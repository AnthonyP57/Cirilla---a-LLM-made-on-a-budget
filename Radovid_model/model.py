from LLM_pieces import (
    RoPE,
    SMoE,
    SlidingWindowAttention,
    get_activation,
    create_static_block_mask,
    create_dynamic_block_mask,
    sliding_window_causal,
    Expert
)
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
from modules import select_torch_device
from typing import Optional
import warnings
import torch
from attn_gym.mods import generate_tanh_softcap

@dataclass
class Args:
    device:torch.device = select_torch_device()
    vocab_size:int = 50_000
    context_window:int = 2048 # seq len
    dim:int = 1024
    d_ff:int = 2048
    window_size:int = 1024
    n_heads:int = 8
    n_kv_heads:int = 4
    static_mask:bool = True
    soft_cap:Optional[int] = 20
    n_layers:int = 16
    theta:float = 10_000.0
    num_experts:int = 8
    k:int = 4
    dtype:torch.dtype = torch.bfloat16

    def __post_init__(self):
        if not torch.cuda.is_available():
            warnings.warn("hf kernels only work on cuda")
        assert self.dim % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0


class InputEmbeddings(nn.Module):
    def __init__(self, args:Args):
        super().__init__()

        self.embeddings = nn.Embedding(args.vocab_size, args.dim)
    
    def forward(self, x):
        return self.embeddings(x)


class Radovid(nn.Module):
    def __init__(self, args:Args):
        super().__init__()
        self.args = args
        self.emb = InputEmbeddings(args)
        self.rope = RoPE(args.dim // args.n_heads, args.context_window, args.device, args.theta, args.device)
        activation = get_activation('Motif-Technologies/activation')
        self.rmsnorm = activation.layers.RMSNorm(dim=args.dim) if args.device == torch.cuda.is_available() else nn.RMSNorm(args.dim)
        
        if args.static_mask:
            self.mask = create_static_block_mask(sliding_window_causal,args.context_window,
                                             args.context_window, args.device, args.window_size)

            self.attentions = nn.ModuleList([
                torch.compile(
                SlidingWindowAttention(args, self.rope, self.mask, generate_tanh_softcap(args.soft_cap, approx=False) if args.soft_cap is not None else None),
                mode='max-autotune') for _ in range(args.n_layers)
            ])

        else:
            self.attentions = nn.ModuleList([
                torch.compile(SlidingWindowAttention(args, self.rope,
                create_dynamic_block_mask,
                generate_tanh_softcap(args.soft_cap, approx=False) if args.soft_cap is not None else None),
                mode='max-autotune') for _ in range(args.n_layers)
            ])

        self.smoes = nn.ModuleList([
            torch.compile(SMoE(args, [Expert(args) for _ in range(args.num_experts)]), mode='max-autotune')
            for _ in range(args.n_layers)
        ])

        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.output.weight = self.emb.embeddings.weight # tied params

        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.to(args.device, dtype=args.dtype)
        
    def pred(self, x):
        
        x = self.emb(x)

        for attention, moe in zip(self.attentions, self.smoes):
            x = x + attention(x)
            x = x + moe(x)

        x = self.output(x)

        return x
    
if __name__ == '__main__':
    import time
    import numpy as np

    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.capture_scalar_outputs = True

    x = torch.randint(0, 50_000, (4, 2048), dtype=torch.long, device='cuda')
    y = torch.randint(0, 50_000, (4, 2048), dtype=torch.long, device='cuda')
    model = Radovid(Args())

    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    print(model)
    print(model.n_params/1e6, 'M')

    criterion = torch.nn.CrossEntropyLoss()

    optimizer_dict = {p: torch.optim.AdamW([p], fused=True, foreach=False, lr=5e-5) for p in model.parameters()}

    def optimizer_hook(parameter) -> None:
        optimizer_dict[parameter].step()
        optimizer_dict[parameter].zero_grad(set_to_none=True)

    for p in model.parameters():
        p.register_post_accumulate_grad_hook(optimizer_hook)

    times = []

    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
    torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None

    for i in range(5): #warm up for benchmark
        torch.compiler.cudagraph_mark_step_begin()
        out = model.pred(x)
        loss = criterion(out.view(-1, 50_000), x.view(-1))

        loss.backward()

    torch.cuda.synchronize()

    for i in range(100):
        torch.compiler.cudagraph_mark_step_begin()
        out = model.pred(x)
        loss = criterion(out.view(-1, 50_000), y.view(-1))
        loss_item = loss.item()

        loss.backward()
        
        times.append(time.time())
        print(f'average time: {np.mean(np.diff(times))} loss: {loss_item}', end='\r')