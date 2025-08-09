from LLM_pieces import (
    RoPE,
    SMoE,
    SlidingWindowAttention,
    get_activation,
    create_static_block_mask,
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

@dataclass
class Args:
    device:torch.device = select_torch_device()
    vocab_size:int = 30_000
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
    static_mask:bool = True
    dtype:torch.dtype = torch.bfloat16

    warnings.warn("hf kernels only work on cuda")
    assert dim % n_heads == 0
    assert n_heads % n_kv_heads == 0


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
        self.rope = RoPE(args.dim // args.n_heads, args.context_window, args.device, args.theta)
        activation = get_activation('Motif-Technologies/activation')
        self.rmsnorm = activation.layers.RMSNorm(dim=args.dim) if args.device == torch.cuda.is_available() else nn.RMSNorm(args.dim)
        self.mask = create_static_block_mask(sliding_window_causal,args.context_window,
                                             args.context_window, args.device)
        self.attentions = nn.ModuleList([
            SlidingWindowAttention(args, self.rope, self.mask)
            for _ in range(args.n_layers)
        ])

        self.smoes = nn.ModuleList([
            SMoE(args, [Expert(args) for _ in range(args.num_experts)])
            for _ in range(args.n_layers)
        ])

        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.to(args.device, dtype=args.dtype)

    def pred(self, x):
        
        x = self.emb(x).to(self.args.dtype)

        for attention, moe in zip(self.attentions, self.smoes):
            x = x + attention(self.rmsnorm(x))
            x = x + moe(self.rmsnorm(x))

        x = self.output(self.rmsnorm(x))

        return x
    
if __name__ == '__main__':
    from torchao.optim import _AdamW

    x = torch.randint(0, 30_000, (4, 2048), dtype=torch.long, device='cuda')
    model = Radovid(Args())
    # out = model.pred(x)
    # print(out.shape)
    # print(model.n_params/1e6, 'M')

    model = torch.compile(model, mode='reduce-overhead')

    # run inference
    # out = model.pred(x)
    # print(out.shape)
    print(model.n_params/1e6, 'M')

    # optim = AdamFp8(model.parameters(), lr=1e-3)
    # optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    optim = _AdamW(model.parameters(), bf16_stochastic_round=True, lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for i in range(100):
        out = model.pred(x)
        loss = criterion(out.view(-1, 30_000), x.view(-1))

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()