from LLM_pieces import (
    RoPE,
    SMoE,
    SlidingWindowAttention,
    get_activation,
    create_static_block_mask,
    create_dynamic_block_mask,
    sliding_window_causal,
    Expert,
    MegablockMoE
)
from dataclasses import dataclass
import torch.nn as nn
from modules import select_torch_device, get_args_from_hub
from typing import Optional
import warnings
import torch
from attn_gym.mods import generate_tanh_softcap
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from safetensors.torch import load_file

@dataclass
class Args:
    device:str = select_torch_device()
    vocab_size:int = 50_000
    context_window:int = 2048 # seq len
    dim:int = 1024
    d_ff:int = 2048
    window_size:int = 1024
    n_heads:int = 8
    n_kv_heads:int = 4
    static_mask:bool = True
    soft_cap:Optional[int] = 20
    # n_layers:int = 16
    n_layers:int = 8
    theta:float = 10_000.0
    num_experts:int = 8
    k:int = 4
    capacity_factor: float = 1.0
    impl: str = "grouped"   # or "sparse" Sparse MLP is not supported with triton >=3.2.0
    dtype_str:str = 'bfloat16'

    @property
    def dtype(self):
        return getattr(torch, self.dtype_str)

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


class Radovid(
            nn.Module,
            PyTorchModelHubMixin,
            pipeline_tag="text-generation",
            library_name="pytorch",
            license="mit"
    ):
    def __init__(self, args:Args=None):
        super().__init__()

        if isinstance(args, dict):
               args = Args(**args)
        if args is None:
            args = Args()

        self.args = args
        self._prepare_model()

    def _prepare_model(self):

        self.emb = InputEmbeddings(self.args)
        self.rope = RoPE(self.args.dim // self.args.n_heads, self.args.context_window, self.args.device, self.args.theta, self.args.device)
        activation = get_activation('Motif-Technologies/activation')
        self.rmsnorm = activation.layers.RMSNorm(dim=self.args.dim) if self.args.device == torch.cuda.is_available() else nn.RMSNorm(self.args.dim)
        
        if self.args.static_mask:
            self.mask = create_static_block_mask(sliding_window_causal,self.args.context_window,
                                             self.args.context_window, self.args.device, self.args.window_size)

            self.attentions = nn.ModuleList([
                torch.compile(
                SlidingWindowAttention(self.args, self.rope, self.mask, generate_tanh_softcap(self.args.soft_cap, approx=False) if self.args.soft_cap is not None else None),
                mode='max-autotune') for _ in range(self.args.n_layers)
            ])

        else:
            self.attentions = nn.ModuleList([
                torch.compile(SlidingWindowAttention(self.args, self.rope,
                create_dynamic_block_mask,
                generate_tanh_softcap(self.args.soft_cap, approx=False) if self.args.soft_cap is not None else None),
                mode='max-autotune') for _ in range(self.args.n_layers)
            ])

        # self.smoes = nn.ModuleList([
        #     torch.compile(SMoE(self.args, [Expert(self.args) for _ in range(self.args.num_experts)]), mode='max-autotune')
        #     for _ in range(self.args.n_layers)
        # ])

        self.smoes = nn.ModuleList([
            MegablockMoE(self.args)
            for _ in range(self.args.n_layers)
        ])

        self.output = nn.Linear(self.args.dim, self.args.vocab_size, bias=False)
        self.output.weight = self.emb.embeddings.weight # tied params

        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.to(self.args.device, dtype=self.args.dtype)
        
    def pred(self, x):
        
        x = self.emb(x)

        for attention, moe in zip(self.attentions, self.smoes):
            x = x + attention(x)
            x = x + moe(x)

        x = self.output(x)

        return x
    
    def pull_model_from_hub(self, hf_repo_id:str):
        model_args = self.args
        pulled_args = get_args_from_hub(hf_repo_id)

        if model_args != pulled_args:
            print(f"Current model args don't correspond to the HF model's args.\nCurrent args:\n{model_args}\nThe model will use the HF args:\n{pulled_args}")
            self.args = pulled_args
            self._prepare_model()

        file_path = hf_hub_download(
            repo_id=hf_repo_id,
            filename="model.safetensors",
        )

        loaded = load_file(file_path)
        if "output.weight" not in loaded:
            loaded['output.weight'] = loaded["emb.embeddings.weight"]

        self.load_state_dict(loaded)
    
if __name__ == '__main__':
    import time
    import numpy as np
    from hf_hub import push_model_to_hub, hf_hub_download
    import json

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

    # # optimizer_dict = {p: torch.optim.AdamW([p], fused=True, foreach=False, lr=5e-5) for p in model.parameters()}

    optim_kwargs = dict(fused=True, foreach=False, lr=5e-5)

    # 1) CREATE per-parameter optimizers (keyed by parameter NAME)
    optimizer_by_name = {}
    for name, p in model.named_parameters():
        optimizer_by_name[name] = torch.optim.AdamW([p], **optim_kwargs)

    # 2) register hooks using a param->optimizer lookup for fast access in the hook
    # build param object -> optimizer mapping
    params_by_name = dict(model.named_parameters())
    optimizer_by_param = {params_by_name[name]: opt for name, opt in optimizer_by_name.items()}

    def optimizer_hook(parameter) -> None:
        optimizer_by_param[parameter].step()
        optimizer_by_param[parameter].zero_grad(set_to_none=True)

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

    for i in range(20):
        torch.compiler.cudagraph_mark_step_begin()
        out = model.pred(x)
        loss = criterion(out.view(-1, 50_000), y.view(-1))
        loss_item = loss.item()

        loss.backward()
        
        times.append(time.time())
        print(f'average time: {np.mean(np.diff(times))} loss: {loss_item}', end='\r')

    print(f'average time: {np.mean(np.diff(times))} loss: {loss_item}')
    

    # push_model_to_hub("AnthonyPa57/HF-torch-demo-R", model, loss, 'pretraining', save_locally='./test_model')

    # repo_id = "AnthonyPa57/HF-torch-demo-R"
    # filename = "config.json"

    # file_path = hf_hub_download(
    #     repo_id=repo_id,
    #     filename=filename,
    # )

    # with open(file_path, "r") as f:
    #     config = json.load(f)

    # args = Args(**config[list(config.keys())[0]])

    torch.save(model.state_dict(), './test_model/model.pt')
    optim_states = {name: opt.state_dict() for name, opt in optimizer_by_name.items()}
    torch.save(optim_states, './test_model/optimizer.pt')
    print("Saved model + optimizers")

    # --- loading ---

    model = Radovid(Args())
    model.load_state_dict(torch.load('./test_model/model.pt', map_location=model.args.device))
    print("Loaded model weights")

    # Load optimizer states
    loaded_states = torch.load('./test_model/optimizer.pt', map_location='cuda')

    # Recreate optimizers for *new* parameter objects
    params_by_name = dict(model.named_parameters())
    optimizer_by_name = {}
    for name, state in loaded_states.items():
        if name not in params_by_name:
            print(f"Skipping unknown param: {name}")
            continue
        p = params_by_name[name]
        opt = torch.optim.AdamW([p], fused=True, foreach=False, lr=5e-5)
        # Move saved state tensors to param's device
        # for s in state["state"].values():
        #     for k, v in s.items():
        #         if isinstance(v, torch.Tensor):
        #             s[k] = v.to(p.device)
        opt.load_state_dict(state)
        optimizer_by_name[name] = opt

    # Build param->optimizer mapping
    optimizer_by_param = {params_by_name[n]: o for n, o in optimizer_by_name.items()}

    # Register hooks again
    def optimizer_hook(parameter):
        optimizer_by_param[parameter].step()
        optimizer_by_param[parameter].zero_grad(set_to_none=True)

    for p in model.parameters():
        if p in optimizer_by_param:
            p.register_post_accumulate_grad_hook(optimizer_hook)

    print("Restored optimizer states + hooks — training can resume")

    times = []

    for i in range(20):
        torch.compiler.cudagraph_mark_step_begin()
        out = model.pred(x)
        loss = criterion(out.view(-1, 50_000), y.view(-1))
        loss_item = loss.item()

        loss.backward()
        
        times.append(time.time())
        print(f'average time: {np.mean(np.diff(times))} loss: {loss_item}', end='\r')