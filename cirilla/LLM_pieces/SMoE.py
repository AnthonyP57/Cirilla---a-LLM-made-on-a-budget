import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import torch
from .activations import get_activation
from megablocks import Arguments, MoE, dMoE

activation = get_activation("kernels-community/activation")

@dataclass
class ExpertArgs:
    dim:int=128
    d_ff:int=256 # hidden dim
    assert d_ff % 2 == 0
    drop:float=0.1

class Expert(nn.Module):
    def __init__(self, args: ExpertArgs):
        super().__init__()
        self.dim = args.dim
        self.d_ff = args.d_ff

        self.w1 = nn.Linear(args.dim, args.d_ff * 2)
        self.w2 = nn.Linear(args.d_ff, args.dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:  # SwiGLU 
        x = self.w1(x)

        d = x.shape[-1] // 2
        out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)

        activation.silu_and_mul(out, x)

        out = self.w2(out)
        return out

@dataclass
class SMoEArgs:
    num_experts:int=8
    k:int=2
    dim:int=128
    dtype_str:str = 'bfloat16'
    device:str = 'cuda'
    d_ff:int=256 # hidden dim

    @property
    def dtype(self):
        return getattr(torch, self.dtype_str)

class SMoE(nn.Module):
    def __init__(self, args:SMoEArgs, experts:list[Expert]):
        super().__init__()
        self.n_experts = args.num_experts
        self.k = args.k
        self.gating = nn.Linear(args.dim, args.num_experts)
        self.experts = nn.ModuleList(experts)
        self.args = args

        activation = get_activation('Motif-Technologies/activation')
        self.rmsnorm = activation.layers.RMSNorm(dim=self.args.dim) if self.args.device == torch.cuda.is_available() else nn.RMSNorm(self.args.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rmsnorm(x)                           # (B,S,D)

        logits = self.gating(x)                       # (B,S,E)
        topk_vals, topk_idx = torch.topk(logits, self.k, dim=-1)

        topk_w = F.softmax(topk_vals, dim=-1)        # (B,S,k)
        one_hot = F.one_hot(topk_idx, num_classes=self.n_experts).to(x.dtype)

        weights_per_expert = (one_hot * topk_w.unsqueeze(-1)).sum(dim=2)  # (B,S,E)

        out = torch.zeros_like(x)
        for ex_idx, expert in enumerate(self.experts):
            w = weights_per_expert[..., ex_idx].unsqueeze(-1)  # (B,S,1)
            out = out + w * expert(x)                         # expert(x) -> (B,S,D)
        return out, weights_per_expert

@dataclass
class MegablockArgs:
    num_experts: int = 4
    k: int = 2
    dim: int = 128
    d_ff: int = 256
    capacity_factor: float = 1.0
    impl: str = "grouped"   # or "sparse" Sparse MLP is not supported with triton >=3.2.0
    dtype_str:str = 'bfloat16'
    device:str = 'cuda'
    moe_zloss_weight:float = 0.1

    @property
    def dtype(self):
        return getattr(torch, self.dtype_str)

class MegablockMoE(nn.Module):
    def __init__(self, args:MegablockArgs):
        super().__init__()

        self.args = args

        activation = get_activation('Motif-Technologies/activation')
        self.rmsnorm = activation.layers.RMSNorm(dim=self.args.dim) if self.args.device == torch.cuda.is_available() else nn.RMSNorm(self.args.dim)

        init_method = torch.nn.init.xavier_uniform_

        self.args = Arguments(
                hidden_size=args.dim,
                ffn_hidden_size=args.d_ff,
                moe_num_experts=args.num_experts,
                moe_capacity_factor=args.capacity_factor,
                moe_top_k=args.k,
                init_method=init_method,
                memory_optimized_mlp=True,
                mlp_type="mlp",
                mlp_impl=args.impl,
                fp16= args.dtype_str == 'float16',
                bf16= args.dtype_str == 'bfloat16',
                device=args.device,
                moe_zloss_weight=args.moe_zloss_weight
            )
        
        self.moe = MoE(
            self.args
        )

    def forward(self, x: torch.Tensor):

        x = self.rmsnorm(x)
        # MegaBlocks expects (seq, batch, dim)
        x = x.transpose(0, 1).contiguous()

        x, _ = self.moe(x)
        del _

        x = x.transpose(0, 1)  # back to (batch, seq, dim)
        return (x,)

class MegablockdMoE(nn.Module):
    def __init__(self, args:MegablockArgs):
        super().__init__()

        self.args = args

        activation = get_activation('Motif-Technologies/activation')
        self.rmsnorm = activation.layers.RMSNorm(dim=self.args.dim) if self.args.device == torch.cuda.is_available() else nn.RMSNorm(self.args.dim)

        init_method = torch.nn.init.xavier_uniform_
        
        self.args = Arguments(
                hidden_size=args.dim,
                ffn_hidden_size=args.d_ff,
                moe_num_experts=args.num_experts,
                moe_capacity_factor=args.capacity_factor,
                moe_top_k=args.k,
                init_method=init_method,
                memory_optimized_mlp=True,
                mlp_type="mlp",
                mlp_impl=args.impl,
                fp16= args.dtype_str == 'float16',
                bf16= args.dtype_str == 'bfloat16',
                device=args.device,
                moe_zloss_weight=args.moe_zloss_weight
            )
        
        self.moe = dMoE(
            self.args
        )

    def forward(self, x: torch.Tensor):

        x = self.rmsnorm(x)
        # MegaBlocks expects (seq, batch, dim)
        x = x.transpose(0, 1).contiguous()

        x, _ = self.moe(x)
        del _

        x = x.transpose(0, 1)  # back to (batch, seq, dim)
        return (x,)
