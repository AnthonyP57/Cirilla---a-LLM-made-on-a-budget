import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import torch
from .activations import get_activation

activation = get_activation("kernels-community/activation")

@dataclass
class SMoEArgs:
    num_experts:int=8
    k:int=2
    dim:int=128

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

class SMoE(nn.Module):
    def __init__(self, args:SMoEArgs, experts:list[Expert]):
        super().__init__()
        self.n_experts = args.num_experts
        self.k = args.k
        self.gating = nn.Linear(args.dim, args.num_experts)
        self.experts = nn.ModuleList(experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        logits = self.gating(x)                       # (B,S,E)
        topk_vals, topk_idx = torch.topk(logits, self.k, dim=-1)
        topk_w = F.softmax(topk_vals, dim=-1)        # (B,S,k)
        one_hot = F.one_hot(topk_idx, num_classes=self.n_experts).to(x.dtype)
        weights_per_expert = (one_hot * topk_w.unsqueeze(-1)).sum(dim=2)  # (B,S,E)

        out = torch.zeros_like(x)
        for ex_idx, expert in enumerate(self.experts):
            w = weights_per_expert[..., ex_idx].unsqueeze(-1)  # (B,S,1)
            out = out + w * expert(x)                         # expert(x) -> (B,S,D)
        return out
    
if __name__=='__main__':
    moe = SMoE(
        SMoEArgs(num_experts=4, k=2, dim=2),
        [Expert(ExpertArgs(dim=2, d_ff=4)) for _ in range(4)]
    ).to("cuda") # hf kernel only work on cuda

    x = torch.randn(2, 10, 2, device='cuda') # (b, seq_len, dim)
    out = moe(x)
    print(out.shape)