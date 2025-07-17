import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import torch
from activations import get_activation

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

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        gate_logits = self.gating(x) # (b, seq_len, n_experts)
        weights, selected_experts_ids = torch.topk(gate_logits, self.k) # (b, seq_len, k) ; (b, seq_len, k)
        weights = F.softmax(weights, dim=-1, dtype=x.dtype) # (b, seq_len, k)
        out = torch.zeros_like(x)

        for ex_idx, expert in enumerate(self.experts):
            if ex_idx in selected_experts_ids:
                batch_idx, token_idx, expert_idx = torch.where(selected_experts_ids == ex_idx) # (n_true) ; (n_true) ; (n_true)
                out[batch_idx, token_idx] += weights[batch_idx, token_idx, expert_idx].unsqueeze(-1) * expert(
                    x[batch_idx, token_idx]
                )
        return out
    
if __name__=='__main__':
    moe = SMoE(
        SMoEArgs(num_experts=4, k=2, dim=2),
        [Expert(ExpertArgs(dim=2, d_ff=4)) for _ in range(4)]
    ).to("cuda") # hf kernel only work on cuda

    x = torch.randn(2, 10, 2, device='cuda') # (b, seq_len, dim)
    out = moe(x)
    print(out.shape)