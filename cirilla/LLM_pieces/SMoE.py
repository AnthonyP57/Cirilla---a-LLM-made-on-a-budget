import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from cirilla.LLM_pieces.activations import Dynamic_erf, DynamicTanh


@dataclass
class SwiGLUArgs:
    dim:int=128
    d_ff:int=256 # hidden dim
    assert d_ff % 2 == 0
    drop:float=0.1

class SwiGLU(nn.Module):
    def __init__(self, args: SwiGLUArgs):
        super().__init__()
        self.dim = args.dim
        self.d_ff = args.d_ff

        self.w1a = nn.Linear(args.dim, args.d_ff)
        self.w1b = nn.Linear(args.dim, args.d_ff)
        self.w2 = nn.Linear(args.d_ff, args.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.w1a(x)
        b = self.w1b(x)

        x = F.silu(a) * b

        x = self.w2(x)
        return x

@dataclass
class SMoEArgs:
    num_experts:int=8
    k:int=2
    dim:int=128
    dtype_str:str = 'bfloat16'
    device:str = 'cuda'
    d_ff:int=256 # hidden dim
    layer_norm:str = "RMSNorm"

    output_moe_weights:bool = False

    @property
    def dtype(self):
        return getattr(torch, self.dtype_str)

class SMoE(nn.Module):
    def __init__(self, args:SMoEArgs, experts:list[SwiGLU]):
        super().__init__()
        self.n_experts = args.num_experts
        self.k = args.k
        self.gating = nn.Linear(args.dim, args.num_experts)
        self.experts = nn.ModuleList(experts)
        self.args = args

        if self.args.layer_norm == "RMSNorm":
            self.layer_norm = nn.RMSNorm(self.args.dim)
        elif self.args.layer_norm == "Derf":
            self.layer_norm = Dynamic_erf(self.args.dim)
        elif self.args.layer_norm == "DyT":
            self.layer_norm = DynamicTanh(self.args.dim)
        else:
            raise ValueError(f"allowed layer norms: 'RMSNorm', 'Derf', 'DyT' ; got: {self.args.layer_norm}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        x = self.layer_norm(x) # (B,S,D)

        logits = self.gating(x) # (B,S,E)
        topk_vals, topk_idx = torch.topk(logits, self.k, dim=-1) # (B,S,K)

        topk_w = F.softmax(topk_vals, dim=-1) # (B,S,K)
        
        if self.args.output_moe_weights:
            one_hot = F.one_hot(topk_idx, num_classes=self.n_experts).to(x.dtype) # (B,S,K,E)
            weights_per_expert = (one_hot * topk_w.unsqueeze(-1)).sum(dim=2) # (B,S,K,E) * (B,S,K,1) -> (B,S,E)
        else:
            weights_per_expert = None

        x_flat = x.view(-1, D) # (B*S,D)
        out = torch.zeros_like(x_flat) # (B*S,D)
        topk_idx_flat = topk_idx.view(-1, self.k) # (B*S,K)
        topk_w_flat = topk_w.view(-1, self.k) # (B*S,K)

        for expert_id, expert in enumerate(self.experts):

            is_selected = (topk_idx_flat == expert_id) # (B*S,K)
            
            batch_ids, k_ids = torch.where(is_selected) # (B*S,K), (B*S,K)

            if batch_ids.numel() == 0:
                continue

            x_selected = x_flat[batch_ids] # (B'*S,D)

            expert_out = expert(x_selected) # (B'*S,D)

            gate_w = topk_w_flat[batch_ids, k_ids].unsqueeze(-1) # (B'*S,1)
            expert_out = expert_out * gate_w.to(self.args.dtype) # (B'*S,D)

            out.index_add_(dim=0, index=batch_ids, source=expert_out)

        out = out.view(B,S,D)

        return out, weights_per_expert
