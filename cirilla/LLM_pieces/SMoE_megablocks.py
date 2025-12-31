import torch
import torch.nn as nn
from dataclasses import dataclass
from megablocks import Arguments, MoE, dMoE
from .activations import Dynamic_erf, DynamicTanh

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

        if self.args.layer_norm == "RMSNorm":
            self.layer_norm = nn.RMSNorm(self.args.dim)
        elif self.args.layer_norm == "Derf":
            self.layer_norm = Dynamic_erf(self.args.dim)
        elif self.args.layer_norm == "DyT":
            self.layer_norm = DynamicTanh(self.args.dim)
        else:
            raise ValueError(f"allowed layer norms: 'RMSNorm', 'Derf', 'DyT' ; got: {self.args.layer_norm}")

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.layer_norm(x)
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

        if self.args.layer_norm == "RMSNorm":
            self.layer_norm = nn.RMSNorm(self.args.dim)
        elif self.args.layer_norm == "Derf":
            self.layer_norm = Dynamic_erf(self.args.dim)
        elif self.args.layer_norm == "DyT":
            self.layer_norm = DynamicTanh(self.args.dim)
        else:
            raise ValueError(f"allowed layer norms: 'RMSNorm', 'Derf', 'DyT' ; got: {self.args.layer_norm}")

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.layer_norm(x)
        # MegaBlocks expects (seq, batch, dim)
        x = x.transpose(0, 1).contiguous()

        x, _ = self.moe(x)
        del _

        x = x.transpose(0, 1)  # back to (batch, seq, dim)
        return (x,)
