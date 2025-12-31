from .RoPE import RoPE
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
import torch
from .activations import Dynamic_erf, DynamicTanh
from torch.nn.attention.flex_attention import flex_attention
from functools import partial


@dataclass
class BertAttentionArgs:
    n_heads:int = 16
    n_kv_heads:int = 4
    dim:int = 128*16
    soft_cap:Optional[int] = 20
    device:str = 'cuda:0'
    layer_norm:str = "RMSNorm"

class BertAttention(nn.Module):
    def __init__(self, args: BertAttentionArgs, rope:RoPE, score_mod:callable=None):
        super().__init__()

        self.args = args

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        if self.args.layer_norm == "RMSNorm":
            self.layer_norm = nn.RMSNorm(self.args.dim)
        elif self.args.layer_norm == "Derf":
            self.layer_norm = Dynamic_erf(self.args.dim)
        elif self.args.layer_norm == "DyT":
            self.layer_norm = DynamicTanh(self.args.dim)
        else:
            raise ValueError(f"allowed layer norms: 'RMSNorm', 'Derf', 'DyT' ; got: {self.args.layer_norm}")

        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        
        self.hq_dim = self.n_heads_q * self.head_dim
        self.hkv_dim = self.n_kv_heads * self.head_dim

        self.rope = rope

        self.attn = partial(flex_attention, score_mod=score_mod, enable_gqa= self.n_heads_q != self.n_kv_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape

        x = self.layer_norm(x)

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = self.rope.apply_rotary_embeddings(xq, xk)

        # (b, seq, h_q, head_dim) -> (b, h_q, seq, head_dim)
        xq = xq.transpose(1,2)
        xk = xk.transpose(1,2)
        xv = xv.transpose(1,2)

        out = self.attn(xq, xk, xv)
        
        out = out.transpose(1,2).contiguous().view(batch_size, seq_len, dim) # (b, seq, dim)
        return self.wo(out) #(b, seq, dim)
