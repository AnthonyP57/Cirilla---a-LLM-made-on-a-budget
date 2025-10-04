from torch.nn.attention.flex_attention import create_block_mask, flex_attention, BlockMask
from functools import lru_cache, partial
from .RoPE import RoPE
import torch.nn as nn
from dataclasses import dataclass
from typing import Union, Callable, Optional
import torch
from .activations import get_activation

SLIDING_WINDOW = 512

def sliding_window_causal(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    window_mask = q_idx - kv_idx <= SLIDING_WINDOW
    return causal_mask & window_mask

def create_static_block_mask(sliding_window_causal, q_len, kv_len, device='cuda', window_size=512):
    global SLIDING_WINDOW
    SLIDING_WINDOW = window_size
    # B,H set to None means that the mask is broadcasted for those dimentions as it doesn't require any calculation anyway
    return create_block_mask(sliding_window_causal, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, _compile=True, device=device)

@lru_cache(maxsize=32)
def create_dynamic_block_mask(sliding_window_causal, q_len=2048, kv_len=2048, device='cuda', window_size=512):
    global SLIDING_WINDOW
    SLIDING_WINDOW = window_size
    # B,H set to None means that the mask is broadcasted for those dimentions as it doesn't require any calculation anyway
    return create_block_mask(sliding_window_causal, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device)

@dataclass
class AttentionArgs:
    n_heads:int = 16
    n_kv_heads:int = 4
    dim:int = 128*16
    static_mask:bool = True
    window_size:int = 512
    soft_cap:Optional[int] = 20
    device:str = 'cuda:0'

class SlidingWindowAttention(nn.Module):
    def __init__(self, args: AttentionArgs, rope:RoPE, mask:Union[BlockMask, create_dynamic_block_mask]=None, score_mod:Callable=None):
        super().__init__()

        self.args = args

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.static_mask = args.static_mask
        
        activation = get_activation('Motif-Technologies/activation')
        self.rmsnorm = activation.layers.RMSNorm(dim=self.args.dim) if self.args.device == torch.cuda.is_available() else nn.RMSNorm(self.args.dim)

        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        
        self.hq_dim = self.n_heads_q * self.head_dim
        self.hkv_dim = self.n_kv_heads * self.head_dim

        self.rope = rope
        self.mask = mask if not isinstance(mask, BlockMask) else None
        self.score_mode = score_mod
        self.window_size = args.window_size

        self.attn = partial(flex_attention, block_mask=mask if mask is not None and isinstance(mask, BlockMask) else None,
                            score_mod=score_mod if score_mod is not None else None, enable_gqa= self.n_heads_q != self.n_kv_heads)\
                        if self.static_mask else None

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, dim = x.shape

        x = self.rmsnorm(x)

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

        if self.static_mask:
            out = self.attn(xq, xk, xv)
        else:
            mask = self.mask(sliding_window_causal, xq.shape[2], xk.shape[2], device=xq.device, window_size=self.window_size)
            out = flex_attention(xq, xk, xv, block_mask=mask, score_mod=self.score_mode)
        
        out = out.transpose(1,2).contiguous().view(batch_size, seq_len, dim) # (b, seq, dim)
        return self.wo(out) #(b, seq, dim)
