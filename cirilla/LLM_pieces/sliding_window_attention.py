from torch.nn.attention.flex_attention import create_block_mask, flex_attention, BlockMask
from functools import lru_cache, partial
from .RoPE import RoPE
import torch.nn as nn
from dataclasses import dataclass
from typing import Union
import torch
from .activations import Dynamic_erf, DynamicTanh

SLIDING_WINDOW = 512

def sliding_window_causal(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    window_mask = q_idx - kv_idx <= SLIDING_WINDOW
    return causal_mask & window_mask

def create_static_block_mask(sliding_window_causal, q_len, kv_len, device='cuda', window_size=512):
    global SLIDING_WINDOW
    SLIDING_WINDOW = window_size
    # B,H set to None means that the mask is broadcasted for those dimentions as it doesn't require any calculation anyway
    return create_block_mask(sliding_window_causal, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device)

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
    device:str = 'cuda:0'

class SlidingWindowAttention(nn.Module):
    def __init__(self, args: AttentionArgs, rope:RoPE, mask:Union[BlockMask, create_dynamic_block_mask]=None, score_mod:callable=None):
        super().__init__()

        self.args = args

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.static_mask = args.static_mask
        
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
        self.mask = mask if not isinstance(mask, BlockMask) else None
        self.score_mode = score_mod
        self.window_size = args.window_size

        self.attn = partial(flex_attention, block_mask=mask if mask is not None and isinstance(mask, BlockMask) else None,
                            score_mod=score_mod if score_mod is not None else None, enable_gqa= self.n_heads_q != self.n_kv_heads)\
                        if self.static_mask else None

    def forward(self, x: torch.Tensor):
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

        if self.static_mask:
            out = self.attn(xq, xk, xv)
        else:
            mask = self.mask(sliding_window_causal, xq.shape[2], xk.shape[2], device=xq.device, window_size=self.window_size)
            out = flex_attention(xq, xk, xv, block_mask=mask, score_mod=self.score_mode, enable_gqa= self.n_heads_q != self.n_kv_heads)
        
        out = out.transpose(1,2).contiguous().view(batch_size, seq_len, dim) # (b, seq, dim)
        return self.wo(out) #(b, seq, dim)
    
    @torch.no_grad
    def forward_with_cache(self, x: torch.Tensor, cur_pos:int, seq_len:int=1, max_batch:int=1):

        batch_size, seq_len, dim = x.shape

        if not hasattr(self, 'k_cache'):
            self.k_cache = torch.zeros(max_batch,
                                       self.args.context_window,
                                       self.n_kv_heads,
                                       self.head_dim,
                                       device=self.args.device,
                                       dtype=x.dtype)
            
            self.v_cache = torch.zeros(max_batch,
                                       self.args.context_window,
                                       self.n_kv_heads,
                                       self.head_dim,
                                       device=self.args.device,
                                       dtype=x.dtype)
        
        x = self.layer_norm(x)

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        cos = self.rope.cos[:, cur_pos:cur_pos+seq_len, :, :]
        sin = self.rope.sin[:, cur_pos:cur_pos+seq_len, :, :]

        xq_even, xq_odd = xq[..., ::2], xq[..., 1::2]
        xk_even, xk_odd = xk[..., ::2], xk[..., 1::2]

        xq_rot = torch.stack([xq_even * cos - xq_odd * sin,
                              xq_even * sin + xq_odd * cos], dim=-1)
        xk_rot = torch.stack([xk_even * cos - xk_odd * sin,
                              xk_even * sin + xk_odd * cos], dim=-1)

        xq = xq_rot.flatten(-2)
        xk = xk_rot.flatten(-2)

        self.k_cache[:batch_size, cur_pos:cur_pos+seq_len, :, :] = xk
        self.v_cache[:batch_size, cur_pos:cur_pos+seq_len, :, :] = xv

        xk = self.k_cache[:batch_size, :cur_pos+seq_len, :, :]
        xv = self.v_cache[:batch_size, :cur_pos+seq_len, :, :]

        # (b, seq, h_q, head_dim) -> (b, h_q, seq, head_dim)
        xq = xq.transpose(1,2)
        xk = xk.transpose(1,2)
        xv = xv.transpose(1,2)

        mask = create_dynamic_block_mask(sliding_window_causal,
                                         q_len=seq_len,
                                         kv_len=cur_pos+seq_len,
                                         device=xq.device,
                                         window_size=self.window_size)

        out = flex_attention(xq, xk, xv,
                             block_mask=mask,
                             score_mod=self.score_mode,
                             enable_gqa=\
                                self.n_heads_q != self.n_kv_heads
                            )

        out = out.transpose(1,2).contiguous().view(batch_size, seq_len, dim) # (b, seq, dim)
        return self.wo(out) #(b, seq, dim)
