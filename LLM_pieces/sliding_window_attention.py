from torch.nn.attention.flex_attention import create_block_mask, flex_attention, BlockMask
from functools import lru_cache, partial
from attn_gym.mods import generate_tanh_softcap
from .RoPE import RoPE
import torch.nn as nn
from dataclasses import dataclass
from typing import Union, Callable, Optional
import torch

SLIDING_WINDOW = 512
SOFT_CAP = 20

def sliding_window_causal(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    window_mask = q_idx - kv_idx <= SLIDING_WINDOW 
    return causal_mask & window_mask

def create_static_block_mask(sliding_window_causal, q_len, kv_len, device='cuda'):
    # B,H set to None means that the mask is broadcasted for those dimentions as it doesn't require any calculation anyway
    return create_block_mask(sliding_window_causal, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, _compile=True, device=device)

@lru_cache(maxsize=32)
def create_dynamic_block_mask(sliding_window_causal, q_len=2048, kv_len=2048, device='cuda'):
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

class SlidingWindowAttention(nn.Module):
    def __init__(self, args: AttentionArgs, rope:RoPE, mask:Union[BlockMask, create_dynamic_block_mask]=None, score_mod:Callable=None):
        super().__init__()

        global SLIDING_WINDOW, SOFT_CAP
        SLIDING_WINDOW = args.window_size
        SOFT_CAP = args.soft_cap

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.static_mask = args.static_mask

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.rope = rope
        self.mask = mask if not isinstance(mask, BlockMask) else None
        self.score_mode = score_mod

        self.attn = partial(flex_attention, block_mask=mask if mask is not None and isinstance(mask, BlockMask) else None,
                            score_mod=score_mod if score_mod is not None else None)\
                        if static_mask else None
    
    @staticmethod
    def _repeat_kv(x:torch.Tensor, n_rep:int):
        batch, seq, n_kv, head_dim = x.shape
        if n_rep == 1:
            return x
        else:
            return (
                x[:, :, :, None, :].expand(batch, seq, n_kv, n_rep, head_dim)
                .reshape(batch, seq, n_kv*n_rep, head_dim)
            )

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, dim = x.shape
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = self.rope.apply_rotary_embeddings(xq, xk)

        xk = self._repeat_kv(xk, self.n_rep)
        xv = self._repeat_kv(xv, self.n_rep)

        # (b, seq, h_q, head_dim) -> (b, h_q, seq, head_dim)
        xq = xq.transpose(1,2)
        xk = xk.transpose(1,2)
        xv = xv.transpose(1,2)

        if self.static_mask:
            out = self.attn(xq, xk, xv)
        else:
            mask = self.mask(sliding_window_causal, xq.shape[2], xk.shape[2], device=xq.device)
            out = flex_attention(xq, xk, xv, block_mask=mask, score_mod=self.score_mode)
        
        out = out.transpose(1,2).contiguous().view(batch_size, seq_len, dim) # (b, seq, dim)
        return self.wo(out) #(b, seq, dim)

if __name__=='__main__':

    # EXAMPLE USAGE

    import torch
    # print(torch.cuda.get_device_properties(0))

    # softcap = generate_tanh_softcap(SOFT_CAP, approx=False) # approximation of tanh for performance
    # # UserWarning: There is a performance drop because we have not yet implemented the batching rule for approx::tanh.

    # block_mask = create_dynamic_block_mask(sliding_window_causal)

    # query = torch.rand((1,16,2048,128), device='cuda', dtype=torch.bfloat16) # (b, h, seq, head_dim)
    # key = torch.rand((1,16,2048,128), device='cuda', dtype=torch.bfloat16)
    # value = torch.rand((1,16,2048,128), device='cuda', dtype=torch.bfloat16)

    # out = flex_attention(query, key, value, block_mask=block_mask)
    # print(out[0,0,:8,:8])
    # flex_attention = torch.compile(flex_attention, dynamic=False, mode='max-autotune') # for bigger q k v sizes this will throw an error - out of resource: shared memory, Required: 335872, Hardware limit: 101376.
    # out = flex_attention(query, key, value, block_mask=block_mask) # after compilation the block_mask may change and this won't trigger recompilation
    
    # query_ = torch.rand((1,16,2048,128), device='cuda', dtype=torch.bfloat16)
    # key_ = torch.rand((1,16,2048,128), device='cuda', dtype=torch.bfloat16)
    # value_ = torch.rand((1,16,2048,128), device='cuda', dtype=torch.bfloat16)
    
    # block_mask = create_dynamic_block_mask(sliding_window_causal, 2048, 2048)
    # out = flex_attention(query_, key_, value_, block_mask=block_mask, score_mod=softcap)
    # print(out[0,0,:8,:8])

    # #static
    # static_mask = create_static_block_mask(sliding_window_causal, 2048, 2048)
    # causal_attention = partial(flex_attention, block_mask=static_mask) # partial safes the arguments
    # causal_attention = partial(flex_attention, block_mask=static_mask, score_mod=softcap)
    # out = causal_attention(query, key, value)
    # print(out[0,0,:8,:8])

    x = torch.rand((1,2048,128*16), device='cuda', dtype=torch.bfloat16) # (b, seq, head_dim*h)

    static_mask = create_static_block_mask(sliding_window_causal, 2048, 2048)
    softcap = generate_tanh_softcap(SOFT_CAP, approx=False)

    rope = RoPE(128, 2048)

    attention_layer = SlidingWindowAttention(AttentionArgs(), rope, mask=static_mask, score_mod=softcap).to('cuda', dtype=torch.bfloat16)
    attention_layer = torch.compile(attention_layer, mode='max-autotune')
    out = attention_layer(x)
    print(out.shape)#, out)

    # dynamic mask
    dynamic_args = AttentionArgs(static_mask=False)
    attention_layer = SlidingWindowAttention(dynamic_args, mask=create_dynamic_block_mask, rope=rope, score_mod=softcap).to('cuda', dtype=torch.bfloat16)
    out = attention_layer(x)
    print(out.shape)#, out)

    x = torch.rand((1,512,128*16), device='cuda', dtype=torch.bfloat16) # (b, seq, head_dim*h)

    out = attention_layer(x)
    print(out.shape)#, out)


    x = torch.rand((1,256,128*16), device='cuda', dtype=torch.bfloat16) # (b, seq, head_dim*h)

    out = attention_layer(x)
    print(out.shape)#, out)

    
    x = torch.rand((1,2048,128*16), device='cuda', dtype=torch.bfloat16) # (b, seq, head_dim*h)

    out = attention_layer(x)
    print(out.shape)#, out)
    print(create_dynamic_block_mask.cache_info())
