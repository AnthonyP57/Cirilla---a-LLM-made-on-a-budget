from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from functools import lru_cache, partial
from attn_gym.mods import generate_tanh_softcap

SLIDING_WINDOW = 1024
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

if __name__=='__main__':
    import torch
    print(torch.cuda.get_device_properties(0))

    softcap = generate_tanh_softcap(SOFT_CAP, approx=False) # approximation of tanh for performance
    # UserWarning: There is a performance drop because we have not yet implemented the batching rule for approx::tanh.

    block_mask = create_dynamic_block_mask(sliding_window_causal)

    query = torch.rand((1,16,2048,128), device='cuda', dtype=torch.bfloat16)
    key = torch.rand((1,16,2048,128), device='cuda', dtype=torch.bfloat16)
    value = torch.rand((1,16,2048,128), device='cuda', dtype=torch.bfloat16)

    out = flex_attention(query, key, value, block_mask=block_mask)
    print(out[0,0,:8,:8])
    flex_attention = torch.compile(flex_attention, dynamic=False, mode='max-autotune') # for bigger q k v sizes this will throw an error - out of resource: shared memory, Required: 335872, Hardware limit: 101376.
    out = flex_attention(query, key, value, block_mask=block_mask) # after compilation the block_mask may change and this won't trigger recompilation
    
    query_ = torch.rand((1,16,2048,128), device='cuda', dtype=torch.bfloat16)
    key_ = torch.rand((1,16,2048,128), device='cuda', dtype=torch.bfloat16)
    value_ = torch.rand((1,16,2048,128), device='cuda', dtype=torch.bfloat16)
    
    block_mask = create_dynamic_block_mask(sliding_window_causal, 2048, 2048)
    out = flex_attention(query_, key_, value_, block_mask=block_mask, score_mod=softcap)
    print(out[0,0,:8,:8])

    #static
    static_mask = create_static_block_mask(sliding_window_causal, 2048, 2048)
    causal_attention = partial(flex_attention, block_mask=static_mask) # partial safes the arguments
    causal_attention = partial(flex_attention, block_mask=static_mask, score_mod=softcap)
    out = causal_attention(query, key, value)
    print(out[0,0,:8,:8])