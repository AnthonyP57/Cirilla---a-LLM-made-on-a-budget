from cirilla.LLM_pieces import (
    SlidingWindowAttention,
    create_dynamic_block_mask,
    create_static_block_mask,
    sliding_window_causal,
    RoPE,
    )
from cirilla.LLM_pieces.sliding_window_attention import AttentionArgs
import torch
import pytest
from attn_gym.mods import generate_tanh_softcap

@pytest.fixture
def get_rope():
    return RoPE(head_dim=128, seq_len=2048, device='cuda')

def test_static_mask(get_rope):
    SOFT_CAP = 20
    softcap = generate_tanh_softcap(SOFT_CAP, approx=False)
    static_mask = create_static_block_mask(sliding_window_causal, 2048, 2048)
    x = torch.rand((1,2048,128*16), device='cuda', dtype=torch.bfloat16) # (b, seq, head_dim*h)

    attention_layer = SlidingWindowAttention(AttentionArgs(
                                                n_heads=16,
                                                n_kv_heads=4,
                                                dim=128*16,
                                                static_mask=True,
                                                window_size=512,
                                                device='cuda'
                                            ),
                                            rope=get_rope,
                                            mask=static_mask,
                                            score_mod=softcap).to('cuda', dtype=torch.bfloat16)
    # verify forward pass
    out = attention_layer(x)
    print(out.shape) # torch.Size([1, 2048, 2048])
    assert out.shape == (1, 2048, 2048)
    
    # verify backward pass
    loss = out.sum()
    loss.backward()

    grads_ok = all(p.grad is not None for p in attention_layer.parameters() if p.requires_grad)
    if not grads_ok:
        for name, p in attention_layer.named_parameters():
            if p.requires_grad and p.grad is None:
                print(f"Parameter {name} has no gradient.")
        raise AssertionError("Not all parameters have gradients after initialization.")

def test_dynamic_mask(get_rope):
    """" dynamic mask - won't trigger recompilation """
    SOFT_CAP = 20
    softcap = generate_tanh_softcap(SOFT_CAP, approx=False)
    x = torch.rand((1,2048,128*16), device='cuda', dtype=torch.bfloat16) # (b, seq, head_dim*h)
    dynamic_args = AttentionArgs(static_mask=False,
                                n_heads=16,
                                n_kv_heads=4,
                                dim=128*16,
                                window_size=512,
                                device='cuda'
                                )
    attention_layer = SlidingWindowAttention(dynamic_args,
                                            mask=create_dynamic_block_mask,
                                            rope=get_rope,
                                            score_mod=softcap).to('cuda', dtype=torch.bfloat16)
    out = attention_layer(x)
    print(out.shape) # torch.Size([1, 2048, 2048])
    assert out.shape == (1, 2048, 2048)

    loss = out.sum()
    loss.backward()
    grads_ok = all(p.grad is not None for p in attention_layer.parameters() if p.requires_grad)
    if not grads_ok:
        for name, p in attention_layer.named_parameters():
            if p.requires_grad and p.grad is None:
                print(f"Parameter {name} has no gradient.")
        raise AssertionError("Not all parameters have gradients after initialization.")

    for tensor_shape in [(1,512,128*16), (1,256,128*16), (1,2048,128*16)]:

        x = torch.rand(tensor_shape, device='cuda', dtype=torch.bfloat16) # (b, seq, head_dim*h)

        out = attention_layer(x)
        print(out.shape)
        assert out.shape == tensor_shape

        loss = out.sum()
        loss.backward()
        grads_ok = all(p.grad is not None for p in attention_layer.parameters() if p.requires_grad)
        if not grads_ok:
            for name, p in attention_layer.named_parameters():
                if p.requires_grad and p.grad is None:
                    print(f"Parameter {name} has no gradient.")
            raise AssertionError("Not all parameters have gradients after initialization.")

    print(create_dynamic_block_mask.cache_info()) # how many times the mask template was reused
    # CacheInfo(hits=1, misses=3, maxsize=32, currsize=3)
    assert create_dynamic_block_mask.cache_info().hits == 1
    assert create_dynamic_block_mask.cache_info().misses == 3
    assert create_dynamic_block_mask.cache_info().currsize == 3