from cirilla.LLM_pieces.BERT_attention import BertAttention, BertAttentionArgs
from cirilla.LLM_pieces import RoPE
import pytest
import torch

@pytest.fixture
def get_rope():
    return RoPE(head_dim=128, seq_len=1024, device='cuda:0')

@pytest.mark.parametrize("n_heads, n_kv_heads, dim, soft_cap, device", [
    (4, 2, 128*4, 20, 'cuda:0'),
])
def test_bert_attention_initialization(n_heads, n_kv_heads, dim, soft_cap, device, get_rope):
    args = BertAttentionArgs(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        dim=dim,
        soft_cap=soft_cap,
        device=device
    )
    attention = BertAttention(args, get_rope).to(device, dtype=torch.bfloat16)
    
    # verify forward pass
    x = torch.randn(2, 512, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    out = attention(x)
    assert out.shape == (2, 512, dim)

    # very backward pass
    loss = out.sum()
    loss.backward()

    grads_ok = all(p.grad is not None for p in attention.parameters() if p.requires_grad)
    if not grads_ok:
        for name, p in attention.named_parameters():
            if p.requires_grad and p.grad is None:
                print(f"Parameter {name} has no gradient.")
        raise AssertionError("Not all parameters have gradients after initialization.")