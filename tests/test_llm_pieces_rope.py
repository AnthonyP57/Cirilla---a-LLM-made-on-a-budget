from cirilla.LLM_pieces import RoPE
import pytest
import torch

@pytest.fixture
def get_rope():
    return RoPE(head_dim=128, seq_len=1024, device='cuda:0')

@pytest.mark.parametrize("seq_len", [
    256,
    512,
    1024,
])
def test_rope_initialization(seq_len, get_rope):
    rope = get_rope
    x = torch.randn(2, seq_len, 4, 128, device='cuda:0')

    out = rope.apply_rotary_embeddings(x, x)
    assert out[0].shape == x.shape
    assert out[1].shape == x.shape