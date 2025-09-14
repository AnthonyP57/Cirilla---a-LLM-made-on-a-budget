import torch.nn as nn
import torch

class RoPE(nn.Module):
    def __init__(self, head_dim: int, seq_len: int, device="cuda", theta: float = 10000.0, dtype=torch.bfloat16):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even"
        self.dtype = dtype

        theta_numerator = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (theta_numerator / head_dim))  # (head_dim/2)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)  # (seq_len)
        freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)

        # (1, seq_len, 1, head_dim/2)
        cos = torch.cos(freqs)[None, :, None, :].to(dtype)
        sin = torch.sin(freqs)[None, :, None, :].to(dtype)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def apply_rotary_embeddings(self, xq: torch.Tensor, xk: torch.Tensor):
        seq_len = xq.size(1)

        cos = self.cos[:, :seq_len, :, :]  # [1, seq_len, 1, head_dim/2]
        sin = self.sin[:, :seq_len, :, :]  # same

        # Split last dim into even/odd
        xq_even, xq_odd = xq[..., ::2], xq[..., 1::2]
        xk_even, xk_odd = xk[..., ::2], xk[..., 1::2]

        xq_rot = torch.stack([xq_even * cos - xq_odd * sin,
                              xq_even * sin + xq_odd * cos], dim=-1)
        xk_rot = torch.stack([xk_even * cos - xk_odd * sin,
                              xk_even * sin + xk_odd * cos], dim=-1)

        xq_out = xq_rot.flatten(-2)
        xk_out = xk_rot.flatten(-2)

        return xq_out, xk_out
