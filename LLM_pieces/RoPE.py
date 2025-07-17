import torch.nn as nn
import torch

class RoPE(nn.Module):
    def __init__(self, head_dim:int, seq_len:int, device:str='cuda', theta:float = 10000.0):
        super().__init__()
        self.register_buffer('freqs_complex',
                             self._precompute_theta_pos_frequencies(head_dim, seq_len, device, theta))

    @staticmethod
    def _precompute_theta_pos_frequencies(head_dim:int, seq_len:int, device:str, theta:float):
        assert head_dim % 2 == 0, 'embedding cannot be applied to odd number of head (dimentions)'
        theta_numerator = torch.arange(0, head_dim, 2, device=device).float()
        theta = 1.0 / (theta ** (theta_numerator / head_dim))
        m = torch.arange(seq_len, device=device)
        freqs = torch.outer(m, theta).float()
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_complex.unsqueeze(0).unsqueeze(2)
    
    def apply_rotary_embeddings(self, xq: torch.Tensor, xk: torch.Tensor):
        xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

        xq_rotated = xq_complex * self.freqs_complex[:, :xq.shape[1], :, :]
        xk_rotated = xk_complex * self.freqs_complex[:, :xk.shape[1], :, :]

        xq_out = torch.view_as_real(xq_rotated)
        xk_out = torch.view_as_real(xk_rotated)

        xq_out = xq_out.reshape(*xq.shape)
        xk_out = xk_out.reshape(*xk.shape)

        return xq_out.type_as(xq), xk_out.type_as(xk)
    

if __name__ == '__main__':
    rope = RoPE(128, 512)
    xq = torch.randn(2, 512, 4, 128, device='cuda', dtype=torch.bfloat16) # (b, seq_len, h, head_dim)
    xk = torch.randn(2, 512, 4, 128, device='cuda', dtype=torch.bfloat16)
    xq_out, xk_out = rope.apply_rotary_embeddings(xq, xk)
    print(xq.shape, xq_out.shape, xq_out.dtype, xq_out.device)
    print(xk.shape, xk_out.shape, xk_out.dtype, xk_out.device)
    print(rope.freqs_complex.device)