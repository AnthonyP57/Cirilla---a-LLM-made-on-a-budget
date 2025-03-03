import torch.nn as nn
from einops import rearrange #"Einstein-inpired notation"
import torch

class DilatedAttention(nn.Module):
    def __init__(self, 
                 w=[4,8,16], # segment len
                 r=[1,2,4], # dilated rate
                 drop=0.1 # dropout rate
                 ):
        super().__init__()

        assert len(w) == len(r), "w and r need to have equal number of elements"
        self.w = w
        self.r = r
        self.drop = nn.Dropout(drop)
        self.n_groups = len(w)
        self.alpha = nn.Parameter(torch.ones(self.n_groups)) #learnable param for weighted sum

    def forward(self, q, k, v):
        b, h, s, d = q.shape # (batch, heads, seq, head_dim)

        assert any([s % w == 0 for w in self.w]), f"Input sequence length {s} is not divisible by any in w: {self.w}"
        
        sub_outputs=[]
        weights = torch.softmax(self.alpha, dim=-1)

        for i, (w, r) in enumerate(zip(self.w, self.r)):

            q_seg = rearrange(q, 'b h (n w) d -> b h n w d', w=w) #split the sequence into segments of length w[i]
            k_seg = rearrange(k, 'b h (n w) d -> b h n w d', w=w) #(batch, heads, seq, head_dim)
            v_seg = rearrange(v, 'b h (n w) d -> b h n w d', w=w) # -> (batch, heads, seq/w, w, head_dim)

            indices = torch.arange(0, w, step=r, device=q.device)
            if indices.size(0) == 0:
                indices = torch.tensor([w - 1], device=q.device)

            # L = indices.size(0)
            
            q_seg = q_seg[:, :, :, indices, :]
            k_seg = k_seg[:, :, :, indices, :]
            v_seg = v_seg[:, :, :, indices, :]

            attn = torch.einsum('b h n i d, b h n j d -> b h n i j', q_seg, k_seg) # (batch, heads, seq/w, L, L) for original: (batch, heads, seq/w, w, w)
            att_shape = attn.shape
            attn = attn * (d ** -0.5)
            attn = torch.softmax(attn, dim=-1)
            attn = self.drop(attn)

            padded_attn = torch.zeros((att_shape[0], att_shape[1], att_shape[2], w, w), device=q.device)
            padded_attn[:, :, :, indices, indices] = attn
            padded_attn = padded_attn.to_sparse()

            padded_v = torch.zeros((att_shape[0], att_shape[1], att_shape[2], w, d), device=q.device)
            padded_v[:, :, :, indices, :] = v_seg
            padded_v = padded_v.to_sparse()

            B, H, N, W, _ = padded_attn.shape
            padded_attn = padded_attn.reshape(B * H * N, W, W)
            padded_v = padded_v.reshape(B * H * N, W, d)
            
            out_blocks = []
            for j in range(B * H * N):
                out_blocks.append(torch.sparse.mm(padded_attn[j], padded_v[j]))
            out_blocks = torch.stack(out_blocks, dim=0)  # (B*H*N, seg, d)
            
            out_blocks = out_blocks.reshape(B, H, N, W, d)
            out_blocks = rearrange(out_blocks, 'b h n w d -> b h (n w) d') # (batch, heads, seq, head_dim)
            sub_outputs.append(out_blocks)

        stacked = torch.stack(sub_outputs, dim=2)  # shape: (b, h, n_groups, s, d)
        weights = weights.view(1, 1, self.n_groups, 1, 1)
        out = (stacked * weights).sum(dim=2)
        return out