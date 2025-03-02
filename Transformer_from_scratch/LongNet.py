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
        self.w = w
        self.r = r
        self.drop = nn.Dropout(drop)
        self.n_groups = len(w)

        assert len(w) == len(r), "w and r need to have equal number of elements"

    def forward(self, q, k, v):
        b, h, s, d = q.shape # (batch, heads, seq, head_dim)
        
        out = torch.zeros_like(q)

        assert h % self.n_groups == 0, "num_heads must be divisible by n_groups" #lets naively assume that num_heads has to be divisible by n_groups

        group_sizes = [h // self.n_groups] * self.n_groups  # how many unique attention heads there are

        for i, (g, w, r) in enumerate(zip(group_sizes, self.w, self.r)):
            q = rearrange(q, 'b h (n s) d -> b h n s d', s=w) #split the sequence into segments of length w[i]
            k = rearrange(k, 'b h (n s) d -> b h n s d', s=w) #(batch, heads, seq, head_dim)
            v = rearrange(v, 'b h (n s) d -> b h n s d', s=w) # -> (batch, heads, seq/w, w, head_dim)

            seg_offset = i*r #offset for group
            # size for each group = top - bottom
            bottom = i*g
            top = (i+1)*g

            

