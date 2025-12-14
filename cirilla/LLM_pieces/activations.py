from kernels import get_kernel
from functools import lru_cache
import torch
import torch.nn as nn

@lru_cache(maxsize=8)
def get_activation(path: str = "kernels-community/activation"):
    return get_kernel(path)

class Dynamic_erf(nn.Module): # Stronger Normalization-Free Transformers Mingzhi Chen, Taiming Lu, Jiachen Zhu, Mingjie Sun, and Zhuang Liu. https://github.com/zlab-princeton/Derf/blob/main/language%20model/model_derf.patch
    def __init__(self, normalized_shape, alpha_init_value=0.5, shift_init_value=0.0):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.shift_init_value = shift_init_value

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.shift = nn.Parameter(torch.ones(1) * shift_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, x):
        return self.weight * torch.erf(self.alpha * x + self.shift) + self.bias

class DynamicTanh(nn.Module): # Stronger Normalization-Free Transformers Mingzhi Chen, Taiming Lu, Jiachen Zhu, Mingjie Sun, and Zhuang Liu. https://github.com/zlab-princeton/Derf/blob/main/language%20model/model_derf.patch
    def __init__(self, normalized_shape, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, x):
        return self.weight * torch.tanh(self.alpha * x) + self.bias
