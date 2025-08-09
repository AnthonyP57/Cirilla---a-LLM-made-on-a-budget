# [Huggingface Kernel Hub](https://github.com/huggingface/kernels)
Customs kernels can be used to maximize performance of various functions, however writing them requires knowledge of lower-lever packages/langauges like [Triton](https://github.com/triton-lang/triton) or [CUDA](https://developer.nvidia.com/cuda-faq). [HF's Kernel Hub](https://huggingface.co/models?other=kernel) simplifies this dramatically e.g.

```python
import torch
from kernels import get_kernel

activation = get_kernel("kernels-community/activation")

x = torch.randn((10, 10), dtype=torch.float16, device="cuda")

y = torch.empty_like(x)
activation.gelu_fast(y, x)

print(y)
```

RMS Norm
```python
from kernels import get_kernel

layer_norm_kernel_module = get_kernel("kernels-community/triton-layer-norm")

@use_kernel_forward_from_hub("LlamaRMSNorm")
class OriginalRMSNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = variance_epsilon
        self.hidden_size = hidden_size

    def forward(self, x):
        # Assumes x is (batch_size, ..., hidden_size)
        input_dtype = x.dtype
        # Calculate variance in float32 for stability
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)

        # Apply weight and convert back to original dtype
        return (self.weight * x).to(input_dtype)
```
HF kernels should have a documentation on how to use them or check with dir()

```python
kernel = get_kernel("user-or-org/kernel-name")

# inspect the kernel
print(dir(loaded_kernel))

#e.g. for
# >>> layer_norm_kernel_module = get_kernel("kernels-community/triton-layer-norm")

# >>> print(dir(layer_norm_kernel_module))
# ['Optional', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__kernel_metadata__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 'layer_norm', 'layer_norm_fn', 'layer_norm_linear_fn', 'layers', 'rms_norm_fn', 'torch']
```

Example of how to use can be found in
```bash
./HF_kernels/examples
```

For making sure all previous GPU operations are complete (for benchmarking) we can use:

```python
torch.cuda.synchronize()
```