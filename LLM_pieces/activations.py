from kernels import get_kernel
from functools import lru_cache

@lru_cache(maxsize=8)
def get_activation(path: str = "kernels-community/activation"):
    return get_kernel(path)
