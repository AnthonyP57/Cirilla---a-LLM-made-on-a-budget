from .activations import get_activation, Dynamic_erf, DynamicTanh
from .SMoE import SMoE, SwiGLU
from .RoPE import RoPE
from .sliding_window_attention import (
                                    SlidingWindowAttention,
                                    create_dynamic_block_mask,
                                    create_static_block_mask,
                                    sliding_window_causal
                                      )
from .BERT_attention import BertAttention

__all__ = [
    'get_activation',
    'SMoE',
    'SwiGLU',
    'MegablockMoE',
    'MegablockdMoE',
    'RoPE',
    'SlidingWindowAttention',
    'create_dynamic_block_mask',
    'create_static_block_mask',
    'sliding_window_causal',
    'BertAttention',
    'Dynamic_erf',
    'DynamicTanh'
]

try:
    from .SMoE_megablocks import MegablockMoE, MegablockdMoE

    __all__ += [
        "MegablockMoE",
        "MegablockdMoE",
    ]
except ImportError:
    pass