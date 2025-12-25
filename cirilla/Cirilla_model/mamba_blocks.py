from ..LLM_pieces import (
    RoPE,
    SMoE,
    SwiGLU,
    MegablockMoE,
    MegablockdMoE,
    SlidingWindowAttention,
    create_static_block_mask,
    create_dynamic_block_mask,
    sliding_window_causal,
    DynamicTanh,
    Dynamic_erf
)
from mamba_ssm import Mamba2
from .blocks import DecoderArgs
from attn_gym.mods import generate_tanh_softcap
from dataclasses import dataclass
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchao.float8 import convert_to_float8_training, Float8LinearConfig
from torchao.sparsity.training import (
    SemiSparseLinear,
    swap_linear_with_semi_sparse_linear,
)

@dataclass
class HybridDecoderArgs(DecoderArgs):
    layer_pattern:str = 'MAMAM' # M - Mamba2 block, A - Attention block
    d_state:int = 64 # typically 64 or 128
    d_conv:int = 4

    def __post_init__(self):
        super().__post_init__()
        _layers=[]
        _number=''
        for layer in self.layer_pattern:
            if layer.isdigit():
                _number+=layer
            else:
                n_repeat = int(_number) if _number else 1
                _layers.append(layer*n_repeat)
                _number=''

        self.layer_pattern = "".join(_layers)

        if len(self.layer_pattern) != self.n_layers:
            raise ValueError(f"layer_pattern length {len(self.layer_pattern)} must match n_layers {self.n_layers}")
        
        assert self.d_ff % self.dim == 0, "d_ff must be multiple of dim"

class HybridDecoder(nn.Module):

    def __init__(self, args:HybridDecoderArgs=None):
        super().__init__()

        if isinstance(args, dict):
            args = HybridDecoderArgs(**args)

        if args is None:
            args = HybridDecoderArgs()

        self.args = args
        self._prepare_model()

    def _prepare_model(self):

        self.rope = RoPE(self.args.dim // self.args.n_heads, self.args.context_window, self.args.device, self.args.theta, self.args.device)
        if self.args.layer_norm == "RMSNorm":
            self.layer_norm = nn.RMSNorm(self.args.dim)
        elif self.args.layer_norm == "Derf":
            self.layer_norm = Dynamic_erf(self.args.dim)
        elif self.args.layer_norm == "DyT":
            self.layer_norm = DynamicTanh(self.args.dim)
        else:
            raise ValueError(f"allowed layer norms: 'RMSNorm', 'Derf', 'DyT' ; got: {self.args.layer_norm}")
    
        if self.args.static_mask:
            self.mask = create_static_block_mask(sliding_window_causal,self.args.context_window,
                                            self.args.context_window, self.args.device, self.args.window_size)

        self.attentions = []

        for layer_type in self.args.layer_pattern:

            if layer_type == 'A':

                if self.args.static_mask:

                    self.attentions.append(
                        SlidingWindowAttention(self.args, self.rope, self.mask, generate_tanh_softcap(self.args.soft_cap, approx=False) if self.args.soft_cap is not None else None)
                    )

                else:
                    self.attentions.append(
                        SlidingWindowAttention(self.args, self.rope,
                            create_dynamic_block_mask,
                            generate_tanh_softcap(self.args.soft_cap, approx=False) if self.args.soft_cap is not None else None)
                    )
            
            elif layer_type == 'M':
                self.attentions.append(
                    Mamba2(
                        d_model = self.args.dim,
                        d_state = self.args.d_state,
                        d_conv = self.args.d_conv,
                        expand = self.args.d_ff // self.args.dim,
                        device = self.args.device,
                        dtype = self.args.dtype

                    )
                )
            else:
                raise ValueError(f"allowed layer types: 'A' for attention, 'M' for Mamba2 ; got: {layer_type}")

        if self.args.dtype_str == 'fp8':

            config = Float8LinearConfig.from_recipe_name(self.args.fp8_recipe)

            def module_filter_fn(mod: torch.nn.Module, fqn: str):
                # don't convert the last module
                if fqn == "1":
                    return False
                # don't convert linear modules with weight dimensions not divisible by 16
                if isinstance(mod, torch.nn.Linear):
                    if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                        return False
                return True
            
            self.attentions = [convert_to_float8_training(attention, config=config, module_filter_fn=module_filter_fn) for attention in self.attentions]

        if self.args.use_sparse:

            def get_sparse_config(model, sparse_cls=SemiSparseLinear):
                config = {}
                for name, m in model.named_modules():
                    if isinstance(m, torch.nn.Linear):
                        out, inp = m.out_features, m.in_features
                        if out % 128 == 0 and inp % 128 == 0:
                            config[name] = sparse_cls
                return config
            
            for attention in self.attentions:
                swap_linear_with_semi_sparse_linear(attention, get_sparse_config(attention))

        if self.args.torch_compile:
            self.attentions = nn.ModuleList([
                torch.compile(attention.to(dtype=self.args.dtype), mode='max-autotune') for attention in self.attentions
                ])
        else:
            self.attentions = nn.ModuleList(self.attentions)
        
        if self.args.moe_type == 'pytorch':
            self.smoes = [
                SMoE(self.args, [SwiGLU(self.args) for _ in range(self.args.num_experts)])
                for _ in range(self.args.n_layers)
            ]

            if self.args.dtype_str == 'fp8':
                self.smoes = [convert_to_float8_training(smoe, config=config, module_filter_fn=module_filter_fn) for smoe in self.smoes]

            if self.args.use_sparse:
                for smoe in self.smoes:
                    swap_linear_with_semi_sparse_linear(smoe, get_sparse_config(smoe))        

            if self.args.torch_compile:
                self.smoes = nn.ModuleList([
                    torch.compile(smoe.to(dtype=self.args.dtype), mode='max-autotune') for smoe in self.smoes
                ])
            else:
                self.smoes = nn.ModuleList(self.smoes)

        elif self.args.moe_type == 'megablocks-moe':
            self.smoes = nn.ModuleList([
                MegablockMoE(self.args)
                for _ in range(self.args.n_layers)
            ])

        elif self.args.moe_type == 'megablocks-dmoe':
            self.smoes = nn.ModuleList([
                MegablockdMoE(self.args)
                for _ in range(self.args.n_layers)
            ])
        
        else:
            print(self.args.moe_type)
            raise ValueError(f"allowed moe types: 'pytorch',  'megablocks-moe', 'megablocks-dmoe' ; got: {self.args.moe_type}")

        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.to(dtype=self.args.dtype)
        
    def pred(self, x):
        
        if self.args.output_moe_weights:
            moe_weights = []

            for attention, moe in zip(self.attentions, self.smoes):

                x = x + attention(x)
                moe_out, moe_w = moe(x)
                moe_weights.append(moe_w)
                x = x + moe_out

            return x, moe_weights

        else:
            for attention, moe in zip(self.attentions, self.smoes):
                x = x + attention(x)
                x = x + moe(x)[0]
        
            return x

    def forward(self, x):
        return self.pred(x)