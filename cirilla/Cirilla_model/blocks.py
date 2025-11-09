from ..LLM_pieces import (
    RoPE,
    SMoE,
    get_activation,
    SwiGLU,
    MegablockMoE,
    MegablockdMoE,
    BertAttention,
    SlidingWindowAttention,
    create_static_block_mask,
    create_dynamic_block_mask,
    sliding_window_causal,
)
from attn_gym.mods import generate_tanh_softcap
from dataclasses import dataclass
from .modules import select_torch_device
import torch.nn as nn
from typing import Optional
import warnings
import torch
import torch.nn.functional as F
from torchao.float8 import convert_to_float8_training, Float8LinearConfig
from torchao.sparsity.training import (
    SemiSparseLinear,
    swap_linear_with_semi_sparse_linear,
)

@dataclass
class EncoderArgs:
    """general"""
    dim:int = 256
    d_ff:int = 256
    n_layers:int = 2
    output_moe_weights:bool = False
    
    """attention"""
    context_window:int = 512 # max seq len
    n_heads:int = 2
    n_kv_heads:int = 2
    soft_cap:Optional[int] = 20

    """MoE"""
    num_experts:int = 2
    k:int = 1
    moe_type:str = "pytorch" # or "pytorch" or "megablocks-moe" or "megablocks-dmoe"
    moe_zloss_weight:float = 0.1
    capacity_factor: float = 1.0
    impl: str = "grouped"   # or "sparse" Sparse MLP is not supported with triton >=3.2.0
    
    """misc"""
    dtype_str:str = 'bfloat16'
    fp8_recipe:str="tensorwise" # tensorwise (fastest), rowwise, rowwise_with_gw_hp (most accurate)
    use_sparse:bool = False
    theta:float = 10_000.0
    device:str = select_torch_device()
    torch_compile:bool=True

    @property
    def dtype(self):
        if self.dtype_str == "fp8":
            return torch.bfloat16 # for initialization, then convert to FP8
        return getattr(torch, self.dtype_str)

    def __post_init__(self):
        if not torch.cuda.is_available():
            warnings.warn("hf kernels only work on cuda")
        assert self.dim % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
        if self.use_sparse:
            assert self.dtype_str != "fp8"
        if self.output_moe_weights:
            assert self.moe_type == "pytorch"

class Encoder(nn.Module):
    
    def __init__(self, args:EncoderArgs=None):
        super().__init__()

        if isinstance(args, dict):
            args = EncoderArgs(**args)

        if args is None:
            args = EncoderArgs()

        self.args = args
        self._prepare_model()

    def _prepare_model(self):

        self.rope = RoPE(self.args.dim // self.args.n_heads, self.args.context_window, self.args.device, self.args.theta, self.args.device)
        activation = get_activation('Motif-Technologies/activation')
        self.rmsnorm = activation.layers.RMSNorm(dim=self.args.dim) if self.args.device == torch.cuda.is_available() else nn.RMSNorm(self.args.dim)


        self.attentions = [
            BertAttention(self.args, self.rope)
            for _ in range(self.args.n_layers)
            ]

        if self.args.dtype_str == "fp8":

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


@dataclass
class DecoderArgs:
    """general"""
    dim:int = 1024
    d_ff:int = 2048
    n_layers:int = 16
    output_moe_weights:bool = False

    """attention"""
    context_window:int = 2048 # max seq len
    window_size:int = 1024
    n_heads:int = 8
    n_kv_heads:int = 4
    static_mask:bool = True
    soft_cap:Optional[int] = 20

    """MoE"""
    num_experts:int = 8
    k:int = 4
    moe_type:str = "pytorch" # or "pytorch" or "megablocks-moe" or "megablocks-dmoe"
    capacity_factor: float = 1.0
    moe_zloss_weight:float = 0.1
    impl: str = "grouped"   # or "sparse" Sparse MLP is not supported with triton >=3.2.0
    
    """misc"""
    dtype_str:str = 'bfloat16'
    fp8_recipe:str="tensorwise" # tensorwise (fastest), rowwise, rowwise_with_gw_hp (most accurate)
    use_sparse:bool = False
    theta:float = 10_000.0
    device = select_torch_device()
    torch_compile:bool=True

    @property
    def dtype(self):
        if self.dtype_str == "fp8":
            return torch.bfloat16 # for initialization, then convert to FP8
        return getattr(torch, self.dtype_str)

    def __post_init__(self):
        if not torch.cuda.is_available():
            warnings.warn("hf kernels only work on cuda")
        assert self.dim % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
        if self.use_sparse:
            assert self.dtype_str != "fp8"
        if self.output_moe_weights:
            assert self.moe_type == "pytorch"

class Decoder(nn.Module):

    def __init__(self, args:DecoderArgs=None):
        super().__init__()

        if isinstance(args, dict):
            args = DecoderArgs(**args)

        if args is None:
            args = DecoderArgs()

        self.args = args
        self._prepare_model()

    def _prepare_model(self):

        self.rope = RoPE(self.args.dim // self.args.n_heads, self.args.context_window, self.args.device, self.args.theta, self.args.device)
        activation = get_activation('Motif-Technologies/activation')
        self.rmsnorm = activation.layers.RMSNorm(dim=self.args.dim) if self.args.device == torch.cuda.is_available() else nn.RMSNorm(self.args.dim)
        
        if self.args.static_mask:
            self.mask = create_static_block_mask(sliding_window_causal,self.args.context_window,
                                            self.args.context_window, self.args.device, self.args.window_size)

            self.attentions = [
                SlidingWindowAttention(self.args, self.rope, self.mask, generate_tanh_softcap(self.args.soft_cap, approx=False) if self.args.soft_cap is not None else None)
                    for _ in range(self.args.n_layers)
            ]

        else:
            self.attentions = [
                SlidingWindowAttention(self.args, self.rope,
                create_dynamic_block_mask,
                generate_tanh_softcap(self.args.soft_cap, approx=False) if self.args.soft_cap is not None else None)
                    for _ in range(self.args.n_layers)
            ]

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


@dataclass
class MixerArgs:
    """general"""
    dim: int = 256
    depth: int = 8
    context_window: int = 512
    expansion_factor: float = 4
    expansion_factor_token: float = 0.5
    dropout: float = 0.0
    dtype_str: str = 'bfloat16'
    device = select_torch_device()

    @property
    def dtype(self):
        return getattr(torch, self.dtype_str)

class MLPMixer1D(nn.Module):
    def __init__(self, args: Optional[MixerArgs] = None):
        super().__init__()

        if isinstance(args, dict):
            args = MixerArgs(**args)
        if args is None:
            args = MixerArgs()

        self.args = args
        self._prepare_model()

    def _prepare_model(self):
        layers = []
        for _ in range(self.args.depth):
            # Token mixing
            layers.append(PreNormResidual(
                self.args.dim,
                FeedForward(self.args.context_window, int(self.args.expansion_factor * self.args.dim),
                                self.args.dropout, dense_type='conv1d')
            ))
            # Channel mixing
            layers.append(PreNormResidual(
                self.args.dim,
                FeedForward(self.args.dim, int(self.args.expansion_factor_token * self.args.dim),
                                self.args.dropout, dense_type='linear')
            ))

        self.norm = nn.LayerNorm(self.args.dim, bias = False)

        layers.append(self.norm)
        self.mixer = nn.Sequential(*layers)

        self.to(dtype=self.args.dtype, device=self.args.device)

        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def pred(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pred(x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim, bias = False)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class FeedForward(nn.Module):
    def __init__(self, dim_in, dim_hidden, dropout=0., dense_type='linear'):
        super().__init__()
        if dense_type == 'conv1d':
            self.net = nn.Sequential(
                nn.Conv1d(dim_in, dim_hidden, kernel_size=1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(dim_hidden, dim_in, kernel_size=1),
                nn.Dropout(dropout)
            )
        elif dense_type == 'linear':
            self.net = nn.Sequential(
                nn.Linear(dim_in, dim_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_hidden, dim_in),
                nn.Dropout(dropout)
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.net(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=768, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x ~ (B, C, H, W)
        x = self.proj(x)  # (B, D, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, H/ps * W/ps, D)
        x = self.norm(x)
        return x

class VisionEmbeddingModel(nn.Module):
    def __init__(self,
                    in_ch=3,
                    embed_dim=768,
                    patch_size=16,
                    H=16,
                    W=16
                    ):
        super().__init__()
        self.patch_embed = PatchEmbed(in_ch=in_ch, embed_dim=embed_dim, patch_size=patch_size)
        self.token_norm = nn.LayerNorm(embed_dim)
        self.pos_embed = self._make_pos_embed((H, W), embed_dim)

    def _make_pos_embed(self, grid_hw, embed_dim):
        H, W = grid_hw
        pe = nn.Parameter(torch.zeros(1, H * W, embed_dim))
        nn.init.trunc_normal_(pe, std=0.02)
        return pe

    def forward(self, x):
        # x ~ (B, C, H, W)
        tokens = self.patch_embed(x)  # (B, N, D)

        tokens = tokens + self.pos_embed
        tokens = self.token_norm(tokens)

        return tokens

class KeylessAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.wi = nn.Linear(dim, 1, bias=False)
        self.wt = nn.Linear(dim, 1, bias=False)

    def forward(self, cls_image, cls_text):
        ei = self.wi(cls_image)
        et = self.wt(cls_text)
        lmb = F.sigmoid(ei - et)

        return lmb * cls_image + ((1 - lmb) * cls_text)

@dataclass
class EmbedArgs:
    vocab_size: int
    dim: int

class InputEmbeddings(nn.Module):
    def __init__(self, args: EmbedArgs):
        super().__init__()

        self.embeddings = nn.Embedding(args.vocab_size, args.dim)
    
    def forward(self, x):
        return self.embeddings(x)