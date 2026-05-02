from ..LLM_pieces import (
    RoPE,
    SMoE,
    SwiGLU,
    BertAttention,
    SlidingWindowAttention,
    create_static_block_mask,
    create_dynamic_block_mask,
    sliding_window_causal,
    DynamicTanh,
    Dynamic_erf
)
from attn_gym.mods import generate_tanh_softcap
from torch.nn.attention.flex_attention import flex_attention as _flex_attention
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
    moe_type:str = "pytorch" # "pytorch"
    
    """misc"""
    dtype_str:str = 'bfloat16'
    fp8_recipe:str="tensorwise" # tensorwise (fastest), rowwise, rowwise_with_gw_hp (most accurate)
    use_sparse:bool = False
    theta:float = 10_000.0
    device:str = select_torch_device()
    torch_compile:bool=True
    layer_norm:str = "RMSNorm" # or "Derf" or "DyT"

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
        if self.args.layer_norm == "RMSNorm":
            self.layer_norm = nn.RMSNorm(self.args.dim)
        elif self.args.layer_norm == "Derf":
            self.layer_norm = Dynamic_erf(self.args.dim)
        elif self.args.layer_norm == "DyT":
            self.layer_norm = DynamicTanh(self.args.dim)
        else:
            raise ValueError(f"allowed layer norms: 'RMSNorm', 'Derf', 'DyT' ; got: {self.args.layer_norm}")

        self.attentions = [
            BertAttention(self.args, self.rope, generate_tanh_softcap(self.args.soft_cap, approx=False) if self.args.soft_cap is not None else None)
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
        
        else:
            print(self.args.moe_type)
            raise ValueError(f"allowed moe types: 'pytorch' ; got: {self.args.moe_type}")

        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.to(dtype=self.args.dtype)
        
    def pred(self, x) -> torch.Tensor:
        
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
            
    def forward(self, x) -> torch.Tensor:
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
    moe_type:str = "pytorch" # "pytorch"
    
    """misc"""
    dtype_str:str = 'bfloat16'
    fp8_recipe:str="tensorwise" # tensorwise (fastest), rowwise, rowwise_with_gw_hp (most accurate)
    use_sparse:bool = False
    theta:float = 10_000.0
    device = select_torch_device()
    torch_compile:bool=True
    layer_norm:str = "RMSNorm" # or "Derf" or "DyT"

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
        
        else:
            print(self.args.moe_type)
            raise ValueError(f"allowed moe types: 'pytorch'; got: {self.args.moe_type}")

        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.to(dtype=self.args.dtype)
        
    def pred(self, x) -> torch.Tensor:
        
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

    def forward(self, x) -> torch.Tensor:
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

    def forward(self, x) -> torch.Tensor:
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

    def forward(self, x) -> torch.Tensor:
        x = self.net(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=768, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x) -> torch.Tensor:
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

    def forward(self, x) -> torch.Tensor:
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

    def forward(self, cls_image, cls_text) -> torch.Tensor:
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
    
    def forward(self, x) -> torch.Tensor:
        return self.embeddings(x)


# Swin Transformer Encoder

def _window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """(B, H, W, C) → (B*nW, ws, ws, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def _window_reverse(windows: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    """(B*nW, ws, ws, C) → (B, H, W, C)"""
    nW = (H // ws) * (W // ws)
    B = windows.shape[0] // nW
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class _SwinWindowAttn(nn.Module):
    """Window multi-head self-attention with learnable relative position bias, using FlexAttention."""

    def __init__(self, dim: int, window_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.ws = window_size

        self.rel_pos_bias = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing='ij'))
        coords_flat = coords.flatten(1)                              # (2, ws*ws)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]     # (2, ws*ws, ws*ws)
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer('rel_idx', rel.sum(-1))                 # (ws*ws, ws*ws)

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B_, num_heads, N, head_dim)

        # score_mod: add relative position bias (+ SW-MSA additive mask when present)
        rel_bias = self.rel_pos_bias  # ((2ws-1)^2, num_heads)
        rel_idx  = self.rel_idx       # (ws*ws, ws*ws)

        if mask is not None:
            # mask: (nW, ws*ws, ws*ws) with 0 (allow) or -100 (block) — same convention as before
            nW = mask.shape[0]
            def score_mod(score, b, h, q_idx, kv_idx):
                return score + rel_bias[rel_idx[q_idx, kv_idx], h] + mask[b % nW, q_idx, kv_idx]
        else:
            def score_mod(score, b, h, q_idx, kv_idx):
                return score + rel_bias[rel_idx[q_idx, kv_idx], h]

        out = _flex_attention(q, k, v, score_mod=score_mod)
        x = out.transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class _SwinBlock(nn.Module):
    """One Swin Transformer block (W-MSA or SW-MSA + MLP)."""

    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int,
                 mlp_ratio: float, dropout: float):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _SwinWindowAttn(dim, window_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, H: int, W: int,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, _, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        if pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = _window_partition(x, self.window_size).view(-1, self.window_size ** 2, C)
        attn_out = self.attn(windows, mask=attn_mask).view(-1, self.window_size, self.window_size, C)
        x = _window_reverse(attn_out, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + x.view(B, H * W, C)
        x = x + self.mlp(self.norm2(x))
        return x


class _PatchMerging(nn.Module):
    """2× spatial downsampling with learned projection (dim → 2*dim)."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, _, C = x.shape
        x = x.view(B, H, W, C)
        pad_H, pad_W = H % 2, W % 2
        if pad_H or pad_W:
            x = F.pad(x, (0, 0, 0, pad_W, 0, pad_H))
        Hp, Wp = H + pad_H, W + pad_W
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)
        x = self.reduction(self.norm(x.view(B, -1, 4 * C)))
        return x, Hp // 2, Wp // 2


class _SwinStage(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int,
                 mlp_ratio: float, dropout: float, downsample: bool):
        super().__init__()
        self.blocks = nn.ModuleList([
            _SwinBlock(dim, num_heads, window_size,
                       shift_size=0 if i % 2 == 0 else window_size // 2,
                       mlp_ratio=mlp_ratio, dropout=dropout)
            for i in range(depth)
        ])
        self.downsample = _PatchMerging(dim) if downsample else None

    @staticmethod
    def _make_attn_mask(Hp: int, Wp: int, ws: int, shift: int, device) -> torch.Tensor:
        img = torch.zeros(1, Hp, Wp, 1, device=device)
        cnt = 0
        for h in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
            for w in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
                img[:, h, w, :] = cnt
                cnt += 1
        wins = _window_partition(img, ws).view(-1, ws * ws)
        mask = wins.unsqueeze(1) - wins.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

    def forward(self, x: torch.Tensor, H: int, W: int):
        for block in self.blocks:
            pad_b = (block.window_size - H % block.window_size) % block.window_size
            pad_r = (block.window_size - W % block.window_size) % block.window_size
            Hp, Wp = H + pad_b, W + pad_r
            mask = self._make_attn_mask(Hp, Wp, block.window_size, block.shift_size, x.device) \
                   if block.shift_size > 0 else None
            x = block(x, H, W, mask)
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        return x, H, W


@dataclass
class SwinArgs:
    img_size: int = 224
    patch_size: int = 4
    in_channels: int = 3
    embed_dim: int = 96
    depths: tuple = (2, 2, 6, 2)
    num_heads: tuple = (3, 6, 12, 24)
    window_size: int = 7
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    dtype_str: str = 'bfloat16'
    device: str = select_torch_device()

    @property
    def dtype(self):
        return getattr(torch, self.dtype_str)

    @property
    def out_dim(self) -> int:
        return self.embed_dim * (2 ** (len(self.depths) - 1))

    def __post_init__(self):
        assert len(self.depths) == len(self.num_heads), "depths and num_heads must have the same length"
        assert self.img_size % self.patch_size == 0, "img_size must be divisible by patch_size"


class SwinEncoder(nn.Module):
    """Swin Transformer V1 image encoder. Input (B, C, H, W) → output (B, N_tokens, out_dim)."""

    def __init__(self, args: SwinArgs = None):
        super().__init__()
        if isinstance(args, dict):
            args = SwinArgs(**args)
        if args is None:
            args = SwinArgs()
        self.args = args
        self._prepare_model()

    def _prepare_model(self):
        a = self.args
        self.patch_embed = nn.Conv2d(a.in_channels, a.embed_dim, kernel_size=a.patch_size, stride=a.patch_size)
        self.patch_norm = nn.LayerNorm(a.embed_dim)
        self.pos_drop = nn.Dropout(a.dropout)
        self.stages = nn.ModuleList([
            _SwinStage(
                dim=a.embed_dim * (2 ** i),
                depth=a.depths[i],
                num_heads=a.num_heads[i],
                window_size=a.window_size,
                mlp_ratio=a.mlp_ratio,
                dropout=a.dropout,
                downsample=(i < len(a.depths) - 1),
            )
            for i in range(len(a.depths))
        ])
        self.norm = nn.LayerNorm(a.out_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.to(dtype=a.dtype, device=a.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, N_tokens, out_dim)"""
        x = self.patch_embed(x)                                     # (B, embed_dim, H/ps, W/ps)
        B, C, H, W = x.shape
        x = self.patch_norm(x.flatten(2).transpose(1, 2))           # (B, H*W, embed_dim)
        x = self.pos_drop(x)
        for stage in self.stages:
            x, H, W = stage(x, H, W)
        return self.norm(x)                                         # (B, N_tokens, out_dim)