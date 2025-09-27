from cirilla.LLM_pieces import (
    RoPE,
    SMoE,
    SlidingWindowAttention,
    get_activation,
    create_static_block_mask,
    create_dynamic_block_mask,
    sliding_window_causal,
    Expert,
    MegablockMoE,
    MegablockdMoE,
)
from dataclasses import dataclass
import torch.nn as nn
from .modules import select_torch_device, get_args_from_hub
from typing import Optional
import warnings
import torch
from attn_gym.mods import generate_tanh_softcap
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from safetensors.torch import load_file
from torchao.float8 import convert_to_float8_training, Float8LinearConfig
from torchao.sparsity.training import (
    SemiSparseLinear,
    swap_linear_with_semi_sparse_linear,
)

@dataclass
class Args:
    """general"""
    vocab_size:int = 60_000
    dim:int = 1024
    d_ff:int = 2048
    n_layers:int = 16
    tie_params:bool = False
    out_bias:bool = True
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
    device:str = select_torch_device()

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

class InputEmbeddings(nn.Module):
    def __init__(self, args:Args):
        super().__init__()

        self.embeddings = nn.Embedding(args.vocab_size, args.dim)
    
    def forward(self, x):
        return self.embeddings(x)

class Cirilla(
            nn.Module,
            PyTorchModelHubMixin,
            pipeline_tag="text-generation",
            library_name="pytorch",
            license="mit"
    ):
    def __init__(self, args:Args=None):
        super().__init__()

        if isinstance(args, dict):
            args = Args(**args)

        if args is None:
            args = Args()

        self.args = args
        self._prepare_model()

    def _prepare_model(self):

        self.emb = InputEmbeddings(self.args)
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

        self.attentions = nn.ModuleList([
            torch.compile(attention, mode='max-autotune') for attention in self.attentions
            ])
        
        if self.args.moe_type == 'pytorch':
            self.smoes = [
                SMoE(self.args, [Expert(self.args) for _ in range(self.args.num_experts)])
                for _ in range(self.args.n_layers)
            ]

            if self.args.dtype_str == 'fp8':
                self.smoes = [convert_to_float8_training(smoe, config=config, module_filter_fn=module_filter_fn) for smoe in self.smoes]

            if self.args.use_sparse:
                for smoe in self.smoes:
                    swap_linear_with_semi_sparse_linear(smoe, get_sparse_config(smoe))        

            self.smoes = nn.ModuleList([
                torch.compile(smoe, mode='max-autotune') for smoe in self.smoes
            ])

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

        self.output = nn.Linear(self.args.dim, self.args.vocab_size, bias=self.args.out_bias)
        if self.args.tie_params:
            self.output.weight = self.emb.embeddings.weight

        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.to(self.args.device, dtype=self.args.dtype)
        
    def pred(self, x):
        
        x = self.emb(x)

        if self.args.output_moe_weights:
            moe_weights = []

            for attention, moe in zip(self.attentions, self.smoes):

                x = x + attention(x)
                moe_out, moe_w = moe(x)
                moe_weights.append(moe_w)
                x = x + moe_out

        else:
            for attention, moe in zip(self.attentions, self.smoes):
                x = x + attention(x)
                x = x + moe(x)[0]

        x = self.rmsnorm(x)
        x = self.output(x)
        
        if self.args.output_moe_weights:
            return x, moe_weights
        
        return x

    def forward(self, x):
        return self.pred(x)
    
    def pull_model_from_hub(self, hf_repo_id:str):
        model_args = self.args
        pulled_args = get_args_from_hub(hf_repo_id)

        if model_args != pulled_args:
            print(f"Current model args don't correspond to the HF model's args.\nCurrent args:\n{model_args}\nThe model will use the HF args:\n{pulled_args}")
            self.args = pulled_args
            self._prepare_model()

        file_path = hf_hub_download(
            repo_id=hf_repo_id,
            filename="model.safetensors",
        )

        loaded = load_file(file_path)
        if "output.weight" not in loaded:
            loaded['output.weight'] = loaded["emb.embeddings.weight"]

        self.load_state_dict(loaded)
