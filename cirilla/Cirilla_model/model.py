from cirilla.LLM_pieces import (
    get_activation,
)
from dataclasses import dataclass
import torch.nn as nn
from .modules import select_torch_device, get_args_from_hub
from .blocks import Decoder
from typing import Optional
import warnings
import torch
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from safetensors.torch import load_file

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
        activation = get_activation('Motif-Technologies/activation')
        self.rmsnorm = activation.layers.RMSNorm(dim=self.args.dim) if self.args.device == torch.cuda.is_available() else nn.RMSNorm(self.args.dim)
        self.decoder = Decoder(self.args)

        self.output = nn.Linear(self.args.dim, self.args.vocab_size, bias=self.args.out_bias)
        if self.args.tie_params:
            self.output.weight = self.emb.embeddings.weight

        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.to(self.args.device, dtype=self.args.dtype)
        
    def pred(self, x):
        
        x = self.emb(x)

        if self.args.output_moe_weights:
            x, moe_weights = self.decoder(x)

            x = self.rmsnorm(x)
            x = self.output(x)

            return x, moe_weights
        
        else:
            x = self.decoder(x)

            x = self.rmsnorm(x)
            x = self.output(x)
        
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
