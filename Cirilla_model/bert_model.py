from LLM_pieces import (
    RoPE,
    SMoE,
    get_activation,
    Expert,
    MegablockMoE,
    MegablockdMoE,
    BertAttention
)
from in_embeddings import InputEmbeddings
from dataclasses import dataclass
import torch.nn as nn
from modules import select_torch_device, get_bertargs_from_hub
from typing import Optional
import warnings
import torch
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from safetensors.torch import load_file

@dataclass
class BertArgs:
    """general"""
    vocab_size:int = 50_000
    dim:int = 1024
    d_ff:int = 2048
    n_layers:int = 4
    output_what:bool = 'meanpool' # 'meanpool' or 'tokens' or 'vocab'
    tie_params:bool = False
    
    """attention"""
    context_window:int = 2048 # max seq len
    n_heads:int = 8
    n_kv_heads:int = 4
    soft_cap:Optional[int] = 20

    """MoE"""
    num_experts:int = 8
    k:int = 4
    moe_type:str = "megablocks-moe" # or "pytorch" or "megablocks-dmoe"
    capacity_factor: float = 1.0
    impl: str = "grouped"   # or "sparse" Sparse MLP is not supported with triton >=3.2.0
    
    """misc"""
    dtype_str:str = 'bfloat16'
    theta:float = 10_000.0
    device:str = select_torch_device()

    @property
    def dtype(self):
        return getattr(torch, self.dtype_str)

    def __post_init__(self):
        if not torch.cuda.is_available():
            warnings.warn("hf kernels only work on cuda")
        assert self.dim % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0

class CirillaBERT(
            nn.Module,
            PyTorchModelHubMixin,
            pipeline_tag="text-generation",
            library_name="pytorch",
            license="mit"
    ):
    def __init__(self, args:BertArgs=None):
        super().__init__()

        if isinstance(args, dict):
               args = BertArgs(**args)
        if args is None:
            args = BertArgs()

        self.args = args
        self._prepare_model()

    def _prepare_model(self):

        self.emb = InputEmbeddings(self.args)
        self.rope = RoPE(self.args.dim // self.args.n_heads, self.args.context_window, self.args.device, self.args.theta, self.args.device)
        activation = get_activation('Motif-Technologies/activation')
        self.rmsnorm = activation.layers.RMSNorm(dim=self.args.dim) if self.args.device == torch.cuda.is_available() else nn.RMSNorm(self.args.dim)

        self.attentions = nn.ModuleList([
            torch.compile(
            BertAttention(self.args, self.rope),
            mode='max-autotune') for _ in range(self.args.n_layers)
        ])
        
        if self.args.moe_type == 'pytorch':
            self.smoes = nn.ModuleList([
                torch.compile(SMoE(self.args, [Expert(self.args) for _ in range(self.args.num_experts)]), mode='max-autotune')
                for _ in range(self.args.n_layers)
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

        if self.args.output_what == 'vocab':

            self.output = nn.Linear(self.args.dim, self.args.vocab_size, bias=False)
            if self.args.tie_params:
                self.output.weight = self.emb.embeddings.weight

        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.to(self.args.device, dtype=self.args.dtype)

    @staticmethod
    def mean_pooling(out, attention_mask):
        if attention_mask is None:
            return torch.mean(out, dim=1)
        
        mask_expanded = attention_mask.unsqueeze(-1).expand(out.size())
        
        sum_embeddings = torch.sum(out * mask_expanded, dim=1)
        
        sum_mask = mask_expanded.sum(dim=1)
        
        return sum_embeddings / torch.clamp(sum_mask, min=1e-9)
        
    def pred(self, x, attention_mask=None):
        
        x = self.emb(x)

        for attention, moe in zip(self.attentions, self.smoes):
            x = x + attention(x) + moe(x)

        if self.args.output_what == 'meanpool':
            return self.mean_pooling(x, attention_mask)
        
        if self.args.output_what == 'tokens':
            return x
        
        x = self.rmsnorm(x)
        x = self.output(x)

        return x
    
    def pull_model_from_hub(self, hf_repo_id:str):
        model_args = self.args
        pulled_args = get_bertargs_from_hub(hf_repo_id)

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
