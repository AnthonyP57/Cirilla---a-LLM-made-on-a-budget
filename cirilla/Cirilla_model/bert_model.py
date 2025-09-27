from ..LLM_pieces import (
    RoPE,
    SMoE,
    get_activation,
    Expert,
    MegablockMoE,
    MegablockdMoE,
    BertAttention
)
from dataclasses import dataclass
import torch.nn as nn
from .modules import select_torch_device, get_bertargs_from_hub
from typing import Optional
import warnings
import torch
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from safetensors.torch import load_file
from torchao.float8 import convert_to_float8_training, Float8LinearConfig
from torchao.sparsity.training import (
    SemiSparseLinear,
    swap_linear_with_semi_sparse_linear,
)

@dataclass
class BertArgs:
    """general"""
    vocab_size:int = 50_000
    dim:int = 1024
    d_ff:int = 2048
    n_layers:int = 4
    output_what:bool = 'meanpool' # 'meanpool' or 'tokens' or 'vocab' or 'classify'
    cls_index:int = None
    n_classes:int = 2
    tie_params:bool = False
    out_bias:bool = True
    output_moe_weights:bool = False
    
    """attention"""
    context_window:int = 2048 # max seq len
    n_heads:int = 8
    n_kv_heads:int = 4
    soft_cap:Optional[int] = 20

    """MoE"""
    num_experts:int = 8
    k:int = 4
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
    def __init__(self, args:BertArgs):
        super().__init__()

        self.embeddings = nn.Embedding(args.vocab_size, args.dim)
    
    def forward(self, x):
        return self.embeddings(x)

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

        if self.args.output_what == 'vocab':

            self.output = nn.Linear(self.args.dim, self.args.vocab_size, bias=self.args.out_bias)
            if self.args.tie_params:
                self.output.weight = self.emb.embeddings.weight

        elif self.args.output_what == 'classify':
            self.output = nn.Linear(self.args.dim, self.args.n_classes, bias=self.args.out_bias)

        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.to(self.args.device, dtype=self.args.dtype)

    @staticmethod
    def mean_pooling(out, attention_mask):
        if attention_mask is None:
            return torch.mean(out, dim=1)
        
        mask_expanded = attention_mask.unsqueeze(-1).expand(out.size()).to(out.dtype)
        
        sum_embeddings = torch.sum(out * mask_expanded, dim=1)
        
        sum_mask = mask_expanded.sum(dim=1)
        
        return sum_embeddings / torch.clamp(sum_mask, min=1e-9)
        
    def pred(self, x, attention_mask=None):
        
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

        if self.args.output_what == 'meanpool':
            if self.args.output_moe_weights:
                return self.mean_pooling(x, attention_mask), moe_weights
            
            return self.mean_pooling(x, attention_mask)
        
        if self.args.output_what == 'tokens':
            if self.args.output_moe_weights:
                return x, moe_weights
            
            return x
        
        x = self.rmsnorm(x)

        if self.args.output_what == 'classify':
            if self.args.cls_index is None:
                x = self.mean_pooling(x, attention_mask)
            else:
                x = x[:, self.args.cls_index]

        x = self.output(x)

        if self.args.output_moe_weights:
            return x, moe_weights
        
        return x
    
    def forward(self, x, attention_mask=None):
        return self.pred(x, attention_mask)
    
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
