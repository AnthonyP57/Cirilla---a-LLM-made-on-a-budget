from dataclasses import dataclass
import torch.nn as nn
from .modules import select_torch_device, get_args_from_hub
import warnings
import torch
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from safetensors.torch import load_file
from contextlib import nullcontext
from einops.layers.torch import Rearrange

@dataclass
class TRMArgs:
    """general"""
    vocab_size:int = 70
    dim:int = 256
    tie_params:bool = False
    out_bias:bool = True
    
    """misc"""
    dtype_str:str = 'bfloat16'
    n_total_refinements:int = 4
    n_latent_refinements:int = 2
    device:str = select_torch_device()

    @property
    def dtype(self):
        if self.dtype_str == "fp8":
            return torch.bfloat16 # for initialization, then convert to FP8
        return getattr(torch, self.dtype_str)

    def __post_init__(self):
        if not torch.cuda.is_available():
            warnings.warn("hf kernels only work on cuda")

class InputEmbeddings(nn.Module):
    def __init__(self, args:TRMArgs):
        super().__init__()

        self.embeddings = nn.Embedding(args.vocab_size, args.dim)
    
    def forward(self, x):
        return self.embeddings(x)

class CirillaTRM(
            nn.Module,
            PyTorchModelHubMixin,
            pipeline_tag="text-generation",
            library_name="pytorch",
            license="mit"
    ):
    def __init__(self, network:nn.Module, args:TRMArgs=None):
        super().__init__()

        if isinstance(args, dict):
            args = TRMArgs(**args)

        if args is None:
            args = TRMArgs()

        self.args = args

        self.network = network

        self.emb = InputEmbeddings(self.args)

        self.y_hat_init = nn.Parameter(torch.randn(self.args.dim) * 1e-2)
        self.z_init = nn.Parameter(torch.randn(self.args.dim) * 1e-2)

        self.output = nn.Linear(self.args.dim, self.args.vocab_size, bias=self.args.out_bias)
        if self.args.tie_params:
            self.output.weight = self.emb.embeddings.weight

        self.to_halt = nn.Sequential(
                        nn.Linear(self.args.dim, 1),
                        Rearrange('... 1 -> ...')
                        )

        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.to(self.args.device, dtype=self.args.dtype)

    def get_init(self):
        return self.y_hat_init, self.z_init
    
    def get_halt(self, x, attention_mask=None):
        return self.to_halt(self.mean_pooling(x, attention_mask))
    
    def single_refinement_step(self, x, y_hat, z):

        for _ in range(self.args.n_latent_refinements):

            z = self.network(x + y_hat + z)

        y_hat = self.network(y_hat + z)
        return y_hat, z
    
    def refine(self, x, y_hat, z):

        for step in range(self.args.n_total_refinements):

            is_last_step = step == self.args.n_total_refinements - 1

            context = torch.no_grad if not is_last_step else nullcontext

            with context():
                y_hat, z = self.single_refinement_step(x, y_hat, z)

        return y_hat, z
    
    @staticmethod
    def mean_pooling(out, attention_mask):
        if attention_mask is None:
            return torch.mean(out, dim=1)
        
        mask_expanded = attention_mask.unsqueeze(-1).expand(out.size()).to(out.dtype)
        
        sum_embeddings = torch.sum(out * mask_expanded, dim=1)
        
        sum_mask = mask_expanded.sum(dim=1)
        
        return sum_embeddings / torch.clamp(sum_mask, min=1e-9)

    def forward(self, x, y_hat, z, attention_mask=None):
        
        x = self.emb(x)

        y_hat, z = self.refine(x, y_hat, z)

        pred = self.output(y_hat)

        haltp = self.get_halt(y_hat, attention_mask)

        return pred, y_hat, z, haltp
    
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
