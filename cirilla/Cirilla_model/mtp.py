from cirilla.LLM_pieces import get_activation
from dataclasses import dataclass
import torch.nn as nn
from .modules import CirillaBaseModel
from .blocks import Decoder, DecoderArgs, InputEmbeddings
import torch

@dataclass
class MTPArgs(DecoderArgs):
    n_token_heads:int = 4
    vocab_size:int = 60_000
    tie_params:bool = False
    out_bias:bool = False

class CirillaMTP(
            nn.Module,
            CirillaBaseModel,
            pipeline_tag="text-generation",
            library_name="pytorch",
            license="mit"
    ):
    def __init__(self, args:MTPArgs=None):
        super().__init__()

        if isinstance(args, dict):
            args = MTPArgs(**args)

        if args is None:
            args = MTPArgs()

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

        token_args = {k:v for k,v in self.args.__dict__.items() if k in DecoderArgs.__dataclass_fields__}
        token_args['n_layers'] = 1
        self.token_head_args = DecoderArgs(**token_args)

        if torch.cuda.is_available():
            self.token_heads = [nn.Sequential(Decoder(self.token_head_args), activation.layers.RMSNorm(dim=self.args.dim)) for _ in range(self.args.n_token_heads)]
        else:
            self.token_heads = [nn.Sequential(Decoder(self.token_head_args), nn.RMSNorm(self.args.dim)) for _ in range(self.args.n_token_heads)]
        
        self.token_heads = nn.ModuleList(self.token_heads)

        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.to(self.args.device, dtype=self.args.dtype)
        
    def get_z(self, x):
        
        x = self.emb(x)

        if self.args.output_moe_weights:
            x, moe_weights = self.decoder(x)

            x = self.rmsnorm(x)

            return x, moe_weights
        
        else:
            x = self.decoder(x)

            x = self.rmsnorm(x)
        
            return x
        
    def get_heads(self, idx, z):
        return self.output(self.token_heads[idx](z))

    def forward(self, x):
        x = self.get_z(x)
        return [head for head in self.get_heads(x)]
