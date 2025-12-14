from cirilla.LLM_pieces import get_activation, DynamicTanh, Dynamic_erf
from dataclasses import dataclass
import torch.nn as nn
from .modules import CirillaBaseModel
from .blocks import Decoder, DecoderArgs, InputEmbeddings
import torch

@dataclass
class Args(DecoderArgs):
    vocab_size:int = 60_000
    tie_params:bool = False
    out_bias:bool = False

class Cirilla(
            nn.Module,
            CirillaBaseModel,
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
        if self.args.layer_norm == "RMSNorm":
            self.layer_norm = activation.layers.RMSNorm(dim=self.args.dim) if self.args.device == torch.cuda.is_available() else nn.RMSNorm(self.args.dim)
        elif self.args.layer_norm == "Derf":
            self.layer_norm = Dynamic_erf(self.args.dim)
        elif self.args.layer_norm == "DyT":
            self.layer_norm = DynamicTanh(self.args.dim)
        else:
            raise ValueError(f"allowed layer norms: 'RMSNorm', 'Derf', 'DyT' ; got: {self.args.layer_norm}")
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

            x = self.layer_norm(x)
            x = self.output(x)

            return x, moe_weights
        
        else:
            x = self.decoder(x)

            x = self.layer_norm(x)
            x = self.output(x)
        
            return x

    def forward(self, x):
        return self.pred(x)
