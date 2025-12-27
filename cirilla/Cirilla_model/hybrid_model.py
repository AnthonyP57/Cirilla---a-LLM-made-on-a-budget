from cirilla.Cirilla_model import Cirilla
from cirilla.LLM_pieces import DynamicTanh, Dynamic_erf
from .mamba_blocks import HybridDecoder, HybridDecoderArgs
from dataclasses import dataclass
import torch.nn as nn
from .blocks import InputEmbeddings


@dataclass
class HybridArgs(HybridDecoderArgs):
    vocab_size:int = 60_000
    tie_params:bool = False
    out_bias:bool = False

class HybridCirilla(Cirilla):

    def __init__(self, args:HybridArgs=None):
        super().__init__(args)

    def _prepare_model(self):

        self.emb = InputEmbeddings(self.args)
        if self.args.layer_norm == "RMSNorm":
            self.layer_norm = nn.RMSNorm(self.args.dim)
        elif self.args.layer_norm == "Derf":
            self.layer_norm = Dynamic_erf(self.args.dim)
        elif self.args.layer_norm == "DyT":
            self.layer_norm = DynamicTanh(self.args.dim)
        else:
            raise ValueError(f"allowed layer norms: 'RMSNorm', 'Derf', 'DyT' ; got: {self.args.layer_norm}")
        self.decoder = HybridDecoder(self.args)

        self.output = nn.Linear(self.args.dim, self.args.vocab_size, bias=self.args.out_bias)
        if self.args.tie_params:
            self.output.weight = self.emb.embeddings.weight

        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.to(self.args.device, dtype=self.args.dtype)