from cirilla.LLM_pieces import get_activation
from dataclasses import dataclass
import torch.nn as nn
from .modules import CirillaBaseModel
from .blocks import Decoder, DecoderArgs, InputEmbeddings
import torch
import torch.nn.functional as F

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

def mtp_training_step(self, data, pad_id):
    step_loss = 0.0
    n = 0

    torch.compiler.cudagraph_mark_step_begin()
    
    x = data[0]
    y = data[1]
    
    y_fill = torch.tensor([[pad_id] * self.model.args.n_token_heads] * y.shape[0], dtype=y.dtype, device=y.device)
    y = torch.hstack([y, y_fill])

    z = self.model.get_z(x)
    zd = z.detach()
    zd.requires_grad = True

    for i in range(self.model.args.n_token_heads):
        preds = self.model.get_heads(i, zd)
        loss = F.cross_entropy(
            preds.view(-1, self.model.args.vocab_size),
            y[:, i:-(self.model.args.n_token_heads - i)].reshape(-1),
            ignore_index=pad_id)
        step_loss += loss.item()
        n += 1
        loss.backward()

    z.backward(gradient=zd.grad)

    return step_loss / n

@torch.inference_mode()
def mtp_inference_step(self, data, pad_id):

    x = data[0]
    y = data[1]
    
    z = self.model.get_z(x)
    zd = z.detach()
    zd.requires_grad = True

    preds = self.model.get_heads(0, zd)
    loss = F.cross_entropy(
        preds.view(-1, self.model.args.vocab_size),
        y.reshape(-1),
        ignore_index=pad_id)

    return loss.item()