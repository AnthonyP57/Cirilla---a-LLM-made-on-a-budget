from ..LLM_pieces import get_activation, DynamicTanh, Dynamic_erf
from dataclasses import dataclass
import torch.nn as nn
from .modules import CirillaBaseModel
from .blocks import Encoder, EncoderArgs, InputEmbeddings
import torch
from einops.layers.torch import Rearrange

@dataclass
class BertArgs(EncoderArgs):
    vocab_size:int = 50_000
    output_what:bool = 'meanpool' # 'meanpool' or 'tokens' or 'vocab' or 'classify'
    cls_index:int = None
    n_classes:int = 2
    tie_params:bool = False
    out_bias:bool = True

    def __post_init__(self):
        assert self.output_what in ['meanpool', 'tokens', 'vocab', 'classify']

class CirillaBERT(
            nn.Module,
            CirillaBaseModel,
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
        activation = get_activation('Motif-Technologies/activation')
        if self.args.layer_norm == "RMSNorm":
            self.layer_norm = activation.layers.RMSNorm(dim=self.args.dim) if self.args.device == torch.cuda.is_available() else nn.RMSNorm(self.args.dim)
        elif self.args.layer_norm == "Derf":
            self.layer_norm = Dynamic_erf(self.args.dim)
        elif self.args.layer_norm == "DyT":
            self.layer_norm = DynamicTanh(self.args.dim)
        else:
            raise ValueError(f"allowed layer norms: 'RMSNorm', 'Derf', 'DyT' ; got: {self.args.layer_norm}")
        self.encoder = Encoder(self.args)

        if self.args.output_what == 'vocab':

            self.output = nn.Linear(self.args.dim, self.args.vocab_size, bias=self.args.out_bias)
            if self.args.tie_params:
                self.output.weight = self.emb.embeddings.weight

        elif self.args.output_what == 'classify':
            if self.args.n_classes == 1:
                self.output = nn.Sequential(nn.Linear(self.args.dim, 1, bias=self.args.out_bias), nn.Sigmoid(), Rearrange('... 1 -> ...'))
            else:
                self.output = nn.Linear(self.args.dim, self.args.n_classes, bias=self.args.out_bias)

        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.to(self.args.device, dtype=self.args.dtype)
        
    def pred(self, x, attention_mask=None):
        
        x = self.emb(x)

        if self.args.output_moe_weights:
            x, moe_weights = self.encoder(x)

        else:
            x = self.encoder(x)

        if self.args.output_what == 'meanpool':
            if self.args.output_moe_weights:
                return self.mean_pooling(x, attention_mask), moe_weights
            
            return self.mean_pooling(x, attention_mask)
        
        if self.args.output_what == 'tokens':
            if self.args.output_moe_weights:
                return x, moe_weights
            
            return x
        
        x = self.layer_norm(x)

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

def bert_training_step(self, data): # define a custom training step
    torch.compiler.cudagraph_mark_step_begin()

    out = self.model.pred(data[0], data[1]) # tokens, mask
    loss = self.criterion(out, data[2])
    loss_item = loss.item()
    loss.backward()
    return loss_item

@torch.inference_mode()
def bert_inference_step(self, data): # define a custom inference step
    out = self.model.pred(data[0], data[1]) # tokens, mask
    loss = self.criterion(out, data[2])
    loss_item = loss.item()
    return loss_item