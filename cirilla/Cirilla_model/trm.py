from dataclasses import dataclass
import torch.nn as nn
from .modules import select_torch_device, CirillaBaseModel
import warnings
import torch
from contextlib import nullcontext
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from .blocks import InputEmbeddings

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

class CirillaTRM(
            nn.Module,
            CirillaBaseModel,
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

    def forward(self, x, y_hat, z, attention_mask=None):
        
        x = self.emb(x)

        y_hat, z = self.refine(x, y_hat, z)

        pred = self.output(y_hat)

        haltp = self.get_halt(y_hat, attention_mask)

        return pred, y_hat, z, haltp
    
    @torch.inference_mode()
    def predict(self, x, attention_mask=None, halt_thresh=0.5, max_recurrent_step=16):
        
        y_hat, z = self.get_init()

        preds = []
        pred_indices = []
        n_steps = []
        active_batch_indices = torch.arange(x.shape[0], device=x.device)

        for step in range(max_recurrent_step):

            pred, y_hat, z, haltp = self.forward(x, y_hat, z, attention_mask)

            halt_mask = (F.sigmoid(haltp) < halt_thresh) & (step < max_recurrent_step - 1)

            if halt_mask.all():
                continue
            
            y_hat = y_hat[halt_mask]
            z = z[halt_mask]
            x = x[halt_mask]
            attention_mask = attention_mask[halt_mask]

            preds.append(pred[~halt_mask])
            n_steps.extend([step] * (~halt_mask).sum().item())
            pred_indices.append(active_batch_indices[~halt_mask])
            active_batch_indices = active_batch_indices[halt_mask]

            if z.numel() == 0: # if is empty
                break

        preds = torch.cat(preds, dim=0)
        n_steps = torch.tensor(n_steps).to(x.device)
        pred_indices = torch.cat(pred_indices, dim=0).to(x.device)
        
        preds = preds[pred_indices]
        n_steps = n_steps[pred_indices]

        return preds, n_steps

def trm_training_step(self, data, max_recurrent_step=16, halt_weight=0.5, halt_thresh=0.5, ema_model=None):

    step_loss = 0
    n = 0

    y_hat, z = self.model.get_init()
    x = data[0]
    mask = data[1]

    for _ in range(max_recurrent_step):

        torch.compiler.cudagraph_mark_step_begin()

        pred, y_hat, z, haltp = self.model(x, y_hat, z, mask)

        loss = F.cross_entropy(pred.view(-1, self.model.args.vocab_size), x.view(-1))

        all_correct = (pred.argmax(dim=-1) == x).all(dim=-1)

        halt_loss = F.binary_cross_entropy_with_logits(haltp, all_correct.to(haltp.dtype))

        loss = loss + halt_weight * halt_loss

        step_loss += loss.item()
        n += 1

        loss.backward()
        if ema_model is not None:
            ema_model.update() # model needs to have .predict() method

        halt_mask = F.sigmoid(haltp) < halt_thresh
        
        y_hat, z = y_hat.detach(), z.detach()

        if halt_mask.all():
            continue
        
        y_hat = y_hat[halt_mask]
        z = z[halt_mask]
        x = x[halt_mask]
        mask = mask[halt_mask]

        if z.numel() == 0: # if is empty
            break

    return step_loss / n

@torch.inference_mode()
def trm_inference_step(self, data, max_recurrent_step=16, halt_thresh=0.5):
    
    x = data[0]
    mask = data[1]

    preds, n_steps = self.model.predict(x, mask, halt_thresh, max_recurrent_step)

    loss = F.cross_entropy(preds.view(-1, self.model.args.vocab_size), x.view(-1))

    return loss.item()