import time
import numpy as np
import torch
import torch.nn as nn
from model import Radovid, Args
from torch.utils.data import DataLoader

def train_radovid(n_epochs:int, model:Radovid, args:Args, dataloader:DataLoader, optim:torch.optim=None, lr:float=5e-5, **optim_kwargs) -> Radovid:

    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.capture_scalar_outputs = True

    model = Radovid(args)

    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    print(f'{(model.n_params/1e6):.2f} M')

    criterion = torch.nn.CrossEntropyLoss()

    if optim is None:
        optimizer_dict = {p: torch.optim.AdamW([p], fused=True, foreach=False, lr=lr) for p in model.parameters()}
    else:
        optimizer_dict = {p: optim([p], lr=lr, **optim_kwargs) for p in model.parameters()}

    def optimizer_hook(parameter) -> None:
        optimizer_dict[parameter].step()
        optimizer_dict[parameter].zero_grad(set_to_none=True)

    for p in model.parameters():
        p.register_post_accumulate_grad_hook(optimizer_hook)


    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
    torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None

    for epoch in range(n_epochs):
        for x, y in dataloader:

            torch.compiler.cudagraph_mark_step_begin()
            out = model.pred(x)
            loss = criterion(out.view(-1, args.vocab_size), y.view(-1))

            loss.backward()

            
            