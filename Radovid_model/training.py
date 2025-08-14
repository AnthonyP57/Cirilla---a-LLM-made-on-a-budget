import torch
import torch.nn as nn
from model import Radovid
from torch.utils.data import DataLoader
from functools import partial
from dataclasses import dataclass, field
from torch.optim import Optimizer, AdamW

@dataclass
class TrainingArgs:
    n_epoch:int = 100
    optim:Optimizer = AdamW
    lr:float = 5e-5
    xavier_init:bool = True
    optim_kwargs:dict[str,str] = field(default_factory=lambda: {'fused':True, 'foreach':False})

class RadovidTrainer:
    def __init__(self, model:Radovid, training_args:TrainingArgs):
        self.model = model
        self.args = training_args
        self.optim = self._prepare_optimizer(**training_args.optim_kwargs)

        print(f'n trainable params: {(model.n_params/1e6):.2f} M')

    def train(self, dataloader:DataLoader):

        self._set_global_vars()

        if self.args.xavier_init:
            self._xavier_init()

        self._fuse_optim()

        self._set_prior_training_vars()

        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.args.n_epoch):

            for x, y in dataloader:

                torch.compiler.cudagraph_mark_step_begin()
                out = self.model.pred(x)
                loss = criterion(out.view(-1, self.model.args.vocab_size), y.view(-1))

                loss.backward()

    def benchmark(self):
        
        self._set_global_vars()

        x = torch.randint(0, self.model.args.vocab_size,
                          (4, self.model.args.context_window),
                          dtype=torch.long, device=self.model.args.device)
        
        y = torch.randint(0, self.model.args.vocab_size,
                          (4, self.model.args.context_window),
                          dtype=torch.long, device=self.model.args.device)

        if self.args.xavier_init:
            self._xavier_init()

        self._fuse_optim()

        self._set_prior_training_vars()

        criterion = nn.CrossEntropyLoss()

        for i in range(5): #warm up for benchmark
            torch.compiler.cudagraph_mark_step_begin()
            out = model.pred(x)
            loss = criterion(out.view(-1, self.model.args.vocab_size), x.view(-1))

            loss.backward()

        torch.cuda.synchronize()
        
        times = []

        for i in range(100):
            torch.compiler.cudagraph_mark_step_begin()
            out = model.pred(x)
            loss = criterion(out.view(-1, self.model.args.vocab_size), y.view(-1))
            loss_item = loss.item()

            loss.backward()
            
            times.append(time.time())
            print(f'average time: {np.mean(np.diff(times)):.4f} loss: {loss_item}', end='\r')

        print(f'average time for epoch: {np.mean(np.diff(times)):.4f}')

    def _xavier_init(self):
        for param in self.model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    @staticmethod
    def _set_global_vars():
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        torch._dynamo.config.capture_scalar_outputs = True
    
    @staticmethod
    def _set_prior_training_vars():
        torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
        torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None

    def _prepare_optimizer(self, **optim_kwargs):
        return partial(self.args.optim, **optim_kwargs, lr=self.args.lr)
    
    def _fuse_optim(self):
        optimizer_dict = {p: self.optim([p]) for p in self.model.parameters()}

        def optimizer_hook(parameter) -> None:
            # if parameter.grad is not None:
            #     parameter.grad.data.clamp_(-1, 1)
            optimizer_dict[parameter].step()
            optimizer_dict[parameter].zero_grad(set_to_none=True)

        for p in self.model.parameters():
            p.register_post_accumulate_grad_hook(optimizer_hook)

if __name__ == '__main__':
    import time
    import numpy as np
    from model import Args

    model = Radovid(Args())

    trainer = RadovidTrainer(model, TrainingArgs())

    trainer.benchmark()