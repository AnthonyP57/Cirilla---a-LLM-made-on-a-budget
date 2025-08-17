import torch
import torch.nn as nn
from model import Radovid
from torch.utils.data import DataLoader
from functools import partial
from dataclasses import dataclass, field
from torch.optim import Optimizer, AdamW
from pathlib import Path
from hf_hub import push_model_to_hub
from huggingface_hub import hf_hub_download
import json
import os
from safetensors.torch import load_file

@dataclass
class TrainingArgs:
    n_epoch:int = 100
    optim:Optimizer = AdamW
    lr:float = 5e-5
    xavier_init:bool = True
    local_checkpoint_folder:Path = './'
    optim_kwargs:dict[str,str] = field(default_factory=lambda: {'fused':True, 'foreach':False})
    
    hf_repo_id:str = None
    private_hf_repo:bool=True
    hf_tags:list[str] = field(default_factory=lambda: ["pytorch", "text-generation", "moe", "custom_code"])
    hf_license:str = 'mit'
    languages:list[str] = field(default_factory=lambda: ["en"])
    model_card:str = None

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
        self.optimizer_by_name = {}
        for name, p in self.model.named_parameters():
            self.optimizer_by_name[name] = self.optim([p])

        params_by_name = dict(self.model.named_parameters())
        optimizer_dict = {params_by_name[name]: opt for name, opt in self.optimizer_by_name.items()}

        self._register_hooks(optimizer_dict)

    def _register_hooks(self, optimizer_dict):
        
        def optimizer_hook(parameter):
            optimizer_dict[parameter].step()
            optimizer_dict[parameter].zero_grad(set_to_none=True)

        for p in self.model.parameters():
            if p in optimizer_dict:
                p.register_post_accumulate_grad_hook(optimizer_hook)
            else:
                print(f"Unknown param: {p.shape}")

    def _save_local_checkpoint(self):
        if not hasattr(self, 'optimizer_by_name'):
            self._fuse_optim()
            
        torch.save(self.model.state_dict(), os.path.join(self.args.local_checkpoint_folder, 'model.pt'))
        optim_states = {name: opt.state_dict() for name, opt in self.optimizer_by_name.items()}
        torch.save(optim_states, os.path.join(self.args.local_checkpoint_folder, 'optimizer_states.pt'))

    def _load_local_checkpoint(self):
        self.model.load_state_dict(torch.load(\
            os.path.join(self.args.local_checkpoint_folder,'model.pt')))

        loaded_states = torch.load(\
            os.path.join(self.args.local_checkpoint_folder, 'optimizer_states.pt'),
            map_location=self.model.args.device)
        
        self._load_optim_from_checkpoint(loaded_states)
        
    def _load_optim_from_checkpoint(self, loaded_states):

        params_by_name = dict(self.model.named_parameters())
        self.optimizer_by_name = {}

        for name, state in loaded_states.items():

            if name not in params_by_name:
                print(f"Skipping unknown param: {name}")
                continue

            p = params_by_name[name]
            opt = self.optim([p])
            opt.load_state_dict(state)
            self.optimizer_by_name[name] = opt

        optimizer_dict = {params_by_name[n]: o for n, o in self.optimizer_by_name.items()}

        self._register_hooks(optimizer_dict)

    def _push_all_to_hub(self, loss, dataset_name):
        push_model_to_hub(
            repo_id = self.args.hf_repo_id,
            model = self.model,
            loss = loss,
            dataset_name = dataset_name,
            private = self.args.private_hf_repo,
            optmizer_states_path = os.path.join(self.args.local_checkpoint_folder, 'optimizer_states.pt'),
            tags = self.args.hf_tags,
            license = self.args.hf_license,
            languages = self.args.languages,
            model_card = self.args.model_card
        )

    def _get_args_from_hub(self):
        file_path = hf_hub_download(
            repo_id=self.args.hf_repo_id,
            filename="config.json",
        )
        with open(file_path, "r") as f:
            config = json.load(f)
        args = Args(**config[list(config.keys())[0]])

        return args
    
    def _pull_optim_from_hub(self):
        file_path = hf_hub_download(
            repo_id=self.args.hf_repo_id,
            filename="optimizer_states.pt",
        )

        if not os.path.exists(file_path):
            print('no optimizer states file found')

        with open(file_path, "rb") as f:
            loaded_states = torch.load(f, map_location=self.model.args.device)

        self._load_optim_from_checkpoint(loaded_states)

    def _pull_model_from_hub(self):
        model_args = self.model.args
        pulled_args = self._get_args_from_hub()

        if model_args != pulled_args:
            self.model = Radovid(pulled_args)
            print(f"current model args don't correspond to the HF model's args.\nRight now the model uses the HF args:\n{pulled_args}")

        file_path = hf_hub_download(
            repo_id=self.args.hf_repo_id,
            filename="model.safetensors",
        )

        if not os.path.exists(file_path):
            print('no model file found')

        loaded = load_file(file_path)
        if "output.weight" not in loaded:
            loaded['output.weight'] = loaded["emb.embeddings.weight"]

        self.model.load_state_dict(loaded)

    def _pull_all_from_hub(self):
        self._pull_model_from_hub()
        self._pull_optim_from_hub()

if __name__ == '__main__':
    import time
    import numpy as np
    from model import Args

    model = Radovid(Args())

    targs = TrainingArgs(hf_repo_id='AnthonyPa57/HF-torch-demo-R', local_checkpoint_folder='./test_model')
    trainer = RadovidTrainer(model, targs)


    # trainer._load_local_checkpoint()
    trainer._pull_all_from_hub()
    # trainer._pull_model_from_hub()

    # trainer._fuse_optim()
    # trainer._save_local_checkpoint()
    # trainer._push_all_to_hub(0, 'test')

    trainer.benchmark()