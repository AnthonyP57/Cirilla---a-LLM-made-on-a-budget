import torch
import torch.nn as nn
from model import Cirilla
from torch.utils.data import DataLoader
from dataloader import JSONLDataset
from functools import partial
from dataclasses import dataclass, field
from torch.optim import Optimizer, AdamW, SGD
from pathlib import Path
from hf_hub import push_model_to_hub
from huggingface_hub import hf_hub_download
import os
from safetensors.torch import load_file
from modules import get_args_from_hub
import time
from modules import cache_or_fetch
import threading
from progress_table import ProgressTable

@dataclass
class TrainingArgs:
    n_epoch:int = 100
    optim:Optimizer = AdamW
    lr:float = 5e-5
    batch_size:int = 4
    xavier_init:bool = True
    local_checkpoint_folder:Path = './'
    optim_kwargs:dict[str,str] = field(default_factory=lambda: {'fused':True, 'foreach':False})

    renew_training:bool = True
    save_checkpoint_n_iterations:int = None
    save_checkpoint_min:int = 0.5

    push_checkpoint_to_hub:bool = False
    push_checkpoint_to_hub_n_local_saves:int = 4
    
    hf_repo_id:str = None
    private_hf_repo:bool=True
    hf_tags:list[str] = field(default_factory=lambda: ["pytorch", "text-generation", "moe", "custom_code"])
    hf_license:str = 'mit'
    languages:list[str] = field(default_factory=lambda: ["en"])
    model_card:str = None

    @property
    def stateful_optim(self):
        if self.optim == SGD:
            return False
        return True

class CirillaTrainer:
    def __init__(self, model:Cirilla, training_args:TrainingArgs):
        self.model = model
        self.args = training_args
        self.optim = self._prepare_optimizer(**training_args.optim_kwargs)
        self.criterion = nn.CrossEntropyLoss(ignore_index=1, # tokenizer.convert_tokens_to_ids(padding token) ; by default it's 1
                                             label_smoothing=0.1)

        self.n_checkpoints = 0

        print(f'n trainable params: {(model.n_params/1e6):.2f} M')

    def train(self, dataset:JSONLDataset, valid_dataset:JSONLDataset=None):

        assert cache_or_fetch('DATA_LEN', dataset.path_signature) % self.args.batch_size == 0, f"Dataset length: {cache_or_fetch('DATA_LEN', dataset.path_signature)} is not divisible by batch size: {self.args.batch_size}. It has to be for optimal training."
        
        dataset_path = dataset.path_signature

        dataloader = DataLoader(dataset, shuffle=False, batch_size=self.args.batch_size)
        del dataset

        if valid_dataset is not None:
            valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=self.args.batch_size)
            del valid_dataset

        start_time = time.time()

        self._set_global_vars()

        skip_n_data_points = cache_or_fetch('N_DATA_POINTS', dataset_path)
        if skip_n_data_points is None:
            skip_n_data_points = 0

        n_iter = -1

        os.makedirs(self.args.local_checkpoint_folder, exist_ok=True)

        if self.args.renew_training:
            if os.path.exists(os.path.join(self.args.local_checkpoint_folder, 'optimizer_states.pt')) or not self.args.stateful_optim:
                if os.path.exists(os.path.join(self.args.local_checkpoint_folder, 'model.pt')):
                    self._load_local_checkpoint()
                else:
                    if skip_n_data_points > 0:
                        raise FileNotFoundError(f"Couldn't find model path at: {os.path.join(self.args.local_checkpoint_folder, 'model.pt')}")
            else:
                if skip_n_data_points > 0:
                    raise FileNotFoundError(f"Couldn't find optimizer states path for a {'stateful' if self.args.stateful_optim else 'non-stateful'} optimizer at: {os.path.join(self.args.local_checkpoint_folder, 'optimizer_states.pt')}")

        if not hasattr(self, 'optimizer_by_name'): # if didn't load from checkpoint
            print("Training from scratch")

            if self.args.xavier_init:
                self._xavier_init()

            self._fuse_optim()

        self._set_prior_training_vars()

        if self.criterion is None:
            self.criterion = nn.CrossEntropyLoss()

        times = [time.time()]

        for epoch in range(1, self.args.n_epoch + 1):

            self.model.train()

            for data in dataloader:

                n_iter += 1

                if n_iter * self.args.batch_size < skip_n_data_points:
                    continue

                torch.compiler.cudagraph_mark_step_begin()

                loss = self.training_step(data)
                loss_item = loss.item()
                loss.backward()

                times.append(time.time())

                print(f"iter: {n_iter}, loss: {loss_item:.4f}, time: {times[-1]-times[-2]:.2f}")
                do_checkpoint, push_hub = self._check_if_do_checkpoint(time.time() - start_time, n_iter)
                
                if do_checkpoint:
                    start_time = time.time()
                    try:
                        self._save_local_checkpoint_async()
                    except Exception as e:
                        print(f"Failed to save local checkpoint asynchronously:{e}\nSaving synchronously")
                        self._save_local_checkpoint()

                    cache_or_fetch('N_DATA_POINTS', dataset_path, n_iter * self.args.batch_size)
                    if push_hub and self.args.push_checkpoint_to_hub:
                        try:
                            self._push_all_to_hub_async(loss, dataset_path.split('/')[-1].split('.')[0])
                            
                        except Exception as e:
                            print(f"Failed to push asynchronously to HF hub: {e}\nPushing synchronously")
                            self._push_all_to_hub(loss, dataset_path.split('/')[-1].split('.')[0])

                if valid_dataloader is not None:
                    self.model.eval()

                    total_loss = 0
                    with torch.no_grad():
                        for data in valid_dataloader:
                            loss = self.training_step(data)
                            loss_item = loss.item()

                            total_loss += loss_item

                            print(f"iter: {n_iter}, loss: {loss_item:.4f}")

                    print(f'{iter} valid loss: {total_loss / len(valid_dataloader)}')

    def training_step(self, data):
        out = self.model.pred(data[0])
        loss = self.criterion(out.view(-1, self.model.args.vocab_size), data[1].view(-1))
        return loss

    def benchmark(self):
        
        self._set_global_vars()

        x = torch.randint(0, self.model.args.vocab_size,
                          (4, self.model.args.context_window),
                          dtype=torch.long, device=self.model.args.device)
        
        y = torch.randint(0, self.model.args.vocab_size,
                          (4, self.model.args.context_window),
                          dtype=torch.long, device=self.model.args.device)
        
        def loss_color(distance):
            if distance < 8:
                return "green"
            elif distance < 9:
                return "yellow"
            else:
                return "red"
            
        def time_color(distance):
            if distance < 0.76:
                return "green"
            elif distance < 0.9:
                return "yellow"
            else:
                return "red"

        if self.args.xavier_init:
            self._xavier_init()

        self._fuse_optim()

        self._set_prior_training_vars()

        for i in range(5): #warm up for benchmark
            torch.compiler.cudagraph_mark_step_begin()
            loss = self.training_step((x, x))

            loss.backward()

        torch.cuda.synchronize()

        ptable = ProgressTable(
                pbar_show_progress=False,
                pbar_show_throughput=False,
                pbar_show_eta=True,
                default_column_width=8,
                default_header_color="bold",
                                )
        
        main_pbar = ptable.pbar(
                        100,
                        position=1,
                        show_progress=True,
                        style="rich alt lightmagenta_ex lightwhite_ex",
                    )
        
        times = [time.time()]
        losses = []

        for i in range(100):

            if i % 5 == 0:
                ptable['epoch'] = i

            torch.compiler.cudagraph_mark_step_begin()
            loss = self.training_step((x, y))
            loss_item = loss.item()

            loss.backward()
            
            times.append(time.time())
            losses.append(loss_item)
            
            ptable.update('train loss', loss_item, aggregate='mean', color='cyan')
            ptable.update('time', round(times[-1] - times[-2], 4), aggregate='mean', color='blue')

            if i % 5 == 0: # new row every 5 iterations
                ptable.next_row(split=True, color={'time': time_color(np.mean(np.diff(times[-5:]))), 'train loss': loss_color(np.mean(losses[-5:]))})

            main_pbar.update(1)

        ptable.close()
        print(f'average time for epoch: {np.mean(np.diff(times)):.4f}')

    def _check_if_do_checkpoint(self, time, iter_step):
        if self.args.save_checkpoint_min is not None:
            if time >= self.args.save_checkpoint_min * 60:
                self.n_checkpoints += 1
                return True, self.n_checkpoints % self.args.push_checkpoint_to_hub_n_local_saves == 0
            
        if self.args.save_checkpoint_n_iterations is not None:
            if iter_step % self.args.save_checkpoint_n_iterations == 0:
                self.n_checkpoints += 1
                return True, self.n_checkpoints % self.args.push_checkpoint_to_hub_n_local_saves == 0
            
        return False, False

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
                print(f"Unknown param of shape: {p.shape}")

    def _save_local_checkpoint(self):
        if not hasattr(self, 'optimizer_by_name'):
            self._fuse_optim()
            
        torch.save(self.model.state_dict(), os.path.join(self.args.local_checkpoint_folder, 'model.pt'))

        if self.args.stateful_optim:

            optim_states = {name: opt.state_dict() for name, opt in self.optimizer_by_name.items()}
            torch.save(optim_states, os.path.join(self.args.local_checkpoint_folder, 'optimizer_states.pt'))

    def _save_local_checkpoint_async(self):
        def worker():
            self._save_local_checkpoint()

        threading.Thread(target=worker, daemon=True).start()

    def _load_local_checkpoint(self):
        self.model.load_state_dict(torch.load(\
            os.path.join(self.args.local_checkpoint_folder,'model.pt')))
        
        if self.args.stateful_optim:

            loaded_states = torch.load(\
                os.path.join(self.args.local_checkpoint_folder, 'optimizer_states.pt'),
                map_location=self.model.args.device)

            self._load_optim_from_checkpoint(loaded_states)

        else:
            self._fuse_optim()
        
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

    def _push_all_to_hub_async(self, loss, dataset_name):
        args = (loss, dataset_name)

        def worker(loss_value, dataset_name):
            self._push_all_to_hub(loss_value, dataset_name)

        t = threading.Thread(target=worker, args=args, daemon=True)
        t.start()
    
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
        pulled_args = get_args_from_hub(self.args.hf_repo_id)

        if model_args != pulled_args:
            print(f"Current model args don't correspond to the HF model's args.\nCurrent args:\n{model_args}\nThe model will use the HF args:\n{pulled_args}")
            self.model = Cirilla(pulled_args)

        file_path = hf_hub_download(
            repo_id=self.args.hf_repo_id,
            filename="model.safetensors",
        )

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
    from tokenizer_modules import CirillaTokenizer

    model = Cirilla(Args())

    targs = TrainingArgs(hf_repo_id='AnthonyPa57/HF-torch-demo-R', local_checkpoint_folder='./test_model')
    trainer = CirillaTrainer(model, targs)

    tokenizer = CirillaTokenizer(hub_url='AnthonyPa57/HF-torch-demo2')
    dl = JSONLDataset(['./example.jsonl', './example.jsonl'], shuffle_path=True, tokenizer=tokenizer, max_len=model.args.context_window)

    trainer.train(dl, dl)

    # trainer._fuse_optim()
    # trainer._save_local_checkpoint()
    # trainer._push_all_to_hub_async(0, 'test')

    # trainer._load_local_checkpoint()
    # trainer._pull_all_from_hub()
    # trainer._pull_model_from_hub()

    # trainer.benchmark()

    # time.sleep(60)