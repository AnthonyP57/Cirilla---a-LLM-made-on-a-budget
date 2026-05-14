import torch
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True
torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit=None
torch.set_float32_matmul_precision('medium')

import torch.nn.functional as F
from cirilla.Cirilla_model import CirillaMTP, MTPArgs, get_optims
from cirilla.Cirilla_model import CirillaTrainer, TrainingArgs, CirillaTokenizer, JSONLDataset
from types import MethodType


hf_repo = 'AnthonyPa57/CirillaMTP-0.1B-3E'

model = CirillaMTP(MTPArgs(
                    dim=512,
                    d_ff=1024,
                    out_bias=True,
                    tie_params=False,
                    n_heads=4,
                    n_kv_heads=4,
                    n_layers=8,
                    num_experts=3,
                    k=2,
                    vocab_size=30_000,
                    n_token_heads=3,
                    window_size=256,
                    )
                )
tokenizer = CirillaTokenizer(hub_url=hf_repo)

# dl = JSONLDataset(
#     [
#     './training_datasets/pretraining/pretraining.jsonl',
#     './training_datasets/mid_training/reason_gym_synth.jsonl',
#     './training_datasets/mid_training/fandom_summaries.jsonl',
#     './training_datasets/domain_training/witcher_synthetic_instruct.jsonl',
#     './training_datasets/domain_training/witcher_instruct.jsonl',
#     './training_datasets/domain_training/synth_multi_round.jsonl',
#     './training_datasets/domain_training/fandom_summaries_instruct.jsonl',
#     ], 
#     shuffle_path=False
#     )

# tokenizer.train(dl, min_frequency=5, vocab_size=30_000)
# tokenizer.push_to_hub(hf_repo)

pad_token_id = tokenizer.tokenizer.pad_token_id

muon_opt, adam_opt = get_optims(
                                model,
                                use_muon_optim=True,
                                optim=torch.optim.AdamW,
                                lr=5e-4, weight_decay=1e-5,
                                )

micro_batch_size = 12

def mtp_training_step_grad_acc(self, data) -> float:
    step_loss = 0.0

    torch.compiler.cudagraph_mark_step_begin()
    
    x = data[0]
    y = data[1]
    
    y_fill = torch.tensor([[pad_token_id] * self.model.args.n_token_heads] * y.shape[0], dtype=y.dtype, device=y.device)
    y = torch.hstack([y, y_fill])

    n_micro_steps = max(1, x.size(0) // micro_batch_size)

    for micro_step in range(n_micro_steps):

        x_ = x[micro_step*micro_batch_size:(micro_step+1)*micro_batch_size]
        y_ = y[micro_step*micro_batch_size:(micro_step+1)*micro_batch_size]

        z = self.model.get_z(x_)

        zd = z.detach()
        zd.requires_grad = True

        for i in range(self.model.args.n_token_heads):
            preds = self.model.get_heads(i, zd)

            loss = (F.cross_entropy(
                preds.view(-1, self.model.args.vocab_size),
                y_[:, i:-(self.model.args.n_token_heads - i)].reshape(-1),
                ignore_index=pad_token_id, label_smoothing=0.1)\
                    ) / (n_micro_steps * self.model.args.n_token_heads)
            
            step_loss += loss.item()
            loss.backward()
        
        z.backward(gradient=zd.grad)
    
    for group in muon_opt.param_groups:
        for p in group['params']:
            if p.grad is not None:
                p.grad = p.grad.clone() # this cretes a separate cudagraph that doesnt touch the models one, but were we create a copy of params, so more memory
    
    muon_opt.step()
    adam_opt.step()

    muon_opt.zero_grad(set_to_none=True)
    adam_opt.zero_grad(set_to_none=True)

    return step_loss

@torch.inference_mode()
def mtp_inference_step_grad_acc(self, data) -> float:
    step_loss = 0.0

    x = data[0]
    y = data[1]
    
    n_micro_steps = max(1, x.size(0) // micro_batch_size)

    for micro_step in range(n_micro_steps):

        x_ = x[micro_step*micro_batch_size:(micro_step+1)*micro_batch_size]
        y_ = y[micro_step*micro_batch_size:(micro_step+1)*micro_batch_size]

        z = self.model.get_z(x_)

        preds = self.model.get_heads(0, z)
        loss = F.cross_entropy(
            preds.view(-1, self.model.args.vocab_size),
            y_.reshape(-1),
            ignore_index=pad_token_id) / n_micro_steps
            
        step_loss += loss.item()

    return step_loss

dl = JSONLDataset(
                './training_datasets/pretraining/pretraining.jsonl',
                # './examples/data/example.jsonl',
                shuffle_path=True,
                tokenizer=tokenizer,
                max_len=model.args.context_window,
                )

trainer = CirillaTrainer(model,
                            TrainingArgs(
                                        n_epoch=10,
                                        save_checkpoint_min=15,
                                        use_muon_optim=True,
                                        fuse_optim=False,
                                        batch_size=120,
                                        local_checkpoint_folder=f'./{hf_repo.split("/")[-1]}',
                                        hf_repo_id=hf_repo
                                        )
                                    )

trainer.training_step = MethodType(mtp_training_step_grad_acc, trainer)
trainer.inference_step = MethodType(mtp_inference_step_grad_acc, trainer)
trainer.criterion = None
trainer.optims_to_save = {'muon_opt': muon_opt, 'adam_opt': adam_opt}

# trainer._pull_all_from_hub()

trainer.train(dl)

final_loss = input('Enter final loss: ')
trainer._push_all_to_hub(float(final_loss), 'pretraining')
