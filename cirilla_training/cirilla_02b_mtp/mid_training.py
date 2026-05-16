import torch
import math
import torch.nn.functional as F
from cirilla.Cirilla_model import CirillaMTP, MTPArgs, get_optims
from cirilla.Cirilla_model import CirillaTrainer, TrainingArgs, CirillaTokenizer, JSONDynamicDatset, DynamicCollator
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
                    torch_compile=False,
                    static_mask=False
                    )
                )
# model.pull_model_from_hub(hf_repo, force_eager=True, force_dynamic_mask=True)

tokenizer = CirillaTokenizer(hub_url=hf_repo)

pad_token_id = tokenizer.tokenizer.pad_token_id

new_lr = 1e-5

muon_opt, adam_opt = get_optims(
                                model,
                                use_muon_optim=True,
                                optim=torch.optim.AdamW,
                                lr=new_lr, weight_decay=1e-5,
                                )

micro_batch_size = 12

dl = JSONDynamicDatset(
                path=(
                        './training_datasets/mid_training/fandom_summaries.jsonl',
                        './training_datasets/mid_training/reason_gym_synth.jsonl'
                    ),
                shuffle_path=True,
                tokenizer=tokenizer,
                max_len=model.args.context_window,
                )

def mtp_training_step_grad_acc(self, data) -> float:
    step_loss = 0.0

    # torch.compiler.cudagraph_mark_step_begin()
    
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

            head_loss_weight = ((self.model.args.n_token_heads - i) / math.factorial(self.model.args.n_token_heads)) * self.model.args.n_token_heads

            preds = self.model.get_heads(i, zd)

            loss = head_loss_weight * \
                (F.cross_entropy(
                preds.view(-1, self.model.args.vocab_size),
                y_[:, i:-(self.model.args.n_token_heads - i)].reshape(-1),
                ignore_index=pad_token_id, label_smoothing=0.1)\
                    ) / (n_micro_steps * self.model.args.n_token_heads)
            
            step_loss += loss.item()
            loss.backward()
        
        z.backward(gradient=zd.grad)
    
    # for group in muon_opt.param_groups:
    #     for p in group['params']:
    #         if p.grad is not None:
    #             p.grad = p.grad.clone() # this cretes a separate cudagraph that doesnt touch the models one, but were we create a copy of params, so more memory
    
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

trainer = CirillaTrainer(model,
                            TrainingArgs(
                                        n_epoch=16,
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

# trainer._pull_all_from_hub(force_eager=True, force_dynamic_mask=True)

# trainer._pull_optim_from_hub()
# trainer.pulled_from_hub = True

# for optimizer in [adam_opt, muon_opt]:
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = new_lr

trainer.train(dl, collator=DynamicCollator(pad_token_id=pad_token_id))

final_loss = input('Enter final loss: ')
trainer._push_all_to_hub(float(final_loss), 'mid_training')
