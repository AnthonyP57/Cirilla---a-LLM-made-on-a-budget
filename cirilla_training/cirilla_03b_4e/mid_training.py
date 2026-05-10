import torch
import torch.nn.functional as F
from cirilla.Cirilla_model import Cirilla, Args, load_balancing_loss, get_optims
from cirilla.Cirilla_model import CirillaTrainer, TrainingArgs, CirillaTokenizer, JSONLDataset
from types import MethodType

hf_repo = 'AnthonyPa57/Cirilla-0.3B-4E'

model = Cirilla(Args())
model.pull_model_from_hub(hf_repo)

tokenizer = CirillaTokenizer(hub_url=hf_repo)

pad_token_id = tokenizer.tokenizer.pad_token_id

new_lr = 1e-5

muon_opt, adam_opt = get_optims(
                                model,
                                use_muon_optim=True,
                                optim=torch.optim.AdamW,
                                lr=new_lr, weight_decay=1e-5,
                                )

micro_batch_size = 8

dl = JSONLDataset(
                path=('./training_datasets/mid_training/fandom_summaries.jsonl', './training_datasets/mid_training/reason_gym_synth.jsonl'),
                shuffle_path=True,
                tokenizer=tokenizer,
                max_len=model.args.context_window,
                )

micro_batch_size = 8

def cirilla_grad_acc(self, data):

    x = data[0]
    y = data[1]

    train_loss = torch.zeros(1, device=x.device)

    torch.compiler.cudagraph_mark_step_begin()

    n_micro_steps = max(1, x.size(0) // micro_batch_size)

    for micro_step in range(n_micro_steps):

        # clear_router_zloss()
        # clear_load_balancing_loss()

        x_ = x[micro_step*micro_batch_size:(micro_step+1)*micro_batch_size]
        y_ = y[micro_step*micro_batch_size:(micro_step+1)*micro_batch_size]

        logits, moe_weight_list = model.pred(x_)
        # logits = model.pred(x)

        lb_losses = [
            load_balancing_loss(w, num_experts=model.args.num_experts, top_k=model.args.k)
            for w in moe_weight_list
                ]
        lb_loss = torch.stack(lb_losses).mean()

        loss = (F.cross_entropy(
            logits.view(-1, logits.size(-1)), y_.view(-1), ignore_index=pad_token_id, label_smoothing=0.1
        # ) + 0.1 * batched_router_zloss(model.smoes[0].args).mean()) / n_micro_steps
        ) + 0.01 * lb_loss) / n_micro_steps

        train_loss += loss.detach()

        loss.backward()

    muon_opt.step()
    adam_opt.step()

    muon_opt.zero_grad(set_to_none=True)
    adam_opt.zero_grad(set_to_none=True)

    return train_loss.item()

@torch.inference_mode()
def cirilla_grad_acc_inference(self, data):

    x = data[0]
    y = data[1]

    train_loss = torch.zeros(1, device=x.device)

    n_micro_steps = max(1, x.size(0) // micro_batch_size)

    for micro_step in range(n_micro_steps):

        # clear_router_zloss()
        # clear_load_balancing_loss()

        x_ = x[micro_step*micro_batch_size:(micro_step+1)*micro_batch_size]
        y_ = y[micro_step*micro_batch_size:(micro_step+1)*micro_batch_size]

        logits, moe_weight_list = model.pred(x_)
        # logits = model.pred(x)

        # lb_losses = [
        #     load_balancing_loss(w, num_experts=model.args.num_experts, top_k=model.args.k)
        #     for w in moe_weight_list
        #         ]
        # lb_loss = torch.stack(lb_losses).mean()

        loss = (F.cross_entropy(
            logits.view(-1, logits.size(-1)), y_.view(-1), ignore_index=pad_token_id
        # ) + 0.1 * batched_router_zloss(model.smoes[0].args).mean()) / n_micro_steps
        # ) + 0.01 * lb_loss) / n_micro_steps
        )) / n_micro_steps

        train_loss += loss.detach()

    return train_loss.item()

trainer = CirillaTrainer(model,
                            TrainingArgs(
                                        n_epoch=10,
                                        save_checkpoint_min=15,
                                        use_muon_optim=True,
                                        fuse_optim=False,
                                        batch_size=64,
                                        local_checkpoint_folder=f'./{hf_repo.split("/")[-1]}',
                                        hf_repo_id=hf_repo
                                        )
                                    )

trainer.training_step = MethodType(cirilla_grad_acc, trainer)
trainer.inference_step = MethodType(cirilla_grad_acc_inference, trainer)
trainer.criterion = None
trainer.optims_to_save = {'muon_opt': muon_opt, 'adam_opt': adam_opt}

# trainer._pull_optim_from_hub()
# trainer.pulled_from_hub = True

# for optimizer in [adam_opt, muon_opt]:
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = new_lr

trainer.train(dl)

final_loss = input('Enter final loss: ')
trainer._push_all_to_hub(float(final_loss), 'mid_training')
