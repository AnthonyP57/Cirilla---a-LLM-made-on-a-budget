import torch
import torch.nn.functional as F
from cirilla.Cirilla_model import Cirilla, Args, load_balancing_loss
from cirilla.Cirilla_model import CirillaTrainer, TrainingArgs, CirillaTokenizer, JSONLDataset
from cirilla.LLM_pieces import get_activation
from types import MethodType

model = Cirilla(Args(output_moe_weights=True, tie_params=True, torch_compile=False, n_layers=6))
tokenizer = CirillaTokenizer(hub_url='AnthonyPa57/HF-torch-demo2')

pad_token_id = tokenizer.tokenizer.pad_token_id

moun_optim = get_activation("motif-technologies/optimizer")

def get_optims(model, use_muon_optim, optim, lr, weight_decay):

    if use_muon_optim:
        get_default_muon_param_groups = moun_optim.muon.get_default_muon_param_groups
        muon_param_groups = get_default_muon_param_groups(model)

        moptim = moun_optim.Muon(muon_param_groups, lr=lr, weight_decay=weight_decay)

        rest_of_params = [p for n, p in model.named_parameters() if n not in muon_param_groups[0]['names']]

        roptim = optim(rest_of_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))

        return moptim, roptim
    
    else:
        return optim(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))

muon_opt, adam_opt = get_optims(model, use_muon_optim=True, optim=torch.optim.AdamW, lr=1e-3, weight_decay=1e-4)

micro_batch_size = 2

def cirilla_grad_acc(self, data):

    x = data[0]
    y = data[1]

    train_loss = torch.zeros(1, device=x.device)

    torch.compiler.cudagraph_mark_step_begin()

    n_micro_steps = x.size(0) // micro_batch_size

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
            logits.view(-1, logits.size(-1)), y_.view(-1), ignore_index=pad_token_id
        # ) + 0.1 * batched_router_zloss(model.smoes[0].args).mean()) / n_micro_steps
        ) + 0.01 * lb_loss) / n_micro_steps

        train_loss += loss.detach()

        loss.backward()

    muon_opt.step()
    muon_opt.zero_grad(set_to_none=True)

    adam_opt.step()
    adam_opt.zero_grad(set_to_none=True)

    return train_loss.item()

@torch.inference_mode()
def cirilla_grad_acc_inference(self, data):
    x = data[0]
    y = data[1]

    train_loss = torch.zeros(1, device=x.device)

    n_micro_steps = x.size(0) // micro_batch_size

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
            logits.view(-1, logits.size(-1)), y_.view(-1), ignore_index=pad_token_id
        # ) + 0.1 * batched_router_zloss(model.smoes[0].args).mean()) / n_micro_steps
        ) + 0.01 * lb_loss) / n_micro_steps

        train_loss += loss.detach()

    return train_loss.item()

dl = JSONLDataset(['./examples/data/example.jsonl', './examples/data/example.jsonl'], shuffle_path=True,
                    tokenizer=tokenizer, max_len=model.args.context_window,
                    random_spelling_mistake_prob=0.1,
                    random_missing_char_prob=0.05)

dl_valid = JSONLDataset(['./examples/data/example.jsonl', './examples/data/example.jsonl'], shuffle_path=True,
                    tokenizer=tokenizer, max_len=model.args.context_window)

trainer = CirillaTrainer(model, TrainingArgs(n_epoch=1000, save_checkpoint_min=9999, use_muon_optim=True, fuse_optim=False))

trainer.training_step = MethodType(cirilla_grad_acc, trainer)
trainer.inference_step = MethodType(cirilla_grad_acc_inference, trainer)

trainer.train(dl, dl_valid)