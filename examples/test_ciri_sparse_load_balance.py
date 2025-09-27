import torch
from types import MethodType
from cirilla.Cirilla_model import Cirilla, Args, CirillaTokenizer, TrainingArgs, CirillaTrainer, JSONLDataset, load_balancing_loss

trainargs = TrainingArgs(n_epoch=1000, save_checkpoint_min=9999, use_muon_optim=True)
model = Cirilla(Args(moe_type='pytorch', n_layers=14, output_moe_weights=True, use_sparse=True))
trainer = CirillaTrainer(model, trainargs)

tokenizer = CirillaTokenizer(hub_url='AnthonyPa57/HF-torch-demo2')
dl = JSONLDataset(['./examples/data/example.jsonl', './examples/data/example.jsonl'], shuffle_path=True,
                    tokenizer=tokenizer, max_len=model.args.context_window)

def new_training_step(self, data): # define a custom training step

    out, moe_weight_list = self.model.pred(data[0]) # tokens, mask

    loss = self.criterion(out.view(-1, self.model.args.vocab_size), data[1].view(-1))

    lb_losses = [
    load_balancing_loss(w, num_experts=self.model.args.num_experts, top_k=self.model.args.k)
    for w in moe_weight_list
        ]
    lb_loss = torch.stack(lb_losses).mean()

    return loss + 0.1 * lb_loss

trainer.training_step = MethodType(new_training_step, trainer) # assign the custom training step to the trainer

trainer.train(dl, dl)

# trainer.benchmark()
