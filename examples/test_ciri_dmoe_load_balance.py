from types import MethodType
from cirilla.Cirilla_model import Cirilla, Args, CirillaTokenizer, TrainingArgs, CirillaTrainer, JSONLDataset
from megablocks.layers.moe import clear_load_balancing_loss
from megablocks.layers.router import clear_router_zloss, batched_router_zloss
import torch

trainargs = TrainingArgs(n_epoch=1000, save_checkpoint_min=9999, use_muon_optim=True)
model = Cirilla(Args(moe_type='megablocks-dmoe', n_layers=14, output_moe_weights=False, layer_norm="DyT"))
trainer = CirillaTrainer(model, trainargs)

tokenizer = CirillaTokenizer(hub_url='AnthonyPa57/HF-torch-demo2')
dl = JSONLDataset(['./examples/data/example.jsonl', './examples/data/example.jsonl'], shuffle_path=True,
                    tokenizer=tokenizer, max_len=model.args.context_window)

def new_training_step(self, data): # define a custom training step
    torch.compiler.cudagraph_mark_step_begin()

    # for megablocks implementation clearing the losses is crucial as they will accumulate and cause a out of memory (OOM) error
    clear_router_zloss()
    clear_load_balancing_loss()

    out = self.model.pred(data[0])

    loss = self.criterion(out.view(-1, self.model.args.vocab_size), data[1].view(-1))
    loss = loss + 0.1 * batched_router_zloss(self.model.decoder.smoes[0].args).mean()
    loss_item = loss.item()
    loss.backward()
    return loss_item

trainer.training_step = MethodType(new_training_step, trainer) # assign the custom training step to the trainer

# trainer.benchmark()

trainer.train(dl, dl)
