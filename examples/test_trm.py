from cirilla.Cirilla_model import (
                            CirillaTRM,
                            TRMArgs,
                            TrainingArgs,
                            CirillaTrainer,
                            CirillaTokenizer,
                            JSONLDataset,
                            Encoder,
                            EncoderArgs
                            )
import torch.nn as nn
import torch.nn.functional as F
import torch
# from ema_pytorch import EMA

encoder = Encoder(EncoderArgs())

model = CirillaTRM(encoder, TRMArgs())

targs = TrainingArgs(n_epoch=1000, save_checkpoint_min=9999, use_muon_optim=False, lr=1e-7)

trainer = CirillaTrainer(model, targs)

tokenizer = CirillaTokenizer(hub_url='AnthonyPa57/HF-torch-demo2')

dl = JSONLDataset(['./examples/data/example_bert.jsonl', './examples/data/example_bert.jsonl'],
                    shuffle_path=True, tokenizer=tokenizer, max_len=encoder.args.context_window)

from types import MethodType

max_recurrent_step = 4
halt_weight = 0.2
halt_thresh = 0.5

# ema_model = EMA(
#                 model,
#                 beta=0.999,
#                 update_model_with_ema_every=10_000,
#                 forward_method_names=('predict',)
#                 )

def new_training_step(self, data): # define a custom training step
    # out = self.model.pred(data[0], data[1]) # tokens, mask
    # loss = self.criterion(out, data[2])
    # return loss
    y_hat, z = self.model.get_init()

    for step in range(max_recurrent_step):

        if not step == 0:
            torch.compiler.cudagraph_mark_step_begin()

        pred, y_hat, z, haltp = self.model(data[0], y_hat, z, data[1])

        loss = self.criterion(pred.view(-1, self.model.args.vocab_size), data[0].view(-1))

        all_correct = (pred.argmax(dim=-1) == data[0]).all(dim=-1)

        # print(max(haltp), min(haltp), haltp.shape, all_correct.shape)
        halt_loss = F.binary_cross_entropy_with_logits(haltp, all_correct.to(haltp.dtype))
        print(loss.item(), halt_loss.item())

        loss = loss + halt_weight * halt_loss

        # ema_model.update()

        # halt_mask = haltp < halt_thresh
        # print(halt_mask, haltp)

        # if (~halt_mask).any():
        #     print(halt_mask)
        #     continue
        
        # y_hat = y_hat[halt_mask]
        # z = z[halt_mask]
        # data[0] = data[0][halt_mask]
        # data[1] = data[1][halt_mask]

        # print(z.shape, loss.item(), halt_loss.item())

        if z.numel() == 0: # if is empty
            return loss
        
        if step == max_recurrent_step - 1:
            return loss
        else:
            loss.backward()

        # y_hat, z = y_hat.detach(), z.detach()

trainer.training_step = MethodType(new_training_step, trainer) # assign the custom training step to the trainer
trainer.criterion = nn.CrossEntropyLoss() # override the default criterion

trainer.train(dl, dl)