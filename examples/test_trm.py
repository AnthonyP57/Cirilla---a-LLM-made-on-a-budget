from cirilla.Cirilla_model import (
                            CirillaTRM,
                            TRMArgs,
                            TrainingArgs,
                            CirillaTrainer,
                            CirillaTokenizer,
                            JSONLDataset,
                            # Encoder,
                            # EncoderArgs,
                            MixerArgs,
                            MLPMixer1D
                            )
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from ema_pytorch import EMA

# encoder = Encoder(EncoderArgs())
mixer = MLPMixer1D(MixerArgs())

model = CirillaTRM(mixer, TRMArgs())

targs = TrainingArgs(n_epoch=1000, save_checkpoint_min=9999, use_muon_optim=False)

trainer = CirillaTrainer(model, targs)
trainer._set_global_vars()
trainer._weights_init()
trainer._fuse_optim()

tokenizer = CirillaTokenizer(hub_url='AnthonyPa57/HF-torch-demo2')

dl = JSONLDataset(['./examples/data/example_bert.jsonl', './examples/data/example_bert.jsonl'],
                    shuffle_path=True, tokenizer=tokenizer, max_len=mixer.args.context_window)

ds = DataLoader(dl, batch_size=4)

max_recurrent_step = 16
halt_weight = 0.5
halt_thresh = 0.5

ema_model = EMA(
                model,
                beta=0.999,
                update_model_with_ema_every=1_000,
                forward_method_names=('predict',)
                )

for _ in range(100):
    epoch_loss = 0
    n = 0

    for data in ds:
        
        y_hat, z = model.get_init()
        x = data[0]
        mask = data[1]

        # preds, n_steps = model.predict(x, mask)
        # print(preds.argmax(-1), n_steps)

        for step in range(max_recurrent_step):

            torch.compiler.cudagraph_mark_step_begin()

            pred, y_hat, z, haltp = model(x, y_hat, z, mask)

            loss = F.cross_entropy(pred.view(-1, model.args.vocab_size), x.view(-1))

            all_correct = (pred.argmax(dim=-1) == x).all(dim=-1)

            halt_loss = F.binary_cross_entropy_with_logits(haltp, all_correct.to(haltp.dtype))

            loss = loss + halt_weight * halt_loss

            epoch_loss += loss.item()
            n += 1

            loss.backward()
            ema_model.update() # model needs to have .predict() method

            halt_mask = F.sigmoid(haltp) < halt_thresh
            print(f'loss: {epoch_loss / n:.2f} epoch: {_} n_steps: {step}')

            if halt_mask.all():
                continue
            
            y_hat = y_hat[halt_mask]
            z = z[halt_mask]
            x = x[halt_mask]
            mask = mask[halt_mask]

            if z.numel() == 0: # if is empty
                break

            y_hat, z = y_hat.detach(), z.detach()
