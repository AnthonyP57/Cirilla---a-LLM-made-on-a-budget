from cirilla.Cirilla_model import CirillaMTP, MTPArgs, JSONLDataset, CirillaTrainer, TrainingArgs, CirillaTokenizer
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

model = CirillaMTP(MTPArgs(
                        n_layers=4,
                        dim=128,
                        d_ff=256,
                        n_heads=8,
                        context_window=128,
                        torch_compile=False))

targs = TrainingArgs(n_epoch=1000, save_checkpoint_min=9999, use_muon_optim=False)

trainer = CirillaTrainer(model, targs)
trainer._set_global_vars()
trainer._weights_init()
trainer._fuse_optim()

tokenizer = CirillaTokenizer(hub_url='AnthonyPa57/HF-torch-demo2')
pad_id = tokenizer.convert_tokens_to_ids('<pad>')

dl = JSONLDataset(['./examples/data/example.jsonl', './examples/data/example.jsonl'],
                    shuffle_path=True, tokenizer=tokenizer, max_len=model.args.context_window)

ds = DataLoader(dl, batch_size=4)

for e in range(100):

    epoch_loss = 0
    n = 0

    for data in ds:

        torch.compiler.cudagraph_mark_step_begin()
        
        x = data[0]
        y = data[1]
        
        y_fill = torch.tensor([[pad_id] * model.args.n_token_heads] * y.shape[0], dtype=y.dtype, device=y.device)
        y = torch.hstack([y, y_fill])

        z = model.get_z(x)
        zd = z.detach()
        zd.requires_grad = True

        for i in range(model.args.n_token_heads):
            preds = model.get_heads(i, zd)
            loss = F.cross_entropy(
                preds.view(-1, model.args.vocab_size),
                y[:, i:-(model.args.n_token_heads - i)].reshape(-1),
                ignore_index=pad_id)
            epoch_loss += loss.item()
            n += y.shape[0]
            loss.backward()

        z.backward(gradient=zd.grad)

    print(f'Loss: {epoch_loss / n:.2f} Epoch: {e}')