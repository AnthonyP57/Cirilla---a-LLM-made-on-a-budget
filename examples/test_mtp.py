from cirilla.Cirilla_model import (
                            CirillaMTP,
                            MTPArgs,
                            JSONLDataset,
                            CirillaTrainer,
                            TrainingArgs,
                            CirillaTokenizer,
                            mtp_training_step,
                            mtp_inference_step
                            )
from types import MethodType
from functools import partial

model = CirillaMTP(MTPArgs(
                        n_layers=4,
                        dim=128,
                        d_ff=256,
                        n_heads=8,
                        context_window=128,
                        torch_compile=False,
                        layer_norm='Derf'))

targs = TrainingArgs(n_epoch=1000, save_checkpoint_min=9999, use_muon_optim=False)

trainer = CirillaTrainer(model, targs)

tokenizer = CirillaTokenizer(hub_url='AnthonyPa57/HF-torch-demo2')
pad_id = tokenizer.convert_tokens_to_ids('<pad>')

dl = JSONLDataset(['./examples/data/example.jsonl', './examples/data/example.jsonl'],
                    shuffle_path=True, tokenizer=tokenizer, max_len=model.args.context_window)

trainer.training_step = MethodType(partial(mtp_training_step, pad_id=pad_id), trainer)
trainer.inference_step = MethodType(partial(mtp_inference_step, pad_id=pad_id), trainer)
trainer.criterion = None # the criterion is useless in that case

trainer.train(dl, dl)
