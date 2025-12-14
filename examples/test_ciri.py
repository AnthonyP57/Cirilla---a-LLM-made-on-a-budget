from cirilla.Cirilla_model import (
                            Cirilla,
                            Args,
                            CirillaTokenizer,
                            TrainingArgs,
                            CirillaTrainer,
                            JSONLDataset
                            )

trainargs = TrainingArgs(n_epoch=1000, save_checkpoint_min=9999, use_muon_optim=True)
model = Cirilla(Args(moe_type='pytorch', n_layers=7, context_window=512, output_moe_weights=False, layer_norm="Derf", torch_compile=False))
trainer = CirillaTrainer(model, trainargs)

tokenizer = CirillaTokenizer(hub_url='AnthonyPa57/HF-torch-demo2')
dl = JSONLDataset(['./examples/data/example.jsonl', './examples/data/example.jsonl'], shuffle_path=True,
                    tokenizer=tokenizer, max_len=model.args.context_window)

# trainer.benchmark()

trainer.train(dl, dl)
