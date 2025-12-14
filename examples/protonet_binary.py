from cirilla.Few_shot import ProtonetDataset, protonet_training_step, protonet_inference_step
from cirilla.Cirilla_model import (CirillaBERT,
                                    BertArgs,
                                    CirillaTokenizer,
                                    CirillaTrainer,
                                    TrainingArgs)
from types import MethodType

tokenizer = CirillaTokenizer(hub_url='AnthonyPa57/HF-torch-demo2')

dl = ProtonetDataset(path=('examples/data/example_bert.jsonl',
                                    'examples/data/example_bert.jsonl'),
                                    tokenizer=tokenizer)

model = CirillaBERT(BertArgs(output_what='meanpool', moe_type='pytorch', n_layers=2, dim=128, d_ff=256, torch_compile=False))

targs = TrainingArgs(batch_size=16)
trainer = CirillaTrainer(model, targs)

trainer.training_step = MethodType(protonet_training_step, trainer)
trainer.inference_step = MethodType(protonet_inference_step, trainer)

trainer.train(dl, dl)