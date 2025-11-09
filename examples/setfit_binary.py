from cirilla.Few_shot import SetfitDataset, setfit_training_step, setfit_inference_step
from cirilla.Cirilla_model import (CirillaBERT,
                                    BertArgs,
                                    CirillaTokenizer,
                                    CirillaTrainer,
                                    TrainingArgs)
from types import MethodType
import torch.nn as nn

tokenizer = CirillaTokenizer(hub_url='AnthonyPa57/HF-torch-demo2')

dl = SetfitDataset(path=('examples/data/example_bert.jsonl',
                                    'examples/data/example_bert.jsonl'),
                                    tokenizer=tokenizer)

model = CirillaBERT(BertArgs(output_what='meanpool', moe_type='pytorch', n_layers=2, dim=128, d_ff=256, torch_compile=False))

targs = TrainingArgs()
trainer = CirillaTrainer(model, targs)

trainer.criterion = nn.TripletMarginLoss() # override the default criterion
trainer.training_step = MethodType(setfit_training_step, trainer)
trainer.inference_step = MethodType(setfit_inference_step, trainer)

trainer.train(dl, dl)