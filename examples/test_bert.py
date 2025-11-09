from cirilla.Cirilla_model import (
                            CirillaBERT,
                            BertArgs,
                            TrainingArgs,
                            CirillaTrainer,
                            CirillaTokenizer,
                            JSONLDataset,
                            bert_training_step,
                            bert_inference_step
                            )
import torch.nn as nn
from torch.optim import AdamW
from types import MethodType

model = CirillaBERT(BertArgs(output_what='classify', moe_type='pytorch'))

targs = TrainingArgs(
n_epoch = 4,
optim = AdamW,
use_muon_optim=True, # use Muon optimizer for hidden layers
lr = 1e-3,
batch_size = 4,
valid_every_n = 5, # validate on the validation dataset every n epochs
save_local_async = False, # you can save the model asynchronously to the local dir (not recommended if the model can crash but saves time)
init_method_str = 'xavier_uniform_', # initiate with xavier initialization
local_checkpoint_folder = './bert_model',
optim_kwargs = {'fused':True, 'foreach':False},
renew_training = False, # renew the training session if the training failed and there exists a local checkpoint
save_checkpoint_n_iterations = 10,
save_checkpoint_min = 1,
push_checkpoint_to_hub = False, # push the checkpoint to the huggingface hub you specified
push_checkpoint_to_hub_n_local_saves=2,
hf_repo_id='username/repo',
private_hf_repo=False, # set to true if you want to make the repo private
hf_tags = ["pytorch", "text-generation", "moe", "custom_code"],
hf_license = "mit",
model_card = "This is my model card I want to appear on the huggingface hub"
)

trainer = CirillaTrainer(model, targs)

tokenizer = CirillaTokenizer(hub_url='AnthonyPa57/HF-torch-demo2')

dl = JSONLDataset('./examples/data/example_bert.jsonl', shuffle_path=True) # list of paths, obviously they don't need to be the same, it can be one file instead

from cirilla.Cirilla_model.tokenizer_modules import SPECIAL_TOKENS

tokenizer.train(dl, special_tokens=SPECIAL_TOKENS, min_frequency=2)
# tokenizer.push_to_hub('AnthonyPa57/HF-torch-demo2')

dl = JSONLDataset(['./examples/data/example_bert.jsonl', './examples/data/example_bert.jsonl'],
                    shuffle_path=True, tokenizer=tokenizer, max_len=model.args.context_window)

trainer.training_step = MethodType(bert_training_step, trainer) # assign the custom training step to the trainer
trainer.inference_step = MethodType(bert_inference_step, trainer) # assign the custom inference step to the trainer
trainer.criterion = nn.CrossEntropyLoss() # override the default criterion

trainer.train(dl, dl)