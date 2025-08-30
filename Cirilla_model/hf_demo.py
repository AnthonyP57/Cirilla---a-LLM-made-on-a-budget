import torch
import torch.nn as nn
import torch.optim as optim
from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from huggingface_hub import HfApi, PyTorchModelHubMixin
from huggingface_hub.repocard import metadata_eval_result, metadata_save
import json
import tempfile
from typing import Iterator
from pathlib import Path
import os

SPECIAL_TOKENS = {'unk_token':'[UNK]', 'pad_token':'[PAD]', 'mask_token':'[MASK]',
                  'bos_token':'[SOS]', 'eos_token':'[EOS]'}

def sentence_iterator(dataset):
   for item in dataset:
      yield item

def prepare_tokenizer(save_dir, data:Iterator) -> PreTrainedTokenizerFast:
    tokenizer_json = os.path.join(save_dir, 'tokenizer.json')

    if not os.path.exists(tokenizer_json):
        spm = SentencePieceBPETokenizer()
        spm.train_from_iterator(sentence_iterator(data),
                                special_tokens=list(SPECIAL_TOKENS.values()),
                                min_frequency=4,
                                limit_alphabet=1_000,)
                                # vocab_size=50_000,)
        spm.save(tokenizer_json)

    return into_pretrained(tokenizer_json)

def into_pretrained(tokenizer_json):
    return PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json), **SPECIAL_TOKENS)

class NN(
    nn.Module,
    PyTorchModelHubMixin,
    pipeline_tag="text-generation",
    library_name="pytorch",
    license="mit"
):
    def __init__(self, in_size: int=64, h_size: int=16, out_size: int=16):
        super().__init__()
        self.l1 = nn.Linear(in_size, h_size)
        self.l2 = nn.Linear(h_size, out_size)
        self.to(dtype=torch.bfloat16)

    def forward(self, x):
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        return x

hypers = {
    "in_size": 64,
    "out_size": 1,
    "hidden_size": 16,
    "epochs": 3,
    "batch_size": 16,
    "lr": 1e-3
}

dataset = torch.utils.data.TensorDataset(torch.rand(1024, hypers['in_size']), torch.rand(1024, hypers['out_size']))
dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=hypers['batch_size'])

model = NN(hypers['in_size'], hypers['hidden_size'], hypers['out_size'])
optimizer = optim.SGD(model.parameters(), lr=hypers['lr'])
criterion = nn.MSELoss()

for epoch in range(hypers['epochs']):
   for x, y in dataloader:
      x, y = x.to(dtype=torch.bfloat16), y.to(dtype=torch.bfloat16)
      pred = model(x)
      loss = criterion(pred, y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

def get_core_metadata(m: nn.Module) -> dict:
    return {
        "model_size": f"{int(sum(p.numel() / 1000 for p in m.parameters()))} K",
        "tensor_type": 'BF16',
    }

tokenizer_data = ["hello world!",
                  "I am vengeance"]

tokenizer = prepare_tokenizer('./Cirilla_model', tokenizer_data)

def push_to_hub(repo_id,
                model,
                tokenizer,
                hyperparameters,
                ):

  repo_name = repo_id
  api = HfApi()

#   api.delete_repo(repo_id)
  repo_url = api.create_repo(
        repo_id=repo_id,
        # private=True,
        exist_ok=True,
  )

  with tempfile.TemporaryDirectory() as tmpdirname:
    local_directory = Path(tmpdirname)

    # torch.save(model, local_directory / "model.pth")

    with open(local_directory / "hyperparameters.json", "w") as outfile:
      json.dump(hyperparameters, outfile)

    metadata = {}
    metadata["tags"] = [
          "pytorch",
          "text-generation",
          "mixture of experts"
      ]
    metadata['library'] = 'pytorch'
    metadata['license'] = 'mit'

    eval = metadata_eval_result(
        model_pretty_name=repo_name,
        task_pretty_name="text-generation",
        task_id="text-generation",
        metrics_pretty_name="mse",
        metrics_id="mse",
        metrics_value=f"amazing loss",
        dataset_pretty_name='random',
        dataset_id='random',
      )

    metadata = {**metadata, **eval}

    model_card = f"""
# Random Pytorch model used as a demo to show how to push custom models to HF hub
| parameters | precision |
| :--------: | :-------: |
|{get_core_metadata(model)['model_size']}|{get_core_metadata(model)['tensor_type']}|
"""

    readme_path = local_directory / "README.md"
    readme = ""
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
          readme = f.read()
    else:
      readme = model_card

    with readme_path.open("w", encoding="utf-8") as f:
      f.write(readme)

    model.push_to_hub(repo_id)
    # model.save_pretrained("./") # save locally

    tokenizer.push_to_hub(repo_id)
    metadata_save(readme_path, metadata)

    api.upload_folder(
          repo_id=repo_id,
          folder_path=local_directory,
          path_in_repo=".",
    )

    print(f"Your model is pushed to the Hub. You can view your model here: {repo_url}")


push_to_hub("AnthonyPa57/HF-torch-demo", model, tokenizer, hypers)

# Later, to load:
config = {
    "in_size": hypers['in_size'], "out_size": hypers['out_size'], "h_size": hypers['hidden_size']
    }

model_hf = NN(**config)
model_hf.from_pretrained("AnthonyPa57/HF-torch-demo")

tokenizer_hf = AutoTokenizer.from_pretrained("AnthonyPa57/HF-torch-demo")

# print(model, tokenizer_hf)

print(tokenizer_hf.decode(tokenizer_hf.encode("hello world")))