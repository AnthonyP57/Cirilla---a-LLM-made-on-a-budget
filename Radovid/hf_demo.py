import torch
import torch.nn as nn
import torch.optim as optim
from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from huggingface_hub import HfApi, PyTorchModelHubMixin
from huggingface_hub.repocard import metadata_eval_result, metadata_save
from pathlib import Path
import json
import tempfile

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

def prepare_tokenizer(save_dir: Path) -> PreTrainedTokenizerFast:
    save_dir.mkdir(parents=True, exist_ok=True)
    sample = ["hello world! this is a tiny corpus.", "demo sentence."]
    corpus = save_dir / "corpus.txt"
    with open(corpus, "w") as f:
        f.write("\n".join(sample))

    spm = SentencePieceBPETokenizer()
    spm.train(files=[str(corpus)], vocab_size=10, min_frequency=1)

    tokenizer_json = save_dir / "tokenizer.json"
    spm._tokenizer.save(str(tokenizer_json))

    return PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json),
        unk_token="<unk>", pad_token="<pad>",
        cls_token="<s>", sep_token="</s>", mask_token="<mask>"
    )

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
    "out_size": 64,
    "hidden_size": 16,
    "epochs": 3,
    "batch_size": 16,
    'lr': 1e-3
}

data = torch.randn(512, hypers['in_size'], device=device)
labels = torch.randint(0, hypers['out_size'], (512,), device=device)
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(data, labels),
    batch_size=hypers['batch_size'], shuffle=True
)

model = NN(hypers['in_size'], hypers['hidden_size'], hypers['out_size']).to(device)
optimizer = optim.Adam(model.parameters(), lr=hypers['lr'])
criterion = nn.CrossEntropyLoss()
total_loss = 0.0
for epoch in range(1, hypers['epochs'] + 1):
    epoch_loss = 0.0
    for x, y in loader:
        x = x.to(device).to(torch.bfloat16)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, hypers['out_size']), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x.size(0)
    avg_loss = epoch_loss / len(loader.dataset)
    total_loss = avg_loss
    print(f"Epoch {epoch}/{hypers['epochs']} loss: {avg_loss:.3f}")

def get_core_metadata(m: nn.Module) -> dict:
    return {
        "model_size": f"{sum(p.numel() for p in m.parameters()):,}",
        "tensor_type": str(next(m.parameters()).dtype).split('.')[-1],
    }

def push_to_hub(repo_id,
                model,
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
          "custom-implementation",
          "Mixture of Experts"
      ]
    metadata['library'] = 'pytorch'
    metadata['license'] = 'mit'

    eval = metadata_eval_result(
        model_pretty_name=repo_name,
        task_pretty_name="text-generation",
        task_id="text-generation",
        metrics_pretty_name="mse",
        metrics_id="mse",
        metrics_value=f"{total_loss:.3f}",
        dataset_pretty_name='random',
        dataset_id='random123',
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
    # cartpole_policy.save_pretrained("AnthonyPa57/HF-torch-demo") #save locally

    tok = prepare_tokenizer(local_directory)
    tok.push_to_hub(repo_id)
    metadata_save(readme_path, metadata)


    api.upload_folder(
          repo_id=repo_id,
          folder_path=local_directory,
          path_in_repo=".",
    )

    print(f"Your model is pushed to the Hub. You can view your model here: {repo_url}")


# ===== Example Usage =====
if __name__ == "__main__":
    push_to_hub("AnthonyPa57/HF-torch-demo", model, hypers)
    # Later, to load:
    config = {
        "in_size": hypers['in_size'], "out_size": hypers['out_size'], "h_size": hypers['hidden_size'],}
    model_hf = NN(**config)
    model_hf.from_pretrained("AnthonyPa57/HF-torch-demo")
    tokenizer_hf = AutoTokenizer.from_pretrained("AnthonyPa57/HF-torch-demo")

    # print(model, tokenizer_hf)

    print(tokenizer_hf.decode(tokenizer_hf.encode("hello world")))