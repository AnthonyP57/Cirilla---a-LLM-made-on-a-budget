from huggingface_hub import HfApi, PyTorchModelHubMixin, hf_hub_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save
import json
import tempfile
from pathlib import Path
from .model import Cirilla
from dataclasses import dataclass
import torch

def push_model_to_hub(repo_id,
                model:Cirilla,
                loss:float,
                dataset_name:str,
                private:bool=False,
                optmizer_states_path:Path=None,
                tags:list[str]=["pytorch", "text-generation", "moe", "custom_code"],
                license:str='mit',
                languages:list[str] = ['en'],
                model_card:str = None
                ):

  repo_name = repo_id
  api = HfApi()

  repo_url = api.create_repo(
        repo_id=repo_id,
        private=private,
        exist_ok=True,
  )

  with tempfile.TemporaryDirectory() as tmpdirname:
    local_directory = Path(tmpdirname)

    metadata = {}
    metadata["tags"] = tags
    metadata['library'] = 'pytorch'
    metadata['license'] = license
    metadata['language'] = languages

    eval = metadata_eval_result(
        model_pretty_name=repo_name,
        task_pretty_name="text-generation",
        task_id="text-generation",
        metrics_pretty_name="Cross Entropy Loss",
        metrics_id="CEL",
        metrics_value=f'{loss:.3f}',
        dataset_pretty_name=dataset_name,
        dataset_id=dataset_name,
      )

    metadata = {**metadata, **eval}

    if model_card is None:
        model_card = f"""
# Random Pytorch model used as a demo to show how to push custom models to HF hub
| parameters | precision |
| :--------: | :-------: |
|{((model.n_params/1e6) if hasattr(model, 'n_params') else int(sum(p.numel() / 1e6 for p in model.parameters()))):.2f} M|{'BF16' if model.args.dtype == torch.bfloat16 else 'FP32'}|
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

    if optmizer_states_path is not None:
        local_states = local_directory / "optimizer_states.pt"
        states = torch.load(optmizer_states_path, map_location='cpu')
        torch.save(states, local_states)
          

    model.push_to_hub(repo_id)

    metadata_save(readme_path, metadata)

    api.upload_folder(
          repo_id=repo_id,
          folder_path=local_directory,
          path_in_repo=".",
    )

    # print(f"Your model is pushed to the Hub. You can view your model here: {repo_url}")

if __name__ == "__main__":

    from dataclasses import dataclass
    import torch
    import torch.nn as nn
    import torch.optim as optim

    @dataclass
    class hypers:
        in_size: int = 64
        out_size: int = 1
        hidden_size: int = 16
        epochs: int = 3
        batch_size: int = 16
        lr: float = 1e-3
        dtype_str:str = 'bfloat16'

        @property
        def dtype(self):
            return getattr(torch, self.dtype_str)

    class NN(
        nn.Module,
        PyTorchModelHubMixin,
        pipeline_tag="text-generation",
        library_name="pytorch",
        license="mit"
    ):
        def __init__(self, args:dataclass):
            super().__init__()
            if isinstance(args, dict):
               args = hypers(**args)
            self.l1 = nn.Linear(args.in_size, args.hidden_size)
            self.l2 = nn.Linear(args.hidden_size, args.out_size)
            self.to(dtype=args.dtype)

        def forward(self, x):
            x = self.l1(x)
            x = torch.relu(x)
            x = self.l2(x)
            return x

    args = hypers()
    
    print(args.dtype)

    dataset = torch.utils.data.TensorDataset(torch.rand(1024, args.in_size), torch.rand(1024, args.out_size))
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=16)

    model = NN(args)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for x, y in dataloader:
        x, y = x.to(dtype=args.dtype), y.to(dtype=args.dtype)
        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    push_model_to_hub("AnthonyPa57/HF-torch-demo2", model, loss, 'pretraining', save_locally='./test_model')
    
    repo_id = "AnthonyPa57/HF-torch-demo2"
    filename = "config.json"

    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
    )

    with open(file_path, "r") as f:
        config = json.load(f)

    args = hypers(**config[list(config.keys())[0]])
    print(args)

    model_hf = NN(args)
    model_hf.from_pretrained("AnthonyPa57/HF-torch-demo2")

    print(model_hf)

    with open('./test_model/config.json', 'r') as f:
        config = json.load(f)
    args = hypers(**config[list(config.keys())[0]])
    model_hf = NN(args)
    model_hf.from_pretrained('./test_model')
    print(model_hf)