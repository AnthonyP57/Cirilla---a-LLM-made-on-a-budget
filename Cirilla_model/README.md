# Cirilla
## Pushing custom Pytoch models to Huggingface Hub
I suggest checking out `hf_demo.py` for a concrete example.

1. Create the model, it has to inherit `PyTorchModelHubMixin`, you can add tags to the model as well.
```python
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
```
2. define hyperparameters (optional)
```python
hypers = {
    "in_size": 64,
    "out_size": 1,
    "hidden_size": 16,
    "epochs": 3,
    "batch_size": 16,
    "lr": 1e-3
}
```
3. train the model normally
```python
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
```
4. you can simply push with
```python
model.push_to_hub("AnthonyPa57/HF-torch-demo")
```
5. or with function in hf_demo `push_to_hub` that create a repo and pushes the model there

### Saving the tokenizer
6. I suggest training the model and then passing its learnt mappings into `PreTrainedTokenizerFast`
```python
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

tokenizer = prepare_tokenizer('./Radovid_model', tokenizer_data)
```
7. You can then push to hub
```python
tokenizer.push_to_hub("AnthonyPa57/HF-torch-demo")
```
you can see an example repo [here](https://huggingface.co/AnthonyPa57/HF-torch-demo)

## Pulling custom Pytorch models from Huggingface Hub
After you have pushed the model you can pull it as
```python
config = {...}

model_hf = NN(**config)
model_hf.from_pretrained("AnthonyPa57/HF-torch-demo")

tokenizer_hf = AutoTokenizer.from_pretrained("AnthonyPa57/HF-torch-demo")
```