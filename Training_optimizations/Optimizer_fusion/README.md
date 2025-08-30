# [How to save memory by fusing the optimizer step into the backward pass](https://docs.pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html)


## Memory usage during training
1. Model params (P)
2. Activations (A)
3. Gradients (G = P)
4. Optimizer state (if optimizer is statefull - like Adam) (O = 2P)
5. Intermediate tensors

The key take in this approach, is that instead of saving the gradients first and then updating parameters, we apply the optimizer immediatelly once the gradient has been accumulated. This removes the need to hold onto a big buffer of gradients until the optimizer step.

E.g. instead of
```python
model = Cirilla(Args())
model = torch.compile(model, mode='reduce-overhead')
optim = _AdamW(model.parameters(), bf16_stochastic_round=True, lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for i in range(100):
    out = model.pred(x)
    loss = criterion(out.view(-1, 50_000), x.view(-1))

    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
```
we can do
```python
model = Cirilla(Args())
model = torch.compile(model, mode='reduce-overhead')
criterion = torch.nn.CrossEntropyLoss()

optimizer_dict = {p: torch.optim.AdamW([p]) for p in model.parameters()}

def optimizer_hook(parameter) -> None:
    optimizer_dict[parameter].step()
    optimizer_dict[parameter].zero_grad()

for p in model.parameters():
    p.register_post_accumulate_grad_hook(optimizer_hook)

for i in range(100):
    out = model.pred(x)
    loss = criterion(out.view(-1, 50_000), x.view(-1))

    # optim.zero_grad(set_to_none=True)
    loss.backward() # this single call calculates gradients and updates the weights
    # optim.step()
```

In this case way we saved around 0.8GiB of memory, which in that case allows us to increase the vocabulary size from 30'000 to 50'000

```python
@dataclass
class Args:
    device:torch.device = select_torch_device()
    vocab_size:int = 30_000 -> 50_000
    context_window:int = 2048 # seq len
    dim:int = 1024
    d_ff:int = 2048
    window_size:int = 1024
    n_heads:int = 8
    n_kv_heads:int = 4
    static_mask:bool = True
    soft_cap:Optional[int] = 20
    n_layers:int = 16
    theta:float = 10_000.0
    num_experts:int = 8
    k:int = 4
    static_mask:bool = True
    dtype:torch.dtype = torch.bfloat16

    warnings.warn("hf kernels only work on cuda")
    assert dim % n_heads == 0
    assert n_heads % n_kv_heads == 0
```