# What is FlexAttention
There are many implementations of memory and compute efficient attention mechanisms, however they may require e.g. writing a custom kernel.

[FlexAttention](https://pytorch.org/blog/flexattention/) _"allows for the flexibility of Pytorch with the performance of FlashAttention"_. The custom attention mechanism can be turned into a fused kernel with ```torch.compile```, in consequence not materializing any extra memory.

Checkout [Attention Gym](https://github.com/pytorch-labs/attention-gym/blob/6a65742f797c0837200516589b721703b42b2ab3/examples/flex_attn.ipynb) for examples.

## Overview
Attention as we all know and love:

$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$

```
Q, K, V: Tensor[batch_size, num_heads, sequence_length, head_dim]
score: Tensor[batch_size, num_heads, sequence_length, sequence_length] = (Q @ K) / sqrt(head_dim)
probabilities = softmax(score, dim=-1)
output: Tensor[batch_size, num_heads, sequence_length, head_dim] = probabilities @ V
```

FlexAttention allows for a user-defined function ```score_mod```:

$FlexAttention(Q,K,V) = softmax(scoremod(\frac{QK^T}{\sqrt{d_k}}))V$

```
Q, K, V: Tensor[batch_size, num_heads, sequence_length, head_dim]
score: Tensor[batch_size, num_heads, sequence_length, sequence_length] = (Q @ K) / sqrt(head_dim)
modified_scores: Tensor[batch_size, num_heads, sequence_length, sequence_length] = score_mod(score)
probabilities = softmax(modified_scores, dim=-1)
output: Tensor[batch_size, num_heads, sequence_length, head_dim] = probabilities @ V
```

```score_mode``` can be thought to work like:
```python
for b in range(batch_size):
    for h in range(num_heads):
        for q_idx in range(sequence_length):
            for kv_idx in range(sequence_length):
                modified_scores[b, h, q_idx, kv_idx] = score_mod(scores[b, h, q_idx, kv_idx], b, h, q_idx, kv_idx)
```

### Examples
### "Standard" Attention
```python
from torch.nn.attention.flex_attention import flex_attention

def noop(score, b, h, q_idx, kv_idx):
    return score

flex_attention(query, key, value, score_mod=noop).sum().backward()
```

### Relative position encoding
Adjusts the distance between the queries and keys e.g.
```python
Relative Position Matrix:
[[ 0  1  2  3  4]
 [-1  0  1  2  3]
 [-2 -1  0  1  2]
 [-3 -2 -1  0  1]
 [-4 -3 -2 -1  0]]
```
```python
def relative_positional(score, b, h, q_idx, kv_idx):
    return score + (q_idx - kv_idx)
```
This does not materialize the SxS matrix, instead it is calculated on the fly.

### [Alibi bias](https://arxiv.org/pdf/2108.12409)
It is similar to positional encodings, but it has a per-head factor, typically precomputed.
```python
alibi_bias = generate_alibi_bias() # [num_heads]

def alibi(score, b, h, q_idx, kv_idx):
    bias = alibi_bias[h] * (q_idx - kv_idx)
    return score + bias
```

### Soft-capping
Is a techinque used in Gemma2 adn Grok-1 that prevents logits from growing excessiviely.
```python
softcap = 20
def soft_cap(score, b, h, q_idx, kv_idx):
    score = score / softcap
    score = torch.tanh(score)
    score = score * softcap
    return score
```
Even though this implementation is semantically correct we would like to use a tanh approximation.

### Causal mask
```python
def causal_mask(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, -float("inf"))
```
However if something is masked we can completely skip its computation, and causal masks have about 50% sparsity. That's why FlexAttention can use [mask_mod](#mask-mod)

```python
from torch.nn.attention.flex_attention import create_block_mask

def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

# Because the sparsity pattern is independent of batch and heads, we'll set them to None (which broadcasts them) 
block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=1024, KV_LEN=1024)
# In this case, we don't need a score_mod, so we won't pass any in.
# However, score_mod can still be combined with block_mask if you need the additional flexibility.
flex_attention(query, key, value, block_mask=block_mask)
```
Although ```create_block_mask``` is an expensive operation.

#### Mask mod
To tak advantage of sparsity we can create a BlockMask, FlexAttention can use to to take advantage of sprasity. The signature of ```mask_mod``` is similar to ```score_mod```
```python
# returns True if this position should participate in the computation
mask_mod(b, h, q_idx, kv_idx) => bool
```
For masking it is recommended to use ```mask_mod``` and ```create_block_mask``` as it is more performant

### Slinding window + Causal
Used in Mistral, sliding window attention also known as local attention takes advantage of the intuition that the most recent tokens are the most useful, it is often used together with causal attention.

```python
SLIDING_WINDOW = 1024

def sliding_window_causal(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    window_mask = q_idx - kv_idx <= SLIDING_WINDOW 
    return causal_mask & window_mask

# If you want to be cute...
from torch.nn.attention import or_masks

def sliding_window(b, h, q_idx, kv_idx)
    return q_idx - kv_idx <= SLIDING_WINDOW

sliding_window_causal = or_masks(causal_mask, sliding_window)
```

### PrefixLM
Performs full bidirectional attention on a "prefix" and a causal attention on the rest.
```python
prefix_length: [B]
def prefix_mask(b, h, q_idx, kv_idx):
    return kv_idx <= prefix_length[b]

prefix_lm_causal = or_masks(prefix_mask, causal_mask)
# In this case, our mask is different per sequence so we set B equal to our batch size
block_mask = create_block_mask(prefix_lm_causal, B=B, H=None, S, S)
```
The sparsity changes per input, this means that for each new input we will need to recompute the ```BlockMask```, we can instead call ```create_block_mask``` at the beggining and reuse that block_mask for all attention call in the model.

### Document Masking/Jagged Sequences
Imagine that you have sequences of varying lengths, you want to train on all of them but unfortunately most operations only accept rectangular tensors. This is supported through ```BlockMask```:
1. Flatten all sequences into a single sequence.
2. Compute the document_id that each token belongs to.
3. In ```mask_mod``` we pass whether the Q and KV token belongs to the same document.

```python
# The document that each token belongs to.
# e.g. [0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2] corresponds to sequence lengths 3, 2, and 6.
document_id: [SEQ_LEN]

def document_masking(b, h, q_idx, kv_idx):
    return document_id[q_idx] == document_id[kv_idx]
```
We end up with a blockdiagonal mask
![](./img/blockdiagonal.avif)

#### Combination of masks
For combination of masks we can use another function to make it e.g.
```python
def generate_doc_mask_mod(mask_mod, document_id):
    # Get unique document IDs and their counts
    _, counts = torch.unique_consecutive(document_id, return_counts=True)
    # Create cumulative counts (offsets)
    offsets = torch.cat([torch.tensor([0], device=document_id.device), counts.cumsum(0)[:-1]])
    def doc_mask_wrapper(b, h, q_idx, kv_idx):
        same_doc = document_id[q_idx] == document_id[kv_idx]
        q_logical = q_idx - offsets[document_id[q_idx]]
        kv_logical = kv_idx - offsets[document_id[kv_idx]]
        inner_mask = mask_mod(b, h, q_logical, kv_logical)
        return same_doc & inner_mask
    return doc_mask_wrapper
```
```python
prefix_length = torch.tensor(2, dtype=torch.int32, device="cuda")

def prefix_mask(b, h, q_idx, kv_idx):
    return kv_idx < prefix_length
prefix_lm_causal = or_masks(prefix_mask, causal_mask)
doc_prefix_lm_causal_mask = generate_doc_mask_mod(prefix_lm_causal, document_id)
```
We get a block-prefixLM-diagonal shaped mask
![](./img/prefixcausal.avif)

## [FlexAttention for inference](https://pytorch.org/blog/flexattention-for-inference/)
```torch.compile``` lowers ```flex_attention``` to a fused kernel. There is a dedicated FlexDecoding backend optimiez for long-context LLM inference incorporating decoder-specific techniques. ```flex_attention``` automatically switches to the FlexDecoding backend for JIT compilation when given a short query and a long KV cache.

```python
flex_attention = torch.compile(flex_attention)

k_cache = torch.random(B, H, 16384, D) 
v_cache = torch.random(B, H, 16384, D)

...

# Prefill Phase: query shape = [B, H, 8000, D]
flex_attention(q_prefill, k_cache, v_cache, ...) # Uses FlexAttention backend optimized for prefill & training

# Decoding Phase: q_last_token shape = [B, H, 1, D]
flex_attention(q_last_token  , k_cache, v_cache, ...) # Recompiles with the FlexDecoding backend 

# decode 2 tokens at the same time: q_last_2_tokens shape = [B, H, 2, D]
flex_attention(q_last_2_tokens, k_cache, v_cache, ...) # No recompilation needed! Runs the decoding kernel again.
```

### KV Cache
FlexDecoding takes a user-defined ```mask_mod``` and ```score_mod``` functions. In the decoding phas previously calculated tokens are cached and only the latest generated token is used as the query.
```python
# a naive way
def causal(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, -float("inf"))
```
But this is problematic as a nwe token should attend to all previously generated tokens, that's why we can introduce an ```offset```

```python
offset = torch.tensor(i, "cuda")
def causal_w_offset(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx + offset >= kv_idx, score, -float("inf"))

# Attend the i-th token
flex_attention(..., score_mod=causal_w_offset  ) # Compiles the kernel here 
...
# Attend the i+1-th token
offset = offset + 1 # Increment offset
flex_attention(..., score_mod=causal_w_offset ) # Doesn't need to recompile! 
```
Offset becomes a captured tensor so if we changee it we don't need to recompile.

We don't need to handle offset by hand
```python
offset = torch.tensor(i, "cuda")

def get_score_mod_w_offset(score_mod: _score_mod_signature, _offset: tensor):
    def _score_mod(score, b, h, q, kv):
        return score_mod(score, b, h, q + _offset, kv)
    return _score_mod

def get_mask_mod_w_offset(mask_mod: _mask_mod_signature, _offset: tensor):
    def _mask_mod(b, h, q, kv):
        return mask_mod(b, h, q + _offset, kv)
    return _mask_mod

causal_w_offset = get_score_mod_w_offset(causal, offset)
```

### BlockMask for inference
The idea is to precompute the BlockMask once during model setup and use slices of it during decoding.
```python
from torch.nn.attention.flex_attention import create_block_mask

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

block_mask = create_block_mask(causal_mask, B=None, H=None, Q_LEN=MAX_SEQ_LEN,KV_LEN=MAX_SEQ_LEN)
```
For the i-th token
```python
block_offset = i // block_mask.BLOCK_SIZE[0]
block_mask_slice = block_mask[:, :, block_offset]

# don't forget to use the mask_mod with offset! 
block_mask_slice.mask_mod = get_mask_mod_w_offset(causal_mask)
```

### Paged attention
PagedAttention scatters KV cache to reduce memory fragmentation and support higher batch sizes, with PagedAttention we can chunk each request into multiple pages of the same size page_size and scatter into a physical KV cache. This avoids padding requests to the same length and saves memory.

One question is how to reqrite user-specified ```mask_mod``` and ```score_mod``` for PagedAttention. The following code shows an automated conversion at runtime. The ```new_mask_mod``` would take the physical_kv_idx and convert it back into the logical_kv_idx and apply user-specified ```mask_mod``` on the logical_kv_idx, the out-of-boundary blocks are masked with ```torch.where```. After batching logical KV caches into the same physical KV cache there are much more physical blocks then the number of logical blocks. By masking with ```torch.where``` we are ensuring that data from different requests does not interfere with each other.

[Code](https://github.com/pytorch-labs/attention-gym/blob/main/attn_gym/paged_attention/paged_attention.py)
```python
def get_mask_mod(mask_mod: Optional[_mask_mod_signature]) -> _mask_mod_signature:
    if mask_mod is None:
        mask_mod = noop_mask

    def new_mask_mod(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        physical_kv_idx: torch.Tensor,
    ):
        physical_kv_block = physical_kv_idx // page_size
        physical_kv_offset = physical_kv_idx % page_size
        logical_block_idx = physical_to_logical[b, physical_kv_block]
        logical_kv_idx = logical_block_idx * page_size + physical_kv_offset
        return torch.where(
            logical_block_idx >= 0, mask_mod(b, h, q_idx, logical_kv_idx), False
        )

    return new_mask_mod
```

PagedAttention shows 5% less overhead from FlexAttention.

## FAQ
### When flexattention needs to recompile
It does not need ot recompile even if the caputed tensor changes values e.g.
```python
flex_attention = torch.compile(flex_attention)
def create_bias_mod(bias)
    def bias_mod(score, b, h, q_idx, kv_idx):
        return score + bias
    return bias_mod
bias_mod1 = create_bias_mod(torch.tensor(0))
flex_attention(..., score_mod=bias_mod1) # Compiles the kernel here 

bias_mod2 = create_bias_mod(torch.tensor(2))
flex_attention(..., score_mod=bias_mod2) # Doesn't need to recompile! 
```
But is the block-sparsity changes we need to recompile the BlockMask

### When to recompute the BlockMask
Whenever the block-sparsity changes, but recomputing is much cheaper than recompilation (ms vs sec)

### How to compute BlockMask quicker
1. Broadcast with setting arguments to ```None```
2. Compile ```create_block_mask```, it may not work directly, but you can seet ```_compile=True``` to reduce peak memory and runtime
3. Write a custom constructor for BlockMask

### Performance
Generally speaking FlexAttention is nearly as performant as a handwritten Triton kernel

### Performance tuning for FlexAttention
For optimal performance, compile FlexAttention using max-autotune, especially when dealing with complex score_mods and mask_mods.
```python
flex_attention = torch.compile(flex_attention, dynamic=True, mode=’max-autotune’)
```
While compilation takes longer, the optimal configuration is cached for future kernel execution.