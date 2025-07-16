# Mistral

> [!IMPORTANT]
> A lot of these notes were made with the meritorical help of [Umar Jamil](https://www.youtube.com/@umarjamilai) - through his content and comments to Mistral source code. So special thanks to him !!!

[^1]Mistral is a French startup, founded in 2023 that specializes in making LLMs. The first paper that Mistal released is Mistral 7B, describing the model they released of the same name.
## Architecture overview
Mistral has multiple key elements in its architecture, with context length of 8k
### Attention
Mistral uses sliding window attention with a rolling buffer KV cache. It also uses [GQA](../README.md).
### Feed Forward
Uses just SiLU function, but also uses MoE - Misture of Experts.

## Architecture details
### Sliding window attention
[^1] [^2]Obviously as Mistral is a decoder only model we need the attention to be causal, what is new is that we apply additionally a mask thats cuts off some of the values below the diagonal with a given offset - in effect words are just related to only few of their predecessors, thus less dot product calculations. Obviously this shriks the local context (smaller receptive field) for each word, but e.g. if you watch the whole Lord of the Rings series you do not necessarily care about the relation of the first 15 minutes of the first movie with the last 15 minutes of the movie, as they are around 12h apart.

![](./img/sliding_window.png)

However now each embedding captures the information of previous n tokens like

![](./img/information_flow.png)

So in total after n layers each embedding captures information about the previous k tokens, just like in CNNs - their receptive field grows with each layer. For sliding window size of n after each layer we add information for each embedding about the previour k = k+(n-1) tokens (generally speaking).

### KV Cache with rolling buffer cache
Regular KV cache looks like
![](../img/kvcache.png)

Since we are using the sliding window attention. We don't need all K and V values, we can limit it to only n previous tokens, with n being the window size.

![](./img/kvcache_rolling.png)

```python
window_size = 3 (cache size)

time_step:1 -> cache = [<SOS>]
time_step:2 -> cache = [<SOS>I]
time_step:3 -> cache = [<SOS>I am]
time_step:4 -> cache = [I am Anthony]
...
```
### Pre-filling and chunking
In KV cache we calculate the attention for Q~(1,dim) and K,V of sizes~(seq_len,dim), this is good but if we have a user prompt that is very long this is quite wasteful to go token by token as we already know what the prompt is and we won't be generating anything anyway.

#### Pre-filling
Instead imagine our windows size is 16 and user prompt is 8 in length, what we do is we use the whole prompt to calculate the attention~(8,8) (as its length is smaller then the window size) and then we generate a new token, and with this new token we come back to vanilla KV cache with rolling buffer, so we calculate the next attention matrices with Q~(1,dim) (being the last generated token) and K,V~(sliding_window,dim) (being the last sliding_window-number of tokens), so we get attention of size~(1,sliding_window).

#### Chunking
![](./img/kvcache_over.png)
But what is the users prompt is bigger then the sliding window e.g. sliding window is of size 16 and prompt is of size 34. We divide the prompt into chunks.
1. First we calculate the attention same as in prefilling, however we fill the whole window, so our attention is of size~(sliding_window, sliding_window), we keep the K,V values in cache (remember that for each layer and each iteration they will change).
![](./img/kvcache_step1.png)

2. Then we calculate attention matrix with next chunk but we also use the cached K,V values, so our attention is of size~(sliding_window, 2*sliding_window), we cache the K,V values for the last sliding_window-number of tokens that were in the new chunk.
![](./img/kvcache_step2.png)


3. Finally we are left with 2 tokens from the prompt, so we use them together with the ones stored in the KV cache so our attention is~(2, sliding_window + 2).
![](./img/kvcache_step3.png)

4. After that when we generate we come back to vanilla KV cache with rolling buffer.

### Sparse Mixture of Experts
MoE - Mixture of Experts is an ensemble technique, where we have multiple experts that sepcialize in different things and their outputs are combined, usualyy theirs logits are combined as a weighted sum.

In SMoE - Sparse MoE we utilize only a few of the experts e.g. 2 out of 8, we choose the experts with a gating mechanism, and then the weight of each expert is calculated as softmax of the top k chosen experts (for each token).
![](./img/smoe.png)

![](./img/smoe_over.png)

![](./img/smoe_mistral.png)

During training we can have many expert models, but during inference we may use just 2, in vanilla MoE we would use all the experts both during training and inference. For instance if we train the model for multiple languages each Expert may learn one of the langauges.

![](./img/smoe_code.png)

### Model Sharding
When we have a model that cannot fit on a single GPU, we can divide the model into groups/chunks/**shards** and place them on different GPUs. But this method in a naive approach is not very efficient as we wait for each subsequent GPU to finsh calculation
![](./img/shard_problem.png)
a better approach is to shift them an the time scale so that alll the GPUs are utilized all the time (Pipeline Paralellizm). From a bigger batch we create micro batches e.g. we have a macro batch of 8, so we divide it into 4 micro batches of 2 items, we still have bubbles when the GPUs aren't working but this is still a getter utilization of GPUs.
![](./img/shard_solution.png)

At inference we just don't have the backward pass.

### Block attention - xformers library
xformers is a library for optimizing inference. When we have multiple API requests a naive way would be to just pad the requests up the the longest one and pass it through the model, however this is not very efficient as we will calculate a lot of meaningless dot products.
![](./img/naive_1.png)
![](./img/naive_2.png)
![](./img/naive_3.png)

Instead we can combine all the tokens into one big sequence and in the attention mask be can mask out any connections between themso they do not have any impact on each other.

![](./img/xformer1.png)
![](./img/xformer2.png)
Block Diagonal Causal Mask

## Disclaimer
> [!CAUTION]
> This repo does not serve to amazingly describe and explain model architectures, it was made to give a broad simplified overview of the models and implement them.

[^1]: Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., De Las Casas, D., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., Lavaud, L. R., Lachaux, M., Stock, P., Scao, T. L., Lavril, T., Wang, T., Lacroix, T., & Sayed, W. E. (2023). Mistral 7B. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2310.06825

[^2]: Umar J. (2023). Mistral / Mixtral Explained: Sliding Window Attention, Sparse Mixture of Experts, Rolling Buffer https://www.youtube.com/watch?v=UiX8K-xBUpE&ab_channel=UmarJamil