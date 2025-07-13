# LLama 2 Architecture

## Embeddings
The Embeddings for llama 2 share a common principle - they turn sentences into tokens (usually [BPE](https://huggingface.co/learn/llm-course/en/chapter6/5) tokenizer) and then into values that encode some information.

In llama the embeddding dim is 4096.

Then (in vanilla transformer) we add positional embeddings and then we get the stuff we feed into the network - they are called absolute possitional embeddings.

In Llama we use RoPE, we also compute them before the calculation of the attention and they are calculated for Q and K only. RoPE are something between absolute and relative mebeddings - RoPE were introduced as a way to find an inner product between Q and K that only depends on their relative distance, this is done through Euler's formula (the one when we take the power of e^[x] to get a rotation), so the tokens that are in a similar position are kinda similar.

>[!NOTE] inner product is a "generalization" of the dot product

The vanilla rotational metrix is very sparse, so we do this
![](../img/rotary2.png)

to get the embedding - where m is the token position and theta is precomputed as a series. We can precompute the matrices with cos and sin as they depend only on position.

The motivation behind complex numbers in the RoPE are used to roatations as Euler's formula says:
$e^{ix} = \cos (x) + i \cdot \sin (x)$

So we can represent complex numbers in trygonometric forms.