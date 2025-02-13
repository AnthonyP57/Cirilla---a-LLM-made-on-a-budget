# Attention is all you need

## What is a transformer
[^1] Transformer is a model architecture presented initially for transduction problems — like translation — that resolves one of the big problems of sequential information, meaning sequence length and the lack of parallelization possibility.

### Architecture overview

<p align="center">
  <img src="./img/transformerarch.png" alt="transformer-architecture" width="400"/>
</p>

<div align='center'>
<em>fig.1 Transformer architecture overview [^1]</em>
</div>

## Previous SOTA
Previous established approaches mainly focused on using recurrent neural networks, they generate a sequence of hidden state _h<sub>t</sub>_ based on the previous hidden states _h<sub>t-1</sub>_. This is what precludes parallelization — the fact that we need to go through each sequence step one by one.

## What is different
Transformers rely on attention rather than seqential processing. Meaning that we are able to process the whole sequence as one time step _O(1)_ instead of n-time steps _O(n)_.

### Attention
As explaned in [^2] let _x = (sequence length vector)<sup>T</sup>_ denote a sequence of inputs and let _q_ be a query (a vector representation of the query) e.g. the RNN hidden state of the target encoder, and _z_ represents the source position we want to attend to. Our aim is to produce context _c_ based on the sequence and the query, we do this by assuming _attention distribution_ _z~p(z | x,q)_, so the context over a sequence will be the expectation _c=E<sub>z~p(z | x,q)</sub>[f(x,z)]_, where f(x,z) in an annotation function. In the context of deep/neural networks both _annotation function_ and _attention distibution_ are parametrized. In that case our _attention distribution_ could be simply _p(z=i | x,q) = softmax(MLP([x<sub>i</sub> ; q]))_, which gives us:
<div align='center'>
<em>c = E<sub>z~p(z | x,q)</sub>[f(x,z)] = Σ p(z | x,q)x<sub>i</sub> = Σ softmax(MLP([x<sub>i</sub> ; q]))x<sub>i</sub></em>
</div>
</br>
Which with the same general logic can be expressed as "Scaled Dot-Product Attention" in the transformer architecture [^1].

</br>
<div align='center'>
<em>Attention(Q,K,V) = softmax( QK<sup>T</sup> / sqrt(d<sub>k</sub> )) V</em>
</div>
</br>
Reffered to later as attention / self-attention / cross-attention. The idea of attention is best explained with a cross-attention visualization for the example inputs: What can owls do? - Owls can fly, here the input would be the question "What can owls do?" (K and V - encoder output) and expected output "Owls can fly" (Q - decoder input), so the attention can be visualized with <em>softmax( QK<sup>T</sup> / sqrt(d<sub>k</sub> ))</em>

<p align="center">
  <img src="./img/corssattentionmatrix.png" alt="cross-attention-matrix" width="500"/>
</p>

<div align='center'>
<em>fig.2 Corss-attention visualization — brigther colors denote higher attention/importance</em>
</div>

## How to encode seqential information without sequential processing
The model so far — while good at finding important dependencies through attention — doesn't retain any information about the order of the sentence, this information is "injected" with _positional encodings_. In the original paper [^1] sine and cosine functions are used:
<div align='center'>
<em>even positions: PE<sub>(pos, 2i)</sub> = sin(pos/1000<sup>2i / d<sub>model</sub></sup>)</em>
</br>
<em>odd positions: PE<sub>(pos, 2i+1)</sub> = cos(pos/1000<sup>2i / d<sub>model</sub></sup>)</em>
</div>
</br>

<p align="center">
  <img src="./img/posemb.png" alt="positional-encodings"/>
</p>

<div align='center'>
<em>fig.3 Positional Encodings visualization [^3]</em>
</div>

## Tweaks for numerical stability
Some parts of the model in the implementation may differ to what has been originally presented. Namely it is the positional encodings and the order of normalization, both of those changes have been made to speed up convergence of the model

## Results
First making sense results appeared at around epoch 7
```
INPUT: Levin felt so strong and calm that he thought the answer, whatever it might be, could not agitate him, but he did [MASK] at all expect the reply Oblonsky gave him.    

EXPECTED: Levin felt so strong and calm that he thought the answer , whatever it might be , could not agitate him , but he did not at all expect the reply Oblonsky gave him .   
                                                   
PREDICTED: Levin felt so strong and calm that he thought to answer . John , whatever , could be , could not arrange the answer to - door , to the reply Oblonsky gave him .    
```

## Citations


[^1]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is All you Need. arXiv (Cornell University), 30, 5998–6008. https://arxiv.org/pdf/1706.03762v5


[^2]: Kim, Y., Denton, C., Hoang, L., & Rush, A. M. (2017). Structured attention networks. arXiv (Cornell University). https://arxiv.org/pdf/1702.00887


[^3]: https://dongkwan-kim.github.io/blogs/a-short-history-of-positional-encoding/


> [!CAUTION]
> This repo does not serve to amazingly describe and explain model architectures, it was made to give a broad simplified overview of the models and implement them.