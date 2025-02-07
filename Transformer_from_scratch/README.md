# Attention is all you need

## What is a transformer
Transformer is a model architecture made for transduction problems — like translation — that resolves one of the big problems of sequential information, meaning sequence length and the lack of parallelization possibility.

## Previous SOTA
Previous established approaches mainly focused on using recurrent neural networks, they generate a sequence of hidden state _h<sub>t</sub>_ based on the previous hidden states _h<sub>t-1</sub>_. This is what precludes parallelization — the fact that we need to go through each sequence step one by one.

## What is different
Transformers rely on attention rather than seqential processing. Meaning that we are able to process the whole sequence as one time step _O(1)_ instead of n-time steps _O(n)_.

### Attention
As explaned in [ ] let _x = (sequence length vector)<sup>T</sup>_ denote a sequence of inputs e.g. _x = What can owls do?_ and let _q_ be a query (a vector representation of the query) e.g. the RNN hidden state of the target encoder _q = Owls can fly_, and _z_ represents the source position we want to attend to. Our aim is to produce context _c_ based on the sequence and the query, we do this by assuming _attention distribution_ _z~p(z | x,q)_, so the context over a sequence will be the expectation _c=E<sub>z~p(z | x,q)</sub>[f(x,z)]_, where f(x,z) in an annotation function. In the context of deep/neural networks both _annotation function_ and _attention distibution_ are parametrized. In that case our _attention distribution_ is simply _p(z=i | x,q) = softmax(MLP([x<sub>i</sub> ; q]))_, which gives us:
<div align='center'>
<em>c = E<sub>z~p(z | x,q)</sub>[f(x,z)] = Σ p(z | x,q)x<sub>i</sub> = Σ softmax(MLP([x<sub>i</sub> ; q]))x<sub>i</sub></em>
</div>
</br>
Which with the same general logic can be expressed as "Scaled Dot-Product Attention" in the transformer architecture [ ].

</br>
<div align='center'>
<em>Attention(Q,K,V) = softmax( QK<sup>T</sup> / sqrt(d<sub>k</sub> )) V</em>
</div>
</br>
Reffered to later as attention / self-attention / cross-attention. The idea of attention is best explained with a cross-attention visualization for the example inputs: What can owls do? - Owls can fly, here the input would be the question "What can owls do?" (K and V) and expected output "Owls can fly" (Q), so the attention can be visualized with <em>softmax( QK<sup>T</sup> / sqrt(d<sub>k</sub> ))</em>

![cross-attention-matrix](./img/corssattentionmatrix.png)
<div align='center'>
<em>fig.1 Corss-attention visualization — brigther colors denote higher attention/importance</em>
</div>
