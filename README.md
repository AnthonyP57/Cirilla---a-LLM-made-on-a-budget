> [!IMPORTANT]  
> For a much nicer README visit [Cirilla](https://anthonyp57.github.io/Cirilla---a-LLM-made-on-a-budget/)
> 
> *(Note: the site is made for 16:9 1080p displays — I’m not a web developer, so it may look a bit rough on other screen sizes.)*

![](https://github.com/AnthonyP57/Radovid---a-LLM-made-on-a-budget/blob/master/img/ciri_w4_2.png?raw=true)
*Ciri from The Witcher 4 trailer*

# Cirilla
Cirilla is an open source learning project aiming at implmenting various LLMs.
It is focused mainly on showing how to make, train, infer and deploy a LLM from scratch using Pytorch and a budget friendly GPU (RTX 4060Ti 16GiB ~500$).

- [About Cirilla](#who-is-cirilla)

- [Repo organization](#repo-organization)

- [Getting started](#getting-started)

- [Why Cirilla](#why-cirilla)

## Who is Cirilla
**Cirilla Fiona Elen Riannon**, known as *Ciri*, is one of the central characters in 
*The Witcher* saga by Andrzej Sapkowski and its adaptations.  
She is the princess of Cintra, granddaughter of Queen Calanthe, and the sole heir 
to a powerful lineage marked by the mysterious Elder Blood.

Ciri is defined by her destiny, adaptability, and potential. Unlike kings who wield authority by birthright, her strength comes from surviving chaos, learning from mentors like Geralt and Yennefer, and unlocking extraordinary powers.

Her unique abilities make her one of the most pivotal figures in the saga. Known as the *Lady of Space and Time*, the *Lion Cub of Cintra*, and the *Child of the Elder Blood*, she can manipulate space and time, travel between worlds, and influence the course of events in ways few can.


<p align="center">
  <img src="https://github.com/AnthonyP57/Radovid---a-LLM-made-on-a-budget/blob/master/img/fake_ciri.webp?raw=true" width="250"/>
</p>

<div align='center'>
  <em>Fig.1 Ciri Gwent card by Bogna Gawrońska</em>
</div>
</br>

## Why name a LLM Cirilla
Unlike rulers who inherit authority, *Cirilla* embodies potential realized through learning, experience, and adaptability. She is resilient, capable of navigating complex and unpredictable worlds, and able to respond to challenges with skill and precision - qualities that mirror how an language model can shift between tasks, domains, and contexts.

Guided by mentors and shaped by hardships, Ciri develops her abilities quickly, mastering both strategy and instinct while remaining flexible in the face of unforeseen circumstances.

Her combination of innate talent, adaptability, and the capacity for growth makes her an fitting symbol for a language model designed to acquire knowledge, evolve over time, and connect information across domains.

<p align="center">
  <img src="https://github.com/AnthonyP57/Radovid---a-LLM-made-on-a-budget/blob/master/img/Ciri.webp?raw=true" width="220"/>
</p>

<div align='center'>
  <em>Fig.2 Ciri Gwent card by Anna Podedworna</em>
</div>
</br>

## What is a LLM
On a high level: imagine a toddler with an huge amount of knowledge but still possessing a toddler-like way of reasoning and understanding.

On a lower level: an LLM is a neural network trained on so-called big data to recognize patterns, generate human-like responses, and predict the most likely next word in a given context. While it can process and recall information efficiently, it lacks true understanding, reasoning, or consciousness, relying only on statistical correlations rather than genuine comprehension. the reasoning of LLMs is being impoved in projects (most notably) like DeepSeek, which focus on enhancing the ability to understand context and simulating human-like reasoning.

## Repo organization:
```bash
Cirilla - a LLM made on a budget/
  │
  ├── BERT/                           # overview of BERT
  │   └── RAG/                        # overview of RAG
  │
  ├── cirilla/
  │   ├── Cirilla_model/              # implementation of the Cirilla LLM
  │   ├── Few_shot/                   # Few-shot learning techniques
  │   ├── LLM_pieces/                 # building blocks of LLMs
  │   └── synth_data/                 # creating synthetic data
  │
  ├── cirilla_training/               # proper LLM training with the Cirilla package
  │
  ├── Decoder_only_architecture/      # overview of decoder only transformer architecture
  │   ├── Llama2/                     # implementation of Llama 2 inference loop
  │   └── Mistral/                    # overview of the Mistral 7B architecture and inference tricks
  │
  ├── DPO/                            # overview of Direct Preference Optimization (DPO)
  │
  ├── examples/                       # examples how to use this package
  │
  ├── Few_shot/                       # overview of Few-shot ML techniques
  │
  ├── KAN/                            # overview of Kolmogorov-Arnold Networks (KAN)
  │
  ├── Multimodal/                     # overview of Paligemma (VLM)
  │
  ├── Tiny_recursive_model/           # overview of Tiny recursive model (TRM)
  │
  ├── Training_optimizations/
  │   ├── FlexAttention/              # overview of Pytorch's FlexAttention
  │   ├── HF_kernels/                 # overview of HF's kernel hub
  │   ├── Mamba/                      # overview of Mamba
  │   ├── Multi_Token_Prediction/     # overview of MTP
  │   └── Optimizer_dusion/           # fusing Pytorch optimizer into the backward pass
  │
  └── Transformer_from_scratch/       # transformer implementation
      ├── model.py                    # transformer model
      ├── dataset.py                  # dataset for MLM - masked language modelling
      ├── train.py                    # main transformer training loop
      └── LongNet.py                  # LongNet - crude dilated attention implementation
```

## Getting started
### 1. Installing Cirilla
```bash
uv add Cirilla
#or
pip install Cirilla # that's it
```
### 2. Installing Mamaba (not required, but recommended)
```bash
uv add Cirilla[mamba]
```

In case the is some problem, try:
```bash
uv pip install --no-cache-dir --no-binary :all: --no-build-isolation mamba-ssm[causal-conv1d]
```
and then
```bash
uv add Cirilla[mamba]
```
To verify that everything works you can try running: `./examples/cirilla_hybrid.py`

## Why Cirilla

Cirilla is a project focused on building **simple and optimized transformer models**. The goal is to give you access to all the modern bells and whistles, like Mixture of Experts (MoE) and [FlexAttention](https://pytorch.org/blog/flexattention/), without requiring you to implement or learn about them from scratch.

### Modular building blocks
Cirilla is organized around reusable transformer components. Each module is implemented in a clean and transparent way, making it easy to experiment, swap, or optimize parts of the model.

*Some highlights:*
- **Hybrid Architecture**: Transformer architecture containing Mamba blocks (similar to [IBM Granite 4.0](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)).
- **Multimodal models**: Similar to [PaliGemma](https://arxiv.org/pdf/2407.07726).
- **Tiny Recursive Model (TRM)**: A simpler recursive reasoning approach to Hierarchical Reasoning Model (HRM).
- **Few-shot ML techniques**: like [ProtoNet](https://arxiv.org/pdf/1703.05175), [MAML](https://arxiv.org/pdf/1703.03400), [Setfit](https://arxiv.org/pdf/2209.11055)
- **Attention mechanisms**: sliding window attention with PyTorch FlexAttention, and non-causal “BERT-like” attention.
- **Rotary Positional Embeddings (RoPE)**: lightweight and efficient PyTorch implementation.  
- **Muon optimizer**: optimizer for hidden layers
- **Accelerated Sparse Training**: available with [torchao](https://github.com/pytorch/ao/tree/main/torchao/sparsity/training)
- **From-scratch transformer**: complete implementations including dataset handling, model definition, training loops and checkpointing.  

#### LLM blocks - learn where the magic happens
- You can learn about the RMS norm [here](https://github.com/AnthonyP57/Cirilla---a-LLM-made-on-a-budget/tree/master/Decoder_only_architecture#normalization-and-rms-norm)
- RoPE embeddings [here](https://github.com/AnthonyP57/Cirilla---a-LLM-made-on-a-budget/tree/master/Decoder_only_architecture/Llama2#rope)
- Grouped-Query Attention [here](https://github.com/AnthonyP57/Cirilla---a-LLM-made-on-a-budget/tree/master/Decoder_only_architecture#multi-query-attention---mqa)
- Sliding window attention [here](https://github.com/AnthonyP57/Cirilla---a-LLM-made-on-a-budget/tree/master/Decoder_only_architecture/Mistral#sliding-window-attention)
- Rolling buffer cache [here](https://github.com/AnthonyP57/Cirilla---a-LLM-made-on-a-budget/tree/master/Decoder_only_architecture/Mistral#kv-cache-with-rolling-buffer-cache)
- SwiGLU [here](https://github.com/AnthonyP57/Cirilla---a-LLM-made-on-a-budget/tree/master/Decoder_only_architecture#swiglu)
- Mixture of Experts [here](https://github.com/AnthonyP57/Cirilla---a-LLM-made-on-a-budget/tree/master/Decoder_only_architecture/Mistral#sparse-mixture-of-experts)
- BERT models [here](https://github.com/AnthonyP57/Cirilla---a-LLM-made-on-a-budget/tree/master/BERT)
- dropless-MoE (dMoE) [here](https://arxiv.org/abs/2211.15841)

### Focus on efficiency
- **Optimized kernels** from [HuggingFace kernel hub](https://huggingface.co/models?other=kernel).
- **Alternative attention mechanisms** for handling longer contexts and specialized training setups.
- **Sparse Mixture of Experts** to scale models without an increase in compute cost.
- **Fused optimizers** that reduce memory usage.
- **FlexAttention** for efficient and sparse attention computation.

### Research + Education
Cirilla explains and integrates ideas from notable papers. This makes it an great resource for:
- **Researchers**, who want to test new variations of transformer models quickly.  
- **Practitioners**, who need efficient and flexible code for training on limited hardware.  
- **Students and hobbyists**, who want to learn how modern LLMs are built.  

### HuggingFace integration
Cirilla models can be easily pushed to and pulled from the HuggingFace Hub, making collaboration, sharing, and deployment straightforward.

### Data generation tools
The repository also provides scripts for **synthetic data generation**, including multi-turn dialogues, reasoning datasets, and domain-specific examples. This allows users to create datasets for fine-tuning and evaluation without relying solely on large, external corpora of questionable quality.
