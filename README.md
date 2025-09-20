> [!IMPORTANT]  
> For a much nicer README visit [Cirilla](https://anthonyp57.github.io/Cirilla---a-LLM-made-on-a-budget/)



![](https://github.com/AnthonyP57/Radovid---a-LLM-made-on-a-budget/blob/master/img/ciri_w4_2.png?raw=true)
*Ciri from The Witcher 4 trailer*

# Cirilla
Cirilla is an open source learning project aiming at implmenting various LLMs.
It is focused mainly on showing how to make, train, infer and deploy a LLM from scratch using Pytorch and a budget friendly GPU (RTX 4060Ti 16GiB ~500$).

- [About Cirilla](#who-is-cirilla)

- [Repo organization](#repo-organization)

- [Getting started](#getting-started)

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
  ├── Cirilla_model/                  # implementation of Cirilla LLM
  │   ├── model.py
  │   ...
  │
  ├── Decoder_only_architecture/      # overview of decoder only transformer architecture
  │   └── Llama2/                     # implementation of Llama 2 inference loop
  │   └── Mistral/                    # overview of the Mistral 7B architecture and inference tricks
  │
  ├── LLM_pieces/                     # elements of decoder-only model you can use
  │   ├── SMoE.py                     # Sparse mixture of Experts
  │   ...
  │
  ├── synth_data/
  │   ├── multi_turn_vllm.py          # create multi turn instructions with VLLM
  │   ├── Ollama_create.py            # synthetic data creation with Ollama
  │   ├── reason_gym_synthetic.py     # create synthetic reasoning dataset with reasoning_gym
  │   ├── rm_duplicate_instruct.py    # remove duplicate instructions from Ollama
  │   └── witcher_mr_gather.py        # create multi turn instructions with Witcher
  │
  ├── Training_optimizations/
  │   ├──FlexAttention/               # overview of Pytorch's FlexAttention
  │   └── HF_kernels/                 # overview of HF's kernel hub
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
### 2. building megablocks (not required, but recommended)
#### 2.1. check the Pytorch cuda version
```bash
# check pip packages
uv pip list | grep -E "torch|cupy|cudatoolkit|nvidia" # or just pip list ...

# inside Pytorch info
python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)   # linked cuda runtime (e.g. '12.8')
    print("cuda available:", torch.cuda.is_available())
except Exception as e:
    print("torch not installed or import failed:", e)
PY
```
You should see something like:
```bash
cupy-cuda12x                      13.6.0
...
torchvision                       0.22.0+cu128
torch: 2.7.0+cu128
torch.version.cuda: 12.8 # <- your cuda version
cuda available: True
```
#### 2.2. check the driver version
```bash
# toolkit compiler
which nvcc || echo "nvcc not in PATH"

nvcc --version    # prints CUDA compiler version (toolkit version)
```
You should see something like:
```bash
/usr/local/cuda-12.8/bin/nvcc
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:23:50_PST_2025
Cuda compilation tools, release 12.8, V12.8.93 # <- make sure that the CUDA toolkit version matches that of the Pytorch form step 1. (release 12.8 == 12.8 from step 1, so all is good)
Build cuda_12.8.r12.8/compiler.35583870_0
```
#### 2.3. Install the correct CUDA toolkit
You can see a guide of how to install the correct CUDA toolkit [here](https://www.cherryservers.com/blog/install-cuda-ubuntu)

To verify that everything works you can try running: `./examples/train_bert.py`