import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # n head for query 
    n_kv_heads: Optional[int] = None # n heads for k and v
    vocab_size: int = -1 # will be set when we make the tokenizer
    multiple_of:int = 256 # hidden of feed forward
    ffn_dim_multiplier: Optional[int] = None # use to compare the number of params between GQA and vanilla attention
    norm_eps: float = 1e-5

    max_batch_size:int = 32
    max_seq_len:int = 2048

    device:str = None

def precompute_theta_pos_frequencies(head_dim:int, seq_len:int, device:str, theta:float = 10000.0): # 10k is form the paper on RoPE
    assert head_dim % 2 == 0, 'embedding cannot be applied to odd number of head (dimentions)'

    # build theta parameters
    # shape: (head_dim / 2)

    # theta_i = 10k ^ (-2*(i-1)/dim) for i = [1,2,3, ..., dim/2]

    theta_numerator = torch.arange(0, head_dim, 2).float()
    # shape: (head_dim/2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # construct positions
    # shape: (seq_len)
    m = torch.arange(seq_len, device=device)

    # now we multiply m with all the thetas - for that we use outer product - each element with each element, all posiible combinations
    # shape: (seq_len), (head_dim/2) -> (seq_len, head_dim/2)
    freqs = torch.outer(m, theta).float()

    # we want to comput the numbers into the complex form to get the rotation: c = R*exp(i*m*theta), where R=1
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, 'Vocab size must be set'

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.appen(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_fequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device = self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int): #inference only
        # (B, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "only one token at a time for inference" # for kv cache

        # (B, seq_len) -> (B, seq_len, dim)
        h = self.tok_embeddings(tokens)

        # retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos+ seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        # apply the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output
