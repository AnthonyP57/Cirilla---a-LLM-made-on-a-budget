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

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device:str):
    # (b, seq_len, h, head_dim) -> (b, seq_len, h, head_dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (seq_len, head_dim/2) -> (1, seq_len, 2, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex
    # (b, seq_len, head/2) -> (b, seq_len, head/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (b, seq_len, head_dim/2, 2) -> (b, seq_len, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x:torch.Tensor):
        # (b, seq_len, dim)
        return x * torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x:torch.Tensor):
        # (dim) * (b, seq_len, dim) -> (b, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # norm before self attention
        self.attention_norm = RMSNorm(args.dim, eps=args.eps)

        # norm before feed forward
        self.ffn_norm = RMSNorm(args.dim, eps=args.eps)

    def forward(self, x:torch.Tensor, start_pos:int, freqs_complex: torch.Tensor):
        # (b, seq_len, dim) + (b, seq_len, dim) -> (b, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

def repeat_kv(x:torch.Tensor, n_rep:int):
    batch, seq, n_kv, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (
            x[:, :, :, None, :].expand(batch, seq, n_kv, n_rep, head_dim)
            .reshape(batch, seq, n_kv*n_rep, head_dim)
        )

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        # how many times the keys and values should be repeated
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        self.cache_v = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)

    def forward(self, x: torch.Tensor, start_pos:int, freqs_complex:torch.Tensor):
        batch_size, seq_len, _ = x.shape # (b, seq_len, dim) for inference (b, 1, dim)
        xq = self.wq(x) # (b, 1, h_q, head_dim)
        xk = self.wk(x)
        
        xv = self.wv(x)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # doesn't change the size of the vectors
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv

        # get the cached K and V
        # (b, seq_len_kv, h_kv, head_dim)
        keys = self.cache_k[:batch_size, 0:start_pos+seq_len]
        values = self.cache_v[:batch_size, 0:start_pos+seq_len]

        # repeat the head of the k and v to reach the number of heads of the queries
        keys = repeat_kv(keys, self.n_rep) # this is not optimal way to do this
        values = repeat_kv(values, self.n_rep)

        # (b, 1, h_q, head_dim) -> (b, h_q, 1, head_dim)
        xq = xq.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim) # here we do not need to "make it causal" as we only treat the last token (last row in vanilla attention)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # (b, h_q, 1, head_dim)
        out = torch.matmul(scores, values)
        out = out.transpose(1,2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(out) #(b, 1, dim)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        # what happened above is e.g hidden is 7 but multiple_of is 5 so we want to get a multiple of 5 so 10
        # e.g. (7+4) // 5 = 2
        # 2 * 5 = 10

        self.w1= nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x:torch.Tensor):
        swish = F.silu(self.w1(x))
        x_v = self.w1(x)
        x = swish * x_v
        x = self.w2(x)
        return x

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

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device = self.args.device)

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
