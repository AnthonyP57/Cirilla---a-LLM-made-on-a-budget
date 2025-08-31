import torch.nn as nn
from model import Args
from bert_model import BertArgs
from typing import Union

class InputEmbeddings(nn.Module):
    def __init__(self, args:Union[Args, BertArgs]):
        super().__init__()

        self.embeddings = nn.Embedding(args.vocab_size, args.dim)
    
    def forward(self, x):
        return self.embeddings(x)