from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Digits, Whitespace, Sequence
from tokenizers.trainers import WordPieceTrainer
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import torch

# if not Path.exists(tokenizer_path):
#     tokenizer = Tokenizer(WordPiece(unk_token='[UNK]'))
#     tokenizer.pre_tokenizer = Sequence([Whitespace(), Digits(individual_digits=True)])
#     trainer = WordPieceTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
#     tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
#     tokenizer.save(str(tokenizer_path))
# else:
#     tokenizer = Tokenizer.from_file(str(tokenizer_path))

def get_all_sentences(paths):
    for path in paths:
        with open(path) as f:
            for line in f:
                yield line

class FillBlankDataset(Dataset):
    def __init__(self, data, tokenizer, seq_len):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int16)
        self.mask_token = torch.tensor([tokenizer.token_to_id("[MASK]")], dtype=torch.int16)
        self.sos_token = torch.tensor([tokenizer.token_to_id("[SOS]")], dtype=torch.int16)
        self.eos_token = torch.tensor([tokenizer.token_to_id("[EOS]")], dtype=torch.int16)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sentence = self.data[idx]
        masked = self._make_blank(sentence)

        sentence = self.tokenizer.encode(sentence).ids
        masked = self.tokenizer.encode(masked).ids

        enc_num_padding_tokens = self.seq_len - len(sentence) - 2
        dec_num_padding_tokens = self.seq_len - len(masked) - 1

        enc_in = torch.concatenate(
            [
                self.sos_token,
                torch.tensor(masked, dtype=torch.int16),
                self.pad_token * enc_num_padding_tokens,
                self.eos_token
            ]
        )

        dec_in = torch.concatenate(
            [
                self.sos_token,
                torch.tensor(sentence, dtype=torch.int16),
                self.pad_token * dec_num_padding_tokens
            ]
        )

        dec_out = torch.concatenate(
            [
                torch.tensor(sentence, dtype=torch.int16),
                self.eos_token,
                self.pad_token * dec_num_padding_tokens
            ]
        )

        return {
            "enc_in": enc_in,
            "dec_in": dec_in,
            "dec_out": dec_out,
            "enc_mask": mask(enc_in, enc_in, self.pad_token),
            "dec_self_att_mask": mask(dec_in, dec_in, self.pad_token, causal=True),
            "dec_cross_att_mask": mask(dec_in, enc_in, self.pad_token)
        }

    def _make_blank(self, text):
        text = text.split(' ')
        l = len(text)
        n_blank = np.abs(np.random.normal(0, l/3)).astype(int)

        if n_blank == 0:
            n_blank = 1
        if n_blank > l/3:
            n_blank = l//3
        
        n_blank = [np.random.randint(0, l-1) for _ in range(n_blank)]
        n_blank.sort()

        masked=[]
        for i, l in enumerate(text):
            if i not in n_blank:
                masked.append(l)
            else:
                masked.append("[MASK]")

        f = [masked[0]]

        for l in masked[1:]:
            if l == "[MASK]":
                if l != f[-1]:
                    f.append(l)

        masked = " ".join(f)
        

def mask(sentence1, sentence2, pad_token, causal=False):
    non_pad1 = (sentence1 != pad_token).sum().item()
    non_pad2 = (sentence2 != pad_token).sum().item()

    seq_len = sentence1.size(0)
    mask=torch.zeros(seq_len, seq_len, dtype=torch.int16)

    mask[:non_pad1, :non_pad2] = 1

    if causal:
        mask = nn.triu(mask, diagonal=1)

    return mask