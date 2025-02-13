from tokenizers import Tokenizer
from tokenizers.models import WordPiece, WordLevel
from tokenizers.pre_tokenizers import Digits, Whitespace, Sequence
from tokenizers.trainers import WordPieceTrainer, WordLevelTrainer
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch
import os
from datasets import load_dataset

def get_all_sentences(ds):
    for item in ds:
        yield item['translation']['en']

def get_or_build_tokenizer(ds):
    if not os.path.exists('tokenizer.json'):
        # tokenizer = Tokenizer(WordPiece(unk_token='[UNK]'))
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Sequence([Whitespace(), Digits(individual_digits=True)])
        # trainer = WordPieceTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]', '[MASK]'], min_frequency=2)
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]', '[MASK]'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds), trainer=trainer)
        tokenizer.save(str('tokenizer.json'))
    else:
        tokenizer = Tokenizer.from_file(str('tokenizer.json'))

    return tokenizer

def get_dataset(config):
    ds_raw = load_dataset('opus_books', 'en-it', split='train') #en-hu has the most sentences

    tokenizer = get_or_build_tokenizer(ds_raw)

    train_ds_size = int(0.9*len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, val_sd_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = FillBlankDataset(train_ds_raw, tokenizer, config['seq_len'])
    val_ds = FillBlankDataset(val_sd_raw, tokenizer, config['seq_len'])

    max_len=0

    for item in ds_raw:
        ids = tokenizer.encode(item['translation']['en']).ids
        max_len = max(max_len, len(ids))

    print(f"Max length of sentence: {max_len}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer

class FillBlankDataset(Dataset):
    def __init__(self, data, tokenizer, seq_len):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int64)
        self.mask_token = torch.tensor([tokenizer.token_to_id("[MASK]")], dtype=torch.int64)
        self.sos_token = torch.tensor([tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.token_to_id("[EOS]")], dtype=torch.int64)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sentence_text = self.data[idx]['translation']['en']
        masked_text = self._make_blank(sentence_text)

        sentence = self.tokenizer.encode(sentence_text).ids
        masked = self.tokenizer.encode(masked_text).ids

        enc_num_padding_tokens = self.seq_len - len(masked) - 2
        dec_num_padding_tokens = self.seq_len - len(sentence) - 1

        enc_in = torch.concatenate(
            [
                self.sos_token,
                torch.tensor(masked, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        dec_in = torch.concatenate(
            [
                self.sos_token,
                torch.tensor(sentence, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        dec_out = torch.concatenate(
            [
                torch.tensor(sentence, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        return {
            "enc_in": enc_in,
            "dec_in": dec_in,
            "dec_out": dec_out,
            "enc_mask": mask(enc_in, enc_in, self.pad_token, self.mask_token),
            "dec_self_att_mask": mask(dec_in, dec_in, self.pad_token, causal=True),
            "dec_cross_att_mask": mask(dec_in, enc_in, self.pad_token, self.mask_token),
            "masked": masked_text
        }

    def _make_blank(self, text):
        text = text.split(' ')
        l = len(text)
        n_blank = min(np.abs(np.random.normal(0, l//3)).astype(int), 3)

        if n_blank == 0:
            n_blank = 1
        
        n_blank = [np.random.randint(0, max(l-1, 1)) for _ in range(n_blank)]
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
            else:
                f.append(l)

        masked = " ".join(f)

        return masked
        

def mask(sentence1, sentence2, pad_token, mask_token=None, seq_len=320, dec_len=320, causal=False):
    non_pad1 = (sentence1 != pad_token).sum().item()
    non_pad2 = (sentence2 != pad_token).sum().item()

    mask=torch.zeros(dec_len, seq_len, dtype=torch.int64)

    if not causal:
        mask[:non_pad1, :non_pad2] = 1
    else:
        mask[:non_pad1, :non_pad2] = torch.tril(torch.ones(non_pad1, non_pad2), diagonal=0)

    if mask_token is not None:
        mask_idx1 = (sentence1 == mask_token).nonzero(as_tuple=True)[0]
        mask_idx2 = (sentence2 == mask_token).nonzero(as_tuple=True)[0]
        mask[mask_idx1, :] = 0
        mask[:, mask_idx2] = 0

    return mask.unsqueeze(0)