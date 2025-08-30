from pathlib import Path
from modules import cache_or_fetch
import torch
import json
import random
from torch.utils.data import IterableDataset, DataLoader
from typing import Union
from tokenizer_modules import CirillaTokenizer

class JSONLDataset(IterableDataset):
    def __init__(self, path:Union[Path, tuple[Path]]='./training_dataset.jsonl',
                 shuffle_path=False,
                 device:torch.device='cuda',
                 tokenizer:CirillaTokenizer=None,
                 max_len:int=32):
        super().__init__()
        self.path = path
        self.shuffle_path = shuffle_path
        self.device = device
        self.tokenizer = tokenizer
        self.max_len = max_len

        if isinstance(self.path, list):
            self.path = tuple(self.path)

        if isinstance(self.path, str):
            self.path = (self.path,)

        for p in self.path:

            if cache_or_fetch('DATA_LEN', p) is None:
                with open(p, 'r', encoding='utf-8') as f:
                    count = sum(1 for _ in f)
                cache_or_fetch('DATA_LEN', p, count)

            if cache_or_fetch('SHUFFLED', p) is None and shuffle_path:
                with open(p, 'r') as f:
                    data = [json.loads(line) for line in f]
                
                random.shuffle(data)
                with open(p, 'w') as f:
                    for d in data:
                        f.write(json.dumps(d) + '\n')

                del data

            cache_or_fetch('SHUFFLED', p, 1)

    def __len__(self):
        return sum(int(cache_or_fetch('DATA_LEN', p)) for p in self.path)
    
    def __iter__(self):
        for p in self.path:
            with open(p, 'r') as f:
                for line in f:
                    line = json.loads(line)
                    if line['data type'] == 'plain text':

                        if self.tokenizer is not None:
                            tokenized_data =  self.tokenizer(line['text'], return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)
                            tokens = tokenized_data['input_ids'].squeeze(0).to(self.device)
                            mask = tokenized_data['attention_mask'].squeeze(0).to(self.device)
                            yield tokens, mask
                        else:
                            yield line['text']

                    elif line['data type'] == 'conv':

                        if self.tokenizer is not None:
                            tokenized_data =  self.tokenizer.apply_chat_template(line['text'], return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)
                            tokens = tokenized_data['input_ids'].squeeze(0).to(self.device)
                            mask = tokenized_data['attention_mask'].squeeze(0).to(self.device)
                            yield tokens, mask
                        else:
                            yield '\n'.join([l['content'] for l in line['text']])


if __name__ == '__main__':
    from tokenizer_modules import CirillaTokenizer
    tokenizer = CirillaTokenizer(hub_url='AnthonyPa57/HF-torch-demo2')
    dl = JSONLDataset(['./example.jsonl', './example.jsonl'], shuffle_path=True, tokenizer=tokenizer)
    print(len(dl))
    dl = DataLoader(dl, batch_size=2)
    for _ in range(2):
        for i in dl:
            print(i)
