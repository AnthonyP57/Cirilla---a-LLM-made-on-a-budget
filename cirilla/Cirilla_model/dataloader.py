from pathlib import Path
from .modules import cache_or_fetch
import torch
import json
import random
from torch.utils.data import IterableDataset
from typing import Union
from .tokenizer_modules import CirillaTokenizer

KEYBOARD_NEIGHBORS = {
    'a': 'qs', 'b': 'vn', 'c': 'xv', 'd': 'sf', 'e': 'wr', 'f': 'dg', 'g': 'fh',
    'h': 'gj', 'i': 'uo', 'j': 'hk', 'k': 'jl', 'l': 'k', 'm': 'n', 'n': 'bm',
    'o': 'ip', 'p': 'o', 'q': 'wa', 'r': 'et', 's': 'ad', 't': 'ry', 'u': 'iy',
    'v': 'cb', 'w': 'qe', 'x': 'zc', 'y': 'tu', 'z': 'x'
    }

class GenericDataset:
    def __init__(self, path:Union[Path, tuple[Path]]='./training_dataset.jsonl',
                shuffle_path=False,
                device:torch.device='cuda',
                tokenizer:CirillaTokenizer=None,
                max_len:int=1024,
                pad_token:str='<pad>',
                eos_token:str='<eos>',
                sos_token:str='<sos>',
                user_token:str='<|user|>',
                suffix_tokens:list[str]=None,
                prefix_tokens:list[str]=None,
                random_spelling_mistake_prob:float=0.,
                random_missing_char_prob:float=0.
                ):
        
        self.path = path
        self.shuffle_path = shuffle_path
        self.device = device
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.suffix_tokens = suffix_tokens
        self.prefix_tokens = prefix_tokens
        self.random_spelling_mistake_prob = random_spelling_mistake_prob
        self.random_missing_char_prob = random_missing_char_prob

        if self.tokenizer is not None:
            self.sos_token = sos_token
            self.sos_token_id = torch.tensor([tokenizer.convert_tokens_to_ids(sos_token)])
            self.user_token_id = torch.tensor([tokenizer.convert_tokens_to_ids(user_token)])
            self.pad_token_id = torch.tensor([tokenizer.convert_tokens_to_ids(pad_token)])
            self.eos_token_id = torch.tensor([tokenizer.convert_tokens_to_ids(eos_token)])

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
        
        self.path_signature = '-'.join(self.path)
        cache_or_fetch('DATA_LEN', self.path_signature, sum(cache_or_fetch('DATA_LEN', p) for p in self.path))

    def __len__(self):
        return cache_or_fetch('DATA_LEN', self.path_signature)
    
    def _apply_random_spelling_mistake(self, text):
        new_text = []
        for letter in text:
            if not letter.isupper():

                if random.random() < self.random_missing_char_prob: # skip letter
                    continue

                elif random.random() < self.random_spelling_mistake_prob:
                    letter_lower = letter.lower()
                    if letter_lower in KEYBOARD_NEIGHBORS:
                        new_letter = random.choice(KEYBOARD_NEIGHBORS[letter_lower])
                    else:
                        new_letter = letter
                    new_text.append(new_letter)
                else:
                    new_text.append(letter)
            else:
                new_text.append(letter)
        return ''.join(new_text)

class JSONLDataset(IterableDataset, GenericDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __iter__(self):

        for p in self.path:
            with open(p, 'r') as f:

                for line in f:
                    line = json.loads(line)

                    match line['data type']:

                        case 'plain text':

                            if self.tokenizer is not None:

                                if self.random_spelling_mistake_prob > 0. or self.random_missing_char_prob > 0.:

                                    v = self.tokenizer(self.sos_token + line['text'], return_tensors='pt', padding='do_not_pad',
                                                                    truncation=True, max_length=self.max_len+1)
                                    
                                    out_tokens = v['input_ids'].squeeze(0)
                                    token_shape = out_tokens.shape[0]

                                    if token_shape <= self.max_len:
                                        out_tokens = torch.concat([out_tokens, self.eos_token_id] + \
                                                        [self.pad_token_id] * (self.max_len - token_shape), dim=0)
                                    
                                    out_tokens = out_tokens.to(self.device)
                                    
                                    line['text'] = self._apply_random_spelling_mistake(line['text'])
                                    
                                    tokenized_data = self.tokenizer(self.sos_token + line['text'], return_tensors='pt', padding='do_not_pad',
                                                                    truncation=True, max_length=self.max_len+1)
                                    
                                    in_tokens = tokenized_data['input_ids'].squeeze(0)
                                    token_shape = in_tokens.shape[0]

                                    if token_shape <= self.max_len:
                                        in_tokens = torch.concat([in_tokens, self.eos_token_id] + \
                                                        [self.pad_token_id] * (self.max_len - token_shape), dim=0)
                                    
                                    in_tokens = in_tokens.to(self.device)

                                    yield in_tokens[:-1], out_tokens[1:]

                                else:
                                        
                                    tokenized_data =  self.tokenizer(self.sos_token + line['text'], return_tensors='pt', padding='do_not_pad',
                                                                    truncation=True, max_length=self.max_len+1)
                                    
                                    tokens = tokenized_data['input_ids'].squeeze(0)
                                    token_shape = tokens.shape[0]

                                    if token_shape <= self.max_len:
                                        tokens = torch.concat([tokens, self.eos_token_id] + \
                                                        [self.pad_token_id] * (self.max_len - token_shape), dim=0)
                                    
                                    tokens = tokens.to(self.device)
                                    yield tokens[:-1], tokens[1:]

                            else:
                                yield line['text']

                        case 'conv':

                            if self.tokenizer is not None:
                                tokens =  self.tokenizer.apply_chat_template(line['text'], return_tensors='pt', padding='do_not_pad',
                                                                                    truncation=True, max_length=self.max_len+1, add_generation_prompt=False)
                                tokens = tokens.squeeze(0)
                                tokens_shape = tokens.shape[0]

                                if tokens_shape <= self.max_len:
                                    tokens = torch.concat(
                                        [tokens, self.user_token_id] + \
                                        [self.pad_token_id] * (self.max_len - tokens_shape))
                                    
                                tokens = tokens.to(self.device)
                                yield tokens[:-1], tokens[1:]
                            else:
                                yield '\n'.join([l['content'] for l in line['text']])

                        case 'bert':

                            if self.tokenizer is not None:
                                text = line['text']
                                
                                if self.prefix_tokens is not None:
                                    text = "".join(self.prefix_tokens) + text

                                if self.suffix_tokens is not None:
                                    text += "".join(self.suffix_tokens)

                                tokenized_data =  self.tokenizer(text, return_tensors='pt', padding='max_length',
                                                                truncation=True, max_length=self.max_len)
                                
                                tokens = tokenized_data['input_ids'].squeeze(0).to(self.device)
                                mask = tokenized_data['attention_mask'].squeeze(0).to(self.device)
                                yield tokens, mask, torch.tensor(line['label'], dtype=torch.int64 if not isinstance(line['label'], list) else torch.bfloat16, device=self.device)
                            else:
                                yield line['text']

                        case _:
                            raise NotImplementedError(f"Data type {line['data type']} is not supported, use one of: ['plain text', 'conv', 'bert']")

class JSONDynamicDatset(IterableDataset, GenericDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __iter__(self):

        for p in self.path:
            with open(p, 'r') as f:

                for line in f:
                    line = json.loads(line)

                    match line['data type']:

                        case 'plain text':

                            if self.tokenizer is not None:

                                if self.random_spelling_mistake_prob > 0. or self.random_missing_char_prob > 0.:

                                    v = self.tokenizer(line['text'], return_tensors='pt', padding='do_not_pad',
                                                                    truncation=True, max_length=self.max_len)
                                    
                                    out_tokens = v['input_ids'].squeeze(0)
                                    token_shape = out_tokens.shape[0]

                                    if token_shape < self.max_len:
                                        out_tokens = torch.concat([out_tokens, self.eos_token_id], dim=0)
                                    
                                    out_tokens = out_tokens.to(self.device)
                                    
                                    line['text'] = self._apply_random_spelling_mistake(line['text'])
                                    
                                    tokenized_data = self.tokenizer(line['text'], return_tensors='pt', padding='do_not_pad',
                                                                    truncation=True, max_length=self.max_len-1)
                                    
                                    in_tokens = tokenized_data['input_ids'].squeeze(0)

                                    in_tokens = torch.concat([self.sos_token_id, in_tokens], dim=0)
                                    
                                    in_tokens = in_tokens.to(self.device)

                                    yield in_tokens, out_tokens

                                else:

                                    tokenized_data =  self.tokenizer(line['text'], return_tensors='pt', padding='do_not_pad',
                                                                    truncation=True, max_length=self.max_len-1)
                                    
                                    tokens = tokenized_data['input_ids'].squeeze(0)
                                    tokens = torch.concat([self.sos_token_id, tokens], dim=0)
                                    
                                    token_shape = tokens.shape[0]

                                    if token_shape <= self.max_len:
                                        tokens = torch.concat([tokens, self.eos_token_id], dim=0)
                                    
                                    tokens = tokens.to(self.device)
                                    yield tokens[:-1], tokens[1:] # (in tokens, out tokens)

                            else:
                                yield line['text']

                        case 'conv':

                            if self.tokenizer is not None:
                                tokens =  self.tokenizer.apply_chat_template(line['text'], return_tensors='pt', padding='do_not_pad',
                                                                                    truncation=True, max_length=self.max_len, add_generation_prompt=False)
                                tokens = tokens.squeeze(0)
                                tokens_shape = tokens.shape[0]

                                if tokens_shape <= self.max_len:
                                    tokens = torch.concat(
                                        [tokens, self.user_token_id])

                                tokens = tokens.to(self.device)
                                
                                yield tokens[:-1], tokens[1:]
                            else:
                                yield '\n'.join([l['content'] for l in line['text']])

                        case 'bert':

                            if self.tokenizer is not None:
                                text = line['text']
                                
                                if self.prefix_tokens is not None:
                                    text = "".join(self.prefix_tokens) + text

                                if self.suffix_tokens is not None:
                                    text += "".join(self.suffix_tokens)

                                tokenized_data =  self.tokenizer(text, return_tensors='pt', padding='do_not_pad',
                                                                truncation=True, max_length=self.max_len)
                                
                                tokens = tokenized_data['input_ids'].squeeze(0).to(self.device)
                                mask = tokenized_data['attention_mask'].squeeze(0).to(self.device)
                                yield tokens, mask, torch.tensor(line['label'], dtype=torch.int64 if not isinstance(line['label'], list) else torch.bfloat16, device=self.device)
                            else:
                                yield line['text']

                        case _:
                            raise NotImplementedError(f"Data type {line['data type']} is not supported, use one of: ['plain text', 'conv', 'bert']")
                        
class DynamicCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):

        if isinstance(batch[0], tuple):
            return tuple(torch.nn.utils.rnn.pad_sequence(b, batch_first=True, padding_value=self.pad_token_id) for b in batch)
        
        elif isinstance(batch[0], torch.Tensor):
            return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=self.pad_token_id)
        
        else:
            raise NotImplementedError(f"Data type {type(batch[0])} is not supported")
