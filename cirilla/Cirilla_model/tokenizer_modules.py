from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from typing import Iterator, Union
import os
from pathlib import Path
from typing import Iterator

SPECIAL_TOKENS = {'unk_token':'<unk>', 'pad_token':'<pad>', 'mask_token':'<mask>',
                'bos_token':'<sos>', 'eos_token':'<eos>', 'system_token':'<|system|>',
                'assistant_token':'<|assistant|>', 'user_token':'<|user|>', 'class_token':'<cls>'}

class CirillaTokenizer:
    def __init__(self, path:Path=None, hub_url=None):
        self.path = path
        self.hub_url = hub_url

        if path is not None:
            if os.path.exists(path):
                self.tokenizer = self._turn_to_fast(path)
        
        elif hub_url is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(hub_url)

    def train(self, dataset: Union[Iterator[str], Iterator[Iterator[str]]], special_tokens: dict[str, str]=SPECIAL_TOKENS, save_to_path:Path='./tokenizer.json', **kwargs) -> PreTrainedTokenizerFast:
        spm = SentencePieceBPETokenizer()
        spm.train_from_iterator(dataset, special_tokens=list(special_tokens.values()), **kwargs)
        spm.save(str(save_to_path))
        self.tokenizer = self._turn_to_fast(save_to_path)
        return self.tokenizer

    @staticmethod
    def _turn_to_fast(path: Path, special_tokens: dict[str, str] = SPECIAL_TOKENS) -> PreTrainedTokenizerFast:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(path), **special_tokens)

        tok_to_add = []
        for s in special_tokens.values():
            if tokenizer.convert_tokens_to_ids(s) == tokenizer.unk_token_id:
                tok_to_add.append(s)
        if tok_to_add:
            tokenizer.add_tokens(tok_to_add)
            tokenizer.add_special_tokens({k: v for k, v in special_tokens.items() if v in tok_to_add})


        tokenizer.chat_template = (
"""{% for message in messages %}
{% if message['role'] == 'system' %}{{'<|system|>' + message['content']}}
{% elif message['role'] == 'user' %}{{'<|user|>' + message['content']}}
{% elif message['role'] == 'assistant' %}{{'<|assistant|>' + message['content']}}
{% endif %}
{% endfor %}{{ '<|assistant|>' if add_generation_prompt else '' }}"""
                )

        return tokenizer


    def pull_from_hub(self, hub_url):
        self.tokenizer = AutoTokenizer.from_pretrained(hub_url)

    def push_to_hub(self, hub_url):
        self.tokenizer.push_to_hub(hub_url)
    
    def decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens, **kwargs)
    
    def encode(self, text, **kwargs):
        return self.tokenizer.encode(text, **kwargs)
    
    def apply_chat_template(self, texts, **kwargs):
        return self.tokenizer.apply_chat_template(texts, **kwargs)
    
    def __call__(self, text, **kwargs):
        return self.tokenizer(text, **kwargs)
    
    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)
