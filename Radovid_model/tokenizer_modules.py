from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from typing import Iterator, Union
import os
from pathlib import Path
from typing import Iterator

SPECIAL_TOKENS = {'unk_token':'<unk>', 'pad_token':'<pad>', 'mask_token':'<mask>',
                  'bos_token':'<sos>', 'eos_token':'<eos>'}

class RadovidTokenizer:
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

        return tokenizer


    def pull_from_hub(self, hub_url):
        self.tokenizer = AutoTokenizer.from_pretrained(hub_url)

    def push_to_hub(self, hub_url):
        self.tokenizer.push_to_hub(hub_url)
    
    def decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens, **kwargs)
    
    def encode(self, text, **kwargs):
        return self.tokenizer.encode(text, **kwargs)
    
    def __call__(self, text, **kwargs):
        return self.tokenizer(text, **kwargs)
        
if __name__ == '__main__':
    # tokenizer = RadovidTokenizer(hub_url='AnthonyPa57/HF-torch-demo2')

    # tokenizer.pull_from_hub('AnthonyPa57/HF-torch-demo2')
    # tokenizer.push_to_hub('AnthonyPa57/HF-torch-demo2')
    # tokenizer.pull_from_hub('AnthonyPa57/HF-torch-demo2')

    # print(tokenizer.decode(tokenizer.encode('hello world')))
    # print(tokenizer.encode('hello world'))

    from dataloader import JSONLDataset
    # from torch.utils.data import DataLoader
    dl = JSONLDataset('training_datasets/mid_training/witcher_instruct.jsonl', shuffle_path=True)
    # dl = DataLoader(dl, batch_size=2)

    tokenizer = RadovidTokenizer()
    tokenizer.train(dl, special_tokens=SPECIAL_TOKENS, min_frequency=2)

    tokenizer.push_to_hub('AnthonyPa57/HF-torch-demo2')

    print(tokenizer.decode(tokenizer.encode('hello world')))
    print(tokenizer.encode('<sos> What is the capital of France?'))
    print(tokenizer.decode(tokenizer.encode('What is the capital of France?')))

