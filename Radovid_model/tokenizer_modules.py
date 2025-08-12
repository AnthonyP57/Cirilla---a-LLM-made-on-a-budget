from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from typing import Iterator
import os
from pathlib import Path
from typing import Iterator

SPECIAL_TOKENS = {'unk_token':'<unk>', 'pad_token':'[PAD]', 'mask_token':'[MASK]',
                  'bos_token':'[SOS]', 'eos_token':'[EOS]'}

class RadovidTokenizer:
    def __init__(self, path:Path='./tokenizer.json', hub_url=None):
        self.path = path
        self.hub_url = hub_url

        if os.path.exists(path):
            self.tokenizer = self._turn_to_fast(path)
        
        elif hub_url is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(hub_url)

    def train(self, dataset: Iterator, special_tokens: dict[str, str] = SPECIAL_TOKENS, **kwargs) -> PreTrainedTokenizerFast:
        spm = SentencePieceBPETokenizer()
        spm.train_from_iterator(dataset, special_tokens=list(special_tokens.values()), **kwargs)
        spm.save(str(self.path))
        self.tokenizer = self._turn_to_fast(self.path)
        return self.tokenizer

    @staticmethod
    def _turn_to_fast(path: Path, special_tokens: dict[str, str] = SPECIAL_TOKENS) -> PreTrainedTokenizerFast:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(path))

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
    
    def decode(self, tokens):
        return self.tokenizer.decode(tokens)
    
    def encode(self, text):
        return self.tokenizer.encode(text)
        
if __name__ == '__main__':
    tokenizer = RadovidTokenizer()

    tokenizer.pull_from_hub('AnthonyPa57/HF-torch-demo')

    print(tokenizer.decode(tokenizer.encode('hello world')))
    print(tokenizer.encode('hello world'))

    from dataloader import JSONLDataset
    from torch.utils.data import DataLoader
    dl = JSONLDataset('./example.jsonl', shuffle_path=True)
    dl = DataLoader(dl, batch_size=2)

    tokenizer.train(dl, special_tokens=SPECIAL_TOKENS, min_frequency=1)

    print(tokenizer.decode(tokenizer.encode('hello world')))
    print(tokenizer.encode('[SOS] What is the capital of France?'))
    print(tokenizer.decode(tokenizer.encode('What is the capital of France?')))

