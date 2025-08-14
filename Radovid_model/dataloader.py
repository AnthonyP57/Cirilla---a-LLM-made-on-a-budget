from pathlib import Path
from modules import cache_or_fetch
import json
import random
from torch.utils.data import IterableDataset, DataLoader

class JSONLDataset(IterableDataset):
    def __init__(self, path:Path='./training_dataset.jsonl', shuffle_path=False):
        super().__init__()
        self.path = path
        self.shuffle_path = shuffle_path

        if cache_or_fetch('DATA_LEN', path) is None:
            with open(self.path, 'r', encoding='utf-8') as f:
                count = sum(1 for _ in f)
            cache_or_fetch('DATA_LEN', self.path, count)

        if cache_or_fetch('SHUFFLED', path) is None and shuffle_path:
            with open(path, 'r') as f:
                data = [json.loads(line) for line in f]
            
            random.shuffle(data)
            with open(path, 'w') as f:
                for d in data:
                    f.write(json.dumps(d) + '\n')

            del data

            cache_or_fetch('SHUFFLED', path, 1)

    def __len__(self):
        return int(cache_or_fetch('DATA_LEN', self.path))
    
    def __iter__(self):
        with open(self.path, 'r') as f:
            for line in f:
                line = json.loads(line)
                data = line.get('data', None)
                if data is not None:
                    yield data['text']
                else:
                    yield line['question'] + ' ' + line['answer']


if __name__ == '__main__':
    dl = JSONLDataset('./example.jsonl', shuffle_path=True)
    dl = DataLoader(dl, batch_size=2)
    for _ in range(2):
        for i in dl:
            print(i)
