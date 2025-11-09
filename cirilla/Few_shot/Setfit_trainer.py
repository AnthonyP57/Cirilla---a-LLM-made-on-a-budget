from torch.utils.data import Dataset
from ..Cirilla_model.dataloader import GenericDataset
import json
import torch
import random

class SetfitDataset(Dataset, GenericDataset):
    def __init__(self, tokenizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = tokenizer

        self._get_data()

    def __len__(self):
        return int((len(self.texts) * (len(self.texts)-1))/2)

    def _get_data(self):

        self.texts = []
        self.labels = []

        for p in self.path:
            with open(p, 'r') as f:

                for line in f:
                    line = json.loads(line)
                    assert line['data type'] == 'bert', 'Setfit supports only BERT data'

                    text = line['text']

                    if self.prefix_tokens is not None:
                        text = "".join(self.prefix_tokens) + text

                    if self.suffix_tokens is not None:
                        text += "".join(self.suffix_tokens)

                    self.texts.append(text)
                    self.labels.append(line['label'])

        slabels = set(self.labels)
        assert sorted(list(slabels)) == list(range(len(slabels))), "Labels must be consecutive integers starting from 0"
        self.labels = torch.tensor(self.labels, dtype=torch.int64)
        self.n_classes = len(slabels)

        for s in slabels:
            indices = torch.where(self.labels == s)[0]
            if len(indices) < 2:
                print(f"Label {s} has only {len(indices)} samples. Needs at least 2")
                continue

            setattr(self, f'indices_{int(s)}', indices.tolist())

    def __getitem__(self, idx):
                
        anchor_idx = torch.randint(0, self.n_classes, (1,))
        negative_idx = random.choice([i for i in range(self.n_classes) if i != anchor_idx])
        
        anchor_data = getattr(self, f'indices_{anchor_idx.item()}').copy()
        negative_data = getattr(self, f'indices_{negative_idx}').copy()

        _anchor_data = random.choice(anchor_data)
        anchor_data.remove(_anchor_data)
        positive_data = random.choice(anchor_data)
        n = random.choice(negative_data)

        anchor_tokens = self.tokenizer(self.texts[_anchor_data], return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)
        positive_tokens = self.tokenizer(self.texts[positive_data], return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)
        negative_tokens = self.tokenizer(self.texts[n], return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)

        anchor_encoded_id = anchor_tokens['input_ids'].to(self.device).squeeze(0)
        anchor_encoded_mask = anchor_tokens['attention_mask'].to(self.device).squeeze(0)

        positive_encoded_id = positive_tokens['input_ids'].to(self.device).squeeze(0)
        positive_encoded_mask = positive_tokens['attention_mask'].to(self.device).squeeze(0)

        negative_encoded_id =  negative_tokens['input_ids'].to(self.device).squeeze(0)
        negative_encoded_mask = negative_tokens['attention_mask'].to(self.device).squeeze(0)


        return (
            #'anchor': 
            (anchor_encoded_id, anchor_encoded_mask),
            #'positive':
            (positive_encoded_id, positive_encoded_mask),
            #'negative':
            (negative_encoded_id, negative_encoded_mask)
                )
    
def setfit_training_step(self, data):
    torch.compiler.cudagraph_mark_step_begin()

    anchor = data[0]
    positive = data[1]
    negative = data[2]

    anchor_out = self.model(*anchor)
    positive_out = self.model(*positive)
    negative_out = self.model(*negative)

    loss = self.criterion(anchor_out, positive_out, negative_out) # criterion has to be triplet loss
    loss_item = loss.item()
    loss.backward()
    return loss_item

@torch.inference_mode()
def setfit_inference_step(self, data):
    anchor = data[0]
    positive = data[1]
    negative = data[2]

    anchor_out = self.model(*anchor)
    positive_out = self.model(*positive)
    negative_out = self.model(*negative)

    loss = self.criterion(anchor_out, positive_out, negative_out) # criterion has to be triplet loss
    return loss.item()