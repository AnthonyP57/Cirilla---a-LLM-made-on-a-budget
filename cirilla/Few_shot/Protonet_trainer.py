from torch.utils.data import Dataset
from ..Cirilla_model.dataloader import GenericDataset
import json
import torch
import torch.nn.functional as F

class ProtonetDataset(Dataset, GenericDataset):
    def __init__(self, n_support=None, n_query=None, tokenizer=None, **kwargs):
        super().__init__(**kwargs)

        self.n_support = n_support
        self.n_query = n_query
        self.tokenizer = tokenizer

        self._get_data()

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
        self.label_to_indices = {}

        for s in slabels:
            indices = torch.where(self.labels == s)[0]
            if len(indices) < 2:
                print(f"Label {s} has only {len(indices)} samples. Needs at least 2. It will be skipped.")
                continue

            self.label_to_indices[s] = indices

        if self.n_support is None or self.n_query is None:
            min_examples = min([len(indices) for indices in self.label_to_indices.values()])
            self.n_support = self.n_support if self.n_support is not None else min_examples // 2
            self.n_query = self.n_query if self.n_query is not None else min_examples - self.n_support

        for label, indices in self.label_to_indices.items(): # sanity check
            if len(indices) < self.n_support + self.n_query:
                raise ValueError(f"Label {label} has only {len(indices)} samples. Needs at least {self.n_support + self.n_query}")
                
    def __len__(self):
        return len(self.texts) * self.n_classes
    
    def __getitem__(self, idx):
        
        support_inputs_id = []
        support_inputs_mask = []
        query_inputs_id = []
        query_inputs_mask = []
        support_labels = []
        query_labels = []

        for klass, indices in self.label_to_indices.items():

            # sample without replacement
            perm = torch.randperm(indices.size(0))
            indices = indices[perm][: self.n_support + self.n_query]

            support_indices = indices[:self.n_support]
            query_indices = indices[self.n_support:self.n_support + self.n_query]

            support_texts = [self.texts[i] for i in support_indices]
            query_texts = [self.texts[i] for i in query_indices]

            for text in support_texts:
                tokens_mask = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)
                
                support_inputs_id.append(tokens_mask['input_ids'].to(self.device))
                support_inputs_mask.append(tokens_mask['attention_mask'].to(self.device))
                support_labels.append(klass)

            for text in query_texts:
                tokens_mask = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)
                
                query_inputs_id.append(tokens_mask['input_ids'].to(self.device))
                query_inputs_mask.append(tokens_mask['attention_mask'].to(self.device))
                query_labels.append(klass)

        return {
                'support_inputs_id': torch.concat(support_inputs_id),
                'support_inputs_mask': torch.concat(support_inputs_mask),
                'support_labels': torch.tensor(support_labels, dtype=torch.int64, device=self.device),
                'query_inputs_id': torch.concat(query_inputs_id),
                'query_inputs_mask': torch.concat(query_inputs_mask),
                'query_labels': torch.tensor(query_labels, dtype=torch.int64, device=self.device),
        }

def protonet_training_step(self, data):
    torch.compiler.cudagraph_mark_step_begin()
    
    support_inputs_ids = data['support_inputs_id']
    b, sexamples, seq = support_inputs_ids.shape

    support_inputs_ids = support_inputs_ids.view(b*sexamples, seq)

    support_input_masks = data['support_inputs_mask']
    support_input_masks = support_input_masks.view(b*sexamples, seq)

    support_labels = data['support_labels']

    query_inputs_ids = data['query_inputs_id']
    b, qexamples, seq = query_inputs_ids.shape
    query_inputs_ids = query_inputs_ids.view(b*qexamples, seq)

    query_input_masks = data['query_inputs_mask']
    query_input_masks = query_input_masks.view(b*qexamples, seq)

    query_labels = data['query_labels']

    support_emb = self.model(support_inputs_ids, support_input_masks).view(b, sexamples, -1)
    query_emb = self.model(query_inputs_ids, query_input_masks).view(b, qexamples, -1)

    class_prototypes = {}
    for class_id in torch.unique(support_labels):
        a = []
        for b_ in range(b):
            class_mask = support_labels[b_] == class_id
            class_emb = support_emb[b_][class_mask]
            a.append(class_emb.mean(dim=0).unsqueeze(0))

        class_prototypes[class_id.item()] = torch.concat(a)

    distances = torch.empty(b, qexamples, len(class_prototypes), device=getattr(self.model.args, 'device', 'cuda'), dtype=query_emb.dtype)

    for i, class_prototype in enumerate(class_prototypes.values()):
        distances[:, :, i] = - torch.norm(query_emb - class_prototype.unsqueeze(1), dim=2)

    loss = F.cross_entropy(distances.view(b*qexamples, -1), query_labels.view(b*qexamples))
    loss_item = loss.item()
    loss.backward()
    import time
    return loss_item

@torch.inference_mode()
def protonet_inference_step(self, data):
    support_inputs_ids = data['support_inputs_id']
    b, sexamples, seq = support_inputs_ids.shape

    support_inputs_ids = support_inputs_ids.view(b*sexamples, seq)

    support_input_masks = data['support_inputs_mask']
    support_input_masks = support_input_masks.view(b*sexamples, seq)

    support_labels = data['support_labels']

    query_inputs_ids = data['query_inputs_id']
    b, qexamples, seq = query_inputs_ids.shape
    query_inputs_ids = query_inputs_ids.view(b*qexamples, seq)

    query_input_masks = data['query_inputs_mask']
    query_input_masks = query_input_masks.view(b*qexamples, seq)

    query_labels = data['query_labels']

    support_emb = self.model(support_inputs_ids, support_input_masks).view(b, sexamples, -1)
    query_emb = self.model(query_inputs_ids, query_input_masks).view(b, qexamples, -1)

    class_prototypes = {}
    for class_id in torch.unique(support_labels):
        a = []
        for b_ in range(b):
            class_mask = support_labels[b_] == class_id
            class_emb = support_emb[b_][class_mask]
            a.append(class_emb.mean(dim=0).unsqueeze(0))

        class_prototypes[class_id.item()] = torch.concat(a)

    distances = torch.empty(b, qexamples, len(class_prototypes), device=getattr(self.model.args, 'device', 'cuda'), dtype=query_emb.dtype)

    for i, class_prototype in enumerate(class_prototypes.values()):
        distances[:, :, i] = - torch.norm(query_emb - class_prototype.unsqueeze(1), dim=2)

    loss = F.cross_entropy(distances.view(b*qexamples, -1), query_labels.view(b*qexamples))
    loss_item = loss.item()
    
    return loss_item