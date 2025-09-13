from datasets import load_dataset
import json
import re

pretraining = './training_datasets/pretraining.jsonl'
pretraining_ = './training_datasets/pretraining_overview.jsonl'

""""""
dataset = load_dataset("roneneldan/TinyStories", split='train[:2%]')

data = []

with open(pretraining_, 'a') as f:
    f.write(json.dumps({'metadata': {'dataset':"roneneldan/TinyStories", 'split':'train[:2%]'},
        'num_rows':dataset.num_rows}) + '\n')

for text in dataset:
    data.append({'text': text['text'], 'data type': 'plain text', 'metadata': {'dataset':"roneneldan/TinyStories", 'split':'train[:2%]'}})

with open(pretraining, 'a') as f:
    for d in data:
        f.write(json.dumps(d) + '\n')

""""""
dataset = load_dataset("roneneldan/TinyStoriesInstruct", split='train[:1%]')

data = []

with open(pretraining_, 'a') as f:
    f.write(json.dumps({'metadata': {'dataset':"roneneldan/TinyStoriesInstruct", 'split':'train[:1%]'},
        'num_rows':dataset.num_rows}) + '\n')
    
for text in dataset:
    text = text['text']
    if not bool(re.match(r'^(\w+:|<\|)', text)) and len(text) > 10: # if text doesn't start with some_word: or <|somw_word
        data.append({'text': text, 'data type': 'plain text', 'metadata': {'dataset':"roneneldan/TinyStoriesInstruct", 'split':'train[:1%]'}})

with open(pretraining, 'a') as f:
    for d in data:
        f.write(json.dumps(d) + '\n')

""""""

dataset = load_dataset("nyu-mll/glue", "mnli", split="train[:3%]")

with open(pretraining_, 'a') as f:
    f.write(json.dumps({'metadata': {'dataset':"nyu-mll/glue", 'subset':'mnli', 'split':'train[:3%]'},
        'num_rows':dataset.num_rows * 2}) + '\n')

data = []

for text in dataset:
    data.append({'text': text['premise'], 'data type': 'plain text', 'metadata': {'dataset':"nyu-mll/glue", "subset":"mnli", 'split':'train[:3%]'}})
    data.append({'text': text['hypothesis'], 'data type': 'plain text', 'metadata': {'dataset':"nyu-mll/glue", "subset":"mnli", 'split':'train[:3%]'}})

with open(pretraining, 'a') as f:
    for d in data:
        f.write(json.dumps(d) + '\n')

""""""

dataset = load_dataset("SimpleStories/SimpleStories", split="train[:2%]")

data = []

with open(pretraining_, 'a') as f:
    f.write(json.dumps({'metadata': {'dataset':"SimpleStories/SimpleStories", 'split':'train[:2%]'},
        'num_rows':dataset.num_rows}) + '\n')

for text in dataset:
    data.append({'text': text['story'], 'data type': 'plain text', 'metadata': {'dataset':"SimpleStories/SimpleStories", 'split':'train[:2%]'}})

with open(pretraining, 'a') as f:
    for d in data:
        f.write(json.dumps(d) + '\n')

""""""
# dataset = load_dataset("Helsinki-NLP/opus-100", "en-sv", split="train[:5%]")

# with open(pretraining_, 'a') as f:
#     f.write(json.dumps({'metadata': {'dataset':"Helsinki-NLP/opus-100", "subset":"en-sv", 'split':'train[:5%]'},
#         'num_rows':dataset.num_rows}) + '\n')

# data = []

# for text in dataset:
#     data.append({'text': text['translation']['en'], 'metadata': {'dataset':"Helsinki-NLP/opus-100", "subset":"en-sv", 'split':'train[:5%]'}})

# with open(pretraining, 'a') as f:
#     for d in data:
#         f.write(json.dumps(d) + '\n')
