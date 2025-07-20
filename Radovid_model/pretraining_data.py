from datasets import load_dataset
import json
import re

pretraining = './training_datasets/pretraining.jsonl'
pretraining_ = './training_datasets/pretraining_overview.jsonl'

""""""
dataset = load_dataset("roneneldan/TinyStories", split='train')

data = {'metadata': {'dataset':"roneneldan/TinyStories", 'split':'train'},
        'data': []}

with open(pretraining_, 'a') as f:
    f.write(json.dumps({'metadata': {'dataset':"roneneldan/TinyStories", 'split':'train'},
        'num_rows':dataset.num_rows}) + '\n')

for text in dataset:
    data['data'].append(text['text'])

with open(pretraining, 'a') as f:
    f.write(json.dumps(data) + '\n')

print(f'for dataset {data["metadata"]["dataset"]}, {data["metadata"]["split"]} has {dataset.num_rows} rows\n sample: {data["data"][0][:200]}')
""""""
dataset = load_dataset("roneneldan/TinyStoriesInstruct", split='train[:10%]')

data = {'metadata': {'dataset':"roneneldan/TinyStoriesInstruct", 'split':'train[:10%]'},
        'data': []}

with open(pretraining_, 'a') as f:
    f.write(json.dumps({'metadata': {'dataset':"roneneldan/TinyStoriesInstruct", 'split':'train[:10%]'},
        'num_rows':dataset.num_rows}) + '\n')
    
for text in dataset:
    text = text['text']
    if not bool(re.match(r'^(\w+:|<\|)', text)) and len(text) > 10: # if text doesn't start with some_word: or <|somw_word
        data['data'].append(text)

with open(pretraining, 'a') as f:
    f.write(json.dumps(data) + '\n')

print(f'for dataset {data["metadata"]["dataset"]}, {data["metadata"]["split"]} has {dataset.num_rows} rows\n sample: {data["data"][0][:200]}')
""""""

dataset = load_dataset("nyu-mll/glue", "mnli", split="train")

with open(pretraining_, 'a') as f:
    f.write(json.dumps({'metadata': {'dataset':"nyu-mll/glue", 'subset':'mnli', 'split':'train'},
        'num_rows':dataset.num_rows * 2}) + '\n')

data = {'metadata': {'dataset':"nyu-mll/glue", "subset":"mnli", 'split':'train'},
        'data': []}

for text in dataset:
    data['data'].append(text['premise'])
    data['data'].append(text['hypothesis'])

with open(pretraining, 'a') as f:
    f.write(json.dumps(data) + '\n')

print(f'for dataset {data["metadata"]["dataset"]}, {data["metadata"]["split"]} has {dataset.num_rows} rows\n sample: {data["data"][0][:200]}')
""""""

dataset = load_dataset("SimpleStories/SimpleStories", split="train")

data = {'metadata': {'dataset':"SimpleStories/SimpleStories", 'split':'train'},
        'data': []}

with open(pretraining_, 'a') as f:
    f.write(json.dumps({'metadata': {'dataset':"SimpleStories/SimpleStories", 'split':'train'},
        'num_rows':dataset.num_rows}) + '\n')

for text in dataset:
    data['data'].append(text['story'])

with open(pretraining, 'a') as f:
    f.write(json.dumps(data) + '\n')

print(f'for dataset {data["metadata"]["dataset"]}, {data["metadata"]["split"]} has {dataset.num_rows} rows\n sample: {data["data"][0][:200]}')
""""""
dataset = load_dataset("Helsinki-NLP/opus-100", "en-sv", split="train")

with open(pretraining_, 'a') as f:
    f.write(json.dumps({'metadata': {'dataset':"Helsinki-NLP/opus-100", "subset":"en-sv", 'split':'train'},
        'num_rows':dataset.num_rows}) + '\n')

data = {'metadata': {'dataset':"Helsinki-NLP/opus-100", "subset":"en-sv", 'split':'train'},
        'data': []}

for text in dataset:
    data['data'].append(text['translation']['en'])

with open(pretraining, 'a') as f:
    f.write(json.dumps(data) + '\n')

print(f'for dataset {data["metadata"]["dataset"]}, {data["metadata"]["split"]} has {dataset.num_rows} rows\n sample: {data["data"][0][:200]}')