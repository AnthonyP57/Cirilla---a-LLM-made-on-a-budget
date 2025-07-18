from datasets import load_dataset
import json
import re

pretraining = './training_datasets/pretrainig.jsonl'
pretraining_ = './training_datasets/pretrainig_overview.jsonl'

""""""
# dataset = load_dataset("roneneldan/TinyStories", split='validation')

# data = {'metadata': {'dataset':"roneneldan/TinyStories", 'split':'validation'},
#         'data': dataset[:]}

# with open(pretraining_, 'a') as f:
#     json.dump({'metadata': {'dataset':"roneneldan/TinyStories", 'split':'validation'},
#         'num_rows':dataset.num_rows}, f)

# with open(pretraining, 'a') as f:
#     json.dump(data, f)

""""""
# dataset = load_dataset("roneneldan/TinyStories", split='train')

# data = {'metadata': {'dataset':"roneneldan/TinyStories", 'split':'train'},
#         'data': dataset[:]}

# with open(pretraining_, 'a') as f:
#     json.dump({'metadata': {'dataset':"roneneldan/TinyStories", 'split':'train'},
#         'num_rows':dataset.num_rows}, f)

# with open(pretraining, 'a') as f:
#     json.dump(data, f)


""""""
# dataset = load_dataset("roneneldan/TinyStoriesInstruct", split='train[:10%]')

# data = {'metadata': {'dataset':"roneneldan/TinyStoriesInstruct", 'split':'train[:10%]'},
#         'data': []}

# with open(pretraining_, 'a') as f:
#     json.dump({'metadata': {'dataset':"roneneldan/TinyStoriesInstruct", 'split':'train[:10%]'},
#         'num_rows':dataset.num_rows}, f)

# for text in dataset:
#     text = text['text']
#     if not bool(re.match(r'^(\w+:|<\|)', text)) and len(text) > 10:
#         data['data'].append(text)

# with open(pretraining, 'a') as f:
#     json.dump(data, f)

""""""

# dataset = load_dataset("nyu-mll/glue", "mnli", split="train")

# with open(pretraining_, 'a') as f:
#     json.dump({'metadata': {'dataset':"nyu-mll/glue", 'subset':'mnli', 'split':'train'},
#         'num_rows':dataset.num_rows}, f)

# data = {'metadata': {'dataset':"nyu-mll/glue", "subset":"mnli", 'split':'train'},
#         'data': []}

# for text in dataset:
#     data['data'].append(text['premise'])
#     data['data'].append(text['hypothesis'])

# with open(pretraining, 'a') as f:
#     json.dump(data, f)

""""""

# dataset = load_dataset("SimpleStories/SimpleStories", split="train[:1%]")

# data = {'metadata': {'dataset':"SimpleStories/SimpleStories", 'split':'train'},
#         'data': dataset[:]}

# with open(pretraining_, 'a') as f:
#     json.dump({'metadata': {'dataset':"SimpleStories/SimpleStories", 'split':'train'},
#         'num_rows':dataset.num_rows}, f)

# with open(pretraining, 'a') as f:
#     json.dump(data, f)

