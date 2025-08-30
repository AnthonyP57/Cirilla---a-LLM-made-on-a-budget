import os
import json

input_path = './training_datasets/raw/async_fandom_instruct'
out_path = './training_datasets/domain_training/witcher_instruct.jsonl'

for file in os.listdir(input_path):
    with open(os.path.join(input_path, file), 'r') as f:
        qa = json.load(f)
        for k,v in qa.items():
            out = {'subject': file.split('.')[0],
                   'text': [{'role':'user', 'content': k}, {'role': 'assistant', 'content': v}],
                   'data type': 'conv',
                   'source': 'fandom'}
            
            with open(out_path, 'a') as f:
                f.write(json.dumps(out) + '\n')
            