import os
import json

def instructions_into_conv(input_path, out_path):
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
