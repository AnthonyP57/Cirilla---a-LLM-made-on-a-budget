import os
import json
import re

def multiround_instructions_into_conv(input_path, out_path):
    files = []
    for main_path, subfolders, _files in os.walk(input_path):
        for _file in _files:
            if _file.endswith(".json"):
                files.append(os.path.join(main_path, _file))

    for file in files:
        with open(os.path.join(file), 'r') as f:
            qa_list = json.load(f)
            out_data = []
            contexts = []
            if not isinstance(qa_list, list):
                qa_list = [qa_list]
            for qa in qa_list:
                out_data.append([{'role':'user', 'content': qa['question']}, {'role': 'assistant', 'content': qa['answer']}])
                contexts.append(qa['context'])
            out = {'subject': re.sub(r'_\d+$', '', file.split('/')[-1].split('.')[0]),
                'text': out_data,
                'data type': 'conv',
                'source': 'fandom',
                'metadata': {'contexts': contexts}}
                
            with open(out_path, 'a') as f:
                f.write(json.dumps(out) + '\n')

if __name__ == "__main__":

    inp = './training_datasets/raw/synth_multi_round'
    outp = './training_datasets/domain_training/synth_multi_round.jsonl'

    multiround_instructions_into_conv(inp, outp)