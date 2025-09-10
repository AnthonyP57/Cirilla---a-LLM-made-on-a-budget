import os
import json
import re
from pathlib import Path

def multi_turn_gather(input_path:Path, save_to:Path):
    """
    create synthetic multi turn conversations about given topics with vllm
    
    Args:
        input_path (Path): path to the folder containing the .json files, they can be nested
        save_to (Path): path to save the synthetic multi turn dataset to
    
    Returns:
        None
    """
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
                out_data.extend([{'role':'user', 'content': qa['question']}, {'role': 'assistant', 'content': qa['answer']}])
                contexts.append(qa['context'])
            out = {'subject': re.sub(r'_\d+$', '', file.split('/')[-1].split('.')[0]),
                'text': out_data,
                'data type': 'conv',
                'source': 'fandom',
                'metadata': {'contexts': contexts}}
                
            with open(save_to, 'a') as f:
                f.write(json.dumps(out) + '\n')
