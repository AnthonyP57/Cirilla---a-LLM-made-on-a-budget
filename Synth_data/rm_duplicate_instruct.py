from fuzzywuzzy import fuzz
import os
import re
import json
import numpy as np
from pathlib import Path
from typing import Union

def rm_duplicate_instructs(main_dir: Union[Path, list[Path]], save_to: Path):
    files = []

    for main_path, subfolders, _files in os.walk(main_dir):
        for _file in _files:
            if _file.endswith(".json"):
                files.append(os.path.join(main_path, _file))

    print(f"Found {len(files)} files.")

    all_names = set(
        re.sub(r'\d+', '', os.path.basename(file).replace('_', ''))
        for file in files
    )

    valid_paths = []

    for name in all_names:
        same_files = [
            file for file in files
            if re.sub(r'\d+', '', os.path.basename(file).replace('_', '')) == name
        ]
        
        data = []
        for file in same_files:
            with open(file, 'r') as f:
                data.append(json.load(f))

        n = len(data)
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                text_i = data[i].get('question', '') + ' ' + data[i].get('answer', '')
                text_j = data[j].get('question', '') + ' ' + data[j].get('answer', '')
                sim_matrix[i, j] = fuzz.ratio(text_i, text_j)

        ids_delete = set()
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] > 90:
                    ids_delete.add(j)

        for i in range(n):
            if i not in ids_delete:
                valid_paths.append(same_files[i])

    print(f"Remaining files: {len(valid_paths)}")

    data_to_save = []

    for name in all_names:
        same_files = [
            file for file in valid_paths
            if re.sub(r'\d+', '', os.path.basename(file).replace('_', '')) == name
        ]
        for file in same_files:
            model = file.split('/')[-2]
            qa = json.load(open(file, 'r'))
            if (qa['question'] != '') and (qa['answer'] != ''):
                data_ = ({
                    'subject': name.split('.')[0],
                    'text': [{'role':'user', 'content': qa['question']}, {'role': 'assistant', 'content': qa['answer']}],
                    'data type': 'conv',
                    'model': model,
                })

                data_to_save.append(data_)

    with open(save_to, 'w') as f:
        for d in data_to_save:
            f.write(json.dumps(d) + '\n')
