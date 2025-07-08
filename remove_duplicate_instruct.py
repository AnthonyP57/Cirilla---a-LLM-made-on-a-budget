from fuzzywuzzy import fuzz
import os
import re
import json
import numpy as np

main_dir = './witcher_synthetic_instruct'
save_to = './witcher_synthetic_instruct/clean.jsonl'

files = []
for model in os.listdir(main_dir):
    model_dir = os.path.join(main_dir, model)
    if os.path.isdir(model_dir):
        for file in os.listdir(model_dir):
            files.append(os.path.join(model_dir, file))

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
# print(valid_paths[-5:])

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
            data_to_save.append({
                'subject': name.split('.')[0],
                'question': qa['question'],
                'answer': qa['answer'],
                'model': model,
            })

with open(save_to, 'w') as f:
    for d in data_to_save:
        f.write(json.dumps(d) + '\n')