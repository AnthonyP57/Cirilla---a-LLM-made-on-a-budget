import json
import os
import random

def gather_summaries(in_path, out_path):
    summary_paths = []
    for main_path, subfolders, files in os.walk(in_path):
        for file in files:
            if file.endswith(".txt"):
                summary_paths.append(os.path.join(main_path, file))
    
    out_data = []
    for path in summary_paths:
        with open(path, 'r') as f:
            summary = f.read()
        if not summary == '':
            out_data.append(
                    {'subject': path.split('/')[-1].split('.')[0], 'text': summary, 'data type': 'plain text', 'source':'fandom', 'model': path.split('/')[-2]}
            )

    with open(out_path, 'a') as f:
        for d in out_data:
            f.write(json.dumps(d) + '\n')

random_questions_to_summarize_texts = [
    "Tell me about {placeholder}.",
    "What is {placeholder}?",
    "Provide a brief overview of {placeholder}.",
    "Explain {placeholder} in simple terms.",
    "What are the key details about {placeholder}?",
    "Summarize all information about {placeholder}.",
    "Give me important facts about {placeholder}.",
    "What should I know about {placeholder}?",
    "Provide a short summary of {placeholder}.",
    "List the main points about {placeholder}.",
    "Give an introduction to {placeholder}.",
    "What is notable about {placeholder}?",
    "Describe {placeholder} in a few sentences.",
    "What makes {placeholder} significant?",
    "Provide an overview of {placeholder}.",
    "Give some key information about {placeholder}.",
    "What are the main aspects of {placeholder}?",
    "Provide background information on {placeholder}.",
    "Explain the significance of {placeholder}.",
    "What should someone know about {placeholder}?"
]
def summaries_to_instruct(in_path, out_path):
    summary_paths = []
    for main_path, subfolders, files in os.walk(in_path):
        for file in files:
            if file.endswith(".txt"):
                summary_paths.append(os.path.join(main_path, file))
    
    out_data = []
    for path in summary_paths:
        with open(path, 'r') as f:
            summary = f.read()
        if not summary == '':
            out_data.append(
                    {'subject': path.split('/')[-1].split('.')[0], 'text': [
                        {'role':'user', 'content': random.choice(random_questions_to_summarize_texts).format(placeholder=path.split('/')[-1].split('.')[0])},
                        {'role': 'assistant', 'content': summary}
                        ],
                        'data type': 'plain text', 'source':'fandom', 'model': path.split('/')[-2]}
            )

    with open(out_path, 'a') as f:
        for d in out_data:
            f.write(json.dumps(d) + '\n')
