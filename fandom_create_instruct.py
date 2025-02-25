import os
import json

input_directory = './witcher_fandom'
output_directory = './witcher_instruct'
n_instructs=0
for file in os.listdir(input_directory):
    file_path = os.path.join(input_directory, file)
    
    with open(file_path, 'r') as f:
        text = f.read()
        text = text.split('\n')

        capture = False
        extracted_lines = []

        for i, line in enumerate(text):

            if text[i-1].startswith('Quick Answers'):
                capture = True
            elif line.startswith('{'):
                capture = False

            if capture:
                extracted_lines.append(line)

        if len(extracted_lines) > 0:
            questions=[]
            answers=[]
            capture=False
            for l in extracted_lines:
                if l.startswith('						Provided by:'):
                    capture=False
                if capture:
                    answers[-1].extend(l)
                if l.endswith('?'):
                    questions.append(l)
                    answers.append([])
                    capture=True
                

            ans_=[]
            for a in answers:
                a = ''.join(a)
                a = a.replace('\n', ' ')
                a = a.replace('\t', '')
                a = a.replace("'", '')
                ans_.append(a)

            q_a_a = {q:a for q, a in zip(questions, ans_)}
            n_instructs+= len(q_a_a)

            with open(os.path.join(output_directory, f'{file.split('.')[0]}.json'), 'w') as f:
                json.dump(q_a_a, f)

print(f'Created {n_instructs} instructions')
# Created 359 instructions