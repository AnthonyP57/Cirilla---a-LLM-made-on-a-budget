import json

path = './training_datasets/domain_training/fandom_summaries_instruct.jsonl'
out_path = './training_datasets/RL/prompts.jsonl'
out_ans_path= './training_datasets/RL/answers.jsonl'
out = []
ans = []
i = 0
for line in open(path, 'r'):
    line = json.loads(line)
    out.append({
        'id': i,
        'subject': line['subject'],
        'prompt': line['text'][0]['content']
    })
    ans.append({
        'id': i, 'answer': line['text'][1]['content']})
    i += 1

path = './training_datasets/domain_training/synth_multi_round.jsonl'
for line in open(path, 'r'):
    line = json.loads(line)
    for subject, prompt, answer in zip(
        line['metadata']['contexts'], [t['content'] for t in line['text'] if t['role'] == 'user'], [t['content'] for t in line['text'] if t['role'] == 'assistant']
        ):
        out.append({
            'id': i,
            'subject': subject,
            'prompt': prompt
        })
        ans.append({
            'id': i, 'answer': answer
            })
        i += 1

path = './training_datasets/domain_training/witcher_instruct.jsonl'
for line in open(path, 'r'):
    line = json.loads(line)
    out.append({
        'id': i,
        'subject': line['subject'],
        'prompt': line['text'][0]['content']
    })
    ans.append({
        'id': i, 'answer': line['text'][1]['content']})
    i += 1

path = './training_datasets/domain_training/witcher_synthetic_instruct.jsonl'
for line in open(path, 'r'):
    line = json.loads(line)
    out.append({
        'id': i,
        'subject': line['subject'],
        'prompt': line['text'][0]['content']
    })
    ans.append({
        'id': i, 'answer': line['text'][1]['content']})
    i += 1

with open(out_path, 'w') as f:
    for d in out:
        json.dump(d, f)
        f.write('\n')

with open(out_ans_path, 'w') as f:
    for d in ans:
        json.dump(d, f)
        f.write('\n')