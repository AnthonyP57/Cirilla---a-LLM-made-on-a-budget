from reasoning_gym.composite import DatasetSpec
import reasoning_gym
import json
import random

jsonl_path = './training_datasets/mid_training/reason_gym_synth.jsonl'
n_samples = 400

specs = [
    DatasetSpec(name='aiw', weight=2, config={"max_entities":10}),
    DatasetSpec(name='basic_arithmetic', weight=2, config={"whitespace":"random"}),
    DatasetSpec(name='caesar_cipher', weight=2, config={}),
    DatasetSpec(name='calendar_arithmetic', weight=2, config={}),
    DatasetSpec(name='chain_sum', weight=2, config={}),
    DatasetSpec(name='decimal_chain_sum', weight=2, config={}),
    DatasetSpec(name='family_relationships', weight=2, config={}),
    DatasetSpec(name='fraction_simplification', weight=2, config={"styles":["plain"]}),
    # DatasetSpec(name='gsm_symbolic', weight=1, config={"difficulty":1.0}), # <- doesnt work: Object of type Fraction is not JSON serializable
    DatasetSpec(name='knights_knaves', weight=2, config={}),
    DatasetSpec(name='leg_counting', weight=2, config={}),
    DatasetSpec(name='letter_counting', weight=2, config={}),
    DatasetSpec(name='needle_haystack', weight=2, config={"min_num_statements":2, "max_num_statements":10}),
    DatasetSpec(name='number_filtering', weight=2, config={"max_numbers":5}),
    DatasetSpec(name='number_sequence', weight=2, config={"min_terms":4, "max_terms":8}),
    DatasetSpec(name='sentence_reordering', weight=2, config={}),
    DatasetSpec(name='simple_equations', weight=2, config={"min_terms":1, "max_terms":3}),
    DatasetSpec(name='simple_geometry', weight=1, config={}),
    DatasetSpec(name='spell_backward', weight=2, config={}),
    DatasetSpec(name='syllogism', weight=2, config={}),
]

n_failed=0
for seed in random.sample(range(1_000_000), n_samples):
    data = reasoning_gym.create_dataset('composite', size=100, seed=seed, datasets=specs)

    with open(jsonl_path, 'a') as f:
        for i, x in enumerate(data):
            assert data.score_answer(answer=x['answer'], entry=x) == 1.0
            assert type(x) == dict
            try:
                x = {
                    'subject': x['metadata']['source_dataset'],
                    'text': [{'role': 'user', 'text': x['question']}, {'role': 'assistant', 'text': x['answer']}],
                    'data type': 'conv'
                }
                f.write(json.dumps(x) + '\n')
            except Exception as e:
                n_failed += 1
                print(e, x['metadata']['source_dataset'])