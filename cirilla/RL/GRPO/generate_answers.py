from cirilla.Cirilla_model import Cirilla
from cirilla.Cirilla_model import CirillaTokenizer
from cirilla.Cirilla_model.modules import select_torch_device
from datasets import Dataset
import json
import torch
import math

class CirillaResponseGenerator:
    def __init__(self, hub_url):

        self.model = Cirilla()
        self.model.pull_model_from_hub(hub_url, inference_mode=True, map_device=select_torch_device())
        self.tokenizer = CirillaTokenizer(hub_url=hub_url)
        self.termination_tokens = [self.tokenizer.convert_tokens_to_ids('<eos>'), self.tokenizer.convert_tokens_to_ids('<|user|>')]
        self.generation_config = {
                                'kv_cache':True,
                                'top_p':0.2,
                                'top_k':None,
                                'temperature':1.0,
                                'n_beams':3,
                                }

    def generate_batch(self, prompt:str, n:int=1, kv_cache:bool=True):

        if kv_cache:
            batch_prompts = [[{'role':'user', 'content': prompt}] for _ in range(n)] if isinstance(prompt, str) else [[{'role':'user', 'content': p}] for p in prompt]
            x = self.tokenizer.apply_chat_template(batch_prompts, padding='do_not_pad', add_generation_prompt=True)
            out = self.model.generate_kv_cache(x, termination_tokens=self.termination_tokens,
                                                    top_k=self.generation_config['top_k'],
                                                    top_p=self.generation_config['top_p'],
                                                    temperature=self.generation_config['temperature'],
                                                    sample_parallel=False)

            input_lengths = [len(xi) for xi in x]

            texts = [self.tokenizer.decode(o[l:])\
                                            .replace('<pad>', '')\
                                            .replace('<|user|>', '')\
                                            .replace('<|assistant|>', '')\
                                            .replace('<eos>', '')\
                                            .replace('<sos>', '')\
                                            .replace('<unk>', '')\
                                            .strip() for o, l in zip(out, input_lengths)]
        else:
            texts = []
            for _ in range(n):
                x = self.tokenizer.apply_chat_template([{'role':'user', 'content': prompt}], padding='do_not_pad', add_generation_prompt=True, return_tensors='pt')
                out = self.model.generate_naive(x.to(self.model.args.device),
                                                termination_tokens=self.termination_tokens,
                                                top_k=self.generation_config['top_k'],
                                                top_p=self.generation_config['top_p'],
                                                n_beams=self.generation_config['n_beams'],
                                                temperature=self.generation_config['temperature'])[0]
                input_length = x.shape[1]

                texts.append(self.tokenizer.decode(out[input_length:])\
                                                .replace('<pad>', '')\
                                                .replace('<|user|>', '')\
                                                .replace('<|assistant|>', '')\
                                                .replace('<eos>', '')\
                                                .replace('<sos>', '')\
                                                .replace('<unk>', '')\
                                                .strip())
                
        self.model.clear_cache()
        return texts

class CirillaSampler:
    def __init__(self, crg: CirillaResponseGenerator):
        self.crg = crg

    def sample(self, prompt_dataset:Dataset, batch_size:int=32, n_generate_with_kv_cache:int=1, n_generate_naive:int=2):

        crg = self.crg

        out = []
        j = 0

        for line in prompt_dataset:
            id = line['id']
            if id > 3:
                break
            prompt = line['prompt']
            answers_naive = crg.generate_batch(prompt, n=n_generate_naive, kv_cache=False)

            for ans in answers_naive:
                out.append({
                    'id': id,
                    'answer': ans,
                    'log_probs_id':j
                })
                j += 1

        for _ in range(n_generate_with_kv_cache):
            _prompt_dataset = iter(prompt_dataset)
            for _ in range(math.ceil(len(out) / batch_size)):
                ids = []
                prompts = []
                for _ in range(batch_size):
                    line = next(_prompt_dataset)
                    if line != '':
                        if line['id'] > 3:
                            break
                        ids.append(line['id'])
                        prompts.append(line['prompt'])

                answers_kv = crg.generate_batch(prompts, kv_cache=True)

                for ans, id in zip(answers_kv, ids):
                    out.append({
                        'id': id,
                        'answer': ans,
                        'log_probs_id':j
                    })
                    j += 1
                    
        out.sort(key=lambda x: x['id'])
        return Dataset.from_list(out)
    
    def get_log_probs(self, prompt_dataset:Dataset, sampled_dataset:Dataset, batch_size:int=32):

        out_tensor = []
        generate_batch = []
        prompt_lens = []
        js = []

        last_out = sampled_dataset[-1]['id']
        for ans_data in sampled_dataset:
            prompt = prompt_dataset[ans_data['id']]['prompt']
            answer = ans_data['answer']
            j = ans_data['log_probs_id']
            end = ans_data['id'] == last_out
            template = self.crg.tokenizer.apply_chat_template(
                    [
                        {'role':'user', 'content': prompt},
                        {'role':'assistant', 'content': answer}
                    ],
                    padding='do_not_pad',
                    add_generation_prompt=False
                )
            
            prompt_len = len(self.crg.tokenizer.apply_chat_template(
                [{'role':'user', 'content': prompt}],
                padding='do_not_pad',
                add_generation_prompt=True
            ))
            js.append(j)
            generate_batch.append(template)
            prompt_lens.append(prompt_len)
            if len(generate_batch) == batch_size or end:
                x = torch.full((len(generate_batch), max(len(t) for t in generate_batch)), self.crg.tokenizer.convert_tokens_to_ids('<pad>'), dtype=torch.int64) # one is padding
                for i, t in enumerate(generate_batch):
                    x[i, :len(t)] = torch.tensor(t, dtype=torch.int64)
                per_token_log_probs = self.crg.model.get_per_token_log_probs(x.to(self.crg.model.args.device))
                for j, row_ptlp, plen, batch in zip(js, per_token_log_probs, prompt_lens, generate_batch):
                    out_tensor.append({'log_probs_id':j, 'per_token_log_probs':row_ptlp[plen-1:len(batch)-1].tolist()})
                generate_batch = []
                prompt_lens = []
                js = []

        return Dataset.from_list(out_tensor)

if __name__ == '__main__':
    from datasets import load_dataset
    crg = CirillaResponseGenerator('AnthonyPa57/Cirilla-0.3B-4E')

    sampler = CirillaSampler(crg)

    prompt_dataset = load_dataset('AnthonyPa57/Witcher-GRPO-prompts', split='train')

    sampled_dataset = sampler.sample(prompt_dataset, batch_size=32, n_generate_with_kv_cache=2)
    sampled_dataset.to_json('./training_datasets/RL/sampled_cirilla.jsonl')

    log_probs_dataset = sampler.get_log_probs(prompt_dataset, sampled_dataset, batch_size=32)
    log_probs_dataset.to_json('./training_datasets/RL/per_token_log_probs.jsonl')
