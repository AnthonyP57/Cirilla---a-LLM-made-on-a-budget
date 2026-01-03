from cirilla.Cirilla_model import Cirilla
from cirilla.Cirilla_model import CirillaTokenizer
from cirilla.Cirilla_model.modules import select_torch_device

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

if __name__ == '__main__':
    import json
    import math

    crg = CirillaResponseGenerator('AnthonyPa57/Cirilla-0.3B-4E')
    # print(crg.generate_batch('Who is Geralt?', 3, kv_cache=True))
    # print(crg.generate_batch('Who is Geralt?', 3, kv_cache=False))

    prompts_path = './training_datasets/RL/prompts.jsonl'
    out_path = './training_datasets/RL/sampled_cirilla.jsonl'
    out = []

    for line in open(prompts_path, 'r'):
        line = json.loads(line)
        id = line['id']
        if id > 3:
            break
        prompt = line['prompt']
        answers_naive = crg.generate_batch(prompt, 2, kv_cache=False)

        for ans in answers_naive:
            out.append({
                'id': id,
                'answer': ans
            })

    batch_size = 32
    with open(prompts_path, 'r') as f:
        for _ in range(math.ceil(len(out) / batch_size)):
            ids = []
            prompts = []
            for _ in range(batch_size):
                line = f.readline()
                if line != '':
                    line = json.loads(line)
                    if line['id'] > 3:
                        break
                    ids.append(line['id'])
                    prompts.append(line['prompt'])

            answers_kv = crg.generate_batch(prompts, kv_cache=True)

            for ans, id in zip(answers_kv, ids):
                out.append({
                    'id': id,
                    'answer': ans
                })

    with open(out_path, 'w') as f:
        for d in out:
            json.dump(d, f)
            f.write('\n')