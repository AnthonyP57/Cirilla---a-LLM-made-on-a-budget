from cirilla.Cirilla_model import Cirilla
from cirilla.Cirilla_model import CirillaTokenizer
from cirilla.Cirilla_model.modules import select_torch_device
from datasets import Dataset
import torch
import math
from mistralai import Mistral
from dotenv import load_dotenv
import os
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

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
                                'n_beams':2,
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
    def __init__(self, crg: CirillaResponseGenerator, user_token='<|user|>'):
        self.crg = crg
        self.user_token = [self.crg.tokenizer.convert_tokens_to_ids(user_token)]

    def sample(self, prompt_dataset:Dataset, batch_size:int=32, n_generate_with_kv_cache:int=1, n_generate_naive:int=2, generate_mistral:int=0, store_mistral_answers:str='./mistral_answers.jsonl'):

        crg = self.crg

        out = []
        j = 0

        if n_generate_naive > 0:

            for line in tqdm(prompt_dataset, desc="Naive sampling"):
                id = line['id']
                prompt = line['prompt']
                answers_naive = crg.generate_batch(prompt, n=n_generate_naive, kv_cache=False)

                for ans in answers_naive:
                    out.append({
                        'id': id,
                        'answer': ans,
                        'log_probs_id':j
                    })
                    j += 1

        if n_generate_with_kv_cache > 0:

            for i in range(n_generate_with_kv_cache):
                _prompt_dataset = iter(prompt_dataset)
                for _ in tqdm(range(math.ceil(len(prompt_dataset) / batch_size)), desc=f"KV sampling {i+1}/{n_generate_with_kv_cache}"):
                    ids = []
                    prompts = []
                    for _ in range(batch_size):
                        try:
                            line = next(_prompt_dataset)
                            if line != '':
                                ids.append(line['id'])
                                prompts.append(line['prompt'])
                        except StopIteration:
                            break
                    
                    if not prompts:
                        continue

                    answers_kv = crg.generate_batch(prompts, kv_cache=True)

                    for ans, id in zip(answers_kv, ids):
                        out.append({
                            'id': id,
                            'answer': ans,
                            'log_probs_id':j
                        })
                        j += 1

        if generate_mistral > 0:

            if store_mistral_answers is not None and os.path.exists(store_mistral_answers):
                with open(store_mistral_answers, 'r') as f:
                    for line in f:
                        line = json.loads(line)
                        out.append({
                            'id': line['id'],
                            'answer': line['answer'],
                            'log_probs_id':j
                        })
                        j += 1

            else:

                load_dotenv()

                SYS_PROMPT = """\
                Based on the context and the prompt you should generate an answer that would satisfy the following criteria:
                - correct grammar
                - correct logic
                - the answer is only based on the context
                - the answer has to be as short as possible, it does not need to encompass the whole prompt but it has to answer it, use up to 20 words to answer the prompt
                - the answer has to use simple terms, pretend you are a very simple chatbot that is supposed to accurately answer the question based on the context, no elaboration
                - the answer cannot contain any markdown, like ** or _ or * etc.
                - the answer cannot contain any code blocks, like ``` or \`\`\`
                
                example prompt: Which two kings did Dethmold serve in The Witcher 2: Assassins of Kings?
                example answer: King Esterad and King Henselt of Kaedwen.
                
                example prompt: In which book does the story of Ciri entering a portal and becoming trapped in a different world first appear?
                example answer: The Lady of the Lake.
                
                example prompt: Who is Geralt of Rivia, and what is his role in The Witcher universe?
                example answer: Geralt of Rivia is a highly skilled witcher known for his ability to hunt and defeat supernatural creatures. He serves as the protagonist in the Witcher series."""

                MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "your_api_key_here")
                client = Mistral(api_key=MISTRAL_API_KEY, timeout_ms=10000)

                def _mistral_call(client, prompt, context):
                    try:
                        response = client.chat.complete(
                            model="ministral-8b-2512",
                            messages=[
                                {"role": "system", "content": SYS_PROMPT},
                                {"role": "user", "content": f"Context: {context}\nPrompt: {prompt}"}
                            ],
                            temperature=0.2,
                        )
                        return response.choices[0].message.content.replace('*', '')
                    except Exception as e:
                        print(f"API Error: {e}")
                        return ""

                with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 2)) as executor: # rate limits

                    if store_mistral_answers is not None:
                        f = open(store_mistral_answers, 'w')

                    for ans, sample_id in tqdm(zip(
                        executor.map(
                            _mistral_call,
                            [client] * len(prompt_dataset),
                            [line['prompt'] for line in prompt_dataset],
                            [line['context'] for line in prompt_dataset],
                        ),
                        [line['id'] for line in prompt_dataset],
                    ), total=len(prompt_dataset), desc="Generating Mistral answers"):
                        
                        out.append({
                            'id': sample_id,
                            'answer': ans,
                            'log_probs_id': j
                        })
                        j += 1

                        if ans and store_mistral_answers is not None:
                                f.write(json.dumps({'id': sample_id, 'answer': ans}) + '\n')
                    
                    if store_mistral_answers is not None:
                        f.close()
                    
        out.sort(key=lambda x: x['id'])
        return Dataset.from_list(out)
    
    def get_log_probs(self, prompt_dataset:Dataset, sampled_dataset:Dataset, batch_size:int=32, max_len:int=2048):

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
                        {'role':'assistant', 'content': answer if answer is not None else ''}
                    ],
                    padding='do_not_pad',
                    max_len=max_len,
                    add_generation_prompt=False
                )
            if len(template) < max_len:
                template += self.user_token
            
            prompt_len = len(self.crg.tokenizer.apply_chat_template(
                [{'role':'user', 'content': prompt}],
                padding='do_not_pad',
                max_len=max_len,
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
