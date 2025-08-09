import ollama
from pydantic import BaseModel, Field
import time
import os
import json

# Ollama options
#   numa: Optional[bool] = None
#   num_ctx: Optional[int] = None
#   num_batch: Optional[int] = None
#   num_gpu: Optional[int] = None
#   main_gpu: Optional[int] = None
#   low_vram: Optional[bool] = None
#   f16_kv: Optional[bool] = None
#   logits_all: Optional[bool] = None
#   vocab_only: Optional[bool] = None
#   use_mmap: Optional[bool] = None
#   use_mlock: Optional[bool] = None
#   embedding_only: Optional[bool] = None
#   num_thread: Optional[int] = None

#   # runtime options
#   num_keep: Optional[int] = None
#   seed: Optional[int] = None
#   num_predict: Optional[int] = None
#   top_k: Optional[int] = None
#   top_p: Optional[float] = None
#   tfs_z: Optional[float] = None
#   typical_p: Optional[float] = None
#   repeat_last_n: Optional[int] = None
#   temperature: Optional[float] = None
#   repeat_penalty: Optional[float] = None
#   presence_penalty: Optional[float] = None
#   frequency_penalty: Optional[float] = None
#   mirostat: Optional[int] = None
#   mirostat_tau: Optional[float] = None
#   mirostat_eta: Optional[float] = None
#   penalize_newline: Optional[bool] = None
#   stop: Optional[Sequence[str]] = None

class OllamaInstructCurate:
    def __init__(self, model, system_prompt, response_template:BaseModel):
        """
        atacker (str): The name of the attacker model e.g. 'llama3.2:3b'\n
        atacker_sys_prompt (str): The system prompt for the attacker model\n
        defender (str): The name of the defender model e.g. 'llama3.2:3b'\n
        defender_sys_prompt (str): The system prompt for the defender model
        """
        self.system_prompt = {'role': 'system', 'content': system_prompt}
        self.model = model
        self.response_template = response_template
        self.convo = {}

    def __call__(self, paths:list[str], save_to:str='./example', seed:int=42, checkpoint:int=10, skip=True):

        start = time.time()
        n_skipped = 0
        n_failed = 0
        not_failed = 0
        
        os.makedirs(save_to, exist_ok=True)

        for i,p in enumerate(paths):

            data = open(p, 'r').read()
            data = {'role': 'user', 'content': data}
            
            chat = [self.system_prompt]

            if '/' in p:
                p = p.split('/')[-1]
            p = p.split('.')[0]

            if os.path.exists(f'{save_to}/{p}.json'):
                if skip:
                    n_skipped += 1
                    continue
                else:
                    with open(f'{save_to}/{p}.json', 'r') as f:
                        qa = json.load(f)
                    
                    questions=f"find a question and answer pair that is different from:\nquestion: {qa['question']} answer: {qa['answer']}\n"
                    p = f"{p}_"
                    if os.path.exists(f'{save_to}/{p}.json'):
                        j=1
                        while True:
                            p_ = f"{p}{j}"
                            if not os.path.exists(f'{save_to}/{p_}.json'):
                                p = p_
                                break
                            j += 1
                            with open(f'{save_to}/{p_}.json', 'r') as f:
                                qa = json.load(f)
                            questions = questions + f"question: {qa['question']} answer: {qa['answer']}\n"

                    chat.append({'role': 'user', 'content': questions})

            chat.append(data)

            response = ollama.chat(
                model=self.model,
                messages = chat,
                format=self.response_template.model_json_schema(),
                options={
                    'num_predict': 512, # max num tokens
                    'seed': seed
                }
            )
            try:
                response = self.response_template.model_validate_json(response.message.content)
                response = response.model_dump()
                self.convo[p] = response
                not_failed += 1
            except:
                n_failed += 1
                print(f'failed on {p} failed:not failed {n_failed}:{not_failed}')
            
            print(f' ETA: {((time.time() - start) / (i+1 - n_skipped) * (len(paths) - i))/60:3.1f} min ', end='\r')

            if (i % checkpoint == 0) or (i == len(paths)-1):
                for k, v in self.convo.items():
                    # if not os.path.exists(f'{save_to}/{k}.json'):
                    with open(f'{save_to}/{k}.json', 'w') as f:
                        json.dump(v, f)
                
                self.convo = {}

        os.system(f'ollama stop {self.model}')

if __name__ == '__main__':
    import random

    os.environ["OLLAMA_FLASH_ATTENTION"] = "1"
    # os.environ["OLLAMA_KV_CACHE_TYPE"] = "fp16"

    class Response(BaseModel):
        question: str = Field(description="What question is appropriate to this text?")
        answer: str = Field(description="Answer to the question")
    
    sys_prompt = system_prompt = """
You are an expert dataset annotator for instruction-tuning large language models. Your task is to create high-quality question-answer pairs from provided texts for training instruct models.

Guidelines:
- Keep the question relevant and informative for learners.
- Avoid using markdown or any unnecessary formatting.
- You can ask to elaborate based on a keyword or phrase in the text.
- You can ask about the plot if the text is a story.
- Do not use overly formal language.
- Use only the information provided in the text.
- If the text states that any part of it is from Netflix, or mentions that a section is from Netflix, ignore that part and do not include it in the question or answer.
- If user specifies already created question and answer pair, find a different question and answer pair that is different from the one provided. If this is impossible use different words then the ones provided.
- Return the output strictly as a JSON with two fields: "question" and "answer".
"""
    folder = './training_datasets/raw/witcher_fandom'
    paths = os.listdir(folder)
    paths = [os.path.join(folder, p) for p in paths]
    print(f"{len(paths)} paths found")

    for _ in range(3):

        dual = OllamaInstructCurate('qwen3:8b',
                        sys_prompt,
                        Response)
        dual(paths, save_to='./training_datasets/raw/witcher_synthetic_instruct/qwen3:8b', skip=False, seed=random.randint(0, 1000))
