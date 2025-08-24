import ollama
import random
from pydantic import BaseModel, Field
import os
import json
import copy

class Response(BaseModel):
    question: str = Field(description="What question is appropriate to this text?")
    answer: str = Field(description="Answer to the question")

sys_prompt = """You are an AI assistant engaged in a multi-turn question answering conversation.  

Your task:
- Read and understand the provided content.
- Generate a clear and relevant **question** about the content.
- Provide a factual and concise **answer** to that question.  

Content to analyze:
{content}

Conversation rules:
- You may sometimes ask a very general question about "The Witcher" universe (books, lore, characters, history, monsters, or games), but you must **never** ask or answer about the Netflix adaptation, series, actors, or production.
- If new content is introduced, you must adapt your question and answer to that **new context**, ensuring it feels like a natural continuation of the conversation.
- Each response should stay grounded in either the current context or the newly provided content.
- Do not repeat earlier questions unless the new context requires it.
- Keep answers informative, but concise and accurate.  
- You may ask for clarification or a follow-up question. DO NOT ASK THE SAME QUESTION, you may ask to clarify or rephrase instead.
- If the context is very vague or confusing you can ask about some obvious fact or element present in the context.

Format:
Return a JSON object that matches the provided schema with the following keys:
- "question": the generated question
- "answer": the corresponding answer

Already asked questions: (DO NOT REPEAT THEM AGAIN)
"""


def multi_turn(paths, save_to='./convos', bar=None, system_prompt=sys_prompt, n_turns_range=(2,5), template=Response, model='llama3.2:3b', seed=random.randint(0, 1000), prob_chance_new_context=0.3):

    not_failed = 0
    n_failed = 0

    for p in paths:

        basename = os.path.basename(p).split('.')[0]

        with open(p, 'r') as f:
            path_content = f.read()

        clean_convo = []
        model_convo = [{'role': 'system', 'content': system_prompt.format(content=path_content)}]
        random_path = None

        for i in range(random.randint(n_turns_range[0], n_turns_range[1])):

            if (random.randint(0, 100) < prob_chance_new_context * 100) and (i > 0):
                random_path = random.choice(paths)
                with open(random_path, 'r') as f:
                    random_content = f.read()

                model_convo.append({'role':'user', 'content':f"""Now, based on the NEW CONTENT below, generate a new question and answer.  

Rules:
- The question must be relevant to the NEW CONTENT.
- The answer must be factual and concise.
- You may ask a general question about The Witcher universe if it fits naturally, but you must never ask or answer about the Netflix adaptation, series, actors, or production.
- Do not repeat earlier questions unless the NEW CONTENT explicitly makes it necessary.
- Ensure the new question and answer feel like a natural continuation of the ongoing conversation.  

NEW CONTENT:
{random_content}"""})
            
            response = ollama.chat(
                model=model,
                messages = model_convo,
                format=template.model_json_schema(),
                options={
                    'num_predict': 512, # max num tokens
                    'seed': seed
                }
            )
            try:
                response = template.model_validate_json(response.message.content)
                response = response.model_dump()
                model_convo[0]['content'] += f'\nQuestion: {response["question"]} Answer: {response["answer"]}'
                response['context'] = basename if random_path is None else os.path.basename(random_path).split('.')[0]
                if not (response['question'] == "" or response['answer'] == ""):
                    clean_convo.append(response)
                not_failed += 1
            except:
                n_failed += 1
                print(f'failed on {p} failed:not failed {n_failed}:{not_failed}')

        if len(clean_convo) > n_turns_range[0]:

            clean_convo = tuple(clean_convo)

            basename_ = copy.copy(basename)
            i = 0
            while True:
                if os.path.exists(f'{save_to}/{basename_}.json'):
                    basename_ = copy.copy(basename) + str(i)
                    i += 1
                else:
                    break
            
            os.makedirs(save_to, exist_ok=True)

            with open(f'{save_to}/{basename_}.json', 'w') as f:
                json.dump(clean_convo, f)

            if bar is not None:
                bar.update(1)
        
    os.system(f'ollama stop {model}')


if __name__ == "__main__":
    from tqdm import tqdm

    paths = os.listdir('./training_datasets/raw/witcher_fandom')
    paths = [os.path.join('./training_datasets/raw/witcher_fandom', p) for p in paths]

    bar = tqdm(total=3*3*len(paths))
    for model in ['qwen3:8b', 'phi4', 'llama3.1:8b']:
        for _ in range(3):
        
            multi_turn(paths, save_to=f'./training_datasets/raw/synth_multi_round/{model}', bar=bar, model=model)