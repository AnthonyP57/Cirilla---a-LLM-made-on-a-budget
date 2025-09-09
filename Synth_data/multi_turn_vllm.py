import random
from pydantic import BaseModel, Field
import os
import json
# from tqdm import tqdm
# from openai import OpenAI
import os
import copy
import re
from typing import Optional, Any
import vllm
from transformers import AutoTokenizer
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, ProgressColumn
from datetime import timedelta
from pathlib import Path

llm = None
llm_model_name = None

class Response(BaseModel):
    question: str = Field(description="What question is appropriate to this text?")
    answer: str = Field(description="Answer to the question")

sys_prompt = """
You are an assistant in a multi-turn Q&A about *The Witcher* universe.

Task:
1. Read the provided content.
2. Write ONE clear, unique question about it.
3. Write ONE concise, factual answer.

Rules:
- Use only the English language.
- Stay within Witcher lore (books, games, monsters, history, characters). 
- Never mention the Netflix series, actors, or production.
- Each Q must be different from all "Already asked questions by the user".
- No duplicates, paraphrases, or overlapping questions. If stuck, pivot to a new angle.
- Keep Q&A short, precise, and directly tied to the content.

Output:
Return only ONE valid JSON object, ending with `}`:

{
    "question": "<concise question>",
    "answer": "<concise factual answer>"
}

Content:
{content}

Already asked questions by the user:
"""

new_context_prompt = """
New content has been added. Based only on this NEW CONTENT:

1. Ask ONE clear, unique question.
2. Provide ONE concise, factual answer.

Rules:
- Must be about the NEW CONTENT.
- Never mention Netflix/actors/production.
- Do not repeat or rephrase any previous or "Already asked" questions.
- If no new unique question is possible, ask a short clarification instead.
- Keep Q&A minimal and factual.

Output:
Return only ONE valid JSON object, ending with `}`:

{
    "question": "<concise question>",
    "answer": "<concise factual answer>"
}

NEW CONTENT:
{random_content}
"""

random_user_prompt1 = \
"""Ask to elaborate."""

random_user_prompt2 = \
"""Ask to clarify."""

random_user_prompt3 = \
"""Ask to rephrase the answer."""

random_user_prompt4 = \
"""Cite some of the context and ask to summarize."""

random_user_prompt5 = \
"""Cite some of the context and ask to explain."""

random_user_prompt6 = \
"""Ask for a simpler answer."""

random_user_prompt7 = \
"""Ask if a a particular fact is true or false. The stated fact has to be based on the context and has to be true."""

random_user_prompt8 = \
"""Ask if a particular fact is true or false. The stated fact has to be based on the context and has to be false."""

random_user_prompt9 = \
"""Ask a very underspecific question."""

random_user_prompt10 = \
"""Ask for an example related to the answer."""

random_user_prompt11 = \
"""Ask why the answer is important."""

random_user_prompt12 = \
"""Ask how this relates to other events or characters."""

random_user_prompt13 = \
"""Ask for a comparison with another character, place, or event."""

random_user_prompt14 = \
"""Ask the assistant to list two more related facts or characters or events."""

random_user_prompt15 = \
"""Ask to put the answer into chronological context if possible."""

random_user_prompt16 = \
"""Ask to explain the political or historical significance of the answer."""


random_user_prompts = (random_user_prompt1, random_user_prompt2,
                       random_user_prompt3, random_user_prompt4,
                       random_user_prompt5, random_user_prompt6,
                       random_user_prompt7, random_user_prompt8,
                       random_user_prompt9, random_user_prompt10,
                       random_user_prompt11, random_user_prompt12,
                       random_user_prompt13, random_user_prompt14,
                       random_user_prompt15, random_user_prompt16)

class StepEtaColumn(ProgressColumn):
    """ETA updates only when task.completed changes"""
    def __init__(self):
        super().__init__()
        self.last_completed = {}
        self.cached_eta = {}

    def render(self, task):
        # Initialize per-task tracking
        if task.id not in self.last_completed:
            self.last_completed[task.id] = -1
            self.cached_eta[task.id] = "ETA: --:--"

        # Update ETA only if completed count changed
        if task.completed != self.last_completed[task.id] and task.completed > 0:
            percent = task.completed / task.total if task.total else 0
            elapsed = task.elapsed or 0
            if percent > 0:
                eta_seconds = elapsed / percent - elapsed
                self.cached_eta[task.id] = f"ETA: {str(timedelta(seconds=int(eta_seconds)))}"
            self.last_completed[task.id] = task.completed

        return TextColumn(self.cached_eta[task.id]).render(task)

def best_effort_parse(text: str, schema: Optional[BaseModel] = None) -> dict[str, Any]:
    text = text.strip()

    try:
        data = json.loads(text)
        if schema:
            return schema(**data).model_dump()
        return data
    except Exception:
        pass

    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            candidate = match.group(0)
            data = json.loads(candidate)
            if schema:
                return schema(**data).model_dump()
            return data
    except Exception:
        pass

    candidate = re.sub(r",\s*}", "}", text)
    candidate = re.sub(r",\s*]", "]", candidate)
    try:
        data = json.loads(candidate)
        if schema:
            return schema(**data).dict()
        return data
    except Exception:
        pass

    return {"question": "", "answer": ""}

def vllm_multi_turn(paths:list[Path],
                    save_to:Path='./convos',
                    batch_size:int=256,
                    system_prompt:str=sys_prompt,
                    n_turns:int=5,
                    template:BaseModel=Response,
                    model:str="unsloth/Meta-Llama-3.1-8B-Instruct",
                    prob_chance_new_context:float=0.3):
    """
    Generate multi-turn conversations
    
    Args:
        paths (list[Path]): paths to .txt files containing texts to create multi turn question answer pairs
        save_to (Path): folder to save generated multi turn question answer pairs to. Defaults to './convos'.
        batch_size (int): batch size for the model. Defaults to 256.
        system_prompt (str): system prompt for the model. Defaults to sys_prompt.
        n_turns (int): number of turns to generate. Defaults to 5.
        template (BaseModel): template for the model. Defaults to Response.
        model (str): model to use. Defaults to "unsloth/Meta-Llama-3.1-8B-Instruct".
        prob_chance_new_context (float): probability of starting a new context for the question answer pairs. Defaults to 0.3.
    
    Returns:
        None
    """

    global llm, llm_model_name


    if llm is None or llm_model_name != model:
        llm = vllm.LLM(model=model, max_model_len=4096, gpu_memory_utilization=0.8)
        llm_model_name = model

    sampling = vllm.SamplingParams(
                max_tokens=2048,
            )

    tokenizer = AutoTokenizer.from_pretrained(model)

    # client = OpenAI(base_url=f"http://0.0.0.0:{vllm_port}/v1", api_key="dummy")

    # bar = tqdm(total=len(paths) * n_turns, desc=f"{model} conversations")
    os.makedirs(save_to, exist_ok=True)

    progress = Progress(
                        TextColumn("{task.description}"),
                        BarColumn(),
                        TextColumn("{task.completed}/{task.total}"),
                        TextColumn("{task.percentage:>3.0f}%"),
                        TimeElapsedColumn(),
                        StepEtaColumn(),
                        )
    
    writer = progress.add_task(f"{model.split('/')[1]}", total=len(paths) * n_turns)

    qa = []
    model_convo_batched = []
    contexts = []

    with progress:
        for p in paths:

            with open(p, 'r') as f:
                path_content = f.read()

            model_convo = [{'role': 'system', 'content': system_prompt.replace('{content}', path_content)}]
            model_convo_batched.append(model_convo)
            contexts.append(os.path.basename(p).split('.')[0])

        for turn in range(n_turns):
        
            batched_output = []

            for i in range(0, len(model_convo_batched), batch_size):

                chat_template_batch  = [
                                        tokenizer.apply_chat_template(
                                            prompt,
                                            tokenize=False,
                                            add_generation_prompt=True
                                        ) for prompt in model_convo_batched[i:i+batch_size]
                                        ]

                out = llm.generate(chat_template_batch, sampling_params=sampling)
                out = [best_effort_parse(o.outputs[0].text, template) for o in out]
                batched_output.extend(out)

                # batch = model_convo_batched[i:i+batch_size]
                # try:
                #     responses = [
                #         client.beta.chat.completions.parse(
                #             model=model,
                #             messages=conv,
                #             response_format=template,
                #             timeout=timeout
                #         )
                #         for conv in batch
                #     ]
                #     parsed = [resp.choices[0].message.parsed.model_dump() for resp in responses]
                
                # except:
                #     bar.write('Failed to generate response - timeout')
                #     parsed = [{'question': '', 'answer': ''} for _ in range(len(batch))]

                # batched_output.extend(parsed)

                progress.update(writer, advance=len(out))
            
            for i, b in enumerate(batched_output):
                b['context'] = contexts[i]

            qa.append(batched_output)

            for i in range(len(model_convo_batched)):

                model_convo_batched[i].append({'role':'assistant', 'content':f'Question: {batched_output[i]["question"]}\nAnswer: {batched_output[i]["answer"]}\n\n'})

                if (random.randint(0, 100) < prob_chance_new_context * 100) and (turn > 0):

                        random_path = random.choice(paths)
                        with open(random_path, 'r') as f:
                            random_content = f.read()

                        model_convo_batched[i].append({'role':'user', 'content':new_context_prompt.replace("{random_content}", random_content)})
                        contexts[i] = os.path.basename(random_path).split('.')[0]

                else:
                    model_convo_batched[i].append({'role':'user', 'content':random.choice(random_user_prompts)})


    qa_gathered = [[] for _ in range(len(paths))]

    for turn in qa:
        assert len(turn) == len(paths)

        for i in range(len(paths)):
            if (not turn[i]['question'] == '') and (not turn[i]['answer'] == ''):
                qa_gathered[i].append(turn[i])
    
    for i, q in enumerate(qa_gathered):
        if len(q) > 0:
            
            path = f'{save_to}/{os.path.basename(paths[i]).split(".")[0]}'
            path_ = copy.deepcopy(path)
            i = 1
            while os.path.exists(path_+'.json'):
                path_ = copy.deepcopy(path) + f'_{i}'
                i += 1

            with open(f'{path_}.json', 'w') as f:
                json.dump(q, f, indent=2)
