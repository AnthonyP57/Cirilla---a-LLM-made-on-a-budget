import random
from pydantic import BaseModel, Field
import os
import json
from tqdm import tqdm
from openai import OpenAI
import os
import copy

class Response(BaseModel):
    question: str = Field(description="What question is appropriate to this text?")
    answer: str = Field(description="Answer to the question")

sys_prompt = \
"""You are an AI assistant engaged in a multi-turn question answering conversation.  

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

**NON-REPETITION RULE (ABSOLUTE)**:
- **You MUST NOT repeat any question that has already been asked.** This prohibition includes:
  - exact duplicates,
  - close paraphrases,
  - rephrasings, or
  - any question that seeks the same information as a previously asked question.
- Before finalizing the question, explicitly compare it against the list labeled **"Already asked questions by the user"** and the conversation history. If there is *any* semantic overlap, produce a different, clearly distinct question.
- If the new content makes it impossible to produce a non-repeating question about the same narrow topic, you must:
  - either ask a brief clarification that does **not** repeat earlier questions, **or**
  - pivot to a related but distinct angle (for example: ask about causes instead of consequences, or ask about a different character, time period, or mechanic).
- Under no circumstances output a question that duplicates earlier content, even if slightly reworded.

Other constraints:
- Keep answers informative, but concise and accurate.
- You may ask for clarification or a follow-up question — but **DO NOT ASK THE SAME QUESTION**; if you need clarification, phrase it so it is unambiguously different from prior questions.
- Do not include any preamble, sign-off, or extra commentary (e.g., "I'm ready to assist", "Here is the JSON", etc.). **Return only the JSON object** described below and nothing else.
- If the model is tempted to produce multiple candidate questions, choose one that is unique and does not overlap with the "Already asked questions" list.

Format:
Return a JSON object that matches the provided schema with the following keys and nothing else:

{"question": <the generated question>, "answer": <the corresponding answer>}

Already asked questions by the user: (DO NOT REPEAT THEM AGAIN)"""

new_context_prompt = \
"""Now, based on the NEW CONTENT below, generate a new question and answer.  

Rules:
- The question must be relevant to the NEW CONTENT.
- The answer must be factual and concise.
- You may ask a general question about The Witcher universe if it fits naturally, but you must never ask or answer about the Netflix adaptation, series, actors, or production.
- **ABSOLUTELY DO NOT repeat any question** that appears in the conversation or in the "Already asked questions by the user" list. This includes paraphrases and rephrasings. Compare your candidate question to that list and the conversation history; if there is any overlap, generate a different question.
- If the NEW CONTENT would otherwise force a repeat, ask a short clarification that does not duplicate earlier questions, or pivot to a different angle on the content.
- Ensure the new question and answer feel like a natural continuation of the ongoing conversation and are non-redundant.

NEW CONTENT:
{random_content}"""


def multi_turn(paths, save_to='./convos', batch_size=32, system_prompt=sys_prompt, n_turns_range=(2,5), template=Response, model="unsloth/Meta-Llama-3.1-8B-Instruct", prob_chance_new_context=0.3, vllm_port=8000):

    os.makedirs(save_to, exist_ok=True)

    # llm = vllm.LLM(model=model, max_model_len=32000)

    client = OpenAI(base_url=f"http://0.0.0.0:{vllm_port}/v1", api_key="dummy")
    # tokenizer = AutoTokenizer.from_pretrained(model)

    n_turns = random.randint(n_turns_range[0], n_turns_range[1])

    bar = tqdm(total=len(paths) * n_turns, desc=f"{model} conversations")

    qa = []
    model_convo_batched = []
    contexts = []

    for p in paths:

        with open(p, 'r') as f:
            path_content = f.read()

        model_convo = [{'role': 'system', 'content': system_prompt.replace('{content}', path_content)}]
        model_convo_batched.append(model_convo)
        contexts.append(os.path.basename(p).split('.')[0])

    for turn in range(n_turns):
    
        batched_output = []

        for i in range(0, len(model_convo_batched), batch_size):

            # chat_template_batch  = [
            #                         tokenizer.apply_chat_template(
            #                             prompt,
            #                             tokenize=False,
            #                             add_generation_prompt=True
            #                         ) for prompt in model_convo_batched[i:i+batch_size]
            #                         ]

            # out= llm.generate(chat_template_batch)
            # out = [best_effort_parse(o.outputs[0].text) for o in out]
            # batched_output.extend(out)

            batch = model_convo_batched[i:i+batch_size]

            responses = [
                client.beta.chat.completions.parse(
                    model=model,
                    messages=conv,
                    response_format=Response,   # enforce Pydantic schema
                    # extra_body=dict(guided_decoding_backend="outlines"),
                )
                for conv in batch
            ]

            parsed = [resp.choices[0].message.parsed.model_dump() for resp in responses]
            batched_output.extend(parsed)

            bar.update(len(parsed))
        
        for i, b in enumerate(batched_output):
            b['context'] = contexts[i]

        qa.append(batched_output)

        for i in range(len(model_convo_batched)):

            model_convo_batched[i].append({'role':'assistant', 'content':f'Question: {batched_output[i]["question"]}\nAnswer: {batched_output[i]["answer"]}\n\n'})

            if (random.randint(0, 100) < prob_chance_new_context * 100) and (turn > 0):

                    random_path = random.choice(paths)
                    with open(random_path, 'r') as f:
                        random_content = f.read()

                    model_convo_batched[i].append({'role':'user', 'content':new_context_prompt.format(random_content=random_content)})
                    contexts[i] = os.path.basename(random_path).split('.')[0]


    qa_gathered = [[] for _ in range(len(paths))]

    for turn in qa:
        assert len(turn) == len(paths)

        for i in range(len(paths)):
            qa_gathered[i].append(turn[i])
    
    for i, q in enumerate(qa_gathered):
            
        path = f'{save_to}/{os.path.basename(paths[i]).split(".")[0]}'
        path_ = copy.deepcopy(path)
        i = 1
        while os.path.exists(path_+'.json'):
            path_ = copy.deepcopy(path) + f'_{i}'
            i += 1

        with open(f'{path_}.json', 'w') as f:
            json.dump(q, f, indent=2)

if __name__ == "__main__":

    paths = os.listdir('./training_datasets/raw/witcher_fandom')
    paths = [os.path.join('./training_datasets/raw/witcher_fandom', p) for p in paths][:10]

    for model in ["unsloth/Llama-3.2-3B-Instruct-bnb-4bit"]:
        for _ in range(3):
        
            multi_turn(paths, save_to=f'./training_datasets/raw/synth_multi_round/{model.split("/")[1]}', model=model)