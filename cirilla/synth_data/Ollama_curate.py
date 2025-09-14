import ollama
from pydantic import BaseModel
import time
import os
import json
from tqdm import tqdm
from typing import Optional
import copy
from .multi_turn_vllm import random_user_prompts
import random
from pathlib import Path

"""
Ollama options
    numa: Optional[bool] = None
    num_ctx: Optional[int] = None
    num_batch: Optional[int] = None
    num_gpu: Optional[int] = None
    main_gpu: Optional[int] = None
    low_vram: Optional[bool] = None
    f16_kv: Optional[bool] = None
    logits_all: Optional[bool] = None
    vocab_only: Optional[bool] = None
    use_mmap: Optional[bool] = None
    use_mlock: Optional[bool] = None
    embedding_only: Optional[bool] = None
    num_thread: Optional[int] = None

    # runtime options
    num_keep: Optional[int] = None
    seed: Optional[int] = None
    num_predict: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    tfs_z: Optional[float] = None
    typical_p: Optional[float] = None
    repeat_last_n: Optional[int] = None
    temperature: Optional[float] = None
    repeat_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    mirostat: Optional[int] = None
    mirostat_tau: Optional[float] = None
    mirostat_eta: Optional[float] = None
    penalize_newline: Optional[bool] = None
    stop: Optional[Sequence[str]] = None
"""

class OllamaCurate:
    def __init__(self, model, system_prompt, response_template:BaseModel):
        self.system_prompt = {'role': 'system', 'content': system_prompt}
        self.model = model
        self.response_template = response_template
        self.convo = {}
    
    @staticmethod
    def try_repair_json(text: str) -> str:
        text = text.strip()
        try:
            json.loads(text)
            return text
        except:
            pass

        if text.count('{') > text.count('}'):
            text += "}"
        if text.count('[') > text.count(']'):
            text += "]"
        if text.count('"') % 2 != 0:
            text += '"'

        try:
            json.loads(text)
            return text
        except:
            return json.dumps({"summary": text[:1000]})


    def single_pass_summary(
        self,
        paths: list[Path],
        save_to: Path = "./summaries",
        seed: int = 42,
        num_predict: int = 4096,  # allow longer completion
        use_response_template: bool = False,
    ):
        """
        Summarize a list of files using a single pass of the model
        
        Args:
            paths (list[Path]): List of paths of files to summarize.
            save_to (Path): Directory to save summaries. Defaults to './summaries'.
            seed (int): Random seed for reproducibility. Defaults to 42.
            num_predict (int): Number of tokens to predict. Defaults to 4096.
            use_response_template (bool: Use a response template. Defaults to False.
        
        Returns:
            None
        """
        os.makedirs(save_to, exist_ok=True)

        def _call_model(prompt: str):
            if use_response_template and hasattr(self, "response_template") and self.response_template is not None:
                resp = ollama.chat(
                    model=self.model,
                    messages=[self.system_prompt, {"role": "user", "content": prompt}],
                    format=self.response_template.model_json_schema(),
                    options={"num_predict": num_predict, "seed": seed},
                )
                raw = resp.message.content.strip()
                raw = self.try_repair_json(raw)
                resp_valid = self.response_template.model_validate_json(raw)
                return resp_valid.summary
            else:
                resp = ollama.chat(
                    model=self.model,
                    messages=[self.system_prompt, {"role": "user", "content": prompt}],
                    options={"num_predict": num_predict, "seed": seed},
                )
                return resp.message.content.strip()

        bar = tqdm(total=len(paths), desc="Processing files", unit="file")

        for path in paths:
            basename = os.path.basename(path).split(".")[0]
            final_path = os.path.join(save_to, f"{basename}.txt")

            if not os.path.exists(final_path):
                with open(path, "r", encoding="utf-8") as f:
                    full_text = f.read()

                prompt = \
f"""Summarize the TEXT into one detailed, single-paragraph summary.

Rules:
- Include ALL concrete details: names, places, monsters, dates, events, factions, motives, artifacts, numbers, outcomes.
- Stay only within Witcher lore (books, games, monsters, history, characters). Ignore anything about Netflix, actors, or production.
- Do not add or invent facts. Use only what is in the text.
- Be factual, clear, and exhaustive while keeping one coherent paragraph.
- If the text has almost no real facts, instead list up to 12 observable items (comma-separated).

TEXT:
{full_text}

"""
                try:
                    final_summary = _call_model(prompt).strip()
                    final_summary = final_summary.replace("{", "").replace("}", "")
                except Exception as e:
                    print(f"Model error on {path}: {e}")
                    final_summary = full_text[:1000]  # truncated fallback

                with open(final_path, "w", encoding="utf-8") as f:
                    f.write(final_summary)

            bar.update(1)

        os.system(f"ollama stop {self.model}")

    def dynamic_hierarchical_summary(
        self,
        paths: list[Path],
        save_to: Path = './summaries',
        chunk_lines: int = 100, # number of lines per chunk
        seed: int = 42,
        num_predict: int = 2048, # max number of tokens
        max_words_summary: int = 500, # target maximum words per summary block
        use_response_template: bool = False
    ):
        """
        Summarize a list of files hierarchically. Meaning that for a text divided into chunks we subsequently generate a summary for each chunk, and then generate a final summary based on the summaries of the chunks.
        
        Args:
            paths (list[Path]): List of paths of files to summarize.
            save_to (Path): Directory to save summaries. Defaults to './summaries'.
            chunk_lines (int): Number of lines per chunk. Defaults to 100.
            seed (int): Random seed for reproducibility. Defaults to 42.
            num_predict (int): Number of tokens to predict. Defaults to 2048.
            max_words_summary (int): Maximum number of words per summary block and the final summary. Defaults to 500.
            use_response_template (bool): Use a response template. Defaults to False.
        
        Returns:
            None
        """
        os.makedirs(save_to, exist_ok=True)

        def _call_model(prompt: str):
            if use_response_template and hasattr(self, 'response_template') and self.response_template is not None:
                resp = ollama.chat(
                    model=self.model,
                    messages=[self.system_prompt, {'role': 'user', 'content': prompt}],
                    format=self.response_template.model_json_schema(),
                    options={'num_predict': num_predict, 'seed': seed}
                )

                raw = resp.message.content.strip()
                raw = self.try_repair_json(raw)

                resp_valid = self.response_template.model_validate_json(raw)
                data = resp_valid.model_dump()
                return data['summary']

            else:
                resp = ollama.chat(
                    model=self.model,
                    messages=[self.system_prompt, {'role': 'user', 'content': prompt}],
                    options={'num_predict': num_predict, 'seed': seed}
                )
                raw_text = resp.message.content.strip()

                return raw_text


        def summarize_chunk(chunk_text: str, context_summary: Optional[str] = None, target_words: int = max_words_summary):
            if context_summary:
                prompt = f"""
    You are an objective, concise summarizer. Use the EXISTING SUMMARY together with the NEW CHUNK to produce a single, self-contained updated summary.
    Rules:
    - Do NOT invent facts or add information not present in the EXISTING SUMMARY or NEW CHUNK.
    - Prioritize new, important facts introduced by the NEW CHUNK; integrate them into the EXISTING SUMMARY and remove redundancies.
    - If the NEW CHUNK adds no substantive facts, look for any noteworthy detail in the NEW CHUNK and highlight it in one sentence; if there is literally nothing (only noise: whitespace, headers, repeated filler), instead output a compact list (comma-separated, max 8 items) of tokens/phrases or types of content that do appear (e.g. "table of contents, URL, image caption").
    - Be factual, concise and clear. Return a single paragraph (no headings, no metadata).
    - Keep length ≤ {target_words} words and avoid repetition.

    EXISTING SUMMARY:
    {context_summary}

    NEW CHUNK:
    {chunk_text}
    """
            else:
                prompt = f"""
    You are an objective, concise summarizer. Summarize the TEXT into a single, self-contained paragraph.
    Rules:
    - Preserve concrete facts, key points, entities, dates, and actions. Do NOT invent details.
    - If the text contains no substantive facts (only noise or empty lines), find the most salient thing present and describe it in one sentence; if there truly is nothing useful, output a compact list (comma-separated, max 8 items) of observable elements or token types (e.g. "heading, URL, bullet list").
    - Prefer clarity and specificity. Return only the summary (no headings or explanations).
    - Keep length ≤ {target_words} words.
    TEXT:
    {chunk_text}
    """

            return _call_model(prompt).strip()

        def _chunks_from_file(path, lines_per_chunk):
            with open(path, 'r', encoding='utf-8') as f:
                all_lines = f.read().splitlines()

            for i in range(0, len(all_lines), lines_per_chunk):
                chunk_lines_list = all_lines[i:i + lines_per_chunk]
                yield "\n".join(chunk_lines_list)

        bar = tqdm(total=len(paths), desc="Processing files", unit="file")
        for file_idx, path in enumerate(paths):
            basename = os.path.basename(path).split('.')[0]

            if not os.path.exists(f"{save_to}/{basename}.txt"):

                summary_blocks = []
                current_summary = None
                processed_chunks = 0

                for chunk_idx, chunk_text in enumerate(_chunks_from_file(path, chunk_lines)):
                    # skip empty chunk (all whitespace)
                    if not chunk_text.strip():
                        continue

                    # summarize (current_summary + chunk_text) into a new current_summary
                    try:
                        if current_summary is None:
                            new_summary = summarize_chunk(chunk_text, context_summary=None)
                        else:
                            # include the running summary as context
                            new_summary = summarize_chunk(chunk_text, context_summary=current_summary)
                    except Exception as e:
                        print(f"Model error on chunk {processed_chunks} ({path} chunk {chunk_idx}): {e}")
                        # fallback: just store the chunk itself as a "summary" to avoid data loss
                        new_summary = chunk_text[:1000]  # truncated fallback

                    block_id = f"{file_idx:03d}_{chunk_idx:03d}"
                    summary_blocks.append({'id': block_id, 'summary': new_summary, 'source': f"{basename}:{chunk_idx}"})
                    current_summary = new_summary  # update running summary
                    processed_chunks += 1

                    # elapsed = time.time() - start
                    # print(f"Processed chunk #{processed_chunks} (block {block_id}) — elapsed {elapsed/60:.2f} min", end='\r')
                if len(summary_blocks) > 1:
                    cohesive_blocks = [b['summary'] for b in summary_blocks]
                    cohesive_blocks = "\n\n".join(cohesive_blocks)

                    prompt = f"""
    You are an objective, concise summarizer. You are given BLOCK SUMMARIES that each describe parts of a larger document. Produce ONE cohesive, comprehensive summary of the entire document.
    Rules:
    - Integrate facts across blocks, remove redundancy, resolve obvious repetition, and create a smooth, logical single-paragraph narrative.
    - Do NOT hallucinate or add information not present in the blocks. If a fact appears in multiple blocks, mention it once.
    - If many blocks are sparse, prioritize blocks that contain specific facts or named entities; if none contain substantive facts, list (comma-separated) the observable items across blocks (max 12).
    - Return only the final summary (no headings). Aim for ≈ {max_words_summary} words; do not exceed that by more than ~20%.
    BLOCK SUMMARIES:
    {cohesive_blocks}
    """

                    final_summary = _call_model(prompt).strip()
                else:
                    final_summary = summary_blocks[0]['summary']

                final_path = os.path.join(save_to, f"{basename}.txt")
                with open(final_path, 'w', encoding='utf-8') as f:
                    f.write(final_summary)

            bar.update(1)

        os.system(f'ollama stop {self.model}')

    def __call__(self, paths:list[Path], save_to:Path='./example', seed:int=42, checkpoint:int=10, skip:bool=True) -> None:
        """
        Generate instructions from files.

        Args:
            paths (list[Path]): List of paths to files to summarize.
            save_to (Path): Directory to save summaries. Defaults to './example'.
            seed (int): Random seed for reproducibility. Defaults to 42.
            checkpoint (int): Number of files to process before saving a checkpoint. Defaults to 10.
            skip (bool): Skip files that already have summaries. Defaults to True. Else, add index to file name.

        Returns:
            None
        """

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
                        json.dump(v, f, indent=2)
                
                self.convo = {}

        os.system(f'ollama stop {self.model}')

    def multi_turn(self,
        paths: list[Path],
        save_to: Path='./convos',
        bar: tqdm=None,
        n_turns_range: tuple[int,int]=(2,5),
        seed: int=random.randint(0, 1000),
        prob_chance_new_context: float=0.3):
        """
        Generate multi turn question answer pairs from files.

        Args:
            paths (list[Path]): List of paths to files to summarize.
            save_to (Path): Directory to save summaries. Defaults to './example'.
            bar (tqdm.tqdm): tqdm bar to track progress. Defaults to None.
            n_turns_range (tuple[int,int]): Range of number of turns to generate. Defaults to (2, 5).
            seed (int): Random seed for reproducibility. Defaults to 42.
            prob_chance_new_context (float): Probability of starting a new context for the question answer pairs. Defaults to 0.3.

        Returns:
            None
        """
    
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

        new_content_prompt = \
    """Now, based on the NEW CONTENT below, generate a new question and answer.  

    Rules:
    - The question must be relevant to the NEW CONTENT.
    - The answer must be factual and concise.
    - You may ask a general question about The Witcher universe if it fits naturally, but you must never ask or answer about the Netflix adaptation, series, actors, or production.
    - Do not repeat earlier questions unless the NEW CONTENT explicitly makes it necessary.
    - Ensure the new question and answer feel like a natural continuation of the ongoing conversation.  

    NEW CONTENT:
    {random_content}
    """

        not_failed = 0
        n_failed = 0

        for p in paths:

            basename = os.path.basename(p).split('.')[0]

            with open(p, 'r') as f:
                path_content = f.read()

            clean_convo = []
            model_convo = [{'role': 'system', 'content': sys_prompt.format(content=path_content)}]
            random_path = None

            for i in range(random.randint(n_turns_range[0], n_turns_range[1])):

                if (random.randint(0, 100) < prob_chance_new_context * 100) and (i > 0):
                    random_path = random.choice(paths)
                    with open(random_path, 'r') as f:
                        random_content = f.read()

                    model_convo.append({'role':'user', 'content':new_content_prompt.replace('{random_content}', random_content)})
                    
                else:
                    model_convo.append({'role':'user', 'content':random.choice(random_user_prompts)})
                
                response = ollama.chat(
                    model=self.model,
                    messages = model_convo,
                    format=self.response_template.model_json_schema(),
                    options={
                        'num_predict': 512, # max num tokens
                        'seed': seed
                    }
                )
                try:
                    response = self.response_template.model_validate_json(response.message.content)
                    response = response.model_dump()
                    model_convo.append({'role':'assistant', 'content':f"Question: {response['question']}\nAnswer: {response['answer']}"})
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
            
        os.system(f'ollama stop {self.model}')
