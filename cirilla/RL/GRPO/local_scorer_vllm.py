import json
import copy
import os
from typing import List, Any, Dict
from pathlib import Path
from vllm import LLM, SamplingParams
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from mistralai import Mistral 

MODEL_NAME = "mistralai/Ministral-3-3B-Instruct-2512"
MAX_MODEL_LEN = 1024
GPU_UTILIZATION = 0.9
MAX_RETRIES = 3
CONTEXT_LENGTH_THRESHOLD = 350
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "your_api_key_here") 

SYS_PROMPT = """You are an expert evaluator for a RAG (Retrieval Augmented Generation) system. 
Your task is to score an Answer based strictly on a provided Context and the User Prompt.

Output a valid JSON object with the following structure:
```json
{
    "reasoning": <str, very short explanation why did you score the answer in such a way, max 30 words>,
    "grammar": <int, score this answer from 0 to 5 for grammar>,
    "logic": <int, score this answer from 0 to 10 based on the logic, e.g. if there are repetitions or the answer doesn't make sense or isn't grounded in the context>,
    "encompassment": <int, the answer should answer the question as much as the information can be found in the context, if the answer doesn't make sense or answer the question then 0, score this metric from 0 to 5>
}
```"""

def load_jsonl(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl(data: List[Dict], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

def parse_model_output(text: str) -> Dict[str, Any]:
    text_clean = text.strip()
    
    try:
        return json.loads(text_clean)
    except json.JSONDecodeError:
        pass

    text_clean = text_clean.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text_clean)
    except json.JSONDecodeError:
        pass

    start = text_clean.find('{')
    end = text_clean.rfind('}')
    
    if start != -1 and end != -1 and start < end:
        candidate = text_clean[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None

def call_mistral_api(client, user_content):
    try:
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API Error: {e}")
        return None

def run_evaluation(input_file: str, output_file: str, batch_size: int = 256, id_col_name:str='id'):
    try:
        data = load_jsonl(input_file)
    except FileNotFoundError:
        print("Input file not found.")
        return

    if not data:
        print("No data to process.")
        return
    
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)

    print(f"Loading Model {MODEL_NAME}...")
    llm = LLM(
        model=MODEL_NAME, 
        max_model_len=MAX_MODEL_LEN, 
        gpu_memory_utilization=GPU_UTILIZATION, 
        trust_remote_code=True
    )
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(temperature=0.2, max_tokens=256)

    local_processing_queue = []
    api_processing_queue = []
    
    for idx, entry in enumerate(data):
        user_content = (
            f"Based solely on this context:\n{entry.get('context', '')}\n\n"
            f"And this prompt:\n{entry.get('prompt', '')}\n\n"
            f"Assess this answer:\n{entry.get('answer', '')}"
        )
        
        context_len = len(entry.get('context', '').split())
        if context_len > CONTEXT_LENGTH_THRESHOLD:
            api_processing_queue.append((idx, user_content))
        else:
            conversation = [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": user_content}
            ]
            text_prompt = tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True)
            local_processing_queue.append((idx, text_prompt))

    results = [None] * len(data) # Pre-allocate results to maintain order
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(), 
    ) as progress:
        
        local_prompts = [p for _, p in local_processing_queue]
        local_indices_map = [i for i, _ in local_processing_queue]
        
        task = progress.add_task("Scoring (Local)...", total=len(local_prompts))

        for i in range(0, len(local_prompts), batch_size):
            batch_indices = list(range(i, min(i + batch_size, len(local_prompts))))
            active_indices = copy.deepcopy(batch_indices)
            
            for attempt in range(MAX_RETRIES + 1):
                if not active_indices: break

                current_prompts = [{"prompt_token_ids": local_prompts[idx]} for idx in active_indices]
                outputs = llm.generate(current_prompts, sampling_params=sampling_params, use_tqdm=False)
                
                next_retry_indices = []

                for j, output in enumerate(outputs):
                    local_idx = active_indices[j]
                    global_idx = local_indices_map[local_idx]
                    
                    raw_text = output.outputs[0].text
                    parsed_json = parse_model_output(raw_text)

                    if parsed_json:
                        entry = {id_col_name: data[global_idx].get(id_col_name)}
                        entry['score'] = parsed_json
                        try:
                            entry['total_score'] = (
                                float(parsed_json.get('grammar', 0)) + 
                                float(parsed_json.get('logic', 0)) + 
                                float(parsed_json.get('encompassment', 0))
                            )
                        except: entry['total_score'] = 0.0
                        results[global_idx] = entry
                    else:
                        if attempt < MAX_RETRIES:
                            next_retry_indices.append(local_idx)
                        else:
                            entry = {id_col_name: data[global_idx].get(id_col_name)}
                            entry['score'] = {"error": "parsing_failed", "raw": raw_text}
                            entry['total_score'] = 0.0
                            results[global_idx] = entry

                active_indices = next_retry_indices
            progress.update(task, advance=len(batch_indices))

        if api_processing_queue:
            api_task = progress.add_task("Scoring (API)...", total=len(api_processing_queue))
            for global_idx, user_content in api_processing_queue:
                
                raw_text = call_mistral_api(mistral_client, user_content)
                parsed_json = parse_model_output(raw_text) if raw_text else None
                
                entry = {id_col_name: data[global_idx].get(id_col_name)}
                if parsed_json:
                    entry['score'] = parsed_json
                    try:
                        entry['total_score'] = (
                            float(parsed_json.get('grammar', 0)) + 
                            float(parsed_json.get('logic', 0)) + 
                            float(parsed_json.get('encompassment', 0))
                        )
                    except: entry['total_score'] = 0.0
                else:
                    entry['score'] = {"error": "api_failed", "raw": raw_text}
                    entry['total_score'] = 0.0
                
                results[global_idx] = entry
                progress.update(api_task, advance=1)

    clean_results = [r for r in results if r is not None]
    save_jsonl(clean_results, output_file)

if __name__ == "__main__":
    test_input = "input_eval.jsonl"
    if not Path(test_input).exists():
        with open(test_input, 'w') as f:
            for _ in range(1000):
                dummy = {
                    "id": 0,
                    "context": "Geralt hates portals. " * 100,
                    "prompt": "What does Geralt think of portals?",
                    "answer": "He loves them."
                }
                f.write(json.dumps(dummy) + "\n")

            for _ in range(10):
                dummy_long = {
                    "id": 1,
                    "context": "Long Context " * 1000,
                    "prompt": "Test",
                    "answer": "Test"
                }
                f.write(json.dumps(dummy_long) + "\n")
            
    run_evaluation(input_file=test_input, output_file="scored_output.jsonl")
