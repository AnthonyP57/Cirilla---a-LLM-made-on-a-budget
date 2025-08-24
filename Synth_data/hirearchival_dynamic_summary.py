import os
import time
import json
from typing import List, Optional
from Ollama_create_instruct import OllamaInstructCurate
import ollama
from tqdm import tqdm

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
        return json.dumps({"summary": text[:1000]})  # truncated

def dynamic_hierarchical_summary(
    self,
    paths: List[str],
    save_to: str = './summaries',
    chunk_lines: int = 100, # number of lines per chunk (your "100 lines")
    seed: int = 42,
    num_predict: int = 2048, # max number of tokens
    max_words_summary: int = 200, # target maximum words per summary block
    use_response_template: bool = False
):
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
            raw = try_repair_json(raw)

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

OllamaInstructCurate.dynamic_hierarchical_summary = dynamic_hierarchical_summary


if __name__ == '__main__':
    
    from pydantic import BaseModel, Field
    import os
    import random

    class Response(BaseModel):
        summary: str = Field(description="Summary of the text, without the thinking process")

    paths = os.listdir('./training_datasets/raw/witcher_fandom')
    paths = [os.path.join('./training_datasets/raw/witcher_fandom', p) for p in paths]

    print(f"{len(paths)} paths found")

    for m in ['llama3.1:8b', 'mistral-small3.2:24b', 'qwen3:8b']:

        model = OllamaInstructCurate(m,
                                    "",
                                    Response)
        
        model.dynamic_hierarchical_summary(paths, save_to=f'./training_datasets/raw/synth_sumarries/fandom/{m}', seed=random.randint(0, 1000), use_response_template= m=='qwen3:8b')