import os
import random
from typing import List
from tqdm import tqdm
from vllm import LLM, SamplingParams
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
import torch
torch.set_float32_matmul_precision("high")

def single_pass_summary_vllm(
    model_name: str,
    paths: List[str],
    save_to: str = "./summaries",
    seed: int = 42,
    max_tokens: int = 8192,
    batch_size: int = 16,
):
    os.makedirs(save_to, exist_ok=True)

    # vLLM engine
    llm = LLM(model=model_name, tensor_parallel_size=1, seed=seed, gpu_memory_utilization=0.85)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
    )

    def build_prompt(text: str) -> str:
        return f"""Summarize the TEXT into one detailed, single-paragraph summary.

Rules:
- Include ALL concrete details: names, places, monsters, dates, events, factions, motives, artifacts, numbers, outcomes.
- Stay only within Witcher lore (books, games, monsters, history, characters). Ignore anything about Netflix, actors, or production.
- Do not add or invent facts. Use only what is in the text.
- Be factual, clear, and exhaustive while keeping one coherent paragraph.
- If the text has almost no real facts, instead list up to 12 observable items (comma-separated).

TEXT:
{text}

"""

    # filter only unsummarized files
    jobs = []
    for path in paths:
        basename = os.path.basename(path).split(".")[0]
        final_path = os.path.join(save_to, f"{basename}.txt")
        if not os.path.exists(final_path):
            jobs.append((path, final_path))

    with Progress(
        SpinnerColumn(),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TextColumn("[bold blue]{task.fields[status]}"),
        TimeRemainingColumn(),
    ) as progress:

        task = progress.add_task(
            "[cyan]Summarizing Witcher texts...", 
            total=len(jobs), 
            status="starting"
        )

        for i in range(0, len(jobs), batch_size):
            batch = jobs[i:i + batch_size]
            texts = []
            for path, _ in batch:
                with open(path, "r", encoding="utf-8") as f:
                    texts.append(f.read())

            prompts = [build_prompt(t) for t in texts]

            progress.update(task, status=f"Batch {i//batch_size+1}/{(len(jobs)+batch_size-1)//batch_size}")

            try:
                outputs = llm.generate(prompts, sampling_params=sampling_params)
                summaries = [out.outputs[0].text.strip() for out in outputs]
            except Exception as e:
                print(f"Model error on batch starting with {batch[0][0]}: {e}")
                summaries = [t[:1000] for t in texts]  # truncated fallback

            for (path, final_path), summary in zip(batch, summaries):
                with open(final_path, "w", encoding="utf-8") as f:
                    f.write(summary)
            
            progress.update(task, advance=len(batch))

        progress.update(task, status="done")


if __name__ == "__main__":
    paths = os.listdir("./training_datasets/raw/async_fandom")
    paths = [os.path.join("./training_datasets/raw/async_fandom", p) for p in paths]

    print(f"{len(paths)} paths found")

    model_name = "unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit"
    single_pass_summary_vllm(
        model_name,
        paths,
        save_to=f"./training_datasets/raw/async_summaries/{model_name.split('/')[1]}",
        seed=random.randint(0, 1000),
        batch_size=16,
    )
