# How to use

## Hierarchical dynamic summary
```python
import os
from Ollama_curate import OllamaCurate

if __name__ == '__main__':
    
    from pydantic import BaseModel, Field
    import os
    import random

    class Response(BaseModel):
        summary: str = Field(description="Summary of the text, without the thinking process and without any introduction. Provide only pure summary, be expressive but stick to the maximum number of words that were provided.")

    paths = os.listdir('./training_datasets/raw/witcher_texts')
    paths = [os.path.join('./training_datasets/raw/witcher_texts', p) for p in paths]

    print(f"{len(paths)} paths found")

    for m in ['llama3.2:3b', 'llama3.1:8b', 'qwen3:8b']:

        model = OllamaCurate(m,
                            "",
                            Response
                            )
        
        model.dynamic_hierarchical_summary(paths, save_to=f'./training_datasets/raw/synth_sumarries/witcher_texts/{m}', seed=random.randint(0, 1000), use_response_template=True)

```

## Multi turn ollama
```python
from Ollama_curate import OllamaCurate
from pydantic import BaseModel, Field
import os
import random

if __name__ == "__main__":
    from tqdm import tqdm

    class Response(BaseModel):
        question: str = Field(description="What question is appropriate to this text?")
        answer: str = Field(description="Answer to the question")

    paths = os.listdir('./training_datasets/raw/witcher_fandom')
    paths = [os.path.join('./training_datasets/raw/witcher_fandom', p) for p in paths]

    bar = tqdm(total=3*3*len(paths))
    for model in ['qwen3:8b', 'phi4', 'llama3.1:8b']:
        for _ in range(3):

            ol = OllamaCurate(model, "", Response)
            ol.multi_turn(paths, save_to=f'./training_datasets/raw/synth_multi_round/{model}', bar=bar, seed=random.randint(0, 1000))

```

## Synthetic instructions
```python
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

    dual = OllamaCurate('qwen3:8b',
                    sys_prompt,
                    Response)
    dual(paths, save_to='./training_datasets/raw/witcher_synthetic_instruct/qwen3:8b', skip=False, seed=random.randint(0, 1000))
```

## Ollama summarizer
```python
import os
import random
from Ollama_curate import OllamaCurate

if __name__ == "__main__":
    from pydantic import BaseModel, Field

    class Response(BaseModel):
        summary: str = Field(description="Final summary of the entire text. Pure summary only, no introduction or reasoning.")

    paths = os.listdir("./training_datasets/raw/async_fandom")
    paths = [os.path.join("./training_datasets/raw/async_fandom", p) for p in paths]

    print(f"{len(paths)} paths found")

    for m in ["granite3.1-moe:3b"]:
        model = OllamaCurate(m, "", Response)
        model.single_pass_summary(
            paths,
            save_to=f"./training_datasets/raw/async_summaries/{m}",
            seed=random.randint(0, 1000),
            use_response_template=True,
        )

```

## Synthetic resoning dataset
```python
from reson_gym_synthetic import get_synth_resoning_dataset

out_path = './training_datasets/mid_training/reason_gym_synth.jsonl'
n_samples = 400

get_synth_resoning_dataset(out_path, n_samples)
```

## Remove duplicated synthetic instructions
```python
from rm_duplicate_instruct import main

main_dir = './training_datasets/raw/witcher_synthetic_instruct'
save_to = './training_datasets/domain_training/witcher_synthetic_instruct.jsonl'

rm_duplicate_instructs(main_dir, save_to)
```

## Turn fandom scraped instructions into conversations
```python
form witcher_instruct_gather import instructions_into_conv

input_path = './training_datasets/raw/async_fandom_instruct'
out_path = './training_datasets/domain_training/witcher_instruct.jsonl'

instructions_into_conv(input_path, out_path)
```

## Fandom scraper
```python
from fandom_scraper import scrape_fandom

in_path = Path("./training_datasets/raw/witcher_json")
out_path = Path("./training_datasets/raw/async_fandom")
instruct_path = Path("./training_datasets/raw/async_fandom_instruct")

scrape_fandom(in_path, out_path, instruct_path)
```

