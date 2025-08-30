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
