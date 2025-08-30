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
