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
