from cirilla.Cirilla_model import Cirilla
from cirilla.Cirilla_model import CirillaTokenizer

hf_model_id = 'AnthonyPa57/Cirilla-0.3B-4E'

model = Cirilla()

model.pull_model_from_hub(hf_model_id, inference_mode=True)
tokenizer = CirillaTokenizer(hub_url=hf_model_id)

prompts = [
    "What is notable about Ravik?",
    "Summarize all information about Novigradian Union.",
    "Who is Geralt of Rivia?"
]

for p in prompts:
    # x = tokenizer.apply_chat_template([{"role": "user", "content": p}],
    #                                 padding='do_not_pad', add_generation_prompt=True)
    # out = model.generate_kv_cache([x], termination_tokens=[tokenizer.convert_tokens_to_ids('<eos>'), tokenizer.convert_tokens_to_ids('<|user|>')])
    x = tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                    return_tensors='pt', padding='do_not_pad', add_generation_prompt=True)
    out = model.generate_naive(x.to(model.args.device), top_k=3, n_beams=3, termination_tokens=[tokenizer.convert_tokens_to_ids('<eos>'), tokenizer.convert_tokens_to_ids('<|user|>')])
    print(tokenizer.decode(out[0]))

batch_prompts = [[{"role": "user", "content": p}] for p in prompts]
x = tokenizer.apply_chat_template(batch_prompts, padding='do_not_pad', add_generation_prompt=True)
out = model.generate_kv_cache(x, termination_tokens=[tokenizer.convert_tokens_to_ids('<eos>'), tokenizer.convert_tokens_to_ids('<|user|>')])
for o in out:
    print(tokenizer.decode(o).replace('<pad>', ''))

model.clear_cache()