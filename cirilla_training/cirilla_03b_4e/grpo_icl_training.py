import torch
from cirilla.RL.GRPO import GRPO_DBMS, GRPO_Engine
from cirilla.Cirilla_model import get_optims, Cirilla, Args, CirillaTokenizer

if __name__ == '__main__':

    save_gen_path = './saved_grpo_answers.jsonl'
    hf_repo = 'AnthonyPa57/Cirilla-0.3B-4E-grpo-icl'

    gen_config = {'batch_size': 256, 'n_generate_with_kv_cache': 3,
                'n_generate_naive': 0, 'generate_mistral': 1,
                'store_mistral_answers' : './mistral_answers.jsonl'}

    dbms = GRPO_DBMS(
        model_hub_url="AnthonyPa57/Cirilla-0.3B-4E", 
        prompt_dataset_hub_url="AnthonyPa57/Witcher-GRPO-prompts",
        generation_config=gen_config,
        random_data_subset=15_000
    )

    tokenizer = CirillaTokenizer(hub_url='AnthonyPa57/Cirilla-0.3B-4E')

    model = Cirilla(Args(torch_compile=False))
    model.pull_model_from_hub('AnthonyPa57/Cirilla-0.3B-4E', map_device='cuda:0', force_dynamic_mask=True, force_eager=True)

    muon_opt, adam_opt = get_optims(
                                model,
                                use_muon_optim=True,
                                optim=torch.optim.AdamW,
                                lr=1e-5, weight_decay=1e-5,
                                )

    grpo_engine = GRPO_Engine(
        grpo_dbms=dbms,
        tokenizer=tokenizer,
        model=model,
        hf_repo=hf_repo,
        optimizers_dict={'muon_opt': muon_opt, 'adam_opt': adam_opt},
        beta=0.5
    )

    grpo_engine.train(
        n_epochs=3,
        batch_size=4,
        n_iter_checkpoints=1000,
        save_gen_path=save_gen_path,
        pull_optim_from_hub='AnthonyPa57/Cirilla-0.3B-4E',
    )
