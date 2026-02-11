from cirilla.RL.GRPO.generate_answers import CirillaResponseGenerator, CirillaSampler
from datasets import load_dataset
from cirilla.RL.GRPO.local_scorer_vllm import run_evaluation
from torch.utils.data import IterableDataset
import sqlite3
import torch
import json
import tempfile
import os
from pathlib import Path

class GRPO_DBMS:
    def __init__(self, model_hub_url:str, prompt_dataset_hub_url:str, 
                generation_config:dict,
                local_folder:str='./GRPO_data', max_len:int=2048):
        
        self.model_hub_url = model_hub_url
        self.prompt_dataset = load_dataset(prompt_dataset_hub_url, split='train[:6]')
        self.generation_config = generation_config

        self.crg = CirillaResponseGenerator(model_hub_url)
        self.sampler = CirillaSampler(self.crg)

        Path(local_folder).mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(f'{local_folder}/grpo.db')
        self.sql_cursor = self.conn.cursor()
        self.local_folder = local_folder
        self.max_len = max_len

        self.sql_cursor.execute(
            """CREATE TABLE IF NOT EXISTS Prompt_Dataset (
                id INTEGER PRIMARY KEY,
                subject TEXT,
                prompt TEXT,
                context TEXT
            )""")
        
        self.sql_cursor.execute("""CREATE TABLE IF NOT EXISTS Generated_Dataset (
                log_probs_id INTEGER PRIMARY KEY,
                answer TEXT,
                id INTEGER
            )""")
        
        self.sql_cursor.execute("""CREATE TABLE IF NOT EXISTS Log_Probs_Dataset (
                log_probs_id INTEGER PRIMARY KEY,
                per_token_log_probs TEXT
            )""")
        
        self.sql_cursor.execute("""CREATE TABLE IF NOT EXISTS Generated_Dataset_Scores (
                log_probs_id INTEGER PRIMARY KEY,
                score REAL
            )""")
        self.conn.commit()

    def populate_prompt_table(self):

        print(f"Inserting {len(self.prompt_dataset)} rows into SQL database")

        rows_to_insert = []
        for row in self.prompt_dataset:
            rows_to_insert.append((
                row['id'],
                row['subject'],
                row['prompt'],
                row['context']
            ))

        try:
            self.sql_cursor.executemany(
                "INSERT INTO Prompt_Dataset (id, subject, prompt, context) VALUES (?, ?, ?, ?)", 
                rows_to_insert
            )
            self.conn.commit()
            print("DB populated with prompt data.")
        except sqlite3.Error as e:
            print(f"An error occurred during insertion: {e}")

    def drop_generated_tables(self):
        self.sql_cursor.execute("DELETE FROM Generated_Dataset")
        self.sql_cursor.execute("DELETE FROM Log_Probs_Dataset")
        self.sql_cursor.execute("DELETE FROM Generated_Dataset_Scores")
        self.conn.commit()

    def get_generated(self):
        to_fetch = self.sql_cursor.execute("""
                        SELECT gd.id, gd.answer, gs.score
                            FROM Generated_Dataset AS gd
                            JOIN Generated_Dataset_Scores AS gs ON gd.log_probs_id = gs.log_probs_id
                        """)
        return to_fetch.fetchall()

    def sample(self):
        sampled_ds = self.sampler.sample(self.prompt_dataset, **self.generation_config)
        
        try:
            self.sql_cursor.executemany(
                """INSERT INTO Generated_Dataset (log_probs_id, answer, id) VALUES (?, ?, ?)""",
                [(ans['log_probs_id'], ans['answer'], ans['id']) for ans in sampled_ds]
            )
            self.conn.commit()
            print("Successfully inserted answers into SQL database")
        except sqlite3.Error as e:
            print(f"An error occurred during insertion: {e}")
        
        log_probs_ds = self.sampler.get_log_probs(self.prompt_dataset, sampled_ds, self.generation_config['batch_size'], self.max_len)
        
        del sampled_ds

        try:
            self.sql_cursor.executemany(
                """INSERT INTO Log_Probs_Dataset (log_probs_id, per_token_log_probs) VALUES (?, ?)""",
                [(ans['log_probs_id'], json.dumps(ans['per_token_log_probs'])) for ans in log_probs_ds]
            )
            self.conn.commit()
            print("Successfully inserted log_probs into SQL database")
        except sqlite3.Error as e:
            print(f"An error occurred during insertion: {e}")

        del log_probs_ds

    def evaluate(self):
        eval_data_cursor = self.sql_cursor.execute(
            """SELECT gd.log_probs_id, pd.prompt, pd.context, gd.answer 
                FROM Prompt_Dataset AS pd
                JOIN Generated_Dataset AS gd ON pd.id = gd.id"""
        )
        rows = eval_data_cursor.fetchall()

        if not rows:
            print("No data found to evaluate.")
            return

        temp_input = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.jsonl', encoding='utf-8', dir=self.local_folder)
        temp_output_path = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.jsonl', encoding='utf-8', dir=self.local_folder)
        
        try:
            for row in rows:
                temp_input.write(json.dumps({
                    'log_probs_id': row[0],
                    'prompt': row[1],
                    'context': row[2],
                    'answer': row[3]
                }) + '\n')
            temp_input.close()

            run_evaluation(
                input_file=temp_input.name, 
                output_file=temp_output_path.name, 
                id_col_name='log_probs_id'
            )

            scored_results = []
            if os.path.exists(temp_output_path.name): 
                with open(temp_output_path.name, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            scored_results.append(json.loads(line))
            
            if scored_results:
                try:
                    self.sql_cursor.executemany(
                        """INSERT INTO Generated_Dataset_Scores (log_probs_id, score) VALUES (?, ?)""",
                        [(item['log_probs_id'], item['total_score']) for item in scored_results]
                    )
                    self.conn.commit()
                    print(f"Successfully inserted {len(scored_results)} scores into SQL database")
                except sqlite3.Error as e:
                    print(f"An error occurred during score insertion: {e}")
            else:
                print("Evaluation finished but no results were found in the output file.")

        finally:
            if os.path.exists(temp_input.name):
                os.remove(temp_input.name)
            if os.path.exists(temp_output_path.name):
                os.remove(temp_output_path.name)

    def get_dataset(self, id):
        train_data_cursor = self.sql_cursor.execute(
            f"""SELECT pd.prompt, gd.answer, ld.per_token_log_probs, gs.score
                FROM Prompt_Dataset AS pd
                JOIN Generated_Dataset AS gd ON pd.id = gd.id
                JOIN Log_Probs_Dataset AS ld ON gd.log_probs_id = ld.log_probs_id
                JOIN Generated_Dataset_Scores AS gs ON gd.log_probs_id = gs.log_probs_id
                WHERE pd.id = {id}"""
        )
        train_data = train_data_cursor.fetchall()
        return [
            {'text': [{
                'role': 'user',
                'content': prompt},
                {
                    'role': 'assistant',
                    'content': answer
                }],
                'per_token_log_probs': json.loads(per_token_log_probs),
                'score': score
            }
            for prompt, answer, per_token_log_probs, score in train_data
        ]

class GRPO_IterablaDataset(IterableDataset):
    def __init__(self, grpo_dbms:GRPO_DBMS,
                tokenizer, 
                device,
                pad_token='<pad>',
                user_token='<|user|>',
                max_len=2048):
        
        self.grpo_dbms = grpo_dbms
        self.tokenizer = tokenizer
        self.device = device

        self.pad_token_id_raw = self.tokenizer.convert_tokens_to_ids(pad_token)
        self.pad_token_id = torch.tensor([self.pad_token_id_raw])
        self.user_token_id = torch.tensor([self.tokenizer.convert_tokens_to_ids(user_token)])
        self.max_len = max_len

    def __len__(self):
        return len(self.grpo_dbms.prompt_dataset)

    def __iter__(self):
        for id in range(len(self.grpo_dbms.prompt_dataset)):
            grpo_group = self.grpo_dbms.get_dataset(id)

            out = [[], [], [], []]

            for line in grpo_group:
                tokens =  self.tokenizer.apply_chat_template(line['text'], return_tensors='pt', padding='do_not_pad',
                                                                    truncation=True, max_length=self.max_len+1, add_generation_prompt=False)
                tokens = tokens.squeeze(0)
                tokens_shape = tokens.shape[0]

                if tokens_shape <= self.max_len:
                    tokens = torch.concat(
                        [tokens, self.user_token_id])
                    
                tokens = tokens.to(self.device)

                out[0].append(tokens[:-1])
                out[1].append(tokens[1:])
                out[2].append(torch.tensor(line['per_token_log_probs'], dtype=torch.bfloat16))
                out[3].append(torch.tensor(line['score'], dtype=torch.bfloat16))
                
            yield (
                torch.nn.utils.rnn.pad_sequence(out[0], batch_first=True, padding_value=self.pad_token_id_raw),
                torch.nn.utils.rnn.pad_sequence(out[1], batch_first=True, padding_value=self.pad_token_id_raw),
                out[2],
                torch.tensor(out[3], dtype=torch.bfloat16)
            )
        
class GRPO_Collator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        
        batch_inputs = [item[0] for item in batch] # List of B tensors, each (G, Seq)
        batch_labels = [item[1] for item in batch] 
        batch_logs   = [item[2] for item in batch] # List of B lists
        batch_scores = [item[3] for item in batch] # List of B tensors (G,)

        max_len = max(t.size(1) for t in batch_inputs)
        
        padded_inputs = []
        padded_labels = []

        for b_input, b_label in zip(batch_inputs, batch_labels):
            if b_input.size(1) < max_len:
                pad_amt = max_len - b_input.size(1)
                b_input = torch.nn.functional.pad(b_input, (0, pad_amt), value=self.pad_token_id)
                b_label = torch.nn.functional.pad(b_label, (0, pad_amt), value=self.pad_token_id)
            padded_inputs.append(b_input)
            padded_labels.append(b_label)

        return (
            torch.stack(padded_inputs),      # Shape: (B, G, SeqLen)
            torch.stack(padded_labels),      # Shape: (B, G, SeqLen)
            batch_logs,                      # List of size B, containing lists of size G
            torch.stack(batch_scores)        # Shape: (B, G)
        )


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from cirilla.Cirilla_model import CirillaTokenizer
    from cirilla.RL.GRPO.grpo_loss import GRPO
    from cirilla.Cirilla_model import Cirilla, Args, get_optims
    
    gen_config = {'batch_size': 2, 'n_generate_with_kv_cache': 1, 'n_generate_naive': 1}
    
    dbms = GRPO_DBMS(
        model_hub_url="AnthonyPa57/Cirilla-0.3B-4E", 
        prompt_dataset_hub_url="AnthonyPa57/Witcher-GRPO-prompts",
        generation_config=gen_config
    )

    tokenizer = CirillaTokenizer(hub_url='AnthonyPa57/Cirilla-0.3B-4E')

    # Pull model (force_eager=True is often safer for debugging custom loops)
    model = Cirilla(Args())
    model.pull_model_from_hub('AnthonyPa57/Cirilla-0.3B-4E', map_device='cuda:0', force_dynamic_mask=True, force_eager=True)

    muon_opt, adam_opt = get_optims(
                                model,
                                use_muon_optim=True,
                                optim=torch.optim.AdamW,
                                lr=5e-4, weight_decay=1e-5,
                                )
    
    criterion = GRPO(epsilon=0.1, beta=0.1)

    dbms.populate_prompt_table()
    dbms.sample() 
    dbms.crg.model.to('cpu')
    dbms.evaluate()
    
    dataset = GRPO_IterablaDataset(
        grpo_dbms=dbms,
        tokenizer=tokenizer,
        device='cpu',
        pad_token='<pad>',
        user_token='<|user|>'
    )
    
    pad_token_id = tokenizer.convert_tokens_to_ids('<pad>')
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=GRPO_Collator(pad_token_id))

    model.train()

    for batch_idx, batch in enumerate(dataloader):

        input_ids, labels, log_probs, scores = batch
        input_ids = input_ids.to(model.args.device)
        labels = labels.to(model.args.device)
        scores = scores.to(model.args.device)

        B, G, S = input_ids.shape

        # Initialize full size first
        _old_log_probs = torch.zeros((B, G, S), dtype=torch.bfloat16, device=model.args.device)
        training_mask = (labels != pad_token_id)

        for i, group in enumerate(log_probs):
            for j, seq_tensor in enumerate(group):
                ans_len = len(seq_tensor)
                full_seq_len = (input_ids[i, j] != pad_token_id).sum().item()
                offset = full_seq_len - ans_len
                
                if offset < 0: offset = 0 

                _old_log_probs[i, j, offset : offset + ans_len] = seq_tensor.to(model.args.device)
                training_mask[i, j, :offset] = False

        # --- FIX START ---
        # Flatten input for the model
        input_ids_flat = input_ids.view(-1, S)
        
        # Get new log probs (Shape: B*G, S-1)
        new_log_probs_flat = model.get_per_token_log_probs(input_ids_flat) 
        
        # Reshape to (B, G, S-1)
        new_log_probs = new_log_probs_flat.view(B, G, S - 1)

        # Slice old_log_probs and mask to match (drop the first token)
        # new_log_probs[t] predicts token t+1. 
        # _old_log_probs[t+1] stores the probability of token t+1.
        # So we align new[0:] with old[1:]
        _old_log_probs_sliced = _old_log_probs[:, :, 1:]
        training_mask_sliced = training_mask[:, :, 1:]

        loss = criterion(
            model_log_probs=new_log_probs,
            old_model_log_probs=_old_log_probs_sliced,
            reference_model_log_probs=_old_log_probs_sliced,
            rewards=scores,
            mask=training_mask_sliced
        )
        # --- FIX END ---
        
        print(f"Batch {batch_idx} Loss: {loss.item()}")

        loss.backward()

        muon_opt.step()
        adam_opt.step()

        muon_opt.zero_grad()
        adam_opt.zero_grad()