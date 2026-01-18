from cirilla.RL.GRPO.generate_answers import CirillaResponseGenerator, CirillaSampler
from datasets import load_dataset, DatasetDict
from cirilla.RL.GRPO.grpo_loss import GRPO
from cirilla.RL.GRPO.local_scorer_vllm import run_evaluation
from cirilla.Cirilla_model import JSONDynamicDatset, DynamicCollator
from torch.utils.data import IterableDataset
import sqlite3
import torch
import json
import tempfile
import os

class GRPO_DBMS:
    def __init__(self, model_hub_url:str, prompt_dataset_hub_url:str, 
                generation_config:dict, evaluation_config:dict,
                local_folder:str='./GRPO_data'):
        
        self.model_hub_url = model_hub_url
        self.prompt_dataset = load_dataset(prompt_dataset_hub_url, split='train')
        self.generation_config = generation_config
        self.evaluation_config = evaluation_config

        self.crg = CirillaResponseGenerator(model_hub_url)
        self.sampler = CirillaSampler(self.crg)

        self.conn = sqlite3.connect(f'{local_folder}/grpo.db')
        self.sql_cursor = self.conn.cursor()
        self.local_folder = local_folder

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
            print("Insertion complete.")
        except sqlite3.Error as e:
            print(f"An error occurred during insertion: {e}")

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
        
        del sampled_ds

        log_probs_ds = self.sampler.get_log_probs(self.prompt_dataset, sampled_ds, self.generation_config['batch_size'])
        
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
                output_file=temp_output_path, 
                id_col_name='log_probs_id'
            )

            scored_results = []
            if os.path.exists(temp_output_path):
                with open(temp_output_path, 'r', encoding='utf-8') as f:
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

        self.pad_token_id = self.tokenizer.pad_token_id
        self.pad_token_id = torch.tensor([tokenizer.convert_tokens_to_ids(pad_token)])
        self.user_token_id = torch.tensor([tokenizer.convert_tokens_to_ids(user_token)])
        self.max_len = max_len

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
                        [tokens, self.user_token_id] + \
                        [self.pad_token_id] * (self.max_len - tokens_shape))
                    
                tokens = tokens.to(self.device)

                out[0].append(tokens[:-1])
                out[1].append(tokens[1:])
                out[2].append(torch.tensor(line['per_token_log_probs'], dtype=torch.bfloat16))
                out[3].append(torch.tensor(line['score'], dtype=torch.bfloat16))
                
            return (
                torch.nn.utila.rnn.pad_sequence(out[0], batch_first=True, padding_value=self.pad_token_id),
                torch.nn.utila.rnn.pad_sequence(out[1], batch_first=True, padding_value=self.pad_token_id),
                out[2],
                torch.tensor(out[3], dtype=torch.bfloat16)
            )
        
class GRPO_Collator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        return (
            torch.nn.utils.rnn.pad_sequence(batch[0], batch_first=True, padding_value=self.pad_token_id),
            torch.nn.utils.rnn.pad_sequence(batch[1], batch_first=True, padding_value=self.pad_token_id),
            batch[2],
            batch[3]
        )
