from cirilla.Few_shot import MAMLBinaryAdapterTrainer, MamlPretrainingDataset
from cirilla.Cirilla_model import CirillaBERT, BertArgs
from cirilla.Cirilla_model import CirillaTokenizer

tasks = MamlPretrainingDataset(path=('examples/data/example_bert.jsonl',
                                    'examples/data/example_bert.jsonl'), batch_size=2)
print(f"n tasks: {len(tasks)}")

model = CirillaBERT(BertArgs(output_what='meanpool', moe_type='pytorch', n_layers=2, dim=128, d_ff=256, torch_compile=False))

tokenizer = CirillaTokenizer(hub_url='AnthonyPa57/HF-torch-demo2')

trainer = MAMLBinaryAdapterTrainer(model, tokenizer)

trainer.meta_train(tasks)

finetune_texts = ['some other text, different text', 'some other text']
finetune_labels = [0, 1]
test_texts = ['some other text', 'some other text, different text']
test_labels = [1, 0]

trainer.fine_tune_native(finetune_texts, finetune_labels, test_texts, test_labels, verbose=True)
