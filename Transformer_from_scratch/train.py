from dataset import get_dataset
from model import Transformer
from modules import ModelConfig, show_valid
import torch.nn.functional as F
import torch
import warnings
import torch.nn as nn
from tqdm import tqdm

warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_properties(0))
else:
    device = torch.device("cpu")
    print("using CPU, GPU is not available")


data_train, data_val, tokenizer = get_dataset({
    'batch_size': 4,
    'seq_len': 320
})

config = ModelConfig('./Transformer_from_scratch/checkpoints',
                     lr=1e-4, wd=1e-6, d_model=512, num_heads=8, num_layers=3,
                     vocab_size=tokenizer.get_vocab_size(), max_seq_len=350, d_ff=2048, drop=0.1, resume=False)

model = Transformer(config.d_model, config.num_heads, config.num_layers, config.vocab_size, config.max_seq_len, config.d_ff, config.drop).to(device)

for param in model.parameters():
    if param.dim() > 1:
        nn.init.xavier_uniform_(param)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {n_params/1e6:.2f}M')

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.wd)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'), label_smoothing=0.1)

epochs=50
iterator = tqdm(range(epochs), desc='Masek fill training')

for epoch in iterator:
    torch.cuda.empty_cache()
    model.train()

    for batch in data_train:

        enc_in = batch['enc_in'].to(device)
        dec_in = batch['dec_in'].to(device)
        dec_out = batch['dec_out'].to(device)
        enc_mask = batch['enc_mask'].to(device)
        dec_self_att_mask = batch['dec_self_att_mask'].to(device)
        dec_cross_att_mask = batch['dec_cross_att_mask'].to(device)
        

        pred = model(enc_in, dec_in, enc_mask, dec_self_att_mask, dec_cross_att_mask)

        loss = criterion(pred.view(-1, tokenizer.get_vocab_size()), dec_out.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
    
    scheduler.step()
    show_valid(data_val, model, tokenizer, iterator, config.max_seq_len)