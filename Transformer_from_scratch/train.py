from dataset import get_dataset
from model import Transformer
from modules import ModelConfig, show_valid
import torch
import warnings
import torch.nn as nn
from tqdm import tqdm

torch.set_float32_matmul_precision('high')

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_properties(0))
else:
    device = torch.device("cpu")
    print("using CPU, GPU is not available")


data_train, data_val, tokenizer = get_dataset({
    'batch_size': 8,
    'seq_len': 320
})

config = ModelConfig('/home/anthonyp57/VSCode_projects/Radovid/Transformer_from_scratch/checkpoints/',
                     lr=1e-4, wd=0, d_model=512, num_heads=8, num_layers=3,
                     vocab_size=tokenizer.get_vocab_size(), max_seq_len=320, d_ff=2048, drop=0.1, resume=True)

model = Transformer(config.d_model, config.num_heads, config.num_layers, config.vocab_size, config.max_seq_len, config.d_ff, config.drop).to(device)
model = torch.compile(model)

for param in model.parameters():
    if param.dim() > 1:
        nn.init.xavier_uniform_(param)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {n_params/1e6:.2f}M')

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.wd, eps=1e-9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.85)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

if hasattr(config, 'loaded_checkpoint'):
    model.load_state_dict(config.loaded_checkpoint['model_state_dict'])
    optimizer.load_state_dict(config.loaded_checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(config.loaded_checkpoint['scheduler_state_dict'])

past_epochs = config.epoch
epochs = 30
epochs -= past_epochs
iterator = tqdm(range(epochs), desc='Masked fill training')

for epoch in iterator:
    torch.cuda.empty_cache()
    model.train()

    for batch in data_train:

        with torch.autocast(device_type="cuda:0", dtype=torch.bfloat16):

            enc_in = batch['enc_in'].to(device)
            dec_in = batch['dec_in'].to(device)
            dec_out = batch['dec_out'].to(device)
            enc_mask = batch['enc_mask'].to(device)
            dec_self_att_mask = batch['dec_self_att_mask'].to(device)
            dec_cross_att_mask = batch['dec_cross_att_mask'].to(device)
        
            enc_in = model.encode(enc_in, enc_mask)
            dec_in = model.decode(dec_in, enc_in, dec_self_att_mask, dec_cross_att_mask)
            pred = model.project(dec_in)

            loss = criterion(pred.view(-1, tokenizer.get_vocab_size()), dec_out.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
    
    config.checkpoint(model, optimizer, scheduler, epoch + past_epochs)    
    scheduler.step()
    model.eval()
    show_valid(data_val, model, tokenizer, iterator, config.max_seq_len)
    
    
torch.save(model,'/home/anthonyp57/VSCode_projects/Radovid/Transformer_from_scratch/checkpoints/model.pth')
torch.save(model.state_dict(), '/home/anthonyp57/VSCode_projects/Radovid/Transformer_from_scratch/checkpoints/model_state_dict.pth')