import torch
import os
from dataset import mask

def greedy_decode(model, enc_in, enc_mask, tokenizer, seq_len, device = torch.device('cuda')):
    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_ind = tokenizer.token_to_id('[EOS]')

    encoder_output = model.encode(enc_in, enc_mask)

    decoder_input = torch.tensor([[sos_idx]], dtype=torch.int64).to(device)

    while True:
        if decoder_input.size(1) == seq_len:
            break

        out = model.decode(decoder_input, encoder_output, mask(decoder_input, decoder_input, pad_token=eos_ind, seq_len=decoder_input.size(1), dec_len=decoder_input.size(1), causal=True).to(device),
                           mask(decoder_input, encoder_output, pad_token=eos_ind, dec_len=decoder_input.size(1)).to(device))[:,-1]

        _, next_word = torch.max(out, dim=1) #greedy

        decoder_input = torch.cat([decoder_input, torch.tensor([[next_word]], dtype=torch.int64).to(device)], dim=1)

        if next_word == eos_ind:
            break

    return decoder_input.squeeze(0)

def show_valid(data_valid, model, tokenizer, writer, seq_len, device=torch.device('cuda'), n_samples=2):
    sample=0
    with torch.no_grad():
        for data in data_valid:
            assert data['enc_in'].size(0) == 1, 'Only batch size 1 supported'
            sample+=1
            enc_in = data['enc_in'].to(device)
            expected = data['dec_out']
            enc_mask = data['enc_mask'].to(device)

            dec_out = greedy_decode(model, enc_in, enc_mask, tokenizer, seq_len, device)

            dec_out = tokenizer.decode(dec_out.detach().cpu().numpy())
            enc_in = tokenizer.decode(enc_in[0].detach().cpu().numpy())
            expected = tokenizer.decode(expected[0].detach().cpu().numpy())

            writer.write(f'INPUT: {enc_in}')
            writer.write(f'EXPECTED: {expected}')
            writer.write(f'PREDICTED: {dec_out}')
            writer.write('\n')

            if sample > n_samples:
                break

class ModelConfig:
    def __init__(self, checkpoint_path, lr, wd, d_model, num_heads, num_layers, vocab_size, max_seq_len, d_ff=2048, drop=0.1, resume=False):
        self.lr = lr
        self.wd = wd
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_ff = d_ff
        self.drop = drop
        self.checkpoint_path = checkpoint_path
        self.epoch = 0

        if resume and os.path.exists(self.checkpoint_path + f'{self.lr}-{self.wd}-{self.d_model}-{self.num_heads}-{self.num_layers}-{self.vocab_size}-{self.max_seq_len}-{self.d_ff}-{self.drop}.pth'):
            config = torch.load(self.checkpoint_path + f'{self.lr}-{self.wd}-{self.d_model}-{self.num_heads}-{self.num_layers}-{self.vocab_size}-{self.max_seq_len}-{self.d_ff}-{self.drop}.pth')
            self.loaded_checkpoint = config
            config = config['config']
            self.lr = config['lr']
            self.wd = config['wd']
            self.d_model = config['d_model']
            self.num_heads = config['num_heads']
            self.num_layers = config['num_layers']
            self.vocab_size = config['vocab_size']
            self.max_seq_len = config['max_seq_len']
            self.d_ff = config['d_ff']
            self.drop = config['drop']
            self.epoch = config['epoch']

            print(f'resuming from epoch: {self.epoch}')

    def get_config(self, epoch):
        return {
            'lr': self.lr,
            'wd': self.wd,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            'd_ff': self.d_ff,
            'drop': self.drop,
            'epoch': epoch
        }

    def checkpoint(self, model, optimizer, scheduler, epoch):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': self.get_config(epoch+1)},
                   self.checkpoint_path+f'{self.lr}-{self.wd}-{self.d_model}-{self.num_heads}-{self.num_layers}-{self.vocab_size}-{self.max_seq_len}-{self.d_ff}-{self.drop}.pth')