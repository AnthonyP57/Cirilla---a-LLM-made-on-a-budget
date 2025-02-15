import torch
import os
from dataset import mask
import math

def beam_search_decode(model, enc_in, enc_mask, tokenizer, seq_len, beam_size=3, temperature=1, device = torch.device('cuda')):
    sos_token = tokenizer.token_to_id('[SOS]')
    eos_token = tokenizer.token_to_id('[EOS]')
    mask_token = tokenizer.token_to_id('[MASK]')
    pad_token = tokenizer.token_to_id('[PAD]')

    encoder_output = model.encode(enc_in, enc_mask)

    decoder_input = torch.tensor([[sos_token]]).type_as(enc_in).to(device)
    candidates = [(decoder_input, 0)] #log(1) = 0

    while True:

        if any([cand.size(1) == seq_len for cand, _ in candidates]):
            break
        new_candidates = []

        for candidate, score in candidates:

            if candidate[0][-1].item() == eos_token:
                continue

            out = model.decode(candidate, encoder_output,
                        mask(candidate, candidate, pad_token=pad_token, seq_len=candidate.size(1),dec_len=candidate.size(1), causal=True).to(device),
                        mask(candidate, encoder_output, pad_token=pad_token,  dec_len=candidate.size(1)).to(device))
        
            out = model.project(out[:, -1])

            if temperature != 1:
                out = out / temperature
            
            out = torch.log_softmax(out, dim=-1)*100 #log for stability, *100 for it not to be too small

            topk_prob, topk_idx = torch.topk(out, beam_size, dim=1)

            for i in range(beam_size):
                token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)
                token_prob = topk_prob[0][i].item()
                new_candidate = torch.cat([candidate, token], dim=1)
                new_candidates.append((new_candidate, score + token_prob))

        candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
        candidates = candidates[:beam_size]

        if all([cand[0][-1].item() == sos_token for cand, _ in candidates]):
            break

    # Return the best candidate
    return candidates[0][0].squeeze()

def greedy_decode(model, enc_in, enc_mask, tokenizer, seq_len, device = torch.device('cuda')):
    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_ind = tokenizer.token_to_id('[EOS]')
    mask_token = tokenizer.token_to_id('[MASK]')
    pad_token = tokenizer.token_to_id('[PAD]')

    encoder_output = model.encode(enc_in, enc_mask)

    decoder_input = torch.tensor([[sos_idx]]).type_as(enc_in).to(device)

    while True:
        if decoder_input.size(1) == seq_len:
            break

        out = model.decode(decoder_input, encoder_output,
                           mask(decoder_input, decoder_input, pad_token=pad_token, seq_len=decoder_input.size(1),dec_len=decoder_input.size(1), causal=True).to(device),
                           mask(decoder_input, encoder_output, pad_token=pad_token, dec_len=decoder_input.size(1)).to(device))
        
        out = model.project(out[:, -1])

        _, next_word = torch.max(out, dim=1) #greedy

        decoder_input = torch.cat([decoder_input, torch.tensor([[next_word]]).type_as(enc_in).to(device)], dim=1)

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
            masked_text = data['masked']

            dec_out = greedy_decode(model, enc_in, enc_mask, tokenizer, seq_len, device)
            try:
                dec_out_beam = beam_search_decode(model, enc_in, enc_mask, tokenizer, seq_len, device=device)
            except:
                writer.write('Beam search failed')
                dec_out_beam = None

            dec_out = tokenizer.decode(dec_out.detach().cpu().numpy())
            enc_in = tokenizer.decode(enc_in[0].detach().cpu().numpy())
            expected = tokenizer.decode(expected[0].detach().cpu().numpy())

            writer.write(f'INPUT: {masked_text[0]}')
            writer.write(f'EXPECTED: {expected}')
            if dec_out_beam is not None:
                dec_out_beam = tokenizer.decode(dec_out_beam.detach().cpu().numpy())
                writer.write(f'PREDICTED BEAM SEARCH: {dec_out_beam}')
            writer.write(f'PREDICTED GREEDY SEARCH: {dec_out}')
            writer.write('\n')

            if sample >= n_samples:
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