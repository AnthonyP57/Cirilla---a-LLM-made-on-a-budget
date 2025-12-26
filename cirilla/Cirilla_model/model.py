from cirilla.LLM_pieces import DynamicTanh, Dynamic_erf
from dataclasses import dataclass
import torch.nn as nn
from .modules import CirillaBaseModel
from .blocks import Decoder, DecoderArgs, InputEmbeddings
import torch

@dataclass
class Args(DecoderArgs):
    vocab_size:int = 60_000
    tie_params:bool = False
    out_bias:bool = False

class Cirilla(
            nn.Module,
            CirillaBaseModel,
            pipeline_tag="text-generation",
            library_name="pytorch",
            license="mit"
    ):
    def __init__(self, args:Args=None):
        super().__init__()

        if isinstance(args, dict):
            args = Args(**args)

        if args is None:
            args = Args()

        self.args = args
        self._prepare_model()

    def _prepare_model(self):

        self.emb = InputEmbeddings(self.args)
        if self.args.layer_norm == "RMSNorm":
            self.layer_norm = nn.RMSNorm(self.args.dim)
        elif self.args.layer_norm == "Derf":
            self.layer_norm = Dynamic_erf(self.args.dim)
        elif self.args.layer_norm == "DyT":
            self.layer_norm = DynamicTanh(self.args.dim)
        else:
            raise ValueError(f"allowed layer norms: 'RMSNorm', 'Derf', 'DyT' ; got: {self.args.layer_norm}")
        self.decoder = Decoder(self.args)

        self.output = nn.Linear(self.args.dim, self.args.vocab_size, bias=self.args.out_bias)
        if self.args.tie_params:
            self.output.weight = self.emb.embeddings.weight

        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.to(self.args.device, dtype=self.args.dtype)
        
    def pred(self, x):
        
        x = self.emb(x)

        if self.args.output_moe_weights:
            x, moe_weights = self.decoder(x)

            x = self.layer_norm(x)
            x = self.output(x)

            return x, moe_weights
        
        else:
            x = self.decoder(x)

            x = self.layer_norm(x)
            x = self.output(x)
        
            return x

    def forward(self, x):
        return self.pred(x)
    
    @torch.no_grad()
    def infer_with_cache(self, x, cur_pos:int, seq_len:int=1, max_batch:int=1):
        
        x = self.emb(x)

        if self.args.output_moe_weights:

            moe_weights = []
            for attention, moe in zip(self.decoder.attentions, self.decoder.smoes):

                x = x + attention.forward_with_cache(x, cur_pos, seq_len, max_batch)
                moe_out, moe_w = moe(x)
                moe_weights.append(moe_w)
                x = x + moe_out

            x = self.layer_norm(x)
            x = self.output(x)

            return x, moe_weights
        
        else:

            for attention, moe in zip(self.decoder.attentions, self.decoder.smoes):
                x = x + attention.forward_with_cache(x, cur_pos, seq_len, max_batch)
                x = x + moe(x)[0]

            x = self.layer_norm(x)
            x = self.output(x)
        
            return x

    @torch.no_grad()
    def infer(self, x):
        if self.args.output_moe_weights:
            logits, moe_weights = self.pred(x)
            return logits
        else:
            logits = self.pred(x)
            return logits
    
    def _greedy_next_token(self, x):
        logits = self.infer(x)
        probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)
        return next_token
    
    def generate_naive(self, x:torch.Tensor,
                       max_new_tokens:int=1024,
                       top_k:int=None,
                       top_p:float=None,
                       n_beams:int=None,
                       temperature:float=1.0,
                       termination_tokens:list[int]=None
                       ):

        if top_k is None and top_p is None and n_beams is None and temperature == 1.0: # pure greedy
            for _ in range(max_new_tokens):
                next_token = self._greedy_next_token(x)
                if termination_tokens is not None and next_token.item() in termination_tokens:
                    break
                x = torch.cat((x, next_token), dim=1)
            return x
        
        else:

            with torch.no_grad():

                if n_beams is None:
                    n_beams = 1

                _beams = [[x, 0, False] for _ in range(n_beams)]

                for _ in range(max_new_tokens):
                    
                    n_remaining_top_p = None

                    if all([beam[2] for beam in _beams]): # all beams have reached termination
                        break
                    _new_beams = []

                    for beam in _beams:

                        if beam[2]: # termination already reached
                            _new_beams.append(beam)
                            continue

                        logits = self.infer(beam[0])
                        logits = logits[:, -1, :] / temperature

                        if top_k is not None:
                            values, indices = torch.topk(logits, top_k)
                            log_probs = torch.full_like(logits, float('-inf'))
                            log_probs = log_probs.scatter_(1, indices, torch.nn.functional.log_softmax(values, dim=-1))

                        elif top_p is not None:
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                            sorted_indices_to_remove[:, 0] = 0

                            indices_to_remove = sorted_indices[sorted_indices_to_remove]
                            n_remaining_top_p = logits.size(-1) - indices_to_remove.size(0)

                            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                            log_probs[0, indices_to_remove] = float('-inf')

                        else: # greedy
                            values, indices = torch.topk(logits, n_beams)
                            log_probs = torch.full_like(logits, float('-inf'))
                            log_probs = log_probs.scatter_(1, indices, torch.nn.functional.log_softmax(values, dim=-1))

                        n_samples = min(n_beams,
                                        top_k if top_k is not None else float('inf'),
                                        n_remaining_top_p if n_remaining_top_p is not None else float('inf')
                                        )

                        next_tokens = torch.multinomial(log_probs.exp(), num_samples=n_samples, replacement=n_samples < n_beams) #batch_size x n_beams
                        next_tokens_probs = log_probs.gather(1, next_tokens)

                        for i in range(next_tokens.size(1)):

                            token = next_tokens[0, i].unsqueeze(0).unsqueeze(0)
                            token_prob = next_tokens_probs[i]

                            _new_beams.append([torch.cat([beam[0], token], dim=1),
                                               beam[1] + token_prob.item(),
                                               beam[2] or (termination_tokens is not None and token.item() in termination_tokens)
                                               ])

                    _beams = _new_beams
                
                    _beams = sorted(_beams, key=lambda x: x[1], reverse=True)[:n_beams]

                return _beams[0][0]

def generate_kv_cache(self,
                          prompt_tokens_list: list[list[int]],
                          max_new_tokens: int = 1024,
                          top_k: int = None,
                          top_p: float = None,
                          temperature: float = 1.0,
                          termination_tokens: list[int] = None,
                          pad_token_id: int = 0
                          ):
        
        batch_size = len(prompt_tokens_list)
        
        prompt_lens = torch.tensor([len(t) for t in prompt_tokens_list], device=self.args.device)
        max_prompt_len = prompt_lens.max().item()
        min_prompt_len = prompt_lens.min().item()
        
        total_len = min(self.args.context_window, max_prompt_len + max_new_tokens)
        
        tokens = torch.full((batch_size, total_len), pad_token_id, dtype=torch.long, device=self.args.device)
        
        for k, t in enumerate(prompt_tokens_list):
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=self.args.device)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.args.device)
        
        with torch.no_grad():
            
            cur_pos = 0
            
            if min_prompt_len > 0:
                initial_input = tokens[:, :min_prompt_len]
                
                logits = self.infer_with_cache(initial_input, cur_pos=0, seq_len=min_prompt_len, max_batch=batch_size)
                
                next_token_logits = logits[:, -1, :]
                
                cur_pos = min_prompt_len

            while cur_pos < total_len:
                
                next_token_logits = next_token_logits / temperature

                if top_k is not None:
                    v, i = torch.topk(next_token_logits, top_k)
                    log_probs = torch.full_like(next_token_logits, float('-inf'))
                    log_probs.scatter_(1, i, torch.nn.functional.log_softmax(v, dim=-1))
                    probs = torch.exp(log_probs)

                elif top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    mask = torch.zeros_like(next_token_logits, dtype=torch.bool).scatter_(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[mask] = float('-inf')
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

                else: # Greedy
                    if temperature == 1.0 and top_k is None and top_p is None:
                        probs = None
                    else:
                        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

                if probs is not None:
                    next_token_sample = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_token_sample = torch.argmax(next_token_logits, dim=-1)


                is_prompt_phase = cur_pos < prompt_lens
                safe_lookup_idx = torch.clamp(torch.tensor(cur_pos), max=total_len-1)
                ground_truth = tokens[:, safe_lookup_idx]

                next_token = torch.where(is_prompt_phase, ground_truth, next_token_sample)

                if cur_pos < total_len:
                    tokens[:, cur_pos] = next_token

                if termination_tokens is not None:
                    active_generation_mask = ~is_prompt_phase
                    for t in termination_tokens:
                        has_terminated = (next_token == t) & active_generation_mask
                        finished = finished | has_terminated
                
                if finished.all() and cur_pos >= max_prompt_len:
                    break

                if cur_pos == total_len - 1:
                    break
                    
                input_token = next_token.unsqueeze(1)
                
                logits = self.infer_with_cache(input_token, cur_pos=cur_pos, seq_len=1, max_batch=batch_size)
                next_token_logits = logits[:, -1, :]
                
                cur_pos += 1

        return tokens
