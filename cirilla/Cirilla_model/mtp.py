from cirilla.LLM_pieces import DynamicTanh, Dynamic_erf
from dataclasses import dataclass
import torch.nn as nn
from .modules import CirillaBaseModel
from .blocks import Decoder, DecoderArgs, InputEmbeddings
import torch
import torch.nn.functional as F
from math import ceil

@dataclass
class MTPArgs(DecoderArgs):
    n_token_heads:int = 4
    vocab_size:int = 60_000
    tie_params:bool = False
    out_bias:bool = False

class CirillaMTP(
            nn.Module,
            CirillaBaseModel,
            pipeline_tag="text-generation",
            library_name="pytorch",
            license="mit"
    ):
    def __init__(self, args:MTPArgs=None):
        super().__init__()

        if isinstance(args, dict):
            args = MTPArgs(**args)

        if args is None:
            args = MTPArgs()

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

        token_args = {k:v for k,v in self.args.__dict__.items() if k in DecoderArgs.__dataclass_fields__}
        token_args['n_layers'] = 1
        token_args['output_moe_weights'] = False
        self.token_head_args = DecoderArgs(**token_args)

        self.token_heads = nn.ModuleList([nn.Sequential(Decoder(self.token_head_args), type(self.layer_norm)(self.args.dim)) for _ in range(self.args.n_token_heads)])

        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.to(self.args.device, dtype=self.args.dtype)
        
    def get_z(self, x) -> torch.Tensor:
        
        x = self.emb(x)

        if self.args.output_moe_weights:
            x, moe_weights = self.decoder(x)

            x = self.layer_norm(x)

            return x, moe_weights
        
        else:
            x = self.decoder(x)

            x = self.layer_norm(x)
        
            return x
        
    def get_heads(self, idx, z) -> torch.Tensor:
        return self.output(self.token_heads[idx](z))

    @torch.no_grad()
    def _infer_head(self, idx: int, z: torch.Tensor) -> torch.Tensor:
        decoder = self.token_heads[idx][0]
        norm = self.token_heads[idx][1]
        for attention, moe in zip(decoder.attentions, decoder.smoes):
            z = z + attention.forward_with_cache(z, cur_pos=0, max_batch=z.shape[0])
            z = z + moe(z)[0]
        return self.output(norm(z))

    def forward(self, x) -> list[torch.Tensor]:
        if self.args.output_moe_weights:
            x, moe_weights = self.get_z(x)
        else:
            x = self.get_z(x)
        return [self.get_heads(i, x) for i in range(self.args.n_token_heads)]
    
    def pred(self, x, max_heads=None) -> torch.Tensor:
        if max_heads is None:
            max_heads = self.args.n_token_heads
        if self.args.output_moe_weights:
            x, moe_weights = self.get_z(x)
        else:
            x = self.get_z(x)
        return [self.get_heads(i, x) for i in range(max_heads)]
    
    @torch.no_grad()
    def infer_with_cache(self, x, cur_pos:int, max_batch:int=1, chunked_prefill:bool=False, non_finished_ids:torch.Tensor=None) -> torch.Tensor:
        
        x = self.emb(x)

        if self.args.output_moe_weights:

            for attention, moe in zip(self.decoder.attentions, self.decoder.smoes):

                x = x + attention.forward_with_cache(x, cur_pos, max_batch, chunked_prefill, non_finished_ids)
                moe_out, moe_weights = moe(x)
                x = x + moe_out

            x = self.layer_norm(x)
            x = self._infer_head(0, x)

            return x

        else:

            for attention, moe in zip(self.decoder.attentions, self.decoder.smoes):
                x = x + attention.forward_with_cache(x, cur_pos, max_batch, chunked_prefill, non_finished_ids)
                x = x + moe(x)[0]

            x = self.layer_norm(x)
            x = self._infer_head(0, x)

            return x

    @torch.no_grad()
    def infer(self, x, max_heads=None) -> torch.Tensor:
        if max_heads is None:
            max_heads = self.args.n_token_heads
        logits = torch.stack(self.pred(x, max_heads), dim=0) # (head, batch, seq, vocab)
        return logits
    
    def _greedy_next_token(self, x, max_heads=None) -> torch.Tensor:
        logits = self.infer(x, max_heads)
        probs = F.softmax(logits[:, :, -1, :], dim=-1)
        next_token = torch.argmax(probs, dim=-1) # (head, batch)
        return next_token
    
    def generate_naive(self, x:torch.Tensor,
                        max_new_tokens:int=1024,
                        top_k:int=None,
                        top_p:float=None,
                        n_beams:int=None,
                        temperature:float=1.0,
                        termination_tokens:list[int]=None,
                        max_heads=None
                        ) -> torch.Tensor:

        if top_k is None and top_p is None and n_beams is None: # pure greedy
            for _ in range(max_new_tokens):
                next_token = self._greedy_next_token(x, max_heads).transpose(0, 1) # (batch, head)
                x = torch.cat((x, next_token), dim=1)
                if termination_tokens is not None:
                    for i in range(next_token.shape[1]):
                        if next_token[0, i].item() in termination_tokens:
                            if self.args.n_token_heads - 1 - i > 0:
                                x = x[:, :-(self.args.n_token_heads - 1 - i)]
                            break
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

                        logits = self.infer(beam[0], 1)[0]
                        logits = logits[:, -1, :] / temperature

                        if top_k is not None:
                            values, indices = torch.topk(logits, top_k)
                            log_probs = torch.full_like(logits, float('-inf'))
                            log_probs = log_probs.scatter_(1, indices, F.log_softmax(values, dim=-1))

                        elif top_p is not None:
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                            sorted_indices_to_remove[:, 0] = 0

                            indices_to_remove = sorted_indices[sorted_indices_to_remove]
                            n_remaining_top_p = logits.size(-1) - indices_to_remove.size(0)

                            log_probs = F.log_softmax(logits, dim=-1)
                            log_probs[0, indices_to_remove] = float('-inf')

                        else: # greedy
                            values, indices = torch.topk(logits, n_beams)
                            log_probs = torch.full_like(logits, float('-inf'))
                            log_probs = log_probs.scatter_(1, indices, F.log_softmax(values, dim=-1))

                        n_samples = min(n_beams,
                                        top_k if top_k is not None else float('inf'),
                                        n_remaining_top_p if n_remaining_top_p is not None else float('inf')
                                        )

                        next_tokens = torch.multinomial(log_probs.exp(), num_samples=n_samples, replacement=n_samples < n_beams) #batch_size x n_beams
                        next_tokens_probs = log_probs.gather(1, next_tokens)

                        for i in range(next_tokens.size(1)):

                            token = next_tokens[0, i].unsqueeze(0).unsqueeze(0)
                            token_prob = next_tokens_probs[0, i]

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
                            pad_token_id: int = 1,
                            sample_parallel: bool = False
                            ) -> torch.Tensor:
            
            batch_size = len(prompt_tokens_list)
            
            prompt_lens = torch.tensor([len(t) for t in prompt_tokens_list], device=self.args.device)
            max_prompt_len = prompt_lens.max().item()
            min_prompt_len = prompt_lens.min().item()
            n_chunked_prefill_steps = ceil(min_prompt_len / self.args.window_size)
            
            total_len = min(self.args.context_window, max_prompt_len + max_new_tokens)
            
            tokens = torch.full((batch_size, total_len), pad_token_id, dtype=torch.long, device=self.args.device)
            
            for k, t in enumerate(prompt_tokens_list):
                tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=self.args.device)

            non_finished_ids = torch.arange(batch_size, device=self.args.device)

            if sample_parallel:
                response_prob = torch.tensor([0.] * batch_size, device=self.args.device)
            
            with torch.inference_mode():
                
                cur_pos = 0

                for _ in range(n_chunked_prefill_steps):

                    chunk = tokens[:, cur_pos:min(cur_pos + self.args.window_size, min_prompt_len)]
                    logits = self.infer_with_cache(chunk, cur_pos=cur_pos, max_batch=batch_size, chunked_prefill=True)
                    next_token_logits = logits[:, -1, :]
                    cur_pos += chunk.shape[1]

                while cur_pos < total_len: # single token generation loop
                    
                    next_token_logits = next_token_logits / temperature

                    if top_k is not None:
                        v, i = torch.topk(next_token_logits, top_k)
                        probs = torch.full_like(next_token_logits, 0)
                        probs.scatter_(1, i, F.softmax(v, dim=-1))

                    elif top_p is not None:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                        sorted_indices_to_remove[:, 0] = 0
                        
                        mask = torch.zeros_like(next_token_logits, dtype=torch.bool).scatter_(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[mask] = float('-inf')
                        probs = F.softmax(next_token_logits, dim=-1)

                    else: # Greedy
                        max_arg = torch.argmax(next_token_logits, dim=-1) # (b,)
                        probs = torch.zeros_like(next_token_logits)
                        probs[range(probs.size(0)), max_arg] = 1.0

                    next_token_sample = torch.multinomial(probs, num_samples=1).squeeze(1) # (b,1) -> (b,)

                    is_prompt_phase = cur_pos < prompt_lens[non_finished_ids] # (b,)
                    ground_truth = tokens[non_finished_ids, cur_pos] # (b,)

                    next_token = torch.where(is_prompt_phase, ground_truth, next_token_sample) # (b,)

                    tokens[non_finished_ids, cur_pos] = next_token

                    if sample_parallel and ~is_prompt_phase.any():
                        response_prob[non_finished_ids] += F.log_softmax(next_token_logits, dim=-1)[[i for i in range(next_token_logits.size(0))], next_token_sample]

                    if termination_tokens is not None:
                        active_generation_mask = ~is_prompt_phase
                        has_terminated = torch.isin(next_token, torch.tensor(termination_tokens, device=next_token.device, dtype=next_token.dtype)) & active_generation_mask
                        non_finished_ids = non_finished_ids[~has_terminated]
                    
                    if non_finished_ids.size(0) == 0:
                        break
                    
                    if termination_tokens is not None and has_terminated.any():
                        input_token = next_token[~has_terminated].unsqueeze(1) # add seq dim (b, 1)
                    else:
                        input_token = next_token.unsqueeze(1)

                    logits = self.infer_with_cache(input_token, cur_pos=cur_pos, non_finished_ids=non_finished_ids)
                    next_token_logits = logits[:, -1, :]
                    
                    cur_pos += 1

            return tokens[:, :cur_pos+1] if not sample_parallel else tokens[response_prob.argmax().item(), :cur_pos+1]
    
    def clear_cache(self) -> None:
        for att in self.decoder.attentions:
            att._clear_cache()
        for head in self.token_heads:
            for att in head[0].attentions:
                att._clear_cache()

def mtp_training_step(self, data, pad_id) -> float:
    step_loss = 0.0
    n = 0

    torch.compiler.cudagraph_mark_step_begin()
    
    x = data[0]
    y = data[1]
    
    y_fill = torch.tensor([[pad_id] * self.model.args.n_token_heads] * y.shape[0], dtype=y.dtype, device=y.device)
    y = torch.hstack([y, y_fill])

    z = self.model.get_z(x)
    # z, moe_weights = self.model.get_z(x)
    zd = z.detach()
    zd.requires_grad = True

    for i in range(self.model.args.n_token_heads):
        preds = self.model.get_heads(i, zd)
        loss = F.cross_entropy(
            preds.view(-1, self.model.args.vocab_size),
            y[:, i:-(self.model.args.n_token_heads - i)].reshape(-1),
            ignore_index=pad_id, label_smoothing=0.1)
        step_loss += loss.item()
        n += 1
        loss.backward()

    z.backward(gradient=zd.grad)

    return step_loss / n

@torch.inference_mode()
def mtp_inference_step(self, data, pad_id) -> float:

    x = data[0]
    y = data[1]
    
    z = self.model.get_z(x)
    # z, moe_weights = self.model.get_z(x)

    preds = self.model.get_heads(0, z)
    loss = F.cross_entropy(
        preds.view(-1, self.model.args.vocab_size),
        y.reshape(-1),
        ignore_index=pad_id)

    return loss.item()