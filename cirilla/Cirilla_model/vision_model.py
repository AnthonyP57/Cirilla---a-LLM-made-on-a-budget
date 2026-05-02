from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from cirilla.LLM_pieces import DynamicTanh, Dynamic_erf
from .blocks import Decoder, DecoderArgs, InputEmbeddings, EmbedArgs, SwinEncoder, SwinArgs
from .modules import CirillaBaseModel


@dataclass
class VisionArgs(DecoderArgs):
    """Combined args for CirillaVision (decoder + Swin encoder)."""
    vocab_size: int = 60_000
    out_bias: bool = False

    # Swin encoder
    img_size: int = 224
    patch_size: int = 4
    in_channels: int = 3
    swin_embed_dim: int = 96
    swin_depths: tuple = (2, 2, 6, 2)
    swin_num_heads: tuple = (3, 6, 12, 24)
    swin_window_size: int = 7          # Swin attention window; NOT the decoder's sliding window_size
    swin_mlp_ratio: float = 4.0
    swin_dropout: float = 0.0

    @property
    def swin_out_dim(self) -> int:
        return self.swin_embed_dim * (2 ** (len(self.swin_depths) - 1))

    def to_swin_args(self) -> SwinArgs:
        return SwinArgs(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.swin_embed_dim,
            depths=self.swin_depths,
            num_heads=self.swin_num_heads,
            window_size=self.swin_window_size,
            mlp_ratio=self.swin_mlp_ratio,
            dropout=self.swin_dropout,
            dtype_str=self.dtype_str,
            device=self.device,
        )

    @property
    def n_img_tokens(self) -> int:
        """Number of image tokens produced by the Swin encoder for a single image."""
        grid = self.img_size // self.patch_size
        for _ in range(len(self.swin_depths) - 1):
            grid = (grid + grid % 2) // 2
        return grid * grid


class CirillaVision(
            nn.Module,
            CirillaBaseModel,
            pipeline_tag="image-text-to-text",
            library_name="pytorch",
            license="mit"
    ):
    """
    Vision-language model: Swin Transformer encodes the image into prefix tokens,
    then Cirilla's causal decoder generates text conditioned on those tokens.

    Forward:
        image   : (B, C, H, W)
        text_ids: (B, T)  — token IDs (input sequence, not shifted)
        returns : logits (B, n_img_tokens + T, vocab_size)

    Training loss should be computed only over the text positions (index n_img_tokens onward),
    shifting targets by 1 as usual for causal LM.
    """

    def __init__(self, args: VisionArgs = None):
        super().__init__()
        if isinstance(args, dict):
            args = VisionArgs(**args)
        if args is None:
            args = VisionArgs()
        self.args = args
        self._prepare_model()

    def _prepare_model(self):
        a = self.args

        self.swin = SwinEncoder(a.to_swin_args())
        # project image features from swin_out_dim to decoder dim
        self.img_proj = nn.Linear(a.swin_out_dim, a.dim, bias=False)

        self.emb = InputEmbeddings(EmbedArgs(vocab_size=a.vocab_size, dim=a.dim))

        if a.layer_norm == "RMSNorm":
            self.layer_norm = nn.RMSNorm(a.dim)
        elif a.layer_norm == "Derf":
            self.layer_norm = Dynamic_erf(a.dim)
        elif a.layer_norm == "DyT":
            self.layer_norm = DynamicTanh(a.dim)
        else:
            raise ValueError(f"allowed layer norms: 'RMSNorm', 'Derf', 'DyT'; got: {a.layer_norm}")

        self.decoder = Decoder(a)
        self.output = nn.Linear(a.dim, a.vocab_size, bias=a.out_bias)

        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.to(a.device, dtype=a.dtype)

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """image: (B, C, H, W) → (B, N_img, dim)"""
        device = next(self.parameters()).device
        feats = self.swin(image.to(device=device, dtype=self.args.dtype))  # (B, N_img, swin_out_dim)
        return self.img_proj(feats)                     # (B, N_img, dim)

    def pred(self, image: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        img_tokens = self._encode_image(image)          # (B, N_img, dim)
        txt_tokens = self.emb(text_ids)                 # (B, T, dim)
        x = torch.cat([img_tokens, txt_tokens], dim=1) # (B, N_img + T, dim)

        if self.args.output_moe_weights:
            x, moe_weights = self.decoder(x)
            x = self.output(self.layer_norm(x))
            return x, moe_weights

        x = self.decoder(x)
        return self.output(self.layer_norm(x))          # (B, N_img + T, vocab_size)

    def forward(self, image: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        return self.pred(image, text_ids)

    @torch.no_grad()
    def infer(self, image: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        if self.args.output_moe_weights:
            logits, _ = self.pred(image, text_ids)
            return logits
        return self.pred(image, text_ids)

    @torch.no_grad()
    def generate(self,
                 image: torch.Tensor,
                 prompt_ids: torch.Tensor,
                 max_new_tokens: int = 256,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 termination_tokens: Optional[list] = None,
                 ) -> torch.Tensor:
        """
        Autoregressively generate text given an image and a text prompt.

        image      : (1, C, H, W)
        prompt_ids : (1, T_prompt)
        returns    : (1, T_prompt + generated) token IDs
        """
        text_ids = prompt_ids
        for _ in range(max_new_tokens):
            logits = self.infer(image, text_ids)            # (1, N_img+T, vocab_size)
            next_logits = logits[:, -1, :] / temperature   # (1, vocab_size)

            if top_k is not None:
                v, i = torch.topk(next_logits, top_k)
                probs = torch.zeros_like(next_logits).scatter_(1, i, F.softmax(v, dim=-1))
            else:
                probs = F.softmax(next_logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            text_ids = torch.cat([text_ids, next_token], dim=1)

            if termination_tokens is not None and next_token.item() in termination_tokens:
                break

        return text_ids

    @torch.no_grad()
    def infer_with_cache(self, x: torch.Tensor, cur_pos: int, max_batch: int = 1,
                         chunked_prefill: bool = False,
                         non_finished_ids: torch.Tensor = None) -> torch.Tensor:
        # 4-D raw image → Swin encoder
        # 3-D pre-encoded float tokens → pass through directly (already projected)
        # 2-D token IDs → word embedding
        if x.ndim == 4:
            x = self._encode_image(x)
        elif x.ndim == 2:
            x = self.emb(x)

        if self.args.output_moe_weights:
            for attention, moe in zip(self.decoder.attentions, self.decoder.smoes):
                x = x + attention.forward_with_cache(x, cur_pos, max_batch, chunked_prefill, non_finished_ids)
                moe_out, moe_weights = moe(x)
                x = x + moe_out
            x = self.layer_norm(x)
            x = self.output(x)
            return x
        else:
            for attention, moe in zip(self.decoder.attentions, self.decoder.smoes):
                x = x + attention.forward_with_cache(x, cur_pos, max_batch, chunked_prefill, non_finished_ids)
                x = x + moe(x)[0]
            x = self.layer_norm(x)
            x = self.output(x)
            return x

    def generate_kv_cache(self,
                          image: torch.Tensor,
                          max_new_tokens: int = 1024,
                          top_k: int = None,
                          top_p: float = None,
                          temperature: float = 1.0,
                          termination_tokens: list[int] = None,
                          ) -> torch.Tensor:
        """KV-cache generation from an image.  image: (B, C, H, W) → (B, T) token IDs.

        Mirrors Cirilla.generate_kv_cache.  The image is the prompt (prefilled in
        one shot), then tokens are generated one-by-one using the rolling KV cache.
        """
        batch_size = image.shape[0]
        n_img = self.args.n_img_tokens
        device = next(self.parameters()).device

        # encode image outside inference_mode so gradients are not needed and
        # the expensive Swin forward is clearly separated from the token loop
        img_tokens = self._encode_image(image)      # (B, N_img, dim)

        generated = torch.zeros(batch_size, max_new_tokens, dtype=torch.long, device=device)
        non_finished_ids = torch.arange(batch_size, device=device)

        with torch.inference_mode():

            logits = self.infer_with_cache(img_tokens, cur_pos=0, max_batch=batch_size)
            next_token_logits = logits[:, -1, :]    # (B, vocab_size)
            cur_pos = n_img

            gen_pos = 0
            while gen_pos < max_new_tokens:

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
                    mask = torch.zeros_like(next_token_logits, dtype=torch.bool).scatter_(
                        1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[mask] = float('-inf')
                    probs = F.softmax(next_token_logits, dim=-1)

                else:  # greedy
                    max_arg = torch.argmax(next_token_logits, dim=-1)
                    probs = torch.zeros_like(next_token_logits)
                    probs[range(probs.size(0)), max_arg] = 1.0

                next_token_sample = torch.multinomial(probs, num_samples=1).squeeze(1)  # (active_b,)
                generated[non_finished_ids, gen_pos] = next_token_sample

                if termination_tokens is not None:
                    has_terminated = torch.isin(
                        next_token_sample,
                        torch.tensor(termination_tokens, device=device, dtype=next_token_sample.dtype)
                    )
                    non_finished_ids = non_finished_ids[~has_terminated]

                if non_finished_ids.size(0) == 0:
                    gen_pos += 1
                    break

                if termination_tokens is not None and has_terminated.any():
                    input_token = next_token_sample[~has_terminated].unsqueeze(1)
                else:
                    input_token = next_token_sample.unsqueeze(1)

                logits = self.infer_with_cache(input_token, cur_pos=cur_pos, non_finished_ids=non_finished_ids)
                next_token_logits = logits[:, -1, :]
                cur_pos += 1
                gen_pos += 1

        self.clear_cache()
        return generated[:, :gen_pos]   # (B, T_generated)

    def clear_cache(self) -> None:
        for att in self.decoder.attentions:
            att._clear_cache()
