from .blocks import (
                    VisionEmbeddingModel,
                    KeylessAttention,
                    Encoder,
                    EncoderArgs,
                    InputEmbeddings
)
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
import torch
from ..LLM_pieces import get_activation, SwiGLU
from dataclasses import dataclass

@dataclass
class CMAArgs(EncoderArgs):
    in_channels:int = 3
    patch_size:int = 14
    H:int = 16
    W:int = 16
    n_tasks:int = 2
    n_classes:int = [2, 3]
    cls_text_index:int = 0
    cls_image_index:int = 10

class CMA(
        nn.Module,
        PyTorchModelHubMixin,
        pipeline_tag="text-generation",
        library_name="pytorch",
        license="mit"
    ):
    def __init__(self, args:CMAArgs=None):
        super().__init__()
        self.args = args
        self._prepare_model()

    def _prepare_model(self):
        
        self.vision_emb = VisionEmbeddingModel(self.args.in_channels,
                                                self.args.dim,
                                                self.args.patch_size,
                                                self.args.H,
                                                self.args.W
                                                )
        self.text_emb = InputEmbeddings(self.args)
        activation = get_activation('Motif-Technologies/activation')
        self.rmsnorm = activation.layers.RMSNorm(dim=self.args.dim) if self.args.device == torch.cuda.is_available() else nn.RMSNorm(self.args.dim)
        self.encoder = Encoder(self.args)

        assert len(self.args.n_classes) == self.args.n_tasks
        self.outs = nn.ModuleList([nn.Sequential(
                                        KeylessAttention(self.args.dim),
                                        nn.Sequential(
                                            SwiGLU(self.args),
                                            nn.SiLU(),
                                            nn.Linear(self.args.dim, self.args.n_classes[i])
                                                )
                                            )
                                for i in range(self.args.n_tasks)])

        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.to(self.args.device, dtype=self.args.dtype)

    def pred(self, texts, images, cls_image_token_index=None):
        texts = self.text_emb(texts)
        images = self.vision_emb(images)
        if cls_image_token_index is not None:
            cls_i_em = self.text_emb(torch.tensor([cls_image_token_index]).to(self.args.device)).unsqueeze(0).expand(images.shape[0], -1, -1)
            x = torch.cat([texts, cls_i_em, images], dim=1)
        else:
            x = torch.cat([texts, images], dim=1)
        x = self.encoder(x)
        x = self.rmsnorm(x)
        cls_text, cls_image = x[:, self.args.cls_text_index], x[:, self.args.cls_image_index]
        tasks = [out(cls_text, cls_image) for out in self.outs]
        return tasks