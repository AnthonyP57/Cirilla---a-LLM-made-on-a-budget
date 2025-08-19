import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import torch
from activations import get_activation
from functools import partial
from megablocks import Arguments, MoE, dMoE

activation = get_activation("kernels-community/activation")

@dataclass
class SMoEArgs:
    num_experts:int=8
    k:int=2
    dim:int=128
    dtype_str:str = 'bfloat16'

    d_ff:int=256 # hidden dim

    @property
    def dtype(self):
        return getattr(torch, self.dtype_str)

@dataclass
class ExpertArgs:
    dim:int=128
    d_ff:int=256 # hidden dim
    assert d_ff % 2 == 0
    drop:float=0.1

class Expert(nn.Module):
    def __init__(self, args: ExpertArgs):
        super().__init__()
        self.dim = args.dim
        self.d_ff = args.d_ff

        self.w1 = nn.Linear(args.dim, args.d_ff * 2)
        self.w2 = nn.Linear(args.d_ff, args.dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:  # SwiGLU 
        x = self.w1(x)

        d = x.shape[-1] // 2
        out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)

        activation.silu_and_mul(out, x)

        out = self.w2(out)
        return out

class SMoE(nn.Module):
    def __init__(self, args:SMoEArgs, experts:list[Expert]):
        super().__init__()
        self.n_experts = args.num_experts
        self.k = args.k
        self.gating = nn.Linear(args.dim, args.num_experts)
        self.experts = nn.ModuleList(experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gating(x)                       # (B,S,E)
        topk_vals, topk_idx = torch.topk(logits, self.k, dim=-1)
        topk_w = F.softmax(topk_vals, dim=-1)        # (B,S,k)
        one_hot = F.one_hot(topk_idx, num_classes=self.n_experts).to(x.dtype)
        weights_per_expert = (one_hot * topk_w.unsqueeze(-1)).sum(dim=2)  # (B,S,E)

        out = torch.zeros_like(x)
        for ex_idx, expert in enumerate(self.experts):
            w = weights_per_expert[..., ex_idx].unsqueeze(-1)  # (B,S,1)
            out = out + w * expert(x)                         # expert(x) -> (B,S,D)
        return out

moe_kernel = get_activation("RedHatAI/moe")

class FusedSMOE(nn.Module): # inference only
    def __init__(self, args:SMoEArgs):
        super().__init__()

        self.args = args

        self.gating = nn.Linear(args.dim, args.num_experts, dtype=args.dtype)
        self.w1 = nn.Parameter(torch.randn(args.num_experts, args.d_ff, args.dim, dtype=args.dtype))
        self.w2 = nn.Parameter(torch.randn(args.num_experts, args.dim, args.dim, dtype=args.dtype))

        self.fused_moe = partial(moe_kernel.fused_moe, topk=self.args.k, global_num_experts=self.args.num_experts, renormalize=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        hidden = x.view(-1, self.args.dim)
        return self.fused_moe(hidden_states=hidden,
                              w1=self.w1,
                              w2=self.w2,
                              gating_output=self.gating(hidden)).view(B, S, D)

@dataclass
class MegablockArgs:
    num_experts: int = 4
    k: int = 2
    dim: int = 128
    d_ff: int = 256
    capacity_factor: float = 1.0
    impl: str = "grouped"   # or "sparse" Sparse MLP is not supported with triton >=3.2.0
    dtype_str:str = 'bfloat16'

    @property
    def dtype(self):
        return getattr(torch, self.dtype_str)

class MegablockMoE(nn.Module):
    def __init__(self, args:MegablockArgs):
        super().__init__()

        self.args = args

        init_method = torch.nn.init.xavier_uniform_
        
        self.moe = MoE(
            Arguments(
                hidden_size=args.dim,
                ffn_hidden_size=args.d_ff,
                moe_num_experts=args.num_experts,
                moe_capacity_factor=args.capacity_factor,
                moe_top_k=args.k,
                init_method=init_method,
                memory_optimized_mlp=True,
                mlp_type="mlp",
                mlp_impl=args.impl,
                fp16=False,
                bf16=True,
            )
        ).cuda().to(args.dtype)

    def forward(self, x: torch.Tensor):
        # MegaBlocks expects (seq, batch, dim)
        x_mb = x.transpose(0, 1).contiguous()

        out, _ = self.moe(x_mb)

        out = out.transpose(0, 1)  # back to (batch, seq, dim)
        return out

class MegablockdMoE(nn.Module):
    def __init__(self, args:MegablockArgs):
        super().__init__()

        self.args = args

        init_method = torch.nn.init.xavier_uniform_
        
        self.moe = dMoE(
            Arguments(
                hidden_size=args.dim,
                ffn_hidden_size=args.d_ff,
                moe_num_experts=args.num_experts,
                moe_capacity_factor=args.capacity_factor,
                moe_top_k=args.k,
                init_method=init_method,
                memory_optimized_mlp=True,
                mlp_type="mlp",
                mlp_impl=args.impl,
                fp16=False,
                bf16=True,
            )
        ).cuda().to(args.dtype)

    def forward(self, x: torch.Tensor):
        # MegaBlocks expects (seq, batch, dim)
        x_mb = x.transpose(0, 1).contiguous()

        out, _ = self.moe(x_mb)

        out = out.transpose(0, 1)  # back to (batch, seq, dim)
        return out

from fairscale.nn import MOELayer, Top2Gate




if __name__=='__main__':
    import time

    def benchmark(model, x, label=""):
        model.train()
        x = x.contiguous()

        # Warmup (not measured)
        out = model(x)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)

        fwd_times, bwd_times = [], []
        fwd_mems, bwd_mems = [], []

        for _ in range(3):
            # Forward
            torch.cuda.synchronize()
            start_mem = torch.cuda.memory_allocated()
            start_time = time.time()

            out = model(x)
            loss = out.sum()

            torch.cuda.synchronize()
            fwd_times.append(time.time() - start_time)
            fwd_mems.append(torch.cuda.memory_allocated() - start_mem)

            # Backward
            torch.cuda.synchronize()
            start_mem = torch.cuda.memory_allocated()
            start_time = time.time()

            loss.backward()

            torch.cuda.synchronize()
            bwd_times.append(time.time() - start_time)
            bwd_mems.append(torch.cuda.memory_allocated() - start_mem)

            model.zero_grad(set_to_none=True)

        print(f"\n[{label}]")
        print(f"Forward time:   {sum(fwd_times)/len(fwd_times)*1000:.2f} ms")
        print(f"Backward time:  {sum(bwd_times)/len(bwd_times)*1000:.2f} ms")
        print(f"Forward memory: {sum(fwd_mems)/len(fwd_mems)/1024/1024:.2f} MB")
        print(f"Backward memory:{sum(bwd_mems)/len(bwd_mems)/1024/1024:.2f} MB")

    moe = SMoE(
        SMoEArgs(num_experts=4, k=2),
        [Expert(ExpertArgs()) for _ in range(4)]
    ).to("cuda") # hf kernel only work on cuda

    x = torch.randn(4, 1024, 128, device='cuda', requires_grad=True) # (b, seq_len, dim) ; requires grad for smoe
    # start = time.time()
    # out = moe(x)
    # out = moe(x)
    # print(time.time() - start)
    # print(out.shape)

    fused = FusedSMOE(SMoEArgs(num_experts=4, k=2)).to("cuda")
    # x = x.to(dtype=torch.bfloat16)
    # start = time.time()
    # out = fused(x)
    # out = fused(x)
    # print(time.time() - start)
    # print(out.shape)

    megamoe = MegablockMoE(MegablockArgs())

    megadmoe = MegablockdMoE(MegablockArgs())

    benchmark(moe, x, "SMoE")
    x = x.to(dtype=torch.bfloat16)
    torch.cuda.empty_cache()
    benchmark(fused, x, "FusedSMoE")
    torch.cuda.empty_cache()
    benchmark(megamoe, x, "MegablocksMoE")
    torch.cuda.empty_cache()
    benchmark(megadmoe, x, "MegablocksdMoE")