from cirilla.LLM_pieces import SMoE, MegablockdMoE, MegablockMoE
from cirilla.LLM_pieces.SMoE import MegablockArgs, SMoEArgs, SwiGLUArgs, SwiGLU
import torch

def test_expert():
    expert_args = SwiGLUArgs()
    expert = SwiGLU(expert_args).to('cuda')
    x = torch.randn(4, 10, expert_args.dim, device='cuda')
    out = expert(x)
    assert out.shape == (4, 10, expert_args.dim)

    loss = out.sum()
    loss.backward()

    grads_ok = all(p.grad is not None for p in expert.parameters() if p.requires_grad)
    if not grads_ok:
        for name, p in expert.named_parameters():
            if p.requires_grad and p.grad is None:
                print(f"Parameter {name} has no gradient.")
        raise AssertionError("Not all parameters have gradients after initialization.")
    
def test_pytorch_smoe():
    smoeargs = SMoEArgs(num_experts=4, k=2, dim=128, d_ff=256, device='cuda')
    experts = [SwiGLU(SwiGLUArgs(dim=128, d_ff=256)).to('cuda') for _ in range(smoeargs.num_experts)]
    model = SMoE(smoeargs, experts).to('cuda')

    x = torch.randn(4, 10, smoeargs.dim, device='cuda')
    out = model(x)
    assert isinstance(out, tuple)
    assert isinstance(out[0], torch.Tensor)
    assert isinstance(out[1], torch.Tensor)

    out, weights = out
    assert out.shape == (4, 10, smoeargs.dim)
    assert weights.shape == (4, 10, smoeargs.num_experts)

    loss = out.sum()
    loss.backward()

    grads_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    if not grads_ok:
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is None:
                print(f"Parameter {name} has no gradient.")
        raise AssertionError("Not all parameters have gradients after initialization.")
    
def test_megablock_moe():
    megablock_args = MegablockArgs(num_experts=4, k=2, dim=128, d_ff=256, device='cuda')
    model = MegablockMoE(megablock_args).to('cuda')

    x = torch.randn(4, 10, megablock_args.dim, device='cuda', dtype=torch.bfloat16)
    out = model(x)
    assert isinstance(out, tuple)
    assert isinstance(out[0], torch.Tensor)

    out = out[0]
    assert out.shape == (4, 10, megablock_args.dim)

    loss = out.sum()
    loss.backward()

    grads_ok = all(p.grad is not None for n, p in model.named_parameters() if p.requires_grad if n != "moe.experts.bias")
    if not grads_ok:
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is None:
                print(f"Parameter {name} has no gradient.")
        raise AssertionError("Not all parameters have gradients after initialization.")

def test_megablock_dmoe():
    megablock_args = MegablockArgs(num_experts=4, k=2, dim=128, d_ff=256, device='cuda')
    model = MegablockdMoE(megablock_args).to('cuda')

    x = torch.randn(4, 10, megablock_args.dim, device='cuda', dtype=torch.bfloat16)
    out = model(x)
    assert isinstance(out, tuple)
    assert isinstance(out[0], torch.Tensor)

    out = out[0]
    assert out.shape == (4, 10, megablock_args.dim)

    loss = out.sum()
    loss.backward()

    grads_ok = all(p.grad is not None for n, p in model.named_parameters() if p.requires_grad if n != "moe.experts.bias")
    if not grads_ok:
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is None:
                print(f"Parameter {name} has no gradient.")
        raise AssertionError("Not all parameters have gradients after initialization.")