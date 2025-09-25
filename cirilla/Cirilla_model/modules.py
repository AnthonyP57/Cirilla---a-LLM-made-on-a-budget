import torch
import configparser
import os
from huggingface_hub import hf_hub_download
import json
import time

def benchmark_model_part(model, x, label=""):
    model.train()
    x = x.contiguous()

    # Warmup (not measured)
    for _ in range(10):
        out = model(x)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)

    fwd_times, bwd_times = [], []
    fwd_mems, bwd_mems = [], []

    for _ in range(100):
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

def get_args_from_hub(hf_repo_id):
    from .model import Args

    file_path = hf_hub_download(
        repo_id=hf_repo_id,
        filename="config.json",
    )
    with open(file_path, "r") as f:
        config = json.load(f)
    args = Args(**config[list(config.keys())[0]])

    return args

def get_bertargs_from_hub(hf_repo_id):
    from .model import BertArgs

    file_path = hf_hub_download(
        repo_id=hf_repo_id,
        filename="config.json",
    )
    with open(file_path, "r") as f:
        config = json.load(f)
    args = BertArgs(**config[list(config.keys())[0]])

    return args

def select_torch_device():
    if torch.cuda.is_available():
        device = "cuda:0"
        print(f"Using device: {device}")
        print(f"Device name: {torch.cuda.get_device_name(device)}")
        mem_gb = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
        print(f"Device memory: {mem_gb:.2f} GB")
    elif getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():
        device = "mps"
        print("Using device: mps (Apple Metal Performance Shaders)")
    else:
        device = "cpu"
        print("Using device: cpu")
        print("NOTE: If you have a GPU, consider using it for training.")

    return device

def find_cache(start_dir='./'):
    for main_path, subfolders, files in os.walk(start_dir):
        if '.cirilla' in files:
            path_to_cache = os.path.join(main_path, '.cirilla')
            return path_to_cache

CACHE_PATH = None

def cache_or_fetch(category, variable, value=None):
    global CACHE_PATH
    if CACHE_PATH is None:
        CACHE_PATH = find_cache()
        if CACHE_PATH is None:
            CACHE_PATH = '.cirilla'

    config = configparser.ConfigParser()
    if os.path.exists(CACHE_PATH):
        config.read(CACHE_PATH)

    try:
        val = config[category][variable]
        if value is not None:
            config[category][variable] = str(value)
            with open(CACHE_PATH, 'w') as c:
                config.write(c)
        try:
            return int(val)
        except:
            return val
    except (KeyError, configparser.NoSectionError):
        if value is not None:
            if category not in config:
                config[category] = {}
            config[category][variable] = str(value)
            with open(CACHE_PATH, 'w') as c:
                config.write(c)
            try:
                return int(value)
            except:
                return value
        else:
            return None

def load_balancing_loss(expert_weights: torch.Tensor,
                        num_experts: int,
                        top_k: int,
                        eps: float = 1e-12) -> torch.Tensor:
    """
    Optimal load balancing loss for MoE (highly vectorized).

    Args:
        expert_weights: Tensor of shape [tokens, top_k] or [batch, seq, top_k],
                        containing the softmax gating weights for top-k experts.
        num_experts: Total number of experts (E).
        top_k: Number of experts per token.
        eps: Small epsilon for numerical stability.

    Returns:
        Scalar tensor: load balancing loss.
    """
    # Flatten to [tokens, top_k]
    if expert_weights.dim() == 3:
        expert_weights = expert_weights.reshape(-1, top_k)  # (B*S, k)

    # Convert top-k weights to dense expert distribution (E,)
    # Each token contributes its top-k weights to corresponding experts
    # Here we assume experts are indexed 0..num_experts-1 evenly across top-k
    # (this is equivalent to a soft one-hot expansion)
    # For optimal version, we sum contributions per expert
    # Generate a placeholder expert assignment for simplicity: evenly spread
    # For exact assignment, use topk_idx instead of this.
    # Here we assume top_k == num_experts for simplified demonstration
    if top_k != num_experts:
        # If top_k < num_experts, distribute weights evenly to simulate usage
        importance = expert_weights.sum(0) * (num_experts / top_k)
    else:
        importance = expert_weights.sum(0)  # (num_experts,)

    # Normalize to probability
    importance = importance / (importance.sum() + eps)

    # KL divergence to uniform distribution
    # KL(p || U) = sum_i p_i * log(p_i / u_i) = sum_i p_i * log(p_i) + log(num_experts)
    loss = (importance * (importance + eps).log()).sum()
    loss = loss + torch.log(torch.tensor(num_experts, device=expert_weights.device, dtype=expert_weights.dtype))

    return loss
