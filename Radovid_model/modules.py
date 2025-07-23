import torch

def select_torch_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device}")
        print(f"Device name: {torch.cuda.get_device_name(device)}")
        mem_gb = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
        print(f"Device memory: {mem_gb:.2f} GB")
    elif getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: mps (Apple Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")
        print("NOTE: If you have a GPU, consider using it for training.")

    return device
