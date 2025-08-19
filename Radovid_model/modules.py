import torch
import configparser
import os
from huggingface_hub import hf_hub_download
import json

def get_args_from_hub(hf_repo_id):
    from model import Args

    file_path = hf_hub_download(
        repo_id=hf_repo_id,
        filename="config.json",
    )
    with open(file_path, "r") as f:
        config = json.load(f)
    args = Args(**config[list(config.keys())[0]])

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
        if '.radovid' in files:
            path_to_cache = os.path.join(main_path, '.radovid')
            return path_to_cache

CACHE_PATH = None

def cache_or_fetch(category, variable, value=None):
    global CACHE_PATH
    if CACHE_PATH is None:
        CACHE_PATH = find_cache()
        if CACHE_PATH is None:
            CACHE_PATH = '.radovid'

    config = configparser.ConfigParser()
    if os.path.exists(CACHE_PATH):
        config.read(CACHE_PATH)

    try:
        val = config[category][variable]
        if value is not None:
            config[category][variable] = str(value)
            with open(CACHE_PATH, 'w') as c:
                config.write(c)
        return val
    except (KeyError, configparser.NoSectionError):
        if value is not None:
            if category not in config:
                config[category] = {}
            config[category][variable] = str(value)
            with open(CACHE_PATH, 'w') as c:
                config.write(c)
            return value
        else:
            return None
