from pathlib import Path

import torch


def save_checkpoint(state, path):
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path_obj)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu")
