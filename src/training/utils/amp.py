from contextlib import nullcontext

import torch


def maybe_autocast(enabled: bool):
    if enabled and torch.cuda.is_available():
        return torch.cuda.amp.autocast()
    return nullcontext()
