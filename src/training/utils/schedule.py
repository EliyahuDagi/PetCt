"""Optimizer LR scheduling and weight EMA for diffusion training.

``build_lr_scheduler`` returns a warmup + cosine-decay schedule (the common
default for diffusion UNets) when configured, else ``None`` so callers can keep a
constant LR. ``EMA`` maintains an exponential moving average of the model weights;
sampling/validating from the EMA copy is one of the largest quality wins in
diffusion training and is otherwise absent here.
"""

import copy
import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def build_lr_scheduler(optimizer, config):
    """Warmup-then-cosine LR schedule.

    Reads (top-level config, all optional):
        lr_warmup_steps: linear warmup from 0 -> base LR over this many steps (default 0).
        lr_total_steps:  total training steps; required to enable cosine decay.
        lr_min_ratio:    floor LR as a fraction of base LR at the end (default 0.0).

    Returns a per-step LambdaLR, or ``None`` when ``lr_total_steps`` is unset
    (preserving constant-LR behavior).
    """
    if not isinstance(config, dict):
        return None
    total_steps = config.get("lr_total_steps")
    if not total_steps or int(total_steps) <= 0:
        return None
    total_steps = int(total_steps)
    warmup_steps = int(config.get("lr_warmup_steps", 0))
    min_ratio = float(config.get("lr_min_ratio", 0.0))

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


class EMA:
    """Exponential moving average of model parameters.

    Usage::

        ema = EMA(model, decay=0.9999)
        ...
        optimizer.step(); ema.update(model)
        ...
        with ema.average_parameters(model):   # swap EMA weights in for eval/sampling
            validate(model)
    """

    def __init__(self, model, decay=0.9999):
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model.state_dict())
        for k, v in self.shadow.items():
            if v.is_floating_point():
                self.shadow[k] = v.detach().clone()
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            if not v.is_floating_point():
                self.shadow[k] = v.detach().clone()
                continue
            self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=False)

    def store(self, model):
        self._backup = copy.deepcopy(model.state_dict())

    def restore(self, model):
        if self._backup is not None:
            model.load_state_dict(self._backup, strict=False)
            self._backup = None

    def average_parameters(self, model):
        return _EMASwap(self, model)

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = float(state.get("decay", self.decay))
        self.shadow = state["shadow"]


class _EMASwap:  # noqa: E303
    def __init__(self, ema, model):
        self.ema = ema
        self.model = model

    def __enter__(self):
        self.ema.store(self.model)
        self.ema.copy_to(self.model)
        return self.model

    def __exit__(self, *exc):
        self.ema.restore(self.model)
        return False
