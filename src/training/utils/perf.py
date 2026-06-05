"""GPU performance helpers: TF32/cuDNN backend setup, bf16 autocast,
channels-last memory format, and torch.compile.

Everything here is centralized so the four train_* entry points stay in sync, and
every optimization is individually toggleable via an environment variable so a
problematic one can be disabled without touching code (e.g. if torch.compile hits
a graph break on a MONAI model, or channels_last_3d misbehaves):

    PETCT_TF32           TF32 matmul + cuDNN paths        (default on)
    PETCT_AMP            bf16 autocast forward            (default on)
    PETCT_CHANNELS_LAST  channels_last / channels_last_3d (default on)
    PETCT_COMPILE        torch.compile the trained model  (default on)

All four are gated on the *actual training device*: they activate only after
``configure_backends(device)`` is called with a CUDA device. Training on
``--device cpu`` (even on a machine that has a GPU) leaves them fully off, which
keeps CPU runs bit-for-bit deterministic (the resume test relies on this).

The compile-vs-state contract: ``maybe_compile`` returns torch.compile's wrapper
(an ``OptimizedModule`` holding the real module under ``_orig_mod``). Its
``state_dict`` keys carry an ``_orig_mod.`` prefix, which the uncompiled inference
and inflation paths can't load. So callers keep two handles -- the compiled model
for the *forward* pass and ``unwrap_model(...)`` for every state_dict / load /
EMA operation -- which keeps checkpoints prefix-free and loadable everywhere.
"""

import os
from contextlib import nullcontext

import torch

# Set by configure_backends(device); gates every GPU optimization below. Off
# until a CUDA device is explicitly configured, so a process that never calls
# configure_backends (or trains on CPU) gets plain deterministic eager fp32.
_ON_CUDA = False


def _enabled(name, default=True):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off", "")


def configure_backends(device, logger=None):
    """Resolve whether we're training on CUDA and, if so, enable TF32 matmul /
    cuDNN paths and cuDNN autotuning. Call once at the start of training, before
    building the model. A no-op for CPU devices."""
    global _ON_CUDA
    try:
        _ON_CUDA = torch.device(device).type == "cuda" and torch.cuda.is_available()
    except (TypeError, RuntimeError, ValueError):
        _ON_CUDA = False
    if not _ON_CUDA:
        if logger is not None:
            logger.info("perf: device=%s -> GPU optimizations off", device)
        return
    tf32 = _enabled("PETCT_TF32")
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    # Fixed-size slices/crops here, so let cuDNN autotune the fastest kernels.
    torch.backends.cudnn.benchmark = True
    if logger is not None:
        logger.info(
            "perf: cuda tf32=%s cudnn.benchmark=True amp=%s channels_last=%s compile=%s",
            tf32, amp_enabled(), channels_last_enabled(), _enabled("PETCT_COMPILE"),
        )


def amp_enabled():
    return _ON_CUDA and _enabled("PETCT_AMP")


def autocast(enabled=True):
    """bf16 autocast on CUDA -- no GradScaler needed (bf16 has fp32's exponent
    range). Honors ``PETCT_AMP``; a plain no-op context otherwise."""
    if enabled and amp_enabled():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def channels_last_enabled():
    return _ON_CUDA and _enabled("PETCT_CHANNELS_LAST")


def to_model_memory_format(model, spatial_dims):
    """Convert a model's conv weights to channels_last (2D) / channels_last_3d
    (3D). Non-4D/5D params are left untouched by ``.to(memory_format=...)``."""
    if not channels_last_enabled():
        return model
    fmt = torch.channels_last_3d if spatial_dims == 3 else torch.channels_last
    return model.to(memory_format=fmt)


def to_input_memory_format(t):
    """Convert a 4D (NCHW) or 5D (NCDHW) input tensor to the matching
    channels-last layout so it lines up with a channels_last model."""
    if not channels_last_enabled() or not torch.is_tensor(t):
        return t
    if t.ndim == 5:
        return t.contiguous(memory_format=torch.channels_last_3d)
    if t.ndim == 4:
        return t.contiguous(memory_format=torch.channels_last)
    return t


def unwrap_model(model):
    """Return the underlying module behind torch.compile's OptimizedModule, so
    state_dict keys stay free of the ``_orig_mod.`` prefix. Returns ``model``
    unchanged when it was never compiled."""
    return getattr(model, "_orig_mod", model)


def maybe_compile(model, logger=None):
    """torch.compile the model when enabled; fall back to eager on any failure so
    training never hard-fails on a compile/graph issue."""
    if not (_ON_CUDA and _enabled("PETCT_COMPILE")):
        return model
    if not hasattr(torch, "compile"):
        return model
    try:
        compiled = torch.compile(model)
        if logger is not None:
            logger.info("perf: torch.compile enabled")
        return compiled
    except Exception as exc:  # pragma: no cover - environment dependent
        if logger is not None:
            logger.warning("perf: torch.compile failed (%s); using eager", exc)
        return model
