import random
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(state, path):
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path_obj)


def load_checkpoint(path):
    # weights_only=False: our checkpoints embed RNG state (numpy arrays, Python
    # random tuples) for exact resume, which the torch>=2.6 safe loader rejects.
    # These are self-produced, trusted files.
    return torch.load(path, map_location="cpu", weights_only=False)


def save_training_checkpoint(save_dir, state, is_best):
    """Always write last.pt; additionally write best.pt when is_best is True.

    ``state`` should include at least {"model": state_dict, "config": ...} so the
    inference CLI can rebuild the model without external metadata. For resumable
    runs it also carries {"optimizer", "rng", "best_val", "step", "epoch"} (see
    ``capture_rng_state`` / ``resolve_resume_path``).
    """
    save_dir = Path(save_dir)
    save_checkpoint(state, save_dir / "last.pt")
    if is_best:
        save_checkpoint(state, save_dir / "best.pt")


def capture_rng_state(sampler_rng=None):
    """Snapshot all RNG states needed to reproduce a run exactly.

    Covers Python ``random``, NumPy's global RNG, the torch CPU (and CUDA) RNG,
    and the optional per-run ``np.random.RandomState`` sampler that drives slice
    sampling. Returned dict is plain/picklable so it rides inside a checkpoint.
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if sampler_rng is not None:
        state["sampler"] = sampler_rng.get_state()
    return state


def restore_rng_state(state, sampler_rng=None):
    """Restore RNG states captured by :func:`capture_rng_state` (no-op if None)."""
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        except Exception:
            pass
    if sampler_rng is not None and "sampler" in state:
        sampler_rng.set_state(state["sampler"])


def resolve_resume_path(save_dir, resume):
    """Resolve the ``--resume`` CLI value to a checkpoint path, or None.

    ``resume`` is falsy -> no resume; ``True`` or ``"auto"`` -> ``<save_dir>/last.pt``;
    any other string -> that path. Returns None if the resolved file is missing.
    """
    if not resume:
        return None
    if resume is True or resume == "auto":
        path = Path(save_dir) / "last.pt"
    else:
        path = Path(resume)
    return path if path.exists() else None
