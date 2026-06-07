"""Train the 2D AutoencoderKL (encode/decode).

Trains on pooled PET slices (NAC + AC) by default so NAC and AC share one latent
space for the downstream NAC->AC translation. Emits structured JSONL metrics and
saves best/last checkpoints (with config embedded) for the Train Viewer.
"""

import argparse
import os
import random

import numpy as np
import torch

from src.training.data import (
    PatientVolumeCache,
    ae_pool_volumes,
    load_patient_volumes,
    make_depth_split,
    make_patient_split,
    sample_slices,
)
from src.training.dataset_index import enumerate_patients
from src.training.models.autoencoder2d import build_autoencoder_2d
from src.training.utils.checkpointing import (
    capture_rng_state,
    load_checkpoint,
    resolve_resume_path,
    restore_rng_state,
    save_training_checkpoint,
)
from src.training.utils.image_metrics import image_quality_metrics
from src.training.utils.logging import setup_logging
from src.training.utils.metrics import MetricsWriter
from src.training.utils.perf import (
    autocast,
    configure_backends,
    maybe_compile,
    to_input_memory_format,
    to_model_memory_format,
)

TASK = "ae2d"
SPATIAL_DIMS = 2


def _ae_metrics(model, batch):
    batch = to_input_memory_format(batch)
    with autocast():
        outputs = model(batch)
        if not (isinstance(outputs, (list, tuple)) and len(outputs) >= 3):
            raise ValueError("AutoencoderKL output is unexpected.")
        recon, z_mu, z_sigma = outputs[:3]
        recon_loss = torch.mean(torch.abs(recon - batch))
        kl_loss = torch.mean(0.5 * (z_mu.pow(2) + z_sigma.pow(2) - 1.0 - torch.log(z_sigma.pow(2) + 1.0e-6)))
        loss = recon_loss + (1.0e-6 * kl_loss)
    metrics = {
        "loss": float(loss.detach().cpu()),
        "recon_l1": float(recon_loss.detach().cpu()),
        "kl": float(kl_loss.detach().cpu()),
    }
    return loss, metrics, recon


def train_step(model, batch, optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, metrics, _ = _ae_metrics(model, batch)
    loss.backward()
    optimizer.step()
    return metrics


@torch.no_grad()
def eval_step(model, batch):
    model.eval()
    _, metrics, recon = _ae_metrics(model, batch)
    # Image-quality metrics on the reconstruction (val rows carry these).
    metrics.update(image_quality_metrics(recon, batch))
    return metrics


def build_model(config):
    return build_autoencoder_2d(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, nargs="+", help="One or more dataset roots / patient folders")
    parser.add_argument("--patient_index", type=int, default=0)
    parser.add_argument("--config", default=None, help="Optional YAML config path")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=25, help="Validate every N global steps")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--val_batches", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--slice_size", type=int, default=128)
    parser.add_argument("--modality", choices=["pet", "ct"], default="pet")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume training. Bare flag resumes from <save_dir>/last.pt; pass a path to resume from a specific checkpoint.",
    )
    args = parser.parse_args()

    default_config = {
        "seed": 42,
        "output_dir": "outputs/ae2d",
        "batch_size": 4,
        "learning_rate": 1.0e-4,
        # Wider latent (8) -> richer AE code; lifts the recon ceiling that caps the
        # downstream 3D diffusion. The launcher passes no --config, so this in-script
        # default is the source of truth for real runs (ae2d.yaml mirrors it).
        "latent_channels": 8,
        "model": {
            "in_channels": 1,
            "out_channels": 1,
            "block_out_channels": [16, 32, 64],
            "num_res_blocks": 1,
        },
    }

    config = _load_config(args.config, default_config)
    save_dir = args.save_dir or config["output_dir"]
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate

    resume_path = resolve_resume_path(save_dir, args.resume)

    logger = setup_logging(save_dir)
    metrics_writer = MetricsWriter(save_dir, TASK, append=resume_path is not None)

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    configure_backends(device, logger)

    # raw_model owns the weights (state_dict / resume); model is the (optionally)
    # compiled handle used only for the forward pass. They share parameters.
    raw_model = build_model(config).to(device)
    raw_model = to_model_memory_format(raw_model, SPATIAL_DIMS)
    optimizer = torch.optim.Adam(raw_model.parameters(), lr=float(config.get("learning_rate", 1.0e-4)))
    model = maybe_compile(raw_model, logger)

    rng = np.random.RandomState(seed)
    batch_size = int(config.get("batch_size", 4))

    # Enumerate patients across all roots (missing_ok lets monkeypatched / GUI
    # callers pass non-existent paths). With <=1 enumerated patient we fall back
    # to the legacy single-patient depth split so existing behavior is unchanged.
    patients = enumerate_patients(args.data_dir, missing_ok=True)
    multi_patient = len(patients) > 1

    if not multi_patient:
        vols = load_patient_volumes(args.data_dir[0], args.patient_index, device=device)
        pool_vols = ae_pool_volumes(vols, args.modality)
        if not pool_vols and args.modality == "pet":
            logger.info("No PET volumes found; falling back to CT for AE training.")
        if not pool_vols:
            raise ValueError("No volumes available for AE training.")
        train_pool, val_pool = make_depth_split(seed, args.val_fraction)
        logger.info("Training on 1 patient (train 1 / val 1) across %d roots.", len(args.data_dir))

        def sample_train():
            return sample_slices(pool_vols, train_pool, batch_size, args.slice_size, rng)

        def sample_val():
            return sample_slices(pool_vols, val_pool, batch_size, args.slice_size, rng)
    else:
        cache = PatientVolumeCache(patients, device=device, max_cached=4)
        train_idx, val_idx = make_patient_split(len(patients), args.val_fraction, seed)
        full_pool = np.linspace(0.0, 1.0, num=128, endpoint=False)
        logger.info(
            "Training on %d patients (train %d / val %d) across %d roots.",
            len(patients), len(train_idx), len(val_idx), len(args.data_dir),
        )

        def _pool_for(idx):
            vols = cache.get(idx)
            return ae_pool_volumes(vols, args.modality)

        def _sample_from(indices):
            pool_vols = []
            # Try patients until one yields usable volumes (handles unpaired data).
            order = list(indices)
            rng.shuffle(order)
            for idx in order:
                pool_vols = _pool_for(idx)
                if pool_vols:
                    break
            if not pool_vols:
                raise ValueError("No volumes available for AE training across selected patients.")
            return sample_slices(pool_vols, full_pool, batch_size, args.slice_size, rng)

        def sample_train():
            return _sample_from(train_idx)

        def sample_val():
            return _sample_from(val_idx)

    steps_per_epoch = int(args.steps_per_epoch)
    total_steps = int(args.epochs) * steps_per_epoch
    best_val = float("inf")
    start_step = 0
    if resume_path is not None:
        start_step, best_val = _resume(raw_model, optimizer, rng, resume_path, logger)

    for global_step in range(start_step, total_steps):
        epoch = global_step // max(1, steps_per_epoch)
        batch = sample_train()
        metrics = train_step(model, batch, optimizer)
        logger.info("step=%s loss=%.6f recon_l1=%.6f kl=%.6f", global_step, metrics["loss"], metrics["recon_l1"], metrics["kl"])
        metrics_writer.log("train", global_step, metrics, epoch)

        if args.val_every > 0 and global_step % args.val_every == 0:
            val_metrics = _validate(model, sample_val, args.val_batches)
            logger.info("VAL step=%s loss=%.6f recon_l1=%.6f", global_step, val_metrics["loss"], val_metrics["recon_l1"])
            metrics_writer.log("val", global_step, val_metrics, epoch)
            is_best = val_metrics["loss"] < best_val
            best_val = min(best_val, val_metrics["loss"])
            save_training_checkpoint(save_dir, _checkpoint_state(raw_model, optimizer, rng, config, global_step + 1, epoch, val_metrics["loss"], best_val), is_best)

    final_val = _validate(model, sample_val, args.val_batches)
    metrics_writer.log("val", total_steps, final_val, int(args.epochs))
    is_best = final_val["loss"] < best_val
    best_val = min(best_val, final_val["loss"])
    save_training_checkpoint(save_dir, _checkpoint_state(raw_model, optimizer, rng, config, total_steps, int(args.epochs), final_val["loss"], best_val), is_best)
    metrics_writer.close()
    logger.info("Training finished. best_val_loss=%.6f", best_val)


def _validate(model, sample_val, n_batches):
    accum = {}
    for _ in range(max(1, n_batches)):
        batch = sample_val()
        m = eval_step(model, batch)
        for k, v in m.items():
            accum[k] = accum.get(k, 0.0) + v
    return {k: v / max(1, n_batches) for k, v in accum.items()}


def _checkpoint_state(model, optimizer, rng, config, step, epoch, val_loss, best_val):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": capture_rng_state(rng),
        "config": config,
        "task": TASK,
        "step": step,
        "epoch": epoch,
        "val_loss": val_loss,
        "best_val": best_val,
        "latent_channels": int(config.get("latent_channels", 4)),
    }


def _resume(model, optimizer, rng, resume_path, logger):
    """Restore model/optimizer/RNG from a checkpoint; return (start_step, best_val)."""
    state = load_checkpoint(resume_path)
    model.load_state_dict(state["model"])
    if state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    restore_rng_state(state.get("rng"), rng)
    best_val = float(state.get("best_val", float("inf")))
    start_step = int(state.get("step", 0))  # "step" = number of steps already completed
    logger.info("Resumed from %s at step=%s best_val=%.6f", resume_path, start_step, best_val)
    return start_step, best_val


def _load_config(path, fallback):
    if not path or not os.path.exists(path):
        return _deep_copy_config(fallback)
    try:
        import yaml  # type: ignore
    except Exception:
        return _deep_copy_config(fallback)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    merged = _deep_copy_config(fallback)
    merged.update({k: v for k, v in data.items() if k != "model"})
    if "model" in data and isinstance(data["model"], dict):
        model_cfg = dict(fallback.get("model", {}))
        model_cfg.update(data["model"])
        merged["model"] = model_cfg
    return merged


def _deep_copy_config(config):
    out = dict(config)
    if isinstance(config.get("model"), dict):
        out["model"] = dict(config["model"])
    return out


if __name__ == "__main__":
    main()
