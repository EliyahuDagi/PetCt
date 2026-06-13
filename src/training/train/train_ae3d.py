"""Fine-tune a 3D AutoencoderKL inflated from the trained 2D AE.

Stage 2 of the pipeline (between ae2d and the diffusion stages). Loads the frozen
2D AE checkpoint, builds a 3D AutoencoderKL with the *same* architecture, centre-
inflates the 2D weights into it (so each 3D conv starts out behaving like its 2D
counterpart on every z-plane) and fine-tunes on pooled NAC+AC 3D crops. The
result encodes volumetric context and compresses the depth axis, giving the
downstream 3D diffusion a richer, smaller latent than a depth-uncompressed 2D
encoding. Emits JSONL metrics and best/last checkpoints (config embedded for
inference).
"""

import argparse
import os
import random

import numpy as np
import torch

from src.training.data import (
    PrefetchingPatientCache,
    ae_pool_volumes,
    load_patient_volumes,
    make_patient_split3,
    sample_volumes,
    sample_volumes_full,
    write_split_json,
)
from src.training.dataset_index import enumerate_patients
from src.training.utils.augment import build_aug_3d
from src.training.models.autoencoder3d import build_autoencoder_3d
from src.training.models.inflation import map_state_dict_2d_to_3d
from src.training.utils.checkpointing import (
    capture_rng_state,
    load_checkpoint,
    resolve_resume_path,
    restore_rng_state,
    save_training_checkpoint,
)
from src.training.utils.image_metrics import image_quality_metrics
from src.training.utils.logging import setup_logging
from src.training.utils.perceptual import build_perceptual_loss
from src.training.utils.metrics import MetricsWriter
from src.training.utils.perf import (
    autocast,
    configure_backends,
    maybe_compile,
    to_input_memory_format,
    to_model_memory_format,
)

TASK = "ae3d"
SPATIAL_DIMS = 3


def perceptual_term(perceptual, recon, target, weight):
    """Weighted perceptual loss on the 3D AE reconstruction vs the target.

    The AE reconstruction is already in-graph (no x0 decode needed like the
    diffusion stages). ``recon``/``target`` are 3D ``(N,1,D,H,W)`` so the 2D VGG
    backbone runs slice-wise across the three planes inside ``PerceptualLoss`` --
    the exact same slice-wise handling ft3d uses on its 3D decode. Returns
    ``(weighted_loss_tensor_or_None, float_value)``; ``None`` means disabled (no-op).
    """
    if perceptual is None or not weight or weight <= 0:
        return None, 0.0
    pterm = perceptual(recon, target) * float(weight)
    return pterm, float(pterm.detach().cpu())


def _ae_metrics(model, batch, perceptual=None, perceptual_weight=0.0):
    batch = to_input_memory_format(batch)
    with autocast():
        outputs = model(batch)
        if not (isinstance(outputs, (list, tuple)) and len(outputs) >= 3):
            raise ValueError("AutoencoderKL output is unexpected.")
        recon, z_mu, z_sigma = outputs[:3]
        recon_loss = torch.mean(torch.abs(recon - batch))
        kl_loss = torch.mean(0.5 * (z_mu.pow(2) + z_sigma.pow(2) - 1.0 - torch.log(z_sigma.pow(2) + 1.0e-6)))
        loss = recon_loss + (1.0e-6 * kl_loss)
    # Perceptual term is computed (and back-propagated) on the TRAIN path only;
    # callers pass perceptual=None for validation so checkpoint selection stays on
    # the existing recon-based val metric. None weight/loss is a true no-op.
    pterm, pval = perceptual_term(perceptual, recon, batch, perceptual_weight)
    if pterm is not None:
        loss = loss + pterm
    metrics = {
        "loss": float(loss.detach().cpu()),
        "recon_l1": float(recon_loss.detach().cpu()),
        "kl": float(kl_loss.detach().cpu()),
        "perceptual": pval,
    }
    return loss, metrics, recon


def train_step(model, batch, optimizer, perceptual=None, perceptual_weight=0.0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, metrics, _ = _ae_metrics(model, batch, perceptual, perceptual_weight)
    loss.backward()
    optimizer.step()
    return metrics


@torch.no_grad()
def eval_step(model, batch):
    model.eval()
    _, metrics, recon = _ae_metrics(model, batch)
    # Image-quality metrics on the 3D reconstruction (val rows carry these).
    metrics.update(image_quality_metrics(recon, batch))
    return metrics


def build_model(config):
    return build_autoencoder_3d(config)


def append_extra_levels(model_cfg, extra_levels):
    """Append ``extra_levels`` downsample level(s) to a 3D AE's ``model`` config.

    The 3D AE architecture is otherwise inherited wholesale from the 2D AE
    checkpoint, which forces it to share the 2D AE's depth (compression factor).
    Each appended level adds another /2 spatial downsample, so a 128^3 crop with
    one extra level lands a 16^3 latent (factor 8) while ft3d's latent grid -- and
    thus its cost -- is unchanged. The 2D AE's own depth is never modified.

    Pads every per-level list (``block_out_channels`` / ``num_channels``,
    ``attention_levels``, list-valued ``num_res_blocks``) so they stay the same
    length. The fresh 3D level has no 2D counterpart to inflate from; the
    ``strict=False`` inflation path leaves it at its random init (verified by
    test_sizing).

    Returns the same dict (mutated) for convenience. ``extra_levels <= 0`` is a
    no-op so the default run is byte-identical to before.
    """
    extra_levels = int(extra_levels or 0)
    if extra_levels <= 0:
        return model_cfg

    key = "block_out_channels" if "block_out_channels" in model_cfg else "num_channels"
    channels = list(model_cfg.get(key, [64, 128, 256]))
    if not channels:
        channels = [64, 128, 256]
    # New levels keep widening: double the last width per level (no decoder/encoder
    # asymmetry to worry about -- AutoencoderKL mirrors num_channels).
    for _ in range(extra_levels):
        channels.append(channels[-1] * 2)
    model_cfg[key] = channels
    n = len(channels)

    attn = model_cfg.get("attention_levels")
    if isinstance(attn, (list, tuple)):
        attn = list(attn)
        attn += [False] * (n - len(attn))
        model_cfg["attention_levels"] = attn[:n]

    nrb = model_cfg.get("num_res_blocks")
    if isinstance(nrb, (list, tuple)):
        nrb = list(nrb)
        pad = nrb[-1] if nrb else 2
        nrb += [pad] * (n - len(nrb))
        model_cfg["num_res_blocks"] = nrb[:n]

    return model_cfg


def inflate_and_load(model_3d, state_dict_2d):
    mapped, missing = map_state_dict_2d_to_3d(state_dict_2d, model_3d.state_dict())
    model_3d.load_state_dict(mapped, strict=False)
    return mapped, missing


def load_ae2d_config(ae2d_ckpt):
    if not ae2d_ckpt or not os.path.exists(ae2d_ckpt):
        raise FileNotFoundError(
            "2D AE checkpoint not found at %r. Train ae2d first (writes outputs/ae2d/best.pt)." % ae2d_ckpt
        )
    state = load_checkpoint(ae2d_ckpt)
    return state.get("config", {}), state.get("model", state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, nargs="+", help="One or more dataset roots / patient folders")
    parser.add_argument("--patient_index", type=int, default=0)
    parser.add_argument("--config", default=None)
    parser.add_argument("--ae2d_ckpt", default="outputs/ae2d/best.pt", help="2D AE checkpoint to inflate from")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=25)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--test_fraction", type=float, default=0.1,
                        help="By-patient TEST holdout fraction (multi-patient only). "
                             "Test patients are excluded from BOTH train and val.")
    parser.add_argument("--val_batches", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--crop_size", type=int, default=64, help="Cube size of 3D crops fed to the AE")
    parser.add_argument(
        "--extra_levels",
        type=int,
        default=0,
        help=(
            "Append N extra downsample level(s) to the inflated 3D AE (each level "
            "doubles the spatial compression factor). Opt-in: 0 keeps the 2D AE depth. "
            "B2 config uses crop_size 128 + extra_levels 1 -> latent stays 16^3 at factor 8."
        ),
    )
    parser.add_argument("--modality", choices=["pet", "ct"], default="pet")
    parser.add_argument(
        "--perceptual_weight",
        type=float,
        default=None,
        help="Override the perceptual-loss weight (0 disables it). Default comes from the config.",
    )
    parser.add_argument("--prefetch", type=int, default=0,
                        help="Background patient-prefetch depth (0 = synchronous, default). "
                             ">0 overlaps DICOM I/O with GPU compute via a worker thread.")
    parser.add_argument("--cache_size", type=int, default=None,
                        help="Override the resident patient-cache size (default auto).")
    parser.add_argument("--cache_dir", default=None,
                        help="SSD pre-cache dir (or $PETCT_CACHE_DIR). Empty/None = pure DICOM "
                             "(unchanged behavior). Hits skip DICOM; see src.training.precache.")
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
        "output_dir": "outputs/ae3d",
        "batch_size": 1,
        "learning_rate": 1.0e-4,
        # Perceptual loss on the 3D AE reconstruction (2D backbone run slice-wise
        # on the 3D recon, as ft3d does on its 3D decode). >0 enables it; 0/disabled
        # is a no-op. Backend pluggable ("vgg" now, "medical_sam" later).
        "perceptual_weight": 0.1,
        "perceptual_backend": "vgg",
    }

    config = _load_config(args.config, default_config)
    save_dir = args.save_dir or config["output_dir"]
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.perceptual_weight is not None:
        config["perceptual_weight"] = args.perceptual_weight

    resume_path = resolve_resume_path(save_dir, args.resume)

    logger = setup_logging(save_dir)
    metrics_writer = MetricsWriter(save_dir, TASK, append=resume_path is not None)

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    configure_backends(device, logger)

    # Build the 3D AE with the SAME architecture as the 2D AE so weights inflate
    # cleanly; the 2D AE's config is the source of truth and is re-embedded here.
    ae2d_config, ae2d_state = load_ae2d_config(args.ae2d_ckpt)
    config["model"] = dict(ae2d_config.get("model", {}))
    config["latent_channels"] = int(ae2d_config.get("latent_channels", 4))
    # Opt-in: deepen the 3D AE beyond the 2D AE's depth so it compresses harder
    # (e.g. 128^3 -> 16^3). The 2D AE checkpoint is untouched; the extra level
    # warm-starts at fresh init via the strict=False inflation below.
    if args.extra_levels:
        append_extra_levels(config["model"], args.extra_levels)
        logger.info("Appended %d extra 3D AE downsample level(s): %s",
                    args.extra_levels, config["model"].get("block_out_channels", config["model"].get("num_channels")))

    # raw_model owns the weights (inflation / state_dict / resume); model is the
    # (optionally) compiled forward handle. They share parameters.
    raw_model = build_model(config).to(device)
    raw_model = to_model_memory_format(raw_model, SPATIAL_DIMS)
    optimizer = torch.optim.Adam(raw_model.parameters(), lr=float(config.get("learning_rate", 1.0e-4)))

    # Inflation seeds the 3D weights; a --resume checkpoint takes precedence and
    # overwrites them, so only inflate on a fresh run. Inflate before compiling.
    if resume_path is None:
        _, missing = inflate_and_load(raw_model, ae2d_state)
        logger.info("Inflated 3D AE from %s; %s params left at fresh init.", args.ae2d_ckpt, len(missing))

    model = maybe_compile(raw_model, logger)

    # Pluggable perceptual loss (frozen VGG by default), applied to the 3D recon vs
    # target on the TRAIN path only (slice-wise 2D backbone, as ft3d uses). build_*
    # returns None when the backend/weights are unavailable, so training proceeds
    # without the term (graceful degradation).
    perceptual_weight = float(config.get("perceptual_weight", 0.0) or 0.0)
    perceptual = None
    if perceptual_weight > 0:
        perceptual = build_perceptual_loss(
            config.get("perceptual_backend", "vgg"),
            device=device,
            weights_path=config.get("perceptual_weights_path"),
        )
        if perceptual is None:
            logger.warning("Perceptual loss requested but backend unavailable; continuing without it.")
            perceptual_weight = 0.0
        else:
            logger.info("Perceptual loss enabled: backend=%s weight=%.4g",
                        config.get("perceptual_backend", "vgg"), perceptual_weight)

    rng = np.random.RandomState(seed)
    batch_size = int(config.get("batch_size", 1))

    # Geometric-only 3D train augmentation (no-op identity unless config opts in),
    # layered on top of the existing crop/flip/rot90. Val always passes geo_aug=None.
    train_aug = build_aug_3d(config.get("augment"))

    # Enumerate patients across roots; <=1 patient falls back to the legacy
    # single-patient depth-band split (keeps existing behavior/tests unchanged).
    patients = enumerate_patients(args.data_dir, missing_ok=True)
    multi_patient = len(patients) > 1
    cache = None  # set in the multi-patient branch; closed after training

    if not multi_patient:
        vols = load_patient_volumes(args.data_dir[0], args.patient_index, device=device)
        pool_vols = ae_pool_volumes(vols, args.modality)
        if not pool_vols and args.modality == "pet":
            logger.info("No PET volumes found; falling back to CT for AE training.")
        if not pool_vols:
            raise ValueError("No volumes available for 3D AE training.")
        logger.info("Training on 1 patient (train 1 / val 1) across %d roots.", len(args.data_dir))

        def sample_train():
            return sample_volumes(pool_vols, batch_size, args.crop_size, "train", args.val_fraction, rng, geo_aug=train_aug)

        def sample_val():
            return sample_volumes(pool_vols, batch_size, args.crop_size, "val", args.val_fraction, rng, geo_aug=None)
    else:
        train_idx, val_idx, test_idx = make_patient_split3(
            len(patients), args.val_fraction, args.test_fraction, seed)
        logger.info(
            "Training on %d patients (train %d / val %d / test %d) across %d roots. "
            "Test patients are held out from training entirely.",
            len(patients), len(train_idx), len(val_idx), len(test_idx), len(args.data_dir),
        )
        split_path = write_split_json(
            save_dir, TASK, patients, train_idx, val_idx, test_idx,
            args.val_fraction, args.test_fraction, seed)
        if split_path:
            logger.info("Wrote by-patient split to %s.", split_path)
        # Opt-in async prefetch: prefetch>0 overlaps the per-patient DICOM read with
        # GPU compute. prefetch==0 keeps the synchronous LRU path byte-identical.
        cache_kwargs = {} if args.cache_size is None else {"max_cached": args.cache_size}
        cache = PrefetchingPatientCache(
            patients, device=device, prefetch=args.prefetch,
            train_indices=train_idx, rng=rng, cache_dir=args.cache_dir, **cache_kwargs,
        ).start()

        def _sample_sync(indices, augment, geo_aug=None):
            # Original synchronous path (byte-identical for prefetch==0).
            pool_vols = []
            order = list(indices)
            rng.shuffle(order)
            for idx in order:
                pool_vols = ae_pool_volumes(cache.get(idx), args.modality)
                if pool_vols:
                    break
            if not pool_vols:
                raise ValueError("No volumes available for 3D AE training across selected patients.")
            return sample_volumes_full(pool_vols, batch_size, args.crop_size, rng, augment=augment, geo_aug=geo_aug)

        # Train augments (random crop/flip/rotation + optional MONAI geo); val stays clean.
        def sample_train():
            if args.prefetch <= 0:
                return _sample_sync(train_idx, True, geo_aug=train_aug)
            pool_vols = []
            for _ in range(max(1, len(train_idx))):
                _, vols = cache.next_train()
                pool_vols = ae_pool_volumes(vols, args.modality)
                if pool_vols:
                    break
            if not pool_vols:
                raise ValueError("No volumes available for 3D AE training across selected patients.")
            return sample_volumes_full(pool_vols, batch_size, args.crop_size, rng, augment=True, geo_aug=train_aug)

        def sample_val():
            return _sample_sync(val_idx, False, geo_aug=None)

    steps_per_epoch = int(args.steps_per_epoch)
    total_steps = int(args.epochs) * steps_per_epoch
    best_val = float("inf")
    start_step = 0
    if resume_path is not None:
        start_step, best_val = _resume(raw_model, optimizer, rng, resume_path, logger)

    for global_step in range(start_step, total_steps):
        epoch = global_step // max(1, steps_per_epoch)
        batch = sample_train()
        metrics = train_step(model, batch, optimizer, perceptual, perceptual_weight)
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
    if cache is not None:
        cache.close()
    logger.info("Training finished. best_val_loss=%.6f", best_val)


@torch.no_grad()
def _validate(model, sample_val, n_batches):
    accum = {}
    n = max(1, n_batches)
    for _ in range(n):
        batch = sample_val()
        m = eval_step(model, batch)
        for k, v in m.items():
            accum[k] = accum.get(k, 0.0) + v
    return {k: v / n for k, v in accum.items()}


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
    start_step = int(state.get("step", 0))
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
