"""Train the 2D latent diffusion UNet for NAC->AC translation.

A frozen 2D AutoencoderKL encodes paired NAC and AC slices to latents. The UNet
is conditioned on the NAC latent by channel-concatenation (in_channels =
2*latent_channels) and trained to predict the noise added to the AC latent under
the shared DiffusionSchedule. Emits JSONL metrics and best/last checkpoints.
"""

import argparse
import os
import random

import numpy as np
import torch

from src.training.data import (
    PrefetchingPatientCache,
    filter_paired_patients_cached,
    load_patient_volumes,
    make_depth_split,
    make_patient_split3,
    sample_pairs,
    write_split_json,
)
from src.training.dataset_index import enumerate_patients
from src.training.utils.augment import build_aug_2d
from src.training.models.autoencoder2d import ae_decode, ae_encode, build_autoencoder_2d
from src.training.models.diffusion2d import build_diffusion_2d
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
from src.training.utils.sampling import DiffusionSchedule
from src.training.utils.schedule import EMA, build_lr_scheduler

TASK = "diff2d"
SPATIAL_DIMS = 2


def build_model(config):
    return build_diffusion_2d(config)


def load_frozen_ae(ae_ckpt, device):
    if not ae_ckpt or not os.path.exists(ae_ckpt):
        raise FileNotFoundError(
            "AE checkpoint not found at %r. Train ae2d first (it writes outputs/ae2d/best.pt)." % ae_ckpt
        )
    state = load_checkpoint(ae_ckpt)
    ae_config = state.get("config", {})
    ae = build_autoencoder_2d(ae_config).to(device)
    ae.load_state_dict(state.get("model", state))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae, ae_config


def compute_latent_scale(ae, sample_train_pair, n_batches=12, logger=None):
    """Estimate the Stable-Diffusion-style latent-normalization scale.

    Encodes a sample of AC latents (RAW, scale=1.0) and returns 1/(std + eps) so
    the scaled latents have ~unit std -- matching the N(0,1) noise the DDIM sampler
    starts from. Computed once at the start of diffusion training and stored in the
    config (embedded into the checkpoint) so inference/eval reuse the same value.
    """
    samples = []
    for _ in range(max(1, n_batches)):
        nac_img, ac_img = sample_train_pair()
        samples.append(ae_encode(ae, ac_img).flatten())
    std = float(torch.cat(samples).std().cpu())
    scale = 1.0 / (std + 1e-8)
    if logger is not None:
        logger.info("Computed latent_scale=%.6f (AC latent std=%.6f over %d batches)", scale, std, max(1, n_batches))
    return scale


def apply_cond_dropout(nac_lat, prob):
    """Classifier-free guidance: zero the NAC conditioning latent per-sample with
    probability ``prob`` (the "null" condition), so the UNet also learns the
    unconditional score. ``prob<=0`` is a no-op (plain conditional training).
    Returns the (possibly masked) conditioning latent — same shape as input.
    """
    if not prob or prob <= 0:
        return nac_lat
    keep = (torch.rand(nac_lat.shape[0], device=nac_lat.device) >= float(prob)).to(nac_lat.dtype)
    keep = keep.view(nac_lat.shape[0], *([1] * (nac_lat.ndim - 1)))
    return nac_lat * keep


def diffusion_loss(model, schedule, ac_lat, nac_lat, snr_gamma=None):
    """Predict noise added to the AC latent, conditioned on the NAC latent.

    ``snr_gamma`` (e.g. 5.0) enables Min-SNR-gamma loss weighting; ``None`` keeps
    the plain unweighted MSE.
    """
    timesteps = torch.randint(0, schedule.num_train_timesteps, (ac_lat.shape[0],), device=ac_lat.device)
    noise = torch.randn_like(ac_lat)
    x_t = schedule.q_sample(ac_lat, timesteps, noise)
    model_in = to_input_memory_format(torch.cat([x_t, nac_lat], dim=1))
    with autocast():
        pred = model(model_in, timesteps)
    if snr_gamma is None:
        loss = torch.mean((pred - noise) ** 2)
    else:
        per_sample = torch.mean((pred - noise) ** 2, dim=list(range(1, pred.ndim)))
        weights = schedule.min_snr_weights(timesteps, gamma=float(snr_gamma))
        loss = torch.mean(weights * per_sample)
    return loss, x_t, timesteps, pred, noise


def x0_decode_in_graph(ae, schedule, x_t, timesteps, pred, latent_scale=1.0):
    """Decode the one-step x0 estimate to image space, keeping the graph intact.

    Same formula ``_validate`` uses, but ``ae_decode`` is wrapped in ``no_grad``;
    for a perceptual *training* term gradients must reach the UNet, so call the
    AE decoder directly (the AE is frozen, so no AE params are updated -- the
    gradient just passes through it back to ``pred``). ``x0_pred`` lives in scaled
    latent space, so divide by ``latent_scale`` before decoding (no-op at 1.0).
    """
    acp = schedule.alphas_cumprod[timesteps]
    sqrt_acp = schedule._broadcast(torch.sqrt(acp), x_t)
    sqrt_one_minus = schedule._broadcast(torch.sqrt(1.0 - acp), x_t)
    x0_pred = (x_t - sqrt_one_minus * pred) / sqrt_acp
    if latent_scale and latent_scale > 0 and latent_scale != 1.0:
        x0_pred = x0_pred / latent_scale
    return ae.decode(x0_pred)


def perceptual_term(perceptual, ae, schedule, x_t, timesteps, pred, ac_img, weight, latent_scale=1.0):
    """Weighted perceptual loss on the in-graph decoded x0 vs the AC image.

    Returns ``(weighted_loss_tensor_or_None, float_value)``. ``None`` weight/loss
    means "disabled" -- the caller skips adding it (a true no-op).
    """
    if perceptual is None or not weight or weight <= 0:
        return None, 0.0
    recon = x0_decode_in_graph(ae, schedule, x_t, timesteps, pred, latent_scale)
    pterm = perceptual(recon, ac_img) * float(weight)
    return pterm, float(pterm.detach().cpu())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, nargs="+", help="One or more dataset roots / patient folders")
    parser.add_argument("--patient_index", type=int, default=0)
    parser.add_argument("--config", default=None)
    parser.add_argument("--ae_ckpt", default="outputs/ae2d/best.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=25)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--test_fraction", type=float, default=0.1,
                        help="By-patient TEST holdout fraction (multi-patient only). "
                             "Test patients are excluded from BOTH train and val.")
    parser.add_argument("--val_batches", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--latent_size", type=int, default=128, help="Image size of slices fed to the AE encoder")
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
    parser.add_argument("--rescan_pairs", action="store_true",
                        help="Force a fresh NAC/AC pairing scan, ignoring the cached manifest.")
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
        "output_dir": "outputs/diff2d",
        "batch_size": 4,
        "learning_rate": 1.0e-4,
        "latent_channels": 4,
        "noise_schedule": "cosine",
        "snr_gamma": 5.0,
        "rescale_zero_terminal_snr": False,
        # Classifier-free guidance: per-sample probability of replacing the NAC
        # conditioning latent with zeros (the null condition) during training, so
        # the UNet learns both the conditional and unconditional score. 0 = plain
        # conditional training (old behavior). Inference amplifies with guidance.
        "cond_dropout_prob": 0.1,
        "ema_decay": 0.9999,
        "lr_min_ratio": 0.1,
        # Perceptual loss on the in-graph decoded x0 estimate. >0 enables it;
        # 0/disabled is a no-op. Backend is pluggable ("vgg" now, "medical_sam"
        # later) -- see src/training/utils/perceptual.py.
        "perceptual_weight": 0.1,
        "perceptual_backend": "vgg",
        "model": {
            "num_channels": [16, 32, 64],
            "attention_levels": [False, True, True],
            "num_res_blocks": 1,
        },
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

    ae, ae_config = load_frozen_ae(args.ae_ckpt, device)
    latent_channels = int(ae_config.get("latent_channels", config.get("latent_channels", 4)))
    config["latent_channels"] = latent_channels
    # Conditioning by concatenation: input is [noisy_AC | NAC] latents.
    model_cfg = dict(config.get("model", {}))
    model_cfg["in_channels"] = 2 * latent_channels
    model_cfg["out_channels"] = latent_channels
    config["model"] = model_cfg

    # raw_model owns the weights (EMA / state_dict / resume); model is the
    # (optionally) compiled forward handle. They share parameters.
    raw_model = build_model(config).to(device)
    raw_model = to_model_memory_format(raw_model, SPATIAL_DIMS)
    optimizer = torch.optim.Adam(raw_model.parameters(), lr=float(config.get("learning_rate", 1.0e-4)))
    model = maybe_compile(raw_model, logger)
    schedule = DiffusionSchedule(
        schedule=config.get("noise_schedule", "cosine"),
        rescale_zero_terminal_snr=bool(config.get("rescale_zero_terminal_snr", False)),
        device=device,
    )
    snr_gamma = config.get("snr_gamma", 5.0)
    cond_dropout_prob = float(config.get("cond_dropout_prob", 0.0) or 0.0)
    if cond_dropout_prob > 0:
        logger.info("Classifier-free guidance: cond_dropout_prob=%.3g", cond_dropout_prob)

    # Pluggable perceptual loss (frozen VGG by default). build_* returns None when
    # the backend/weights are unavailable, so training proceeds without the term.
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
    batch_size = int(config.get("batch_size", 4))

    # Geometric-only train augmentation (no-op identity unless config opts in).
    # The SAME geometry is applied to NAC and AC (paired). Val passes augment=None.
    train_aug = build_aug_2d(config.get("augment"))

    # Diffusion requires paired NAC+AC. Enumerate across roots, keep only paired
    # patients (logging skips). <=1 paired patient -> legacy within-patient split.
    patients = enumerate_patients(args.data_dir, missing_ok=True)
    if len(patients) > 1:
        patients = filter_paired_patients_cached(patients, rescan=args.rescan_pairs, log=logger.info, cache_dir=args.cache_dir)
        if not patients:
            raise ValueError("No paired NAC+AC patients found for diff2d across the given roots.")
    multi_patient = len(patients) > 1
    cache = None  # set in the multi-patient branch; closed after training

    if not multi_patient:
        vols = load_patient_volumes(args.data_dir[0], args.patient_index, device=device)
        if vols.get("pet_nac") is None or vols.get("pet_ac") is None:
            raise ValueError("Both NAC and AC PET volumes are required for diff2d training.")
        train_pool, val_pool = make_depth_split(seed, args.val_fraction)
        logger.info("Training on 1 patient (train 1 / val 1) across %d roots.", len(args.data_dir))

        def sample_train_pair():
            return sample_pairs(vols["pet_nac"], vols["pet_ac"], train_pool, batch_size, args.latent_size, rng, augment=train_aug)

        def sample_val_pair():
            return sample_pairs(vols["pet_nac"], vols["pet_ac"], val_pool, batch_size, args.latent_size, rng, augment=None)
    else:
        train_idx, val_idx, test_idx = make_patient_split3(
            len(patients), args.val_fraction, args.test_fraction, seed)
        full_pool = np.linspace(0.0, 1.0, num=128, endpoint=False)
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

        def _sample_sync(indices, augment=None):
            idx = int(indices[rng.randint(0, len(indices))])
            vols = cache.get(idx)
            return sample_pairs(vols["pet_nac"], vols["pet_ac"], full_pool, batch_size, args.latent_size, rng, augment=augment)

        def sample_train_pair():
            if args.prefetch <= 0:
                return _sample_sync(train_idx, augment=train_aug)
            _, vols = cache.next_train()
            return sample_pairs(vols["pet_nac"], vols["pet_ac"], full_pool, batch_size, args.latent_size, rng, augment=train_aug)

        def sample_val_pair():
            return _sample_sync(val_idx, augment=None)

    # Latent normalization: scale RAW AE latents to ~unit std so the DDIM sampler's
    # N(0,1) start matches the latent distribution. A non-null config value pins the
    # scale (lets the user fix it / resume deterministically); otherwise compute it
    # once from a sample of AC latents. Stored in config -> embedded in the checkpoint
    # so inference/eval reuse the exact same scale.
    cfg_scale = config.get("latent_scale")
    if cfg_scale is not None and float(cfg_scale) > 0:
        latent_scale = float(cfg_scale)
        logger.info("Using configured latent_scale=%.6f", latent_scale)
    else:
        latent_scale = compute_latent_scale(ae, sample_train_pair, logger=logger)
    config["latent_scale"] = float(latent_scale)

    steps_per_epoch = int(args.steps_per_epoch)
    total_steps = int(args.epochs) * steps_per_epoch

    ema_decay = float(config.get("ema_decay", 0.0))
    # EMA tracks the raw (uncompiled) weights so its shadow keys stay prefix-free.
    ema = EMA(raw_model, decay=ema_decay) if ema_decay > 0 else None
    config.setdefault("lr_total_steps", total_steps)
    config.setdefault("lr_warmup_steps", min(500, max(1, total_steps // 10)))
    lr_scheduler = build_lr_scheduler(optimizer, config)

    best_val = float("inf")
    start_step = 0
    if resume_path is not None:
        start_step, best_val = _resume(raw_model, optimizer, rng, resume_path, logger, ema)
        # Fast-forward the LR schedule so the resumed LR matches an uninterrupted run.
        if lr_scheduler is not None:
            for _ in range(start_step):
                lr_scheduler.step()

    for global_step in range(start_step, total_steps):
        epoch = global_step // max(1, steps_per_epoch)
        nac_img, ac_img = sample_train_pair()
        # Both the target AC latent and the NAC conditioning latent live in the SAME
        # scaled latent space (NAC is channel-concatenated to the noisy AC).
        ac_lat = ae_encode(ae, ac_img, scale=latent_scale)
        nac_lat = ae_encode(ae, nac_img, scale=latent_scale)
        # CFG: drop the conditioning on a fraction of samples so the UNet learns the
        # unconditional score too (the null condition is zeros, matching inference).
        nac_lat = apply_cond_dropout(nac_lat, cond_dropout_prob)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        mse_loss, x_t, timesteps, pred, _ = diffusion_loss(model, schedule, ac_lat, nac_lat, snr_gamma)
        pterm, pval = perceptual_term(perceptual, ae, schedule, x_t, timesteps, pred, ac_img, perceptual_weight, latent_scale)
        loss = mse_loss if pterm is None else mse_loss + pterm
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        if ema is not None:
            ema.update(raw_model)
        metrics = {"loss": float(loss.detach().cpu()), "mse": float(mse_loss.detach().cpu()), "perceptual": pval}
        logger.info("step=%s loss=%.6f", global_step, metrics["loss"])
        metrics_writer.log("train", global_step, metrics, epoch)

        if args.val_every > 0 and global_step % args.val_every == 0:
            val_metrics = _validate(model, raw_model, ema, ae, schedule, sample_val_pair, args.val_batches, snr_gamma, latent_scale)
            logger.info("VAL step=%s loss=%.6f l1=%.6f", global_step, val_metrics["loss"], val_metrics["l1"])
            metrics_writer.log("val", global_step, val_metrics, epoch)
            is_best = val_metrics["loss"] < best_val
            best_val = min(best_val, val_metrics["loss"])
            save_training_checkpoint(save_dir, _checkpoint_state(raw_model, optimizer, rng, config, global_step + 1, epoch, val_metrics["loss"], best_val, ema), is_best)

    final_val = _validate(model, raw_model, ema, ae, schedule, sample_val_pair, args.val_batches, snr_gamma, latent_scale)
    metrics_writer.log("val", total_steps, final_val, int(args.epochs))
    is_best = final_val["loss"] < best_val
    best_val = min(best_val, final_val["loss"])
    save_training_checkpoint(save_dir, _checkpoint_state(raw_model, optimizer, rng, config, total_steps, int(args.epochs), final_val["loss"], best_val, ema), is_best)
    metrics_writer.close()
    if cache is not None:
        cache.close()
    logger.info("Training finished. best_val_loss=%.6f", best_val)


@torch.no_grad()
def _validate(model, raw_model, ema, ae, schedule, sample_val_pair, n_batches, snr_gamma=None, latent_scale=1.0):
    # Evaluate under the EMA weights when available -- consistently higher quality.
    # The EMA swap targets raw_model (shared params); forward still runs via model.
    ctx = ema.average_parameters(raw_model) if ema is not None else _null_context(raw_model)
    with ctx:
        model.eval()
        accum = {"loss": 0.0, "l1": 0.0}
        n = max(1, n_batches)
        for _ in range(n):
            nac_img, ac_img = sample_val_pair()
            # Same scaled latent space as training (NAC + AC scaled identically).
            ac_lat = ae_encode(ae, ac_img, scale=latent_scale)
            nac_lat = ae_encode(ae, nac_img, scale=latent_scale)
            loss, x_t, timesteps, pred, noise = diffusion_loss(model, schedule, ac_lat, nac_lat, snr_gamma)
            # Cheap recon proxy: one-step x0 estimate decoded back to image space.
            # x0_pred is in scaled latent space; ae_decode divides by scale.
            acp = schedule.alphas_cumprod[timesteps]
            sqrt_acp = schedule._broadcast(torch.sqrt(acp), x_t)
            sqrt_one_minus = schedule._broadcast(torch.sqrt(1.0 - acp), x_t)
            x0_pred = (x_t - sqrt_one_minus * pred) / sqrt_acp
            recon = ae_decode(ae, x0_pred, scale=latent_scale)
            accum["loss"] += float(loss.cpu())
            accum["l1"] += float(torch.mean(torch.abs(recon - ac_img)).cpu())
            # Image-quality metrics of the decoded prediction vs AC reference.
            for k, v in image_quality_metrics(recon, ac_img).items():
                accum[k] = accum.get(k, 0.0) + v
    return {k: v / n for k, v in accum.items()}


class _null_context:
    """No-op stand-in for EMA.average_parameters when EMA is disabled."""

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        return self.model

    def __exit__(self, *exc):
        return False


def _checkpoint_state(model, optimizer, rng, config, step, epoch, val_loss, best_val, ema=None):
    state = {
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
    if ema is not None:
        state["ema"] = ema.state_dict()
    return state


def _resume(model, optimizer, rng, resume_path, logger, ema=None):
    """Restore model/optimizer/RNG (and EMA) from a checkpoint; return (start_step, best_val)."""
    state = load_checkpoint(resume_path)
    model.load_state_dict(state["model"])
    if state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if ema is not None and state.get("ema") is not None:
        ema.load_state_dict(state["ema"])
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
