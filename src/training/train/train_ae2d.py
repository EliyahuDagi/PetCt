"""Train the 2D AutoencoderKL (encode/decode).

Trains on pooled PET slices (NAC + AC) by default so NAC and AC share one latent
space for the downstream NAC->AC translation. Emits structured JSONL metrics and
saves best/last checkpoints (with config embedded) for the Train Viewer.
"""

import argparse
import os
import random
import threading

import numpy as np
import torch

from src.training.data import (
    PrefetchingPatientCache,
    ae_pool_volumes,
    load_patient_volumes,
    make_depth_split,
    make_patient_split3,
    sample_slices,
    write_split_json,
)
from src.training.dataset_index import enumerate_patients
from src.training.utils.augment import build_aug_2d
from src.training.models.autoencoder2d import build_autoencoder_2d
from src.training.utils.checkpointing import (
    init_weights_from,
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
from src.training.utils.prefetch_batches import BatchPrefetcher
from src.training.utils.perf import (
    autocast,
    configure_backends,
    maybe_compile,
    to_input_memory_format,
    to_model_memory_format,
)

TASK = "ae2d"
SPATIAL_DIMS = 2


def perceptual_term(perceptual, recon, target, weight):
    """Weighted perceptual loss on the AE reconstruction vs the target.

    The AE reconstruction is already in-graph (no x0 decode needed like the
    diffusion stages), so the term applies directly to ``recon`` vs ``target``.
    Returns ``(weighted_loss_tensor_or_None, float_value)``; ``None`` weight/loss
    means "disabled" -- the caller skips adding it (a true no-op).
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
    parser.add_argument("--test_fraction", type=float, default=0.1,
                        help="By-patient TEST holdout fraction (multi-patient only). "
                             "Test patients are excluded from BOTH train and val.")
    parser.add_argument("--val_batches", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--slice_size", type=int, default=128)
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
    parser.add_argument("--loader_workers", type=int, default=0,
                        help="Background BATCH-prep worker threads (0 = synchronous, default; "
                             "byte-identical to today). >0 overlaps the CPU-bound slice "
                             "extraction + per-sample augment with GPU compute. Multi-patient "
                             "only; makes the data-sampling RNG order nondeterministic (perf path).")
    parser.add_argument("--patient_reuse", type=int, default=1,
                        help="Threaded-loader only (--loader_workers>0): draw this many "
                             "independently-sampled batches from each loaded patient before "
                             "pulling the next one. Amortizes the multi-MB .npz read + zlib "
                             "decompress over K steps so step throughput decouples from disk. "
                             "1 (default) = current behavior, byte-identical.")
    parser.add_argument("--cache_size", type=int, default=None,
                        help="Override the resident patient-cache size (default auto).")
    parser.add_argument("--cache_dir", default=None,
                        help="SSD pre-cache dir (or $PETCT_CACHE_DIR). Empty/None = pure DICOM "
                             "(unchanged behavior). Hits skip DICOM; see src.training.precache.")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--init_from", default=None,
        help="Warm start: load MODEL WEIGHTS ONLY from an existing ae2d checkpoint "
             "(strict=False, so it survives an architecture change such as "
             "latent_channels 8->16). Unlike --resume it keeps a fresh optimizer/LR/step.",
    )
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
        # Perceptual loss on the AE reconstruction (frozen VGG by default). >0
        # enables it; 0/disabled is a no-op. Backend is pluggable ("vgg" now,
        # "medical_sam" later) -- see src/training/utils/perceptual.py.
        "perceptual_weight": 0.1,
        "perceptual_backend": "vgg",
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

    # raw_model owns the weights (state_dict / resume); model is the (optionally)
    # compiled handle used only for the forward pass. They share parameters.
    raw_model = build_model(config).to(device)
    # Warm start (weights only) BEFORE compile, so the compiled graph captures the loaded
    # weights. Skipped when --resume is active: a resume fully restores this run's own
    # state and must take precedence over an external initializer.
    if resume_path is None and args.init_from:
        if not os.path.exists(args.init_from):
            raise FileNotFoundError(f"--init_from checkpoint not found: {args.init_from}")
        init_weights_from(raw_model, args.init_from, logger, label=TASK)
    raw_model = to_model_memory_format(raw_model, SPATIAL_DIMS)
    optimizer = torch.optim.Adam(raw_model.parameters(), lr=float(config.get("learning_rate", 1.0e-4)))
    model = maybe_compile(raw_model, logger)

    # Pluggable perceptual loss (frozen VGG by default), applied to the recon vs
    # target on the TRAIN path only. build_* returns None when the backend/weights
    # are unavailable, so training proceeds without the term (graceful degradation).
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
    # Train uses it; validation always passes augment=None for a stable metric.
    train_aug = build_aug_2d(config.get("augment"))

    # Enumerate patients across all roots (missing_ok lets monkeypatched / GUI
    # callers pass non-existent paths). With <=1 enumerated patient we fall back
    # to the legacy single-patient depth split so existing behavior is unchanged.
    patients = enumerate_patients(args.data_dir, missing_ok=True)
    multi_patient = len(patients) > 1
    cache = None  # set in the multi-patient branch; closed after training
    prefetcher = None  # set when --loader_workers > 0 (multi-patient); closed at the end

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
            return sample_slices(pool_vols, train_pool, batch_size, args.slice_size, rng, augment=train_aug)

        def sample_val():
            return sample_slices(pool_vols, val_pool, batch_size, args.slice_size, rng, augment=None)
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
        # GPU compute on a worker thread. prefetch==0 keeps the synchronous LRU path,
        # so the default run is byte-identical to before.
        cache_kwargs = {} if args.cache_size is None else {"max_cached": args.cache_size}
        # Cache-device selection: with --loader_workers > 0 we keep volumes
        # CPU-resident so the worker threads slice/interpolate/augment with torch CPU
        # ops + MONAI (which release the GIL -> true multicore); the main loop then
        # does one tiny H2D copy of the assembled (~1 MB) batch. With workers == 0 the
        # cache stays on `device` exactly as today, so the default path is unchanged.
        cache_device = "cpu" if args.loader_workers > 0 else device
        cache = PrefetchingPatientCache(
            patients, device=cache_device, prefetch=args.prefetch,
            train_indices=train_idx, rng=rng, cache_dir=args.cache_dir, **cache_kwargs,
        ).start()

        def _pool_for(idx):
            return ae_pool_volumes(cache.get(idx), args.modality)

        def _sample_sync(indices, augment=None):
            # Original synchronous path (preserved byte-for-byte for prefetch==0):
            # shuffle a copy of the index list each call and take the first usable.
            pool_vols = []
            order = list(indices)
            rng.shuffle(order)
            for idx in order:
                pool_vols = _pool_for(idx)
                if pool_vols:
                    break
            if not pool_vols:
                raise ValueError("No volumes available for AE training across selected patients.")
            return sample_slices(pool_vols, full_pool, batch_size, args.slice_size, rng, augment=augment)

        def sample_train():
            if args.prefetch <= 0:
                return _sample_sync(train_idx, augment=train_aug)
            # Prefetch path: walk the shuffled-epoch stream; skip empty patients.
            pool_vols = []
            for _ in range(max(1, len(train_idx))):
                _, vols = cache.next_train()
                pool_vols = ae_pool_volumes(vols, args.modality)
                if pool_vols:
                    break
            if not pool_vols:
                raise ValueError("No volumes available for AE training across selected patients.")
            return sample_slices(pool_vols, full_pool, batch_size, args.slice_size, rng, augment=train_aug)

        def sample_val():
            return _sample_sync(val_idx, augment=None)

        # Opt-in threaded batch prefetcher: when --loader_workers > 0 we move the
        # CPU-bound batch prep (slice extraction + per-sample augment + interpolate)
        # onto background worker threads, overlapping it with GPU compute. The
        # synchronous `batch = sample_train()` path is untouched when workers == 0,
        # so the default run stays byte-identical to today.
        #
        # NOTE: enabling workers makes the data-sampling RNG draw order
        # nondeterministic relative to the single-thread path. Batches are still
        # independently sampled and augmented -- we only overlap/parallelize, not
        # batch the augment. This is an accepted trade-off for a perf-only path.
        if args.loader_workers > 0:
            # The cache is CPU-resident in this mode, so workers do their slice +
            # interpolate + augment entirely on CPU (true multicore). The only shared,
            # non-reentrant state across workers is the cache cursor advance inside
            # cache.next_train(); serialize just that with a lock. The volume tensors
            # are read-only during slicing, so the concurrent CPU work outside the
            # lock is safe. No worker touches CUDA -- the lone H2D copy is the main
            # loop's tiny per-batch move.
            pull_lock = threading.Lock()

            def _pull_train_pool():
                # Mirror the prefetch-path patient walk, but lock-guard the cursor
                # advance so concurrent workers don't corrupt cache.next_train().
                for _ in range(max(1, len(train_idx))):
                    with pull_lock:
                        _, vols = cache.next_train()
                    pool_vols = ae_pool_volumes(vols, args.modality)
                    if pool_vols:
                        return pool_vols
                raise ValueError("No volumes available for AE training across selected patients.")

            patient_reuse = max(1, int(args.patient_reuse))

            def _make_batch(worker_rng, worker_aug, worker_state):
                # Same work as the prefetch sample_train() path, but with the worker's
                # OWN rng + augment (no shared, non-thread-safe RandomState / MONAI
                # Rand*d state) and the lock-guarded patient pull. Returns a CPU batch;
                # the main loop moves it to `device`.
                #
                # Patient reuse: a loaded patient (multi-MB .npz read + zlib decompress)
                # is held in this worker's private state and reused for `patient_reuse`
                # batches before the next lock-guarded cache.next_train(). Each of the K
                # batches is an independent fresh draw of (axis, position) + its own
                # augment, so augment/sampling semantics are unchanged; we only amortize
                # the load. reuse==1 pulls a new patient every batch (prior behavior).
                if worker_state.get("remaining", 0) <= 0:
                    worker_state["pool"] = _pull_train_pool()
                    worker_state["remaining"] = patient_reuse
                worker_state["remaining"] -= 1
                pool_vols = worker_state["pool"]
                return sample_slices(
                    pool_vols, full_pool, batch_size, args.slice_size, worker_rng, augment=worker_aug)

            # Each worker gets a deterministically-seeded RandomState (base seed +
            # worker index) and its own built augment transform.
            prefetcher = BatchPrefetcher(
                make_batch=_make_batch,
                num_workers=int(args.loader_workers),
                make_rng=lambda w: np.random.RandomState(seed + 1 + int(w)),
                make_aug=lambda: build_aug_2d(config.get("augment")),
            )

    # When the prefetcher is active, pull batches from its queue; otherwise keep the
    # synchronous per-step sampler. Validation always stays synchronous.
    if prefetcher is not None:
        get_train_batch = prefetcher.get
    else:
        get_train_batch = sample_train

    # In the threaded path the cache is CPU-resident, so both train batches (from the
    # workers) and val batches (from synchronous sample_val) arrive on CPU and need a
    # tiny H2D copy before train_step/eval_step. In the default path batches already
    # live on `device`, so to_device is a no-op -- this avoids any double-move.
    cpu_batches = prefetcher is not None

    def to_device(batch):
        if cpu_batches:
            return batch.to(device, non_blocking=True)
        return batch

    steps_per_epoch = int(args.steps_per_epoch)
    total_steps = int(args.epochs) * steps_per_epoch
    best_val = float("inf")
    start_step = 0
    if resume_path is not None:
        start_step, best_val = _resume(raw_model, optimizer, rng, resume_path, logger)

    for global_step in range(start_step, total_steps):
        epoch = global_step // max(1, steps_per_epoch)
        batch = to_device(get_train_batch())
        metrics = train_step(model, batch, optimizer, perceptual, perceptual_weight)
        logger.info("step=%s loss=%.6f recon_l1=%.6f kl=%.6f", global_step, metrics["loss"], metrics["recon_l1"], metrics["kl"])
        metrics_writer.log("train", global_step, metrics, epoch)

        if args.val_every > 0 and global_step % args.val_every == 0:
            val_metrics = _validate(model, sample_val, args.val_batches, to_device)
            logger.info("VAL step=%s loss=%.6f recon_l1=%.6f", global_step, val_metrics["loss"], val_metrics["recon_l1"])
            metrics_writer.log("val", global_step, val_metrics, epoch)
            is_best = val_metrics["loss"] < best_val
            best_val = min(best_val, val_metrics["loss"])
            save_training_checkpoint(save_dir, _checkpoint_state(raw_model, optimizer, rng, config, global_step + 1, epoch, val_metrics["loss"], best_val), is_best)

    final_val = _validate(model, sample_val, args.val_batches, to_device)
    metrics_writer.log("val", total_steps, final_val, int(args.epochs))
    is_best = final_val["loss"] < best_val
    best_val = min(best_val, final_val["loss"])
    save_training_checkpoint(save_dir, _checkpoint_state(raw_model, optimizer, rng, config, total_steps, int(args.epochs), final_val["loss"], best_val), is_best)
    metrics_writer.close()
    if prefetcher is not None:
        prefetcher.close()
    if cache is not None:
        cache.close()
    logger.info("Training finished. best_val_loss=%.6f", best_val)


def _validate(model, sample_val, n_batches, to_device=None):
    accum = {}
    for _ in range(max(1, n_batches)):
        batch = sample_val()
        # CPU-resident-cache mode: move the val batch to the model's device. Default
        # mode passes a no-op to_device (batch already on device) -> no double-move.
        if to_device is not None:
            batch = to_device(batch)
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
    # No --config given -> defaults are intended.
    if not path:
        return _deep_copy_config(fallback)
    # An EXPLICIT --config that can't be read must FAIL LOUDLY, never silently fall
    # back to defaults: a silent fallback (e.g. PyYAML missing) once trained many runs
    # with the wrong config (small default model, prediction_type=epsilon instead of
    # the requested flow) without any error -- a very costly, hard-to-spot bug.
    if not os.path.exists(path):
        raise FileNotFoundError(f"--config path does not exist: {path}")
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            f"--config {path} was provided but PyYAML is not importable in this "
            f"environment, so the config cannot be read. Install it (pip install "
            f"pyyaml). Refusing to silently fall back to default hyperparameters."
        ) from exc
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
