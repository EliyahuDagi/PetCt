"""Fine-tune a 3D AutoencoderKL inflated from the trained 2D AE.

Stage 2 of the pipeline (between ae2d and the diffusion stages). Loads the frozen
2D AE checkpoint, builds a 3D AutoencoderKL with the *same* architecture, centre-
inflates the 2D weights into it (so each 3D conv starts out behaving like its 2D
counterpart on every z-plane) and fine-tunes on pooled NAC+AC 3D crops.

Two geometries share this trainer:

* Cube mode (default, ``slab_depth: 0``): the whole volume is squeezed into a
  ``crop_size`` cube, so the depth axis is resampled, and the 3D autoencoder
  compresses depth along with rows and columns. The latent is (B, C, d, h, w) with
  d < D. This path is unchanged.
* Slab mode (``slab_depth > 0``, implies ``model.anisotropic: true``): volumes are
  resized in-plane only and keep every native slice. The autoencoder downsamples
  in-plane only and normalises and attends per slice, so it solves exactly the
  problem the 2D autoencoder solves -- same in-plane compression, depth never
  compressed -- and the centre-inflated 2D weights start out as the 2D autoencoder
  run slice by slice. Training sees random contiguous depth slabs; validation sees a
  fixed centre window of the whole volume at native depth.

Emits JSONL metrics and best/last checkpoints (config embedded for inference, which
is why ``model.anisotropic`` and the slab keys are written back into it).
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
    reuse_split_json,
    sample_volume_native,
    sample_volume_slabs,
    sample_volumes,
    sample_volumes_full,
    write_split_json,
)
from src.training.dataset_index import enumerate_patients
from src.training.utils.augment import build_aug_3d
from src.training.models.anisotropic import is_inplane_only
from src.training.models.autoencoder3d import build_autoencoder_3d
from src.training.models.inflation import plan_inflation
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
from src.training.utils.schedule import build_lr_scheduler

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
    """Centre-inflate the 2D autoencoder weights into the 3D autoencoder.

    Returns the plan from :func:`plan_inflation` so the caller can report how many
    tensors were mapped, how many of those were convolutions that had to be inflated
    (their centre depth tap holds the 2D weights, the other taps are zero), and how many
    found no counterpart and stay at their fresh random init.
    """
    plan = plan_inflation(state_dict_2d, model_3d.state_dict())
    model_3d.load_state_dict(plan["mapped"], strict=False)
    return plan


def _centre_window(vol, window):
    """The middle ``window`` slices of a ``(1, 1, Z, H, W)`` volume (all of it if shorter).

    Slab-mode validation measures a fixed piece of the body, not a randomly chosen one:
    a random depth position makes the validation number wander between checkpoints, and
    between a fresh run and a ``--resume``, for reasons that have nothing to do with the
    weights. The flow run's validation was pure noise for exactly that kind of reason.
    """
    z_dim = int(vol.shape[2])
    w = min(z_dim, int(window))
    z0 = (z_dim - w) // 2
    return vol[:, :, z0:z0 + w].contiguous()


def load_ae2d_config(ae2d_ckpt):
    if not ae2d_ckpt or not os.path.exists(ae2d_ckpt):
        raise FileNotFoundError(
            "2D AE checkpoint not found at %r. Train ae2d first (writes outputs/ae2d/best.pt)." % ae2d_ckpt
        )
    state = load_checkpoint(ae2d_ckpt)
    # Prefer the EMA shadow when the 2D AE checkpoint carries one: inference loads the AE
    # with use_ema=True, so the EMA is the model whose quality the inflated 3D AE inherits.
    # Same selection as init_weights_from / train_ft3d._inflation_source. (train_ae2d
    # writes no EMA today, so this falls through to the raw weights unchanged.)
    ema_state = state.get("ema")
    src_sd = ema_state.get("shadow") if isinstance(ema_state, dict) else None
    src_name = "EMA shadow" if src_sd else "raw model"
    src_sd = src_sd or state.get("model", state)
    return state.get("config", {}), src_sd, src_name


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
    parser.add_argument("--slab_depth", type=int, default=None,
                        help="Slab mode: number of native slices per training slab. 0 = cube "
                             "mode, the default and the original behaviour. Unset falls back to "
                             "the config's slab_depth (itself 0). Implies model.anisotropic and "
                             "refuses --extra_levels.")
    parser.add_argument("--slab_inplane_size", type=int, default=None,
                        help="Slab mode: the in-plane size (rows = columns) every slice is "
                             "resized to; the depth is kept native. This is what --crop_size "
                             "means for the in-plane axes in cube mode. Unset falls back to the "
                             "config's slab_inplane_size (itself 128).")
    parser.add_argument("--slab_window", type=int, default=None,
                        help="Slab mode: number of slices in the FIXED centre window each "
                             "validation volume is scored on. Unset falls back to the config's "
                             "slab_window (itself 64).")
    parser.add_argument("--anisotropic", action="store_true",
                        help="Set model.anisotropic=true: the 3D autoencoder downsamples "
                             "in-plane only (depth stride 1) and normalises and attends per "
                             "slice, so the centre-inflated 2D autoencoder is reproduced exactly "
                             "at step 0. Implied by --slab_depth > 0.")
    parser.add_argument("--split_json", default=None,
                        help="Reuse the by-patient partition from an existing split.json "
                             "(e.g. outputs/ft3d_flow_p/split.json) instead of deriving one. "
                             "Use it whenever this autoencoder will feed a model judged on "
                             "that run's test patients: the same seed and fractions do NOT "
                             "reproduce a partition across runs, so a fresh split would train "
                             "on patients that run holds out. Patients the file does not list "
                             "(the unpaired ones a diffusion split cannot contain) are added "
                             "to training.")
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
        # --- Slab mode (see the module docstring). Every key is written to the checkpoint. ---
        # Native slices per training slab. 0 = cube mode (the whole volume squeezed into
        # a crop_size cube, the original behaviour, unchanged). >0 = slab mode.
        "slab_depth": 0,
        # In-plane size the slices are resized to in slab mode. Recorded so inference and
        # evaluation can rebuild the geometry from the checkpoint alone.
        "slab_inplane_size": 128,
        # Slices in the fixed centre window each validation volume is scored on.
        "slab_window": 64,
    }

    config = _load_config(args.config, default_config)
    save_dir = args.save_dir or config["output_dir"]
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.perceptual_weight is not None:
        config["perceptual_weight"] = args.perceptual_weight

    # --- Slab mode resolution. slab_depth == 0 keeps the cube path exactly as before;
    # the keys are still written to the config so every checkpoint says which geometry
    # produced it (infer.py and evaluate.py rebuild the autoencoder from that config
    # through build_autoencoder_3d, so model.anisotropic in particular MUST survive here
    # or the rebuilt model is a different network).
    if args.slab_depth is not None:
        config["slab_depth"] = int(args.slab_depth)
    slab_depth = int(config.get("slab_depth", 0) or 0)
    config["slab_depth"] = slab_depth
    slab_mode = slab_depth > 0
    if args.slab_inplane_size is not None:
        config["slab_inplane_size"] = int(args.slab_inplane_size)
    slab_inplane_size = int(config.get("slab_inplane_size", 128) or 128)
    config["slab_inplane_size"] = slab_inplane_size
    if args.slab_window is not None:
        config["slab_window"] = int(args.slab_window)
    slab_window = int(config.get("slab_window", 64) or 64)
    config["slab_window"] = slab_window
    # The model block is replaced wholesale by the 2D autoencoder's further down, so read
    # the requested in-plane-only flag out of the config BEFORE that happens.
    want_anisotropic = bool(args.anisotropic or (config.get("model") or {}).get("anisotropic", False))
    if slab_mode and args.extra_levels:
        raise ValueError(
            "slab mode (--slab_depth %d) refuses --extra_levels %d. Every extra level halves "
            "the in-plane grid again, and the whole point of slab mode is to keep the 2D "
            "autoencoder's compression ratio so the inflated weights start out exact. Drop "
            "--extra_levels, or train in cube mode." % (slab_depth, args.extra_levels))

    resume_path = resolve_resume_path(save_dir, args.resume)

    logger = setup_logging(save_dir)
    metrics_writer = MetricsWriter(save_dir, TASK, append=resume_path is not None)

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    configure_backends(device, logger)

    if slab_mode:
        if not want_anisotropic:
            # Without the in-plane-only build the autoencoder still shrinks the depth axis,
            # and the inflated 2D weights are no longer an exact starting point.
            want_anisotropic = True
            logger.info(
                "Slab mode: setting model.anisotropic to true (it was not asked for). A "
                "depth-compressing 3D autoencoder would shrink the depth axis, which is "
                "exactly the harder problem slab mode exists to avoid.")
        logger.info(
            "SLAB MODE: in-plane %dx%d (--slab_inplane_size), native depth kept (slices are "
            "never resampled along depth); training slabs of %d slices; validation on a fixed "
            "centre window of %d slices of the whole volume; the autoencoder solves the same "
            "problem as the 2D one (same in-plane compression, depth never compressed). "
            "--crop_size (%d) is not used in slab mode.",
            slab_inplane_size, slab_inplane_size, slab_depth, slab_window, args.crop_size)

    # Build the 3D AE with the SAME architecture as the 2D AE so weights inflate
    # cleanly; the 2D AE's config is the source of truth and is re-embedded here.
    ae2d_config, ae2d_state, ae2d_src_name = load_ae2d_config(args.ae2d_ckpt)
    config["model"] = dict(ae2d_config.get("model", {}))
    config["latent_channels"] = int(ae2d_config.get("latent_channels", 4))
    # Re-apply the in-plane-only flag: the line above just replaced the model block with
    # the 2D autoencoder's, which of course never carried it. It has to end up in the
    # SAVED config, because that is what inference and evaluation rebuild from.
    if want_anisotropic:
        config["model"]["anisotropic"] = True
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
    # Nothing in slab mode bypasses this: the same load_ae2d_config (which prefers the
    # weight-averaged shadow when the 2D checkpoint has one) feeds the same centre
    # inflation, and with the in-plane-only build that start is the 2D autoencoder
    # applied slice by slice, to floating-point tolerance.
    if resume_path is None:
        plan = inflate_and_load(raw_model, ae2d_state)
        logger.info(
            "Inflated 3D autoencoder from %s (%s): %d tensors mapped, %d convolutions "
            "centre-inflated, %d left at fresh init.",
            args.ae2d_ckpt, ae2d_src_name, len(plan["mapped"]),
            len(plan["inflated_conv_keys"]), len(plan["missing"]))

    # One line saying whether the surgery actually took, so a run's log answers the
    # question "did this start as an exact copy of the 2D autoencoder?" on its own.
    logger.info("Autoencoder is in-plane only (depth never compressed): %s.",
                is_inplane_only(raw_model))

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
    aug_cfg = config.get("augment")
    if slab_mode and isinstance(aug_cfg, dict) and aug_cfg.get("enabled", False):
        # Affine and elastic transforms rotate/warp ACROSS the depth axis and smear
        # neighbouring slices into each other. Slab mode exists to keep native slices
        # intact, so both are pinned to 0 here (flips and quarter-turns stay). An unset
        # affine_prob would otherwise fall back to the augment module's non-zero default.
        affine_p = float(aug_cfg.get("affine_prob") or 0.0)
        elastic_p = float(aug_cfg.get("elastic_prob") or 0.0)
        if affine_p > 0 or elastic_p > 0 or "affine_prob" not in aug_cfg:
            logger.warning(
                "Slab mode: augment.affine_prob (%s) and augment.elastic_prob (%s) forced to 0; "
                "these transforms would resample across the depth axis.",
                aug_cfg.get("affine_prob", "unset"), aug_cfg.get("elastic_prob", "unset"))
        aug_cfg = dict(aug_cfg)
        aug_cfg["affine_prob"] = 0.0
        aug_cfg["elastic_prob"] = 0.0
        config["augment"] = aug_cfg
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

        if slab_mode:
            # Slab mode, one patient: the depth axis is split into a train band and a val
            # band exactly as the cube path's _band_crop does, so the two never overlap.
            # Imported here so the cube path does not depend on the slab-mode helpers.
            from src.training.data import _band_range

            def _depth_band(phase):
                out = []
                for v in pool_vols:
                    z0, z1 = _band_range(int(v.shape[0]), phase, args.val_fraction)
                    out.append(v[z0:z1])
                return out

            train_band = _depth_band("train")
            val_band = _depth_band("val")

            def sample_train():
                return sample_volume_slabs(train_band, batch_size, slab_inplane_size,
                                           slab_depth, rng, augment=True, geo_aug=train_aug)

            def sample_val():
                # The first pooled volume (attenuation-corrected PET when the patient has
                # one), whole val band at native depth, then a FIXED centre window: the
                # same slices at every validation and after a --resume.
                return _centre_window(
                    sample_volume_native(val_band[0], slab_inplane_size), slab_window)
        else:
            def sample_train():
                return sample_volumes(pool_vols, batch_size, args.crop_size, "train", args.val_fraction, rng, geo_aug=train_aug)

            def sample_val():
                return sample_volumes(pool_vols, batch_size, args.crop_size, "val", args.val_fraction, rng, geo_aug=None)
    else:
        if args.split_json:
            # REUSE an existing partition instead of deriving a new one. Required whenever
            # this autoencoder feeds a model that will be judged on another run's test
            # patients: the same seed and fractions do NOT reproduce a partition across
            # runs (the enumerated patient order is not stable), so a fresh split would
            # quietly train on patients that run holds out. extra_to_train=True keeps the
            # patients the split file never listed -- the autoencoder pools unpaired
            # volumes, so it sees patients the paired diffusion split could not contain.
            train_idx, val_idx, test_idx = reuse_split_json(
                args.split_json, patients, logger, extra_to_train=True)
        else:
            train_idx, val_idx, test_idx = make_patient_split3(
                len(patients), args.val_fraction, args.test_fraction, seed)
        logger.info(
            "Training on %d patients (train %d / val %d / test %d) across %d roots. "
            "Test patients are held out from training entirely.%s",
            len(patients), len(train_idx), len(val_idx), len(test_idx), len(args.data_dir),
            f" Split REUSED from {args.split_json}." if args.split_json else "",
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

        def _batch_from_pool(pool_vols, augment, geo_aug):
            # One helper for the prefetch and synchronous paths. Cube mode: the original
            # whole-volume cube sampler. Slab mode: random contiguous runs of slab_depth
            # native slices, resized in-plane only.
            if slab_mode:
                return sample_volume_slabs(pool_vols, batch_size, slab_inplane_size, slab_depth,
                                           rng, augment=augment, geo_aug=geo_aug)
            return sample_volumes_full(pool_vols, batch_size, args.crop_size, rng, augment=augment, geo_aug=geo_aug)

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
            return _batch_from_pool(pool_vols, augment, geo_aug)

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
            return _batch_from_pool(pool_vols, True, train_aug)

        if slab_mode:
            # Slab mode validates the SAME patients at every validation, in the same order,
            # each on a FIXED centre window of its slices. Nothing about the measurement is
            # allowed to move on its own: the cube path draws a random patient (from the
            # TRAINING random generator) and a random crop per call, so consecutive
            # validations of the same weights disagree and best.pt gets picked by luck of
            # the draw. That is exactly how the flow run's validation ended up being pure
            # noise. The fixed list depends only on (seed, val_idx), so a --resume run sees
            # the same patients, and validation no longer consumes the training generator.
            # The two helpers are shared with the flow trainer rather than copied.
            from src.training.train.train_ft3d import _FixedValCursor, _fixed_val_indices

            val_cursor = _FixedValCursor(_fixed_val_indices(val_idx, args.val_batches, seed))
            logger.info(
                "Slab-mode validation: fixed set of %d validation patients (indices %s), each "
                "scored on a fixed centre window of %d slices.",
                len(val_cursor.indices), val_cursor.indices, slab_window)

            def sample_val():
                # Whole volume at native depth, batch of one, no augmentation; then the
                # centre window. Walks the fixed list in order; _validate calls
                # sample_val.reset() once per pass. ae_pool_volumes puts the
                # attenuation-corrected PET first, so the choice of volume is fixed too.
                pool_vols = ae_pool_volumes(cache.get(val_cursor.next_index()), args.modality)
                if not pool_vols:
                    raise ValueError("No volumes available for 3D AE validation on this patient.")
                return _centre_window(
                    sample_volume_native(pool_vols[0], slab_inplane_size), slab_window)

            sample_val.reset = val_cursor.reset
        else:
            def sample_val():
                return _sample_sync(val_idx, False, geo_aug=None)

    steps_per_epoch = int(args.steps_per_epoch)
    total_steps = int(args.epochs) * steps_per_epoch

    # Learning-rate warmup, the same schedule the flow trainer (train_ft3d) uses. The
    # model starts as an exact copy of the 2D autoencoder, and Adam's very first update
    # moves EVERY weight by a full learning rate in the direction of its gradient sign
    # (the second-moment estimate is still just that one gradient squared), the depth
    # taps that are exactly zero included. Measured 2026-09-06 on the slab run: that one
    # step took the fixed-set validation loss from 0.0030 to 0.0151 and the training
    # loss was still 50% above its start 375 steps later. A linear ramp from ~0 over the
    # first lr_warmup_steps keeps the early updates small while Adam's moment estimates
    # settle. build_lr_scheduler then decays the rate along a cosine to lr_min_ratio
    # (default 0) of the base rate at lr_total_steps; both keys are written into the
    # embedded config. Nothing here is slab-specific: cube mode gets the same ramp.
    config.setdefault("lr_total_steps", total_steps)
    config.setdefault("lr_warmup_steps", min(500, max(1, total_steps // 10)))
    lr_scheduler = build_lr_scheduler(optimizer, config)
    logger.info("Learning rate: %.2e with %d warmup steps (%s)",
                float(config.get("learning_rate", 1.0e-4)), int(config["lr_warmup_steps"]),
                _describe_lr_schedule(config, lr_scheduler))

    best_val = float("inf")
    start_step = 0
    if resume_path is not None:
        start_step, best_val = _resume(raw_model, optimizer, rng, resume_path, logger)
        # Fast-forward the schedule so the resumed rate matches an uninterrupted run
        # (train_ft3d does the same; no scheduler state is saved in the checkpoint, the
        # schedule is a pure function of the step count and the config).
        if lr_scheduler is not None:
            for _ in range(start_step):
                lr_scheduler.step()

    for global_step in range(start_step, total_steps):
        epoch = global_step // max(1, steps_per_epoch)
        batch = sample_train()
        # The rate THIS step's update is taken at: read before the scheduler advances,
        # so the logged value is the one the optimizer actually used.
        lr_now = float(optimizer.param_groups[0]["lr"])
        metrics = train_step(model, batch, optimizer, perceptual, perceptual_weight)
        if lr_scheduler is not None:
            lr_scheduler.step()
        metrics["lr"] = lr_now
        logger.info("step=%s loss=%.6f recon_l1=%.6f kl=%.6f lr=%.2e", global_step, metrics["loss"], metrics["recon_l1"], metrics["kl"], lr_now)
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
    # Slab mode: restart the fixed validation list at patient 0 so every pass scores the
    # same patients in the same order. The cube-mode sampler has no reset hook, so this
    # is a no-op there.
    reset = getattr(sample_val, "reset", None)
    if reset is not None:
        reset()
    for _ in range(n):
        batch = sample_val()
        m = eval_step(model, batch)
        for k, v in m.items():
            accum[k] = accum.get(k, 0.0) + v
    return {k: v / n for k, v in accum.items()}


def _describe_lr_schedule(config, lr_scheduler):
    """One plain-English phrase for the startup log line (what happens after the warmup)."""
    if lr_scheduler is None:
        return "constant, no warmup: lr_total_steps is unset or 0"
    base_lr = float(config.get("learning_rate", 1.0e-4))
    min_ratio = float(config.get("lr_min_ratio", 0.0))
    return "linear warmup, then cosine decay to %.2e by step %d (lr_min_ratio=%g)" % (
        base_lr * min_ratio, int(config["lr_total_steps"]), min_ratio)


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
