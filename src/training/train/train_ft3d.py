"""Train the 3D latent diffusion UNet for NAC->AC translation.

A frozen 3D AutoencoderKL (inflated, from train_ae3d) encodes paired NAC/AC 3D
crops into 3D latent volumes (B, C, d, h, w) with depth compression. The 3D UNet
is conditioned on the NAC latent volume by channel-concatenation and trained to
predict noise added to the AC latent volume. Optionally inflates weights from a
trained 2D diffusion checkpoint. Emits JSONL metrics and best/last checkpoints.
"""

import argparse
import json
import os
import random

import numpy as np
import torch

from src.training.data import (
    PrefetchingPatientCache,
    filter_paired_patients_cached,
    load_patient_volumes,
    make_patient_split3,
    sample_pair_volumes,
    sample_pair_volumes_full,
    write_split_json,
)
from src.training.dataset_index import enumerate_patients
from src.training.utils.augment import build_aug_3d
from src.training.models.autoencoder3d import (
    ae3d_decode,
    ae3d_encode,
    build_autoencoder_3d,
)
from src.training.models.diffusion3d import build_diffusion_3d
from src.training.models.inflation import (
    CenterFreeze,
    build_center_freeze_plan,
    map_state_dict_2d_to_3d,
)
# Reuse the dimension-agnostic flow/perceptual helpers from the 2D port (the flow math
# broadcasts to 5-D via schedule._broadcast, the estimate helpers index only shape[0]/
# ndim, and ae.decode is the same MONAI API for the 3D AE); do NOT duplicate them here.
# ft3d previously kept its own epsilon-only copies of x0_decode_in_graph/perceptual_term,
# which is how the flow perceptual path silently never reached 3D.
from src.training.train.train_diff2d import (
    _resume_best_val,
    _selection_metric,
    diffusion_in_channels,
    flow_ac_estimate_latent,
    flow_loss,
    flow_mode_banner,
    flow_x0_decode_in_graph,
    perceptual_term,
    sample_flow_timesteps,
    x0_decode_in_graph,
    x0_estimate_latent,
)
from src.training.utils.checkpointing import (
    init_weights_from,
    capture_rng_state,
    load_checkpoint,
    resolve_resume_path,
    restore_rng_state,
    save_training_checkpoint,
)
from src.training.utils.image_metrics import clamp_unit, image_quality_metrics
from src.training.utils.logging import setup_logging
from src.training.utils.perceptual import build_perceptual_loss
from src.training.utils.quant_losses import build_latent_weight, build_quant_loss
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

TASK = "ft3d"
SPATIAL_DIMS = 3


def build_model(config):
    return build_diffusion_3d(config)


def inflate_and_load(model_3d, state_dict_2d):
    mapped, missing = map_state_dict_2d_to_3d(state_dict_2d, model_3d.state_dict())
    model_3d.load_state_dict(mapped, strict=False)
    return missing


def load_frozen_ae(ae_ckpt, device):
    if not ae_ckpt or not os.path.exists(ae_ckpt):
        raise FileNotFoundError(
            "3D AE checkpoint not found at %r. Train ae3d first (it writes outputs/ae3d/best.pt)." % ae_ckpt
        )
    state = load_checkpoint(ae_ckpt)
    ae_config = state.get("config", {})
    ae = build_autoencoder_3d(ae_config).to(device)
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
        nac_vol, ac_vol = sample_train_pair()
        samples.append(ae3d_encode(ae, ac_vol).flatten())
    std = float(torch.cat(samples).std().cpu())
    scale = 1.0 / (std + 1e-8)
    if logger is not None:
        logger.info("Computed latent_scale=%.6f (AC latent std=%.6f over %d batches)", scale, std, max(1, n_batches))
    return scale


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


def _reuse_split(split_json, patients, logger):
    """Map an existing ``split.json``'s patient PATHS onto indices into ``patients``.

    Returns ``(train_idx, val_idx, test_idx)``. Raises if the file is unusable or if any
    TEST patient is missing from the current enumeration -- silently dropping test
    patients would make the run's numbers incomparable to the split it claims to reuse,
    which is the whole point of passing this flag.

    Patients present now but absent from the split file are dropped from training with a
    warning (they were not part of the original partition, so training on them could leak
    into that split's test set).
    """
    with open(split_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not all(k in data for k in ("train", "val", "test")):
        raise ValueError(f"{split_json} is not a split.json (need train/val/test lists)")
    pos = {str(p): i for i, p in enumerate(patients)}
    out, missing = [], {}
    for key in ("train", "val", "test"):
        idx, miss = [], []
        for p in data[key]:
            i = pos.get(str(p))
            (idx.append(i) if i is not None else miss.append(str(p)))
        out.append(idx)
        missing[key] = miss
    if missing["test"]:
        raise ValueError(
            f"{len(missing['test'])} TEST patient(s) from {split_json} are not present under "
            f"the given --data_dir roots; refusing to reuse a split whose held-out set cannot "
            f"be reproduced. First missing: {missing['test'][0]}")
    for key in ("train", "val"):
        if missing[key]:
            logger.warning("%d %s patient(s) from %s are absent now and will be skipped.",
                           len(missing[key]), key, split_json)
    extra = len(patients) - sum(len(i) for i in out)
    if extra > 0:
        logger.warning(
            "%d enumerated patient(s) are NOT in %s and are excluded from training "
            "(training on them could leak into that split's test set).", extra, split_json)
    return out[0], out[1], out[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, nargs="+", help="One or more dataset roots / patient folders")
    parser.add_argument("--patient_index", type=int, default=0)
    parser.add_argument("--config", default=None)
    parser.add_argument("--ae_ckpt", default="outputs/ae3d/best.pt", help="Frozen 3D AE checkpoint (from train_ae3d)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=20)
    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--test_fraction", type=float, default=0.1,
                        help="By-patient TEST holdout fraction (multi-patient only). "
                             "Test patients are excluded from BOTH train and val.")
    parser.add_argument("--val_batches", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--latent_size", type=int, default=64, help="Cube size of 3D crops fed to the AE encoder")
    parser.add_argument(
        "--perceptual_weight",
        type=float,
        default=None,
        help="Override the perceptual-loss weight (0 disables it). Default comes from the config.",
    )
    parser.add_argument("--inflate_from", default=None, help="Optional 2D diffusion checkpoint to inflate")
    parser.add_argument("--init_from", default=None,
                        help="Fine-tune init: load MODEL WEIGHTS ONLY from an existing 3D "
                             "checkpoint (prefers its EMA shadow). Unlike --resume this keeps a "
                             "fresh optimizer/LR schedule/step/best_val, and unlike "
                             "--inflate_from the source is already 3D.")
    parser.add_argument("--split_json", default=None,
                        help="Reuse the by-patient partition from an existing split.json "
                             "instead of deriving one. REQUIRED for a valid A/B: the same "
                             "seed+fractions do NOT reproduce a partition across runs (the "
                             "enumerated patient order is not stable), so independently "
                             "trained runs otherwise leak each other's test patients.")
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
        "output_dir": "outputs/ft3d",
        "batch_size": 1,
        # micro-batches accumulated per optimizer step; effective batch = batch_size * grad_accum_steps
        "grad_accum_steps": 1,
        "learning_rate": 1.0e-4,
        "latent_channels": 4,
        # "epsilon": standard latent diffusion (start sampling from N(0,1), NAC as a
        # concat condition). "flow": NAC->AC rectified-flow bridge (start sampling FROM
        # the NAC latent so the model cannot ignore it; see flow_loss / flow_sample).
        "prediction_type": "epsilon",
        "noise_schedule": "cosine",
        "snr_gamma": 5.0,
        "rescale_zero_terminal_snr": False,
        "ema_decay": 0.9999,
        "lr_min_ratio": 0.1,
        # --- Gradient-masked center-freeze warm-start (Make-A-Video / Video-LDM) ---
        # When inflating a 2D diffusion checkpoint (--inflate_from) into this 3D UNet,
        # optionally FREEZE the inflated 2D spatial prior and train ONLY the new
        # depth-axis capacity, then optionally release it. OFF by default -- with the
        # flag off the code path is byte-identical to before. See
        # models/inflation.build_center_freeze_plan / CenterFreeze.
        #   True: pin the center depth slice of every inflated 3D conv (the 2D prior)
        #   via a gradient mask, and freeze every other 2D-derived param
        #   (norms/biases/time-embed/attention/1x1x1 convs) with requires_grad=False;
        #   only the zero-initialized off-center depth taps (and any genuinely-fresh
        #   params) train. Requires an actual inflation to have happened.
        "inflate_freeze_backbone": False,
        #   Optimizer step at which to release the freeze and fine-tune the whole net.
        #   0 = never unfreeze (stay depth-only for the entire run); N>0 = unfreeze at
        #   step N. Resume re-establishes the freeze from the checkpoint-embedded plan.
        "inflate_unfreeze_step": 0,
        # Perceptual loss on the in-graph decoded x0 estimate (2D backbone run
        # slice-wise on the 3D decode). >0 enables it; 0/disabled is a no-op.
        # Backend is pluggable ("vgg" now, "medical_sam" later).
        "perceptual_weight": 0.1,
        # Apply perceptual only to the low-noise (high-SNR) timesteps where the one-step
        # x0 estimate is meaningful; see PixelGen.
        "perceptual_active_frac": 0.7,
        "perceptual_backend": "vgg",
        # "lpl" backend only: decoder block indices to tap (null = auto-select interior
        # blocks up to the last Upsample, so the full-resolution tail never runs).
        "perceptual_taps": None,
        # Clamp the decoded validation prediction to the valid [0,1] range so the
        # in-training monitor matches the (clamped) eval path. normalize_volume clips
        # every GT volume to [0,1]; the AE decoder's linear output conv overshoots.
        "clamp_output": True,
        # Flow-only knobs (inert in epsilon mode). See sample_flow_timesteps / flow_loss.
        # "uniform" + "none" reproduce the original flow behaviour exactly.
        "flow_tau_dist": "uniform",     # uniform | ushaped  (RFPP U-shaped density)
        "flow_tau_a": 4.0,              # ushaped sharpness
        "flow_loss_weighting": "none",  # none | rfpp        ((1-tau)-weighted MSE)
        # Quantitative-fidelity term (flow only), off by default.
        # quant_loss: none | slope | intensity_weighted | expectile
        "quant_loss": "none",
        "quant_weight": 0.0,
        "quant_variance_weight": None,  # slope: adds (std(pred)/std(gt)-1)^2
        "quant_lam": None,              # intensity_weighted: weight = 1 + lam*(gt/max)^gamma
        "quant_gamma": None,
        "quant_q": None,                # expectile: q>0.5 penalizes under-prediction more
        # Spatial rebalancing of the velocity MSE: weight for latent positions holding NO
        # anatomy. 1.0 = off (uniform mean, the original behaviour); 0.1 = air counts 10%.
        # Value-INDEPENDENT (a foreground gate, not an intensity ramp), so unlike
        # intensity weighting it cannot encourage saturation.
        "bg_weight": 1.0,
        "bg_fg_threshold": None,
        # Which spatial weight map that floor belongs to (flow only):
        #   "occupancy" -- binary anatomy gate, value-independent (the original behaviour)
        #   "intensity" -- ramps with the GT PET value, w = bg + (1-bg)*(pet/q)^gamma with
        #                  q an outlier-resistant percentile of the foreground, so a single
        #                  saturated voxel cannot shrink every other weight. Evaluated on
        #                  the SMALL latent grid (a pooled GT map), NOT on a full-resolution
        #                  decode -- contrast quant_loss: intensity_weighted, which pays for
        #                  the decode. Intensity-monotone => can reward hallucinated uptake.
        #   "none"      -- off
        "latent_weight": "occupancy",
        "iw_source": "ac",       # ac | nac | union -- whose PET value sets the weight
        "iw_gamma": 1.0,         # >1 concentrates on the hottest tissue, <1 flattens
        "iw_percentile": 99.0,   # foreground percentile used as the normalizer
        "iw_clip": 1.0,          # cap on pet/q before the ramp (1.0 = no runaway weights)
        "iw_pool": "avg",        # avg | max pooling of the GT map onto the latent grid
        "model": {
            "num_channels": [16, 32, 48],
            "attention_levels": [False, False, True],
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
    prediction_type = str(config.get("prediction_type", "epsilon")).lower()
    is_flow = prediction_type == "flow"
    # Channel layout (see diffusion_in_channels): epsilon conditions by concatenation
    # so input is [noisy_AC | NAC] (2C); the flow bridge feeds ONLY the interpolant
    # x_t (C) because the trajectory already starts FROM NAC -- concatenating NAC too
    # would let the net recover AC=(x_t - tau*NAC)/(1-tau) by algebra for tau<1.
    model_cfg = dict(config.get("model", {}))
    model_cfg["in_channels"] = diffusion_in_channels(latent_channels, is_flow)
    model_cfg["out_channels"] = latent_channels
    config["model"] = model_cfg

    # raw_model owns the weights (inflation / EMA / state_dict / resume); model is
    # the (optionally) compiled forward handle. They share parameters.
    raw_model = build_model(config).to(device)
    raw_model = to_model_memory_format(raw_model, SPATIAL_DIMS)
    optimizer = torch.optim.Adam(raw_model.parameters(), lr=float(config.get("learning_rate", 1.0e-4)))
    schedule = DiffusionSchedule(
        schedule=config.get("noise_schedule", "cosine"),
        rescale_zero_terminal_snr=bool(config.get("rescale_zero_terminal_snr", False)),
        device=device,
    )
    snr_gamma = config.get("snr_gamma", 5.0)
    if is_flow:
        logger.info(flow_mode_banner(include_cond_dropout=False))

    # Pluggable perceptual loss. Config-driven in BOTH modes (it used to be force-zeroed
    # in flow mode, which is why the flow perceptual term never reached 3D): epsilon uses
    # the one-step x0 estimate (gated to low-noise t), flow uses the velocity AC estimate
    # x_t - tau*v at ALL tau. Backend "vgg" runs a 2D backbone slice-wise on the decoded
    # volume; "lpl" compares the frozen 3D decoder's own features and skips the decode.
    perceptual_weight = float(config.get("perceptual_weight", 0.0) or 0.0)
    perceptual_active_frac = float(config.get("perceptual_active_frac", 0.7))
    clamp_output = bool(config.get("clamp_output", True))
    perceptual = None
    if perceptual_weight > 0:
        perceptual = build_perceptual_loss(
            config.get("perceptual_backend", "vgg"),
            device=device,
            weights_path=config.get("perceptual_weights_path"),
            # "lpl" taps the frozen AE decoder's own features (ignored by other backends).
            ae=ae,
            taps=config.get("perceptual_taps"),
        )
        if perceptual is None:
            logger.warning("Perceptual loss requested but backend unavailable; continuing without it.")
            perceptual_weight = 0.0
        else:
            logger.info("Perceptual loss enabled: backend=%s weight=%.4g space=%s",
                        config.get("perceptual_backend", "vgg"), perceptual_weight,
                        "latent" if getattr(perceptual, "consumes_latents", False) else "image")

    # Flow-bridge loss shaping (inert in epsilon mode).
    flow_tau_dist = str(config.get("flow_tau_dist", "uniform") or "uniform")
    flow_tau_a = float(config.get("flow_tau_a", 4.0))
    flow_loss_weighting = str(config.get("flow_loss_weighting", "none") or "none")
    if is_flow and (flow_tau_dist != "uniform" or flow_loss_weighting != "none"):
        logger.info("Flow loss shaping: tau_dist=%s (a=%.3g) loss_weighting=%s "
                    "(RFPP, arXiv 2405.20320)", flow_tau_dist, flow_tau_a, flow_loss_weighting)

    # Quantitative-fidelity term: attacks the measured under-dispersion (reg_slope 0.844,
    # hot_band_rel_error -0.10) that plain MSE is *guaranteed* to produce, since its optimum
    # is the conditional mean. See src/training/utils/quant_losses.py.
    bg_weight = float(config.get("bg_weight", 1.0) if config.get("bg_weight") is not None else 1.0)
    bg_fg_threshold = config.get("bg_fg_threshold")
    # Spatial weight map on the velocity MSE: "occupancy" (binary anatomy gate, the
    # original bg_weight behaviour) or "intensity" (ramps with the GT PET value). Both
    # floor at bg_weight, so bg_weight=1.0 leaves either one a no-op.
    latent_weight_mode = str(config.get("latent_weight", "occupancy") or "occupancy")
    latent_weight_fn = build_latent_weight(
        latent_weight_mode, bg_weight=bg_weight, fg_threshold=bg_fg_threshold,
        source=config.get("iw_source", "ac"), gamma=float(config.get("iw_gamma", 1.0)),
        percentile=float(config.get("iw_percentile", 99.0)),
        clip=float(config.get("iw_clip", 1.0)), pool=config.get("iw_pool", "avg"),
    )
    if is_flow and bg_weight < 1.0 and latent_weight_mode == "occupancy":
        logger.info("Background down-weighting ENABLED: empty latent positions weighted %.3g "
                    "(mask from GT anatomy, intensity-independent)", bg_weight)
    elif is_flow and bg_weight < 1.0 and latent_weight_mode == "intensity":
        logger.info("PET-VALUE weighting of the latent velocity MSE ENABLED: w = %.3g + "
                    "%.3g*(pet/p%.4g)^%.3g clipped at %.3g, %s-pooled to the latent grid, "
                    "source=%s. Intensity-MONOTONE: watch for hallucinated uptake.",
                    bg_weight, 1.0 - bg_weight, float(config.get("iw_percentile", 99.0)),
                    float(config.get("iw_gamma", 1.0)), float(config.get("iw_clip", 1.0)),
                    config.get("iw_pool", "avg"), config.get("iw_source", "ac"))
    elif latent_weight_mode != "none" and bg_weight >= 1.0:
        latent_weight_fn = None  # floor 1.0 -> uniform weights; keep the original path.
    quant_weight = float(config.get("quant_weight", 0.0) or 0.0)
    quant_loss = None
    if quant_weight > 0:
        quant_loss = build_quant_loss(
            config.get("quant_loss"),
            fg_threshold=config.get("quant_fg_threshold"),
            **{k: config.get(f"quant_{k}") for k in ("variance_weight", "lam", "gamma", "q")},
        )
    if quant_loss is None and quant_weight > 0:
        logger.warning("quant_weight=%.4g but quant_loss is 'none' -- no term applied.", quant_weight)
    elif quant_loss is not None:
        if not is_flow:
            raise ValueError(
                "quant_loss is implemented for prediction_type=flow only: it acts on the "
                "decoded AC estimate, and the epsilon one-step x0 estimate is meaningless "
                "at high noise.")
        logger.info("Quantitative-fidelity loss enabled: %s weight=%.4g (on the decoded "
                    "x_t - tau*v AC estimate, foreground-masked)",
                    config.get("quant_loss"), quant_weight)

    # Inflation seeds the 3D weights; a --resume checkpoint (loaded below) takes
    # precedence and fully overwrites them, so skip the inflation work when resuming.
    # Inflate before compiling.
    inflated = False
    missing = []
    if resume_path is None and args.inflate_from and os.path.exists(args.inflate_from):
        state = load_checkpoint(args.inflate_from)
        # Inflation only carries over Conv weights whose in/out-channel counts match
        # (see map_state_dict_2d_to_3d). The input conv channel count is mode-dependent
        # (flow=C vs epsilon=2C), so inflating ACROSS a prediction_type switch leaves the
        # input conv shape-mismatched -> it silently falls back to fresh init. Warn loudly
        # and recommend inflating flow->flow (or epsilon->epsilon) only.
        ckpt_pt = str((state.get("config") or {}).get("prediction_type", "epsilon")).lower()
        if ckpt_pt != prediction_type:
            logger.warning(
                "Inflation prediction_type mismatch (2D checkpoint=%s, ft3d config=%s): the input conv "
                "channel count differs (flow=C vs epsilon=2C), so the input conv will NOT inflate and "
                "falls back to fresh init. Inflate flow->flow (or epsilon->epsilon) for a full warm-start.",
                ckpt_pt, prediction_type)
        missing = inflate_and_load(raw_model, state.get("model", state))
        inflated = True
        logger.info("Inflated from 2D checkpoint %s, missing keys: %s", args.inflate_from, len(missing))
    elif resume_path is None and args.init_from:
        # Fine-tune init from an existing 3D checkpoint (weights only). Mutually exclusive
        # with --inflate_from in practice: the source is already 3D, so nothing to inflate.
        if not os.path.exists(args.init_from):
            raise FileNotFoundError(f"--init_from checkpoint not found: {args.init_from}")
        src_cfg, _n, _s = init_weights_from(raw_model, args.init_from, logger, label='ft3d')
        src_pt = str((src_cfg or {}).get("prediction_type", "epsilon")).lower()
        if src_pt != prediction_type:
            raise ValueError(
                f"--init_from prediction_type mismatch: checkpoint is {src_pt!r} but this "
                f"config is {prediction_type!r}. The input conv channel count differs "
                f"(flow=C vs epsilon=2C), so the weights are not transferable.")

    # Gradient-masked center-freeze warm-start (Make-A-Video / Video-LDM). OFF by
    # default; nothing below touches the model unless inflate_freeze_backbone is set.
    # Applied to raw_model BEFORE maybe_compile so the first (lazy) compile captures
    # the frozen requires_grad state. The optimizer keeps ALL params (frozen ones are
    # skipped by Adam while their grad is None, and resume after unfreeze picks them up).
    freeze_backbone = bool(config.get("inflate_freeze_backbone", False))
    unfreeze_step = int(config.get("inflate_unfreeze_step", 0) or 0)
    center_freeze = None
    freeze_active = False
    if freeze_backbone and resume_path is None:
        if inflated:
            plan = build_center_freeze_plan(raw_model, missing)
            # Embed the plan in the config so it rides along in every checkpoint and
            # --resume can rebuild the freeze structurally (no 2D checkpoint needed).
            config["inflate_freeze_plan"] = plan
            center_freeze = CenterFreeze()
            summary = center_freeze.apply(raw_model, plan)
            freeze_active = True
            logger.info(
                "Center-freeze warm-start: masked=%d frozen=%d fresh=%d trainable_tensors=%d; "
                "unfreeze at step=%s",
                summary["n_masked"], summary["n_frozen"], summary["n_fresh"], summary["n_trainable"],
                unfreeze_step if unfreeze_step > 0 else "never",
            )
        else:
            logger.warning(
                "inflate_freeze_backbone is set but no inflation happened (pass a valid "
                "--inflate_from 2D checkpoint); training WITHOUT the freeze.")

    model = maybe_compile(raw_model, logger)

    rng = np.random.RandomState(seed)
    batch_size = int(config.get("batch_size", 1))

    # Geometric-only 3D train augmentation (no-op identity unless config opts in),
    # layered on top of the existing crop/flip/rot90. The SAME geometry is applied
    # to NAC and AC (paired). Val passes geo_aug=None for a stable metric.
    train_aug = build_aug_3d(config.get("augment"))

    # Diffusion requires paired NAC+AC. Enumerate across roots, keep only paired
    # patients (logging skips). <=1 paired patient -> legacy within-patient split.
    patients = enumerate_patients(args.data_dir, missing_ok=True)
    if len(patients) > 1:
        patients = filter_paired_patients_cached(patients, rescan=args.rescan_pairs, log=logger.info, cache_dir=args.cache_dir)
        if not patients:
            raise ValueError("No paired NAC+AC patients found for ft3d across the given roots.")
    multi_patient = len(patients) > 1
    cache = None  # set in the multi-patient branch; closed after training

    if not multi_patient:
        vols = load_patient_volumes(args.data_dir[0], args.patient_index, device=device)
        if vols.get("pet_nac") is None or vols.get("pet_ac") is None:
            raise ValueError("Both NAC and AC PET volumes are required for ft3d training.")
        logger.info("Training on 1 patient (train 1 / val 1) across %d roots.", len(args.data_dir))

        def sample_train_pair():
            return sample_pair_volumes(vols["pet_nac"], vols["pet_ac"], batch_size, args.latent_size, "train", args.val_fraction, rng, geo_aug=train_aug)

        def sample_val_pair():
            return sample_pair_volumes(vols["pet_nac"], vols["pet_ac"], batch_size, args.latent_size, "val", args.val_fraction, rng, geo_aug=None)
    else:
        if args.split_json:
            # REUSE an existing partition instead of deriving a new one. This is required
            # for any A/B: make_patient_split3 is deterministic in INDICES, but those index
            # the enumerated paired-patient list, whose ORDER is not stable across runs --
            # so two runs with the SAME seed and fractions can (and did) land on different
            # partitions, leaking 17/41 test patients into the other's training set.
            train_idx, val_idx, test_idx = _reuse_split(args.split_json, patients, logger)
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

        def _sample_sync(indices, augment, geo_aug=None):
            idx = int(indices[rng.randint(0, len(indices))])
            vols = cache.get(idx)
            return sample_pair_volumes_full(
                vols["pet_nac"], vols["pet_ac"], batch_size, args.latent_size, rng, augment=augment, geo_aug=geo_aug
            )

        # Train augments (random crop/flip/rotation + optional MONAI geo) so the small
        # paired pool yields diverse 3D examples; val stays clean for a stable metric.
        def sample_train_pair():
            if args.prefetch <= 0:
                return _sample_sync(train_idx, True, geo_aug=train_aug)
            _, vols = cache.next_train()
            return sample_pair_volumes_full(
                vols["pet_nac"], vols["pet_ac"], batch_size, args.latent_size, rng, augment=True, geo_aug=train_aug
            )

        def sample_val_pair():
            return _sample_sync(val_idx, False, geo_aug=None)

    # Latent normalization: scale RAW AE latents to ~unit std so the DDIM sampler's
    # N(0,1) start matches the latent distribution. A non-null config value pins the
    # scale (lets the user fix it / resume deterministically); otherwise compute it
    # once from a sample of AC latents. Stored in config -> embedded in the checkpoint
    # so inference/eval reuse the exact same scale.
    if is_flow:
        # The bridge interpolates between the AC and NAC latents directly; a constant
        # global scale cancels out of the trajectory direction, so unit-variance
        # normalization is unnecessary. Force 1.0 (a no-op in ae3d_encode/ae3d_decode) and
        # skip compute_latent_scale -- also avoids shifting the RNG sample sequence.
        latent_scale = 1.0
        logger.info("Flow mode: latent_scale forced to 1.0 (no-op for the bridge).")
    else:
        cfg_scale = config.get("latent_scale")
        if cfg_scale is not None and float(cfg_scale) > 0:
            latent_scale = float(cfg_scale)
            logger.info("Using configured latent_scale=%.6f", latent_scale)
        else:
            latent_scale = compute_latent_scale(ae, sample_train_pair, logger=logger)
    config["latent_scale"] = float(latent_scale)

    steps_per_epoch = int(args.steps_per_epoch)
    total_steps = int(args.epochs) * steps_per_epoch

    # Gradient accumulation: each global_step runs `grad_accum` micro-batches, each
    # with its own sampled pair + backward, then ONE optimizer step. So global_step
    # counts OPTIMIZER steps (lr schedule / val cadence / checkpoint step-counting are
    # unchanged) while the effective batch = batch_size * grad_accum. Default 1 is a
    # no-op (e.g. diff2d, which has no grad_accum_steps key, is unaffected).
    grad_accum = max(1, int(config.get("grad_accum_steps", 1)))
    if grad_accum > 1:
        logger.info("Gradient accumulation: %d micro-batches/step (effective batch=%d).",
                    grad_accum, batch_size * grad_accum)

    ema_decay = float(config.get("ema_decay", 0.0))
    # EMA tracks the raw (uncompiled) weights so its shadow keys stay prefix-free.
    ema = EMA(raw_model, decay=ema_decay) if ema_decay > 0 else None
    config.setdefault("lr_total_steps", total_steps)
    config.setdefault("lr_warmup_steps", min(500, max(1, total_steps // 10)))
    lr_scheduler = build_lr_scheduler(optimizer, config)

    best_val = float("inf")
    start_step = 0
    if resume_path is not None:
        start_step, best_val, resume_freeze_plan = _resume(
            raw_model, optimizer, rng, resume_path, logger, ema, prediction_type)
        # Fast-forward the LR schedule so the resumed LR matches an uninterrupted run.
        if lr_scheduler is not None:
            for _ in range(start_step):
                lr_scheduler.step()
        # Inflation is skipped on resume, so re-establish the freeze from the plan that
        # was embedded in the CHECKPOINT'S config (structural -> no 2D checkpoint needed).
        # Applied to raw_model before the loop; skipped entirely if already past the
        # unfreeze step. Everything here is a no-op unless inflate_freeze_backbone is set.
        if freeze_backbone:
            if resume_freeze_plan is None:
                logger.warning(
                    "inflate_freeze_backbone is set but the resume checkpoint has no "
                    "inflate_freeze_plan; continuing WITHOUT the freeze.")
            else:
                # Keep the plan embedded so subsequent checkpoints from this resumed run
                # retain it (the current run's config came from YAML, not the checkpoint).
                config["inflate_freeze_plan"] = resume_freeze_plan
                if unfreeze_step > 0 and start_step >= unfreeze_step:
                    logger.info(
                        "Resume at step=%d is past inflate_unfreeze_step=%d; backbone already "
                        "released, no freeze re-applied.", start_step, unfreeze_step)
                else:
                    center_freeze = CenterFreeze()
                    summary = center_freeze.apply(raw_model, resume_freeze_plan)
                    freeze_active = True
                    logger.info(
                        "Re-applied center-freeze on resume: masked=%d frozen=%d fresh=%d "
                        "trainable_tensors=%d; unfreeze at step=%s (resumed at step=%d).",
                        summary["n_masked"], summary["n_frozen"], summary["n_fresh"],
                        summary["n_trainable"],
                        unfreeze_step if unfreeze_step > 0 else "never", start_step)

    for global_step in range(start_step, total_steps):
        # Release the center-freeze at the configured optimizer step and fine-tune the
        # whole net (Make-A-Video / Video-LDM stage 2). No-op unless a freeze is active.
        if freeze_active and unfreeze_step > 0 and global_step == unfreeze_step:
            center_freeze.unfreeze()
            freeze_active = False
            logger.info("Unfroze backbone at step=%d; fine-tuning the full net.", global_step)
        epoch = global_step // max(1, steps_per_epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # Accumulate float metrics over the micro-batches and report their MEAN, so the
        # logged values reflect the full effective batch (not just the last micro-batch).
        loss_sum = mse_sum = pval_sum = qval_sum = 0.0
        for _ in range(grad_accum):
            nac_vol, ac_vol = sample_train_pair()
            # Both the target AC latent and the NAC conditioning latent live in the SAME
            # scaled latent space (NAC is channel-concatenated to the noisy AC).
            ac_lat = ae3d_encode(ae, ac_vol, scale=latent_scale)
            nac_lat = ae3d_encode(ae, nac_vol, scale=latent_scale)
            if is_flow:
                # Spatial re-weighting of the velocity MSE, built from the GT volumes and
                # never from the prediction (see quant_losses.build_latent_weight). Either
                # WHERE anatomy is ("occupancy", intensity-independent) or HOW HOT it is
                # ("intensity"); None when disabled, which keeps the plain mean.
                wmap = (None if latent_weight_fn is None
                        else latent_weight_fn(ac_vol, nac_vol, ac_lat.shape[2:]))
                mse_loss, x_t, timesteps, pred, _ = flow_loss(
                    model, schedule, ac_lat, nac_lat,
                    tau_dist=flow_tau_dist, tau_a=flow_tau_a,
                    loss_weighting=flow_loss_weighting, weight_map=wmap,
                )
            else:
                mse_loss, x_t, timesteps, pred, _ = diffusion_loss(model, schedule, ac_lat, nac_lat, snr_gamma)
            # Keyword args deliberately: the shared signature has `is_flow` before
            # `perceptual_active_frac`, so positional calls would silently mis-bind.
            pterm, pval = perceptual_term(
                perceptual, ae, schedule, x_t, timesteps, pred, ac_vol, perceptual_weight,
                latent_scale=latent_scale, is_flow=is_flow,
                perceptual_active_frac=perceptual_active_frac, ac_lat=ac_lat,
            )
            loss = mse_loss if pterm is None else mse_loss + pterm
            # Quantitative-fidelity term (under-dispersion / hot-tissue bias). Needs IMAGE
            # space, so it pays for a decode of the in-graph AC estimate -- unlike the LPL
            # perceptual term, which works on latents. Flow only: the epsilon one-step x0
            # estimate is meaningless at high noise, and these are statistics of the whole
            # image rather than a per-voxel distance.
            qval = 0.0
            if quant_loss is not None and is_flow:
                q_est = flow_x0_decode_in_graph(ae, schedule, x_t, timesteps, pred, latent_scale)
                qterm = quant_loss(q_est, ac_vol) * quant_weight
                loss = loss + qterm
                qval = float(qterm.detach().cpu())
            # Scale by 1/grad_accum so the summed gradients equal the mean over micro-batches.
            (loss / grad_accum).backward()
            loss_sum += float(loss.detach().cpu())
            mse_sum += float(mse_loss.detach().cpu())
            pval_sum += pval
            qval_sum += qval
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        if ema is not None:
            ema.update(raw_model)
        metrics = {"loss": loss_sum / grad_accum, "mse": mse_sum / grad_accum,
                   "perceptual": pval_sum / grad_accum, "quant": qval_sum / grad_accum}
        logger.info("step=%s loss=%.6f", global_step, metrics["loss"])
        metrics_writer.log("train", global_step, metrics, epoch)

        if args.val_every > 0 and global_step % args.val_every == 0:
            val_metrics = _validate(model, raw_model, ema, ae, schedule, sample_val_pair, args.val_batches, snr_gamma, latent_scale, is_flow, clamp_output)
            logger.info("VAL step=%s loss=%.6f l1=%.6f", global_step, val_metrics["loss"], val_metrics["l1"])
            metrics_writer.log("val", global_step, val_metrics, epoch)
            # Flow selects on the honest-rollout L1, not the (shortcut-prone) velocity loss.
            sel = _selection_metric(val_metrics, is_flow)
            is_best = sel < best_val
            best_val = min(best_val, sel)
            save_training_checkpoint(save_dir, _checkpoint_state(raw_model, optimizer, rng, config, global_step + 1, epoch, sel, best_val, ema), is_best)

    final_val = _validate(model, raw_model, ema, ae, schedule, sample_val_pair, args.val_batches, snr_gamma, latent_scale, is_flow, clamp_output)
    metrics_writer.log("val", total_steps, final_val, int(args.epochs))
    sel = _selection_metric(final_val, is_flow)
    is_best = sel < best_val
    best_val = min(best_val, sel)
    save_training_checkpoint(save_dir, _checkpoint_state(raw_model, optimizer, rng, config, total_steps, int(args.epochs), sel, best_val, ema), is_best)
    metrics_writer.close()
    if cache is not None:
        cache.close()
    logger.info("Training finished. best_val_loss=%.6f", best_val)


@torch.no_grad()
def _validate(model, raw_model, ema, ae, schedule, sample_val_pair, n_batches, snr_gamma=None,
              latent_scale=1.0, is_flow=False, clamp_output=True):
    # Evaluate under the EMA weights when available -- consistently higher quality.
    # The EMA swap targets raw_model (shared params); forward still runs via model.
    ctx = ema.average_parameters(raw_model) if ema is not None else _null_context(raw_model)
    with ctx:
        model.eval()
        accum = {"loss": 0.0, "l1": 0.0}
        n = max(1, n_batches)
        for _ in range(n):
            nac_vol, ac_vol = sample_val_pair()
            # Same scaled latent space as training (NAC + AC scaled identically).
            ac_lat = ae3d_encode(ae, ac_vol, scale=latent_scale)
            nac_lat = ae3d_encode(ae, nac_vol, scale=latent_scale)
            if is_flow:
                # Deliberately the UNSHAPED loss (uniform tau, no (1-tau) weighting): the
                # reported val `loss` must stay comparable across A/B arms that differ in
                # exactly those knobs. Model selection uses the rollout `l1` regardless.
                loss, _, _, _, _ = flow_loss(model, schedule, ac_lat, nac_lat)
                # HONEST generation metric: a real short rollout FROM the NAC latent
                # (not the old one-step proxy, which leaked the real AC via x_t and
                # masked the conditioning collapse). Only NAC information enters here.
                # No concat -- matches flow_loss (in_channels = C); NAC enters only as x_init.
                flow_model_fn = lambda x, t: model(to_input_memory_format(x), t)
                ac_pred_lat = schedule.flow_sample(flow_model_fn, nac_lat, num_steps=8, spacing="linear")
                recon = clamp_unit(ae3d_decode(ae, ac_pred_lat, scale=latent_scale), clamp_output)
            else:
                loss, x_t, timesteps, pred, noise = diffusion_loss(model, schedule, ac_lat, nac_lat, snr_gamma)
                # x0_pred is in scaled latent space; ae3d_decode divides by scale.
                acp = schedule.alphas_cumprod[timesteps]
                sqrt_acp = schedule._broadcast(torch.sqrt(acp), x_t)
                sqrt_one_minus = schedule._broadcast(torch.sqrt(1.0 - acp), x_t)
                x0_pred = (x_t - sqrt_one_minus * pred) / sqrt_acp
                recon = clamp_unit(ae3d_decode(ae, x0_pred, scale=latent_scale), clamp_output)
            accum["loss"] += float(loss.cpu())
            # ft3d is whole-volume (batch ~= 1), so compute l1 and the image-quality
            # metrics directly on the 3D volume (no per-slice loop -- that was a 2D-only
            # detail to match the per-slice diff2d eval path).
            accum["l1"] += float(torch.mean(torch.abs(recon - ac_vol)).cpu())
            # Image-quality metrics of the decoded 3D prediction vs AC reference.
            for k, v in image_quality_metrics(recon, ac_vol).items():
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


def _resume(model, optimizer, rng, resume_path, logger, ema=None, prediction_type="epsilon"):
    """Restore model/optimizer/RNG (and EMA) from a checkpoint.

    Returns ``(start_step, best_val, freeze_plan)`` where ``freeze_plan`` is the
    center-freeze plan embedded in the checkpoint's stored config (or ``None`` for
    checkpoints written without one), so the caller can re-establish the freeze.
    """
    state = load_checkpoint(resume_path)
    model.load_state_dict(state["model"])
    if state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if ema is not None and state.get("ema") is not None:
        ema.load_state_dict(state["ema"])
    restore_rng_state(state.get("rng"), rng)
    stored_best = float(state.get("best_val", float("inf")))
    ckpt_config = state.get("config") or {}
    ckpt_pt = str(ckpt_config.get("prediction_type", "epsilon")).lower()
    best_val = _resume_best_val(stored_best, ckpt_pt, prediction_type)
    if best_val != stored_best:
        logger.warning("Resume prediction_type mismatch (checkpoint=%s, config=%s): reset best_val "
                       "%.6f -> inf (selection-metric scale differs between modes).",
                       ckpt_pt, prediction_type, stored_best)
    start_step = int(state.get("step", 0))  # "step" = number of steps already completed
    freeze_plan = ckpt_config.get("inflate_freeze_plan")
    logger.info("Resumed from %s at step=%s best_val=%.6f", resume_path, start_step, best_val)
    return start_step, best_val, freeze_plan


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
