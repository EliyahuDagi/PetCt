"""Batch evaluation of trained NAC->AC diffusion models on the held-out val set.

Produces the SOTA-comparable image-quality metrics (SSIM / PSNR / NRMSE / MAE /
rel_bias) by running FULL DDIM inference (encode NAC -> sample -> decode) on every
held-out validation patient and comparing the generated AC PET against the real AC
PET. This is the number you would report -- unlike the training-time ``l1``, which
is only a one-step x0 proxy on a noisy latent.

The held-out split is reconstructed to MATCH training. There are three modes:
  --mode all  : every enumerated patient (lazy pairing skip in the loop).
  --mode val  : the by-patient VALIDATION split.
  --mode test : the by-patient TEST split -- patients held out from BOTH train and
                val during training (the number you report for an unbiased estimate).

For 'val'/'test' the split is reconstructed as:
  enumerate_patients(roots) -> filter_paired_patients (diffusion needs NAC+AC)
  -> [prefer outputs/<task>/split.json if present] else
     make_patient_split3(n, val_fraction, test_fraction, seed).
split.json (written by the train scripts) stores the exact patient PATHS per
train/val/test, so eval reloads the same set even if the dataset changed since
training. When falling back to fractions you MUST pass the same ``--val_fraction``,
``--test_fraction`` and ``--seed`` the training run used (GUI default was
val_fraction=0.1, seed=42; test_fraction default 0.1).

Metrics are computed in the pipeline's normalized intensity space (not calibrated
SUV). ``rel_bias`` is a normalized-intensity proxy for SUV mean bias; a true SUV
%-bias number still requires calibrated SUV units + organ/lesion VOIs.

Run in WSL/GPU (where training ran), from the project root:

  # diff2d (2D latent diffusion) -- evaluate ~32 slices/patient over all val patients
  python -m src.training.evaluate --task diff2d \
      --data_dir "/mnt/d/DeepTrainingData/Project/Bladder 13.11.25" \
                 "/mnt/d/DeepTrainingData/Project/PetCtNormal/PET_CT_NORMAL_2" \
                 "/mnt/d/DeepTrainingData/Project/ACRIN 6668" \
      --val_fraction 0.1 --seed 42 --size 128 --num_slices 32 --ddim_steps 25 \
      --device cuda --out outputs/eval/diff2d/metrics.json

  # ft3d (3D latent diffusion) -- one whole-volume prediction per val patient
  python -m src.training.evaluate --task ft3d \
      --data_dir "/mnt/d/.../Bladder 13.11.25" "/mnt/d/.../PET_CT_NORMAL_2" "/mnt/d/.../ACRIN 6668" \
      --val_fraction 0.1 --seed 42 --size 64 --ddim_steps 25 \
      --device cuda --out outputs/eval/ft3d/metrics.json

IMPORTANT: ``--size`` must match the input size used when that stage was trained
(the AE/diffusion latents depend on it). Defaults mirror src/training/infer.py
(diff2d=128, ft3d=64); override if your training used a different size.

Slab-mode ft3d checkpoints (config ``slab_depth > 0``) are handled automatically:
the volume is resized in-plane only (to the recorded ``slab_inplane_size`` when
``--size`` is absent), every native slice is kept, the frozen 2D AE encodes slice by
slice, and the UNet runs on overlapping depth windows. The fair 2D control for that
is ``--task diff2d --volumetric``: every native slice is translated by the 2D model
and the metrics are computed once on the whole stacked volume.
"""

import argparse
import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import torch

from src.training.data import (
    filter_paired_patients,
    load_patient_by_path,
    make_patient_split3,
    resize_pair_native_depth,
)
from src.training.dataset_index import enumerate_patients
from src.training.models.autoencoder2d import ae_decode, ae_encode, build_autoencoder_2d
from src.training.models.autoencoder3d import ae3d_decode, ae3d_encode, build_autoencoder_3d
from src.training.models.diffusion2d import build_diffusion_2d
from src.training.models.diffusion3d import build_diffusion_3d
from src.training.utils.checkpointing import load_checkpoint
from src.training.utils.image_metrics import clamp_unit, image_quality_metrics
# Reuse the exact loaders/helpers inference uses, so eval and the GUI agree.
from src.training.infer import (
    DEFAULT_AE_SLICEWISE,
    _load_model,
    _resize_volume,
    _schedule_from_config,
    _slice_2d,
)
from src.training.utils.translate import sample_nac_to_ac

DEFAULT_SIZE = {"diff2d": 128, "ft3d": 64}
DEFAULT_AE = {"diff2d": "outputs/ae2d/best.pt", "ft3d": "outputs/ae3d/best.pt"}
# Image tier + clinical-tier proxies (all normalized-intensity space; see
# src/training/utils/image_metrics.py for the SUV-calibration caveat).
METRIC_KEYS = [
    "psnr", "ssim", "nrmse", "mae",
    "rel_bias", "max_rel_error", "p95_rel_error", "hot_band_rel_error", "voxel_r2",
    "reg_slope", "reg_intercept",
    "ba_mean_bias", "ba_loa_lower", "ba_loa_upper",
]


def _load_split_json(split_json):
    """Load a persisted split.json (written by the train scripts), or None.

    Returns the parsed dict (with ``train``/``val``/``test`` patient-path lists) or
    None if the file is absent/unreadable. See ``data.write_split_json`` for schema.
    """
    if not split_json or not os.path.exists(split_json):
        return None
    try:
        with open(split_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not all(k in data for k in ("train", "val", "test")):
        return None
    return data


def _resolve_candidates(roots, mode, val_fraction, test_fraction, seed, split_json=None):
    """Return ``(candidates, n_reference)`` -- the ordered patients to evaluate.

    mode 'val'  : reconstruct the training by-patient VALIDATION split (paired only).
    mode 'test' : reconstruct the by-patient TEST split (paired only) -- patients
                  held out from BOTH train and val during training.
    mode 'all'  : every enumerated patient (unfiltered); the eval loop lazily skips
                  the unpaired ones and stops at --max_patients, so the expensive
                  pairing scan only touches what it needs.

    For 'val'/'test', a persisted ``split.json`` (``--split_json`` / default
    ``outputs/<task>/split.json``) is preferred: it stores the exact patient PATHS,
    so the held-out set is reproduced even if the dataset changed since training.
    When no split.json is found, fall back to recomputing the by-patient 3-way split
    from ``(val_fraction, test_fraction, seed)`` -- only valid if the dataset is
    unchanged since training (otherwise the indices no longer line up).
    """
    patients = enumerate_patients(roots, missing_ok=True)
    if len(patients) <= 1:
        raise ValueError("Found <=1 enumerated patient. Check --data_dir roots.")
    if mode == "all":
        return patients, len(patients)

    # Prefer a persisted split.json (exact paths) for val/test.
    split = _load_split_json(split_json)
    if split is not None:
        wanted = split.get(mode, [])
        present = set(str(p) for p in patients)
        candidates = [p for p in wanted if str(p) in present]
        missing = len(wanted) - len(candidates)
        print(f"Loaded {mode} split from {split_json}: {len(wanted)} patients "
              f"({len(candidates)} present, {missing} missing).")
        if not candidates:
            raise ValueError(
                f"split.json '{mode}' set has no patients present under the given "
                f"--data_dir roots (dataset moved/changed?).")
        return candidates, len(wanted)

    paired = filter_paired_patients(patients, log=print)
    if not paired:
        raise ValueError("No paired NAC+AC patients found across the given roots.")
    train_idx, val_idx, test_idx = make_patient_split3(
        len(paired), val_fraction, test_fraction, seed)
    pick = val_idx if mode == "val" else test_idx
    if not pick:
        raise ValueError(
            f"The by-patient {mode} split is empty for n={len(paired)} paired "
            f"patients (too few for a {mode} holdout). Add more paired patients or "
            f"adjust fractions.")
    return [paired[i] for i in pick], len(paired)


def _subsample(candidates, max_patients):
    """Evenly subsample candidates (deterministic) when a cap is set."""
    if not max_patients or max_patients >= len(candidates):
        return candidates
    idx = [int(round(x)) for x in np.linspace(0, len(candidates) - 1, num=max_patients)]
    return [candidates[i] for i in sorted(set(idx))]


def _slice_indices(z, num_slices):
    """Evenly spaced slice indices across the depth (inclusive of interior)."""
    if num_slices >= z:
        return list(range(z))
    # Skip the extreme first/last (often empty) by sampling the central span.
    return [int(round(x)) for x in np.linspace(0.05 * z, 0.95 * z - 1, num=num_slices)]


@torch.no_grad()
def _eval_diff2d(args, device, candidates):
    ae, _ = _load_model(args.ae_ckpt, build_autoencoder_2d, device, use_ema=args.use_ema)
    model, diff_config = _load_model(args.diff_ckpt, build_diffusion_2d, device, use_ema=args.use_ema)
    schedule = _schedule_from_config(diff_config, device)
    # Latent normalization scale from the diffusion config (default 1.0 keeps OLD
    # checkpoints, which have no latent_scale key, running unchanged).
    if "latent_scale" not in diff_config:
        warnings.warn(
            "latent_scale not found in the diffusion config; assuming identity "
            "scaling (1.0). This is correct only for OLD checkpoints saved before "
            "latent_scale was recorded."
        )
    scale = float(diff_config.get("latent_scale", 1.0))

    per_patient = []
    for pi, patient in enumerate(candidates):
        if args.max_patients and len(per_patient) >= args.max_patients:
            break
        vols = load_patient_by_path(patient, device=device, load_ct=False, run_segmentation=False)
        if vols.get("pet_nac") is None or vols.get("pet_ac") is None:
            continue
        if getattr(args, "volumetric", False):
            # Fair control for slab-mode ft3d: every native slice, one whole-volume score.
            m = _diff2d_volumetric(ae, model, schedule, diff_config, scale, vols, args)
            m["patient"] = str(patient)
            per_patient.append(m)
            print(f"[diff2d/volumetric] {len(per_patient)} eval'd (cand {pi+1}) "
                  f"ssim={m['ssim']:.4f} psnr={m['psnr']:.2f} nrmse={m['nrmse']:.4f}")
            continue
        z = vols["pet_nac"].shape[0]
        slice_metrics = []
        for s in _slice_indices(z, args.num_slices):
            nac_img = _slice_2d(vols["pet_nac"], s, args.size)
            ac_img = _slice_2d(vols["pet_ac"], s, args.size)
            nac_lat = ae_encode(ae, nac_img, scale=scale)

            # Shared flow-vs-epsilon decision (see src/training/utils/translate.py).
            # x0 is sampled in scaled latent space; ae_decode divides by scale.
            x0 = sample_nac_to_ac(
                model, schedule, nac_lat, diff_config,
                num_steps=args.ddim_steps, spacing=args.spacing,
                guidance_scale=args.guidance_scale, clip_x0=args.clip_x0, tag="diff2d",
            )
            pred = clamp_unit(ae_decode(ae, x0, scale=scale), args.clamp_output)
            slice_metrics.append(image_quality_metrics(pred, ac_img))
        if slice_metrics:
            # Per-slice mean, filtering non-finite (voxel_r2 / slope can be NaN on
            # flat slices); NaN only if every slice was non-finite for that key.
            agg = {}
            for k in METRIC_KEYS:
                vals = [m[k] for m in slice_metrics if math.isfinite(m[k])]
                agg[k] = float(np.mean(vals)) if vals else float("nan")
            agg["n_slices"] = len(slice_metrics)
            # See the ft3d branch: identifies the patient so runs can be compared PAIRED.
            agg["patient"] = str(patient)
            per_patient.append(agg)
            print(f"[diff2d] {len(per_patient)} eval'd (cand {pi+1}) "
                  f"ssim={agg['ssim']:.4f} psnr={agg['psnr']:.2f} nrmse={agg['nrmse']:.4f}")
    return per_patient


@torch.no_grad()
def _diff2d_volumetric(ae, model, schedule, diff_config, scale, vols, args, chunk=64):
    """Score a 2D model on a WHOLE volume: every native slice, one metric on the 3D stack.

    Fair control for slab-mode ft3d: the pair is resized in-plane to ``args.size`` with
    the native slice count kept (the geometry slab mode trains on), every slice is
    translated by the 2D model in chunks of ``chunk`` slices folded into the batch
    axis, and the metrics are computed once on the stacked ``(1, 1, Z, size, size)``
    prediction against the whole AC volume.
    """
    nac_r, ac_r = resize_pair_native_depth(vols["pet_nac"], vols["pet_ac"], args.size)  # (1,Z,S,S)
    nac_slices = nac_r.permute(1, 0, 2, 3).contiguous()  # (Z,1,S,S): depth folded into the batch
    preds = []
    for s0 in range(0, int(nac_slices.shape[0]), int(chunk)):
        nac_lat = ae_encode(ae, nac_slices[s0:s0 + chunk], scale=scale)
        x0 = sample_nac_to_ac(
            model, schedule, nac_lat, diff_config,
            num_steps=args.ddim_steps, spacing=args.spacing,
            guidance_scale=args.guidance_scale, clip_x0=args.clip_x0, tag="diff2d",
        )
        preds.append(clamp_unit(ae_decode(ae, x0, scale=scale), args.clamp_output))
    pred_vol = torch.cat(preds, dim=0).permute(1, 0, 2, 3).unsqueeze(0)  # (1,1,Z,S,S)
    return image_quality_metrics(pred_vol, ac_r.unsqueeze(0))


@torch.no_grad()
def _eval_ft3d(args, device, candidates):
    # The diffusion checkpoint's config says which autoencoder and which geometry it
    # was trained with (cube vs slab), so it is loaded first.
    model, diff_config = _load_model(args.diff_ckpt, build_diffusion_3d, device, use_ema=args.use_ema)
    schedule = _schedule_from_config(diff_config, device)
    ae_mode = str(diff_config.get("ae_mode", "3d") or "3d").lower()
    slab_depth = int(diff_config.get("slab_depth", 0) or 0)
    is_flow = str(diff_config.get("prediction_type", "epsilon")).lower() == "flow"
    if ae_mode == "2d_slicewise":
        # Slab checkpoints: the frozen 2D AE applied slice by slice (latent depth ==
        # image depth). Imported here so the cube path does not depend on it.
        from src.training.models.autoencoder2d import SliceWiseAutoencoder
        ae2d, _ = _load_model(args.ae_ckpt, build_autoencoder_2d, device, use_ema=args.use_ema)
        ae = SliceWiseAutoencoder(ae2d)
    else:
        # The 3D AE from train_ae3d: the cube one (depth compressed) for cube checkpoints,
        # or an in-plane-only one (every slice kept) for slab checkpoints trained on it.
        ae, _ = _load_model(args.ae_ckpt, build_autoencoder_3d, device, use_ema=args.use_ema)
        if slab_depth > 0:
            # A slab checkpoint keeps every native slice, so its autoencoder must too.
            # Read from the BUILT model, not from a config that may not match it.
            from src.training.models.anisotropic import is_inplane_only
            if not is_inplane_only(ae):
                raise ValueError(
                    "This is a slab-mode checkpoint (slab_depth=%d), so the autoencoder must "
                    "keep one latent slice per image slice, but %r builds a depth-compressing "
                    "3D autoencoder. Pass an in-plane-only 3D autoencoder (trained with "
                    "model.anisotropic: true) or the 2D autoencoder used slice by slice."
                    % (slab_depth, args.ae_ckpt))
    # Latent normalization scale from the diffusion config (default 1.0 keeps OLD
    # checkpoints, which have no latent_scale key, running unchanged).
    if "latent_scale" not in diff_config:
        warnings.warn(
            "latent_scale not found in the diffusion config; assuming identity "
            "scaling (1.0). This is correct only for OLD checkpoints saved before "
            "latent_scale was recorded."
        )
    scale = float(diff_config.get("latent_scale", 1.0))
    model_fn_wrapper = None
    if slab_depth > 0:
        if not is_flow:
            raise NotImplementedError(
                "Slab-mode evaluation is implemented for prediction_type=flow only.")
        # Slab mode: in-plane resize only, every native slice kept; the UNet runs on
        # overlapping depth windows whose velocities are blended (as in training val).
        size = int(args.size or diff_config.get("slab_inplane_size", 128))
        slab_window = int(diff_config.get("slab_window") or 2 * slab_depth)
        slab_stride = int(diff_config.get("slab_stride") or slab_depth)
        from src.training.utils.sliding import depth_windowed_model_fn
        model_fn_wrapper = lambda f: depth_windowed_model_fn(f, slab_window, slab_stride)
        print(f"[ft3d] slab-mode checkpoint: in-plane {size}, native depth, depth windows of "
              f"{slab_window} slices, stride {slab_stride}")
    else:
        size = int(args.size or DEFAULT_SIZE["ft3d"])

    per_patient = []
    for pi, patient in enumerate(candidates):
        if args.max_patients and len(per_patient) >= args.max_patients:
            break
        vols = load_patient_by_path(patient, device=device, load_ct=False, run_segmentation=False)
        if vols.get("pet_nac") is None or vols.get("pet_ac") is None:
            continue
        if slab_depth > 0:
            nac_r, ac_r = resize_pair_native_depth(vols["pet_nac"], vols["pet_ac"], size)
            nac_vol, ac_vol = nac_r.unsqueeze(0), ac_r.unsqueeze(0)  # (1,1,Z,size,size)
        else:
            nac_vol = _resize_volume(vols["pet_nac"], size)
            ac_vol = _resize_volume(vols["pet_ac"], size)
        nac_lat = ae3d_encode(ae, nac_vol, scale=scale)

        # Shared flow-vs-epsilon decision (see src/training/utils/translate.py).
        # x0 is sampled in scaled latent space; ae3d_decode divides by scale.
        # NOTE: this ft3d eval path historically passed NO clip_x0 to ddim_sample
        # (unlike infer.run_ft3d which uses 4.0); clip_x0=None preserves that.
        x0 = sample_nac_to_ac(
            model, schedule, nac_lat, diff_config,
            num_steps=args.ddim_steps, spacing=args.spacing,
            guidance_scale=args.guidance_scale, clip_x0=None, tag="ft3d",
            model_fn_wrapper=model_fn_wrapper,
        )
        pred = clamp_unit(ae3d_decode(ae, x0, scale=scale), args.clamp_output)
        m = image_quality_metrics(pred, ac_vol)
        # Record WHICH patient, so two eval runs can be compared PAIRED (same patient,
        # same AE, same split) instead of by mean. The expected effect of a loss-shaping
        # arm is ~1%, far below the between-patient SD, so paired is the only sensitive
        # test -- see scripts/_cmp_eval_paired.py. Non-numeric, and _aggregate iterates
        # METRIC_KEYS only, so this cannot leak into the summary.
        m["patient"] = str(patient)
        per_patient.append(m)
        print(f"[ft3d] {len(per_patient)} eval'd (cand {pi+1}) "
              f"ssim={m['ssim']:.4f} psnr={m['psnr']:.2f} nrmse={m['nrmse']:.4f}")
    return per_patient


def _peek_diff_config(diff_ckpt):
    """Return the config embedded in a diffusion checkpoint, or ``{}`` if it cannot be read.

    Only used to pick defaults before the real load; a missing or broken checkpoint
    still fails later with the normal error from ``_load_model``.
    """
    try:
        state = load_checkpoint(diff_ckpt)
    except Exception:
        return {}
    cfg = state.get("config", {}) if isinstance(state, dict) else {}
    return cfg if isinstance(cfg, dict) else {}


def _aggregate(per_patient):
    """mean +/- SD across patients for each metric (filtering non-finite)."""
    out = {}
    for k in METRIC_KEYS:
        vals = [p[k] for p in per_patient if k in p and math.isfinite(p[k])]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=["diff2d", "ft3d"])
    parser.add_argument("--data_dir", required=True, nargs="+", help="Same roots used for training (space-separated)")
    parser.add_argument("--ae_ckpt", default=None, help="AE checkpoint (defaults: ae2d for diff2d, ae3d for ft3d)")
    parser.add_argument("--diff_ckpt", default=None, help="Diffusion checkpoint (default outputs/<task>/best.pt)")
    parser.add_argument("--mode", choices=["val", "test", "all"], default="all",
                        help="'test' reproduces the by-patient TEST split (held out from train AND val -- "
                             "the number to report); 'val' reproduces the validation split; both prefer "
                             "split.json and otherwise recompute from fractions+seed (only valid if the "
                             "dataset is unchanged since training). 'all' samples paired patients from the "
                             "current dataset (default).")
    parser.add_argument("--split_json", default=None,
                        help="Persisted split (default outputs/<task>/split.json). For 'val'/'test', exact "
                             "patient paths are reloaded from here when present; otherwise fall back to "
                             "recomputing from --val_fraction/--test_fraction/--seed.")
    parser.add_argument("--max_patients", type=int, default=20, help="Cap on patients evaluated (0 = no cap)")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="'val'/'test' fallback only: MUST match training (GUI default 0.1)")
    parser.add_argument("--test_fraction", type=float, default=0.1, help="'val'/'test' fallback only: MUST match training (default 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="'val'/'test' fallback only: MUST match training (default 42)")
    parser.add_argument("--size", type=int, default=None, help="Input size; MUST match training (defaults diff2d=128, ft3d=64)")
    parser.add_argument("--num_slices", type=int, default=32, help="diff2d only: slices evaluated per patient")
    parser.add_argument("--volumetric", action="store_true",
                        help="diff2d only: translate EVERY native slice of each patient (resized "
                             "in-plane to --size, depth kept) and score the stacked 3D volume once, "
                             "instead of averaging per-slice metrics over --num_slices slices. The "
                             "fair control for slab-mode ft3d (same geometry, same metric).")
    parser.add_argument("--ddim_steps", type=int, default=25)
    parser.add_argument("--spacing", choices=["linear", "karras"], default="karras")
    parser.add_argument("--clip_x0", type=float, default=4.0,
                        help="Static thresholding: clamp the per-step DDIM x0 estimate to "
                             "[-clip_x0, clip_x0]. Required for stable sampling under the cosine "
                             "schedule (terminal SNR ~0 otherwise explodes). 0/negative disables.")
    parser.add_argument("--clamp_output", dest="clamp_output", action="store_true", default=True,
                        help="Clamp the decoded prediction to the valid [0,1] intensity range "
                             "(default). normalize_volume clips every GT volume to [0,1], so a "
                             "prediction outside it is invalid; the AE decoder's linear output "
                             "conv overshoots at hot-spot edges.")
    parser.add_argument("--no_clamp_output", dest="clamp_output", action="store_false",
                        help="Disable the [0,1] clamp -- reproduces pre-clamp numbers.")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="Classifier-free guidance scale w: eps = eps_uncond + w*(eps_cond-"
                             "eps_uncond), uncond pass uses a zero NAC latent. 1.0 = plain "
                             "conditional (single pass). Only meaningful for cond-dropout-trained "
                             "checkpoints. Typical 1.5-3.0.")
    parser.add_argument("--use_ema", dest="use_ema", action="store_true", default=True, help="Use EMA weights if present (default)")
    parser.add_argument("--no_ema", dest="use_ema", action="store_false")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None, help="Output JSON path (default outputs/eval/<task>/metrics.json)")
    args = parser.parse_args()

    if args.clip_x0 is not None and args.clip_x0 <= 0:
        args.clip_x0 = None  # explicit opt-out
    if args.diff_ckpt is None:
        args.diff_ckpt = os.path.join("outputs", args.task, "best.pt")
    # Slab-mode ft3d checkpoints record their own geometry and autoencoder kind. Use
    # them when --size / --ae_ckpt are absent: the cube defaults (64, the 3D AE) would
    # be wrong for such a checkpoint.
    diff_cfg = _peek_diff_config(args.diff_ckpt) if args.task == "ft3d" else {}
    is_slab_ckpt = int(diff_cfg.get("slab_depth", 0) or 0) > 0
    if args.size is None:
        if is_slab_ckpt:
            args.size = int(diff_cfg.get("slab_inplane_size", 128))
            print(f"slab-mode checkpoint: --size not given, using its recorded in-plane size {args.size} "
                  f"(depth stays native)")
        else:
            args.size = DEFAULT_SIZE[args.task]
    if args.ae_ckpt is None:
        ckpt_ae_mode = str(diff_cfg.get("ae_mode", "3d") or "3d").lower()
        if is_slab_ckpt and ckpt_ae_mode == "2d_slicewise":
            args.ae_ckpt = DEFAULT_AE_SLICEWISE
            print(f"slab-mode checkpoint: --ae_ckpt not given, using the 2D AE default {args.ae_ckpt}")
        else:
            args.ae_ckpt = DEFAULT_AE[args.task]
            if is_slab_ckpt:
                # A slab checkpoint on a 3D autoencoder needs the in-plane-only one, and
                # this default path is where the cube autoencoder lives by convention.
                print(f"slab-mode checkpoint with ae_mode {ckpt_ae_mode}: --ae_ckpt not given, "
                      f"falling back to {args.ae_ckpt}. It has to be an in-plane-only 3D "
                      f"autoencoder (model.anisotropic: true); a depth-compressing one is refused.")
    if args.out is None:
        args.out = os.path.join("outputs", "eval", args.task, "metrics.json")
    if args.split_json is None:
        args.split_json = os.path.join("outputs", args.task, "split.json")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device} task={args.task} mode={args.mode} size={args.size} "
          f"ddim_steps={args.ddim_steps} max_patients={args.max_patients}")

    candidates, n_total = _resolve_candidates(
        args.data_dir, args.mode, args.val_fraction, args.test_fraction, args.seed,
        split_json=args.split_json)
    if args.mode == "all":
        # Spread the sample across the cohort, oversampling so the lazy pairing
        # skip in the eval loop still reaches --max_patients paired studies.
        candidates = _subsample(candidates, (args.max_patients or 0) * 3)
    ref_label = "enumerated" if args.mode == "all" else f"{args.mode}-set"
    print(f"{n_total} {ref_label} -> {len(candidates)} candidates (cap {args.max_patients})")

    runner = _eval_diff2d if args.task == "diff2d" else _eval_ft3d
    per_patient = runner(args, device, candidates)
    if not per_patient:
        raise SystemExit("No patients evaluated (all skipped?). Check pairing / roots.")

    summary = _aggregate(per_patient)
    result = {
        "task": args.task,
        "mode": args.mode,
        "n_patients_evaluated": len(per_patient),
        "n_total_enumerated": n_total,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "split_json": args.split_json if os.path.exists(args.split_json or "") else None,
        "size": args.size,
        "volumetric": bool(getattr(args, "volumetric", False)),
        "ddim_steps": args.ddim_steps,
        "spacing": args.spacing,
        "clip_x0": args.clip_x0,
        "clamp_output": args.clamp_output,
        "guidance_scale": args.guidance_scale,
        "use_ema": args.use_ema,
        "ae_ckpt": args.ae_ckpt,
        "diff_ckpt": args.diff_ckpt,
        "note": ("normalized-intensity space (not calibrated SUV); rel_bias is a proxy for "
                 "SUV mean bias and max_rel_error a proxy for lesion SUVmax error; voxel_r2 / "
                 "reg_slope / Bland-Altman (ba_*) quantify voxel-wise agreement over foreground. "
                 "max_rel_error is a SINGLE-VOXEL statistic dominated by isolated decoder "
                 "overshoot, and is ~0 by construction once the prediction is clamped -- "
                 "prefer p95_rel_error / hot_band_rel_error. NOTE normalize_volume clips at "
                 "p99, saturating ~3% of foreground at exactly 1.0, so true lesion-SUVmax "
                 "fidelity is NOT measurable without a non-clipping normalization"),
        "summary": summary,
        "per_patient": per_patient,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== {args.mode}-set summary (mean +/- SD over patients) ===")
    for k in METRIC_KEYS:
        if k in summary:
            print(f"  {k:9s} {summary[k]['mean']:.4f} +/- {summary[k]['std']:.4f}  (n={summary[k]['n']})")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
