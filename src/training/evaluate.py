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
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from src.training.data import (
    filter_paired_patients,
    load_patient_by_path,
    make_patient_split3,
)
from src.training.dataset_index import enumerate_patients
from src.training.models.autoencoder2d import ae_decode, ae_encode, build_autoencoder_2d
from src.training.models.autoencoder3d import ae3d_decode, ae3d_encode, build_autoencoder_3d
from src.training.models.diffusion2d import build_diffusion_2d
from src.training.models.diffusion3d import build_diffusion_3d
from src.training.utils.image_metrics import image_quality_metrics
# Reuse the exact loaders/helpers inference uses, so eval and the GUI agree.
from src.training.infer import (
    _load_model,
    _resize_volume,
    _schedule_from_config,
    _slice_2d,
)

DEFAULT_SIZE = {"diff2d": 128, "ft3d": 64}
DEFAULT_AE = {"diff2d": "outputs/ae2d/best.pt", "ft3d": "outputs/ae3d/best.pt"}
# Image tier + clinical-tier proxies (all normalized-intensity space; see
# src/training/utils/image_metrics.py for the SUV-calibration caveat).
METRIC_KEYS = [
    "psnr", "ssim", "nrmse", "mae",
    "rel_bias", "max_rel_error", "voxel_r2",
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


def _cfg_model_fn(model, cond_lat, guidance):
    """Build the DDIM ``model_fn(x_t, t)`` with classifier-free guidance.

    With guidance ``w``: eps = eps_uncond + w*(eps_cond - eps_uncond), where the
    unconditional pass uses a zero (null) conditioning latent -- matching the
    cond-dropout null used in training. ``w==1.0`` is plain conditional sampling
    (single pass); ``w<=0`` is treated as 1.0.
    """
    g = float(guidance)
    cat = torch.cat
    if g == 1.0 or g <= 0:
        def model_fn(x_t, t):
            return model(cat([x_t, cond_lat], dim=1), t)
        return model_fn
    null_lat = torch.zeros_like(cond_lat)

    def model_fn(x_t, t):
        eps_c = model(cat([x_t, cond_lat], dim=1), t)
        eps_u = model(cat([x_t, null_lat], dim=1), t)
        return eps_u + g * (eps_c - eps_u)
    return model_fn


@torch.no_grad()
def _eval_diff2d(args, device, candidates):
    ae, _ = _load_model(args.ae_ckpt, build_autoencoder_2d, device, use_ema=args.use_ema)
    model, diff_config = _load_model(args.diff_ckpt, build_diffusion_2d, device, use_ema=args.use_ema)
    schedule = _schedule_from_config(diff_config, device)
    # Latent normalization scale from the diffusion config (default 1.0 keeps OLD
    # checkpoints, which have no latent_scale key, running unchanged).
    scale = float(diff_config.get("latent_scale", 1.0))

    per_patient = []
    for pi, patient in enumerate(candidates):
        if args.max_patients and len(per_patient) >= args.max_patients:
            break
        vols = load_patient_by_path(patient, device=device, load_ct=False, run_segmentation=False)
        if vols.get("pet_nac") is None or vols.get("pet_ac") is None:
            continue
        z = vols["pet_nac"].shape[0]
        slice_metrics = []
        for s in _slice_indices(z, args.num_slices):
            nac_img = _slice_2d(vols["pet_nac"], s, args.size)
            ac_img = _slice_2d(vols["pet_ac"], s, args.size)
            nac_lat = ae_encode(ae, nac_img, scale=scale)
            model_fn = _cfg_model_fn(model, nac_lat, args.guidance_scale)

            # x0 is sampled in scaled latent space; ae_decode divides by scale.
            x0 = schedule.ddim_sample(model_fn, nac_lat.shape, device, num_steps=args.ddim_steps, spacing=args.spacing, clip_x0=args.clip_x0)
            pred = ae_decode(ae, x0, scale=scale)
            slice_metrics.append(image_quality_metrics(pred, ac_img))
        if slice_metrics:
            # Per-slice mean, filtering non-finite (voxel_r2 / slope can be NaN on
            # flat slices); NaN only if every slice was non-finite for that key.
            agg = {}
            for k in METRIC_KEYS:
                vals = [m[k] for m in slice_metrics if math.isfinite(m[k])]
                agg[k] = float(np.mean(vals)) if vals else float("nan")
            agg["n_slices"] = len(slice_metrics)
            per_patient.append(agg)
            print(f"[diff2d] {len(per_patient)} eval'd (cand {pi+1}) "
                  f"ssim={agg['ssim']:.4f} psnr={agg['psnr']:.2f} nrmse={agg['nrmse']:.4f}")
    return per_patient


@torch.no_grad()
def _eval_ft3d(args, device, candidates):
    ae, _ = _load_model(args.ae_ckpt, build_autoencoder_3d, device, use_ema=args.use_ema)
    model, diff_config = _load_model(args.diff_ckpt, build_diffusion_3d, device, use_ema=args.use_ema)
    schedule = _schedule_from_config(diff_config, device)
    # Latent normalization scale from the diffusion config (default 1.0 keeps OLD
    # checkpoints, which have no latent_scale key, running unchanged).
    scale = float(diff_config.get("latent_scale", 1.0))

    per_patient = []
    for pi, patient in enumerate(candidates):
        if args.max_patients and len(per_patient) >= args.max_patients:
            break
        vols = load_patient_by_path(patient, device=device, load_ct=False, run_segmentation=False)
        if vols.get("pet_nac") is None or vols.get("pet_ac") is None:
            continue
        nac_vol = _resize_volume(vols["pet_nac"], args.size)
        ac_vol = _resize_volume(vols["pet_ac"], args.size)
        nac_lat = ae3d_encode(ae, nac_vol, scale=scale)
        model_fn = _cfg_model_fn(model, nac_lat, args.guidance_scale)

        # x0 is sampled in scaled latent space; ae3d_decode divides by scale.
        x0 = schedule.ddim_sample(model_fn, nac_lat.shape, device, num_steps=args.ddim_steps, spacing=args.spacing)
        pred = ae3d_decode(ae, x0, scale=scale)
        m = image_quality_metrics(pred, ac_vol)
        per_patient.append(m)
        print(f"[ft3d] {len(per_patient)} eval'd (cand {pi+1}) "
              f"ssim={m['ssim']:.4f} psnr={m['psnr']:.2f} nrmse={m['nrmse']:.4f}")
    return per_patient


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
    parser.add_argument("--ddim_steps", type=int, default=25)
    parser.add_argument("--spacing", choices=["linear", "karras"], default="karras")
    parser.add_argument("--clip_x0", type=float, default=4.0,
                        help="Static thresholding: clamp the per-step DDIM x0 estimate to "
                             "[-clip_x0, clip_x0]. Required for stable sampling under the cosine "
                             "schedule (terminal SNR ~0 otherwise explodes). 0/negative disables.")
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
    if args.size is None:
        args.size = DEFAULT_SIZE[args.task]
    if args.ae_ckpt is None:
        args.ae_ckpt = DEFAULT_AE[args.task]
    if args.diff_ckpt is None:
        args.diff_ckpt = os.path.join("outputs", args.task, "best.pt")
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
        "ddim_steps": args.ddim_steps,
        "spacing": args.spacing,
        "clip_x0": args.clip_x0,
        "guidance_scale": args.guidance_scale,
        "use_ema": args.use_ema,
        "ae_ckpt": args.ae_ckpt,
        "diff_ckpt": args.diff_ckpt,
        "note": ("normalized-intensity space (not calibrated SUV); rel_bias is a proxy for "
                 "SUV mean bias and max_rel_error a proxy for lesion SUVmax error; voxel_r2 / "
                 "reg_slope / Bland-Altman (ba_*) quantify voxel-wise agreement over foreground"),
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
