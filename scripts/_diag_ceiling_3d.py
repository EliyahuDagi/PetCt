"""Diagnostic: where is the 3D NAC->AC headroom -- in the AE or in the flow model?

Reports the FULL image-quality metric panel, over the same held-out TEST patients the
flow model is scored on, for four references:

  1. raw NAC vs AC                  -- the do-nothing floor.
  2. decode(encode(NAC)) vs AC      -- PASSTHROUGH: the AE round-trips NAC and pretends
                                       it is AC. Anything the flow model earns must beat
                                       this, or it is contributing nothing.
  3. decode(encode(AC))  vs AC      -- the AE CEILING: the best any latent model can do,
                                       because the flow model only ever produces a latent
                                       that is then decoded by this same frozen AE.
  4. flow prediction     vs AC      -- the actual model, via the shared sampling path.

Rows 2-4 are reported both UNCLAMPED and clamped to the valid [0,1] range (normalize_volume
clips every GT volume to [0,1], so a prediction outside it is invalid by construction).

Why this exists: `outputs/ae3d_p/metrics.jsonl` shows the frozen 3D AE reconstructing the
GROUND-TRUTH AC with max_rel_error 0.824, essentially identical to ft3d_flow_p's 0.836 --
i.e. that metric is AE-limited and no change to the flow model can move it. This script
makes the per-metric split explicit so effort goes where headroom actually exists.
It supersedes _diag_eval_ceiling.py, which is 2D / ae2d / SSIM-only and hardcodes the old
diff2d_flow split.

Usage (WSL, GPU):
  python scripts/_diag_ceiling_3d.py \
      --data_dir "/mnt/d/.../ACRIN 6668" [more roots ...] \
      --ae_ckpt outputs/ae3d_p/best.pt \
      --diff_ckpt outputs/ft3d_flow_p/best.pt \
      --split_json outputs/ft3d_flow_p/split.json \
      --mode test --size 64 --ddim_steps 16 --max_patients 0
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.training.evaluate import _resolve_candidates
from src.training.data import load_patient_by_path
from src.training.infer import _load_model, _resize_volume, _schedule_from_config
from src.training.models.autoencoder3d import (
    ae3d_decode,
    ae3d_encode,
    build_autoencoder_3d,
)
from src.training.models.diffusion3d import build_diffusion_3d
from src.training.utils.image_metrics import clamp_unit, image_quality_metrics
from src.training.utils.translate import sample_nac_to_ac

# Ordered for the report; these are the ones that decide where to spend effort.
KEYS = ["ssim", "psnr", "nrmse", "mae", "voxel_r2", "reg_slope", "reg_intercept",
        "rel_bias", "max_rel_error", "p95_rel_error", "hot_band_rel_error"]


def _aggregate(rows):
    """Mean over patients per key, ignoring non-finite values (as evaluate.py does)."""
    out = {}
    for k in KEYS:
        vals = [r[k] for r in rows if k in r and np.isfinite(r[k])]
        out[k] = (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, nargs="+")
    p.add_argument("--ae_ckpt", default="outputs/ae3d_p/best.pt")
    p.add_argument("--diff_ckpt", default="outputs/ft3d_flow_p/best.pt")
    p.add_argument("--split_json", default="outputs/ft3d_flow_p/split.json")
    p.add_argument("--mode", choices=["val", "test", "all"], default="test")
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--test_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--ddim_steps", type=int, default=16)
    p.add_argument("--spacing", choices=["linear", "karras"], default="linear")
    p.add_argument("--max_patients", type=int, default=0, help="0 = all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip_flow", action="store_true",
                   help="Only the AE ceiling / passthrough rows (no diffusion ckpt needed)")
    p.add_argument("--out", default="outputs/report/ceiling_3d.json")
    args = p.parse_args()

    device = torch.device(args.device)
    ae, _ = _load_model(args.ae_ckpt, build_autoencoder_3d, device, use_ema=True)
    model = diff_config = schedule = None
    if not args.skip_flow:
        model, diff_config = _load_model(args.diff_ckpt, build_diffusion_3d, device, use_ema=True)
        schedule = _schedule_from_config(diff_config, device)
        scale = float(diff_config.get("latent_scale", 1.0))
        print(f"flow ckpt: prediction_type={diff_config.get('prediction_type')} latent_scale={scale}")
    else:
        scale = 1.0

    candidates, n_ref = _resolve_candidates(
        args.data_dir, args.mode, args.val_fraction, args.test_fraction, args.seed,
        split_json=args.split_json,
    )
    if args.max_patients and args.max_patients > 0:
        candidates = candidates[: args.max_patients]
    print(f"{len(candidates)} patients ({args.mode} split of {n_ref})")

    rows = {k: [] for k in ("nac_raw", "passthrough", "passthrough_c",
                            "ae_ceiling", "ae_ceiling_c", "flow", "flow_c")}
    for i, path in enumerate(candidates, 1):
        vols = load_patient_by_path(path, device=device, load_ct=False, run_segmentation=False)
        if vols.get("pet_nac") is None or vols.get("pet_ac") is None:
            print(f"  [{i}] skip (unpaired): {Path(path).name}")
            continue
        nac = _resize_volume(vols["pet_nac"], args.size)
        ac = _resize_volume(vols["pet_ac"], args.size)

        with torch.no_grad():
            nac_rt = ae3d_decode(ae, ae3d_encode(ae, nac))   # AE round-trip of NAC
            ac_rt = ae3d_decode(ae, ae3d_encode(ae, ac))     # AE round-trip of AC = ceiling

        rows["nac_raw"].append(image_quality_metrics(nac, ac))
        rows["passthrough"].append(image_quality_metrics(nac_rt, ac))
        rows["passthrough_c"].append(image_quality_metrics(clamp_unit(nac_rt), ac))
        rows["ae_ceiling"].append(image_quality_metrics(ac_rt, ac))
        rows["ae_ceiling_c"].append(image_quality_metrics(clamp_unit(ac_rt), ac))

        if model is not None:
            with torch.no_grad():
                nac_lat = ae3d_encode(ae, nac, scale=scale)
                x0 = sample_nac_to_ac(
                    model, schedule, nac_lat, diff_config,
                    num_steps=args.ddim_steps, spacing=args.spacing,
                    guidance_scale=1.0, clip_x0=None, tag="ceiling3d",
                )
                pred = ae3d_decode(ae, x0, scale=scale)
            rows["flow"].append(image_quality_metrics(pred, ac))
            rows["flow_c"].append(image_quality_metrics(clamp_unit(pred), ac))
        print(f"  [{i}/{len(candidates)}] {Path(path).name[:44]}", flush=True)

    labels = [
        ("nac_raw", "raw NAC vs AC (floor)"),
        ("passthrough", "decode(encode(NAC)) [passthrough]"),
        ("passthrough_c", "  ... clamped [0,1]"),
        ("ae_ceiling", "decode(encode(AC)) [AE CEILING]"),
        ("ae_ceiling_c", "  ... clamped [0,1]"),
        ("flow", "flow prediction"),
        ("flow_c", "  ... clamped [0,1]"),
    ]
    summary = {k: _aggregate(v) for k, v in rows.items() if v}

    w = 34
    print("\n" + "=" * (w + 12 * len(KEYS)))
    print("3D CEILING / PASSTHROUGH DIAGNOSTIC  (mean over patients, normalized intensity)")
    print("=" * (w + 12 * len(KEYS)))
    print(" " * w + "".join(f"{k:>12}" for k in KEYS))
    for key, label in labels:
        if key not in summary:
            continue
        line = f"{label:<{w}}"
        for k in KEYS:
            v, _ = summary[key][k]
            line += f"{v:>12.4f}" if np.isfinite(v) else f"{'-':>12}"
        print(line)

    if "ae_ceiling" in summary and "flow" in summary:
        print("\nHEADROOM (flow -> AE ceiling), unclamped:")
        for k in KEYS:
            c, _ = summary["ae_ceiling"][k]
            f, _ = summary["flow"][k]
            if not (np.isfinite(c) and np.isfinite(f)):
                continue
            gap = c - f
            # For error-like metrics closer-to-zero is better, so report |gap| and a verdict
            # based on whether the AE alone already exhibits the flow model's error.
            if k in ("nrmse", "mae", "max_rel_error", "p95_rel_error", "hot_band_rel_error", "rel_bias"):
                verdict = "AE-LIMITED (no headroom)" if abs(c) >= 0.8 * abs(f) and abs(f) > 1e-6 else "flow-limited"
            else:
                verdict = "flow-limited" if gap > 0.01 else "at ceiling"
            print(f"  {k:<16} flow={f:>9.4f}  ceiling={c:>9.4f}  ->  {verdict}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "mode": args.mode, "n_patients": len(candidates), "size": args.size,
            "ddim_steps": args.ddim_steps, "ae_ckpt": args.ae_ckpt,
            "diff_ckpt": None if args.skip_flow else args.diff_ckpt,
            "note": "normalized intensity, not calibrated SUV; '_c' rows are clamped to [0,1]",
            "summary": {k: {m: v[0] for m, v in d.items()} for k, d in summary.items()},
        }, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
