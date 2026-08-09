"""Can the hot-tissue underestimation be fixed for FREE by inverting the shrinkage?

The model is measured to behave like  pred ~= a*gt + b  with a~0.84, b~0.057: MSE's optimum
is the conditional mean, which is systematically less extreme than the truth. So try the
post-hoc inverse,  pred' = (pred - b)/a,  which by construction restores slope->1.

LEAK-FREE: for each held-out patient the (a,b) used are the MEAN slope/intercept of the
OTHER patients only (leave-one-patient-out), so the correction is never fitted on the
patient it is applied to.

The point is to quantify the TRADE: dividing by a<1 rescales the residual too, so it must
cost PSNR/NRMSE while fixing slope and hot-band bias. That tension is the real decision --
MSE-style metrics reward shrinkage, quantitative SUV fidelity punishes it.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.training.data import load_patient_by_path
from src.training.evaluate import _resolve_candidates
from src.training.infer import _load_model, _resize_volume, _schedule_from_config
from src.training.models.autoencoder3d import ae3d_decode, ae3d_encode, build_autoencoder_3d
from src.training.models.diffusion3d import build_diffusion_3d
from src.training.utils.image_metrics import clamp_unit, image_quality_metrics
from src.training.utils.translate import sample_nac_to_ac

p = argparse.ArgumentParser()
p.add_argument("--data_dir", required=True, nargs="+")
p.add_argument("--ae_ckpt", default="outputs/ae3d_p/best.pt")
p.add_argument("--diff_ckpt", default="outputs/ft3d_flow_p/best.pt")
p.add_argument("--split_json", default="outputs/eval/_clean24_split.json")
p.add_argument("--size", type=int, default=64)
p.add_argument("--ddim_steps", type=int, default=16)
p.add_argument("--device", default="cuda")
p.add_argument("--out", default="outputs/report/recalibrate.json")
a = p.parse_args()

dev = torch.device(a.device)
ae, _ = _load_model(a.ae_ckpt, build_autoencoder_3d, dev, use_ema=True)
model, cfg = _load_model(a.diff_ckpt, build_diffusion_3d, dev, use_ema=True)
sched = _schedule_from_config(cfg, dev)
scale = float(cfg.get("latent_scale", 1.0))
cands, _ = _resolve_candidates(a.data_dir, "test", 0.2, 0.1, 42, split_json=a.split_json)

# Pass 1: predict once, cache pred/gt on CPU, and fit each patient's own (a,b).
preds, gts, fits = [], [], []
for i, path in enumerate(cands, 1):
    v = load_patient_by_path(path, device=dev, load_ct=False, run_segmentation=False)
    if v.get("pet_nac") is None or v.get("pet_ac") is None:
        continue
    nac = _resize_volume(v["pet_nac"], a.size); ac = _resize_volume(v["pet_ac"], a.size)
    with torch.no_grad():
        lat = ae3d_encode(ae, nac, scale=scale)
        x0 = sample_nac_to_ac(model, sched, lat, cfg, num_steps=a.ddim_steps,
                              spacing="linear", guidance_scale=1.0, clip_x0=None, tag="recal")
        pred = clamp_unit(ae3d_decode(ae, x0, scale=scale))
    m = image_quality_metrics(pred, ac)
    preds.append(pred.cpu()); gts.append(ac.cpu())
    fits.append((m["reg_slope"], m["reg_intercept"]))
    print(f"  [{i}/{len(cands)}] slope={m['reg_slope']:.3f} intercept={m['reg_intercept']:+.4f}", flush=True)

n = len(preds)
sl = np.array([f[0] for f in fits]); ic = np.array([f[1] for f in fits])
print(f"\nper-patient slope: mean={sl.mean():.4f} sd={sl.std():.4f} min={sl.min():.3f} max={sl.max():.3f}")

# Pass 2: leave-one-out global correction.
rows_raw, rows_cal = [], []
for i in range(n):
    keep = [j for j in range(n) if j != i]
    a_loo, b_loo = float(sl[keep].mean()), float(ic[keep].mean())
    corrected = clamp_unit((preds[i] - b_loo) / max(a_loo, 1e-6))
    rows_raw.append(image_quality_metrics(preds[i], gts[i]))
    rows_cal.append(image_quality_metrics(corrected, gts[i]))

KEYS = ["psnr", "ssim", "nrmse", "mae", "voxel_r2", "reg_slope", "reg_intercept",
        "rel_bias", "p95_rel_error", "hot_band_rel_error"]
def mean(rows, k):
    v = [r[k] for r in rows if np.isfinite(r[k])]
    return float(np.mean(v)) if v else float("nan")

print(f"\n{'metric':<20}{'raw':>10}{'recalibrated':>14}{'delta':>10}   note")
print("-" * 70)
for k in KEYS:
    r, c = mean(rows_raw, k), mean(rows_cal, k)
    note = ""
    if k in ("reg_slope",): note = "target 1.0"
    if k in ("hot_band_rel_error", "p95_rel_error", "rel_bias", "reg_intercept"): note = "target 0.0"
    if k in ("psnr", "ssim", "voxel_r2"): note = "higher better"
    if k in ("nrmse", "mae"): note = "lower better"
    print(f"{k:<20}{r:>10.4f}{c:>14.4f}{c - r:>+10.4f}   {note}")

json.dump({"n": n, "loo_global_affine": True,
           "slope_mean": float(sl.mean()), "slope_sd": float(sl.std()),
           "raw": {k: mean(rows_raw, k) for k in KEYS},
           "recalibrated": {k: mean(rows_cal, k) for k in KEYS}},
          open(a.out, "w"), indent=2)
print(f"\nwrote {a.out}")
