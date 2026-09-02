"""THE decisive 64^3-vs-128^3 comparison: score both AEs against NATIVE-resolution truth.

Every number reported so far compares a model to its OWN resized ground truth -- ae3d_p to a
64^3 target, ae3d_2x to a 128^3 target. Those are different yardsticks: the 64^3 target is a
pre-blurred version of the 128^3 one, so it is intrinsically easier to reconstruct and scores
higher for the same quality of autoencoder. Comparing 33.05 dB to 29.90 dB across them is
meaningless.

This scores both on the ONLY common yardstick -- the native volume neither model was trained
on -- by running each one's full round trip and mapping the result back to native space:

    native -> resize S^3 -> AE encode -> AE decode -> upsample to native   vs   native

and reports, for reference, the resize-only ceiling of each path (what a PERFECT autoencoder
at that resolution could achieve), so the AE's own contribution is separable from the
resolution's.

Decision rule: if the 128^3 path beats the 64^3 path in native space, the extra resolution is
genuinely reaching the output and stage 2 (a 128^3 flow model) is worth training. If it does
not, the 8x-harder compression into the same 16^3 latent ate the benefit and the honest call
is to stop.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from src.training.data import load_patient_by_path
from src.training.evaluate import _resolve_candidates
from src.training.infer import _load_model
from src.training.models.autoencoder3d import ae3d_decode, ae3d_encode, build_autoencoder_3d
from src.training.utils.image_metrics import clamp_unit, image_quality_metrics

p = argparse.ArgumentParser()
p.add_argument("--data_dir", required=True, nargs="+")
p.add_argument("--ae64", default="outputs/ae3d_p/best.pt")
p.add_argument("--ae128", default="outputs/ae3d_2x/best.pt")
p.add_argument("--split_json", default="outputs/ft3d_flow_p/split.json")
p.add_argument("--n", type=int, default=12)
p.add_argument("--device", default="cuda")
p.add_argument("--out", default="outputs/report/native_ceiling.json")
a = p.parse_args()

dev = torch.device(a.device)
print("loading AEs ...")
ae64, cfg64 = _load_model(a.ae64, build_autoencoder_3d, dev, use_ema=True)
ae128, cfg128 = _load_model(a.ae128, build_autoencoder_3d, dev, use_ema=True)
print(f"  64^3  AE: {cfg64.get('model', {}).get('block_out_channels')}  latent_ch={cfg64.get('latent_channels')}")
print(f"  128^3 AE: {cfg128.get('model', {}).get('block_out_channels')}  latent_ch={cfg128.get('latent_channels')}")

PATHS = [(64, ae64), (128, ae128)]
KEYS = ["psnr", "ssim", "voxel_r2", "reg_slope", "p95_rel_error", "hot_band_rel_error"]
rows = {f"{s}_ae": [] for s, _ in PATHS}
rows.update({f"{s}_resize_only": [] for s, _ in PATHS})

cands, _ = _resolve_candidates(a.data_dir, "test", 0.2, 0.1, 42, split_json=a.split_json)
done = 0
for path in cands:
    if done >= a.n:
        break
    v = load_patient_by_path(path, device=dev, load_ct=False, run_segmentation=False)
    ac = v.get("pet_ac")
    if ac is None:
        continue
    native = ac.float()[None, None]
    shp = tuple(native.shape[2:])
    for s, ae in PATHS:
        down = F.interpolate(native, size=(s, s, s), mode="trilinear", align_corners=False)
        # (a) resize-only: the ceiling a PERFECT autoencoder at this resolution could hit.
        back = F.interpolate(down, size=shp, mode="trilinear", align_corners=False)
        rows[f"{s}_resize_only"].append(image_quality_metrics(back, native))
        # (b) the real round trip through this AE, then back to native.
        with torch.no_grad():
            rec = clamp_unit(ae3d_decode(ae, ae3d_encode(ae, down)))
        rec_native = F.interpolate(rec, size=shp, mode="trilinear", align_corners=False)
        rows[f"{s}_ae"].append(image_quality_metrics(rec_native, native))
    done += 1
    print(f"  [{done}/{a.n}] {Path(path).name[:38]}  native {shp}", flush=True)

def m(rs, k):
    v = [r[k] for r in rs if np.isfinite(r[k])]
    return float(np.mean(v)) if v else float("nan")

print(f"\n{'=' * 96}")
print(f"SCORED AGAINST NATIVE-RESOLUTION TRUTH  (n={done} held-out patients)")
print(f"{'=' * 96}")
print(f"{'path':<38}" + "".join(f"{k:>16}" for k in KEYS))
print("-" * (38 + 16 * len(KEYS)))
order = [("64_resize_only",  "64^3  resize only (ceiling)"),
         ("64_ae",           "64^3  + ae3d_p        <- CURRENT"),
         ("128_resize_only", "128^3 resize only (ceiling)"),
         ("128_ae",          "128^3 + ae3d_2x       <- NEW")]
for key, label in order:
    if not rows[key]:
        continue
    print(f"{label:<38}" + "".join(f"{m(rows[key], k):>16.4f}" for k in KEYS))

if rows["64_ae"] and rows["128_ae"]:
    print(f"\n{'VERDICT':<38}")
    for k in KEYS:
        lo, hi = m(rows["64_ae"], k), m(rows["128_ae"], k)
        if k in ("psnr", "ssim", "voxel_r2"):
            better, d = ("128^3" if hi > lo else "64^3"), hi - lo
        elif k == "reg_slope":
            better, d = ("128^3" if abs(hi - 1) < abs(lo - 1) else "64^3"), hi - lo
        else:
            better, d = ("128^3" if abs(hi) < abs(lo) else "64^3"), hi - lo
        print(f"  {k:<22} 64^3={lo:>9.4f}   128^3={hi:>9.4f}   delta={d:>+9.4f}   -> {better} better")
    # How much of each resolution's OWN ceiling did its AE actually reach?
    print("\n  AE efficiency (how close each AE gets to its own resolution's ceiling, PSNR):")
    for s in (64, 128):
        c, r = m(rows[f"{s}_resize_only"], "psnr"), m(rows[f"{s}_ae"], "psnr")
        print(f"    {s}^3: ceiling {c:.2f} dB, achieved {r:.2f} dB  -> gives up {c - r:.2f} dB to the AE")

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
json.dump({"n": done, "ae64": a.ae64, "ae128": a.ae128,
           "summary": {k: {kk: m(v, kk) for kk in KEYS} for k, v in rows.items() if v}},
          open(out, "w"), indent=2)
print(f"\nwrote {out}")
