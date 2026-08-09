"""Sanity-check the quantitative-fidelity losses against the ACTUAL measured failure.

The model behaves like ``pred ~= 0.84*gt + 0.057`` (under-dispersion). Each loss must:
  * be ~0 for a perfect prediction,
  * be clearly >0 for that shrunk prediction,
  * prefer the correct dynamic range over the shrunk one,
  * pass gradients to pred and not to gt.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.training.utils.quant_losses import (
    build_quant_loss,
    expectile_loss,
    intensity_weighted_l1,
    slope_penalty,
)

torch.manual_seed(0)
gt = torch.rand(1, 1, 16, 16, 16) * 0.6
gt[0, 0, 6:10, 6:10, 6:10] = 1.5           # a hot lesion
gt = gt.clamp(0, 1)                         # as normalize_volume does
shrunk = (0.84 * gt + 0.057).clamp(0, 1)    # the measured failure mode

print("oracle (pred == gt), all should be ~0")
print(f"  slope     {float(slope_penalty(gt, gt)):.3e}")
print(f"  iw        {float(intensity_weighted_l1(gt, gt)):.3e}")
print(f"  expectile {float(expectile_loss(gt, gt)):.3e}")

print("\nshrunk prediction (pred = 0.84*gt + 0.057) -- must be clearly > 0")
print(f"  slope     {float(slope_penalty(shrunk, gt)):.6f}")
print(f"  iw        {float(intensity_weighted_l1(shrunk, gt)):.6f}")
print(f"  expectile {float(expectile_loss(shrunk, gt, q=0.7)):.6f}")

print("\nslope_penalty ranks candidates (lower = better):")
for name, p in (("perfect", gt), ("shrunk 0.84", shrunk), ("stretched 1.2x", (1.2 * gt).clamp(0, 1))):
    print(f"  {name:16} {float(slope_penalty(p, gt)):.6f}")

print("\nexpectile asymmetry -- under-prediction must cost MORE than over:")
# An ADDITIVE +-delta, and NOT clamped: clamping would remove part of the +delta error
# (gt saturates at 1.0), making the two perturbations unequal and faking an asymmetry.
delta = 0.1
lo, hi = gt - delta, gt + delta
for q in (0.7, 0.5):
    u, o = float(expectile_loss(lo, gt, q=q)), float(expectile_loss(hi, gt, q=q))
    note = "  <- q=0.5 must be EXACTLY symmetric" if q == 0.5 else ""
    print(f"  q={q}  under={u:.6f}  over={o:.6f}  ratio={u / max(o, 1e-12):.3f}{note}")

print("\nintensity weighting must emphasise the HOT region:")
# Controlled: corrupt the SAME NUMBER of voxels by the same ABSOLUTE amount, hot vs cold.
# (Comparing 64 lesion voxels against every cold voxel would just measure how many voxels
# were touched, not the weighting.)
fg = gt > 0.05 * gt.max()
vals = gt[fg]
k = 200
hot_thr = vals.kthvalue(max(1, vals.numel() - k)).values      # top-k boundary
cold_thr = vals.kthvalue(min(vals.numel(), k)).values          # bottom-k boundary
hot_sel = fg & (gt >= hot_thr)
cold_sel = fg & (gt <= cold_thr)
d = 0.1
err_hot = gt.clone(); err_hot[hot_sel] -= d
err_cold = gt.clone(); err_cold[cold_sel] -= d
print(f"  corrupting {int(hot_sel.sum())} HOT voxels by -{d}: iw={float(intensity_weighted_l1(err_hot, gt)):.6f}")
print(f"  corrupting {int(cold_sel.sum())} COLD voxels by -{d}: iw={float(intensity_weighted_l1(err_cold, gt)):.6f}")
print("  (hot must cost MORE -- that is the whole point of the weighting)")
print(f"  plain L1 for reference, hot={float((err_hot - gt).abs().mean()):.6f} "
      f"cold={float((err_cold - gt).abs().mean()):.6f}  (should be ~equal)")

print("\ngradients:")
p = torch.nn.Parameter(shrunk.clone())
g = gt.clone().requires_grad_(True)
slope_penalty(p, g).backward()
print(f"  grad reaches pred: {p.grad is not None and float(p.grad.abs().sum()) > 0}")
print(f"  grad leaked to gt: {g.grad is not None} (must be False)")

print("\nfactory:")
print(f"  'none'  -> {build_quant_loss('none')}")
print(f"  'slope' -> {build_quant_loss('slope', variance_weight=0.5).__name__}")
try:
    build_quant_loss("bogus")
    print("  'bogus' -> DID NOT RAISE (bad: a silent no-op ruins an ablation arm)")
except ValueError as e:
    print(f"  'bogus' -> raises ValueError: {str(e)[:60]}...")
