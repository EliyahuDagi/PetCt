"""Diagnostic: is the apparent mirrored band at the bottom of the coronal view real
anatomy / in the raw loaded volume, or an artifact of the 64^3 resize + rendering?

Saves full-resolution and resized coronal PNGs for one test patient so they can be
compared side by side.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from src.training.data import load_patient_by_path
from src.training.infer import _resize_volume
from scripts.report_triplets import _lut, _panel_png, _window

patient = sys.argv[1] if len(sys.argv) > 1 else (
    "/mnt/d/DeepTrainingData/Project/ACRIN 6668/"
    "1_3_6_1_4_1_14519_5_2_1_7009_2403_100559100256769692836460885176")
out = Path("outputs/report/_diag")
out.mkdir(parents=True, exist_ok=True)
lut = _lut("hot")

vols = load_patient_by_path(patient, device=torch.device("cpu"), load_ct=False,
                            run_segmentation=False)
ac = vols["pet_ac"].float().cpu().numpy()
print("raw AC shape (Z,Y,X):", ac.shape, "spacing:", vols.get("spacing"))

# Full-resolution coronal at mid-Y, and a coronal MIP over Y.
vmin, vmax = _window(ac)
mid_y = ac.shape[1] // 2
_panel_png(np.flipud(ac[:, mid_y, :]), vmin, vmax, lut, out / "full_coronal_mid.png", 0)
_panel_png(np.flipud(ac.max(axis=1)), vmin, vmax, lut, out / "full_coronal_mip.png", 0)

# The 64^3 resized volume the model actually sees.
r = _resize_volume(vols["pet_ac"], 64)[0, 0].float().cpu().numpy()
rmin, rmax = _window(r)
_panel_png(np.flipud(r[:, 32, :]), rmin, rmax, lut, out / "r64_coronal_mid.png", 256)
_panel_png(np.flipud(r.max(axis=1)), rmin, rmax, lut, out / "r64_coronal_mip.png", 256)

# Is the bottom band a literal mirror of the band above it? Correlate the last
# quarter of Z against a flipped copy of the quarter before it (per-slice means).
prof = ac.reshape(ac.shape[0], -1).mean(axis=1)
n = ac.shape[0]
q = n // 4
tail = prof[n - q:]
prev = prof[n - 2 * q:n - q][::-1]
if tail.std() > 0 and prev.std() > 0:
    print("corr(tail, flipped previous quarter) =",
          float(np.corrcoef(tail, prev)[0, 1]))
print("z-profile (mean intensity per slice, 32 bins):")
bins = np.array_split(prof, 32)
print("  " + " ".join(f"{b.mean():.3f}" for b in bins))
print("wrote", out)
