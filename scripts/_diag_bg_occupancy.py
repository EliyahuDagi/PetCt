"""How much of the flow loss is currently spent on empty space?

The velocity MSE averages over EVERY latent position. This measures, on real held-out
patients, what fraction of a 64^3 volume is foreground and what fraction of the 16^3 latent
grid holds no anatomy at all -- i.e. how much signal `bg_weight` actually rebalances, and
how the resulting weight map is distributed.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.training.data import load_patient_by_path
from src.training.evaluate import _resolve_candidates
from src.training.infer import _resize_volume
from src.training.utils.quant_losses import latent_occupancy_weight

p = argparse.ArgumentParser()
p.add_argument("--data_dir", required=True, nargs="+")
p.add_argument("--split_json", default="outputs/ft3d_flow_p/split.json")
p.add_argument("--size", type=int, default=64)
p.add_argument("--latent", type=int, default=16)
p.add_argument("--bg_weight", type=float, default=0.1)
p.add_argument("--n", type=int, default=8)
a = p.parse_args()

cands, _ = _resolve_candidates(a.data_dir, "test", 0.2, 0.1, 42, split_json=a.split_json)
img_fg, lat_air, lat_full, wmeans = [], [], [], []
for path in cands[: a.n]:
    v = load_patient_by_path(path, device=torch.device("cpu"), load_ct=False, run_segmentation=False)
    if v.get("pet_nac") is None or v.get("pet_ac") is None:
        continue
    ac = _resize_volume(v["pet_ac"], a.size)
    nac = _resize_volume(v["pet_nac"], a.size)
    fg = (ac > 0.05 * ac.max()) | (nac > 0.05 * nac.max())
    img_fg.append(float(fg.float().mean()))
    w = latent_occupancy_weight([ac, nac], (a.latent,) * 3, bg_weight=a.bg_weight)
    # A cell is "pure air" when its weight is still exactly bg_weight.
    lat_air.append(float((w <= a.bg_weight + 1e-6).float().mean()))
    lat_full.append(float((w >= 1.0 - 1e-6).float().mean()))
    wmeans.append(float(w.mean()))

n = len(img_fg)
print(f"n = {n} held-out patients, {a.size}^3 image -> {a.latent}^3 latent, bg_weight={a.bg_weight}\n")
print(f"image-space foreground (NAC|AC > 5% max) : {100*np.mean(img_fg):5.1f} %  "
      f"=> {100*(1-np.mean(img_fg)):.1f}% of the VOLUME is air")
print(f"latent cells that are PURE air           : {100*np.mean(lat_air):5.1f} %")
print(f"latent cells FULLY inside anatomy        : {100*np.mean(lat_full):5.1f} %")
print(f"mean weight over the latent grid         : {np.mean(wmeans):.3f}")
print()
share = np.mean(lat_air) * a.bg_weight / np.mean(wmeans)
print(f"With bg_weight={a.bg_weight}, pure-air cells hold ~{100*share:.1f}% of the loss,")
print(f"versus ~{100*np.mean(lat_air):.1f}% before -- i.e. the anatomy's share of the")
print(f"gradient rises by roughly {(1-share)/(1-np.mean(lat_air)):.2f}x.")
print()
print("NOTE the 16^3 latent is 4x-downsampled per axis, so each cell covers a 4^3 image")
print("block; far fewer cells are PURE air than the raw voxel fraction suggests. That caps")
print("how much rebalancing is available -- which is exactly what this measures.")
