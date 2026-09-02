"""What does PET-value weighting of the latent velocity MSE actually do to the loss?

`bg_weight`/`latent_weight: occupancy` rebalances the flow loss by WHERE anatomy is;
`latent_weight: intensity` rebalances it by HOW HOT the anatomy is. Both are normalized by
their mean, so the only thing that matters is how the weight MASS is distributed -- which
is what this prints, on real held-out patients, before any GPU time is spent:

  * the share of the loss each GT-uptake band receives, unweighted vs under each setting,
  * the robust normalizer (foreground percentile) each setting picks, versus the max,
  * how much a synthetic hot outlier moves that normalizer (the reason it is a percentile).

Run (WSL venv or the Windows host -- CPU only, no model, no checkpoint needed):

    python scripts/_diag_intensity_weight.py --data_dir "<root>" [<root> ...] \
        --cache_dir C:/DeepTrainingData/PetCT --n 8
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
from src.training.utils.quant_losses import (
    _pool_to_latent,
    _robust_scale,
    build_latent_weight,
)

# (label, kwargs for build_latent_weight)
SETTINGS = [
    ("occupancy bg=.1", dict(mode="occupancy", bg_weight=0.1)),
    ("intensity g=1 p99", dict(mode="intensity", bg_weight=0.1, gamma=1.0, percentile=99.0)),
    ("intensity g=1 p90", dict(mode="intensity", bg_weight=0.1, gamma=1.0, percentile=90.0)),
    ("intensity g=2 p99", dict(mode="intensity", bg_weight=0.1, gamma=2.0, percentile=99.0)),
    ("intensity g=.5 p99", dict(mode="intensity", bg_weight=0.1, gamma=0.5, percentile=99.0)),
    ("intensity g=1 max-pool", dict(mode="intensity", bg_weight=0.1, gamma=1.0, pool="max")),
    ("intensity union g=1", dict(mode="intensity", bg_weight=0.1, gamma=1.0, source="union")),
]
# GT-uptake bands, as a fraction of the volume max (which normalize_volume pins at 1.0).
BANDS = [(0.0, 0.05, "air"), (0.05, 0.3, "cold"), (0.3, 0.6, "mid"), (0.6, 1.01, "hot")]

p = argparse.ArgumentParser()
p.add_argument("--data_dir", required=True, nargs="+")
p.add_argument("--cache_dir", default=None)
p.add_argument("--split_json", default="outputs/ft3d_flow_p/split.json")
p.add_argument("--mode", default="test", choices=("test", "val", "all"),
               help="'all' skips the split lookup -- handy on the Windows host, whose "
                    "paths do not match a split.json written under WSL.")
p.add_argument("--size", type=int, default=64)
p.add_argument("--latent", type=int, default=16)
p.add_argument("--n", type=int, default=8)
p.add_argument("--floors", type=float, nargs="*", default=None,
               help="Also report intensity settings at these floors. The floor is what an "
                    "air position is worth, i.e. how much of the loss the model still "
                    "spends on being right about background.")
a = p.parse_args()

for fl in (a.floors or []):
    SETTINGS.append((f"intensity g=1 floor={fl:g}",
                     dict(mode="intensity", bg_weight=fl, gamma=1.0, percentile=99.0)))
fns = [(label, build_latent_weight(**kw)) for label, kw in SETTINGS]
mass = {label: np.zeros(len(BANDS)) for label, _ in fns}
mass["unweighted"] = np.zeros(len(BANDS))
scales, spiked_scales, maxes, n = [], [], [], 0

cands, _ = _resolve_candidates(a.data_dir, a.mode, 0.2, 0.1, 42, split_json=a.split_json)
for path in cands[: a.n]:
    v = load_patient_by_path(path, device=torch.device("cpu"), load_ct=False,
                             run_segmentation=False, cache_dir=a.cache_dir)
    if v.get("pet_nac") is None or v.get("pet_ac") is None:
        continue
    ac = _resize_volume(v["pet_ac"], a.size)
    nac = _resize_volume(v["pet_nac"], a.size)
    lat = (a.latent,) * 3

    # Per-latent-cell GT uptake decides which band a cell belongs to.
    cell_ac = _pool_to_latent(ac, lat, "avg")
    band_idx = np.full(cell_ac.numel(), -1)
    flat = cell_ac.flatten().numpy()
    for b, (lo, hi, _) in enumerate(BANDS):
        band_idx[(flat >= lo) & (flat < hi)] = b

    for label, fn in fns:
        w = fn(ac, nac, lat)
        w = torch.ones_like(cell_ac) if w is None else w
        wf = (w / w.mean()).flatten().numpy()          # exactly what flow_loss applies
        for b in range(len(BANDS)):
            mass[label][b] += wf[band_idx == b].sum() / wf.size
    for b in range(len(BANDS)):
        mass["unweighted"][b] += float((band_idx == b).sum()) / band_idx.size

    scales.append(float(_robust_scale(ac, 99.0)))
    maxes.append(float(ac.max()))
    spiked = ac.clone()
    spiked.flatten()[0] = 50.0 * float(ac.max())       # one injection-site-sized outlier
    spiked_scales.append(float(_robust_scale(spiked, 99.0)))
    n += 1

if not n:
    raise SystemExit("no paired patients found")

print(f"n = {n} held-out patients, {a.size}^3 image -> {a.latent}^3 latent\n")
print(f"{'setting':22s} " + " ".join(f"{name:>7s}" for _, _, name in BANDS))
print("-" * 60)
for label in ["unweighted"] + [lab for lab, _ in fns]:
    row = mass[label] / n
    print(f"{label:22s} " + " ".join(f"{100 * x:6.1f}%" for x in row))
print("\nShare of the velocity-MSE gradient each GT-uptake band receives. 'unweighted' is")
print("today's plain mean -- i.e. what fraction of the loss is currently spent on air.\n")
print(f"robust scale (fg p99) : {np.mean(scales):.4f}   volume max: {np.mean(maxes):.4f}")
print(f"same, with a 50x spike: {np.mean(spiked_scales):.4f}   "
      f"(a max normalizer would read {50 * np.mean(maxes):.2f} and collapse every weight)")
