"""What does the 64^3 resize itself destroy? -- the question the ceiling diagnostic cannot answer.

Every metric reported so far (AE ceiling, flow prediction) is computed against the ALREADY
RESIZED 64^3 ground truth. So they are all blind to information the resize threw away before
anything was measured. That makes them useless for deciding whether to retrain the whole
chain at 2x resolution.

This measures the resize loss directly, in NATIVE space:
    native GT   vs   (native -> 64^3 -> back to native)
and the same for 128^3, over the held-out patients. Interpretation:

  * If the 64^3 round-trip is already near-perfect, then 2x resolution buys nothing that any
    current metric could reward, and the flow model (7.1 dB below the AE ceiling) is the
    only thing worth working on.
  * If it is lossy -- especially in the hot band, where lesions live -- then the whole
    pipeline has a resolution ceiling nobody has measured, every reported number is
    optimistic, and a 2x retrain is justified.

CPU-only by default so it can run alongside GPU training.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from src.training.data import load_patient_by_path
from src.training.evaluate import _resolve_candidates
from src.training.utils.image_metrics import (
    hot_band_relative_error,
    percentile_relative_error,
    psnr,
    ssim,
    voxel_r2,
)

p = argparse.ArgumentParser()
p.add_argument("--data_dir", required=True, nargs="+")
p.add_argument("--split_json", default="outputs/ft3d_flow_p/split.json")
p.add_argument("--sizes", type=int, nargs="+", default=[64, 128])
p.add_argument("--n", type=int, default=8)
p.add_argument("--device", default="cpu")
a = p.parse_args()

dev = torch.device(a.device)
cands, _ = _resolve_candidates(a.data_dir, "test", 0.2, 0.1, 42, split_json=a.split_json)

rows = {s: [] for s in a.sizes}
shapes = []
for path in cands[: a.n]:
    v = load_patient_by_path(path, device=dev, load_ct=False, run_segmentation=False)
    ac = v.get("pet_ac")
    if ac is None:
        continue
    native = ac.float()[None, None]           # (1,1,Z,Y,X) as loaded/normalized
    shapes.append(tuple(native.shape[2:]))
    for s in a.sizes:
        down = F.interpolate(native, size=(s, s, s), mode="trilinear", align_corners=False)
        back = F.interpolate(down, size=native.shape[2:], mode="trilinear", align_corners=False)
        rows[s].append({
            "psnr": psnr(back, native),
            "ssim": ssim(back, native),
            "voxel_r2": voxel_r2(back, native),
            "p95_rel_error": percentile_relative_error(back, native),
            "hot_band_rel_error": hot_band_relative_error(back, native),
        })
    print(f"  {Path(path).name[:40]}  native {tuple(native.shape[2:])}", flush=True)

def m(rs, k):
    v = [r[k] for r in rs if np.isfinite(r[k])]
    return float(np.mean(v)) if v else float("nan")

print(f"\nnative shapes seen: {sorted(set(shapes))}")
print(f"n = {len(rows[a.sizes[0]])} patients\n")
KEYS = ["psnr", "ssim", "voxel_r2", "p95_rel_error", "hot_band_rel_error"]
print(f"{'resize round-trip':<24}" + "".join(f"{k:>20}" for k in KEYS))
print("-" * (24 + 20 * len(KEYS)))
for s in a.sizes:
    print(f"{'native -> ' + str(s) + '^3 -> native':<24}" + "".join(f"{m(rows[s], k):>20.4f}" for k in KEYS))

print(f"""
For scale, the flow model scores ~25.4 dB / voxel_r2 0.699 / hot_band -0.100 against the
64^3 GT, and the frozen AE's ceiling is ~32.4 dB / 0.937 / -0.016.

If the 64^3 round-trip above sits FAR above those numbers, resolution is not the binding
constraint and a 2x retrain cannot pay for itself. If it sits near or below them -- or if
its hot_band error is comparable to the flow model's -- the resize is destroying exactly the
signal the project is trying to predict, and 2x is the right call.
""")
