"""Is the uptake weight measuring ABSOLUTE brightness, or brightness relative to the crop?

The uptake-weighted latent loss (quant_losses.latent_intensity_weight) turns the AC
volume into a per-position weight

    w = floor + (1 - floor) * pool(clamp(ac / q, 0, 1))

where ``q`` is a high percentile of the FOREGROUND of the volume it is handed. On the
64x64x64 chain that volume was the whole body, so ``q`` was the body-wide bright level and
``w`` meant "how hot is this, absolutely".

Slab mode hands it a 16-slice crop instead. If a crop through the legs has its own low
``q``, the same faint tissue that would score 0.2 body-wide scores 1.0 inside the crop, and
the weight stops meaning "hot" and starts meaning "the brightest thing in this crop". That
would quietly change what the experiment tests, so measure it before launching:

  * the whole-volume ``q`` (should be ~1.0: data.normalize_volume already clips at the
    volume's own 99th percentile, so the top ~1% of voxels sit exactly at 1.0),
  * the per-slab ``q`` over many random 16-slice crops -- its spread IS the problem,
  * how much the weight map moves if ``q`` is pinned to the whole-volume value instead.

Run inside WSL:
  ~/petct/.venv/bin/python scripts/_diag_iw_slab_scale.py --patients 6 --slabs 24
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.training.data import _crop_slabs, load_patient_by_path, resize_pair_native_depth
from src.training.evaluate import _resolve_candidates
from src.training.utils.quant_losses import _robust_scale, latent_intensity_weight

ROOTS = [
    "/mnt/d/DeepTrainingData/Project/ACRIN 6668",
    "/mnt/d/DeepTrainingData/Project/PetCt/Bladder 13.11.25",
    "/mnt/d/DeepTrainingData/Project/PetCtNormal/PET_CT_NORMAL_2",
    "/mnt/d/DeepTrainingData/Project/NSCLC_Radiogenomics",
    "/mnt/d/DeepTrainingData/Project/TCGA-LUAD",
    "/mnt/d/DeepTrainingData/Project/CPTAC-LUAD",
    "/mnt/d/DeepTrainingData/Project/CPTAC-LSCC",
    "/mnt/d/DeepTrainingData/Project/CPTAC-UCEC",
    "/mnt/d/DeepTrainingData/Project/CPTAC-PDA",
    "/mnt/d/DeepTrainingData/Project/TCGA-THCA",
]


def q_of(v):
    """The weight's normalizer for one (1, Z, H, W) volume, as a float."""
    return float(_robust_scale(v.unsqueeze(0), 99.0).reshape(-1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=6)
    ap.add_argument("--slabs", type=int, default=24, help="random 16-slice crops per patient")
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--floor", type=float, default=0.1)
    ap.add_argument("--cache_dir", default="/root/petct_cache")
    ap.add_argument("--split_json", default="outputs/ft3d_flow_p/split.json")
    args = ap.parse_args()

    cands, _ = _resolve_candidates(ROOTS, "test", 0.2, 0.1, 42, split_json=args.split_json)
    # numpy RandomState: that is what _crop_slabs expects (it calls rng.randint).
    rng = np.random.RandomState(0)

    all_slab_q = []
    print("%-28s %8s %8s %8s %8s %8s" % ("patient", "vol_q", "slab_min", "slab_p25",
                                         "slab_med", "slab_max"))
    done = 0
    for cand in cands:
        if done >= args.patients:
            break
        path = cand[0] if isinstance(cand, (tuple, list)) else cand
        try:
            rec = load_patient_by_path(path, load_ct=False, run_segmentation=False,
                                      cache_dir=args.cache_dir)
        except Exception as exc:                                    # noqa: BLE001
            print("  skip %s (%s)" % (Path(str(path)).name, exc))
            continue
        nac, ac = rec.get("pet_nac"), rec.get("pet_ac")
        if nac is None or ac is None:
            continue
        nac_r, ac_r = resize_pair_native_depth(nac.float(), ac.float(), args.size)
        vol_q = q_of(ac_r)

        qs = []
        for _ in range(args.slabs):
            _, ac_slab = _crop_slabs(nac_r, ac_r, 1, args.depth, rng, False, None)
            qs.append(q_of(ac_slab[0]))
        qs = sorted(qs)
        all_slab_q.extend(qs)
        n = len(qs)
        print("%-28s %8.4f %8.4f %8.4f %8.4f %8.4f"
              % (Path(str(path)).name[:28], vol_q, qs[0], qs[n // 4], qs[n // 2], qs[-1]))
        done += 1

    if not all_slab_q:
        print("no patients scored")
        return
    q = sorted(all_slab_q)
    n = len(q)
    print("\nper-slab normalizer over %d crops: min %.4f  p10 %.4f  median %.4f  max %.4f"
          % (n, q[0], q[n // 10], q[n // 2], q[-1]))
    below = sum(1 for x in q if x < 0.9)
    print("crops whose own normalizer is below 0.9 (i.e. their faint tissue would be "
          "promoted to full weight): %d of %d (%.0f%%)" % (below, n, 100.0 * below / n))

    # How different is the resulting weight map? Compare the per-crop normalizer against
    # pinning it to 1.0 (the whole-volume level this data is normalized to by construction).
    print("\nweight-map effect on one patient's crops (latent grid 16 x 32 x 32):")
    path = cands[0][0] if isinstance(cands[0], (tuple, list)) else cands[0]
    rec = load_patient_by_path(path, load_ct=False, run_segmentation=False,
                               cache_dir=args.cache_dir)
    nac_r, ac_r = resize_pair_native_depth(rec["pet_nac"].float(), rec["pet_ac"].float(),
                                           args.size)
    rng = np.random.RandomState(1)
    print("%8s %10s %10s %10s" % ("slab_q", "w_mean_own", "w_mean_pin", "max_abs_diff"))
    for _ in range(8):
        _, ac_slab = _crop_slabs(nac_r, ac_r, 1, args.depth, rng, False, None)
        lat = (args.depth, args.size // 4, args.size // 4)
        w_own = latent_intensity_weight(ac_slab, lat, floor=args.floor)
        # Pinned: divide by 1.0 by hand, i.e. clamp the raw values then pool.
        rel = ac_slab.clamp(0.0, 1.0)
        pooled = torch.nn.functional.adaptive_avg_pool3d(rel, lat)
        w_pin = args.floor + (1.0 - args.floor) * pooled
        print("%8.4f %10.4f %10.4f %10.4f"
              % (q_of(ac_slab[0]), float(w_own.mean()), float(w_pin.mean()),
                 float((w_own - w_pin).abs().max())))


if __name__ == "__main__":
    main()
