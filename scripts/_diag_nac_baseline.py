"""Score the uncorrected input against the truth at the slab-chain geometry.

The baseline row of the report's main table: no model at all, just the NAC
(non-attenuation-corrected) PET volume compared with the AC (attenuation-corrected)
one, both resized in-plane to --size and keeping every native slice -- the same
geometry the slab chain is scored at, so the row is comparable with the flow arms.

Run inside WSL:
  ~/petct/.venv/bin/python scripts/_diag_nac_baseline.py --device cuda
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.training.data import load_patient_by_path, resize_pair_native_depth
from src.training.evaluate import _resolve_candidates
from src.training.utils.image_metrics import clamp_unit, image_quality_metrics

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
KEYS = ["psnr", "ssim", "nrmse", "mae", "voxel_r2", "reg_slope", "reg_intercept",
        "rel_bias", "hot_band_rel_error", "p95_rel_error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_json", default="outputs/ft3d_flow_p/split.json")
    ap.add_argument("--mode", default="test", choices=["val", "test"])
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--cache_dir", default="/root/petct_cache")
    ap.add_argument("--out", default="outputs/eval/nac_baseline/native_depth_128.json")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    candidates, n_ref = _resolve_candidates(ROOTS, a.mode, 0.2, 0.1, 42, split_json=a.split_json)
    print(f"{len(candidates)} {a.mode} candidates (reference {n_ref}); split {a.split_json}")
    per, t0 = [], time.time()
    for patient in candidates:
        vols = load_patient_by_path(patient, device=device, load_ct=False,
                                    run_segmentation=False, cache_dir=a.cache_dir)
        if vols.get("pet_nac") is None or vols.get("pet_ac") is None:
            continue
        nac_r, ac_r = resize_pair_native_depth(vols["pet_nac"], vols["pet_ac"], a.size)
        nac = nac_r.unsqueeze(0).to(device)
        ac = ac_r.unsqueeze(0).to(device)
        m = image_quality_metrics(clamp_unit(nac), ac)
        m["patient"] = str(patient)
        per.append(m)
        print(f"[{len(per)}/{len(candidates)}] Z={ac_r.shape[1]} psnr={m['psnr']:.2f} "
              f"ssim={m['ssim']:.4f} slope={m['reg_slope']:.3f} hot={m['hot_band_rel_error']:+.4f}",
              flush=True)
    print(f"done: {len(per)} patients in {time.time() - t0:.0f} s")
    summary = {k: sum(r[k] for r in per) / max(1, len(per)) for k in KEYS}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"label": "nac_as_is", "mode": a.mode, "split_json": a.split_json,
                   "size": a.size, "volumetric": False, "n": len(per),
                   "note": "no model; NAC vs AC, in-plane resize only, native depth kept",
                   "summary": summary, "per_patient": per}, f, indent=2)
    print(f"wrote {a.out}")
    for k in KEYS:
        print(f"  {k:<20}{summary[k]:>12.4f}")


if __name__ == "__main__":
    main()
