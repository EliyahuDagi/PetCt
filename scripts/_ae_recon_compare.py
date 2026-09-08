"""Paired reconstruction comparison of two frozen autoencoders at native depth.

Scores ``decode(encode(volume))`` against the volume itself for every held-out test
patient, at the slab-chain geometry (in-plane ``--size`` x ``--size``, every native slice
kept). Two autoencoders are compared on the SAME patients:

  A: the 2D autoencoder applied slice by slice (default outputs/ae2d_p/best.pt)
  B: the in-plane-only 3D autoencoder (default outputs/ae3d_slab/best.pt)

Both AC (attenuation-corrected) and NAC (non-attenuation-corrected) PET round trips are
scored, because the flow chain encodes NAC and decodes AC. Each autoencoder is run on the
whole volume in one call (as evaluate.py does); if that runs out of GPU memory the volume
is processed in overlapping depth windows instead (the run says so).

Writes outputs/eval/ae_recon/<label>_{ac,nac}.json in the evaluate.py per_patient format,
so scripts/_cmp_eval_paired.py can compare them paired, and prints a mean table.

Run inside WSL:
  ~/petct/.venv/bin/python scripts/_ae_recon_compare.py --device cuda
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.training.data import load_patient_by_path, resize_pair_native_depth
from src.training.evaluate import _resolve_candidates
from src.training.infer import _load_model
from src.training.models.anisotropic import is_inplane_only
from src.training.models.autoencoder2d import SliceWiseAutoencoder, build_autoencoder_2d
from src.training.models.autoencoder3d import ae3d_decode, ae3d_encode, build_autoencoder_3d
from src.training.utils.image_metrics import clamp_unit, image_quality_metrics
from src.training.utils.sliding import depth_windowed_model_fn

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
TABLE_KEYS = ["psnr", "ssim", "voxel_r2", "reg_slope", "hot_band_rel_error", "p95_rel_error", "mae"]


def _peek_spatial_dims(ckpt):
    """2 or 3, read from the convolution weight shapes (the embedded config's model block of
    an inflated 3D autoencoder is copied from the 2D run and may still say spatial_dims 2)."""
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = state.get("config", {}) or {}
    weights = state.get("model", state)
    for k, v in weights.items():
        if k.endswith("conv.weight") and hasattr(v, "ndim") and v.ndim in (4, 5):
            return v.ndim - 2, cfg
    return int((cfg.get("model", {}) or {}).get("spatial_dims", 2)), cfg


def load_ae(ckpt, device):
    """Return (module with encode/decode on 5-D volumes, one-line description)."""
    dims, _ = _peek_spatial_dims(ckpt)
    if dims == 2:
        ae2d, _cfg = _load_model(ckpt, build_autoencoder_2d, device, use_ema=True)
        return SliceWiseAutoencoder(ae2d).eval(), "2D autoencoder, slice by slice"
    ae, _cfg = _load_model(ckpt, build_autoencoder_3d, device, use_ema=True)
    if not is_inplane_only(ae):
        raise ValueError(
            "%s builds a depth-compressing 3D autoencoder; this comparison needs one latent "
            "slice per image slice (model.anisotropic: true)." % ckpt)
    return ae.eval(), "in-plane-only 3D autoencoder (depth kept)"


@torch.no_grad()
def round_trip(ae, vol, window):
    """decode(encode(vol)) on the whole volume, or on depth windows when window > 0."""
    def f(x):
        return ae3d_decode(ae, ae3d_encode(ae, x))
    if window > 0:
        stride = max(1, window - 16)
        return depth_windowed_model_fn(lambda x, t: f(x), window, stride)(vol, None)
    return f(vol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae_a", default="outputs/ae2d_p/best.pt")
    ap.add_argument("--ae_b", default="outputs/ae3d_slab/best.pt")
    ap.add_argument("--label_a", default="ae2d_slicewise")
    ap.add_argument("--label_b", default="ae3d_slab")
    ap.add_argument("--split_json", default="outputs/ft3d_flow_p/split.json")
    ap.add_argument("--mode", default="test", choices=["val", "test"])
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--cache_dir", default="/root/petct_cache")
    ap.add_argument("--out_dir", default="outputs/eval/ae_recon")
    ap.add_argument("--max_patients", type=int, default=0)
    ap.add_argument("--window", type=int, default=0,
                    help="0 = whole volume per call (falls back to 64-slice windows on OOM)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)
    aes = []
    for ckpt, label in ((a.ae_a, a.label_a), (a.ae_b, a.label_b)):
        ae, desc = load_ae(ckpt, device)
        print(f"[{label}] {ckpt}: {desc}")
        aes.append((label, ae))

    candidates, n_ref = _resolve_candidates(ROOTS, a.mode, 0.2, 0.1, 42, split_json=a.split_json)
    print(f"{len(candidates)} {a.mode} candidates (reference {n_ref}); split {a.split_json}")
    first = aes[0][0]
    rows = {label: {"ac": [], "nac": []} for label, _ in aes}
    fell_back = set()
    t0 = time.time()
    for patient in candidates:
        if a.max_patients and len(rows[first]["ac"]) >= a.max_patients:
            break
        vols = load_patient_by_path(patient, device=device, load_ct=False,
                                    run_segmentation=False, cache_dir=a.cache_dir)
        if vols.get("pet_nac") is None or vols.get("pet_ac") is None:
            continue
        nac_r, ac_r = resize_pair_native_depth(vols["pet_nac"], vols["pet_ac"], a.size)
        vol_by_kind = {"ac": ac_r.unsqueeze(0).to(device), "nac": nac_r.unsqueeze(0).to(device)}
        line = [f"[{len(rows[first]['ac']) + 1}/{len(candidates)}] Z={ac_r.shape[1]}"]
        for label, ae in aes:
            for kind, vol in vol_by_kind.items():
                window = a.window
                try:
                    rec = round_trip(ae, vol, window)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    window = 64
                    fell_back.add(label)
                    rec = round_trip(ae, vol, window)
                m = image_quality_metrics(clamp_unit(rec), vol)
                m["patient"] = str(patient)
                m["window"] = window
                rows[label][kind].append(m)
                if kind == "ac":
                    line.append(f"{label}: psnr={m['psnr']:.2f} ssim={m['ssim']:.4f} r2={m['voxel_r2']:.4f}")
        print("  ".join(line), flush=True)
    print(f"done: {len(rows[first]['ac'])} patients in {time.time() - t0:.0f} s")
    if fell_back:
        print(f"NOTE: whole-volume call ran out of GPU memory for {sorted(fell_back)}; "
              f"those used 64-slice depth windows (stride 48) for the affected patients.")

    for label, _ in aes:
        for kind in ("ac", "nac"):
            per = rows[label][kind]
            summary = {k: sum(r[k] for r in per) / max(1, len(per)) for k in TABLE_KEYS}
            out = {"label": label, "kind": kind, "mode": a.mode, "split_json": a.split_json,
                   "size": a.size, "n": len(per), "summary": summary, "per_patient": per}
            path = os.path.join(a.out_dir, f"{label}_{kind}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            print(f"wrote {path}")

    # Mean table + paired PSNR win count (B over A), AC round trip then NAC round trip.
    la, lb = aes[0][0], aes[1][0]
    for kind in ("ac", "nac"):
        pa, pb = rows[la][kind], rows[lb][kind]
        if not pa:
            continue
        print(f"\n== {kind.upper()} round trip, {len(pa)} patients, mean over patients ==")
        print(f"{'metric':<20}{la:>16}{lb:>16}{'B-A':>12}")
        for k in TABLE_KEYS:
            ma = sum(r[k] for r in pa) / len(pa)
            mb = sum(r[k] for r in pb) / len(pb)
            print(f"{k:<20}{ma:>16.4f}{mb:>16.4f}{mb - ma:>+12.4f}")
        wins = sum(1 for ra, rb in zip(pa, pb) if rb["psnr"] > ra["psnr"])
        print(f"PSNR: {lb} beats {la} on {wins}/{len(pa)} patients")


if __name__ == "__main__":
    main()
