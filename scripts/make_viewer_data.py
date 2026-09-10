"""PACK NAC / PREDICTION / TRUTH VOLUMES FOR THE STANDALONE HTML VIEWER.

The evaluation dumps hold the prediction and the truth but not the non-corrected input, so
the input is rebuilt here exactly as the scorer built it: load the patient, then
``resize_pair_native_depth(nac, ac, size)``. That is the same call
``src/training/evaluate.py`` makes in slab mode, so the three volumes land on one grid --
checked, not assumed: the rebuilt truth is compared against the dumped truth and the run
aborts if they differ.

Each volume is written as ONE png "atlas": every slice as a tile in a grid. A png of a PET
volume compresses well (most of the body box is air), which is what makes a self-contained
viewer of a few hundred slices practical to hand over as a single file. The viewer decodes
the atlas back into a flat array once, so it can also cut the volume the other two ways.

Values are already normalised to [0,1] by the scorer; they are quantised to 256 levels for
display. That is a display-only loss and no metric in the report is computed from these
files.

    ~/petct/.venv/bin/python scripts/make_viewer_data.py --out docs/viewer_data
"""

import argparse
import base64
import io as _io
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.data import load_patient_by_path, resize_pair_native_depth  # noqa: E402

TILE_COLS = 16

# Three ordinary cases, one per cancer site, so the viewer shows the method holding up
# away from one part of the body. Named rather than ranked: the point is the site, and
# all three sit within about a decibel of the test mean anyway.
SITES = {"lung": "AMC-018", "bladder": "31485548", "uterine": "C3L-00962"}


def atlas_png(vol_u8, cols=TILE_COLS):
    """Lay a (Z,H,W) uint8 volume out as one png. Returns (bytes, cols, rows)."""
    z, h, w = vol_u8.shape
    rows = (z + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w), dtype=np.uint8)
    for i in range(z):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = vol_u8[i]
    buf = _io.BytesIO()
    Image.fromarray(sheet, mode="L").save(buf, format="PNG", optimize=True)
    return buf.getvalue(), cols, rows


def to_u8(vol):
    """[0,1] float -> uint8, clamped. The scorer already clamped, so this only quantises."""
    return np.clip(np.asarray(vol, dtype=np.float32), 0.0, 1.0).__mul__(255.0).round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_json",
                    default="outputs/eval/ft3d_slab_flow/slab_ae3d_cont_final_s16.json")
    ap.add_argument("--dump", default="outputs/eval/dump_ae3d_cont_final")
    ap.add_argument("--out", default="docs/viewer_data")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--patients", default="median,best,worst,lung,bladder,uterine",
                    help="which of the ranked test patients to pack")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    meta_all = json.load(open(args.eval_json))
    pp = meta_all["per_patient"]
    psnr = np.array([e["psnr"] for e in pp])
    order = np.argsort(psnr)
    picks = {"worst": int(order[0]), "median": int(order[len(order) // 2]),
             "best": int(order[-1]), "p25": int(order[len(order) // 4])}
    by_name = {os.path.basename(e["patient"].rstrip("/")): i for i, e in enumerate(pp)}
    for tag, name in SITES.items():
        if name not in by_name:
            raise SystemExit("%s (%s) is not a scored patient in %s"
                             % (name, tag, args.eval_json))
        picks[tag] = by_name[name]

    dumps = sorted(f for f in os.listdir(args.dump) if f.endswith("_pred.npy"))
    manifest = []
    for tag in [t.strip() for t in args.patients.split(",") if t.strip()]:
        idx = picks[tag]
        entry = pp[idx]
        path = entry["patient"]
        pred = np.load(os.path.join(args.dump, dumps[idx]))
        gt_dump = np.load(os.path.join(args.dump, dumps[idx].replace("_pred.npy", "_gt.npy")))

        vols = load_patient_by_path(path, device="cpu", load_ct=False, run_segmentation=False)
        nac_r, ac_r = resize_pair_native_depth(vols["pet_nac"], vols["pet_ac"], args.size)
        nac = nac_r[0].numpy()
        gt = ac_r[0].numpy()

        # The rebuilt truth must be the dumped truth, or the input is off the scored grid.
        if gt.shape != gt_dump.shape:
            raise SystemExit("shape mismatch for %s: rebuilt %s vs dumped %s"
                             % (tag, gt.shape, gt_dump.shape))
        worst = float(np.abs(gt - gt_dump).max())
        if worst > 1e-4:
            raise SystemExit("rebuilt truth differs from the dumped truth for %s "
                             "(max %.2e) -- the input would not be on the scored grid" % (tag, worst))

        files = {}
        for name, vol in (("nac", nac), ("pred", pred), ("gt", gt)):
            png, cols, rows = atlas_png(to_u8(vol))
            fn = "%s_%s.png" % (tag, name)
            with open(os.path.join(args.out, fn), "wb") as fh:
                fh.write(png)
            files[name] = {"file": fn, "bytes": len(png),
                           "b64": base64.b64encode(png).decode("ascii")}
        z, h, w = gt.shape
        manifest.append({
            "tag": tag, "label": os.path.basename(path.rstrip("/")),
            "collection": os.path.basename(os.path.dirname(path.rstrip("/"))),
            "z": int(z), "h": int(h), "w": int(w), "cols": TILE_COLS,
            "rows": (int(z) + TILE_COLS - 1) // TILE_COLS,
            "psnr": float(entry["psnr"]), "ssim": float(entry["ssim"]),
            "slope": float(entry["reg_slope"]),
            "hot": float(entry["hot_band_rel_error"]),
            "voxel_r2": float(entry["voxel_r2"]),
            "files": files,
        })
        print("%-7s %-14s z=%3d  png nac/pred/gt = %.2f / %.2f / %.2f MB  (truth match %.1e)"
              % (tag, manifest[-1]["label"][:14], z,
                 files["nac"]["bytes"] / 1e6, files["pred"]["bytes"] / 1e6,
                 files["gt"]["bytes"] / 1e6, worst))

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump([{k: v for k, v in m.items() if k != "files"} | {
            "files": {n: {kk: vv for kk, vv in f.items() if kk != "b64"}
                      for n, f in m["files"].items()}} for m in manifest], fh, indent=1)
    total = sum(f["bytes"] for m in manifest for f in m["files"].values())
    print("total image payload: %.2f MB (base64 in one html: ~%.2f MB)"
          % (total / 1e6, total * 4 / 3 / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
