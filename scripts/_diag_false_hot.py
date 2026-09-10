"""Count the uptake a NAC->AC model INVENTS, which none of the reported metrics can see.

Every number in the eval JSON is an error measured where the ground truth already has
signal. ``psnr`` / ``ssim`` / ``nrmse`` / ``mae`` average over the whole volume, so a false
hot spot a few hundred voxels across moves them in the third decimal. ``rel_bias``,
``voxel_r2``, ``reg_slope``, ``p95_rel_error`` and ``hot_band_rel_error`` are all computed
over the foreground or over the ground truth's OWN hot band -- they ask "was the model
right about the bright places", never "did it add a bright place". So a model that paints
a lesion into cold tissue still scores well.

That gap matters for an arm whose loss is weighted by ground-truth uptake
(``src/training/utils/quant_losses.latent_intensity_weight``): such a weight rewards
pushing predicted values UP. Predicting uptake where there is none is the dangerous
direction in this domain, because a false hot spot reads as disease.

This script reads a directory of prediction / ground-truth pairs written by
``python -m src.training.evaluate --save_pred_dir DIR`` and reports, in plain numpy:

  bright       the ground truth's 99th percentile over its foreground, foreground being
               ``gt > 5% of gt.max()`` -- the same rule ``image_metrics._foreground_mask``
               uses, so this agrees with the reported metrics. ``data.normalize_volume``
               already clips every volume at its own 99th percentile, so on this data
               bright comes out at ~1.0 by construction.
  false_hot%   of the body voxels the ground truth calls COLD (below 20% of bright), the
               percentage the prediction calls HOT (above 60% of bright). This is the
               invented-uptake number.
  false_n      how many voxels that is.
  blob         the largest connected group of those voxels. The same count spread thinly
               over the body is a texture artefact; the same count in one lump is an
               invented lesion. Face-connected (a voxel joins its six neighbours, not the
               diagonal ones), so two lumps touching only at a corner stay separate.
  missed_hot%  the mirror statistic: of the voxels the ground truth calls hot, the
               percentage the prediction calls cold. An uptake-weighted loss is meant to
               REDUCE this one, so the two have to be read together -- buying fewer missed
               hot spots with more invented ones is not a win.
  pred_cold / gt_cold
               the mean value over the ground-truth-cold body voxels, for the prediction
               and for the ground truth. A general upward drift in cold tissue shows up
               here even when no single voxel crosses the hot threshold.

A per-patient table is printed, then two summary rows. POOLED sums the voxel counts over
all patients and divides once, which is the number to quote: these events are rare, and
averaging per-patient percentages would give a patient with a tiny cold region the same
weight as a whole body. MEAN is the plain average of the per-patient percentages, printed
next to it so a single bad patient driving the pooled figure is visible.

Usage:
  python scripts/_diag_false_hot.py outputs/eval/dump_ctrl
  python scripts/_diag_false_hot.py outputs/eval/dump_ctrl --vs outputs/eval/dump_iw
  python scripts/_diag_false_hot.py --a outputs/eval/dump_ctrl --b outputs/eval/dump_iw

With two directories the per-patient numbers are printed side by side with a win/loss
count on the false-hot percentage, pairing files by name. No GPU, no torch, no model --
only the .npy files.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.ndimage import label as _label
except Exception:                                                       # noqa: BLE001
    # Neither environment here has scipy: not the torch-free Windows one, and not the WSL
    # training venv (checked 2026-09-08). So the fallback below is the normal path, and
    # the scipy import is only a faster route if the package ever shows up.
    _label = None

# Above this many flagged voxels the pure-Python grouping is not worth waiting for. A
# prediction with that much invented uptake is broken on the plain counts alone, so the
# blob size would add nothing.
_BLOB_VOXEL_CAP = 4_000_000

PRED_SUFFIX = "_pred.npy"
GT_SUFFIX = "_gt.npy"


def _patient_key(pred_name):
    """Dump filename -> the key two directories are paired on.

    ``evaluate.py`` names files ``<ordinal>_<folder>_pred.npy``, where the ordinal counts
    the patients it actually evaluated. If one run skipped a patient (an unreadable pair,
    a lower ``--max_patients``) every later ordinal shifts by one while the folder name
    does not, so the ordinal is stripped and only the folder name is matched.
    """
    stem = pred_name[: -len(PRED_SUFFIX)]
    return re.sub(r"^\d+_", "", stem)


def _short(key, width=26):
    """Key shortened FOR DISPLAY, keeping the end.

    These folder names are DICOM study identifiers: they share a long common prefix and
    differ only in the last stretch, so cutting from the left makes every row read as the
    same patient. Cut from the left and keep the tail instead. Only the printed label is
    shortened -- pairing always uses the full key.
    """
    return key if len(key) <= width else "~" + key[-(width - 1):]


def _find_pairs(dump_dir):
    """Return ``[(key, pred_path, gt_path)]`` for one dump directory, sorted by name.

    A prediction whose ground-truth partner is missing is reported and skipped: half a
    pair cannot be scored, and silently dropping it would make two directories look
    comparable when they are not.
    """
    d = Path(dump_dir)
    if not d.is_dir():
        raise SystemExit("not a directory: %s" % (dump_dir,))
    out = []
    seen = set()
    for pred in sorted(d.glob("*" + PRED_SUFFIX)):
        gt = pred.with_name(pred.name[: -len(PRED_SUFFIX)] + GT_SUFFIX)
        if not gt.exists():
            print("  skip %s (no matching %s)" % (pred.name, GT_SUFFIX))
            continue
        key = _patient_key(pred.name)
        if key in seen:
            # Two dataset roots can hold folders with the same basename. Keep the ordinal
            # in the key so neither patient is dropped; that makes this one pairable
            # across directories only if both runs evaluated in the same order.
            key = pred.name[: -len(PRED_SUFFIX)]
            print("  note: duplicate folder name, keying %s by its full stem" % (key,))
        seen.add(key)
        out.append((key, pred, gt))
    if not out:
        raise SystemExit(
            "no '*%s' files in %s -- was eval run with --save_pred_dir?" % (PRED_SUFFIX, dump_dir))
    return out


def _largest_blob(mask):
    """Voxel count of the biggest connected group of True voxels (face-connected, 6-way).

    Why this exists as hand-written code: the flagged voxels are the whole point of this
    script -- the same number of them scattered one-by-one through the body is a texture
    artefact, while the same number in one lump is an invented lesion, and only a grouping
    tells the two apart. Neither Python environment in this project has scipy, so there is
    nothing to call.

    It stays cheap because it never touches the volume, only the voxels that were flagged:
    a merge-as-you-go pass (union-find) over the flagged coordinates, joining each to the
    at-most-three already-seen neighbours behind it. That is linear in the number of
    flagged voxels, and a healthy prediction flags a few thousand out of several million.
    Returns None above ``_BLOB_VOXEL_CAP`` flagged voxels, where the wait stops being worth
    it and the raw count already tells the story.
    """
    n_true = int(mask.sum())
    if n_true == 0:
        return 0
    if _label is not None:
        labels, n = _label(mask)
        if n < 1:
            return 0
        counts = np.bincount(labels.ravel())
        counts[0] = 0  # label 0 is the background
        return int(counts.max())
    if n_true > _BLOB_VOXEL_CAP:
        return None

    # Flat indices of the flagged voxels, ascending, so a voxel's lower-side neighbours
    # (one slice back, one row back, one column back) have always been seen already.
    idx = np.flatnonzero(mask.ravel())
    shape = mask.shape
    stride_z = int(shape[1] * shape[2]) if len(shape) == 3 else 0
    stride_y = int(shape[-1])
    pos = {int(v): i for i, v in enumerate(idx)}

    parent = list(range(len(idx)))

    def root(a):
        # Path halving: point every node on the way at its grandparent, so repeated
        # lookups on a long chain stay flat without a second pass.
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def join(a, b):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[ra] = rb

    cols = int(shape[-1])
    rows = int(shape[-2])
    for i, v in enumerate(idx):
        v = int(v)
        # Both guards below stop a neighbour from WRAPPING: flat index arithmetic happily
        # walks off the end of a row into the start of the next one, which would fuse two
        # blobs that only touch across the volume's edge and inflate the answer.
        if v % cols and (v - 1) in pos:                 # same row, previous column
            join(i, pos[v - 1])
        if (v // cols) % rows and (v - stride_y) in pos:  # same slice, previous row
            join(i, pos[v - stride_y])
        if stride_z and (v - stride_z) in pos:            # previous slice, same row/column
            join(i, pos[v - stride_z])

    sizes = {}
    for i in range(len(idx)):
        r = root(i)
        sizes[r] = sizes.get(r, 0) + 1
    return int(max(sizes.values()))


def _stats(pred, gt, fg_frac, bright_pct, cold_frac, hot_frac):
    """False-hot / missed-hot numbers for one patient. Returns a dict, or None if unusable.

    The foreground rule and its empty-mask fallback are copied from
    ``image_metrics._foreground_mask`` so this reads the volume the same way the reported
    metrics do.
    """
    gmax = float(gt.max())
    fg = gt > fg_frac * gmax
    if not fg.any():
        fg = np.ones(gt.shape, dtype=bool)
    bright = float(np.percentile(gt[fg], bright_pct))
    if not np.isfinite(bright) or bright <= 0.0:
        return None

    cold_level = cold_frac * bright
    hot_level = hot_frac * bright
    cold_body = fg & (gt < cold_level)
    gt_hot = fg & (gt > hot_level)
    false_hot = cold_body & (pred > hot_level)
    missed_hot = gt_hot & (pred < cold_level)

    n_cold = int(cold_body.sum())
    n_hot = int(gt_hot.sum())
    return {
        "bright": bright,
        "n_cold": n_cold,
        "false_n": int(false_hot.sum()),
        "false_blob": _largest_blob(false_hot),
        "n_hot": n_hot,
        "missed_n": int(missed_hot.sum()),
        # Sums, not means, and in float64: the pooled row re-divides them by the pooled
        # voxel count, and a float32 sum over millions of voxels loses digits.
        "pred_cold_sum": float(pred[cold_body].sum(dtype=np.float64)) if n_cold else 0.0,
        "gt_cold_sum": float(gt[cold_body].sum(dtype=np.float64)) if n_cold else 0.0,
    }


def _pct(num, den):
    """num/den as a percentage; NaN when the denominator is empty."""
    return 100.0 * num / den if den else float("nan")


def _blob_str(v):
    return "-" if v is None else "%d" % v


def _scan(dump_dir, args):
    """Per-patient stats for one dump directory, keyed by patient, in file order."""
    rows = {}
    for key, pred_path, gt_path in _find_pairs(dump_dir):
        if args.max_patients and len(rows) >= args.max_patients:
            break
        pred = np.load(pred_path).astype(np.float32, copy=False)
        gt = np.load(gt_path).astype(np.float32, copy=False)
        if pred.shape != gt.shape:
            print("  skip %s (shape %s vs %s)" % (key, pred.shape, gt.shape))
            continue
        s = _stats(pred, gt, args.fg_frac, args.bright_pct, args.cold_frac, args.hot_frac)
        if s is None:
            print("  skip %s (ground truth is flat/empty, no bright level to compare to)" % (key,))
            continue
        rows[key] = s
    if not rows:
        raise SystemExit("nothing scored in %s" % (dump_dir,))
    return rows


def _report_single(label, rows):
    """Per-patient table plus the pooled row for one directory."""
    print("\n%s -- %d patients" % (label, len(rows)))
    hdr = ("%-26s %7s %10s %11s %9s %8s %9s %12s %10s %10s %9s"
           % ("patient", "bright", "cold_vox", "false_hot%", "false_n", "blob",
              "hot_vox", "missed_hot%", "missed_n", "pred_cold", "gt_cold"))
    print(hdr)
    print("-" * len(hdr))
    for key, s in rows.items():
        print("%-26s %7.4f %10d %11.4f %9d %8s %9d %12.4f %10d %10.4f %9.4f"
              % (_short(key), s["bright"], s["n_cold"], _pct(s["false_n"], s["n_cold"]),
                 s["false_n"], _blob_str(s["false_blob"]), s["n_hot"],
                 _pct(s["missed_n"], s["n_hot"]), s["missed_n"],
                 s["pred_cold_sum"] / s["n_cold"] if s["n_cold"] else float("nan"),
                 s["gt_cold_sum"] / s["n_cold"] if s["n_cold"] else float("nan")))

    p = _pooled(rows)
    print("-" * len(hdr))
    print("%-26s %7.4f %10d %11.4f %9d %8s %9d %12.4f %10d %10.4f %9.4f"
          % ("POOLED (voxel-weighted)", p["bright_mean"], p["n_cold"], p["false_pct"],
             p["false_n"], _blob_str(p["false_blob_max"]), p["n_hot"], p["missed_pct"],
             p["missed_n"], p["pred_cold_mean"], p["gt_cold_mean"]))
    print("%-26s %7s %10s %11.4f %9s %8s %9s %12.4f %10s %10s %9s"
          % ("MEAN over patients", "", "", p["false_pct_mean"], "", "", "",
             p["missed_pct_mean"], "", "", ""))
    print("blob is the LARGEST single connected group of those voxels (the pooled row "
          "shows the worst one).")
    if any(r.get("false_blob") is None for r in rows.values()):
        print("a blob reads '-' where there were too many flagged voxels to group (over "
              "%d): the raw count already says the prediction is badly wrong there."
              % _BLOB_VOXEL_CAP)
    return p


def _pooled(rows):
    """Sum the voxel counts across patients, then divide. See the module docstring."""
    n_cold = sum(s["n_cold"] for s in rows.values())
    n_hot = sum(s["n_hot"] for s in rows.values())
    false_n = sum(s["false_n"] for s in rows.values())
    missed_n = sum(s["missed_n"] for s in rows.values())
    blobs = [s["false_blob"] for s in rows.values() if s["false_blob"] is not None]
    per_false = [_pct(s["false_n"], s["n_cold"]) for s in rows.values()]
    per_missed = [_pct(s["missed_n"], s["n_hot"]) for s in rows.values()]
    return {
        "n_cold": n_cold,
        "n_hot": n_hot,
        "false_n": false_n,
        "missed_n": missed_n,
        "false_pct": _pct(false_n, n_cold),
        "missed_pct": _pct(missed_n, n_hot),
        "false_pct_mean": float(np.nanmean(per_false)) if per_false else float("nan"),
        "missed_pct_mean": float(np.nanmean(per_missed)) if per_missed else float("nan"),
        "false_blob_max": max(blobs) if blobs else None,
        "bright_mean": float(np.mean([s["bright"] for s in rows.values()])),
        "pred_cold_mean": (sum(s["pred_cold_sum"] for s in rows.values()) / n_cold
                           if n_cold else float("nan")),
        "gt_cold_mean": (sum(s["gt_cold_sum"] for s in rows.values()) / n_cold
                         if n_cold else float("nan")),
    }


def _report_compare(label_a, rows_a, label_b, rows_b, pooled_a, pooled_b):
    """Side-by-side per-patient false-hot / missed-hot, with a win/loss count."""
    shared = [k for k in rows_a if k in rows_b]
    if not shared:
        raise SystemExit(
            "no patient appears in both directories (A has %d, B has %d) -- were the two "
            "eval runs given the same --split_json?" % (len(rows_a), len(rows_b)))

    print("\nA = %s" % label_a)
    print("B = %s" % label_b)
    print("paired on %d patients (A had %d, B had %d)" % (len(shared), len(rows_a), len(rows_b)))
    hdr = ("%-26s %11s %11s %10s %9s %9s %12s %12s"
           % ("patient", "A false%", "B false%", "delta", "A blob", "B blob",
              "A missed%", "B missed%"))
    print(hdr)
    print("-" * len(hdr))
    wins = losses = ties = 0
    for k in shared:
        sa, sb = rows_a[k], rows_b[k]
        fa = _pct(sa["false_n"], sa["n_cold"])
        fb = _pct(sb["false_n"], sb["n_cold"])
        if np.isfinite(fa) and np.isfinite(fb):
            if fb < fa:
                wins += 1
            elif fb > fa:
                losses += 1
            else:
                ties += 1
        print("%-26s %11.4f %11.4f %+10.4f %9s %9s %12.4f %12.4f"
              % (_short(k), fa, fb, fb - fa, _blob_str(sa["false_blob"]),
                 _blob_str(sb["false_blob"]), _pct(sa["missed_n"], sa["n_hot"]),
                 _pct(sb["missed_n"], sb["n_hot"])))

    print("-" * len(hdr))
    print("false-hot percentage (invented uptake, LOWER is better): B better on %d "
          "patients, worse on %d, equal on %d" % (wins, losses, ties))
    print("pooled false-hot%%:  A %.4f   B %.4f   (delta %+.4f)"
          % (pooled_a["false_pct"], pooled_b["false_pct"],
             pooled_b["false_pct"] - pooled_a["false_pct"]))
    print("pooled missed-hot%%: A %.4f   B %.4f   (delta %+.4f)"
          % (pooled_a["missed_pct"], pooled_b["missed_pct"],
             pooled_b["missed_pct"] - pooled_a["missed_pct"]))
    print("mean over cold body voxels: A pred %.4f / gt %.4f   B pred %.4f / gt %.4f"
          % (pooled_a["pred_cold_mean"], pooled_a["gt_cold_mean"],
             pooled_b["pred_cold_mean"], pooled_b["gt_cold_mean"]))
    print("\nRead the two percentages together. B is only better if its false-hot went "
          "down, or stayed level while missed-hot went down. False-hot up in exchange for "
          "missed-hot down is a trade, not a gain, and it is the bad direction here.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir_a", nargs="?", default=None,
                    help="Directory of pred/gt .npy pairs (positional form of --a)")
    ap.add_argument("--a", default=None, help="Directory of pred/gt .npy pairs")
    ap.add_argument("--b", default=None,
                    help="Second such directory: compare it against A, patient by patient")
    ap.add_argument("--vs", default=None, help="Alias for --b")
    ap.add_argument("--cold_frac", type=float, default=0.20,
                    help="A voxel is 'cold' below this fraction of the bright level (default 0.20)")
    ap.add_argument("--hot_frac", type=float, default=0.60,
                    help="A voxel is 'hot' above this fraction of the bright level (default 0.60)")
    ap.add_argument("--fg_frac", type=float, default=0.05,
                    help="Foreground is gt above this fraction of gt.max(), the rule "
                         "image_metrics._foreground_mask uses (default 0.05)")
    ap.add_argument("--bright_pct", type=float, default=99.0,
                    help="Percentile of the ground-truth foreground taken as the bright "
                         "level (default 99)")
    ap.add_argument("--max_patients", type=int, default=0,
                    help="Score at most this many patients per directory (0 = all)")
    ap.add_argument("--label_a", default=None)
    ap.add_argument("--label_b", default=None)
    args = ap.parse_args()

    dir_a = args.a or args.dir_a
    dir_b = args.b or args.vs
    if dir_a is None:
        ap.error("give a dump directory, positionally or with --a")
    if not 0.0 < args.cold_frac <= args.hot_frac:
        ap.error("need 0 < --cold_frac <= --hot_frac; got %r and %r"
                 % (args.cold_frac, args.hot_frac))

    print("cold < %.2f x bright, hot > %.2f x bright, foreground = gt > %.2f x gt.max(), "
          "bright = foreground p%.4g" % (args.cold_frac, args.hot_frac, args.fg_frac,
                                         args.bright_pct))
    label_a = args.label_a or dir_a
    rows_a = _scan(dir_a, args)
    pooled_a = _report_single(label_a, rows_a)
    if dir_b is None:
        return
    label_b = args.label_b or dir_b
    rows_b = _scan(dir_b, args)
    pooled_b = _report_single(label_b, rows_b)
    _report_compare(label_a, rows_a, label_b, rows_b, pooled_a, pooled_b)


if __name__ == "__main__":
    sys.exit(main())
