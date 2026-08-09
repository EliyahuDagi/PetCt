"""Compare AE reconstruction fidelity: old (buggy perceptual) vs new (fixed) run.

Reads the BEST val row (lowest recon_l1) from two metrics.jsonl files and prints an
aligned side-by-side table with the delta, so you can see at a glance whether the
perceptual-loss fix actually improved reconstruction. Higher is better for psnr/ssim/
voxel_r2; lower is better for recon_l1/nrmse/mae/rel_bias.

Defaults compare outputs/ae2d (old) vs outputs/ae2d_p (new). Use --old/--new/--tag
for the 3D pair (outputs/ae3d vs outputs/ae3d_p). Stdlib-only / torch-free.

  python scripts/_ae_p_compare.py
  python scripts/_ae_p_compare.py --old outputs/ae3d/metrics.jsonl --new outputs/ae3d_p/metrics.jsonl --tag ae3d
"""

import argparse
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

# (key, higher_is_better) in display order; headline fidelity first.
METRICS = [
    ("recon_l1", False), ("psnr", True), ("ssim", True), ("nrmse", False),
    ("mae", False), ("voxel_r2", True), ("rel_bias", False),
]


def _best_val_row(metrics_path):
    """Return the val row with the lowest recon_l1 (best reconstruction), or None."""
    if not os.path.isfile(metrics_path):
        return None
    best = None
    with open(metrics_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("phase") != "val":
                continue
            rl1 = row.get("recon_l1")
            if rl1 is None:
                continue
            if best is None or rl1 < best.get("recon_l1", float("inf")):
                best = row
    return best


def _fmt(v):
    return "{:.4f}".format(v) if isinstance(v, float) else ("-" if v is None else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=os.path.join(_REPO, "outputs", "ae2d", "metrics.jsonl"))
    ap.add_argument("--new", default=os.path.join(_REPO, "outputs", "ae2d_p", "metrics.jsonl"))
    ap.add_argument("--tag", default="ae2d")
    args = ap.parse_args()

    old = _best_val_row(args.old)
    new = _best_val_row(args.new)
    if old is None:
        print("WARN: no val row in OLD %s" % args.old)
    if new is None:
        print("WARN: no val row in NEW %s (has it finished training?)" % args.new)
    if old is None or new is None:
        return

    header = ["metric", "old", "new", "delta", "better?"]
    rows = [header]
    for key, higher_better in METRICS:
        o, n = old.get(key), new.get(key)
        if not isinstance(o, (int, float)) or not isinstance(n, (int, float)):
            rows.append([key, _fmt(o), _fmt(n), "-", "-"])
            continue
        delta = n - o
        improved = (delta > 0) if higher_better else (delta < 0)
        rows.append([key, _fmt(o), _fmt(n), "{:+.4f}".format(delta),
                     "YES" if improved else "no"])

    widths = [max(len(r[c]) for r in rows) for c in range(len(header))]
    print()
    print("%s recon fidelity -- OLD (buggy perceptual) vs NEW (fixed)  [best val row]" % args.tag)
    print()
    for ri, r in enumerate(rows):
        print("  ".join(cell.ljust(widths[c]) for c, cell in enumerate(r)))
        if ri == 0:
            print("  ".join("-" * widths[c] for c in range(len(header))))
    print()
    n_better = sum(1 for r in rows[1:] if r[4] == "YES")
    print("Improved on %d/%d metrics. Headline: recon_l1/psnr/ssim/voxel_r2." % (n_better, len(METRICS)))


if __name__ == "__main__":
    main()
