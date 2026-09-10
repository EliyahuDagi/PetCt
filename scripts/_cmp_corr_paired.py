"""PAIR TWO ARMS ON THE TWO FACTORS OF reg_slope, PATIENT BY PATIENT.

``reg_slope`` is not one quantity, it is a product of two:

    reg_slope = r * (sd_pred / sd_gt)

and the two mean completely different things. Raising ``r`` means the prediction now agrees
with the truth about where the uptake is -- real, and hard. Raising ``sd_pred/sd_gt`` means
the prediction is simply more spread out than before, which lifts the slope metric whether
or not the pattern improved. Measured on the finished chain, a stretch big enough to force
the slope to 1.0 costs 0.43 dB of PSNR and 0.028 of variance explained and buys nothing
real (scripts/_diag_corr_ceiling.py). So an arm's slope gain has to be split before it can
be believed.

This takes two ``--json_out`` files from that script and reports, on the shared patients:

  * the paired change in ``r`` and in ``sd_ratio``, each with its own t;
  * the change in the product, and how much of it each factor contributed. The split uses
    ``d(slope) = r*d(sd_ratio) + sd_ratio*d(r)`` evaluated at arm A, which accounts for the
    total to within the second-order term (negligible at these sizes).

A gain that is mostly the ``sd_ratio`` term is the cosmetic kind, no matter how significant
its t is.

    ~/petct/.venv/bin/python scripts/_cmp_corr_paired.py a.json b.json --label-a x --label-b y
"""

import argparse
import json

import numpy as np


def paired_t(a, b):
    """t for the mean of b-a, and the win/loss count. Positive t means b is larger."""
    d = b - a
    sd = d.std(ddof=1)
    t = float(d.mean() / (sd / np.sqrt(len(d)))) if sd > 0 else float("nan")
    return d.mean(), t, int(np.sum(d > 0)), int(np.sum(d < 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    args = ap.parse_args()

    A = json.load(open(args.a))
    B = json.load(open(args.b))
    la = args.label_a or A.get("label", "A")
    lb = args.label_b or B.get("label", "B")

    keys = sorted(set(A["patients"]) & set(B["patients"]))
    if not keys:
        print("no shared patients between the two files")
        return 2
    print("A = %s\nB = %s" % (la, lb))
    print("paired on %d shared patients (A had %d, B had %d)"
          % (len(keys), len(A["patients"]), len(B["patients"])))

    ra = np.array([A["patients"][k]["r"] for k in keys])
    rb = np.array([B["patients"][k]["r"] for k in keys])
    sa = np.array([A["patients"][k]["sd_ratio"] for k in keys])
    sb = np.array([B["patients"][k]["sd_ratio"] for k in keys])

    print()
    print("%-22s %10s %10s %10s %7s %10s" % ("", "A mean", "B mean", "delta", "t", "win/loss"))
    print("-" * 76)
    for name, xa, xb in (("r (agreement)", ra, rb),
                         ("sd_ratio (spread)", sa, sb),
                         ("product = reg_slope", ra * sa, rb * sb)):
        d, t, w, l = paired_t(xa, xb)
        print("%-22s %10.4f %10.4f %+10.4f %7.2f %6d/%-3d"
              % (name, xa.mean(), xb.mean(), d, t, w, l))

    # Attribute the slope change to its two causes.
    d_slope = float((rb * sb - ra * sa).mean())
    from_spread = float((ra * (sb - sa)).mean())
    from_agree = float((sa * (rb - ra)).mean())
    print()
    print("Where the slope change came from:")
    print("  total                     %+.4f" % d_slope)
    print("  from spread (cosmetic)    %+.4f   (%s of it)"
          % (from_spread, "%.0f%%" % (100.0 * from_spread / d_slope) if d_slope else "n/a"))
    print("  from agreement (real)     %+.4f   (%s of it)"
          % (from_agree, "%.0f%%" % (100.0 * from_agree / d_slope) if d_slope else "n/a"))
    print()
    print("A slope gain carried by the spread term is the kind a constant multiply would")
    print("have produced for free, and it costs accuracy. Only the agreement term is a")
    print("better prediction. Judge calibration arms on that line.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
