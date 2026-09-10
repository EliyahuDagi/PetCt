"""HOW MUCH OF THE FLATNESS IS SCALE, AND HOW MUCH IS SHAPE?

The finished chain predicts ``pred ~= 0.90*gt + 0.038``: the prediction is about 10% less
spread out than the truth. Two very different things could cause that, and they call for
opposite fixes:

  * SCALE. The prediction has the right pattern but is squashed toward the middle. Then the
    cure is a multiplication, needs no retraining at all, and an objective that stops
    caring about scale (a correlation loss) would lose nothing by dropping it.
  * SHAPE. The prediction disagrees with the truth about WHERE the uptake is, and the
    squashing is the least-squares response to that disagreement. Then no multiplication
    can fix it -- inflating a wrong pattern makes the error worse, not better -- and a
    correlation loss is attacking the real problem but has no easy win available.

Squared error settles which one you get. Write the error over the foreground as

    MSE = (mean offset)^2 + (sd_pred - sd_gt)^2 + 2*sd_pred*sd_gt*(1 - r)

with ``r`` the correlation. Minimising that over ``sd_pred`` alone gives

    sd_pred = r * sd_gt          and therefore     reg_slope = r^2

So squared error does not merely permit flatness, it prescribes exactly how much: the
prediction should be flattened by precisely the amount the model is uncertain. A model
sitting at ``sd_pred/sd_gt = r`` has nothing left to gain from squared error and is not
"under-trained" -- it is finished. Anything further has to come from a different objective.

This script measures ``r`` and ``sd_pred/sd_gt`` per patient, checks them against that
prediction, and then rescales the saved predictions four ways to price the alternatives:

  as_is             what was scored.
  varmatch_oracle   stretch each patient so sd_pred = sd_gt. Needs the truth, so it is a
                    CEILING, not a method. Gives reg_slope = r.
  slope1_oracle     stretch further, until reg_slope = 1 exactly (a factor 1/r^2). The
                    "make the metric read 1.0" option, and the one most likely to overshoot
                    into reading high.
  varmatch_global   ONE factor for every patient, the median of the oracle factors, applied
                    around each patient's own mean. Uses nothing from the truth at test
                    time, so unlike the two above this is deployable as-is.

Every variant is clamped back to [0,1] afterwards, as the evaluator does, and scored with
the repository's own metric functions so the numbers line up with docs/PROJECT_REPORT.md.

    ~/petct/.venv/bin/python scripts/_diag_corr_ceiling.py --dump outputs/eval/dump_ae3d_cont_final

CPU only -- safe to run while a training arm has the card.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.utils.image_metrics import (  # noqa: E402
    _foreground_mask,
    image_quality_metrics,
)

VARIANTS = ("as_is", "varmatch_oracle", "slope1_oracle", "varmatch_global")
SHOW = ("psnr", "ssim", "voxel_r2", "reg_slope", "reg_intercept",
        "rel_bias", "hot_band_rel_error", "p95_rel_error")


def pairs(dump):
    """(key, pred path, gt path) for every patient in a dump directory."""
    out = []
    for name in sorted(os.listdir(dump)):
        if not name.endswith("_pred.npy"):
            continue
        key = name[: -len("_pred.npy")]
        gt = os.path.join(dump, key + "_gt.npy")
        if os.path.isfile(gt):
            out.append((key, os.path.join(dump, name), gt))
    return out


def fg_stats(pred, gt):
    """Correlation and spread of pred vs gt over gt's foreground.

    The mask is the metrics' own ``gt > 5% of gt.max()`` rule, so ``r`` and the spread
    ratio are measured on exactly the voxels ``reg_slope`` is measured on.
    """
    mask = _foreground_mask(gt, None)
    p = pred[mask].double()
    g = gt[mask].double()
    pm, gm = p.mean(), g.mean()
    # Population spread (not the n-1 form): these are whole populations of voxels, and the
    # correction would be identical in both terms of the ratio anyway.
    sp = torch.sqrt(((p - pm) ** 2).mean())
    sg = torch.sqrt(((g - gm) ** 2).mean())
    r = float((((p - pm) * (g - gm)).mean()) / (sp * sg))
    return {"r": r, "sd_pred": float(sp), "sd_gt": float(sg),
            "sd_ratio": float(sp / sg), "mean_pred": float(pm), "mean_gt": float(gm)}


def rescale(pred, factor, centre):
    """Stretch the prediction about ``centre`` and clamp, as the evaluator would.

    CAVEAT on reading ssim for a factor BELOW 1. Stretching about the foreground mean holds
    the average fixed, which is what a recalibration should do, but it lifts air off zero:
    an air voxel becomes ``centre * (1 - factor)``, which for factor 0.88 and centre ~0.35
    is a visible grey haze over the whole background. Clamping at 0 cannot undo a POSITIVE
    lift, so ssim drops for reasons that have nothing to do with calibration. It only
    affects variants whose factor is under 1 (here: varmatch_oracle, on the patients whose
    prediction is already more spread than the truth). The factors above 1 push air
    negative, which the clamp does absorb, so slope1_oracle and varmatch_global are clean.
    """
    return ((pred - centre) * factor + centre).clamp(0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="outputs/eval/dump_ae3d_cont_final")
    ap.add_argument("--label", default=None)
    ap.add_argument("--max_patients", type=int, default=0, help="0 = all")
    ap.add_argument("--json_out", default=None,
                    help="write the per-patient r / sd_ratio here, so two arms can be "
                         "paired patient-by-patient with scripts/_cmp_corr_paired.py. The "
                         "means alone cannot tell a consistent gain from a noisy one.")
    ap.add_argument("--stats_only", action="store_true",
                    help="skip the rescale variants (which cost the metric passes) and "
                         "report only r and the spread ratio.")
    args = ap.parse_args()

    label = args.label or os.path.basename(args.dump.rstrip("/\\"))
    items = pairs(args.dump)
    if args.max_patients:
        items = items[: args.max_patients]
    if not items:
        print("no pred/gt pairs under %s" % args.dump)
        return 2
    print("%s: %d patients" % (label, len(items)))

    # Pass one: the per-patient statistics, and the global factor the deployable variant
    # needs. It has to be known before any patient is rescaled, hence two passes.
    stats = {}
    for key, pp, gp in items:
        pred = torch.from_numpy(np.load(pp)).float()
        gt = torch.from_numpy(np.load(gp)).float()
        stats[key] = fg_stats(pred, gt)
    k_global = float(np.median([1.0 / s["sd_ratio"] for s in stats.values()]))

    print()
    print("Is the model sitting where squared error tells it to?")
    print("  Squared error's own optimum is sd_pred/sd_gt = r, so slope should equal r^2.")
    print("  %-14s %8s %8s %8s %8s" % ("", "r", "sd_ratio", "r^2", "gap"))
    rs = np.array([s["r"] for s in stats.values()])
    sr = np.array([s["sd_ratio"] for s in stats.values()])
    print("  %-14s %8.4f %8.4f %8.4f %+8.4f"
          % ("mean", rs.mean(), sr.mean(), (rs ** 2).mean(), (sr - rs).mean()))
    print("  %-14s %8.4f %8.4f %8.4f %+8.4f"
          % ("median", np.median(rs), np.median(sr), np.median(rs ** 2),
             np.median(sr - rs)))
    print("  patients with sd_ratio within 0.02 of r: %d of %d"
          % (int(np.sum(np.abs(sr - rs) <= 0.02)), len(rs)))
    print()
    print("  spread of the two, across patients (a mean hides a stretch factor that is")
    print("  wild on one patient and 1.0 on the rest):")
    print("  %-14s %8s %8s %8s %8s %8s" % ("", "min", "p25", "median", "p75", "max"))
    for nm, arr in (("r", rs), ("sd_ratio", sr)):
        qs = np.percentile(arr, [0, 25, 50, 75, 100])
        print("  %-14s %8.4f %8.4f %8.4f %8.4f %8.4f" % (nm, *qs))
    wild = [(k, s["sd_ratio"]) for k, s in stats.items()
            if abs(1.0 / s["sd_ratio"] - 1.0) > 0.10]
    print("  patients needing a stretch beyond +/-10%%: %d of %d%s"
          % (len(wild), len(rs),
             "".join("\n    ~%s sd_ratio=%.3f factor=%.3f" % (k[-18:], v, 1.0 / v)
                     for k, v in sorted(wild, key=lambda t: t[1]))))
    print()
    print("  global variance-match factor (median 1/sd_ratio, the deployable one): %.4f"
          % k_global)

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"label": label, "dump": args.dump, "patients": stats}, fh, indent=1)
        print("  per-patient statistics -> %s" % args.json_out)
    if args.stats_only:
        return 0

    # Pass two: score every variant.
    acc = {v: [] for v in VARIANTS}
    for key, pp, gp in items:
        pred = torch.from_numpy(np.load(pp)).float()
        gt = torch.from_numpy(np.load(gp)).float()
        s = stats[key]
        variants = {
            "as_is": pred,
            # sd_pred -> sd_gt. Stretch about the prediction's own foreground mean, which is
            # knowable at test time; only the FACTOR is an oracle here.
            "varmatch_oracle": rescale(pred, 1.0 / s["sd_ratio"], s["mean_pred"]),
            # reg_slope = r * sd_pred/sd_gt, so reaching 1 needs sd_pred -> sd_gt / r.
            "slope1_oracle": rescale(pred, 1.0 / (s["sd_ratio"] * s["r"]), s["mean_pred"]),
            "varmatch_global": rescale(pred, k_global, s["mean_pred"]),
        }
        # image_quality_metrics routes through ssim, which insists on (N,C,...); the saved
        # dumps are bare volumes. Add the two leading dims for both sides.
        gt5 = gt[None, None]
        for name, vol in variants.items():
            acc[name].append(image_quality_metrics(vol[None, None], gt5))

    print()
    print("%-17s %s" % ("variant", " ".join(k[:10].rjust(11) for k in SHOW)))
    base = None
    for v in VARIANTS:
        m = {k: float(np.mean([row[k] for row in acc[v]])) for k in SHOW}
        if base is None:
            base = m
        print("%-17s %s" % (v, " ".join("%11.4f" % m[k] for k in SHOW)))
    print()
    print("%-17s %s" % ("delta vs as_is", " ".join(k[:10].rjust(11) for k in SHOW)))
    for v in VARIANTS[1:]:
        m = {k: float(np.mean([row[k] for row in acc[v]])) for k in SHOW}
        print("%-17s %s" % (v, " ".join("%+11.4f" % (m[k] - base[k]) for k in SHOW)))

    print()
    print("Reading it: a variant that lifts reg_slope while holding psnr and voxel_r2 says")
    print("the flatness was SCALE and a multiplication recovers it. One that lifts reg_slope")
    print("while psnr falls says the flatness was buying accuracy, i.e. it is SHAPE, and the")
    print("only real fix is a prediction that correlates better. Watch rel_bias and")
    print("p95_rel_error for the overshoot: they turn positive when the cure starts reading")
    print("high, which in this domain means invented uptake.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
