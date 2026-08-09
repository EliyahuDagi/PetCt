"""Paired per-patient comparison of two eval JSONs.

Why paired: the expected effect of the flow loss-shaping / perceptual arms is ~1% (the 2D
VGG precedent moved best rollout val L1 from 0.01251 to 0.01236), while the BETWEEN-patient
SD on the test split is large (SSIM 0.031 over n=41 -> SEM ~0.005). Comparing means would
drown a 1% effect in patient heterogeneity.

Both arms are evaluated on the SAME patients with the SAME frozen AE and split, so the
per-patient DIFFERENCE cancels almost all of that heterogeneity. This reports, per metric:

  * mean paired delta (b - a) with its SEM and a paired t statistic,
  * a Wilcoxon-style sign count (wins/losses), which makes no normality assumption,
  * whether the delta is an improvement, given each metric's orientation.

Usage:
  python scripts/_cmp_eval_paired.py A.json B.json [--label-a NAME --label-b NAME]

Orientation note: for signed-error metrics (rel_bias, p95_rel_error, hot_band_rel_error,
ba_mean_bias, reg_intercept) ZERO is best, so "better" means smaller |value|; for
reg_slope 1.0 is best. max_rel_error is deliberately excluded from the verdict -- it is
AE-limited and ~0 under the default clamp (see docs/nac_ac_benchmark.md).
"""

import argparse
import json
import math
from pathlib import Path

HIGHER_BETTER = {"ssim", "psnr", "voxel_r2"}
LOWER_BETTER = {"nrmse", "mae"}
ZERO_BEST = {"rel_bias", "p95_rel_error", "hot_band_rel_error", "ba_mean_bias",
             "reg_intercept"}
ONE_BEST = {"reg_slope"}
# Reported but never used for the verdict: AE-limited and ~0 under the default clamp.
INFORMATIONAL = {"max_rel_error"}

ORDER = ["ssim", "psnr", "nrmse", "mae", "voxel_r2", "reg_slope", "reg_intercept",
         "rel_bias", "p95_rel_error", "hot_band_rel_error", "max_rel_error"]


def _per_patient(path):
    """Return ``({key: row}, meta, keyed_by_id)``.

    Prefers the explicit ``patient`` identifier. Eval JSONs written before that field
    existed have none, so fall back to positional keys -- valid ONLY because
    ``_resolve_candidates`` yields a deterministic order from the same ``split.json``, and
    the caller verifies the two runs used the same split/mode and produced equal counts.
    """
    d = json.load(open(path, encoding="utf-8"))
    rows = d.get("per_patient", [])
    ids = [r.get("patient") or r.get("path") or r.get("id") for r in rows]
    if rows and all(i is not None for i in ids):
        return {str(i): r for i, r in zip(ids, rows)}, d, True
    return {f"#{i}": r for i, r in enumerate(rows)}, d, False


def _score(metric, v):
    """Distance-from-ideal, so LOWER is always better."""
    if metric in HIGHER_BETTER:
        return -v
    if metric in LOWER_BETTER:
        return v
    if metric in ZERO_BEST:
        return abs(v)
    if metric in ONE_BEST:
        return abs(v - 1.0)
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--label-a", default=None)
    p.add_argument("--label-b", default=None)
    args = p.parse_args()

    pa, da, id_a = _per_patient(args.a)
    pb, db, id_b = _per_patient(args.b)
    la = args.label_a or Path(args.a).parent.name + "/" + Path(args.a).stem
    lb = args.label_b or Path(args.b).parent.name + "/" + Path(args.b).stem

    keyed_by_id = id_a and id_b
    if not keyed_by_id:
        # Re-key BOTH sides positionally. If only one side has ids its keys are paths while
        # the other's are "#i", so the intersection would be empty and the run would look
        # unpairable when it is merely mixed-vintage.
        pa = {f"#{i}": r for i, r in enumerate(pa.values())}
        pb = {f"#{i}": r for i, r in enumerate(pb.values())}
        # Positional pairing: only sound if both runs walked the same ordered split.
        if len(pa) != len(pb):
            raise SystemExit(
                f"Cannot pair: no patient IDs in at least one JSON and the counts differ "
                f"({len(pa)} vs {len(pb)}). Re-run eval so 'patient' is recorded.")
        sa, sb = da.get("split_json"), db.get("split_json")
        ma, mb = da.get("mode"), db.get("mode")
        if (sa, ma) != (sb, mb):
            raise SystemExit(
                f"Cannot pair positionally: split/mode differ ({sa!r},{ma!r}) vs "
                f"({sb!r},{mb!r}). Re-run eval so 'patient' is recorded.")
        print("WARNING: no 'patient' field -- pairing POSITIONALLY (same split.json and "
              "mode verified, equal counts). Newer eval runs record IDs.")

    shared = sorted(set(pa) & set(pb))
    if not shared:
        raise SystemExit(
            "No shared patients between the two eval JSONs -- a paired test is impossible. "
            f"(A has {len(pa)}, B has {len(pb)}.) Were they run on the same --split_json?"
        )
    print(f"A = {la}")
    print(f"B = {lb}")
    print(f"paired on {len(shared)} shared patients "
          f"(A had {len(pa)}, B had {len(pb)}); keyed by "
          f"{'patient id' if keyed_by_id else 'position'}")
    for tag, d in (("A", da), ("B", db)):
        print(f"  {tag}: steps={d.get('ddim_steps')} clamp_output={d.get('clamp_output')} "
              f"ckpt={d.get('diff_ckpt')}")
    print()

    hdr = (f"{'metric':<20}{'A mean':>10}{'B mean':>10}{'delta':>10}"
           f"{'SEM':>9}{'t':>7}{'win/loss':>10}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for m in ORDER:
        va = [pa[k][m] for k in shared if isinstance(pa[k].get(m), (int, float))
              and math.isfinite(pa[k][m]) and isinstance(pb[k].get(m), (int, float))
              and math.isfinite(pb[k][m])]
        vb = [pb[k][m] for k in shared if isinstance(pa[k].get(m), (int, float))
              and math.isfinite(pa[k][m]) and isinstance(pb[k].get(m), (int, float))
              and math.isfinite(pb[k][m])]
        if len(va) < 3:
            continue
        n = len(va)
        ma, mb = sum(va) / n, sum(vb) / n
        # Paired deltas on the distance-from-ideal scale (negative = B better).
        d = [_score(m, y) - _score(m, x) for x, y in zip(va, vb)]
        md = sum(d) / n
        var = sum((x - md) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
        sem = math.sqrt(var / n) if var > 0 else 0.0
        t = md / sem if sem > 0 else float("nan")
        wins = sum(1 for x in d if x < 0)
        losses = sum(1 for x in d if x > 0)
        if m in INFORMATIONAL:
            verdict = "(informational: AE-limited)"
        elif wins == 0 and losses == 0:
            # Every paired delta is exactly zero -- the metric is invariant to whatever
            # differs between the arms (e.g. p95_rel_error is unaffected by the [0,1]
            # clamp, since the 95th percentile sits below 1.0). Not "no effect": no change.
            verdict = "identical (metric invariant to this change)"
        elif not math.isfinite(t) or abs(t) < 2.0:
            verdict = "no significant change"
        else:
            verdict = "B BETTER" if md < 0 else "B WORSE"
        print(f"{m:<20}{ma:>10.4f}{mb:>10.4f}{mb - ma:>+10.4f}"
              f"{sem:>9.4f}{t:>7.2f}{f'{wins}/{losses}':>10}  {verdict}")

    print("\nRead |t| >= 2 as the rough significance bar (n~41, two-sided ~0.05).")
    print("delta is B-A in the raw metric; t is on the distance-from-ideal scale, so a")
    print("NEGATIVE t always means B is closer to ideal. Judge the arms on psnr /")
    print("voxel_r2 / reg_slope / hot_band_rel_error -- NOT on max_rel_error.")


if __name__ == "__main__":
    main()
