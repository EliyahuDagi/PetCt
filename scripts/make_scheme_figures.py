"""Figures for the short docs/PROJECT_REPORT.md.

Four figures, all recomputed from the files under outputs/:

  fig_problem.png       one whole-body coronal slice: NAC, AC and the CT
  fig_scheme.png        the two chains (a drawing, no data)
  fig_slab_curves.png   validation curves of the two native-depth flow runs
  fig_slab_arms.png     test results of the arms on the shared 41 patients

The longer figure set (fig01..fig19) still comes from scripts/make_report_figures.py.

Run from the repository root with the Windows python:
  python scripts/make_scheme_figures.py
"""
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)
os.chdir(ROOT)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 100, "savefig.dpi": 160, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_IN = "#d9ead3"
C_AE = "#cfe2f3"
C_LAT = "#eeeeee"
C_FLOW = "#fce5cd"
C_OUT = "#ead1dc"
EDGE = "#555555"


def box(ax, x, y, w, h, title, sub="", fc=C_LAT, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=fc, ec=EDGE, lw=1.0))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=fs,
                fontweight="bold")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=fs - 1.4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs)


def arrow(ax, p1, p2, text=None, fs=8, color="#333333"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12, lw=1.3, color=color))
    if text:
        ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 0.022, text, ha="center", va="bottom",
                fontsize=fs, color=color)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def summ(e, k):
    s = e["summary"][k]
    return s["mean"] if isinstance(s, dict) else s


def val_rows(run):
    rows = []
    path = os.path.join("outputs", run, "metrics.jsonl")
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if '"phase": "val"' in line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def by_patient(e):
    return {r["patient"]: r for r in e["per_patient"] if "patient" in r}


# ---------------------------------------------------------------------------
# Figure 0: the problem, on one whole-body coronal slice
# ---------------------------------------------------------------------------
# The slices come from scripts/_extract_problem_slices.py, which has to run in WSL (the
# DICOM stack and torch live there, matplotlib lives here). Skipped if it has not run.
SL = os.path.join("outputs", "eval", "fig_problem_slices.npz")
if os.path.exists(SL):
    d = np.load(SL, allow_pickle=True)
    # Row 0 of each array is the foot end, and imshow draws row 0 at the top, so flip.
    ct = np.flipud(d["ct"])
    nac = np.flipud(d["nac"])
    ac = np.flipud(d["ac"])
    # A body window for the CT: bone and lung both visible, in Hounsfield units.
    ctw = np.clip((ct + 500.0) / 1500.0, 0, 1)
    # Each PET panel is shown on its own bright level, because that is what the pipeline
    # does (normalize_volume divides by the volume's own 99th percentile) and because the
    # two volumes are not in the same units anyway. What the picture shows is therefore the
    # PATTERN, not an absolute brightness: where the signal sits inside the body.
    top = 1.0
    print("PROBLEM slice: patient %s, coronal row %s, panels %s, PET scale 0..%.3f"
          % (d["patient"], d["row"], ct.shape, top))
    dz, dy, dx = (float(v) for v in d["pet_spacing"])
    aspect = dz / dx
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 6.4))
    for ax, img, cm, title, sub in [
        (axs[0], nac, "gray", "non-corrected PET (NAC)",
         "what the scanner measures"),
        (axs[1], ac, "gray", "corrected PET (AC)",
         "what we want to predict"),
        (axs[2], ctw, "bone", "CT",
         "how the correction is worked out today"),
    ]:
        vmax = 1.0 if cm == "bone" else top
        ax.imshow(img, cmap=cm, vmin=0, vmax=vmax, aspect=aspect,
                  interpolation="bilinear")
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlabel(sub, fontsize=9, color="#666666")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axs[2].text(0.5, -0.075, "the scan this project does without",
                transform=axs[2].transAxes, ha="center", va="top", fontsize=9,
                color="#b45f06", fontweight="bold")
    fig.suptitle("One patient, one whole-body coronal slice. Each panel is on its own "
                 "brightness scale, so what to read is the pattern:\nin the non-corrected "
                 "image the body outline glows and the organs inside it are washed out.",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "fig_problem.png")
else:
    print("NOTE: no slices for fig_problem.png; run (in WSL)"
          " PYTHONPATH=. python scripts/_extract_problem_slices.py")

# ---------------------------------------------------------------------------
# Figure 1: the two chains
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 5.6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
xs = [0.005, 0.165, 0.335, 0.505, 0.675, 0.845]
W = 0.15
H = 0.20
rows = [
    (0.68, "2D chain", [
        ("NAC volume", "128x128, Z slices"),
        ("encoder, 2D", "each slice on its own"),
        ("code of NAC", "8 x Z x 32 x 32"),
        ("flow bridge, 2D", "16 Euler steps"),
        ("code of AC", "8 x Z x 32 x 32"),
        ("decoder, 2D", "predicted AC volume"),
    ]),
    (0.20, "3D chain", [
        ("NAC volume", "128x128, Z slices"),
        ("encoder, 3D", "sees neighbour slices"),
        ("code of NAC", "8 x Z x 32 x 32"),
        ("flow bridge, 3D", "16 Euler steps,\n32-slice depth windows"),
        ("code of AC", "8 x Z x 32 x 32"),
        ("decoder, 3D", "predicted AC volume"),
    ]),
]
for y, tag, items in rows:
    ax.text(0.0, y + H + 0.055, tag, fontsize=12, fontweight="bold", ha="left")
    for i, (t, s) in enumerate(items):
        fc = [C_IN, C_AE, C_LAT, C_FLOW, C_LAT, C_OUT][i]
        box(ax, xs[i], y, W, H, t, s, fc=fc)
        if i:
            arrow(ax, (xs[i - 1] + W, y + H / 2), (xs[i], y + H / 2))
    ax.text(xs[1] + W / 2, y - 0.045, "frozen", fontsize=8, ha="center", color="#666666",
            style="italic")
    ax.text(xs[5] + W / 2, y - 0.045, "frozen", fontsize=8, ha="center", color="#666666",
            style="italic")
    ax.text(xs[3] + W / 2, y - 0.045, "the only trained part", fontsize=8, ha="center",
            color="#b45f06")
arrow(ax, (xs[1] + W / 2, 0.68), (xs[1] + W / 2, 0.20 + H), color="#3d85c6")
arrow(ax, (xs[3] + W / 2, 0.68), (xs[3] + W / 2, 0.20 + H), color="#3d85c6")
ax.text(0.50, 0.545,
        "the 3D chain starts as the 2D chain: every 2D filter is copied into the centre depth "
        "slice of the\nmatching 3D filter, the depth taps start at zero, and the depth axis is "
        "never made smaller",
        fontsize=9, ha="center", va="center", color="#3d85c6",
        bbox=dict(fc="white", ec="#3d85c6", lw=0.8, boxstyle="round,pad=0.35"))
ax.text(0.50, 0.045,
        "Z = the scan's own number of slices; nothing is resampled along depth. "
        "The code is 4x smaller than the volume in each in-plane direction.",
        fontsize=8.5, ha="center", color="#666666")
save(fig, "fig_scheme.png")

# ---------------------------------------------------------------------------
# Figure 2: validation curves of the two native-depth flow runs
# ---------------------------------------------------------------------------
# One line per experiment. The 3D-autoencoder run was stopped at 20000 steps and
# resumed, so its two files are joined into a single curve: the resumed file's first
# validation is the same checkpoint as the first file's last one (26.8546 dB on both
# sides), so the duplicate step is dropped and the join is exact.
runs = [
    ("3D flow on the 2D autoencoder", "#3d85c6", [("ft3d_slab_flow", 0)]),
    ("3D flow on the 3D autoencoder", "#38761d", [("ft3d_slab_flow_ae3d", 0),
                                                  ("ft3d_slab_flow_ae3d_cont", 20000)]),
]
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2))
panels = [("psnr", "validation PSNR (dB)"),
          ("reg_slope", "validation regression slope"),
          ("hot_band_rel_error", "validation hot-band relative error")]
for lab, col, parts in runs:
    rows, seen = [], set()
    for run, off in parts:
        for r in val_rows(run):
            step = r["step"] + off
            if step in seen:
                continue
            seen.add(step)
            rows.append((step, r))
    rows.sort(key=lambda x: x[0])
    st = np.array([step for step, _ in rows])
    print("CURVE %s: steps %d..%d n=%d psnr %.3f -> %.3f"
          % (lab, st.min(), st.max(), len(st), rows[0][1].get("psnr"), rows[-1][1].get("psnr")))
    for ax, (key, _t) in zip(axs, panels):
        ax.plot(st, [r.get(key, np.nan) for _, r in rows], "-", marker="o", ms=3.2, color=col,
                label=lab if ax is axs[0] else None)
for ax, (key, title) in zip(axs, panels):
    ax.set_xlabel("optimizer step")
    ax.set_title(title, fontsize=10)
axs[0].axhline(26.94, color="#999999", ls=":", lw=1.2)
axs[0].text(20500, 26.60, "the dotted line is the 2D chain\non these same 8 patients",
            fontsize=8, color="#666666", ha="left")
axs[1].axhline(0.8578, color="#999999", ls=":", lw=1.2)
axs[2].axhline(-0.0996, color="#999999", ls=":", lw=1.2)
axs[0].legend(fontsize=8, loc="lower center")
fig.suptitle("The 3D flow on the 3D autoencoder starts behind and ends ahead "
             "(the same 8 held-back patients, whole volumes, native depth)", fontsize=10.5)
fig.tight_layout()
save(fig, "fig_slab_curves.png")

# ---------------------------------------------------------------------------
# Figure 3: test arms on the shared 41 patients
# ---------------------------------------------------------------------------
E = {
    "NAC\nas it is": "outputs/eval/nac_baseline/native_depth_128.json",
    "2D\nchain": "outputs/eval/ft3d_slab_flow/control2d_vol_s16.json",
    "3D chain,\n2D auto-\nencoder": "outputs/eval/ft3d_slab_flow/slab_last_s16.json",
}
# The main experiment is shown once, at the end of its training. Its 20000-step
# reading (slab_ae3d_last_s16.json) is the same run earlier on, not another arm, so
# it is left out of the bars; the training-length comparison is in the paired
# print-out at the bottom of this file.
CONT = "outputs/eval/ft3d_slab_flow/slab_ae3d_cont_final_s16.json"
if os.path.exists(CONT):
    E["3D chain,\n3D auto-\nencoder"] = CONT
else:
    print("NOTE: the continued run is not scored yet:", CONT)
    E["3D chain,\n3D auto-\nencoder"] = "outputs/eval/ft3d_slab_flow/slab_ae3d_last_s16.json"
labels = list(E.keys())
evals = [load(p) for p in E.values()]
ceil = load("outputs/eval/ae_recon_w64/ae3d_slab_ac.json")
print("CEILING (3D autoencoder round trip on AC):",
      {k: round(summ(ceil, k), 4)
       for k in ["psnr", "ssim", "voxel_r2", "reg_slope", "hot_band_rel_error"]})
for lab, e in zip(labels, evals):
    print("ARM", lab.replace("\n", " "), "n=", e.get("n", e.get("n_patients_evaluated")),
          {k: round(summ(e, k), 4)
           for k in ["psnr", "ssim", "nrmse", "mae", "voxel_r2", "reg_slope",
                     "rel_bias", "hot_band_rel_error", "p95_rel_error"]})
fig, axs = plt.subplots(1, 4, figsize=(15, 4.6))
metrics = [("psnr", "PSNR (dB), higher is better", None),
           ("voxel_r2", "variance explained, 1 is perfect", 1.0),
           ("reg_slope", "regression slope, 1 is perfect", 1.0),
           ("hot_band_rel_error", "hot-band relative error, 0 is perfect", 0.0)]
cols = ["#cccccc", "#9fc5e8", "#3d85c6", "#38761d"][:len(labels)]
for ax, (key, title, ideal) in zip(axs, metrics):
    vals = [summ(e, key) for e in evals]
    ax.bar(np.arange(len(vals)), vals, color=cols, edgecolor="#333")
    for i, v in enumerate(vals):
        ax.text(i, v, ("%.2f" % v) if key == "psnr" else ("%.3f" % v), ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)
    if key in ceil["summary"]:
        c = summ(ceil, key)
        ax.axhline(c, color="#674ea7", ls="--", lw=1.1)
        if abs(c - (ideal if ideal is not None else c)) > 0.02 or ideal is None:
            ax.text(len(vals) - 0.45, c, "ceiling %.2f" % c if key == "psnr" else "ceiling %.3f" % c,
                    fontsize=7, color="#674ea7", ha="right", va="bottom")
    if ideal is not None and abs(summ(ceil, key) - ideal) > 0.02:
        ax.axhline(ideal, color="k", lw=0.8, ls=":")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_title(title, fontsize=9.5)
axs[0].text(-0.45, 43.0, "the purple line is the ceiling of the chain:\nencode and decode the true "
            "AC, no flow model", fontsize=7.5, color="#674ea7", ha="left", va="top")
fig.suptitle("The 41 held-out patients, scored the same way: 128x128 in-plane, every native "
             "slice, 16 Euler steps", fontsize=10.5)
fig.tight_layout()
save(fig, "fig_slab_arms.png")

# ---------------------------------------------------------------------------
# paired win counts, printed for the report text
# ---------------------------------------------------------------------------
P2D = "outputs/eval/ft3d_slab_flow/control2d_vol_s16.json"
PB0 = "outputs/eval/ft3d_slab_flow/slab_last_s16.json"
PB1 = "outputs/eval/ft3d_slab_flow/slab_ae3d_last_s16.json"
pairs = [("2D chain", P2D, "3D chain, 2D AE", PB0),
         ("2D chain", P2D, "3D chain, 3D AE", PB1),
         ("3D chain, 2D AE", PB0, "3D chain, 3D AE", PB1)]
if os.path.exists(CONT):
    pairs += [("2D chain", P2D, "3D chain, 3D AE, longer", CONT),
              ("3D chain, 2D AE", PB0, "3D chain, 3D AE, longer", CONT),
              ("3D chain, 3D AE", PB1, "3D chain, 3D AE, longer", CONT)]
IDEAL = {"reg_slope": 1.0, "hot_band_rel_error": 0.0, "voxel_r2": 1.0, "rel_bias": 0.0,
         "p95_rel_error": 0.0, "nrmse": 0.0, "mae": 0.0}
for la, pa, lb, pb in pairs:
    A, B = by_patient(load(pa)), by_patient(load(pb))
    common = sorted(set(A) & set(B))
    print("PAIRED %s vs %s on %d patients:" % (lb, la, len(common)))
    for k in ["psnr", "ssim", "nrmse", "mae", "voxel_r2", "reg_slope", "rel_bias",
              "hot_band_rel_error", "p95_rel_error"]:
        a = np.array([A[p][k] for p in common])
        b = np.array([B[p][k] for p in common])
        d = b - a
        if k in IDEAL:
            better = np.abs(b - IDEAL[k]) < np.abs(a - IDEAL[k])
            # Significance goes on the DISTANCE-FROM-IDEAL scale, the same as
            # scripts/_cmp_eval_paired.py, so the two never disagree. The raw difference
            # is the wrong scale for the error metrics: they are negative here, so a
            # patient going from -1% to +4% would score as a +5-point "improvement"
            # when it actually got worse. Sign flipped so positive still means B better.
            gap = np.abs(b - IDEAL[k]) - np.abs(a - IDEAL[k])
        else:
            better = d > 0
            gap = -d
        sd = gap.std(ddof=1)
        t = -gap.mean() / (sd / np.sqrt(len(gap))) if sd > 0 else np.nan
        print("   %-20s %.4f -> %.4f  mean %+.4f  t %+.2f  B better on %d/%d"
              % (k, a.mean(), b.mean(), d.mean(), t, int(better.sum()), len(d)))
print("DONE")
