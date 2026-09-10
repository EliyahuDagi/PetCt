"""Generate the figures for docs/PROJECT_REPORT_DETAIL.md into docs/figures/.

Run from the repository root with the Windows python (matplotlib + numpy present).
"""
import json, os, glob, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import image as mpimg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)
os.chdir(ROOT)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 100, "savefig.dpi": 160, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_DATA = "#d9ead3"; C_AE = "#cfe2f3"; C_DIFF = "#fce5cd"; C_OUT = "#ead1dc"; C_GUI = "#fff2cc"; C_GREY = "#eeeeee"
EDGE = "#555555"

def box(ax, x, y, w, h, text, fc=C_GREY, fs=9, bold_first=False, ec=EDGE, lw=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.015", fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    if bold_first and "\n" in text:
        first, rest = text.split("\n", 1)
        ax.text(x + w / 2, y + h - 0.035, first, ha="center", va="top", fontsize=fs, fontweight="bold")
        ax.text(x + w / 2, y + (h - 0.09) / 2, rest.strip("\n"), ha="center", va="center", fontsize=fs - 1)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
    return p

def arrow(ax, p1, p2, text=None, ls="-", color="#333333", fs=8, rad=0.0, tpos=0.5, toff=(0, 0.03)):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12, lw=1.2, color=color, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if text:
        tx = p1[0] + (p2[0] - p1[0]) * tpos + toff[0]
        ty = p1[1] + (p2[1] - p1[1]) * tpos + toff[1]
        ax.text(tx, ty, text, ha="center", va="bottom", fontsize=fs, color=color,
                bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.85))

def blank(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)

# ----------------------------------------------------------------------------------------
# Fig 01: system overview
# ----------------------------------------------------------------------------------------
fig, ax = blank((11, 4.4))
ax.text(0.5, 0.98, "The three parts of the repository", ha="center", va="top", fontsize=12, fontweight="bold")
box(ax, 0.02, 0.30, 0.29, 0.58,
    "DataViewer  (src/DataViewer/)\n\nDesktop viewer (Tkinter).\nShows CT, attenuation-corrected PET and\nnon-corrected PET side by side, with\nsegmentation overlays and the prostate\n'zone of interest' box.\nModel / Presenter / View pattern.",
    fc=C_GUI, fs=8.5, bold_first=True)
box(ax, 0.355, 0.30, 0.29, 0.58,
    "Core algorithms  (src/utils/)\n\nVolumeGeometry: voxel <-> millimetre\nmapping so PET and CT grids can be\ncompared.\nProstate locator (rule based, uses CT\ndensity and PET uptake).\nSegmentation helpers (TotalSegmentator,\nMONAI whole-body bundle).",
    fc=C_GREY, fs=8.5, bold_first=True)
box(ax, 0.69, 0.30, 0.29, 0.58,
    "Training pipeline  (src/training/)\n\nLearns to predict attenuation-corrected\nPET from non-corrected PET, without CT.\nTwo autoencoders + two flow models,\ndata loading and caching, evaluation,\ninference, 306 unit tests.\nTrainViewer (src/TrainViewer/): web GUI\nthat launches training and shows results.",
    fc=C_DIFF, fs=8.5, bold_first=True)
arrow(ax, (0.31, 0.60), (0.355, 0.60), "uses")
arrow(ax, (0.69, 0.50), (0.645, 0.50), "reads DICOM\nthrough", tpos=0.5, toff=(0, 0.02))
box(ax, 0.02, 0.05, 0.96, 0.16,
    "Shared data: DICOM PET/CT studies on disk (10 folders, 628 patients, of which 410 have both PET versions)\n->  float16 cache on the SSD  ->  outputs/<run>/ (metrics, checkpoints, evaluation JSON, HTML reports)",
    fc=C_DATA, fs=8.5)
save(fig, "fig01_system_overview.png")

# ----------------------------------------------------------------------------------------
# Fig 02: training chain
# ----------------------------------------------------------------------------------------
fig, ax = blank((13, 6.4))
ax.text(0.5, 0.985, "The four training stages and how they feed each other", ha="center", va="top", fontsize=12, fontweight="bold")
box(ax, 0.01, 0.38, 0.15, 0.34,
    "Data\n\n10 DICOM folders\n628 patients with any PET\n410 patients with the\nNAC + AC pair\n\nEach volume normalised\nto [0,1] by its own 1st and\n99th percentile, stored as\nfloat16 on the SSD",
    fc=C_DATA, fs=8, bold_first=True)
box(ax, 0.20, 0.60, 0.23, 0.33,
    "Stage 1: 2D autoencoder (ae2d_p)\n\nInput: 128 x 128 PET slices cut from\nall three planes, 628 patients\n(NAC or AC, unpaired is fine)\nLatent: 32 x 32 x 8 channels\nLoss: L1 + 1e-6 KL + 0.1 perceptual\n100k steps, batch 16, 12.3 M params",
    fc=C_AE, fs=8, bold_first=True)
box(ax, 0.20, 0.08, 0.23, 0.33,
    "Stage 3: 2D flow model (diff2d_flow_perc_p)\n\nInput: pairs of 2D-AE latents\n(NAC latent -> AC latent)\n410 paired patients\nU-Net 64/128/256 channels\nPredicts the velocity NAC - AC\n100k steps, batch 16, 11.1 M params",
    fc=C_DIFF, fs=8, bold_first=True)
box(ax, 0.50, 0.60, 0.23, 0.33,
    "Stage 2: 3D autoencoder (ae3d_p)\n\nInput: whole volume resized to 64^3\n628 patients\nLatent: 16^3 x 8 channels\nSame loss as stage 1\n50k steps, batch 1, 35.7 M params\nWeights start from stage 1 (inflated)",
    fc=C_AE, fs=8, bold_first=True)
box(ax, 0.50, 0.08, 0.23, 0.33,
    "Stage 4: 3D flow model (ft3d_flow_p)\n\nInput: pairs of 3D-AE latents\n(16^3 x 8 each), 287 train patients\nSame U-Net, 3D convolutions\n30k steps, batch 1 x 16 accumulated\n29.0 M params\nWeights start from stage 3 (inflated)",
    fc=C_DIFF, fs=8, bold_first=True)
box(ax, 0.79, 0.33, 0.20, 0.38,
    "Deliverables\n\nbest.pt / last.pt per stage\n(config embedded)\nmetrics.jsonl per stage\nevaluate.py -> per-patient JSON\n(41 test patients)\ninfer.py -> pred/gt/nac .npy\nreport_triplets.py -> HTML\nTrainViewer GUI",
    fc=C_OUT, fs=8, bold_first=True)
arrow(ax, (0.16, 0.62), (0.20, 0.76), "slices")
arrow(ax, (0.16, 0.46), (0.20, 0.25), "paired\nlatents", toff=(-0.03, 0.0))
arrow(ax, (0.43, 0.76), (0.50, 0.76), "inflate 2D -> 3D\n(centre tap)", toff=(0, 0.03))
arrow(ax, (0.43, 0.25), (0.50, 0.25), "inflate 2D -> 3D\n(centre tap)", toff=(0, 0.03))
arrow(ax, (0.315, 0.60), (0.315, 0.41), "frozen encoder / decoder", ls="--", color="#1f5f8b", toff=(0.0, 0.0), tpos=0.5)
arrow(ax, (0.615, 0.60), (0.615, 0.41), "frozen encoder / decoder", ls="--", color="#1f5f8b", toff=(0.0, 0.0), tpos=0.5)
arrow(ax, (0.73, 0.28), (0.79, 0.42))
arrow(ax, (0.73, 0.72), (0.79, 0.62))
ax.text(0.5, 0.005, "Blue: autoencoders (compress images to a small latent grid).  Orange: flow models (learn the NAC -> AC change inside the latent grid).\nDashed: the autoencoder is frozen while the flow model trains. 'Inflate' copies each 2D convolution kernel into the centre slice of a 3D kernel.",
        ha="center", va="bottom", fontsize=8.5, color="#333333")
save(fig, "fig02_training_chain.png")

# ----------------------------------------------------------------------------------------
# Fig 03: autoencoder architecture
# ----------------------------------------------------------------------------------------
fig, ax = blank((12, 4.8))
ax.text(0.5, 0.985, "Autoencoder: how a PET volume is compressed to the latent grid and back", ha="center", va="top", fontsize=12, fontweight="bold")
def stack(ax, x0, y_base, sizes, chans, labels, color):
    x = x0
    xs = []
    for s, c, lab in zip(sizes, chans, labels):
        w = 0.012 + 0.00022 * c
        h = 0.58 * s
        y = y_base - h / 2
        ax.add_patch(Rectangle((x, y), w, h, fc=color, ec=EDGE, lw=0.8))
        ax.text(x + w / 2, y - 0.03, lab, ha="center", va="top", fontsize=7.5)
        xs.append((x, w))
        x += w + 0.035
    return xs
yb = 0.56
sizes = [1.0, 1.0, 0.5, 0.25]
chans = [1, 64, 128, 256]
labels = ["input\n64^3 x 1\n(2D: 128^2)", "64^3 x 64", "32^3 x 128", "16^3 x 256"]
enc = stack(ax, 0.06, yb, sizes, chans, labels, C_AE)
lx = enc[-1][0] + enc[-1][1] + 0.05
ax.add_patch(Rectangle((lx, yb - 0.08), 0.04, 0.16, fc="#f4cccc", ec=EDGE, lw=1.2))
ax.text(lx + 0.02, yb - 0.11, "latent\n16^3 x 8\n(2D: 32^2 x 8)", ha="center", va="top", fontsize=7.5, fontweight="bold")
ax.text(lx + 0.02, yb + 0.10, "mean and\nlog-variance\nheads (KL)", ha="center", va="bottom", fontsize=7)
dec = stack(ax, lx + 0.10, yb, sizes[::-1], chans[::-1], ["16^3 x 256", "32^3 x 128", "64^3 x 64", "output\n64^3 x 1"], C_AE)
for (x, w), (x2, w2) in zip(enc[:-1], enc[1:]):
    arrow(ax, (x + w, yb), (x2, yb))
arrow(ax, (enc[-1][0] + enc[-1][1], yb), (lx, yb))
arrow(ax, (lx + 0.04, yb), (dec[0][0], yb))
for (x, w), (x2, w2) in zip(dec[:-1], dec[1:]):
    arrow(ax, (x + w, yb), (x2, yb))
ax.text(0.5, 0.93, "ENCODER (left): 3 levels, each = 2 residual blocks (group normalisation, SiLU, 3x3(x3) convolution) followed by a stride-2 down-sample.     DECODER (right): the mirror image, nearest-neighbour up-sample + convolution.",
        ha="center", va="top", fontsize=8.2)
ax.text(0.5, 0.02,
        "Compression: every axis is divided by 4 (2 down-samples), so 64^3 voxels x 1 channel become 16^3 x 8 = 32,768 numbers (8x fewer). "
        "The 2D version does the same on 128 x 128 slices (32 x 32 x 8).\n"
        "Training loss = L1 reconstruction error + 1e-6 x KL term (keeps the latent near a unit Gaussian) + 0.1 x VGG perceptual term. "
        "The 3D model is built by copying each 2D kernel into the centre depth slice of a 3D kernel, then trained further on volumes.",
        ha="center", va="bottom", fontsize=8.2, color="#333333")
save(fig, "fig03_autoencoder.png")

# ----------------------------------------------------------------------------------------
# Fig 04: U-Net flow vs epsilon input
# ----------------------------------------------------------------------------------------
fig, ax = blank((12, 5.4))
ax.text(0.5, 0.985, "The velocity / denoising U-Net that works inside the latent grid", ha="center", va="top", fontsize=12, fontweight="bold")
levels = [("64 ch\n16^3", 0.21, 0.60), ("128 ch\n8^3\n+ attention", 0.30, 0.45), ("256 ch\n4^3\n+ attention", 0.39, 0.30)]
W = 0.075
for lab, x, y in levels:
    box(ax, x, y, W, 0.12, lab, fc=C_DIFF, fs=7.5)
    box(ax, 1 - x - W, y, W, 0.12, lab, fc=C_DIFF, fs=7.5)
box(ax, 0.5 - 0.055, 0.12, 0.11, 0.12, "middle block\n256 ch, attention", fc=C_DIFF, fs=7.5)
# down path
arrow(ax, (0.21 + W, 0.62), (0.30, 0.55)); arrow(ax, (0.30 + W, 0.47), (0.39, 0.40)); arrow(ax, (0.39 + W, 0.32), (0.5 - 0.055, 0.22))
# up path
arrow(ax, (0.5 + 0.055, 0.22), (1 - 0.39 - W, 0.32)); arrow(ax, (1 - 0.39, 0.40), (1 - 0.30 - W, 0.47)); arrow(ax, (1 - 0.30, 0.55), (1 - 0.21 - W, 0.62))
for (lab, x, y) in levels:
    arrow(ax, (x + W, y + 0.10), (1 - x - W, y + 0.10), "skip", ls=":", color="#777777", fs=7, toff=(0, 0.005))
box(ax, 0.02, 0.62, 0.17, 0.31,
    "Input, FLOW mode\n(the model used)\n\nonly x_tau\n= 8 channels\n(a point on the straight\nline between the NAC\nand AC latents)",
    fc="#d9ead3", fs=8, bold_first=True)
box(ax, 0.02, 0.20, 0.17, 0.31,
    "Input, EPSILON mode\n(earlier attempt)\n\nnoisy AC latent (8 ch)\nstacked with the\nNAC latent (8 ch)\n= 16 channels",
    fc="#f4cccc", fs=8, bold_first=True)
arrow(ax, (0.19, 0.77), (0.21, 0.69))
arrow(ax, (0.19, 0.36), (0.21, 0.62), ls="--")
box(ax, 0.81, 0.62, 0.17, 0.31,
    "Output\n\n8 channels, same 16^3 grid\n\nFLOW: the velocity\nv = NAC - AC (latent)\n\nEPSILON: the noise that\nwas added",
    fc=C_OUT, fs=8, bold_first=True)
arrow(ax, (1 - 0.21, 0.69), (0.81, 0.77))
box(ax, 0.36, 0.81, 0.28, 0.12, "time embedding: sinusoidal(t), 64 -> 256 through a small MLP,\nadded inside every residual block", fc=C_GUI, fs=8)
arrow(ax, (0.50, 0.81), (0.50, 0.73), ls="--", color="#8b6f1f")
ax.text(0.5, 0.01,
        "MONAI DiffusionModelUNet, channels [64, 128, 256], 2 residual blocks per level, self-attention on the two coarse levels only. "
        "The same class is used in 2D (32^2 latent) and 3D (16^3 latent);\nthe 3D network is initialised from the 2D one by copying each 2D kernel into the centre of a 3D kernel. "
        "In flow mode the network never sees the NAC latent as a separate input: it is the starting point of the trajectory instead.",
        ha="center", va="bottom", fontsize=8.2, color="#333333")
save(fig, "fig04_unet_flow_vs_epsilon.png")

# ----------------------------------------------------------------------------------------
# Fig 05: the flow bridge and why the prediction is 'shrunk'
# ----------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(width_ratios=[1.35, 1]))
ax = axs[0]
ax.set_xlim(-0.15, 1.15); ax.set_ylim(-0.30, 1.15); ax.axis("off")
ax.set_title("(a) The straight-line bridge between the two PET versions (latent space)")
ax.plot([0, 1], [0.55, 0.55], color="#999999", lw=2, zorder=1)
ax.scatter([1], [0.55], s=260, color="#e69138", zorder=3); ax.text(1, 0.72, "tau = 1\nNAC latent\n(start of sampling)", ha="center", fontsize=8.5)
ax.scatter([0], [0.55], s=260, color="#3d85c6", zorder=3); ax.text(0, 0.72, "tau = 0\nAC latent\n(what we want)", ha="center", fontsize=8.5)
n = 8
xs = np.linspace(1, 0, n + 1)
for i in range(n):
    ax.annotate("", xy=(xs[i + 1] + 0.006, 0.55), xytext=(xs[i], 0.55),
                arrowprops=dict(arrowstyle="-|>", color="#cc0000", lw=1.3, mutation_scale=10))
ax.text(0.5, 0.42, "sampling: x <- x - (1/steps) * v_theta(x, tau), repeated 16 times (Euler)", ha="center", fontsize=8.5, color="#cc0000")
tau = 0.35
ax.scatter([tau], [0.55], s=120, color="#6aa84f", zorder=4)
ax.text(tau, 0.30, "training: pick a random tau,\nform x_tau = (1 - tau) AC + tau NAC,\nask the network for v; loss = |v - (NAC - AC)|^2",
        ha="center", va="top", fontsize=8.5, color="#38761d")
ax.text(0.5, -0.05, "true velocity along the line: d x_tau / d tau = NAC - AC (constant), so the path is straight\nand one exact step already reaches AC; the network learns E[NAC - AC | x_tau] instead.",
        ha="center", va="top", fontsize=8.2, color="#333333")
ax.text(0.5, 1.08, "epsilon (diffusion) alternative for comparison: start from pure noise, NAC glued on as extra channels, 25 curved DDIM steps",
        ha="center", va="top", fontsize=8, color="#777777", style="italic")
ax = axs[1]
rng = np.random.default_rng(0)
gt = np.clip(rng.gamma(1.6, 0.18, 1500), 0, 1)
pred = 0.84 * gt + 0.057 + rng.normal(0, 0.045, gt.size)
ax.scatter(gt, pred, s=5, alpha=0.35, color="#3d85c6", label="voxels (illustration)")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect prediction")
ax.plot([0, 1], [0.057, 0.84 + 0.057], color="#cc0000", lw=1.5, label="measured fit: pred = 0.84 gt + 0.057")
ax.set_xlabel("true AC intensity (normalised)"); ax.set_ylabel("predicted AC intensity")
ax.set_title("(b) Under-dispersion: hot tissue is under-estimated")
ax.legend(fontsize=7.5, loc="upper left")
ax.text(0.98, 0.04, "slope 0.84 and intercept 0.057 are the\nmeasured test-set values of ft3d_flow_p;\nthe points are synthetic, drawn to show\nwhat such a fit means", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#555555")
fig.tight_layout()
save(fig, "fig05_flow_bridge.png")

# ----------------------------------------------------------------------------------------
# Fig 06: inference path
# ----------------------------------------------------------------------------------------
fig, ax = blank((15, 3.8))
ax.text(0.5, 0.97, "Inference: from a non-corrected PET series to a predicted attenuation-corrected volume", ha="center", va="top", fontsize=12, fontweight="bold")
steps = [
    ("NAC DICOM series\n\nfound by the\nCorrectedImage tag\n(no 'ATTN')", C_DATA),
    ("Load + normalise\n\nsort by z, apply\nrescale slope/intercept,\nclip to 1st-99th pct,\nscale to [0,1]", C_DATA),
    ("Resize to 64^3\n\ntrilinear, whole\nvolume as one cube", C_DATA),
    ("3D encoder\n(frozen ae3d_p)\n\n64^3 -> 16^3 x 8\nposterior mean", C_AE),
    ("Flow sampler\n\nstart x = NAC latent\n16 Euler steps\nx <- x - v/16\ndeterministic", C_DIFF),
    ("3D decoder\n(frozen ae3d_p)\n\n16^3 x 8 -> 64^3", C_AE),
    ("Predicted AC\n\nclamp to [0,1]\nsave pred.npy\n(+ gt/nac, meta.json)", C_OUT),
    ("Scoring / display\n\n14 metrics vs the real\nAC; TrainViewer\ntriplet NAC|pred|GT", C_OUT),
]
n = len(steps); w = 0.108; gap = (1 - n * w) / (n + 1)
x = gap
for i, (t, c) in enumerate(steps):
    box(ax, x, 0.18, w, 0.62, t, fc=c, fs=7.3)
    if i < n - 1:
        arrow(ax, (x + w, 0.49), (x + w + gap, 0.49))
    x += w + gap
ax.text(0.5, 0.03, "Runtime for the whole 41-patient test set: about 6 minutes on one RTX 3090 (16 steps). The output is in normalised intensity units, not calibrated SUV.",
        ha="center", va="bottom", fontsize=8.5, color="#333333")
save(fig, "fig06_inference_path.png")

# ----------------------------------------------------------------------------------------
# Fig 07: TrainViewer <-> WSL file contract
# ----------------------------------------------------------------------------------------
fig, ax = blank((11.5, 4.6))
ax.text(0.5, 0.975, "TrainViewer: a torch-free Windows GUI drives GPU training in WSL, talking only through files", ha="center", va="top", fontsize=12, fontweight="bold")
box(ax, 0.02, 0.10, 0.28, 0.70,
    "Windows: TrainViewer (Gradio)\n\nsrc/TrainViewer/app.py  (UI)\ncontroller.py  (logic, no torch)\nmodel.py  (subprocess + file readers)\n\nTrain tab: pick task, roots, epochs,\nlearning rate, size, WSL venv\nInference tab: patient / slice /\nflow vs epsilon variant\nLive metric plots, NAC | pred | GT",
    fc=C_GUI, fs=8, bold_first=True)
box(ax, 0.36, 0.10, 0.28, 0.70,
    "Shared files: outputs/\n\n(Windows drive, seen from WSL as /mnt/c/...)\n\n<task>/metrics.jsonl  (one JSON per line,\nphase = train | val)\n<task>/runs/<run_id>/metrics.jsonl + meta.json\n<task>/best.pt, last.pt  (config embedded)\n<task>/split.json\ninfer/<task>/pred.npy, gt.npy, nac.npy, meta.json\ntrainviewer_settings.json",
    fc=C_GREY, fs=8, bold_first=True)
box(ax, 0.70, 0.10, 0.28, 0.70,
    "WSL Ubuntu-24.04: training process\n\nwsl -d Ubuntu-24.04 bash -lc\n  \"source ~/petct/.venv/bin/activate &&\n   cd /mnt/c/.../PetCt &&\n   python -m src.training.train.launcher\n     <task> --config ... --data_dir ...\"\n\nCUDA 12.8 torch, RTX 3090 (24 GB)\npython -m src.training.infer for the\ninference tab",
    fc=C_DIFF, fs=8, bold_first=True)
arrow(ax, (0.30, 0.78), (0.70, 0.78), "spawns the subprocess (command line at right)", rad=-0.22, toff=(0, 0.075))
arrow(ax, (0.70, 0.40), (0.64, 0.40), "writes")
arrow(ax, (0.36, 0.45), (0.30, 0.45), "tails by\nbyte offset", toff=(0, 0.02))
ax.text(0.5, 0.005, "No sockets, no shared Python objects: every state change is a file write on one side and a file read on the other, which also makes each run re-openable later.",
        ha="center", va="bottom", fontsize=8.5, color="#333333")
save(fig, "fig07_trainviewer_contract.png")

# ----------------------------------------------------------------------------------------
# Data-driven figures
# ----------------------------------------------------------------------------------------
def val_rows(run):
    p = f"outputs/{run}/metrics.jsonl"
    rows = []
    with open(p, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if '"phase": "val"' not in line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def load_eval(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def summ(e, k):
    s = e["summary"][k]
    return s["mean"] if isinstance(s, dict) else s

def std(e, k):
    s = e["summary"][k]
    return s.get("std", np.nan) if isinstance(s, dict) else np.nan

# Fig 08: training curves
runs_ae = [("ae2d_p", "2D autoencoder (ae2d_p)"), ("ae3d_p", "3D autoencoder (ae3d_p)")]
runs_flow = [("diff2d_flow_perc_p", "2D flow (diff2d_flow_perc_p)"), ("ft3d_flow_p", "3D flow (ft3d_flow_p)"),
             ("ft3d_flow_lpl_rfpp", "3D flow + LPL + RFPP"), ("ft3d_2x", "3D flow, 128^3 chain (ft3d_2x)")]
fig, axs = plt.subplots(2, 2, figsize=(12, 7.5))
endpoints = {}
ax = axs[0, 0]
for run, lab in runs_ae:
    rows = val_rows(run)
    st = np.array([r["step"] for r in rows]); l1 = np.array([r.get("recon_l1", np.nan) for r in rows]); ps = np.array([r.get("psnr", np.nan) for r in rows])
    ax.plot(st, l1, label=lab)
    i = int(np.nanargmin(l1)); endpoints[run] = dict(best_step=int(st[i]), best_recon_l1=float(l1[i]), psnr_at_best=float(ps[i]), ssim_at_best=float(rows[i].get("ssim", np.nan)), last_step=int(st[-1]), n_val=len(rows))
ax.set_xlabel("optimizer step"); ax.set_ylabel("validation L1 reconstruction error"); ax.set_title("Autoencoders: reconstruction error on held-out patients"); ax.legend(fontsize=8)
ax.set_yscale("log")
ax = axs[0, 1]
for run, lab in runs_ae:
    rows = val_rows(run)
    st = np.array([r["step"] for r in rows]); ps = np.array([r.get("psnr", np.nan) for r in rows])
    ax.plot(st, ps, label=lab)
ax.set_xlabel("optimizer step"); ax.set_ylabel("validation PSNR (dB)"); ax.set_title("Autoencoders: peak signal-to-noise ratio"); ax.legend(fontsize=8)
ax = axs[1, 0]
for run, lab in runs_flow:
    rows = val_rows(run)
    st = np.array([r["step"] for r in rows]); l1 = np.array([r.get("l1", np.nan) for r in rows]); ps = np.array([r.get("psnr", np.nan) for r in rows])
    ax.plot(st, l1, label=lab)
    i = int(np.nanargmin(l1)); endpoints[run] = dict(best_step=int(st[i]), best_l1=float(l1[i]), psnr_at_best=float(ps[i]), ssim_at_best=float(rows[i].get("ssim", np.nan)), last_step=int(st[-1]), last_l1=float(l1[-1]), n_val=len(rows))
ax.set_xlabel("optimizer step"); ax.set_ylabel("validation rollout L1 (8 Euler steps)"); ax.set_title("Flow models: honest rollout error on held-out patients"); ax.legend(fontsize=8)
ax.set_ylim(0.008, 0.05)
ax = axs[1, 1]
for run, lab in runs_flow:
    rows = val_rows(run)
    st = np.array([r["step"] for r in rows]); ps = np.array([r.get("psnr", np.nan) for r in rows])
    ax.plot(st, ps, label=lab)
ax.set_xlabel("optimizer step"); ax.set_ylabel("validation PSNR (dB)"); ax.set_title("Flow models: PSNR of the rollout"); ax.legend(fontsize=8)
ax.set_ylim(15, 30)
fig.suptitle("Training curves (validation patients, never augmented). Model selection uses the lowest rollout L1.", fontsize=11)
fig.tight_layout()
save(fig, "fig08_training_curves.png")
print("ENDPOINTS", json.dumps(endpoints, indent=1))

for run in ["diff2d", "ft3d", "diff2d_cfg", "diff2d_flow", "diff2d_flow2", "diff2d_flow_perc", "ae2d", "ae3d", "ae2d_2x", "ae3d_2x", "ae3d_2x2", "diff2d_2x", "diff2d_2x_long", "ft3d_flow", "ft3d_ft_ctl5e5", "ft3d_ft_iwlat"]:
    try:
        rows = val_rows(run)
        key = "recon_l1" if "recon_l1" in rows[-1] else ("l1" if "l1" in rows[-1] else "loss")
        vals = np.array([r.get(key, np.nan) for r in rows]); st = np.array([r["step"] for r in rows])
        i = int(np.nanargmin(vals))
        print(f"EP {run:22s} key={key:8s} best={vals[i]:.5f} @step {st[i]:>7d}  psnr={rows[i].get('psnr', float('nan')):.2f} ssim={rows[i].get('ssim', float('nan')):.4f} last_step={st[-1]} n_val={len(rows)} loss_last={rows[-1].get('loss', float('nan')):.5f}")
    except Exception as e:
        print("EP", run, "ERR", e)

# Fig 09: result families
ceil = load_eval("outputs/report/ceiling_3d.json")["summary"]
print("CEIL keys", list(ceil.keys()))
fam = [
    ("NAC as-is\n(no model)", None, ceil["nac_raw"]),
    ("2D diffusion\nepsilon\n(diff2d)", "outputs/eval/diff2d/metrics.json", None),
    ("2D diffusion\nepsilon, CFG 3\n(diff2d_cfg)", "outputs/eval/diff2d_cfg/g3.0.json", None),
    ("3D diffusion\nepsilon\n(ft3d)", "outputs/eval/ft3d/metrics.json", None),
    ("2D flow\n8 steps\n(diff2d_flow_\nperc_p)", "outputs/eval/diff2d_flow_perc_p/test_s8.json", None),
    ("3D flow\n16 steps\n(ft3d_flow_p)", "outputs/eval/ft3d_flow_p/test_s16_clamped.json", None),
    ("3D flow\n128^3 chain\n(ft3d_2x)", "outputs/eval/ft3d_2x/test_s16.json", None),
    ("3D flow\nPET-weighted\nfine-tune\n(ft3d_ft_\niwlat)", "outputs/eval/ft3d_ft_iwlat/test_s16.json", None),
    ("Autoencoder\nceiling\ndecode(\nencode(AC))", None, ceil["ae_ceiling_c"]),
]
names, ps, ss, r2, sl, pse, sse = [], [], [], [], [], [], []
for nm, path, direct in fam:
    if direct is not None:
        d = direct; names.append(nm); ps.append(d["psnr"]); ss.append(d["ssim"]); r2.append(d["voxel_r2"]); sl.append(d["reg_slope"]); pse.append(0); sse.append(0)
        print(f"FAM {nm.replace(chr(10),' '):50s} psnr={d['psnr']:.2f} ssim={d['ssim']:.4f} r2={d['voxel_r2']:.3f} slope={d['reg_slope']:.3f} bias={d.get('rel_bias', float('nan')):.4f} hot={d.get('hot_band_rel_error', float('nan')):.4f}")
    else:
        if not os.path.exists(path):
            print("MISSING", path); continue
        e = load_eval(path); names.append(nm)
        ps.append(summ(e, "psnr")); ss.append(summ(e, "ssim")); r2.append(summ(e, "voxel_r2")); sl.append(summ(e, "reg_slope"))
        pse.append(std(e, "psnr")); sse.append(std(e, "ssim"))
        print(f"FAM {nm.replace(chr(10),' '):50s} n={e.get('n_patients_evaluated')} psnr={summ(e,'psnr'):.2f}+-{std(e,'psnr'):.2f} ssim={summ(e,'ssim'):.4f}+-{std(e,'ssim'):.3f} nrmse={summ(e,'nrmse'):.4f} mae={summ(e,'mae'):.4f} r2={summ(e,'voxel_r2'):.3f} slope={summ(e,'reg_slope'):.3f} icpt={summ(e,'reg_intercept'):.3f} bias={summ(e,'rel_bias'):.4f} hot={summ(e,'hot_band_rel_error') if 'hot_band_rel_error' in e['summary'] else float('nan'):.4f}")
cols = ["#999999", "#f4cccc", "#f4cccc", "#f4cccc", "#b6d7a8", "#6aa84f", "#93c47d", "#38761d", "#3d85c6"]
fig, axs = plt.subplots(2, 2, figsize=(15, 8.6))
xi = np.arange(len(names))
for ax, vals, err, lab, ref in [(axs[0, 0], ps, pse, "PSNR (dB), higher is better", None), (axs[0, 1], ss, sse, "SSIM (1 = identical), higher is better", 1.0),
                                (axs[1, 0], r2, None, "voxel R^2 (fraction of variance explained)", 1.0), (axs[1, 1], sl, None, "regression slope of pred vs true (1 = no shrinkage)", 1.0)]:
    b = ax.bar(xi, vals, color=cols[:len(names)], edgecolor="#333333", yerr=err if err is not None else None, capsize=3)
    ax.set_xticks(xi); ax.set_xticklabels(names, fontsize=7)
    ax.set_title(lab)
    if ref is not None:
        ax.axhline(ref, color="k", ls="--", lw=0.8)
    if lab.startswith("voxel"):
        ax.set_ylim(-3, 1.15); ax.text(0.02, 0.05, "epsilon models are below the axis (R^2 down to -98)", transform=ax.transAxes, fontsize=7.5, color="#990000")
    if lab.startswith("regression"):
        ax.set_ylim(-0.1, 1.15)
    if lab.startswith("PSNR"):
        ax.set_ylim(-5, 37)
    if lab.startswith("SSIM"):
        ax.set_ylim(0, 1.15)
    y0, y1 = ax.get_ylim()
    for rect, v, e_ in zip(b, vals, err if err is not None else [0] * len(vals)):
        yy = min(max(v, 0) + e_, y1 - 0.06 * (y1 - y0))
        ax.text(rect.get_x() + rect.get_width() / 2, yy, f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}", ha="center", va="bottom", fontsize=7, clip_on=False)
    ax.tick_params(axis="x", labelsize=6.6)
fig.suptitle("Held-out test set, all metrics on normalised intensity. Error bars: standard deviation across patients (n = 19 for the three epsilon runs, 41 otherwise).", fontsize=10)
fig.tight_layout()
save(fig, "fig09_results_families.png")

# Fig 10: under-dispersion metrics across 3D arms
arms = [
    ("ft3d_flow_p\n(baseline,\n30k steps)", "outputs/eval/ft3d_flow_p/test_s16_clamped.json"),
    ("+6000 steps\nplain control\n(ctl5e5)", "outputs/eval/ft3d_ft_ctl5e5/test_s16.json"),
    ("+6000 steps\nPET-value\nweights (iwlat)", "outputs/eval/ft3d_ft_iwlat/test_s16.json"),
    ("LPL + RFPP\n(own split)", "outputs/eval/ft3d_flow_lpl_rfpp/test_s16.json"),
    ("128^3 chain\n(ft3d_2x)", "outputs/eval/ft3d_2x/test_s16.json"),
]
fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.8))
labels = [a[0] for a in arms]
E = [load_eval(a[1]) for a in arms]
for ax, key, title, ref in [(axs[0], "reg_slope", "regression slope (1 = perfect scale)", 1.0), (axs[1], "hot_band_rel_error", "hot-band relative error (GT 90th-98th pct)\n0 = unbiased, negative = under-estimated", 0.0), (axs[2], "p95_rel_error", "95th-percentile relative error\n0 = unbiased", 0.0)]:
    vals = [summ(e, key) for e in E]
    b = ax.bar(np.arange(len(vals)), vals, color=["#6aa84f", "#93c47d", "#38761d", "#e69138", "#cc4125"], edgecolor="#333")
    ax.axhline(ref, color="k", ls="--", lw=0.8); ax.set_xticks(np.arange(len(vals))); ax.set_xticklabels(labels, fontsize=6.8); ax.set_title(title, fontsize=9)
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)
    ax.axhline(0, color="#888", lw=0.5)
fig.suptitle("Does the prediction reach the true intensity range? Three views of the same under-dispersion, 3D arms, 16 Euler steps, clamped output, test set", fontsize=10)
fig.tight_layout()
save(fig, "fig10_under_dispersion.png")
for e, k in zip(E, ["ft3d_flow_p", "ctl5e5", "iwlat", "lpl_rfpp", "ft3d_2x"]):
    print(f"ARM {k:12s} n={e.get('n_patients_evaluated')} " + " ".join(f"{m}={summ(e,m):.4f}" for m in ["psnr", "ssim", "nrmse", "mae", "rel_bias", "p95_rel_error", "hot_band_rel_error", "voxel_r2", "reg_slope", "reg_intercept", "ba_mean_bias", "ba_loa_lower", "ba_loa_upper"] if m in e["summary"]))

# Fig 11: paired per-patient deltas iwlat - ctl5e5
def by_patient(e):
    return {r["patient"]: r for r in e["per_patient"] if "patient" in r}
A = by_patient(load_eval("outputs/eval/ft3d_ft_ctl5e5/test_s16.json")); B = by_patient(load_eval("outputs/eval/ft3d_ft_iwlat/test_s16.json"))
common = sorted(set(A) & set(B))
fig, axs = plt.subplots(1, 4, figsize=(13, 4))
for ax, key, lab in [(axs[0], "reg_slope", "regression slope"), (axs[1], "hot_band_rel_error", "hot-band relative error"), (axs[2], "psnr", "PSNR (dB)"), (axs[3], "ssim", "SSIM")]:
    d = np.array([B[p][key] - A[p][key] for p in common])
    order = np.argsort(d)
    colors = ["#38761d" if v > 0 else "#cc0000" for v in d[order]]
    ax.bar(np.arange(len(d)), d[order], color=colors)
    ax.axhline(0, color="k", lw=0.8)
    wins = int((d > 0).sum()); losses = int((d < 0).sum())
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std(ddof=1) > 0 else np.nan
    ax.set_title(f"{lab}\nweighted minus control, per patient\nmean {d.mean():+.4f}, t = {t:.1f}, {wins} up / {losses} down", fontsize=8.5)
    ax.set_xlabel("patients (sorted)"); ax.set_xticks([])
    print(f"PAIRED iwlat-ctl {key}: mean={d.mean():+.5f} t={t:.2f} wins={wins} losses={losses} n={len(d)}")
fig.suptitle("Fine-tuning with PET-value weighting vs the matched control: same 41 test patients, same starting weights, same 6000 steps", fontsize=10)
fig.tight_layout()
save(fig, "fig11_paired_iwlat_vs_control.png")

# Fig 12: gradient share (from scripts/_diag_intensity_weight.py, quoted in ft3d_ft_iwlat.yaml)
share = {"plain mean\n(baseline loss)": [80.6, 10.8, 6.2, 2.4], "occupancy gate\nbg_weight 0.1": [60.3, 21.6, 12.9, 5.2], "PET-value weight\n(iwlat arm)": [53.1, 16.5, 18.4, 12.1]}
fig, ax = plt.subplots(figsize=(8.5, 4.2))
bands = ["air (< 0.05)", "cold (0.05-0.3)", "mid (0.3-0.6)", "hot (>= 0.6)"]
bcol = ["#cccccc", "#9fc5e8", "#f6b26b", "#cc0000"]
bottom = np.zeros(3)
for i, bnd in enumerate(bands):
    vals = np.array([share[k][i] for k in share])
    ax.bar(np.arange(3), vals, bottom=bottom, color=bcol[i], edgecolor="#333", label=bnd)
    for j, v in enumerate(vals):
        ax.text(j, bottom[j] + v / 2, f"{v:.1f}%", ha="center", va="center", fontsize=8)
    bottom += vals
ax.set_xticks(np.arange(3)); ax.set_xticklabels(list(share.keys()), fontsize=9); ax.set_ylabel("share of the loss gradient (%)")
ax.set_title("Where the flow-loss gradient goes, by true-uptake band (6 held-out patients, measured before training)", fontsize=9.5)
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
fig.tight_layout()
save(fig, "fig12_gradient_share.png")

# Fig 13: step-count dependence
def steps_curve(dirpat):
    pts = []
    for p in glob.glob(dirpat):
        e = load_eval(p)
        if e.get("mode") != "test":
            continue
        pts.append((e["ddim_steps"], summ(e, "psnr"), summ(e, "ssim"), os.path.basename(p)))
    pts.sort()
    return pts
curves = [
    ("2D flow (diff2d_flow_perc_p)", steps_curve("outputs/eval/diff2d_flow_perc_p/test_s*.json")),
    ("2D flow, first fixed run (diff2d_flow2)", steps_curve("outputs/eval/diff2d_flow_nc/test_s*.json")),
    ("3D flow (ft3d_flow_p, unclamped)", [(e["ddim_steps"], summ(e, "psnr"), summ(e, "ssim"), os.path.basename(p)) for p in ["outputs/eval/ft3d_flow_p/test_s16.json", "outputs/eval/ft3d_flow_p/test_s32.json"] if os.path.exists(p) for e in [load_eval(p)]]),
    ("3D flow, 128^3 chain (ft3d_2x)", steps_curve("outputs/eval/ft3d_2x/test_s*.json")),
    ("'flow' run that was really epsilon (diff2d_flow)", steps_curve("outputs/eval/diff2d_flow/*.json")),
]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
for lab, pts in curves:
    if not pts:
        print("NO POINTS", lab); continue
    print("STEPS", lab, [(a, round(b, 3), round(c, 4), d) for a, b, c, d in pts])
    axs[0].plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=lab)
    axs[1].plot([p[0] for p in pts], [p[2] for p in pts], "o-", label=lab)
for ax, lab in [(axs[0], "PSNR (dB)"), (axs[1], "SSIM")]:
    ax.set_xscale("log", base=2); ax.set_xlabel("number of sampling steps"); ax.set_ylabel(lab); ax.set_xticks([8, 16, 32, 64]); ax.set_xticklabels(["8", "16", "32", "64"])
axs[0].legend(fontsize=7, loc="center right")
fig.suptitle("A straight bridge needs few steps: test quality barely moves from 8 to 32 steps; the mis-configured epsilon run is worse and erratic", fontsize=10)
fig.tight_layout()
save(fig, "fig13_step_invariance.png")

# Fig 14: headroom / ceilings against native truth
nf = load_eval("outputs/report/native_flow.json")["summary"]
nc = load_eval("outputs/report/native_ceiling.json")["summary"]
print("NF keys", list(nf.keys())); print("NC keys", list(nc.keys()))
def pick(d, *cands):
    for c in cands:
        if c in d:
            return d[c]
    for k in d:
        if all(tok in k for tok in cands[0].split()):
            return d[k]
    raise KeyError(cands)
f64 = pick(nf, "64^3  ft3d_flow_p", "64 ft3d_flow_p"); f128 = pick(nf, "128^3 ft3d_2x", "128 ft3d_2x")
labels = ["NAC as-is\n(64^3 reference,\nn=41)", "3D flow 64^3\n(ft3d_flow_p)", "3D AE only\n64^3 (ae3d_p)", "resize to 64^3\nand back\n(no model)", "3D flow 128^3\n(ft3d_2x)", "3D AE only\n128^3 (ae3d_2x2)", "resize to 128^3\nand back\n(no model)"]
psnr_v = [ceil["nac_raw"]["psnr"], f64["psnr"], nc["64_ae"]["psnr"], nc["64_resize_only"]["psnr"], f128["psnr"], nc["128_ae"]["psnr"], nc["128_resize_only"]["psnr"]]
hot_v = [ceil["nac_raw"]["hot_band_rel_error"], f64["hot_band_rel_error"], nc["64_ae"]["hot_band_rel_error"], nc["64_resize_only"]["hot_band_rel_error"], f128["hot_band_rel_error"], nc["128_ae"]["hot_band_rel_error"], nc["128_resize_only"]["hot_band_rel_error"]]
print("NATIVE psnr", [round(v, 2) for v in psnr_v]); print("NATIVE hot", [round(v, 3) for v in hot_v])
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
cc = ["#999999", "#6aa84f", "#3d85c6", "#cfe2f3", "#cc4125", "#3d85c6", "#cfe2f3"]
b = axs[0].bar(np.arange(7), psnr_v, color=cc, edgecolor="#333"); axs[0].set_title("PSNR against the native-resolution truth (dB)"); axs[0].set_xticks(np.arange(7)); axs[0].set_xticklabels(labels, fontsize=7)
for rect, v in zip(b, psnr_v): axs[0].text(rect.get_x() + rect.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
b = axs[1].bar(np.arange(7), hot_v, color=cc, edgecolor="#333"); axs[1].set_title("hot-band relative error against native truth (0 = unbiased)"); axs[1].set_xticks(np.arange(7)); axs[1].set_xticklabels(labels, fontsize=7); axs[1].axhline(0, color="k", lw=0.8)
for rect, v in zip(b, hot_v): axs[1].text(rect.get_x() + rect.get_width() / 2, v, f"{v:.3f}", ha="center", va="top", fontsize=8)
fig.suptitle("Where is the remaining error? Each pipeline compared with what its own autoencoder and its own resize would allow (12 held-out patients unless noted)", fontsize=10)
fig.tight_layout()
save(fig, "fig14_headroom_ceilings.png")

# Fig 15: per-patient distributions
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
sets = [("ft3d_flow_p", "outputs/eval/ft3d_flow_p/test_s16_clamped.json"), ("ctl5e5", "outputs/eval/ft3d_ft_ctl5e5/test_s16.json"), ("iwlat", "outputs/eval/ft3d_ft_iwlat/test_s16.json"), ("ft3d_2x", "outputs/eval/ft3d_2x/test_s16.json"), ("2D flow\n(perc_p, s8)", "outputs/eval/diff2d_flow_perc_p/test_s8.json")]
for ax, key, lab in [(axs[0], "psnr", "PSNR (dB)"), (axs[1], "ssim", "SSIM"), (axs[2], "reg_slope", "regression slope")]:
    data = [[r[key] for r in load_eval(p)["per_patient"] if np.isfinite(r.get(key, np.nan))] for _, p in sets]
    ax.boxplot(data, tick_labels=[s[0] for s in sets], showmeans=True)
    ax.set_title(lab); ax.tick_params(axis="x", labelsize=8)
fig.suptitle("Spread across the 41 test patients (box = middle half, line = median, triangle = mean)", fontsize=10)
fig.tight_layout()
save(fig, "fig15_per_patient_spread.png")

# Fig 16: qualitative triplet from the HTML report images
coh = load_eval("outputs/report/test41_cohort_metrics.json")
pp = sorted(coh["per_patient"], key=lambda r: r["metrics"]["ssim"])
picks = [pp[len(pp) // 2], pp[-1], pp[2]]
picknames = ["median SSIM", "best SSIM", "3rd-worst SSIM"]
fig, axs = plt.subplots(len(picks) * 2, 3, figsize=(9.5, 3.0 * len(picks) * 2))
for pi, (rec, pn) in enumerate(zip(picks, picknames)):
    d = os.path.join("outputs/report/ft3d_flow_p/img", rec["pid"])
    for ri, plane in enumerate(["axial", "coronal"]):
        for ci, kind in enumerate(["nac", "pred", "gt"]):
            cands = sorted(glob.glob(os.path.join(d, f"{plane}_*_{kind}.png")))
            ax = axs[pi * 2 + ri, ci]
            ax.axis("off")
            if not cands:
                continue
            f = cands[len(cands) // 2]
            img = mpimg.imread(f)
            ax.imshow(img, cmap="hot" if img.ndim == 2 else None)
            title = {"nac": "input: non-corrected PET", "pred": "model output: predicted AC", "gt": "truth: measured AC"}[kind]
            if pi == 0 and ri == 0:
                ax.set_title(title, fontsize=9)
            if ci == 0:
                ax.text(-0.04, 0.5, f"{pn} patient, {plane}\nSSIM {rec['metrics']['ssim']:.3f}, PSNR {rec['metrics']['psnr']:.1f} dB", transform=ax.transAxes, ha="right", va="center", fontsize=7.5, rotation=90)
fig.suptitle("ft3d_flow_p on test patients (64^3, 16 steps). Prediction and truth share one intensity window, so brightness differences are real errors.", fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, 0.965))
save(fig, "fig16_qualitative_triplets.png")
print("PICKS", [(p["pid"][-14:], p.get("dataset"), round(p["metrics"]["ssim"], 3), round(p["metrics"]["psnr"], 2)) for p in picks])

# Fig 17: what the 64^3 resize destroys
fig, axs = plt.subplots(1, 2, figsize=(9, 5.2))
for ax, f, t in [(axs[0], "outputs/report/_diag/full_coronal_mip.png", "native resolution"), (axs[1], "outputs/report/_diag/r64_coronal_mip.png", "after resizing the whole volume to 64^3")]:
    if os.path.exists(f):
        img = mpimg.imread(f); ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    ax.set_title(t); ax.axis("off")
fig.suptitle("Coronal maximum-intensity projection of one attenuation-corrected PET volume: the working resolution costs detail", fontsize=10)
fig.tight_layout()
save(fig, "fig17_resolution_loss.png")

# Fig 18: CFG sweep (epsilon era)
pts = []
for p in sorted(glob.glob("outputs/eval/diff2d_cfg/*.json")):
    e = load_eval(p)
    g = e.get("guidance_scale")
    if g is None:
        continue
    pts.append((float(g), summ(e, "ssim"), summ(e, "psnr"), summ(e, "rel_bias"), os.path.basename(p)))
pts.sort()
print("CFG", pts)
if pts:
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    axs[0].plot([p[0] for p in pts], [p[1] for p in pts], "o-", color="#cc4125"); axs[0].set_xlabel("guidance scale"); axs[0].set_ylabel("test SSIM"); axs[0].set_title("Classifier-free guidance on the epsilon 2D model")
    axs[1].plot([p[0] for p in pts], [p[2] for p in pts], "o-", color="#cc4125"); axs[1].set_xlabel("guidance scale"); axs[1].set_ylabel("test PSNR (dB)"); axs[1].set_title("PSNR stays near 14-15 dB whatever the scale")
    axs[0].axhline(0.886, color="#38761d", ls="--", lw=1); axs[0].text(1.05, 0.80, "2D flow bridge, same AE: 0.886", color="#38761d", fontsize=8)
    axs[0].set_ylim(0, 1)
    fig.tight_layout(); save(fig, "fig18_cfg_sweep.png")

# Fig 19: fine-tune ablation (1500 steps @1e-5) inertness
ft = ["control", "bgw", "iw", "lpl", "ushaped", "expectile", "slope", "bgw_slope", "slope_w200", "slope_w600", "slope_w1800", "slope_w5400"]
vals = []
for a in ft:
    p = f"outputs/eval/ft3d_ft_{a}/test_s16.json"
    if os.path.exists(p):
        e = load_eval(p); vals.append((a, summ(e, "psnr"), summ(e, "reg_slope"), summ(e, "hot_band_rel_error"), summ(e, "ssim")))
print("FT", vals)
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
base = load_eval("outputs/eval/ft3d_flow_p/test_s16_clamped.json")
for ax, i, lab, bk in [(axs[0], 1, "PSNR (dB)", "psnr"), (axs[1], 2, "regression slope", "reg_slope"), (axs[2], 3, "hot-band relative error", "hot_band_rel_error")]:
    ax.bar(np.arange(len(vals)), [v[i] for v in vals], color="#93c47d", edgecolor="#333")
    ax.axhline(summ(base, bk), color="k", ls="--", lw=0.9, label="starting model (ft3d_flow_p)")
    ax.set_xticks(np.arange(len(vals))); ax.set_xticklabels([v[0] for v in vals], rotation=60, fontsize=7.5); ax.set_title(lab)
    lo = min([v[i] for v in vals] + [summ(base, bk)]); hi = max([v[i] for v in vals] + [summ(base, bk)]); pad = (hi - lo) * 0.6 + 1e-6
    ax.set_ylim(lo - pad, hi + pad)
axs[0].legend(fontsize=7.5)
fig.suptitle("The short fine-tune ablation (1500 steps at learning rate 1e-5, 12 arms): every arm lands on top of the starting model. Note the zoomed y-axes.", fontsize=9.5)
fig.tight_layout()
save(fig, "fig19_finetune_ablation_inert.png")
print("DONE")
