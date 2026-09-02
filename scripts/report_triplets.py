"""Run the held-out TEST set through a trained NAC->AC model and build an HTML report.

Produces, per test patient, a scrubbable **triplet** viewer -- NAC input | predicted AC |
ground-truth AC -- in all three planes, plus the same image-quality metrics
``src/training/evaluate.py`` reports (identical split resolution, identical sampler via
``src.training.utils.translate.sample_nac_to_ac``, so the numbers reproduce).

Rendering is PIL-only (no matplotlib in the WSL training venv). Panels are written as
PNGs under ``<out_dir>/img/`` and referenced relatively from ``<out_dir>/index.html``,
so the report stays a lightweight, lazily-loaded local page.

Display windowing (stated in the report too, because it decides what the eye sees):
  * predicted AC and ground-truth AC share ONE window, taken from the GT volume
    (p0.5..p99.5) -- so a brightness difference between those two panels is a real
    prediction error, not a windowing artifact.
  * NAC gets its OWN window (p0.5..p99.5 of NAC). It has to: the whole point of
    attenuation correction is that NAC lives on a different intensity scale.

Run in WSL/GPU from the project root, e.g. the best 3D flow bridge:

  python scripts/report_triplets.py --task ft3d \
      --data_dir "/mnt/d/DeepTrainingData/Project/ACRIN 6668" ... \
      --diff_ckpt outputs/ft3d_flow_p/best.pt --ae_ckpt outputs/ae3d_p/best.pt \
      --split_json outputs/ft3d_flow_p/split.json \
      --size 64 --ddim_steps 16 --spacing linear \
      --out_dir outputs/report/ft3d_flow_p --label "ft3d_flow_p (3D rectified flow)"
"""

import argparse
import base64
import html
import io
import json
import math
import os
import sys
import time
from pathlib import Path

# Run as a plain script (`python scripts/report_triplets.py`): only scripts/ is on
# sys.path, so add the project root for the `src.` imports below.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from src.training.data import load_patient_by_path
from src.training.evaluate import METRIC_KEYS, _resolve_candidates, _subsample
from src.training.infer import _load_model, _resize_volume, _schedule_from_config, _slice_2d
from src.training.models.autoencoder2d import ae_decode, ae_encode, build_autoencoder_2d
from src.training.models.autoencoder3d import ae3d_decode, ae3d_encode, build_autoencoder_3d
from src.training.models.diffusion2d import build_diffusion_2d
from src.training.models.diffusion3d import build_diffusion_3d
from src.training.utils.image_metrics import image_quality_metrics
from src.training.utils.translate import sample_nac_to_ac

KINDS = ("nac", "pred", "gt")
KIND_TITLE = {"nac": "NAC (input)", "pred": "Predicted AC", "gt": "Ground-truth AC"}
# Metrics surfaced on the patient cards / sort menu, in display order.
CARD_METRICS = ["ssim", "psnr", "voxel_r2", "mae", "rel_bias", "max_rel_error"]
METRIC_LABEL = {
    "ssim": "SSIM", "psnr": "PSNR (dB)", "nrmse": "NRMSE", "mae": "MAE",
    "rel_bias": "rel. bias", "max_rel_error": "max rel. err",
    "voxel_r2": "voxel R²", "reg_slope": "reg. slope",
    "reg_intercept": "reg. intercept", "ba_mean_bias": "B-A bias",
    "ba_loa_lower": "B-A LoA low", "ba_loa_upper": "B-A LoA high",
}
# Higher-is-better metrics get the "good = high" colour ramp in the table.
HIGHER_BETTER = {"ssim", "psnr", "voxel_r2", "reg_slope"}
# Signed metrics where zero is best (either sign is equally bad).
SIGNED_ZERO_BEST = {"rel_bias", "ba_mean_bias", "reg_intercept"}


# --------------------------------------------------------------------------- #
# colour mapping / PNG panels
# --------------------------------------------------------------------------- #

def _lut(name):
    """256x3 uint8 lookup table. 'hot' mirrors matplotlib's hot; 'gray' is linear."""
    x = np.linspace(0.0, 1.0, 256)
    if name == "gray":
        rgb = np.stack([x, x, x], axis=1)
    elif name == "hot":
        r = np.clip(x / 0.375, 0, 1)
        g = np.clip((x - 0.375) / 0.375, 0, 1)
        b = np.clip((x - 0.75) / 0.25, 0, 1)
        rgb = np.stack([r, g, b], axis=1)
    else:  # 'magma'-ish perceptual ramp through purple/orange
        anchors = np.array([
            [0.001, 0.000, 0.014], [0.161, 0.048, 0.276], [0.397, 0.084, 0.433],
            [0.622, 0.165, 0.388], [0.851, 0.324, 0.243], [0.972, 0.611, 0.256],
            [0.988, 0.909, 0.720],
        ])
        pos = np.linspace(0.0, 1.0, len(anchors))
        rgb = np.stack([np.interp(x, pos, anchors[:, c]) for c in range(3)], axis=1)
    return (np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def _panel_bytes(plane_img, vmin, vmax, lut, px, interp="smooth"):
    """Window ``plane_img`` (2D float) to [vmin,vmax], colour-map it, return PNG bytes.

    Upscaling to ``px`` happens on the windowed FLOAT image (before colour mapping), so
    it is ordinary display interpolation -- the same thing any DICOM viewer does -- and
    is applied identically to all three panels. ``interp='nearest'`` instead shows the
    raw voxel grid blockily, with no invented intermediate values.

    The colour map has exactly 256 entries, so the PNG is written as a PALETTED image
    (mode 'P') with the LUT as its palette: identical pixels to an RGB PNG at roughly a
    third of the bytes, which matters a lot when panels are base64-embedded into a
    single-file report.
    """
    denom = max(float(vmax) - float(vmin), 1e-6)
    norm = np.clip((plane_img.astype(np.float32) - float(vmin)) / denom, 0.0, 1.0)
    if px and px != norm.shape[0]:
        resample = Image.NEAREST if interp == "nearest" else Image.LANCZOS
        norm = np.asarray(
            Image.fromarray(norm, mode="F").resize((px, px), resample), dtype=np.float32)
        norm = np.clip(norm, 0.0, 1.0)  # LANCZOS can ring slightly out of range
    idx = (norm * 255.0 + 0.5).astype(np.uint8)
    img = Image.fromarray(idx, mode="P")
    img.putpalette(lut.reshape(-1).tolist())
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _panel_png(plane_img, vmin, vmax, lut, out_path, px, interp="smooth"):
    """``_panel_bytes`` written to ``out_path``."""
    data = _panel_bytes(plane_img, vmin, vmax, lut, px, interp)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)


def _window(vol):
    """Robust display window for a normalized-intensity PET volume."""
    finite = vol[np.isfinite(vol)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(finite, 0.5))
    vmax = float(np.percentile(finite, 99.5))
    if vmax <= vmin:
        vmax = vmin + 1e-3
    return vmin, vmax


def _plane_slice(vol, plane, index):
    """Extract a display-oriented 2D slice from a (Z,Y,X) numpy volume.

    Coronal/sagittal are flipped with ``np.flipud`` so head-up matches the
    DataViewer convention (see CLAUDE.md 'Coordinate-system gotchas').
    """
    if plane == "axial":
        return vol[index, :, :]
    if plane == "coronal":
        return np.flipud(vol[:, index, :])
    return np.flipud(vol[:, :, index])


def _positions(n, count, lo=0.15, hi=0.85):
    """Slice indices to render: all ``n`` when ``count >= n``, else ``count`` evenly
    spaced, de-duplicated indices over the central span (the ends are usually empty)."""
    if n <= 1:
        return [0]
    # Asking for at least as many panels as there are slices means "all of them" --
    # return the full range rather than duplicating indices inside the central span.
    if count >= n:
        return list(range(n))
    raw = np.linspace(lo * (n - 1), hi * (n - 1), num=count)
    return sorted({int(round(v)) for v in raw})


def _render_patient(vols_np, out_root, pid, planes, px, lut, interp="smooth", embed=False):
    """Render every panel for one patient.

    Returns ``(slices, windows, panels)``. With ``embed=False`` the panels are written as
    PNG files under ``<out_root>/img/<pid>/`` and ``panels`` is empty; with ``embed=True``
    nothing is written and ``panels`` maps ``"<plane>_<kk>_<kind>"`` to a base64 data URI
    for inlining into a single-file report.
    """
    gt_win = _window(vols_np["gt"])
    # pred and gt MUST share a window, otherwise a brightness error looks like a
    # windowing choice. NAC needs its own -- different intensity scale by definition.
    wins = {"nac": _window(vols_np["nac"]), "pred": gt_win, "gt": gt_win}
    axis_len = {"axial": 0, "coronal": 1, "sagittal": 2}
    slices, panels = {}, {}
    for plane, count in planes:
        n = vols_np["gt"].shape[axis_len[plane]]
        idxs = _positions(n, count)
        slices[plane] = idxs
        for k, idx in enumerate(idxs):
            for kind in KINDS:
                sl = _plane_slice(vols_np[kind], plane, idx)
                key = f"{plane}_{k:02d}_{kind}"
                if embed:
                    data = _panel_bytes(sl, wins[kind][0], wins[kind][1], lut, px, interp)
                    panels[key] = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
                else:
                    _panel_png(sl, wins[kind][0], wins[kind][1], lut,
                               out_root / "img" / pid / f"{key}.png", px, interp)
    return slices, {k: [round(v, 6) for v in wins[k]] for k in KINDS}, panels


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #

def _patient_id(path, used):
    """Short, unique, filesystem-safe label: '<dataset>__<patient folder>'."""
    p = Path(path)
    base = f"{p.parent.name}__{p.name}" if p.parent.name else p.name
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in base)[:80]
    pid, n = safe, 2
    while pid in used:
        pid, n = f"{safe}_{n}", n + 1
    used.add(pid)
    return pid


@torch.no_grad()
def _predict_ft3d(args, device, candidates, out_root, planes, lut):
    ae, _ = _load_model(args.ae_ckpt, build_autoencoder_3d, device, use_ema=args.use_ema)
    model, diff_config = _load_model(args.diff_ckpt, build_diffusion_3d, device, use_ema=args.use_ema)
    schedule = _schedule_from_config(diff_config, device)
    scale = float(diff_config.get("latent_scale", 1.0))
    used, records = set(), []

    for pi, patient in enumerate(candidates):
        if args.max_patients and len(records) >= args.max_patients:
            break
        vols = load_patient_by_path(patient, device=device, load_ct=False, run_segmentation=False)
        if vols.get("pet_nac") is None or vols.get("pet_ac") is None:
            print(f"[skip] unpaired: {patient}")
            continue
        nac_vol = _resize_volume(vols["pet_nac"], args.size)
        ac_vol = _resize_volume(vols["pet_ac"], args.size)
        nac_lat = ae3d_encode(ae, nac_vol, scale=scale)
        x0 = sample_nac_to_ac(
            model, schedule, nac_lat, diff_config,
            num_steps=args.ddim_steps, spacing=args.spacing,
            guidance_scale=args.guidance_scale, clip_x0=None, tag="ft3d",
        )
        pred = ae3d_decode(ae, x0, scale=scale)
        metrics = image_quality_metrics(pred, ac_vol)

        pid = _patient_id(patient, used)
        vols_np = {
            "nac": nac_vol[0, 0].float().cpu().numpy(),
            "pred": pred[0, 0].float().cpu().numpy(),
            "gt": ac_vol[0, 0].float().cpu().numpy(),
        }
        slices, wins, panels = _render_patient(
            vols_np, out_root, pid, planes, args.panel_px, lut, args.interp, args.embed)
        records.append({
            "pid": pid, "path": str(patient), "dataset": Path(patient).parent.name,
            "name": Path(patient).name, "metrics": metrics, "slices": slices,
            "windows": wins, "shape": list(vols_np["gt"].shape), "panels": panels,
        })
        print(f"[ft3d] {len(records)} done (cand {pi+1}/{len(candidates)}) {pid} "
              f"ssim={metrics['ssim']:.4f} psnr={metrics['psnr']:.2f} r2={metrics['voxel_r2']:.3f}")
    return records


@torch.no_grad()
def _predict_diff2d(args, device, candidates, out_root, planes, lut):
    """2D model: predict slice-by-slice, then stack into a volume so the report's
    three-plane viewer works exactly as it does for the 3D model."""
    ae, _ = _load_model(args.ae_ckpt, build_autoencoder_2d, device, use_ema=args.use_ema)
    model, diff_config = _load_model(args.diff_ckpt, build_diffusion_2d, device, use_ema=args.use_ema)
    schedule = _schedule_from_config(diff_config, device)
    scale = float(diff_config.get("latent_scale", 1.0))
    used, records = set(), []

    for pi, patient in enumerate(candidates):
        if args.max_patients and len(records) >= args.max_patients:
            break
        vols = load_patient_by_path(patient, device=device, load_ct=False, run_segmentation=False)
        if vols.get("pet_nac") is None or vols.get("pet_ac") is None:
            print(f"[skip] unpaired: {patient}")
            continue
        z = vols["pet_nac"].shape[0]
        idxs = _positions(z, args.num_slices, lo=0.05, hi=0.95)
        nac_s, pred_s, gt_s, slice_metrics = [], [], [], []
        for s in idxs:
            nac_img = _slice_2d(vols["pet_nac"], s, args.size)
            ac_img = _slice_2d(vols["pet_ac"], s, args.size)
            nac_lat = ae_encode(ae, nac_img, scale=scale)
            x0 = sample_nac_to_ac(
                model, schedule, nac_lat, diff_config,
                num_steps=args.ddim_steps, spacing=args.spacing,
                guidance_scale=args.guidance_scale, clip_x0=args.clip_x0, tag="diff2d",
            )
            pred = ae_decode(ae, x0, scale=scale)
            slice_metrics.append(image_quality_metrics(pred, ac_img))
            nac_s.append(nac_img[0, 0].float().cpu().numpy())
            pred_s.append(pred[0, 0].float().cpu().numpy())
            gt_s.append(ac_img[0, 0].float().cpu().numpy())
        if not slice_metrics:
            continue
        metrics = {}
        for k in METRIC_KEYS:
            vals = [m[k] for m in slice_metrics if math.isfinite(m[k])]
            metrics[k] = float(np.mean(vals)) if vals else float("nan")

        pid = _patient_id(patient, used)
        vols_np = {"nac": np.stack(nac_s), "pred": np.stack(pred_s), "gt": np.stack(gt_s)}
        # The stacked depth is a sparse resample of the body, so only axial slices are
        # anatomically faithful; other planes are still rendered but are coarse in Z.
        slices, wins, panels = _render_patient(
            vols_np, out_root, pid, planes, args.panel_px, lut, args.interp, args.embed)
        records.append({
            "pid": pid, "path": str(patient), "dataset": Path(patient).parent.name,
            "name": Path(patient).name, "metrics": metrics, "slices": slices,
            "windows": wins, "shape": list(vols_np["gt"].shape), "panels": panels,
        })
        print(f"[diff2d] {len(records)} done (cand {pi+1}/{len(candidates)}) {pid} "
              f"ssim={metrics['ssim']:.4f} psnr={metrics['psnr']:.2f}")
    return records


def _aggregate(records):
    out = {}
    for k in METRIC_KEYS:
        vals = [r["metrics"][k] for r in records
                if k in r["metrics"] and math.isfinite(r["metrics"][k])]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                      "median": float(np.median(vals)), "min": float(np.min(vals)),
                      "max": float(np.max(vals)), "n": len(vals)}
    return out


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

CSS = """
*,*::before,*::after{box-sizing:border-box}
:root{
  --bg:#0a0c10; --bg2:#11141b; --card:#141821; --card2:#1a1f2b; --line:#242b39;
  --txt:#e8ecf4; --dim:#8b95a8; --accent:#5eead4; --accent2:#818cf8;
  --good:#34d399; --mid:#fbbf24; --bad:#f87171; --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.3);
}
html{scroll-behavior:smooth}
body{margin:0;background:radial-gradient(1200px 600px at 50% -10%,#151a26 0%,var(--bg) 60%);
  color:var(--txt);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased}
a{color:var(--accent)}
.wrap{max-width:1220px;margin:0 auto;padding:0 24px 80px}
header.hero{padding:56px 0 28px;border-bottom:1px solid var(--line);margin-bottom:28px}
.eyebrow{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);font-weight:600}
h1{margin:.35em 0 .25em;font-size:clamp(26px,3.4vw,40px);line-height:1.1;letter-spacing:-.02em}
h1 .sub{color:var(--dim);font-weight:400}
.lede{color:var(--dim);max-width:74ch;margin:0}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:20px}
.chip{background:var(--card);border:1px solid var(--line);border-radius:999px;padding:5px 13px;
  font-size:12.5px;color:var(--dim);font-variant-numeric:tabular-nums}
.chip b{color:var(--txt);font-weight:600}
h2{font-size:19px;letter-spacing:-.01em;margin:44px 0 14px;display:flex;align-items:center;gap:10px}
h2::before{content:"";width:3px;height:18px;background:linear-gradient(var(--accent),var(--accent2));border-radius:2px}
.note{color:var(--dim);font-size:13.5px;max-width:80ch}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;box-shadow:var(--shadow)}
.stat .k{font-size:11.5px;text-transform:uppercase;letter-spacing:.09em;color:var(--dim)}
.stat .v{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums;margin-top:4px;letter-spacing:-.02em}
.stat .sd{font-size:12px;color:var(--dim);font-variant-numeric:tabular-nums}
.toolbar{position:sticky;top:0;z-index:20;display:flex;flex-wrap:wrap;gap:14px;align-items:center;
  background:rgba(10,12,16,.86);backdrop-filter:blur(12px);border-bottom:1px solid var(--line);
  padding:12px 0;margin:32px 0 20px}
.toolbar label{font-size:12.5px;color:var(--dim);display:flex;align-items:center;gap:7px}
select,input[type=search]{background:var(--card2);color:var(--txt);border:1px solid var(--line);
  border-radius:8px;padding:6px 9px;font:inherit;font-size:13px}
input[type=range]{accent-color:var(--accent)}
.pcard{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px;
  margin-bottom:18px;box-shadow:var(--shadow)}
.phead{display:flex;flex-wrap:wrap;gap:12px;align-items:baseline;justify-content:space-between;margin-bottom:14px}
.pname{font-weight:650;font-size:15.5px;letter-spacing:-.01em}
.pname .ds{color:var(--dim);font-weight:400;font-size:13px}
.pmetrics{display:flex;flex-wrap:wrap;gap:7px}
.m{font-size:12px;font-variant-numeric:tabular-nums;background:var(--card2);border:1px solid var(--line);
  border-radius:7px;padding:3px 9px;color:var(--dim)}
.m b{color:var(--txt);font-weight:600}
.m.good{border-color:rgba(52,211,153,.42)} .m.mid{border-color:rgba(251,191,36,.42)}
.m.bad{border-color:rgba(248,113,113,.42)}
.trip{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.panel{background:#000;border:1px solid var(--line);border-radius:12px;overflow:hidden;position:relative}
.panel img{display:block;width:100%;height:auto;image-rendering:auto}
.plabel{position:absolute;top:8px;left:8px;font-size:11px;letter-spacing:.05em;text-transform:uppercase;
  font-weight:600;background:rgba(6,8,12,.72);border:1px solid rgba(255,255,255,.12);
  border-radius:6px;padding:3px 8px;color:#dfe6f2;backdrop-filter:blur(4px)}
.panel.pred .plabel{color:#a5f3ec;border-color:rgba(94,234,212,.36)}
.pctl{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin-top:13px}
.tabs{display:flex;gap:5px;background:var(--card2);border:1px solid var(--line);border-radius:9px;padding:3px}
.tabs button{background:none;border:0;color:var(--dim);font:inherit;font-size:12.5px;font-weight:600;
  padding:4px 12px;border-radius:6px;cursor:pointer}
.tabs button[aria-pressed=true]{background:var(--accent);color:#04211c}
.slider{flex:1;min-width:180px;display:flex;align-items:center;gap:10px}
.slider input{flex:1}
.slider .idx{font-size:12px;color:var(--dim);font-variant-numeric:tabular-nums;min-width:76px;text-align:right}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:7px 9px;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal}
thead th{position:sticky;top:52px;background:var(--bg2);color:var(--dim);font-size:11.5px;
  text-transform:uppercase;letter-spacing:.07em;font-weight:600;z-index:5}
tbody tr:hover{background:var(--card2)}
tfoot td{font-weight:650;border-top:2px solid var(--line);border-bottom:0}
.tblwrap{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:var(--card)}
.good-t{color:var(--good)} .mid-t{color:var(--mid)} .bad-t{color:var(--bad)}
footer{color:var(--dim);font-size:12.5px;border-top:1px solid var(--line);margin-top:48px;padding-top:20px}
code{background:var(--card2);border:1px solid var(--line);border-radius:5px;padding:1px 5px;font-size:12.5px}
@media (max-width:720px){.trip{grid-template-columns:1fr}.wrap{padding:0 14px 60px}thead th{top:auto;position:static}}
"""

JS = """
const DATA = __DATA__;
const EMBED = __EMBED__;
const q = (s,r=document)=>r.querySelector(s);
const fmt = (v,d=3)=> (v===null||v===undefined||Number.isNaN(v)) ? "\\u2014" : v.toFixed(d);
// Panel source: an inlined data URI in the single-file build, else a relative PNG path.
const srcFor = (rec,plane,k,kind)=>{
  const key = `${plane}_${String(k).padStart(2,"0")}_${kind}`;
  return EMBED ? rec.panels[key] : `img/${rec.pid}/${key}.png`;
};

function cardHTML(rec, i){
  const planes = Object.keys(rec.slices);
  const p0 = planes[0];
  const ms = __CARD_METRICS__.map(k=>{
    const v = rec.metrics[k];
    return `<span class="m ${rec.grades[k]||''}">${__LABELS__[k]} <b>${fmt(v, k==='psnr'?2:3)}</b></span>`;
  }).join("");
  const panels = ["nac","pred","gt"].map(kind=>`
    <div class="panel ${kind}">
      <span class="plabel">${__KINDT__[kind]}</span>
      <img loading="lazy" data-kind="${kind}" alt="${__KINDT__[kind]} — ${rec.name}"
           src="${srcFor(rec,p0,0,kind)}">
    </div>`).join("");
  const tabs = planes.map((pl,j)=>
    `<button type="button" data-plane="${pl}" aria-pressed="${j===0}">${pl}</button>`).join("");
  return `<article class="pcard" data-i="${i}" data-plane="${p0}" data-k="0">
    <div class="phead">
      <div class="pname">#${i+1} &nbsp;${rec.name}<span class="ds"> &nbsp;· ${rec.dataset}</span></div>
      <div class="pmetrics">${ms}</div>
    </div>
    <div class="trip">${panels}</div>
    <div class="pctl">
      <div class="tabs">${tabs}</div>
      <div class="slider">
        <input type="range" min="0" max="${rec.slices[p0].length-1}" value="0" step="1">
        <span class="idx"></span>
      </div>
    </div>
  </article>`;
}

function refresh(card){
  const rec = DATA.patients[+card.dataset.i];
  const plane = card.dataset.plane, k = +card.dataset.k;
  const idxs = rec.slices[plane];
  const kk = Math.min(k, idxs.length-1);
  card.querySelectorAll(".trip img").forEach(img=>{
    img.src = srcFor(rec, plane, kk, img.dataset.kind);
  });
  const r = card.querySelector(".slider input");
  r.max = idxs.length-1; r.value = kk;
  card.querySelector(".idx").textContent = `${plane[0]} ${kk+1}/${idxs.length} · z=${idxs[kk]}`;
  // Warm the neighbouring slices so dragging the slider does not flash blank panels.
  // Pointless when embedded -- those bytes are already in memory.
  if(!EMBED){
    for(const d of [1,-1,2,-2]){
      const j = kk+d; if(j<0 || j>=idxs.length) continue;
      for(const kind of ["nac","pred","gt"]){
        const im = new Image();
        im.src = srcFor(rec, plane, j, kind);
      }
    }
  }
}

function render(){
  const sortKey = q("#sort").value, dir = q("#dir").value === "desc" ? -1 : 1;
  const needle = q("#filter").value.trim().toLowerCase();
  const rows = DATA.patients.map((r,i)=>({r,i}))
    .filter(({r})=> !needle || (r.name+" "+r.dataset).toLowerCase().includes(needle))
    .sort((a,b)=>{
      if(sortKey==="index") return (a.i-b.i)*dir;
      const av=a.r.metrics[sortKey], bv=b.r.metrics[sortKey];
      const an=(av===null||Number.isNaN(av)), bn=(bv===null||Number.isNaN(bv));
      if(an&&bn) return 0; if(an) return 1; if(bn) return -1;
      return (av-bv)*dir;
    });
  q("#cards").innerHTML = rows.map(({r,i})=>cardHTML(r,i)).join("");
  q("#shown").textContent = rows.length;
  document.querySelectorAll(".pcard").forEach(refresh);
}

document.addEventListener("click", e=>{
  const b = e.target.closest(".tabs button"); if(!b) return;
  const card = b.closest(".pcard");
  card.querySelectorAll(".tabs button").forEach(x=>x.setAttribute("aria-pressed", x===b));
  card.dataset.plane = b.dataset.plane;
  refresh(card);
});
document.addEventListener("input", e=>{
  if(e.target.matches(".slider input")){
    const card = e.target.closest(".pcard");
    card.dataset.k = e.target.value; refresh(card);
  } else if(e.target.matches("#filter")){ render(); }
});
document.addEventListener("change", e=>{
  if(e.target.matches("#sort,#dir")) render();
  if(e.target.matches("#syncPlane")){
    const pl = e.target.value;
    document.querySelectorAll(".pcard").forEach(card=>{
      const b = card.querySelector(`.tabs button[data-plane="${pl}"]`); if(!b) return;
      card.querySelectorAll(".tabs button").forEach(x=>x.setAttribute("aria-pressed", x===b));
      card.dataset.plane = pl; refresh(card);
    });
  }
});
render();
"""


def _grade(key, value, summary):
    """Traffic-light class for a per-patient metric, relative to the cohort spread.

    ``z`` is oriented so higher is always better: raw z for higher-is-better metrics,
    negated for error metrics, and distance-from-zero for the signed bias metrics
    (where both a large positive and a large negative value are bad).
    """
    if value is None or not math.isfinite(value) or key not in summary:
        return ""
    mean, sd = summary[key]["mean"], summary[key]["std"]
    if sd <= 0:
        return ""
    if key in SIGNED_ZERO_BEST:
        z = -abs(value) / sd
        return "good" if z >= -0.5 else ("bad" if z <= -1.5 else "mid")
    z = (value - mean) / sd
    if key not in HIGHER_BETTER:
        z = -z
    return "good" if z >= 0.6 else ("bad" if z <= -0.9 else "mid")


def _stat_cards(summary):
    show = [("ssim", 3), ("psnr", 2), ("voxel_r2", 3), ("mae", 4),
            ("rel_bias", 4), ("max_rel_error", 3)]
    out = []
    for k, d in show:
        if k not in summary:
            continue
        s = summary[k]
        out.append(
            f'<div class="stat"><div class="k">{html.escape(METRIC_LABEL[k])}</div>'
            f'<div class="v">{s["mean"]:.{d}f}</div>'
            f'<div class="sd">&plusmn; {s["std"]:.{d}f} SD &nbsp;·&nbsp; '
            f'median {s["median"]:.{d}f}</div></div>')
    return "".join(out)


def _table(records, summary, foot_label=None):
    keys = ["ssim", "psnr", "nrmse", "mae", "voxel_r2", "reg_slope", "rel_bias", "max_rel_error"]
    head = "".join(f"<th>{html.escape(METRIC_LABEL[k])}</th>" for k in keys)
    rows = []
    for i, r in enumerate(records):
        cells = []
        for k in keys:
            v = r["metrics"].get(k, float("nan"))
            g = _grade(k, v, summary)
            cls = {"good": "good-t", "bad": "bad-t", "mid": ""}.get(g, "")
            cells.append(f'<td class="{cls}">{"—" if not math.isfinite(v) else f"{v:.4f}"}</td>')
        rows.append(
            f'<tr><td>{i+1}. {html.escape(r["name"])} '
            f'<span style="color:var(--dim)">· {html.escape(r["dataset"])}</span></td>'
            + "".join(cells) + "</tr>")
    foot = "".join(
        f'<td>{summary[k]["mean"]:.4f}</td>' if k in summary else "<td>&mdash;</td>" for k in keys)
    fl = foot_label or f"Cohort mean (n={len(records)})"
    return (f'<div class="tblwrap"><table><thead><tr><th>Patient</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody>'
            f'<tfoot><tr><td>{html.escape(fl)}</td>{foot}</tr></tfoot></table></div>')


def _cohort_line(cohort):
    """One sentence giving the FULL test-set numbers next to a subset's numbers."""
    s = cohort.get("summary", {})
    bits = []
    for k, d in (("ssim", 3), ("psnr", 2), ("voxel_r2", 3)):
        if k in s:
            bits.append(f"{html.escape(METRIC_LABEL[k])} {s[k]['mean']:.{d}f}"
                        f"&thinsp;&plusmn;&thinsp;{s[k]['std']:.{d}f}")
    if not bits:
        return ""
    return (f'<b>For reference, the full {cohort["n"]}-patient test set scores '
            f'{", ".join(bits)}</b> &mdash; the samples below were chosen to span that '
            f'distribution, not to flatter it.')


def write_html(out_root, records, summary, run, label, out_html=None, cohort=None):
    embed = bool(records and records[0].get("panels"))
    for r in records:
        r["grades"] = {k: _grade(k, r["metrics"].get(k), summary) for k in CARD_METRICS}
    patients = []
    for r in records:
        rec = {"pid": r["pid"], "name": r["name"], "dataset": r["dataset"],
               "slices": r["slices"], "grades": r["grades"],
               "metrics": {k: (None if not math.isfinite(v) else round(v, 6))
                           for k, v in r["metrics"].items()}}
        if embed:
            rec["panels"] = r["panels"]
        patients.append(rec)
    js = (JS.replace("__DATA__", json.dumps({"patients": patients}))
            .replace("__EMBED__", "true" if embed else "false")
            .replace("__CARD_METRICS__", json.dumps(CARD_METRICS))
            .replace("__LABELS__", json.dumps(METRIC_LABEL))
            .replace("__KINDT__", json.dumps(KIND_TITLE)))
    planes = list(records[0]["slices"].keys()) if records else ["axial"]
    sort_opts = '<option value="index">test-set order</option>' + "".join(
        f'<option value="{k}">{html.escape(METRIC_LABEL[k])}</option>' for k in CARD_METRICS)
    plane_opts = "".join(f'<option value="{p}">{p}</option>' for p in planes)
    # Curated header chips; the full checkpoint/split paths live in the footer.
    chip_keys = ["task", "split", "resolution", "steps", "sampler", "EMA", "runtime"]
    chips = "".join(
        f'<span class="chip">{html.escape(k)} <b>{html.escape(str(run[k]))}</b></span>'
        for k in chip_keys if k in run)

    # When only a subset of the split is shown, say so plainly and carry the full-cohort
    # numbers alongside -- a hand-picked handful with no cohort context is how demo
    # figures mislead.
    n_cohort = (cohort or {}).get("n")
    subset = bool(n_cohort) and n_cohort != len(records)
    if subset:
        lede = (f"<b>{len(records)} of the {n_cohort}</b> patients in the by-patient "
                f"<b>test</b> split (held out from both training and validation), rendered "
                f"as a triplet: the non-attenuation-corrected input, the model's predicted "
                f"AC PET, and the real AC PET. Scrub slices with the slider and switch "
                f"plane with the tabs.")
        metrics_head = f"Metrics &mdash; these {len(records)} samples"
        scope = (f"Mean &plusmn; SD over the {len(records)} samples shown. "
                 f"{_cohort_line(cohort)}")
        foot_label = f"Mean of these {len(records)}"
    else:
        lede = ("Every patient in the by-patient <b>test</b> split (held out from both "
                "training and validation), rendered as a triplet: the "
                "non-attenuation-corrected input, the model's predicted AC PET, and the "
                "real AC PET. Scrub slices with the slider and switch plane with the tabs.")
        metrics_head = "Cohort metrics"
        scope = f"Mean &plusmn; SD over {len(records)} test patients."
        foot_label = None

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NAC&rarr;AC test set &mdash; {html.escape(label)}</title>
<style>{CSS}</style></head><body>
<div class="wrap">
<header class="hero">
  <div class="eyebrow">Held-out test set &middot; PET attenuation correction</div>
  <h1>NAC &rarr; AC translation <span class="sub">&mdash; {html.escape(label)}</span></h1>
  <p class="lede">{lede}</p>
  <div class="meta">{chips}</div>
</header>

<h2>{metrics_head}</h2>
<p class="note">{scope} Computed on the foreground of the normalized-intensity volumes
&mdash; <b>not calibrated SUV</b>. <code>rel_bias</code> proxies SUV mean bias;
<code>max_rel_error</code> proxies lesion SUV<sub>max</sub> error and is the known weak
spot of the depth-compressed 3D latent.</p>
<div class="cards">{_stat_cards(summary)}</div>

<div class="toolbar">
  <label>Sort by <select id="sort">{sort_opts}</select></label>
  <label><select id="dir"><option value="desc">high &rarr; low</option>
    <option value="asc">low &rarr; high</option></select></label>
  <label>Plane (all) <select id="syncPlane"><option value="">&mdash;</option>{plane_opts}</select></label>
  <label>Filter <input id="filter" type="search" placeholder="patient / dataset"></label>
  <span class="chip"><b id="shown">{len(records)}</b> shown</span>
</div>

<h2>Triplets &mdash; NAC &middot; predicted AC &middot; ground-truth AC</h2>
<p class="note"><b>Windowing:</b> predicted AC and ground-truth AC share one window taken
from the GT volume (p0.5&ndash;p99.5), so a brightness difference between those two panels
is a real error. NAC has its own window &mdash; it lives on a different intensity scale by
definition. Colour map: <code>{html.escape(str(run.get('colormap', 'hot')))}</code>.<br>
<b>Geometry:</b> panels are shown on the isotropic <code>{html.escape(str(run.get('resolution', '')))}</code>
grid the model actually consumes, so coronal/sagittal aspect ratios are resampled, not
anatomically scaled (a whole-body scan is compressed along the long axis).</p>
<div id="cards"></div>

<h2>Per-patient metrics</h2>
{_table(records, summary, foot_label)}

<footer>
Generated {html.escape(run.get('generated', ''))} by <code>scripts/report_triplets.py</code>.
Model <code>{html.escape(str(run.get('diff_ckpt', '')))}</code> on autoencoder
<code>{html.escape(str(run.get('ae_ckpt', '')))}</code>, sampled at
{html.escape(str(run.get('steps', '')))} steps. Split reloaded from
<code>{html.escape(str(run.get('split_json', '')))}</code>.
Green/red highlighting is relative to the cohort spread, not to a clinical threshold.
</footer>
</div>
<script>{js}</script>
</body></html>
"""
    path = Path(out_html) if out_html else (out_root / "index.html")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #

def _load_cohort(path):
    """Load a previous full-split metrics.json for the reference line + sample picking."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--cohort_json not found: {p}")
    meta = json.loads(p.read_text(encoding="utf-8"))
    return {"n": meta["n_patients_evaluated"], "summary": meta["summary"],
            "per_patient": [_cohort_row(r) for r in meta["per_patient"]]}


def _cohort_row(row):
    """Normalize one cohort row to this report's shape: pid/path/name/dataset/metrics.

    Accepts BOTH this report's own metrics.json (already nested under "metrics") and an
    ``src.training.evaluate`` result, whose rows are flat metric keys plus "patient".
    Reusing the eval json matters: it is the same 41-patient run the arm is judged on, so
    --pick_n spans the distribution the numbers actually came from, and no separate
    full-split report has to be rendered first just to choose a subset.
    """
    if "metrics" in row:
        return row
    path = row.get("patient") or row.get("path") or ""
    metrics = {k: v for k, v in row.items() if k not in ("patient", "path")}
    return {"pid": Path(path).name, "path": str(path), "name": Path(path).name,
            "dataset": Path(path).parent.name, "metrics": metrics}


def _pick_spread(cohort, n_pick, metric):
    """Pick ``n_pick`` patient paths spanning the cohort's ``metric`` distribution.

    Takes the worst, best, and evenly spaced percentiles in between, so a short demo
    report shows the real range of behaviour instead of the top-n cherry pick.
    Returns ``(paths, picked_summary_rows)``.
    """
    rows = [r for r in cohort["per_patient"]
            if math.isfinite(r["metrics"].get(metric, float("nan")))]
    if not rows:
        raise SystemExit(f"No finite '{metric}' values in the cohort json.")
    rows.sort(key=lambda r: r["metrics"][metric])
    n_pick = max(1, min(n_pick, len(rows)))
    idx = sorted({int(round(v)) for v in np.linspace(0, len(rows) - 1, num=n_pick)})
    picked = [rows[i] for i in idx]
    for i, r in zip(idx, picked):
        pct = 100.0 * i / max(len(rows) - 1, 1)
        print(f"  pick {metric}={r['metrics'][metric]:.4f} (p{pct:3.0f} of "
              f"{len(rows)}) {r['name']}")
    return [r["path"] for r in picked], picked


def _rebuild_html(out_root, planes, label):
    """Regenerate index.html from ``<out_root>/metrics.json`` and the existing PNGs.

    Used by ``--html_only`` to iterate on the page without paying for inference again.
    ``slices`` is taken from metrics.json when present, else recomputed from each
    record's volume shape with the same ``_positions`` rule that rendered the PNGs --
    so the plane counts passed here must match the original run.
    """
    meta_path = out_root / "metrics.json"
    if not meta_path.exists():
        raise SystemExit(f"--html_only needs {meta_path} (run without it first).")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    axis_len = {"axial": 0, "coronal": 1, "sagittal": 2}
    records = []
    for r in meta["per_patient"]:
        slices = r.get("slices")
        if not slices:
            slices = {pl: _positions(r["shape"][axis_len[pl]], n) for pl, n in planes}
        records.append({**r, "slices": slices})
    if not records:
        raise SystemExit(f"{meta_path} has no per-patient records.")
    return write_html(out_root, records, meta["summary"], meta["run"], label)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="ft3d", choices=["ft3d", "diff2d"])
    p.add_argument("--data_dir", required=True, nargs="+")
    p.add_argument("--ae_ckpt", required=True)
    p.add_argument("--diff_ckpt", required=True)
    p.add_argument("--split_json", default=None)
    p.add_argument("--mode", default="test", choices=["test", "val", "all"])
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--test_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=None, help="MUST match training (ft3d 64, diff2d 128)")
    p.add_argument("--num_slices", type=int, default=32, help="diff2d only: slices predicted per patient")
    p.add_argument("--ddim_steps", type=int, default=16)
    p.add_argument("--spacing", choices=["linear", "karras"], default="linear")
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--clip_x0", type=float, default=4.0, help="diff2d/epsilon only")
    p.add_argument("--max_patients", type=int, default=0, help="0 = whole split")
    p.add_argument("--use_ema", dest="use_ema", action="store_true", default=True)
    p.add_argument("--no_ema", dest="use_ema", action="store_false")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--label", default=None, help="Model name shown in the report title")
    p.add_argument("--panel_px", type=int, default=256, help="Rendered panel size in px")
    # Panels rendered per plane. A count >= the number of slices renders EVERY slice
    # (so the slider scrubs the whole volume); a smaller count samples the central span.
    # 0 disables the plane. Defaults cover a 64^3 volume completely.
    p.add_argument("--n_axial", type=int, default=64)
    p.add_argument("--n_coronal", type=int, default=64)
    p.add_argument("--n_sagittal", type=int, default=64)
    p.add_argument("--colormap", default="hot", choices=["hot", "gray", "magma"])
    p.add_argument("--interp", default="smooth", choices=["smooth", "nearest"],
                   help="Display upscaling of each panel: 'smooth' (LANCZOS on the "
                        "windowed float, like any DICOM viewer) or 'nearest' (raw voxel "
                        "grid, no invented values). Applied identically to all 3 panels.")
    p.add_argument("--embed", action="store_true",
                   help="Write ONE self-contained .html with every panel inlined as a "
                        "base64 data URI -- no img/ folder, so the single file can be "
                        "emailed or uploaded anywhere. Keep the patient count and panel "
                        "counts modest; the file grows by ~1.4x the raw PNG bytes.")
    p.add_argument("--out_html", default=None,
                   help="Explicit output .html path (default <out_dir>/index.html).")
    p.add_argument("--cohort_json", default=None,
                   help="A previous FULL-split metrics.json. Its numbers are shown next to "
                        "a subset's, so a short report still states the whole-cohort result.")
    p.add_argument("--pick_n", type=int, default=0,
                   help="With --cohort_json: show only N patients, chosen to span the "
                        "--pick_metric distribution (worst .. best) instead of the top N.")
    p.add_argument("--pick_metric", default="ssim",
                   help="Metric whose distribution --pick_n spans (default ssim).")
    p.add_argument("--html_only", action="store_true",
                   help="Rebuild index.html from an existing <out_dir>/metrics.json + the "
                        "already-rendered PNGs, without re-running inference. The panel "
                        "counts (--n_axial/--n_coronal/--n_sagittal) and --colormap must "
                        "match the run that rendered the PNGs. Folder builds only -- an "
                        "--embed report cannot be rebuilt this way (metrics.json "
                        "deliberately does not store the base64 panels); re-run instead.")
    args = p.parse_args()

    if args.size is None:
        args.size = 64 if args.task == "ft3d" else 128
    if args.clip_x0 is not None and args.clip_x0 <= 0:
        args.clip_x0 = None
    if args.out_dir is None:
        args.out_dir = os.path.join("outputs", "report", Path(args.diff_ckpt).parent.name)
    label = args.label or Path(args.diff_ckpt).parent.name
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    planes = [("axial", args.n_axial), ("coronal", args.n_coronal), ("sagittal", args.n_sagittal)]
    planes = [(p_, n) for p_, n in planes if n > 0]

    if args.html_only:
        page = _rebuild_html(out_root, planes, label)
        print(f"Wrote {page} (html_only; PNGs untouched)")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device} task={args.task} mode={args.mode} size={args.size} "
          f"steps={args.ddim_steps} out={out_root}")

    cohort = _load_cohort(args.cohort_json)
    candidates, n_total = _resolve_candidates(
        args.data_dir, args.mode, args.val_fraction, args.test_fraction, args.seed,
        split_json=args.split_json)
    if args.mode == "all":
        candidates = _subsample(candidates, (args.max_patients or 0) * 3)
    print(f"{n_total} {args.mode}-set -> {len(candidates)} candidates")

    if args.pick_n:
        if cohort is None:
            raise SystemExit("--pick_n needs --cohort_json (it picks from those metrics).")
        print(f"Picking {args.pick_n} patients spanning the {args.pick_metric} distribution:")
        wanted, _ = _pick_spread(cohort, args.pick_n, args.pick_metric)
        wanted_set = {str(w) for w in wanted}
        candidates = [c for c in candidates if str(c) in wanted_set]
        if len(candidates) != len(wanted):
            print(f"WARNING: {len(wanted) - len(candidates)} picked patients are not "
                  f"present under the given --data_dir roots.")
        if not candidates:
            raise SystemExit("None of the picked patients resolved under --data_dir.")

    lut = _lut(args.colormap)

    t0 = time.time()
    runner = _predict_ft3d if args.task == "ft3d" else _predict_diff2d
    records = runner(args, device, candidates, out_root, planes, lut)
    if not records:
        raise SystemExit("No patients evaluated (all skipped?). Check pairing / roots.")
    summary = _aggregate(records)
    elapsed = time.time() - t0

    run = {
        "model": label, "task": args.task, "split": f"{args.mode} (n={len(records)})",
        "resolution": f"{args.size}³" if args.task == "ft3d" else f"{args.size}²",
        "steps": args.ddim_steps, "sampler": args.spacing, "EMA": str(args.use_ema),
        "colormap": args.colormap, "interp": args.interp,
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "runtime": f"{elapsed/60:.1f} min",
        "diff_ckpt": args.diff_ckpt, "ae_ckpt": args.ae_ckpt,
        "split_json": args.split_json or "(fractions+seed)",
    }
    if cohort and cohort["n"] != len(records):
        run["split"] = f"{args.mode} ({len(records)} of {cohort['n']} shown)"
    # metrics.json never carries the base64 panels -- it stays a small, diffable record.
    metrics_path = out_root / "metrics.json"
    metrics_path.write_text(json.dumps({
        "run": run, "n_patients_evaluated": len(records), "summary": summary,
        # 'slices' is persisted so --html_only can rebuild the page verbatim.
        "per_patient": [{k: r[k] for k in ("pid", "path", "dataset", "name", "metrics",
                                           "windows", "shape", "slices")}
                        for r in records],
    }, indent=2), encoding="utf-8")

    page = write_html(out_root, records, summary, run, label,
                      out_html=args.out_html, cohort=cohort)
    print(f"\n=== {args.mode}-set summary (mean +/- SD over {len(records)} patients) ===")
    for k in METRIC_KEYS:
        if k in summary:
            print(f"  {k:14s} {summary[k]['mean']:.4f} +/- {summary[k]['std']:.4f}  (n={summary[k]['n']})")
    kb = page.stat().st_size / 1024
    size = f"{kb/1024:.1f} MB" if kb > 1024 else f"{kb:.0f} KB"
    print(f"\nWrote {page} ({size}{', self-contained' if args.embed else ''})"
          f"\nWrote {metrics_path}\nElapsed {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
