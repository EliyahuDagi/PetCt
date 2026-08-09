"""Gradio front-end for the Train Viewer (replaces the old Tkinter view).

Mirrors the behaviour of the previous TrainView/TrainPresenter pair but uses
sliders instead of mouse drags for window/level + slice navigation, a native
gr.LinePlot for the metric curves, and a gr.Timer for the live poll loop.

The UI logic stays thin: all stateful work lives in TrainController; this module
just wires Gradio components to controller methods and formats arrays for display.

Visual layer is a dark "OLED" clinical dashboard: a Base theme + custom CSS
(stashed on demo._theme_kwargs for launch(), as Gradio 6 requires theme/css/js
at launch() rather than the Blocks constructor).
"""

import numpy as np
import gradio as gr

try:
    # matplotlib only used for the diverging colormap on the diff image (no figure).
    from matplotlib import colormaps as _mpl_cm
except Exception:  # pragma: no cover - matplotlib should be present
    _mpl_cm = None

from src.TrainViewer.model import TrainModel
from src.TrainViewer.controller import (
    TrainController,
    METRIC_KEYS,
    LIVE_RUN_LABEL,
    _format_smoke_report,
    _to_int,
    _to_float,
)

# Task selector display labels -> internal keys (mirrors the old view).
TASK_DISPLAY = ["AE 2D (encode/decode)", "AE 3D (encode/decode)", "NAC->AC 2D", "NAC->AC 3D"]
# Label shown next to the size field, per task.
SIZE_LABEL = {"ae2d": "slice_size", "ae3d": "crop_size", "diff2d": "latent_size", "ft3d": "latent_size"}

# Model-variant control: display label -> internal variant key.
VARIANT_DISPLAY = ["Diffusion (ε)", "Flow"]
VARIANT_MAP = {"Diffusion (ε)": "eps", "Flow": "flow"}
# Flow-first: the whole point of this build is applying the flow model.
DEFAULT_VARIANT_LABEL = "Flow"

# Tasks the model-variant (eps/flow) concept applies to.
_DIFFUSION_TASK_KEYS = {"diff2d", "ft3d"}

# How often the live poll loop fires (seconds); matches the old 750ms tick.
POLL_INTERVAL = 0.75


# === Theme / styling (dark, black-canvas medical look) ===

def _dark_theme():
    """A cyan-on-black clinical Gradio theme with an amber accent CTA. Built
    lazily so importing this module without gradio's theme assets never fails at
    import time."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.cyan,
        secondary_hue=gr.themes.colors.amber,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Fira Sans"), gr.themes.GoogleFont("Inter"),
              "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ).set(
        body_background_fill_dark="#05070a",
        background_fill_primary_dark="#0b1017",
        background_fill_secondary_dark="#0b1017",
        block_background_fill_dark="#0e141c",
        block_border_color_dark="#1a2531",
        block_border_width="1px",
        block_radius="12px",
        block_label_text_color_dark="#8194a5",
        block_title_text_color_dark="#d5e1ec",
        body_text_color_dark="#d5e1ec",
        body_text_color_subdued_dark="#8194a5",
        border_color_primary_dark="#1a2531",
        border_color_accent_dark="#24333f",
        # Amber accent CTA (primary buttons) on near-black text.
        button_primary_background_fill_dark="#f59e0b",
        button_primary_background_fill_hover_dark="#fbbf24",
        button_primary_text_color_dark="#0a0e14",
        button_primary_border_color_dark="#f59e0b",
        # Secondary buttons: quiet surface with a strong border.
        button_secondary_background_fill_dark="#0e141c",
        button_secondary_background_fill_hover_dark="#141c26",
        button_secondary_border_color_dark="#24333f",
        button_secondary_text_color_dark="#d5e1ec",
        input_background_fill_dark="#0b1017",
        input_border_color_dark="#1a2531",
        input_border_color_focus_dark="#22d3ee",
        slider_color_dark="#22d3ee",
    )


# Inline logo mark (stylized scan crosshair + waveform), cyan stroke, ~28px.
_LOGO_SVG = (
    '<svg width="30" height="30" viewBox="0 0 32 32" fill="none" '
    'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<rect x="2.5" y="2.5" width="27" height="27" rx="7" stroke="#22d3ee" '
    'stroke-width="1.4" opacity="0.45"/>'
    '<circle cx="16" cy="16" r="8.5" stroke="#22d3ee" stroke-width="1.4" opacity="0.8"/>'
    '<path d="M1.5 16h4M26.5 16h4M16 1.5v4M16 26.5v4" stroke="#22d3ee" '
    'stroke-width="1.4" stroke-linecap="round"/>'
    '<path d="M8.5 17.5c1.8-6.5 3.3 5.5 5.5-0.5 1.2-3.3 2 6 3.5 1 0.9-2.4 1.6 2 3.5 1" '
    'stroke="#67e8f9" stroke-width="1.5" stroke-linecap="round" '
    'stroke-linejoin="round" fill="none"/>'
    "</svg>"
)

# Status-pill colors keyed by state.
_PILL_COLORS = {
    "idle": "#64748b",
    "running": "#f59e0b",
    "done": "#34d399",
    "error": "#f87171",
}


APP_CSS = """
:root {
  --bg:#05070a; --surface:#0b1017; --card:#0e141c;
  --border:#1a2531; --border-strong:#24333f;
  --text:#d5e1ec; --muted:#8194a5;
  --cyan:#22d3ee; --cyan-hi:#67e8f9;
  --amber:#f59e0b; --amber-hi:#fbbf24; --on-accent:#0a0e14;
  --ok:#34d399; --warn:#f59e0b; --err:#f87171; --idle:#64748b;
}
.gradio-container { max-width: 100% !important; }

/* --- Header / hero bar --- */
.app-header {
  justify-content: space-between !important;
  align-items: center !important;
  gap: 14px !important;
  background: linear-gradient(180deg, #0b1017 0%, #080c12 100%);
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 12px 18px !important;
  margin-bottom: 6px;
}
.app-header > * { flex: 0 1 auto !important; min-width: 0 !important; }
.brand { display: flex; align-items: center; gap: 13px; }
.brand-mark { display: inline-flex; align-items: center; }
.brand-text { display: flex; flex-direction: column; line-height: 1.15; }
.brand-title { font-family: 'Fira Sans', system-ui, sans-serif; font-weight: 600;
  font-size: 20px; letter-spacing: .3px; color: var(--text); }
.brand-sub { font-size: 12px; color: var(--muted); letter-spacing: .2px; margin-top: 2px; }

/* --- Status pill (text + colored dot; never color-alone) --- */
.status-pill {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 5px 13px; border-radius: 999px;
  border: 1px solid var(--pill-color, var(--idle));
  background: rgba(148,163,184,0.10);
  color: var(--text); font-family: 'JetBrains Mono', monospace;
  font-size: 11px; letter-spacing: .6px; text-transform: uppercase;
  white-space: nowrap;
}
.pill-dot { width: 8px; height: 8px; border-radius: 50%;
  background: var(--pill-color, var(--idle));
  box-shadow: 0 0 8px var(--pill-color, var(--idle)); flex: 0 0 auto; }
.pill-text { line-height: 1; }
@media (prefers-reduced-motion: no-preference) {
  .pill-dot.pulse { animation: pill-pulse 1.4s ease-in-out infinite; }
  @keyframes pill-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: .45; transform: scale(.8); }
  }
}

/* --- Section labels + control cards --- */
.section-label { font-family: 'Fira Sans', system-ui, sans-serif; font-weight: 600;
  font-size: 12px; letter-spacing: .8px; text-transform: uppercase;
  color: var(--cyan); margin: 2px 0 4px; }
.ctrl-card { background: var(--card) !important; border: 1px solid var(--border) !important;
  border-radius: 12px !important; padding: 12px 14px !important; }

/* --- Scan panels (cyan glow on black canvas) --- */
#scan-row { gap: 12px; }
.scan-img, .scan-img > div, .scan-img img { background: #000 !important; }
.scan-img { border: 1px solid var(--border-strong) !important; border-radius: 12px;
  box-shadow: 0 0 24px rgba(34,211,238,0.08); overflow: hidden;
  transition: box-shadow 160ms ease-out; }
.scan-img:hover { box-shadow: 0 0 30px rgba(34,211,238,0.16); }
.scan-img img { object-fit: contain !important; image-rendering: auto; }
/* labels as small chips on the panels (best-effort across gradio label markup) */
.scan-img label, .scan-img .block-label {
  background: rgba(11,16,23,0.88) !important; color: var(--cyan-hi) !important;
  border: 1px solid var(--border-strong) !important; border-radius: 999px !important;
  font-size: 11px !important; letter-spacing: .5px; padding: 2px 10px !important; }

/* --- Metric stat tiles --- */
.stat-row { display: flex; gap: 12px; margin-top: 12px; }
.stat-tile { flex: 1 1 0; background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 12px 16px; text-align: left;
  transition: border-color 160ms ease-out; }
.stat-tile:hover { border-color: var(--border-strong); }
.stat-label { font-family: 'JetBrains Mono', monospace; font-size: 11px;
  letter-spacing: 1.2px; text-transform: uppercase; color: var(--muted); }
.stat-value { font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums;
  font-size: 26px; font-weight: 600; color: var(--cyan-hi); margin-top: 4px; line-height: 1.1; }
.stat-unit { font-size: 13px; color: var(--muted); margin-left: 5px; font-weight: 400; }

/* --- mono, tabular numerals for readouts --- */
#metrics-bar, .stat-value, .status-pill, .ckpt-info { font-variant-numeric: tabular-nums; }
.ckpt-info { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--muted); }

/* --- Buttons / focus / motion --- */
button { cursor: pointer; transition: filter 160ms ease-out, box-shadow 160ms ease-out; }
button.primary, button[variant="primary"] { font-weight: 600; letter-spacing: .3px; }
button:focus-visible, a:focus-visible, input:focus-visible,
select:focus-visible, [tabindex]:focus-visible {
  outline: 2px solid var(--amber); outline-offset: 2px; }
"""

# Force the dark palette regardless of the browser/OS preference.
FORCE_DARK_JS = """
function () {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""


# === HTML fragment builders (header, status pill, metric tiles) ===

def _brand_html():
    """The header brand block: logo mark + title + muted subtitle (no emoji)."""
    return (
        '<div class="brand">'
        f'<span class="brand-mark">{_LOGO_SVG}</span>'
        '<span class="brand-text">'
        '<span class="brand-title">PET-CT Studio</span>'
        '<span class="brand-sub">NAC→AC PET latent-diffusion &middot; '
        'training &amp; inference</span>'
        "</span>"
        "</div>"
    )


def _status_pill_html(state="idle", text=None):
    """A pill with a colored (optionally pulsing) dot + text label. The text
    carries the meaning too, so state is never conveyed by color alone."""
    color = _PILL_COLORS.get(state, _PILL_COLORS["idle"])
    label = (text if text is not None else state)
    dot_cls = "pill-dot pulse" if state == "running" else "pill-dot"
    return (
        f'<div class="status-pill" data-state="{state}" style="--pill-color:{color}">'
        f'<span class="{dot_cls}"></span>'
        f'<span class="pill-text">{label}</span>'
        "</div>"
    )


def _pill_for_status(status):
    """Map a training/smoke status string to a status-pill update (or no-op)."""
    if status is None:
        return gr.update()
    s = str(status)
    low = s.lower()
    if "running" in low or "stopping" in low:
        return _status_pill_html("running", "training" if "running" in low else "stopping")
    if "exit code 0" in low or "smoke ok" in low or low == "done":
        return _status_pill_html("done", "done")
    if "error" in low or "fail" in low or "exit code" in low:
        return _status_pill_html("error", "error")
    return _status_pill_html("idle", s)


# === Display helpers ===

# Orientation labels for the View tab radio (mirrors DataViewer's three planes).
ORIENTATIONS = ["AXIAL", "CORONAL", "SAGITTAL"]
# Volume axis (for a (Z, Y, X) array) the slice slider scans, per orientation.
_ORIENT_AXIS = {"AXIAL": 0, "CORONAL": 1, "SAGITTAL": 2}


def _orient_key(label):
    """Map the View tab radio label ('Axial'/'Coronal'/'Sagittal') to the internal
    uppercase orientation key used by `_slice_of`/`_axis_len`."""
    key = str(label or "Axial").upper()
    return key if key in _ORIENT_AXIS else "AXIAL"


def _variant_key(label):
    """Map the model-variant radio label to the internal 'eps'/'flow' key."""
    return VARIANT_MAP.get(label, "eps")


def _cta_label(task, variant):
    """Dynamic primary-button text for the current task + variant."""
    if task in _DIFFUSION_TASK_KEYS:
        return "Apply Flow Model" if variant == "flow" else "Run Diffusion (ε)"
    return "Run AE inference"


def _axis_len(vol, orientation):
    """Length of the slider axis for `orientation` on a (Z, Y, X) volume.

    Returns None for non-3D inputs (2D ae2d/diff2d arrays), where there is no
    orientation axis to scan.
    """
    if vol is None:
        return None
    vol = np.asarray(vol)
    if vol.ndim != 3:
        return None
    return vol.shape[_ORIENT_AXIS.get(orientation, 0)]


def _slice_of(vol, idx, orientation="AXIAL"):
    """Pick a 2D frame from a (Z, Y, X) volume for the given orientation, matching
    DataViewer's conventions (axial = vol[idx], coronal/sagittal flipped vertically).

    A 2D input array (ae2d/diff2d) is returned unchanged regardless of orientation.
    """
    if vol is None:
        return None
    vol = np.asarray(vol)
    if vol.ndim != 3:
        return vol
    if orientation == "CORONAL":
        idx = max(0, min(int(idx), vol.shape[1] - 1))
        # Volume is (Z, Y, X). Coronal is (Z, X) at fixed Y; flipud to match DataViewer.
        return np.flipud(vol[:, idx, :])
    if orientation == "SAGITTAL":
        idx = max(0, min(int(idx), vol.shape[2] - 1))
        # Volume is (Z, Y, X). Sagittal is (Z, Y) at fixed X; flipud to match DataViewer.
        return np.flipud(vol[:, :, idx])
    # AXIAL (default): vol[idx, :, :].
    idx = max(0, min(int(idx), vol.shape[0] - 1))
    return vol[idx]


def _gray_uint8(frame, vmin, vmax):
    """Window a 2D float frame to [vmin, vmax] -> uint8 grayscale (HxW)."""
    if frame is None:
        return None
    frame = np.asarray(frame, dtype=np.float32)
    span = max(1e-6, float(vmax) - float(vmin))
    norm = (frame - float(vmin)) / span
    norm = np.clip(norm, 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


# Robust percentile for the diff colormap's symmetric range (see _diff_rgb).
_DIFF_PCTL = 99.0


def _diff_rgb(frame_pred, frame_gt):
    """Map (pred - gt) through a diverging colormap symmetric around 0 -> RGB uint8.

    The colour range is robust: vmax = the 99th percentile of |pred - gt| and
    vmin = -vmax, so a handful of noisy voxels no longer saturate the whole map
    and a zero difference (including empty background, which is ~0 in both PET
    volumes) lands on the neutral mid colour of RdBu_r. This replaces the old
    max-abs / bwr scaling that rendered as a harsh red/white/blue speckle.
    """
    if frame_pred is None or frame_gt is None:
        return None
    diff = np.asarray(frame_pred, dtype=np.float32) - np.asarray(frame_gt, dtype=np.float32)
    if diff.size:
        dmax = float(np.percentile(np.abs(diff), _DIFF_PCTL))
    else:
        dmax = 0.0
    if not np.isfinite(dmax) or dmax <= 0:
        dmax = 1.0
    norm = (diff / dmax + 1.0) * 0.5  # [-dmax, dmax] -> [0, 1]; 0.5 == zero diff
    norm = np.clip(norm, 0.0, 1.0)
    if _mpl_cm is not None:
        # RdBu_r: blue (pred<gt) -> near-white (pred==gt) -> red (pred>gt).
        rgba = _mpl_cm["RdBu_r"](norm)  # HxWx4 float in [0,1]
        return (rgba[..., :3] * 255.0).astype(np.uint8)
    # Fallback diverging map (blue<0, near-white at 0, red>0) w/o matplotlib.
    r = np.clip(norm * 2.0, 0.0, 1.0)
    b = np.clip((1.0 - norm) * 2.0, 0.0, 1.0)
    g = 1.0 - np.abs(norm - 0.5) * 2.0
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _compute_metrics(pred_frame, gt_frame):
    """Return (mae, rmse, psnr_str) between two 2D frames. psnr_str has no unit
    ('∞' when RMSE/range is degenerate)."""
    p = np.asarray(pred_frame, dtype=np.float32)
    g = np.asarray(gt_frame, dtype=np.float32)
    diff = p - g
    mae = float(np.mean(np.abs(diff))) if diff.size else 0.0
    rmse = float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0
    drange = float(np.max(g) - np.min(g)) if g.size else 0.0
    if rmse > 0 and drange > 0:
        psnr_str = f"{20.0 * np.log10(drange / rmse):.2f}"
    else:
        psnr_str = "∞"
    return mae, rmse, psnr_str


def _slice_metrics_md(pred_frame, gt_frame):
    """MAE / RMSE / PSNR between the displayed pred & gt slice, as markdown.

    Kept for backward compatibility; the View tab renders `_metrics_tiles_html`.
    """
    if pred_frame is None or gt_frame is None:
        return "—"
    mae, rmse, psnr_str = _compute_metrics(pred_frame, gt_frame)
    psnr_disp = psnr_str if psnr_str == "∞" else f"{psnr_str} dB"
    return f"**MAE** {mae:.4g}  ·  **RMSE** {rmse:.4g}  ·  **PSNR** {psnr_disp}"


def _stat_tile(label, value, unit=""):
    unit_html = f'<span class="stat-unit">{unit}</span>' if unit else ""
    return (
        '<div class="stat-tile">'
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}{unit_html}</div>'
        "</div>"
    )


def _metrics_tiles_html(pred_frame, gt_frame):
    """Three big-number stat tiles (MAE / RMSE / PSNR) for the displayed slice."""
    if pred_frame is None or gt_frame is None:
        tiles = (
            _stat_tile("MAE", "—")
            + _stat_tile("RMSE", "—")
            + _stat_tile("PSNR", "—", "dB")
        )
        return f'<div class="stat-row">{tiles}</div>'
    mae, rmse, psnr_str = _compute_metrics(pred_frame, gt_frame)
    tiles = (
        _stat_tile("MAE", f"{mae:.4g}")
        + _stat_tile("RMSE", f"{rmse:.4g}")
        + _stat_tile("PSNR", psnr_str, "" if psnr_str == "∞" else "dB")
    )
    return f'<div class="stat-row">{tiles}</div>'


def _render_panels(nac_vol, pred_vol, gt_vol, idx, vmin, vmax, orientation="AXIAL"):
    """Return (nac_img, pred_img, gt_img, diff_img, metrics_html) for the four
    gr.Image panels + the stat-tile readout.

    All outputs derive from the same oriented frame so NAC/pred/gt/diff/metrics
    stay in lockstep with the slice + orientation selectors. NAC is a PET volume
    too, so it uses the same grayscale windowing as pred/gt. When nac_vol is None
    (AE tasks, or no nac.npy) the NAC image is None (blank) -- the panel is hidden
    by the run-inference handler in that case, so it never shows a stale frame.
    """
    nac = _slice_of(nac_vol, idx, orientation)
    pred = _slice_of(pred_vol, idx, orientation)
    gt = _slice_of(gt_vol, idx, orientation)
    return (
        _gray_uint8(nac, vmin, vmax),
        _gray_uint8(pred, vmin, vmax),
        _gray_uint8(gt, vmin, vmax),
        _diff_rgb(pred, gt),
        _metrics_tiles_html(pred, gt),
    )


def _curves_df(rows, metric_key):
    """Build a tidy dict-of-lists (step, value, phase) for gr.LinePlot."""
    steps, values, phases = [], [], []
    for r in rows or []:
        if metric_key not in r or r.get(metric_key) is None:
            continue
        step = r.get("step")
        if step is None:
            continue
        steps.append(step)
        values.append(r[metric_key])
        phases.append("val" if r.get("phase") == "val" else "train")
    return {"step": steps, "value": values, "phase": phases}


def _run_choices(options):
    """gr.Dropdown choices as (label, run_id) pairs; run_id None == live option."""
    return [(o["label"], o["run_id"]) for o in options]


# === App ===

def build_app(model=None):
    """Build and return the Gradio Blocks app for the Train Viewer."""
    model = model or TrainModel()
    controller = TrainController(model)
    cfg = model.config

    # Initial values pulled from config (mirrors the old view.load_config).
    init_dirs = list(cfg.data_dirs) if cfg.data_dirs else ([cfg.data_dir] if cfg.data_dir else [])
    init_size_n = controller.refresh_dataset_size(init_dirs)
    init_run_options = controller.refresh_runs()
    init_size_label = SIZE_LABEL.get(controller.task, "slice_size")
    init_variant = _variant_key(DEFAULT_VARIANT_LABEL)
    init_cta = _cta_label(controller.task, init_variant)

    def _size_text(n):
        return "Dataset: — patients" if n is None else f"Dataset: {int(n)} patients"

    # Gradio 6 moved theme/css/js from the Blocks constructor to launch(); we
    # build them here and stash them on the returned demo for launch() to apply.
    with gr.Blocks(title="Pet-CT Train Viewer") as demo:
        # --- shared state for the View tab (loaded volumes + window/level) ---
        st_pred = gr.State(None)
        st_gt = gr.State(None)
        # NAC input volume (diff2d/ft3d only); None hides the NAC panel.
        st_nac = gr.State(None)
        # Append-only log buffer (we keep it in a State so the Timer can extend it).
        st_log = gr.State("")

        # === Header / hero bar: brand + live status pill ===
        with gr.Row(elem_classes="app-header"):
            gr.HTML(_brand_html())
            status_pill = gr.HTML(_status_pill_html("idle", "idle"))

        # Shared task selector (drives Train / Log / View; single source of truth).
        task_dd = gr.Dropdown(
            choices=TASK_DISPLAY, value=TASK_DISPLAY[0], label="Task", interactive=True
        )

        with gr.Tabs():
            # === Train tab ===
            with gr.Tab("Train"):
                with gr.Group(elem_classes="ctrl-card"):
                    gr.Markdown("Dataset", elem_classes="section-label")
                    # Multi-root data_dirs editor (single-column dataframe + add/remove).
                    dirs_df = gr.Dataframe(
                        headers=["root"],
                        value=[[d] for d in init_dirs] or [[""]],
                        column_count=(1, "fixed"),
                        datatype="str",
                        interactive=True,
                        label="data_dirs (dataset roots)",
                    )
                    size_label = gr.Markdown(_size_text(init_size_n))

                with gr.Group(elem_classes="ctrl-card"):
                    gr.Markdown("Hyperparameters", elem_classes="section-label")
                    with gr.Row():
                        patient_index = gr.Number(value=cfg.patient_index, label="patient_index", precision=0)
                        epochs = gr.Number(value=cfg.epochs, label="epochs", precision=0)
                        val_every = gr.Number(value=cfg.val_every, label="val_every (steps)", precision=0)
                    with gr.Row():
                        batch_size = gr.Number(value=cfg.batch_size, label="batch_size", precision=0)
                        learning_rate = gr.Number(value=cfg.learning_rate, label="learning_rate")
                        size = gr.Number(value=cfg.size, label=init_size_label, precision=0)
                    with gr.Row():
                        val_fraction = gr.Number(value=cfg.val_fraction, label="val_fraction (0-1)")
                        device = gr.Dropdown(choices=["cuda", "cpu"], value=cfg.device, label="device")

                with gr.Group(elem_classes="ctrl-card"):
                    gr.Markdown("WSL settings", elem_classes="section-label")
                    distro = gr.Textbox(value=cfg.distro, label="distro")
                    venv_path = gr.Textbox(value=cfg.venv_path, label="venv_path")
                    project_path = gr.Textbox(value=cfg.project_path, label="project_path")

                with gr.Row():
                    start_btn = gr.Button("Start Training", variant="primary")
                    stop_btn = gr.Button("Stop")
                    smoke_btn = gr.Button("Smoke Test")
                status = gr.Textbox(value="stopped", label="Status", interactive=False)

            # === Log tab ===
            with gr.Tab("Log"):
                with gr.Group(elem_classes="ctrl-card"):
                    gr.Markdown("Run &amp; metric", elem_classes="section-label")
                    with gr.Row():
                        run_dd = gr.Dropdown(
                            choices=_run_choices(init_run_options),
                            value=None,
                            label="run",
                            interactive=True,
                        )
                        metric_dd = gr.Dropdown(
                            choices=METRIC_KEYS, value="loss", label="metric", interactive=True
                        )
                curve = gr.LinePlot(
                    value=_curves_df([], "loss"),
                    x="step",
                    y="value",
                    color="phase",
                    x_title="step",
                    y_title="loss",
                    height=320,
                )
                log_box = gr.Textbox(
                    value="", label="log", lines=20, max_lines=20,
                    interactive=False, autoscroll=True,
                )

            # === View tab (clinical workstation) ===
            with gr.Tab("View"):
                with gr.Row():
                    # --- left: Model & sample control card ---
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes="ctrl-card"):
                            gr.Markdown("Model &amp; sample", elem_classes="section-label")
                            variant_radio = gr.Radio(
                                choices=VARIANT_DISPLAY, value=DEFAULT_VARIANT_LABEL,
                                label="Model", interactive=True,
                            )
                            infer_data_dir = gr.Textbox(
                                value="", label="infer data_dir (blank = use train roots)"
                            )
                            infer_patient = gr.Number(
                                value=cfg.infer_patient_index, label="patient_index", precision=0
                            )
                            with gr.Row():
                                orient_radio = gr.Radio(
                                    choices=["Axial", "Coronal", "Sagittal"], value="Axial",
                                    label="orientation", interactive=True,
                                )
                                infer_device = gr.Dropdown(
                                    choices=["cuda", "cpu"], value=cfg.device, label="device"
                                )
                            # Sampling controls that switch with the variant.
                            flow_steps = gr.Slider(
                                minimum=2, maximum=50, step=1, value=25,
                                label="Flow steps", visible=(init_variant == "flow"),
                            )
                            ddim_steps = gr.Slider(
                                minimum=5, maximum=50, step=1, value=25,
                                label="DDIM steps", visible=(init_variant == "eps"),
                            )
                            guidance = gr.Slider(
                                minimum=1.0, maximum=5.0, step=0.1, value=1.0,
                                label="Guidance", visible=(init_variant == "eps"),
                            )
                            run_infer_btn = gr.Button(init_cta, variant="primary")
                            with gr.Row():
                                infer_status = gr.Textbox(
                                    value="idle", label="Status", interactive=False, scale=2
                                )
                                load_best_btn = gr.Button("Refresh checkpoint", scale=1)
                            ckpt_info = gr.Markdown(
                                controller.refresh_checkpoint(init_variant),
                                elem_classes="ckpt-info",
                            )

                    # --- right: viewport (scan panels + slice + metrics) ---
                    with gr.Column(scale=3):
                        with gr.Row(elem_id="scan-row", equal_height=True):
                            img_nac = gr.Image(label="NAC input", interactive=False, height=460,
                                               elem_classes="scan-img", visible=False)
                            img_pred = gr.Image(label="Predicted AC", interactive=False, height=460,
                                                elem_classes="scan-img")
                            img_gt = gr.Image(label="Ground-truth AC", interactive=False, height=460,
                                              elem_classes="scan-img")
                            img_diff = gr.Image(label="Difference (pred − gt)", interactive=False,
                                                height=460, elem_classes="scan-img")
                        slice_slider = gr.Slider(minimum=0, maximum=1, step=1, value=0, label="slice")
                        with gr.Row():
                            vmin_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, label="window min")
                            vmax_slider = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, label="window max")
                        metrics_md = gr.HTML(_metrics_tiles_html(None, None), elem_id="metrics-bar")

        # === Event handlers ===

        def on_task(label, variant_label):
            key = controller.set_task(label)
            variant = _variant_key(variant_label)
            opts = controller.refresh_runs()
            return (
                gr.update(label=SIZE_LABEL.get(key, "slice_size")),   # size field label
                controller.refresh_checkpoint(variant),                # ckpt info (task+variant)
                gr.update(choices=_run_choices(opts), value=None),     # run dropdown
                _curves_df([], metric_dd.value if hasattr(metric_dd, "value") else "loss"),
                gr.update(value=_cta_label(key, variant)),             # dynamic CTA label
            )

        task_dd.change(
            on_task,
            inputs=[task_dd, variant_radio],
            outputs=[size, ckpt_info, run_dd, curve, run_infer_btn],
        )

        # --- data_dirs editor ---
        def _df_to_dirs(df):
            """Flatten the single-column dataframe into a list of non-empty roots."""
            dirs = []
            if df is None:
                return dirs
            # gradio may hand us a pandas DataFrame or a list-of-rows.
            try:
                values = df.values.tolist()  # pandas
            except AttributeError:
                values = list(df)
            for row in values:
                cell = row[0] if isinstance(row, (list, tuple)) else row
                if cell is not None and str(cell).strip():
                    dirs.append(str(cell).strip())
            return dirs

        def on_dirs_changed(df):
            dirs = _df_to_dirs(df)
            n = controller.on_data_dirs_changed(dirs)
            return _size_text(n)

        dirs_df.change(on_dirs_changed, inputs=[dirs_df], outputs=[size_label])

        # --- form -> config sync (mirrors the old view.dump_config) ---
        def _dump_form(df, pidx, ep, ve, bs, lr, sz, vf, dev, dis, venv, proj, ifp):
            cfg.data_dirs = _df_to_dirs(df)
            cfg.data_dir = cfg.data_dirs[0] if cfg.data_dirs else ""
            cfg.patient_index = _to_int(pidx, cfg.patient_index)
            cfg.epochs = _to_int(ep, cfg.epochs)
            cfg.val_every = _to_int(ve, cfg.val_every)
            cfg.batch_size = _to_int(bs, cfg.batch_size)
            cfg.learning_rate = _to_float(lr, cfg.learning_rate)
            cfg.size = _to_int(sz, cfg.size)
            cfg.val_fraction = _to_float(vf, cfg.val_fraction)
            cfg.device = dev
            cfg.distro = dis
            cfg.venv_path = venv
            cfg.project_path = proj
            cfg.task = controller.task
            cfg.infer_patient_index = _to_int(ifp, cfg.infer_patient_index)

        _form_inputs = [
            dirs_df, patient_index, epochs, val_every, batch_size, learning_rate,
            size, val_fraction, device, distro, venv_path, project_path, infer_patient,
        ]

        # --- training ---
        def on_start(*form_vals):
            _dump_form(*form_vals)
            st = controller.start_training()
            return st, _pill_for_status(st)

        start_btn.click(on_start, inputs=_form_inputs, outputs=[status, status_pill])

        def on_stop():
            st = controller.stop_training()
            return st, _pill_for_status(st)

        stop_btn.click(on_stop, inputs=None, outputs=[status, status_pill])

        # --- smoke test (blocking; streams the report into the log) ---
        def on_smoke(log_text, *form_vals):
            _dump_form(*form_vals)
            report_lines, smoke_status = controller.run_smoke_blocking()
            if report_lines:
                appended = log_text + ("\n" if log_text else "") + "\n".join(report_lines)
            else:
                appended = log_text
            return smoke_status, appended, appended, _pill_for_status(smoke_status)

        smoke_btn.click(
            on_smoke,
            inputs=[st_log] + _form_inputs,
            outputs=[status, st_log, log_box, status_pill],
        )

        # --- run / metric selectors ---
        def on_run_changed(run_id, metric_key):
            result = controller.on_run_changed(run_id)
            return _curves_df(result.metric_rows, metric_key)

        run_dd.change(on_run_changed, inputs=[run_dd, metric_dd], outputs=[curve])

        def on_metric_changed(metric_key):
            # Redraw with the already-accumulated rows (no model read).
            return _curves_df(controller.metric_rows, metric_key)

        metric_dd.change(on_metric_changed, inputs=[metric_dd], outputs=[curve])

        # --- model variant toggle (CTA label + control visibility + ckpt info) ---
        def on_variant_change(variant_label):
            variant = _variant_key(variant_label)
            is_flow = variant == "flow"
            return (
                gr.update(value=_cta_label(controller.task, variant)),  # CTA label
                gr.update(visible=is_flow),                             # flow steps
                gr.update(visible=not is_flow),                         # ddim steps
                gr.update(visible=not is_flow),                         # guidance
                controller.refresh_checkpoint(variant),                # ckpt info
            )

        variant_radio.change(
            on_variant_change,
            inputs=[variant_radio],
            outputs=[run_infer_btn, flow_steps, ddim_steps, guidance, ckpt_info],
        )

        # --- checkpoints ---
        def on_load_best(variant_label):
            return controller.refresh_checkpoint(_variant_key(variant_label))

        load_best_btn.click(on_load_best, inputs=[variant_radio], outputs=[ckpt_info])

        # --- inference (blocking) ---
        def on_run_inference(infer_dir, ipatient, slice_idx, orient, idevice,
                             variant_label, flow_steps_v, ddim_steps_v, guidance_v, *form_vals):
            _dump_form(*form_vals)
            cfg.device = idevice  # View-tab device wins over the Train-tab dropdown
            variant = _variant_key(variant_label)
            is_flow = variant == "flow"
            steps = _to_int(flow_steps_v if is_flow else ddim_steps_v, 25)
            guide = None if is_flow else _to_float(guidance_v, 1.0)

            # Pre-flight: don't launch a doomed run when the variant's ckpt is absent.
            if not controller.checkpoint_present(variant):
                if controller.task in _DIFFUSION_TASK_KEYS:
                    kind = "flow" if is_flow else "diffusion (ε)"
                    msg = f"no {kind} checkpoint for {controller.task} — train it first"
                else:
                    msg = f"no checkpoint for {controller.task} — train it first"
                return (msg, gr.update(), gr.update(), gr.update(),
                        gr.update(visible=False), None, None, None,
                        _metrics_tiles_html(None, None), None, None, None,
                        _status_pill_html("error", "missing ckpt"))

            data_dirs = [infer_dir.strip()] if infer_dir and infer_dir.strip() else cfg.data_dirs
            pred, gt, nac, meta, st = controller.run_inference_blocking(
                data_dirs, _to_int(ipatient, 0), _to_int(slice_idx, 0),
                variant=variant, steps=steps, guidance=guide,
            )
            if pred is None:
                return (st, gr.update(), gr.update(), gr.update(),
                        gr.update(visible=False), None, None, None,
                        _metrics_tiles_html(None, None), None, None, None,
                        _status_pill_html("error", "error"))
            pred = np.asarray(pred)
            gt = np.asarray(gt)
            # NAC is optional (present for diff2d/ft3d). Keep None so the panel
            # blanks + hides for AE tasks or when nac.npy was absent.
            nac = np.asarray(nac) if nac is not None else None
            vmin = float(meta.get("vmin", 0.0))
            vmax = float(meta.get("vmax", 1.0))
            orientation = _orient_key(orient)
            # Slice slider range from the oriented axis length (None for 2D tasks).
            axis_len = _axis_len(pred, orientation)
            if axis_len is not None:
                cur = axis_len // 2
                slider_upd = gr.update(minimum=0, maximum=max(0, axis_len - 1), value=cur)
            else:
                cur = 0
                slider_upd = gr.update(minimum=0, maximum=1, value=0)
            n_img, p_img, g_img, d_img, metrics = _render_panels(
                nac, pred, gt, cur, vmin, vmax, orientation
            )
            # Show the NAC panel only when we actually loaded a NAC volume.
            nac_upd = gr.update(value=n_img, visible=nac is not None)
            # Window sliders span a little beyond [vmin, vmax] so the user can adjust.
            lo = min(vmin, float(np.min(gt)) if gt.size else vmin)
            hi = max(vmax, float(np.max(gt)) if gt.size else vmax)
            vmin_upd = gr.update(minimum=lo, maximum=hi, value=vmin)
            vmax_upd = gr.update(minimum=lo, maximum=hi, value=vmax)
            return (st, slider_upd, vmin_upd, vmax_upd,
                    nac_upd, p_img, g_img, d_img, metrics, nac, pred, gt,
                    _status_pill_html("done", "inference done"))

        run_infer_btn.click(
            on_run_inference,
            inputs=([infer_data_dir, infer_patient, slice_slider, orient_radio, infer_device,
                     variant_radio, flow_steps, ddim_steps, guidance] + _form_inputs),
            outputs=[infer_status, slice_slider, vmin_slider, vmax_slider,
                     img_nac, img_pred, img_gt, img_diff, metrics_md,
                     st_nac, st_pred, st_gt, status_pill],
        )

        # --- re-render the panels when slice / window / orientation changes ---
        def on_view_change(nac_vol, pred_vol, gt_vol, idx, vmin, vmax, orient):
            return _render_panels(nac_vol, pred_vol, gt_vol, idx, vmin, vmax, _orient_key(orient))

        for ctrl in (slice_slider, vmin_slider, vmax_slider):
            ctrl.change(
                on_view_change,
                inputs=[st_nac, st_pred, st_gt, slice_slider, vmin_slider, vmax_slider, orient_radio],
                outputs=[img_nac, img_pred, img_gt, img_diff, metrics_md],
            )

        # --- orientation change: re-range the slider to the new axis (middle slice)
        #     then re-render the oriented frames + metrics. ---
        def on_orient_change(nac_vol, pred_vol, gt_vol, vmin, vmax, orient):
            orientation = _orient_key(orient)
            axis_len = _axis_len(pred_vol, orientation)
            if axis_len is not None:
                cur = axis_len // 2
                slider_upd = gr.update(minimum=0, maximum=max(0, axis_len - 1), value=cur)
            else:
                # 2D task (ae2d/diff2d): orientation + slider are inert.
                cur = 0
                slider_upd = gr.update(minimum=0, maximum=1, value=0)
            n_img, p_img, g_img, d_img, metrics = _render_panels(
                nac_vol, pred_vol, gt_vol, cur, vmin, vmax, orientation
            )
            return slider_upd, n_img, p_img, g_img, d_img, metrics

        orient_radio.change(
            on_orient_change,
            inputs=[st_nac, st_pred, st_gt, vmin_slider, vmax_slider, orient_radio],
            outputs=[slice_slider, img_nac, img_pred, img_gt, img_diff, metrics_md],
        )

        # === Live poll loop ===
        def on_poll(log_text, metric_key, run_id):
            result = controller.poll()
            # Append any new stdout lines to the running log buffer.
            if result.new_log_lines:
                joined = "\n".join(str(x) for x in result.new_log_lines)
                log_text = (log_text + ("\n" if log_text else "") + joined)
            # Run dropdown choices (keep current selection).
            run_upd = gr.update(choices=_run_choices(result.run_options))
            # Curve only redraws when something changed this tick.
            curve_upd = (
                _curves_df(result.metric_rows, metric_key)
                if result.curves_dirty else gr.update()
            )
            status_upd = result.status if result.status is not None else gr.update()
            pill_upd = _pill_for_status(result.status)
            return log_text, log_text, curve_upd, status_upd, run_upd, pill_upd

        # Prefer gr.Timer (Gradio >= 4.x). The caller verifies availability.
        timer = gr.Timer(POLL_INTERVAL)
        timer.tick(
            on_poll,
            inputs=[st_log, metric_dd, run_dd],
            outputs=[st_log, log_box, curve, status, run_dd, status_pill],
        )

    # Stash the controller so launch()/callers (and shutdown hooks) can reach it.
    demo._train_controller = controller
    # Gradio 6: theme/css/js are launch() kwargs. Stash them so any caller of
    # demo.launch(**demo._theme_kwargs) gets the dark look without rebuilding.
    demo._theme_kwargs = dict(theme=_dark_theme(), css=APP_CSS, js=FORCE_DARK_JS)
    return demo


def launch(model=None, inbrowser=True, share=False, **kwargs):
    """Build and launch the Gradio app with the dark theme (no public share)."""
    demo = build_app(model)
    launch_kwargs = dict(demo._theme_kwargs)
    launch_kwargs.update(kwargs)
    demo.launch(inbrowser=inbrowser, share=share, **launch_kwargs)
    return demo
