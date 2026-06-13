"""Gradio front-end for the Train Viewer (replaces the old Tkinter view).

Mirrors the behaviour of the previous TrainView/TrainPresenter pair but uses
sliders instead of mouse drags for window/level + slice navigation, a native
gr.LinePlot for the metric curves, and a gr.Timer for the live poll loop.

The UI logic stays thin: all stateful work lives in TrainController; this module
just wires Gradio components to controller methods and formats arrays for display.
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

# How often the live poll loop fires (seconds); matches the old 750ms tick.
POLL_INTERVAL = 0.75


# === Theme / styling (dark, black-canvas medical look) ===

def _dark_theme():
    """A cyan-on-black Gradio theme. Built lazily so importing this module
    without gradio's theme assets never fails at import time."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.cyan,
        secondary_hue=gr.themes.colors.teal,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ).set(
        body_background_fill_dark="#000000",
        background_fill_primary_dark="#0b0f14",
        background_fill_secondary_dark="#0b0f14",
        block_background_fill_dark="#0d1117",
        block_border_color_dark="#1f2a37",
        block_label_text_color_dark="#3fd0e0",
        body_text_color_dark="#d6e2ea",
        button_primary_background_fill_dark="#0891b2",
        button_primary_background_fill_hover_dark="#22d3ee",
    )


# Black canvas behind the scan images + a faint cyan frame/glow so they pop.
APP_CSS = """
.gradio-container { max-width: 100% !important; }
#scan-row { gap: 12px; }
.scan-img, .scan-img > div, .scan-img img { background: #000 !important; }
.scan-img { border: 1px solid #163a44 !important; border-radius: 10px;
            box-shadow: 0 0 18px rgba(34,211,238,0.10); }
.scan-img img { object-fit: contain !important; image-rendering: auto; }
#app-title h1 { letter-spacing: .5px; margin-bottom: 0; }
#app-title { border-bottom: 1px solid #163a44; padding-bottom: 6px; }
#metrics-bar { font-family: 'JetBrains Mono', monospace; color: #7fe7f2; }
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


def _diff_rgb(frame_pred, frame_gt):
    """Map (pred - gt) through a diverging colormap symmetric around 0 -> RGB uint8."""
    if frame_pred is None or frame_gt is None:
        return None
    diff = np.asarray(frame_pred, dtype=np.float32) - np.asarray(frame_gt, dtype=np.float32)
    dmax = float(np.max(np.abs(diff))) if diff.size else 1.0
    if dmax <= 0:
        dmax = 1.0
    norm = (diff / dmax + 1.0) * 0.5  # [-dmax, dmax] -> [0, 1]
    norm = np.clip(norm, 0.0, 1.0)
    if _mpl_cm is not None:
        rgba = _mpl_cm["bwr"](norm)  # HxWx4 float in [0,1]
        return (rgba[..., :3] * 255.0).astype(np.uint8)
    # Fallback diverging map (blue<0, red>0) if matplotlib is unavailable.
    r = np.clip(norm * 2.0, 0.0, 1.0)
    b = np.clip((1.0 - norm) * 2.0, 0.0, 1.0)
    g = 1.0 - np.abs(norm - 0.5) * 2.0
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _slice_metrics_md(pred_frame, gt_frame):
    """MAE / RMSE / PSNR between the displayed pred & gt slice, as markdown."""
    if pred_frame is None or gt_frame is None:
        return "—"
    p = np.asarray(pred_frame, dtype=np.float32)
    g = np.asarray(gt_frame, dtype=np.float32)
    diff = p - g
    mae = float(np.mean(np.abs(diff))) if diff.size else 0.0
    rmse = float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0
    drange = float(np.max(g) - np.min(g)) if g.size else 0.0
    if rmse > 0 and drange > 0:
        psnr = 20.0 * np.log10(drange / rmse)
        psnr_s = f"{psnr:.2f} dB"
    else:
        psnr_s = "∞"
    return f"**MAE** {mae:.4g}  ·  **RMSE** {rmse:.4g}  ·  **PSNR** {psnr_s}"


def _render_three(pred_vol, gt_vol, idx, vmin, vmax, orientation="AXIAL"):
    """Return (pred_img, gt_img, diff_img, metrics_md) for the three gr.Image + readout.

    All four outputs derive from the same oriented frame so pred/gt/diff/metrics stay
    in lockstep with the orientation selector.
    """
    pred = _slice_of(pred_vol, idx, orientation)
    gt = _slice_of(gt_vol, idx, orientation)
    return (
        _gray_uint8(pred, vmin, vmax),
        _gray_uint8(gt, vmin, vmax),
        _diff_rgb(pred, gt),
        _slice_metrics_md(pred, gt),
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

    def _size_text(n):
        return "Dataset: — patients" if n is None else f"Dataset: {int(n)} patients"

    # Gradio 6 moved theme/css/js from the Blocks constructor to launch(); we
    # build them here and stash them on the returned demo for launch() to apply.
    with gr.Blocks(title="Pet-CT Train Viewer") as demo:
        # --- shared state for the View tab (loaded volumes + window/level) ---
        st_pred = gr.State(None)
        st_gt = gr.State(None)
        # Append-only log buffer (we keep it in a State so the Timer can extend it).
        st_log = gr.State("")

        # === Top bar: title + task selector ===
        gr.Markdown("# 🧠 Pet-CT Train Viewer\nNAC→AC PET latent-diffusion training & inference",
                    elem_id="app-title")
        task_dd = gr.Dropdown(
            choices=TASK_DISPLAY, value=TASK_DISPLAY[0], label="Task", interactive=True
        )

        with gr.Tabs():
            # === Train tab ===
            with gr.Tab("Train"):
                # Multi-root data_dirs editor (single-column dataframe + add/remove).
                gr.Markdown("**data_dirs** (dataset roots)")
                dirs_df = gr.Dataframe(
                    headers=["root"],
                    value=[[d] for d in init_dirs] or [[""]],
                    column_count=(1, "fixed"),
                    datatype="str",
                    interactive=True,
                    label=None,
                )
                size_label = gr.Markdown(_size_text(init_size_n))

                patient_index = gr.Number(value=cfg.patient_index, label="patient_index", precision=0)
                epochs = gr.Number(value=cfg.epochs, label="epochs", precision=0)
                val_every = gr.Number(value=cfg.val_every, label="val_every (steps)", precision=0)
                batch_size = gr.Number(value=cfg.batch_size, label="batch_size", precision=0)
                learning_rate = gr.Number(value=cfg.learning_rate, label="learning_rate")
                size = gr.Number(value=cfg.size, label=init_size_label, precision=0)
                val_fraction = gr.Number(value=cfg.val_fraction, label="val_fraction (0-1)")
                device = gr.Dropdown(choices=["cuda", "cpu"], value=cfg.device, label="device")

                with gr.Group():
                    gr.Markdown("**WSL settings**")
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

            # === View tab ===
            with gr.Tab("View"):
                with gr.Row():
                    load_best_btn = gr.Button("Load best model")
                    ckpt_info = gr.Markdown(controller.refresh_checkpoint())
                infer_data_dir = gr.Textbox(value="", label="infer data_dir (blank = use train roots)")
                infer_patient = gr.Number(value=cfg.infer_patient_index, label="patient_index", precision=0)
                orient_radio = gr.Radio(
                    choices=["Axial", "Coronal", "Sagittal"], value="Axial",
                    label="orientation", interactive=True,
                )
                slice_slider = gr.Slider(minimum=0, maximum=1, step=1, value=0, label="slice")
                with gr.Row():
                    run_infer_btn = gr.Button("Run inference", variant="primary")
                    infer_device = gr.Dropdown(choices=["cuda", "cpu"], value=cfg.device, label="device")
                    infer_status = gr.Textbox(value="idle", label="Status", interactive=False)
                with gr.Row():
                    vmin_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, label="window min")
                    vmax_slider = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, label="window max")
                with gr.Row(elem_id="scan-row", equal_height=True):
                    img_pred = gr.Image(label="Predicted", interactive=False, height=480,
                                        elem_classes="scan-img")
                    img_gt = gr.Image(label="Ground Truth", interactive=False, height=480,
                                      elem_classes="scan-img")
                    img_diff = gr.Image(label="Difference (pred − gt)", interactive=False, height=480,
                                        elem_classes="scan-img")
                metrics_md = gr.Markdown("—", elem_id="metrics-bar")

        # === Event handlers ===

        def on_task(label):
            key = controller.set_task(label)
            opts = controller.refresh_runs()
            return (
                gr.update(label=SIZE_LABEL.get(key, "slice_size")),   # size field label
                controller.refresh_checkpoint(),                       # ckpt info (new task)
                gr.update(choices=_run_choices(opts), value=None),     # run dropdown
                _curves_df([], metric_dd.value if hasattr(metric_dd, "value") else "loss"),
            )

        task_dd.change(
            on_task,
            inputs=[task_dd],
            outputs=[size, ckpt_info, run_dd, curve],
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
            return controller.start_training()

        start_btn.click(on_start, inputs=_form_inputs, outputs=[status])

        def on_stop():
            return controller.stop_training()

        stop_btn.click(on_stop, inputs=None, outputs=[status])

        # --- smoke test (blocking; streams the report into the log) ---
        def on_smoke(log_text, *form_vals):
            _dump_form(*form_vals)
            report_lines, smoke_status = controller.run_smoke_blocking()
            if report_lines:
                appended = log_text + ("\n" if log_text else "") + "\n".join(report_lines)
            else:
                appended = log_text
            return smoke_status, appended, appended

        smoke_btn.click(
            on_smoke,
            inputs=[st_log] + _form_inputs,
            outputs=[status, st_log, log_box],
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

        # --- checkpoints ---
        def on_load_best():
            return controller.refresh_checkpoint()

        load_best_btn.click(on_load_best, inputs=None, outputs=[ckpt_info])

        # --- inference (blocking) ---
        def on_run_inference(infer_dir, ipatient, slice_idx, orient, idevice, *form_vals):
            _dump_form(*form_vals)
            cfg.device = idevice  # View-tab device wins over the Train-tab dropdown
            data_dirs = [infer_dir.strip()] if infer_dir and infer_dir.strip() else cfg.data_dirs
            pred, gt, meta, st = controller.run_inference_blocking(
                data_dirs, _to_int(ipatient, 0), _to_int(slice_idx, 0)
            )
            if pred is None:
                return (st, gr.update(), gr.update(), gr.update(),
                        None, None, None, "—", None, None)
            pred = np.asarray(pred)
            gt = np.asarray(gt)
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
            p_img, g_img, d_img, metrics = _render_three(pred, gt, cur, vmin, vmax, orientation)
            # Window sliders span a little beyond [vmin, vmax] so the user can adjust.
            lo = min(vmin, float(np.min(gt)) if gt.size else vmin)
            hi = max(vmax, float(np.max(gt)) if gt.size else vmax)
            vmin_upd = gr.update(minimum=lo, maximum=hi, value=vmin)
            vmax_upd = gr.update(minimum=lo, maximum=hi, value=vmax)
            return (st, slider_upd, vmin_upd, vmax_upd,
                    p_img, g_img, d_img, metrics, pred, gt)

        run_infer_btn.click(
            on_run_inference,
            inputs=[infer_data_dir, infer_patient, slice_slider, orient_radio, infer_device] + _form_inputs,
            outputs=[infer_status, slice_slider, vmin_slider, vmax_slider,
                     img_pred, img_gt, img_diff, metrics_md, st_pred, st_gt],
        )

        # --- re-render the three images when slice / window / orientation changes ---
        def on_view_change(pred_vol, gt_vol, idx, vmin, vmax, orient):
            return _render_three(pred_vol, gt_vol, idx, vmin, vmax, _orient_key(orient))

        for ctrl in (slice_slider, vmin_slider, vmax_slider):
            ctrl.change(
                on_view_change,
                inputs=[st_pred, st_gt, slice_slider, vmin_slider, vmax_slider, orient_radio],
                outputs=[img_pred, img_gt, img_diff, metrics_md],
            )

        # --- orientation change: re-range the slider to the new axis (middle slice)
        #     then re-render the three oriented frames + metrics. ---
        def on_orient_change(pred_vol, gt_vol, vmin, vmax, orient):
            orientation = _orient_key(orient)
            axis_len = _axis_len(pred_vol, orientation)
            if axis_len is not None:
                cur = axis_len // 2
                slider_upd = gr.update(minimum=0, maximum=max(0, axis_len - 1), value=cur)
            else:
                # 2D task (ae2d/diff2d): orientation + slider are inert.
                cur = 0
                slider_upd = gr.update(minimum=0, maximum=1, value=0)
            p_img, g_img, d_img, metrics = _render_three(
                pred_vol, gt_vol, cur, vmin, vmax, orientation
            )
            return slider_upd, p_img, g_img, d_img, metrics

        orient_radio.change(
            on_orient_change,
            inputs=[st_pred, st_gt, vmin_slider, vmax_slider, orient_radio],
            outputs=[slice_slider, img_pred, img_gt, img_diff, metrics_md],
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
            return log_text, log_text, curve_upd, status_upd, run_upd

        # Prefer gr.Timer (Gradio >= 4.x). The caller verifies availability.
        timer = gr.Timer(POLL_INTERVAL)
        timer.tick(
            on_poll,
            inputs=[st_log, metric_dd, run_dd],
            outputs=[st_log, log_box, curve, status, run_dd],
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
