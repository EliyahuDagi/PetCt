import tkinter as tk
from tkinter import ttk, filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

# Display labels for the task selector (kept here to avoid importing model into view-only tests).
TASK_DISPLAY = ["AE 2D (encode/decode)", "AE 3D (encode/decode)", "NAC->AC 2D", "NAC->AC 3D"]
DISPLAY_TO_KEY = {
    "AE 2D (encode/decode)": "ae2d",
    "AE 3D (encode/decode)": "ae3d",
    "NAC->AC 2D": "diff2d",
    "NAC->AC 3D": "ft3d",
}
KEY_TO_DISPLAY = {v: k for k, v in DISPLAY_TO_KEY.items()}
LATENT_TASKS = {"diff2d", "ft3d"}
# Label shown next to the size field, per task.
SIZE_LABEL = {"ae2d": "slice_size", "ae3d": "crop_size", "diff2d": "latent_size", "ft3d": "latent_size"}

# Metric keys the user can plot.
METRIC_KEYS = ["loss", "recon_l1", "kl", "psnr"]


class TrainView(tk.Tk):
    def __init__(self, presenter=None):
        super().__init__()
        self.title("Pet-CT Train Viewer")
        self.geometry("1200x800")

        self.presenter = presenter
        self.task_key = "ae2d"

        self._build_top_bar()
        self._build_notebook()

        # Periodic refresh of log/curves; never blocks the mainloop.
        self.after(750, self._tick)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def set_presenter(self, presenter):
        self.presenter = presenter

    # === Top bar: task selector ===
    def _build_top_bar(self):
        bar = ttk.Frame(self)
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(bar, text="Task:").pack(side=tk.LEFT, padx=(8, 4), pady=6)
        self.task_var = tk.StringVar(value=TASK_DISPLAY[0])
        self.task_combo = ttk.Combobox(
            bar, textvariable=self.task_var, values=TASK_DISPLAY, state="readonly", width=24
        )
        self.task_combo.pack(side=tk.LEFT, padx=4, pady=6)
        self.task_combo.bind("<<ComboboxSelected>>", self._on_task_change)

    def _on_task_change(self, event=None):
        if self.presenter:
            self.presenter.set_task(self.task_var.get())

    def set_task_key(self, key):
        self.task_key = key
        # Keep combobox in sync.
        if key in KEY_TO_DISPLAY:
            self.task_var.set(KEY_TO_DISPLAY[key])
        # Adapt the size label per task.
        if hasattr(self, "size_label"):
            self.size_label.config(text=SIZE_LABEL.get(key, "slice_size"))

    # === Notebook ===
    def _build_notebook(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.tab_train = ttk.Frame(self.notebook)
        self.tab_log = ttk.Frame(self.notebook)
        self.tab_view = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_train, text="Train")
        self.notebook.add(self.tab_log, text="Log")
        self.notebook.add(self.tab_view, text="View")

        self._build_train_tab()
        self._build_log_tab()
        self._build_view_tab()

    # === Train tab ===
    def _build_train_tab(self):
        frm = ttk.Frame(self.tab_train, padding=10)
        frm.pack(side=tk.TOP, fill=tk.X)

        # Hyperparameter form variables.
        self.var_patient_index = tk.StringVar()
        self.var_epochs = tk.StringVar()
        self.var_val_every = tk.StringVar()
        self.var_batch_size = tk.StringVar()
        self.var_learning_rate = tk.StringVar()
        self.var_size = tk.StringVar()
        self.var_val_fraction = tk.StringVar()
        self.var_device = tk.StringVar(value="cuda")

        row = 0
        # data_dirs: a multi-root list with Add / Remove and a live size readout.
        ttk.Label(frm, text="data_dirs").grid(row=row, column=0, sticky=tk.NW, pady=3)
        list_frame = ttk.Frame(frm)
        list_frame.grid(row=row, column=1, sticky=tk.W)
        self.data_dirs_listbox = tk.Listbox(list_frame, height=4, width=48)
        ddscroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.data_dirs_listbox.yview)
        self.data_dirs_listbox.configure(yscrollcommand=ddscroll.set)
        self.data_dirs_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ddscroll.pack(side=tk.RIGHT, fill=tk.Y)
        dd_btns = ttk.Frame(frm)
        dd_btns.grid(row=row, column=2, sticky=tk.NW, padx=4)
        ttk.Button(dd_btns, text="Add…", command=self._add_data_dir).pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
        ttk.Button(dd_btns, text="Remove", command=self._remove_data_dir).pack(side=tk.TOP, fill=tk.X)
        row += 1
        # Dataset size readout (recomputed on Add/Remove and on load).
        self.dataset_size_var = tk.StringVar(value="Dataset: — patients")
        ttk.Label(frm, textvariable=self.dataset_size_var).grid(
            row=row, column=1, sticky=tk.W, pady=(0, 4)
        )
        row += 1

        def add_field(label, var, widget="entry", values=None):
            nonlocal row
            lbl = ttk.Label(frm, text=label)
            lbl.grid(row=row, column=0, sticky=tk.W, pady=3)
            if widget == "spin":
                w = ttk.Spinbox(frm, from_=0, to=1000000, textvariable=var, width=16)
            elif widget == "combo":
                w = ttk.Combobox(frm, textvariable=var, values=values or [], state="readonly", width=14)
            else:
                w = ttk.Entry(frm, textvariable=var, width=18)
            w.grid(row=row, column=1, sticky=tk.W)
            row += 1
            return lbl, w

        add_field("patient_index", self.var_patient_index, "spin")
        add_field("epochs", self.var_epochs, "spin")
        add_field("val_every (steps)", self.var_val_every, "spin")
        add_field("batch_size", self.var_batch_size, "spin")
        add_field("learning_rate", self.var_learning_rate)
        # size label is dynamic per task.
        self.size_label, _ = add_field("slice_size", self.var_size, "spin")
        add_field("val_fraction (0-1)", self.var_val_fraction)
        add_field("device", self.var_device, "combo", values=["cuda", "cpu"])

        # WSL settings group.
        wsl = ttk.LabelFrame(self.tab_train, text="WSL settings", padding=10)
        wsl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)
        self.var_distro = tk.StringVar(value="Ubuntu")
        self.var_venv_path = tk.StringVar(value="~/petct/.venv")
        self.var_project_path = tk.StringVar()
        ttk.Label(wsl, text="distro").grid(row=0, column=0, sticky=tk.W, pady=3)
        ttk.Entry(wsl, textvariable=self.var_distro, width=24).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(wsl, text="venv_path").grid(row=1, column=0, sticky=tk.W, pady=3)
        ttk.Entry(wsl, textvariable=self.var_venv_path, width=40).grid(row=1, column=1, sticky=tk.W)
        ttk.Label(wsl, text="project_path").grid(row=2, column=0, sticky=tk.W, pady=3)
        ttk.Entry(wsl, textvariable=self.var_project_path, width=48).grid(row=2, column=1, sticky=tk.W)

        # Buttons + status.
        btns = ttk.Frame(self.tab_train, padding=10)
        btns.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(btns, text="Start Training", command=self._on_start).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Stop", command=self._on_stop).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Smoke Test", command=self._on_smoke).pack(side=tk.LEFT, padx=4)
        self.status_var = tk.StringVar(value="stopped")
        ttk.Label(btns, text="Status:").pack(side=tk.LEFT, padx=(16, 2))
        ttk.Label(btns, textvariable=self.status_var).pack(side=tk.LEFT)

    # --- data_dirs list widget ---
    def _add_data_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.data_dirs_listbox.insert(tk.END, path)
            self._on_data_dirs_changed()

    def _remove_data_dir(self):
        sel = self.data_dirs_listbox.curselection()
        for idx in reversed(sel):
            self.data_dirs_listbox.delete(idx)
        self._on_data_dirs_changed()

    def _on_data_dirs_changed(self):
        if self.presenter:
            self.presenter.on_data_dirs_changed()

    def get_data_dirs(self):
        return list(self.data_dirs_listbox.get(0, tk.END))

    def set_data_dirs(self, dirs):
        self.data_dirs_listbox.delete(0, tk.END)
        for d in (dirs or []):
            self.data_dirs_listbox.insert(tk.END, d)

    def set_dataset_size(self, n):
        if n is None:
            self.dataset_size_var.set("Dataset: — patients")
        else:
            self.dataset_size_var.set(f"Dataset: {int(n)} patients")

    def _on_start(self):
        if self.presenter:
            self.presenter.start_training()

    def _on_stop(self):
        if self.presenter:
            self.presenter.stop_training()

    def _on_smoke(self):
        if self.presenter:
            self.presenter.run_smoke()

    def set_status(self, text):
        self.status_var.set(text)

    def show_smoke_report(self, report):
        """Render the smoke-test report into the Log pane and switch to it."""
        self.append_log(_format_smoke_report(report))
        try:
            self.notebook.select(self.tab_log)
        except Exception:
            pass

    # === Log tab ===
    def _build_log_tab(self):
        top = ttk.Frame(self.tab_log)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="metric:").pack(side=tk.LEFT, padx=(8, 4), pady=4)
        self.metric_var = tk.StringVar(value="loss")
        self.metric_combo = ttk.Combobox(
            top, textvariable=self.metric_var, values=METRIC_KEYS, state="readonly", width=12
        )
        self.metric_combo.pack(side=tk.LEFT, padx=4, pady=4)
        self.metric_combo.bind("<<ComboboxSelected>>", self._on_metric_change)

        # Loss curves figure.
        self.log_fig = Figure(figsize=(6, 3), dpi=100)
        self.ax_curve = self.log_fig.add_subplot(111)
        self.ax_curve.set_xlabel("step")
        self.ax_curve.set_ylabel("loss")
        self.line_train, = self.ax_curve.plot([], [], label="train", color="tab:blue")
        self.line_val, = self.ax_curve.plot([], [], label="val", color="tab:orange")
        self.ax_curve.legend(loc="upper right")
        self.log_canvas = FigureCanvasTkAgg(self.log_fig, master=self.tab_log)
        self.log_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Scrolling, append-only log pane.
        text_frame = ttk.Frame(self.tab_log)
        text_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(text_frame, height=10, state=tk.DISABLED, wrap=tk.NONE)
        scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _on_metric_change(self, event=None):
        if self.presenter:
            self.presenter.on_metric_changed()

    def get_metric_key(self):
        return self.metric_var.get()

    def append_log(self, lines):
        if not lines:
            return
        self.log_text.configure(state=tk.NORMAL)
        for line in lines:
            self.log_text.insert(tk.END, str(line) + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def update_curves(self, rows, metric_key):
        """Plot train vs val for the chosen metric key against step."""
        train_x, train_y, val_x, val_y = [], [], [], []
        for r in rows:
            if metric_key not in r or r.get(metric_key) is None:
                continue
            step = r.get("step")
            if step is None:
                continue
            val = r[metric_key]
            if r.get("phase") == "val":
                val_x.append(step)
                val_y.append(val)
            else:
                train_x.append(step)
                train_y.append(val)
        self.line_train.set_data(train_x, train_y)
        self.line_val.set_data(val_x, val_y)
        self.ax_curve.set_ylabel(metric_key)
        self.ax_curve.relim()
        self.ax_curve.autoscale_view()
        self.log_canvas.draw_idle()

    # === View tab ===
    def _build_view_tab(self):
        top = ttk.Frame(self.tab_view, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Load best model", command=self._on_load_best).pack(side=tk.LEFT, padx=4)
        self.ckpt_var = tk.StringVar(value="no checkpoint yet")
        ttk.Label(top, textvariable=self.ckpt_var).pack(side=tk.LEFT, padx=10)

        form = ttk.Frame(self.tab_view, padding=8)
        form.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(form, text="data_dir").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.var_infer_data_dir = tk.StringVar()
        ttk.Entry(form, textvariable=self.var_infer_data_dir, width=48).grid(row=0, column=1, sticky=tk.W)
        ttk.Button(form, text="Browse", command=self._browse_infer_data_dir).grid(row=0, column=2, padx=4)
        ttk.Label(form, text="patient_index").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.var_infer_patient = tk.StringVar(value="0")
        ttk.Spinbox(form, from_=0, to=100000, textvariable=self.var_infer_patient, width=10).grid(
            row=1, column=1, sticky=tk.W
        )

        # Slice slider scrolls all three axes for 3D volumes.
        self.infer_slice_scale = tk.Scale(
            self.tab_view, from_=0, to=0, orient=tk.HORIZONTAL, label="slice",
            command=self._on_infer_slice_change,
        )
        self.infer_slice_scale.pack(side=tk.TOP, fill=tk.X, padx=10)

        runbar = ttk.Frame(self.tab_view, padding=8)
        runbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(runbar, text="Run inference", command=self._on_run_inference).pack(side=tk.LEFT, padx=4)
        self.infer_status_var = tk.StringVar(value="idle")
        ttk.Label(runbar, text="Status:").pack(side=tk.LEFT, padx=(16, 2))
        ttk.Label(runbar, textvariable=self.infer_status_var).pack(side=tk.LEFT)

        # Three-panel display: Predicted | Ground Truth | Difference.
        self.view_fig = Figure(figsize=(12, 4), dpi=100)
        self.ax_pred = self.view_fig.add_subplot(131)
        self.ax_gt = self.view_fig.add_subplot(132)
        self.ax_diff = self.view_fig.add_subplot(133)
        for ax, title in [(self.ax_pred, "Predicted"), (self.ax_gt, "Ground Truth"), (self.ax_diff, "Difference")]:
            ax.set_title(title)
            ax.set_axis_off()
        blank = np.zeros((8, 8))
        self.img_pred = self.ax_pred.imshow(blank, cmap="gray")
        self.img_gt = self.ax_gt.imshow(blank, cmap="gray")
        self.img_diff = self.ax_diff.imshow(blank, cmap="bwr")
        self.view_canvas = FigureCanvasTkAgg(self.view_fig, master=self.tab_view)
        self.view_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Window/level drag + synced zoom, mirrored from DataViewer.
        self.view_canvas.mpl_connect("scroll_event", self._on_view_scroll)
        self.view_canvas.mpl_connect("button_press_event", self._on_view_press)
        self.view_canvas.mpl_connect("motion_notify_event", self._on_view_move)
        self.view_canvas.mpl_connect("button_release_event", self._on_view_release)
        self.view_dragging = False
        self.view_mode = "WL"
        self.last_mx = 0
        self.last_my = 0

        # Holders for loaded arrays.
        self.pred_vol = None
        self.gt_vol = None
        self.meta = {}
        self.vmin = 0.0
        self.vmax = 1.0

    def _browse_infer_data_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.var_infer_data_dir.set(path)

    def _on_load_best(self):
        if self.presenter:
            self.presenter.refresh_checkpoint()

    def set_checkpoint_info(self, text):
        self.ckpt_var.set(text)

    def _on_run_inference(self):
        if self.presenter:
            self.presenter.run_inference()

    def set_infer_status(self, text):
        self.infer_status_var.set(text)

    def get_data_dir(self):
        # Backward-compatible single-root accessor (first root or empty).
        dirs = self.get_data_dirs()
        return dirs[0] if dirs else ""

    def get_infer_data_dirs(self):
        # The View tab's own infer root takes precedence; otherwise fall back to
        # the Train tab's data_dirs list (multi-root).
        own = self.var_infer_data_dir.get()
        if own:
            return [own]
        return self.get_data_dirs()

    def get_infer_data_dir(self):
        # Backward-compatible single-root accessor (first root or empty).
        dirs = self.get_infer_data_dirs()
        return dirs[0] if dirs else ""

    def get_infer_patient_index(self):
        return _to_int(self.var_infer_patient.get(), 0)

    def get_infer_slice(self):
        return int(self.infer_slice_scale.get())

    def _on_infer_slice_change(self, value):
        self._render_slices(int(float(value)))

    # === Inference display ===
    def set_inference_data(self, pred, gt, meta):
        self.meta = meta or {}
        self.vmin = float(self.meta.get("vmin", 0.0))
        self.vmax = float(self.meta.get("vmax", 1.0))
        self.pred_vol = np.asarray(pred)
        self.gt_vol = np.asarray(gt)
        # 3D volumes: enable the slider along axis 0. 2D: single frame.
        if self.pred_vol.ndim == 3:
            n = self.pred_vol.shape[0]
            self.infer_slice_scale.config(to=max(0, n - 1))
            mid = n // 2
            self.infer_slice_scale.set(mid)
            self._render_slices(mid)
        else:
            self.infer_slice_scale.config(to=0)
            self._render_slices(0)

    def _slice_of(self, vol, idx):
        if vol is None:
            return None
        if vol.ndim == 3:
            idx = max(0, min(idx, vol.shape[0] - 1))
            return vol[idx]
        return vol

    def _render_slices(self, idx):
        pred = self._slice_of(self.pred_vol, idx)
        gt = self._slice_of(self.gt_vol, idx)
        if pred is None or gt is None:
            return
        diff = pred - gt
        self.img_pred.set_data(pred)
        self.img_pred.set_clim(self.vmin, self.vmax)
        self.img_pred.set_extent([0, pred.shape[1], pred.shape[0], 0])
        self.img_gt.set_data(gt)
        self.img_gt.set_clim(self.vmin, self.vmax)
        self.img_gt.set_extent([0, gt.shape[1], gt.shape[0], 0])
        dmax = float(np.max(np.abs(diff))) if diff.size else 1.0
        if dmax <= 0:
            dmax = 1.0
        self.img_diff.set_data(diff)
        self.img_diff.set_clim(-dmax, dmax)
        self.img_diff.set_extent([0, diff.shape[1], diff.shape[0], 0])
        for ax in (self.ax_pred, self.ax_gt, self.ax_diff):
            ax.set_xlim(0, pred.shape[1])
            ax.set_ylim(pred.shape[0], 0)
        self.view_canvas.draw_idle()

    # === Mouse handlers (WL drag + synced zoom) ===
    def _sync_zoom_pan(self, xlim, ylim):
        for ax in (self.ax_pred, self.ax_gt, self.ax_diff):
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        self.view_canvas.draw_idle()

    def _on_view_scroll(self, event):
        if not event.inaxes:
            return
        if event.key == "control":
            base_scale = 1.2
            scale = 1 / base_scale if event.button == "up" else base_scale
            ax = event.inaxes
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            if event.xdata is None or event.ydata is None:
                return
            new_w = (xlim[1] - xlim[0]) * scale
            new_h = (ylim[1] - ylim[0]) * scale
            relx = (xlim[1] - event.xdata) / (xlim[1] - xlim[0])
            rely = (ylim[1] - event.ydata) / (ylim[1] - ylim[0])
            new_xlim = [event.xdata - new_w * (1 - relx), event.xdata + new_w * relx]
            new_ylim = [event.ydata - new_h * (1 - rely), event.ydata + new_h * rely]
            self._sync_zoom_pan(new_xlim, new_ylim)
        else:
            # Plain scroll changes the slice for 3D volumes.
            cur = int(self.infer_slice_scale.get())
            delta = 1 if event.button == "up" else -1
            self.infer_slice_scale.set(cur + delta)

    def _on_view_press(self, event):
        if event.button == 3:
            self.view_dragging = True
            self.view_mode = "WL"
        elif event.button == 2:
            self.view_dragging = True
            self.view_mode = "PAN"
        self.last_mx = event.x
        self.last_my = event.y

    def _on_view_move(self, event):
        if not self.view_dragging:
            return
        dx = event.x - self.last_mx
        dy = event.y - self.last_my
        if self.view_mode == "WL":
            # Adjust the shared window/level for pred + gt.
            span = max(1e-6, self.vmax - self.vmin)
            self.vmax += dx * span * 0.01
            self.vmin -= dy * span * 0.01
            if self.vmax <= self.vmin:
                self.vmax = self.vmin + 1e-6
            self.img_pred.set_clim(self.vmin, self.vmax)
            self.img_gt.set_clim(self.vmin, self.vmax)
            self.view_canvas.draw_idle()
        elif self.view_mode == "PAN" and event.inaxes:
            ax = event.inaxes
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            bbox = ax.get_window_extent().transformed(self.view_fig.dpi_scale_trans.inverted())
            wpx = bbox.width * self.view_fig.dpi
            hpx = bbox.height * self.view_fig.dpi
            if wpx > 0 and hpx > 0:
                sx = (xlim[1] - xlim[0]) / wpx
                sy = (ylim[1] - ylim[0]) / hpx
                self._sync_zoom_pan([xlim[0] - dx * sx, xlim[1] - dx * sx],
                                    [ylim[0] + dy * sy, ylim[1] + dy * sy])
        self.last_mx = event.x
        self.last_my = event.y

    def _on_view_release(self, event):
        self.view_dragging = False

    # === Config form load/dump ===
    def load_config(self, cfg):
        """Populate the form from a TrainViewerConfig."""
        dirs = list(cfg.data_dirs) if cfg.data_dirs else ([cfg.data_dir] if cfg.data_dir else [])
        self.set_data_dirs(dirs)
        self.var_patient_index.set(str(cfg.patient_index))
        self.var_epochs.set(str(cfg.epochs))
        self.var_val_every.set(str(cfg.val_every))
        self.var_batch_size.set(str(cfg.batch_size))
        self.var_learning_rate.set(str(cfg.learning_rate))
        self.var_size.set(str(cfg.size))
        self.var_val_fraction.set(str(cfg.val_fraction))
        self.var_device.set(cfg.device)
        self.var_distro.set(cfg.distro)
        self.var_venv_path.set(cfg.venv_path)
        self.var_project_path.set(cfg.project_path)
        # Leave the infer-tab entry empty by default so inference falls back to
        # the Train tab's multi-root data_dirs list.
        self.var_infer_data_dir.set("")
        self.var_infer_patient.set(str(cfg.infer_patient_index))

    def dump_config(self, cfg):
        """Write current form values back into a TrainViewerConfig."""
        cfg.data_dirs = self.get_data_dirs()
        # Keep the legacy single field in sync (first root) for compatibility.
        cfg.data_dir = cfg.data_dirs[0] if cfg.data_dirs else ""
        cfg.patient_index = _to_int(self.var_patient_index.get(), cfg.patient_index)
        cfg.epochs = _to_int(self.var_epochs.get(), cfg.epochs)
        cfg.val_every = _to_int(self.var_val_every.get(), cfg.val_every)
        cfg.batch_size = _to_int(self.var_batch_size.get(), cfg.batch_size)
        cfg.learning_rate = _to_float(self.var_learning_rate.get(), cfg.learning_rate)
        cfg.size = _to_int(self.var_size.get(), cfg.size)
        cfg.val_fraction = _to_float(self.var_val_fraction.get(), cfg.val_fraction)
        cfg.device = self.var_device.get()
        cfg.distro = self.var_distro.get()
        cfg.venv_path = self.var_venv_path.get()
        cfg.project_path = self.var_project_path.get()
        cfg.task = self.task_key
        cfg.infer_patient_index = _to_int(self.var_infer_patient.get(), cfg.infer_patient_index)
        cfg.infer_slice = int(self.infer_slice_scale.get())

    # === Tick / UI thread marshalling ===
    def _tick(self):
        if self.presenter:
            try:
                self.presenter.tick()
            except Exception as e:
                print(f"tick error: {e}")
        self.after(750, self._tick)

    def run_on_ui(self, fn):
        """Schedule fn to run on the Tk main thread (safe from worker threads)."""
        self.after(0, fn)

    def _on_close(self):
        if self.presenter:
            try:
                self.presenter.on_close()
            except Exception as e:
                print(f"close error: {e}")
        self.destroy()


def _fmt_eta(seconds):
    if seconds is None:
        return "-"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _format_smoke_report(report):
    """Turn a smoke_report.json dict into readable log lines (mirrors the CLI table)."""
    report = report or {}
    checks = report.get("checks", {})
    plan = report.get("plan", {})
    lines = [
        "===== SMOKE TEST REPORT =====",
        f"torch={checks.get('torch')}  cuda={checks.get('cuda_available')}  device={checks.get('device_name')}",
        f"data: patients={checks.get('data_patients')} ok={checks.get('data_ok')}",
        f"plan: epochs={plan.get('epochs')} x steps/epoch={plan.get('steps_per_epoch')}",
        f"{'stage':8} {'ok':5} {'ms/step':>10} {'peak MB':>9} {'ETA':>10}",
    ]
    for s in report.get("stages", []):
        ms = f"{s['ms_per_step']:.1f}" if s.get("ms_per_step") is not None else "-"
        mem = f"{s['peak_mem_mb']:.0f}" if s.get("peak_mem_mb") is not None else "-"
        ok = "OK" if s.get("ok") else "FAIL"
        lines.append(f"{s.get('stage',''):8} {ok:5} {ms:>10} {mem:>9} {_fmt_eta(s.get('eta_seconds')):>10}")
        if s.get("error"):
            lines.append(f"    error: {s['error']}")
    lines.append(f"{'TOTAL':8} {'':5} {'':>10} {'':>9} {_fmt_eta(report.get('eta_total_seconds')):>10}")
    lines.append("(ETA = train-step time only; excludes validation/checkpoint overhead)")
    return lines


def _to_int(s, default):
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return default


def _to_float(s, default):
    try:
        return float(s)
    except (ValueError, TypeError):
        return default
