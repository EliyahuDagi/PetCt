"""UI-agnostic controller for the Train Viewer.

Absorbs all of the mediator logic that used to live in presenter.py, but it
returns plain data instead of pushing into a Tk view. The Gradio layer in
app.py consumes these return values; the controller itself imports neither
gradio nor tkinter so it stays unit-testable in isolation.
"""

from src.TrainViewer.model import TASK_DISPLAY_MAP, TASK_KEYS, DIFFUSION_TASKS

# Label for the auto-follow option in the Run selector (run_id == None). Must
# match what the UI shows; the UI rebuilds its label->id map from what we send.
LIVE_RUN_LABEL = "Latest (live)"

# Metric keys the user can plot (mirrors the old view's METRIC_KEYS).
METRIC_KEYS = ["loss", "recon_l1", "kl", "psnr"]


def _run_label(run_id):
    """Human-readable label for an archived run id (a YYYYMMDD-HHMMSS stamp)."""
    if run_id == "current":
        return "current run"
    digits = run_id.replace("-", "")
    if len(run_id) == 15 and run_id[8] == "-" and digits.isdigit():
        d, t = run_id[:8], run_id[9:]
        return f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}:{t[4:6]}"
    return run_id


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


class PollResult:
    """Structured snapshot returned by poll()/tick() for the UI to consume.

    Attributes:
        new_log_lines: list of stdout lines drained this tick (append to the log).
        metric_rows:   snapshot (copy) of the accumulated metric rows.
        status:        training status string ("running" / "stopped (exit code N)"
                       / None if unchanged-and-not-running).
        run_options:   list of {"run_id", "label"} for the Run selector.
        selected_run_id: the currently-selected run id (None == live).
        is_training:   whether the trainer process is alive.
        curves_dirty:  True when metric_rows changed this tick (UI may redraw).
    """

    def __init__(self, new_log_lines, metric_rows, status, run_options,
                 selected_run_id, is_training, curves_dirty):
        self.new_log_lines = new_log_lines
        self.metric_rows = metric_rows
        self.status = status
        self.run_options = run_options
        self.selected_run_id = selected_run_id
        self.is_training = is_training
        self.curves_dirty = curves_dirty


class TrainController:
    """Mediator between TrainModel and the (Gradio) UI. Holds no UI objects."""

    def __init__(self, model):
        self.model = model
        # Current internal task key.
        self.task = self.model.config.task if self.model.config.task in TASK_KEYS else "ae2d"
        # Accumulated metric rows for the Log-tab plot.
        self.metric_rows = []
        # Selected run: None == "Latest (live)" auto-follow of the newest run.
        self.selected_run_id = None
        # Metrics file currently being tailed; switching it clears the plot.
        self._active_path = None

    # --- task selection ---
    def set_task(self, display_label):
        """Called when the task selector changes; display_label is the human label.

        Returns the resolved internal task key.
        """
        key = TASK_DISPLAY_MAP.get(display_label, "ae2d")
        self.task = key
        self.model.config.task = key
        # New task -> follow its live run again and start the plot fresh.
        self.selected_run_id = None
        self.metric_rows = []
        self._active_path = None
        return key

    # --- run selection ---
    def refresh_runs(self):
        """Rebuild the Run selector options from the runs on disk for the current task.

        Returns a list of {"run_id", "label"} (the live option is always first).
        """
        options = [{"run_id": None, "label": LIVE_RUN_LABEL}]
        for r in self.model.list_runs(self.task):
            options.append({"run_id": r["run_id"], "label": _run_label(r["run_id"])})
        return options

    def on_run_changed(self, run_id):
        """Run selector changed: switch the tailed file and reset the plot.

        Returns the PollResult from a fresh tick (the UI redraws from it).
        """
        self.selected_run_id = run_id
        self.metric_rows = []
        self._active_path = None  # force tick() to re-resolve and read in full
        return self.tick()

    def _resolve_run_path(self):
        """Path of the metrics file for the current selection, or None."""
        if self.selected_run_id is None:
            latest = self.model.latest_run(self.task)
            return latest["path"] if latest else None
        path = self.model.run_path(self.task, self.selected_run_id)
        if path is None:
            # Selected run vanished -> fall back to live.
            self.selected_run_id = None
            latest = self.model.latest_run(self.task)
            return latest["path"] if latest else None
        return path

    # --- data_dirs / dataset size ---
    def on_data_dirs_changed(self, dirs):
        """Roots changed: persist into config, return the recomputed patient count."""
        self.model.config.data_dirs = list(dirs or [])
        return self.refresh_dataset_size(self.model.config.data_dirs)

    def refresh_dataset_size(self, dirs):
        """Recompute the patient count across the given roots (int or None)."""
        return self.model.count_patients(list(dirs or []))

    # --- training ---
    def start_training(self):
        """Save settings and launch training. Returns a status string."""
        self.model.save_settings()
        # Follow the new run as it appears.
        self.selected_run_id = None
        self.metric_rows = []
        self._active_path = None
        started = self.model.start_training(self.task)
        return "running" if started else "already running / failed to start"

    def stop_training(self):
        self.model.stop_training()
        return "stopping..."

    # --- smoke test ---
    def run_smoke_blocking(self):
        """Run the smoke test and block until it completes.

        Returns (report_lines, status). status is "smoke OK" /
        "smoke completed with failures" / "smoke failed: <err>" /
        "smoke test already running". On the already-running path report_lines
        is None.
        """
        if self.model.is_smoke_running():
            return None, "smoke test already running"
        self.model.save_settings()

        # The model's run_smoke is callback-async; wrap it with an Event so we
        # can present a blocking variant to the (generator-friendly) UI layer.
        import threading

        done = threading.Event()
        captured = {"report": None, "error": None}

        def _on_done(report, error):
            captured["report"] = report
            captured["error"] = error
            done.set()

        self.model.run_smoke(_on_done)
        done.wait()

        if captured["error"] is not None:
            return None, f"smoke failed: {captured['error']}"
        report = captured["report"]
        ok = report.get("all_ok")
        status = "smoke OK" if ok else "smoke completed with failures"
        return _format_smoke_report(report), status

    # --- periodic poll/tick ---
    def tick(self):
        """Drain new stdout + metric rows and return a PollResult. Never blocks.

        Preserves the exact semantics of the old presenter.tick():
        - switching files clears metric_rows + resets the reader;
        - an external restart (file truncated/recreated) drops stale rows;
        - status reflects running / "stopped (exit code N)".
        """
        new_lines = []
        # Training stdout.
        new_lines.extend(self.model.drain_train_lines())
        # Inference stdout (mirror into the same log).
        new_lines.extend(self.model.drain_infer_lines())
        # Smoke-test stdout (mirror into the same log).
        new_lines.extend(self.model.drain_smoke_lines())

        # Keep the Run selector in sync with runs appearing/finishing on disk.
        run_options = self.refresh_runs()

        curves_dirty = False
        # Resolve which run's metrics file to tail (live = newest run).
        path = self._resolve_run_path()
        if path is not None:
            # Switching files (live-follow jumped to a new run, or the user picked
            # a different run) starts the plot over from that file's contents.
            if path != self._active_path:
                self.metric_rows = []
                self.model.reset_metrics_path(path)
                self._active_path = path
                curves_dirty = True

            # New metric rows. If the file was truncated/recreated (a new run
            # started outside the GUI), drop the previous run's rows so the plot
            # does not mix stale data with the live run -- both restart at step 0.
            rows = self.model.read_new_metrics_path(path)
            restarted = self.model.consume_metrics_restart_path(path)
            if restarted:
                self.metric_rows = []
            if rows:
                self.metric_rows.extend(rows)
            if restarted or rows:
                curves_dirty = True

        # Status reflects running state / exit code.
        is_training = self.model.is_training()
        status = None
        if is_training:
            status = "running"
        else:
            code = self.model.training_exit_code()
            if code is not None:
                status = f"stopped (exit code {code})"

        return PollResult(
            new_log_lines=new_lines,
            metric_rows=list(self.metric_rows),
            status=status,
            run_options=run_options,
            selected_run_id=self.selected_run_id,
            is_training=is_training,
            curves_dirty=curves_dirty,
        )

    # poll() is an alias so the UI timer can call either name.
    def poll(self):
        return self.tick()

    # --- checkpoints ---
    def refresh_checkpoint(self, variant="eps"):
        """Return the checkpoint label text for the current task.

        For the NAC->AC diffusion tasks the variant selects the epsilon vs flow
        checkpoint dir (outputs/<task> vs outputs/<task>_flow) and the label names
        which one. AE tasks ignore the variant and keep the original wording so
        existing callers/tests are unaffected.
        """
        if self.task in DIFFUSION_TASKS:
            info = self.model.diffusion_ckpt_info(self.task, variant)
            kind = "flow" if variant == "flow" else "diffusion (ε)"
            if info["present"]:
                return f"{kind} best.pt present (modified {info['mtime_str']})"
            return f"no {kind} checkpoint yet"
        info = self.model.checkpoint_info(self.task, "best")
        if info["present"]:
            return f"best.pt present (modified {info['mtime_str']})"
        return "no checkpoint yet"

    def checkpoint_present(self, variant="eps"):
        """True when the checkpoint the next inference run would load exists.

        Diffusion tasks resolve the variant's dir (flow vs eps); other tasks use
        the base best.pt. Lets the UI block a doomed run with a clear message.
        """
        if self.task in DIFFUSION_TASKS:
            return bool(self.model.diffusion_ckpt_info(self.task, variant)["present"])
        return bool(self.model.checkpoint_info(self.task, "best")["present"])

    # --- inference ---
    def run_inference_blocking(self, data_dirs, patient_index, slice_idx,
                               variant="eps", steps=None, guidance=None):
        """Run inference and block until it completes.

        variant/steps/guidance select the eps vs flow checkpoint and sampling
        controls (threaded through to model.run_inference / infer.py).
        Returns (pred, gt, nac, meta, status); nac is the NAC input volume when
        the task wrote one (diff2d/ft3d) else None. On failure pred/gt/nac/meta
        are None and status is "error: <msg>"; on success status is "done".
        """
        self.model.save_settings()

        import threading

        done = threading.Event()
        captured = {"result": None, "error": None}

        def _on_done(result, error):
            captured["result"] = result
            captured["error"] = error
            done.set()

        self.model.run_inference(
            self.task, data_dirs, patient_index, slice_idx, _on_done,
            variant=variant, steps=steps, guidance=guidance,
        )
        done.wait()

        if captured["error"] is not None:
            return None, None, None, None, f"error: {captured['error']}"
        pred, gt, nac, meta = captured["result"]
        return pred, gt, nac, meta, "done"

    # --- shutdown ---
    def on_close(self):
        """App shutdown: persist settings and stop any running training."""
        self.model.save_settings()
        self.model.stop_training()
