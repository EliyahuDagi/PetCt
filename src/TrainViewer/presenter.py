from src.TrainViewer.model import TASK_DISPLAY_MAP, TASK_KEYS


class TrainPresenter:
    """Mediator between TrainModel and TrainView. Holds no Tk objects."""

    def __init__(self, model, view):
        self.model = model
        self.view = view
        # Current internal task key.
        self.task = self.model.config.task if self.model.config.task in TASK_KEYS else "ae2d"
        # Accumulated metric rows for the Log-tab plot, keyed by phase.
        self.metric_rows = []

        # Wire view -> presenter and push initial config into the form.
        self.view.set_presenter(self)
        self.view.load_config(self.model.config)
        self.view.set_task_key(self.task)
        # Initial dataset-size readout from the loaded roots.
        self.refresh_dataset_size()

    # --- task selection ---
    def set_task(self, display_label):
        """Called when the task combobox changes; display_label is the human label."""
        key = TASK_DISPLAY_MAP.get(display_label, "ae2d")
        self.task = key
        self.model.config.task = key
        # Reset the plotted metrics; we are now tailing a different file.
        self.metric_rows = []
        self.model.reset_metrics(key)
        self.view.set_task_key(key)
        # Refresh the View-tab checkpoint label for the new task.
        self.refresh_checkpoint()

    # --- form sync ---
    def _pull_form_into_config(self):
        """Read current form values from the view into model.config."""
        self.view.dump_config(self.model.config)

    # --- data_dirs / dataset size ---
    def on_data_dirs_changed(self):
        """Listbox Add/Remove changed: persist roots into config, refresh size."""
        self.model.config.data_dirs = self.view.get_data_dirs()
        self.refresh_dataset_size()

    def refresh_dataset_size(self):
        """Recompute the patient count across the current roots (defensive)."""
        roots = self.view.get_data_dirs()
        n = self.model.count_patients(roots)
        self.view.set_dataset_size(n)

    # --- training ---
    def start_training(self):
        self._pull_form_into_config()
        self.model.save_settings()
        self.metric_rows = []
        started = self.model.start_training(self.task)
        if started:
            self.view.set_status("running")
        else:
            self.view.set_status("already running / failed to start")

    def stop_training(self):
        self.model.stop_training()
        self.view.set_status("stopping...")

    # --- smoke test ---
    def run_smoke(self):
        if self.model.is_smoke_running():
            self.view.set_status("smoke test already running")
            return
        self._pull_form_into_config()
        self.model.save_settings()
        self.view.set_status("smoke test running...")

        def _on_done(report, error):
            if error is not None:
                self.view.run_on_ui(lambda: self.view.set_status(f"smoke failed: {error}"))
                return
            self.view.run_on_ui(lambda: self._show_smoke(report))

        self.model.run_smoke(_on_done)

    def _show_smoke(self, report):
        ok = report.get("all_ok")
        self.view.set_status("smoke OK" if ok else "smoke completed with failures")
        self.view.show_smoke_report(report)

    # --- periodic tick (driven by the view's after() loop) ---
    def tick(self):
        """Pull new stdout lines and metric rows; update the view. Never blocks."""
        # Training stdout.
        lines = self.model.drain_train_lines()
        if lines:
            self.view.append_log(lines)
        # Inference stdout (mirror into the same log pane).
        infer_lines = self.model.drain_infer_lines()
        if infer_lines:
            self.view.append_log(infer_lines)
        # Smoke-test stdout (mirror into the same log pane).
        smoke_lines = self.model.drain_smoke_lines()
        if smoke_lines:
            self.view.append_log(smoke_lines)
        # New metric rows.
        rows = self.model.read_new_metrics(self.task)
        if rows:
            self.metric_rows.extend(rows)
            self.view.update_curves(self.metric_rows, self.view.get_metric_key())
        # Status reflects running state / exit code.
        if self.model.is_training():
            self.view.set_status("running")
        else:
            code = self.model.training_exit_code()
            if code is not None:
                self.view.set_status(f"stopped (exit code {code})")

    def on_metric_changed(self):
        """Combobox metric key changed; redraw curves with the existing rows."""
        self.view.update_curves(self.metric_rows, self.view.get_metric_key())

    # --- checkpoints ---
    def refresh_checkpoint(self):
        info = self.model.checkpoint_info(self.task, "best")
        if info["present"]:
            text = f"best.pt present (modified {info['mtime_str']})"
        else:
            text = "no checkpoint yet"
        self.view.set_checkpoint_info(text)

    # --- inference ---
    def run_inference(self):
        self._pull_form_into_config()
        self.model.save_settings()
        data_dirs = self.view.get_infer_data_dirs()
        patient_index = self.view.get_infer_patient_index()
        slice_idx = self.view.get_infer_slice()
        self.view.set_infer_status("running inference...")

        def _on_done(result, error):
            # Marshal back onto the Tk main thread.
            if error is not None:
                self.view.run_on_ui(lambda: self.view.set_infer_status(f"error: {error}"))
                return
            pred, gt, meta = result
            self.view.run_on_ui(lambda: self._show_inference(pred, gt, meta))

        self.model.run_inference(self.task, data_dirs, patient_index, slice_idx, _on_done)

    def _show_inference(self, pred, gt, meta):
        self.view.set_infer_status("done")
        self.view.set_inference_data(pred, gt, meta)

    # --- shutdown ---
    def on_close(self):
        self._pull_form_into_config()
        self.model.save_settings()
        self.model.stop_training()
