# TrainViewer Architecture Summary

**Project:** PetCt Train Viewer (Python/Tkinter/Matplotlib)
**Architecture:** Model-View-Presenter (MVP), mirroring `src/DataViewer`.

A single Tk window that launches/monitors training jobs in WSL, tails their
metrics + stdout, and runs inference to compare Predicted vs Ground Truth.

The GUI never imports torch. It only spawns WSL subprocesses and reads files
(metrics.jsonl, *.pt presence, infer/*.npy) directly via Windows paths.

---

## 1. Files

### `main.py`
Entry point. Adds the repo root to `sys.path`, then:
`model = TrainModel(); view = TrainView(); presenter = TrainPresenter(model, view); view.mainloop()`.
Runnable as `python -m src.TrainViewer.main` or `python src/TrainViewer/main.py`.

### `model.py`
All subprocess + file logic. No Tk.
- `TrainViewerConfig` (dataclass): form + WSL defaults; `project_path` auto-fills
  with the WSL form of the repo root.
- `win_to_wsl_path(p)`: `C:\\x` -> `/mnt/c/x`; passes through `/...` and `~...`.
- `MetricsReader(path)`: incremental tail of `metrics.jsonl` by byte offset;
  returns only new fully-parsed rows; resets if the file shrinks/recreates.
- `CheckpointIndex(outputs_root)`: torch-free `os.path.exists` + mtime for
  `outputs/<task>/best.pt` / `last.pt`.
- `WslRunner`: builds & spawns the training command via `subprocess.Popen`,
  a daemon thread reads stdout into a `queue.Queue`. `.start/.stop/.is_running/.drain_lines`.
- `InferenceRunner`: spawns the infer CLI on a background thread, then loads
  `pred.npy`, `gt.npy`, `meta.json`; calls an `on_done(result, error)` callback.
- `TrainModel`: owns config/runners/readers/checkpoints and settings persistence
  to `outputs/trainviewer_settings.json`.

### `view.py` (`TrainView(tk.Tk)`)
All Tk + matplotlib. Top bar task `ttk.Combobox` + a `ttk.Notebook` with tabs:
- **Train**: hyperparameter form (Entry/Spinbox/Combobox), Browse for data_dir,
  WSL settings group, Start/Stop buttons, status label. The size field label
  adapts: `slice_size` (ae2d), `crop_size` (ae3d), `latent_size` (diff2d/ft3d).
- **Log**: a matplotlib figure plotting train vs val for a chosen metric key
  (combobox: loss/recon_l1/kl/psnr) vs step, plus an append-only `tk.Text` log.
- **View**: Load best model, data_dir/patient_index/slice inputs, Run inference,
  and a 3-axis figure: Predicted | Ground Truth | Difference (pred-gt, `bwr`).
  Predicted/GT use `gray`. Window/level drag (right-click) and synced zoom/pan
  (ctrl+scroll / middle-drag) mirror DataViewer; the slice slider scrolls all
  three axes for 3D (ft3d) volumes; 2D tasks show a single frame.
A `self.after(750, self._tick)` loop drives `presenter.tick()` without blocking
the mainloop. `run_on_ui(fn)` marshals worker-thread callbacks back to Tk.

### `presenter.py` (`TrainPresenter`)
Orchestration only; no Tk objects. Maps task display labels to internal keys,
pulls form values into config, starts/stops training, polls model queues on each
tick, updates curves/log, refreshes checkpoint info, and drives inference.

---

## 2. Training-side contract (implemented separately; do NOT edit here)

- Repo root (Windows): `c:\Users\algo\VScodeProjects\PetCt\PetCt`; outputs under
  `<repo>/outputs/`, read directly via the Windows path.
- Tasks: `ae2d` (2D AE), `ae3d` (3D AE, inflated from ae2d), `diff2d` (NAC->AC 2D),
  `ft3d` (NAC->AC 3D, uses the 3D AE).
- Metrics: `outputs/<task>/metrics.jsonl`, one JSON object per line with keys
  `task, phase ("train"|"val"), step, epoch?, loss, recon_l1?, kl?, psnr?, wall_time`.
- Checkpoints: `outputs/<task>/best.pt`, `last.pt` (presence + mtime only).
- Train launch:
  ```
  wsl -d <distro> bash -lc "source <venv>/bin/activate && cd <proj> && \
    python -m src.training.train.launcher <task> --data_dir <wsl_data> \
    --patient_index N --device <dev> --epochs E --val_every V \
    --val_fraction F --batch_size B --learning_rate LR <SIZE_ARG>"
  ```
  `<SIZE_ARG>` = `--slice_size S` (ae2d), `--crop_size S` (ae3d), or
  `--latent_size S` (diff2d/ft3d). For ft3d, appends
  `--inflate_from outputs/diff2d/best.pt` if that file exists.
- Infer launch:
  ```
  wsl -d <distro> bash -lc "source <venv>/bin/activate && cd <proj> && \
    python -m src.training.infer --task <task> --data_dir <wsl_data> \
    --patient_index N --slice K --ae_ckpt <AE_CKPT> \
    [--diff_ckpt outputs/<task>/best.pt] --out outputs/infer/<task>"
  ```
  `<AE_CKPT>` = `outputs/ae3d/best.pt` for ae3d/ft3d, else `outputs/ae2d/best.pt`.
  `--diff_ckpt` is included only for diff2d/ft3d. On exit 0, loads
  `outputs/infer/<task>/{pred.npy, gt.npy, meta.json}`. `meta.json` may carry
  `spacing, origin, vmin, vmax`; missing vmin/vmax are computed from the arrays.

---

## 3. Usage
```bash
python -m src.TrainViewer.main
# or
python src/TrainViewer/main.py
```
1. Pick a task in the top combobox.
2. Train tab: set hyperparameters + WSL settings, click **Start Training**.
3. Log tab: watch train/val curves and streaming stdout.
4. View tab: **Load best model**, set patient/slice, **Run inference**, compare
   Predicted | Ground Truth | Difference.

Settings persist to `outputs/trainviewer_settings.json` on Start/Run/close and
reload on startup.

**Dependencies:** `tkinter`, `matplotlib`, `numpy`. (No torch, no pydicom.)
