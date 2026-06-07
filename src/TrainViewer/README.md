# TrainViewer Architecture Summary

**Project:** PetCt Train Viewer (Python/Gradio)
**Architecture:** Model + UI-agnostic Controller + Gradio UI.

A Gradio web app that launches/monitors training jobs in WSL, tails their
metrics + stdout, and runs inference to compare Predicted vs Ground Truth.

The GUI never imports torch. It only spawns WSL subprocesses and reads files
(metrics.jsonl, *.pt presence, infer/*.npy) directly via Windows paths.
`model.py` and `controller.py` import neither gradio nor tkinter, so the logic
stays unit-testable; only `app.py` imports gradio.

---

## 1. Files

### `main.py`
Entry point. Adds the repo root to `sys.path`, then:
`model = TrainModel(); app = build_app(model); app.launch(inbrowser=True, share=False)`.
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

### `app.py` (Gradio UI)
`build_app(model=None) -> gr.Blocks` + `launch()`. All gradio lives here. A top
`gr.Dropdown` task selector + `gr.Tabs`:
- **Train**: hyperparameter form (`gr.Number`/`gr.Dropdown`), a single-column
  `gr.Dataframe` of dataset roots with a live `Dataset: N patients` count, WSL
  settings group, Start/Stop/Smoke buttons, status box. The size field label
  adapts: `slice_size` (ae2d), `crop_size` (ae3d), `latent_size` (diff2d/ft3d).
- **Log**: run `gr.Dropdown` + metric `gr.Dropdown` (loss/recon_l1/kl/psnr), a
  native `gr.LinePlot` (x=step, y=value, color=phase) of train vs val, and an
  append-only scrolling log `gr.Textbox`.
- **View**: Load best model, infer data_dir/patient_index inputs, a **slice
  slider** + **vmin/vmax window sliders**, Run inference, and three `gr.Image`s:
  Predicted | Ground Truth | Difference. Pred/GT are windowed grayscale uint8;
  Difference maps `pred-gt` through matplotlib's `bwr` colormap (symmetric about
  0) to RGB uint8 (no figure). 3D (ft3d) volumes drive the slice slider range;
  2D tasks show a single frame.
A `gr.Timer(0.75)` drives `controller.poll()` each tick, appending stdout to the
log, redrawing the curve when metric rows change, and refreshing status + runs.

### `controller.py` (`TrainController`, no gradio/tkinter)
UI-agnostic orchestration. Maps task display labels to internal keys, starts/
stops training, polls model queues via `tick()`/`poll()` returning a
`PollResult`, resolves the live/pinned run path, recomputes dataset size,
refreshes checkpoint info, and provides blocking wrappers around the model's
callback-async smoke + inference runners. Settings are saved at the same points
the old presenter saved them.

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
1. Pick a task in the top dropdown.
2. Train tab: set hyperparameters + WSL settings, click **Start Training**.
3. Log tab: watch train/val curves and streaming stdout.
4. View tab: **Load best model**, set patient, **Run inference**, then use the
   slice + vmin/vmax sliders to inspect Predicted | Ground Truth | Difference.

Settings persist to `outputs/trainviewer_settings.json` on Start/Run/smoke and
reload on startup.

**Dependencies:** `gradio`, `matplotlib`, `numpy`. (No torch, no tkinter, no
pydicom.)
