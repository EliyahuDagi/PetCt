# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

PET/CT medical-imaging research code with three loosely-coupled parts:

1. **DataViewer** (`src/DataViewer/`) — a Tkinter desktop app for viewing CT, PET,
   and fusion volumes with segmentation overlays and a prostate "zone of interest"
   (ZOI) box.
2. **Core algorithms** (`src/utils/`) — prostate locator, volume geometry
   (voxel↔mm transforms), and segmentation utilities consumed by the viewer.
3. **Training pipeline** (`src/training/`) — a latent-diffusion pipeline for
   **NAC→AC PET translation** (predict attenuation-corrected PET from non-corrected),
   plus a **TrainViewer** GUI (`src/TrainViewer/`) that drives training/inference.

> Note: the root `README.md` describes an Organ-Conditioned Swin-UNETR segmentation
> design (`src/config.py`, `src/dataset.py`, `src/model.py`). Those files do **not**
> exist — treat that README as aspirational/stale. The sources below are ground truth.

## Layout

| Path | What lives here |
| --- | --- |
| `src/DataViewer/` | Viewer app: `main.py` (entry), `model.py` (data + locator bridge), `presenter.py`, `view.py` (MVP pattern). |
| `src/utils/` | `prostate_locator.py`, `geometry.py` (`VolumeGeometry`), `segmentors.py`, `segmentation_utils.py`, `config.py`. |
| `src/training/models/` | `autoencoder2d.py`, `autoencoder3d.py`, `diffusion2d.py`, `diffusion3d.py`, `inflation.py` (Conv2d→Conv3d centre inflation). |
| `src/training/train/` | `train_ae2d.py`, `train_ae3d.py`, `train_diff2d.py`, `train_ft3d.py`, `launcher.py`. |
| `src/training/utils/` | `sampling.py` (`DiffusionSchedule`: cosine/linear betas, zero-terminal-SNR, Min-SNR, DDIM/Karras), `schedule.py` (LR warmup/cosine + EMA), `checkpointing.py`, `logging.py`, `metrics.py`, `amp.py`. |
| `src/training/configs/` | YAML configs per stage (`ae2d.yaml`, `diff2d.yaml`, `ft3d.yaml`, `slices.yaml`). |
| `src/training/{data,infer}.py` | Dataset/manifest loading and the inference CLI. |
| `src/TrainViewer/` | Tkinter GUI to launch training, watch metrics, and view inference (MVP, mirrors DataViewer). |
| `src/tests/` | `pytest` suite (models, training steps, scheduling, viewers, resume, CPU end-to-end). |
| `scripts/`, `src/scripts/` | Data download/prep (TCIA), profiling, segmentation, `run_all_training.py`. |

## Training pipeline (the core algorithmic work)

Latent diffusion for NAC→AC PET, conditioned by **channel-concat** (UNet
`in_channels = 2×latent_channels`; input = `concat(noisy_AC_latent, NAC_latent)`).
Five stages; each writes JSONL metrics + `best.pt`/`last.pt` (the embedded `config`
lets inference rebuild models):

```
ae2d   2D AutoencoderKL on pooled NAC+AC slices      -> outputs/ae2d/best.pt
  └─ inflate ──> ae3d  spatial_dims=3 AE, depth-compressing -> outputs/ae3d/best.pt
diff2d 2D latent diffusion UNet on 2D-AE latents     -> outputs/diff2d/best.pt
  └─ inflate UNet ──> ft3d  3D latent diffusion on 3D-AE latents -> outputs/ft3d/best.pt
```

Output dir name == task key (the 3D diffusion stage is **`ft3d`**, never `diff3d`).

Key pitfalls (see `.github/agents/core-summary.md` for the full list):
- `diff2d` uses the 2D AE while `ft3d` uses the 3D AE — inflating the diff2d UNet
  into ft3d is a filter warm-start, **not** a latent-distribution match.
- `rescale_zero_terminal_snr` drives `alphas_cumprod[-1]→0`, breaking the one-step x0
  estimate used in validation — leave off unless validation is adapted.
- 3D self-attention is the main 3D-AE memory risk; attention is off by default.

### TrainViewer ⇄ training file contract (non-obvious)
- The GUI runs on **Windows and is torch-free**; all torch work runs in **WSL**
  (venv) via `subprocess`. They communicate **only through files** under `outputs/`
  (visible to WSL at `/mnt/c/...`).
- Training writes `outputs/<task>/metrics.jsonl` (one JSON row/line, `phase` =
  train|val) + `best.pt`/`last.pt`.
- Inference: `python -m src.training.infer` writes
  `outputs/infer/<task>/{pred.npy,gt.npy,meta.json}`.
- **Multi-root datasets**: every entry point takes `--data_dir` as `nargs="+"`
  (one flag, space-separated roots); `--patient_index` is a single global index
  across all roots, resolved via the torch-free `src/training/dataset_index.py`
  `enumerate_patients(...)`. AE stages (ae2d/ae3d) pool any available PET volume
  (unpaired NAC *or* AC is fine); diffusion stages require paired NAC+AC and skip
  patients missing a pair. Train/val is by-patient holdout when >1 usable patient,
  else the original within-patient depth split. The GUI Train tab lists roots in an
  Add/Remove listbox and shows a `Dataset: N patients` count.
- See `memory/train-viewer.md` for the durable design record.

## Running things

```bash
python src/DataViewer/main.py                 # viewer
python src/TrainViewer/main.py                # training GUI (set WSL venv path in Train tab)
python -m src.training.train.launcher ae2d --config src/training/configs/ae2d.yaml
python -m src.training.infer ...              # inference CLI
pytest src/tests/                             # tests (CPU end-to-end via test_train_scripts_cpu.py)
python -m src.tests.prostate_locator_debug --patient <id>
```

Environment: Windows host, PowerShell shell. GPU training runs under **WSL** with the
`requirements.txt` (CUDA 12.8 torch build) venv.

## Coordinate-system gotchas
- PET and CT grids can differ — always map through physical (mm) space via
  `VolumeGeometry` when comparing/combining; ZOI uses physical `bbox_mm`.
- Coronal/sagittal views are flipped with `np.flipud` in image extraction; any ZOI
  projection must match.

## Sub-agent delegation

Three project sub-agents in `.claude/agents/` split the codebase by ownership. Detailed
domain notes live in `.github/agents/` (`core-summary.md`, `viewer-summary.md`).

| Agent | Owns | Use it for | Must NOT touch |
| --- | --- | --- | --- |
| **core-agent** | Algorithms, geometry, numeric logic, training/diffusion pipeline | prostate locator, segmentation, `VolumeGeometry`, AE/diffusion models, noise schedules, samplers, LR/EMA | viewer/UI code |
| **viewer-agent** | Viewer/UI layer | DataViewer/TrainViewer presenter & view, image display, slice nav, overlays, ZOI rendering, Tkinter issues | core algorithm math |
| **manager-agent** | Coordination only | splitting cross-cutting work, routing, merging findings | editing files or running shell commands (delegates instead) |

Guidance:
- A change confined to algorithms/geometry/training → delegate to **core-agent**.
- A change confined to viewer/UI behavior → delegate to **viewer-agent**.
- A task spanning both (e.g. a locator change that needs a viewer overlay update) →
  use **manager-agent** to split and route, or invoke the two specialists directly.
- Keep edits within an agent's domain; both specialists are constrained to stay in
  their lane unless explicitly told otherwise.
