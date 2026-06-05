# Training

This package holds the training pipeline for 2D pretraining, 2D diffusion, and 3D fine-tuning.

Phases:
1) Build JSONL manifests for slices and volumes.
2) Train 2D AutoencoderKL (`train_ae2d`).
3) Inflate the 2D AE to 3D and fine-tune on volumes (`train_ae3d`) — gives the
   encoder volumetric context and compresses the depth axis.
4) Train 2D DiffusionModelUNet in latent space (`train_diff2d`).
5) Inflate the diffusion UNet to 3D and fine-tune on 3D-AE latents (`train_ft3d`).

## Datasets: multiple roots, multiple patients

Every stage's `--data_dir` accepts **one or more** dataset roots (space-separated
after a single flag):

```
python -m src.training.train.launcher ae2d --data_dir /data/siteA /data/siteB
python -m src.training.infer --task diff2d --data_dir /data/siteA /data/siteB --patient_index 3
```

Each root is interpreted exactly like the viewer's dataset loader: a root that
directly contains a patient-marker subdir (DICOM/CT/PT/PET/Segmentation/SECTRA) is
itself treated as one patient; otherwise each immediate subdir is a patient.
Patients are enumerated in root order (sorted within a root) and de-duplicated.
`--patient_index` (used by single-patient training, `infer`, and `smoke`) is a
**global** index across all roots. The enumeration lives in the torch-free module
`src/training/dataset_index.py` (`enumerate_patients(data_dirs, missing_ok=False)`)
so the GUI can call it on a Windows host without torch.

### Unpaired AE vs paired diffusion

- **Autoencoder stages (`ae2d`, `ae3d`)** pool *all available* PET volumes across
  patients. A patient may have only NAC, only AC, or both — unpaired data is fine
  (per patient, `ae_pool_volumes` prefers `[AC, NAC]` PET and falls back to `[CT]`).
- **Diffusion stages (`diff2d`, `ft3d`)** require **paired** NAC+AC per patient.
  Enumeration runs across all roots, then patients missing a pair are skipped (and
  logged); the run errors clearly if zero paired patients remain.

### Train/val split

With more than one usable patient, all stages use a deterministic **by-patient
holdout** (`make_patient_split`): `val_fraction` of patients are held out for
validation, guaranteeing at least one train and one val patient. Per step a random
train (or val) patient is loaded through a bounded LRU `PatientVolumeCache`
(`max_cached=4`) and sampled over its full depth. With a **single** usable patient
the stages fall back to the original within-patient split (a 2D depth-position
shuffle for `ae2d`/`diff2d`, a contiguous depth band for `ae3d`/`ft3d`), so
single-patient behavior is unchanged.

## Encoder: why a 3D AE

Earlier the 3D diffusion ran on a 2D AE applied slice-by-slice, so the latent had
no cross-slice features and no depth compression (depth stayed full-resolution).
`train_ae3d` fixes this: it builds a `spatial_dims=3` `AutoencoderKL` with the
*same* architecture as the 2D AE and centre-inflates the 2D weights into it (every
parameter maps; see `map_state_dict_2d_to_3d`). Each 3D conv starts out acting like
its 2D counterpart and learns z-mixing during a short volume fine-tune. The latent
becomes `(B, C, d, h, w)` with `d < D` (e.g. a 64³ crop → 16³ latent, 4× depth
compression), giving the diffusion stage a richer, smaller latent and consistent
(non-flickering) 3D reconstructions.

Stage flow and checkpoints:

```
ae2d  (2D AE)            -> outputs/ae2d/best.pt
  └─ inflate ──> ae3d    -> outputs/ae3d/best.pt   (frozen 3D encoder for ft3d)
diff2d (2D LDM, 2D AE)   -> outputs/diff2d/best.pt
  └─ inflate UNet ──> ft3d (3D LDM on 3D-AE latents) -> outputs/ft3d/best.pt
```

Note: `diff2d` still uses the 2D AE, so inflating its UNet into `ft3d` (which uses
the 3D-AE latent) is a filter warm-start, not a distribution match — useful but not
exact.

## Diffusion scheduling knobs

The diffusion trainers (`train_diff2d`, `train_ft3d`) read these optional config
keys (all have sensible defaults; set them in the YAML `--config`):

| Key | Default | Effect |
| --- | --- | --- |
| `noise_schedule` | `cosine` | Forward-process beta schedule: `cosine` (Nichol & Dhariwal) or `linear`. Cosine keeps more signal mid-trajectory and suits scans with large flat backgrounds. |
| `rescale_zero_terminal_snr` | `false` | Force SNR(T)→0 (Lin et al.) to remove the train/inference brightness mismatch. NOTE: drives `alphas_cumprod[-1]`→0, which breaks the one-step x0 estimate used in validation — leave off unless you adapt validation. |
| `snr_gamma` | `5.0` | Min-SNR-γ loss weighting (Hang et al.). Down-weights easy low-noise steps for faster, more balanced convergence. Set `null` for plain MSE. |
| `ema_decay` | `0.9999` | Exponential moving average of weights; validation/inference run under the EMA copy. Set `0` to disable. |
| `lr_warmup_steps` | `min(500, total//10)` | Linear LR warmup. |
| `lr_min_ratio` | `0.1` | Cosine-decay LR floor as a fraction of base LR. (`lr_total_steps` defaults to the run's total steps.) |

The noise schedule and EMA choice are recorded in the checkpoint config, so
`infer.py` rebuilds the matching schedule and loads EMA weights automatically.

### Inference (`infer.py`)

`--spacing karras` (default) places DDIM steps on a Karras σ grid and reaches good
quality in noticeably fewer steps; `--ddim_steps 25` is the new default (was 50).
`--use_ema`/`--no_ema` toggles EMA-weight inference (EMA on by default).
