# NAC -> AC PET: Standardized SOTA Comparison

This document standardizes how we evaluate the latent-diffusion NAC->AC PET
attenuation-correction models (`diff2d`, `ft3d`) and how we compare against the
published literature. It is the reference for any "are we SOTA?" claim.

> Honest framing up front: our evaluation runs on the **public ACRIN 6668
> (ACRIN-NSCLC-FDG-PET)** cohort, in this pipeline's **normalized-intensity
> space** (not calibrated SUV). The published references below are **private
> whole-body FDG** cohorts in calibrated SUV. Absolute SSIM/PSNR/NRMSE are
> therefore **not strictly comparable** -- see Caveats.

---

## 1. Evaluation protocol

### Dataset
- **ACRIN 6668 / ACRIN-NSCLC-FDG-PET** (TCIA, public). This is the project's only
  paired NAC+AC cohort, so our numbers are reproducible even though the
  references' cohorts are not. NSCLC (lung) FDG, not whole-body.
- Train/val split is by-patient holdout, reconstructed in eval to match training:
  `enumerate_patients(roots) -> filter_paired_patients (NAC+AC) ->
  make_patient_split(n, val_fraction, seed)`. Pass the **same `--val_fraction`
  and `--seed`** the training run used (GUI default `val_fraction=0.1`,
  `seed=42`).

### Commands (run in WSL/GPU from the project root)

`diff2d` (2D latent diffusion; ~32 slices/patient over val patients):

```bash
python -m src.training.evaluate --task diff2d \
    --data_dir "/mnt/d/DeepTrainingData/Project/ACRIN 6668" \
    --mode val --val_fraction 0.1 --seed 42 \
    --size 128 --num_slices 32 --ddim_steps 25 --spacing karras \
    --device cuda --out outputs/eval/diff2d/metrics.json
```

`ft3d` (3D latent diffusion; one whole-volume prediction per val patient):

```bash
python -m src.training.evaluate --task ft3d \
    --data_dir "/mnt/d/DeepTrainingData/Project/ACRIN 6668" \
    --mode val --val_fraction 0.1 --seed 42 \
    --size 64 --ddim_steps 25 --spacing karras \
    --device cuda --out outputs/eval/ft3d/metrics.json
```

Notes:
- `--size` MUST match the input size used when that stage was trained (latents
  depend on it). Defaults mirror `infer.py` (diff2d=128, ft3d=64).
- `--mode val` only reproduces the training split if the dataset is unchanged
  since training; otherwise use `--mode all` (lazily skips unpaired patients).
- Output lands in `outputs/eval/<task>/metrics.json` (per-patient + summary).

### Metric panel

**Image tier** (`src/training/utils/image_metrics.py`):
| Key | Metric |
| --- | --- |
| `ssim` | Gaussian-windowed SSIM (2D/3D) |
| `psnr` | Peak SNR (dB), range-normalized |
| `nrmse` | Range-normalized RMSE |
| `mae` | Mean absolute error |

**Clinical tier** (the panel that decides SOTA -- all normalized-intensity
proxies until SUV calibration + VOIs exist):
| Key | Metric | SOTA analogue |
| --- | --- | --- |
| `rel_bias` | Foreground mean relative difference | SUV mean %-bias (organ) |
| `max_rel_error` | Foreground max relative error | Lesion SUVmax error |
| `voxel_r2` | R^2 of pred vs gt over foreground voxels | Voxel-wise R^2 |
| `reg_slope` / `reg_intercept` | LS slope/intercept of pred on gt (ideal slope=1) | Regression slope |
| `ba_mean_bias` | Mean bias of (pred-gt) over foreground | Bland-Altman mean bias |
| `ba_loa_lower` / `ba_loa_upper` | mean_diff +/- 1.96*SD | Bland-Altman 95% LoA |

All metrics are reported as mean +/- SD across evaluated patients; non-finite
values (R^2/slope on flat regions) are filtered per metric in `_aggregate`.

---

## 2. Comparison table

Published references are private cohorts in calibrated SUV; "Ours" columns are
public ACRIN 6668 (NSCLC FDG) in normalized intensity -- fill after the WSL/GPU
eval run. See Caveats before reading any row as a head-to-head.

| Method | Cohort (private?) | Tracer | Region | n (train/test) | SSIM | PSNR (dB) | NRMSE | Organ SUV bias | Lesion SUV bias/SUVmax |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Shiri et al. 2021, 2D U-Net (image-domain) [ref] | private | FDG | whole-body | (large) | 0.98 +/- 0.01 | 36.3 +/- 3.0 | 0.21 +/- 0.05 | -0.8 +/- 8.6% (organ) | 0.9 +/- 9.2% (lesion) |
| Eur Radiol 2024, CycleGAN total-body [ref] | private | FDG | total-body | n=122 | 0.980 +/- 0.041 | 36.92 +/- 5.49 | -- | -- | -- |
| GAN WB-FDG (pseudo-CT) [ref] | private | FDG | whole-body | 100 / 25 | -- | -- | -- | -- | -- |
| MADM, 2.5D multi-view diffusion [ref] | private | FDG | (per paper) | (per paper) | -- | -- | -- | -- | -- |
| **Ours - diff2d (2D latent diffusion)** | **public ACRIN 6668 (+ Bladder/Normal for AE)** | FDG | **NSCLC (lung)** | 130 / 19 (paired) | 0.185 ± 0.034 | 14.30 ± 0.90 | 0.196 ± 0.021 | -0.797 ± 0.071 (`rel_bias`) | 0.141 ± 0.173 (`max_rel_error`) |
| **Ours - ft3d (3D latent diffusion)** | **public ACRIN 6668 (+ Bladder/Normal for AE)** | FDG | **NSCLC (lung)** | 130 / 19 (paired) | 0.012 ± 0.005 | -2.24 ± 0.33 | 1.295 ± 0.048 | 3.77 ± 0.76 (`rel_bias`) | 20.2 ± 3.0 (`max_rel_error`) |

> **These rows are a NEGATIVE result, not a SOTA claim.** They are the honest
> held-out TEST numbers from the 2026-06-12 full run (clamped DDIM, EMA weights),
> and they are far below the references because the diffusion models do **not** yet
> perform the NAC→AC translation — see §5. They are recorded for reproducibility and
> to track progress, not as a head-to-head.

Reference-method anchor: Shiri et al. 2021 is the image-domain reference. MADM is
the closest diffusion competitor. Our "Lesion SUV" cells report the
`max_rel_error` proxy until SUV calibration + lesion VOIs are added.

---

## 3. Caveats (read before comparing)

1. **NSCLC vs whole-body FDG.** ACRIN 6668 is lung-cancer (NSCLC) FDG; the
   references are whole-body / total-body FDG. Body region and disease
   distribution differ, so absolute metrics are not directly comparable.
2. **Normalized intensity vs calibrated SUV.** Our pipeline carries arbitrarily
   scaled intensities, not calibrated SUV. SSIM/PSNR/NRMSE depend on the data
   range and intensity normalization, so absolute values are **not strictly
   comparable** to SUV-domain papers; trends and within-pipeline deltas are the
   honest readout.
3. **SUV %-bias and lesion SUVmax are proxies.** `rel_bias` (organ SUV mean bias)
   and `max_rel_error` (lesion SUVmax error) are normalized-intensity proxies.
   True calibrated SUV %-bias and lesion SUVmax error require **SUV calibration
   (Bq/mL, injected dose, body weight) + organ/lesion VOIs**, which the pipeline
   does **not** currently track. `voxel_r2`, `reg_slope`, and the Bland-Altman
   numbers are scale-aware but still in normalized intensity.
4. **Generation must be working.** If the diffusion latents are not normalized to
   ~unit scale before sampling, DDIM (which starts from N(0,1)) generates
   near-blank output and the metrics collapse (SSIM ~ 0, PSNR ~ 5 dB). Verify a
   latent scale factor is applied consistently across train_diff2d / train_ft3d /
   infer / evaluate before trusting any number in this table.

---

## 5. 2026-06-12 full run — results & findings (READ THIS)

### Run configuration
- **Data**: 461 patients SSD-pre-cached to `C:\DeepTrainingData\PetCT` (7.0 GB,
  float16); **186 paired NAC+AC** (all from ACRIN 6668). AE stages pooled all PET
  (ACRIN + Bladder + PET_CT_NORMAL_2); diffusion stages used the 186 paired only.
- **Split**: by-patient train/val/test. Paired (diffusion): **130 / 37 / 19**.
  Test patients held out from BOTH train and val. Eval below is on the **test** set.
- **Augmentation**: geometric-only (flip/rot90/affine) enabled on TRAIN for all 4
  stages; val/test clean.
- **Training budget (full-day run)**: ae2d 100k steps, ae3d 25k, diff2d **300k**,
  ft3d 60k (all rc=0, ~17h on an RTX 3090 with the warm SSD cache). diff2d
  in-training val: loss 0.78→0.033, one-step-proxy SSIM 0.48→0.78. (An earlier
  10×-shorter run — diff2d 30k — gave diff2d TEST SSIM 0.098; the long run lifted
  it to 0.185, see below.)

### Finding: generation collapses despite healthy training metrics
Held-out TEST full-DDIM generation is far below the in-training validation. Two
distinct, independently-confirmed causes:

1. **DDIM sampler exploded at terminal SNR (now FIXED).** The cosine schedule has
   `alphas_cumprod[-1] ≈ 2.4e-9`, so the first sampling step's
   `x0 = (x - √(1-acp)·eps)/√(acp)` divides by ~5e-5 and amplifies any eps error
   ~10⁴×; the sampled latent std blew up to ~3×10³ and decoded output was noise
   (SSIM~0, PSNR~4 dB). **Fix**: added static thresholding (`clip_x0`, default 4.0)
   to `DiffusionSchedule.ddim_sample`, wired through `evaluate.py`/`infer.py`. This
   recovered diff2d to PSNR 14.4 / NRMSE 0.19, but **ft3d remained collapsed** (its
   3D latent needs its own clip/retune).

2. **The model under-uses the NAC conditioning (the real ceiling).** The one-step
   x0 proxy used in training-val is *optimistic*: it denoises a noised copy of the
   **real AC** latent, so it scores well without needing NAC. Conditioning ablation
   (t=500 one-step proxy, 12 test slices), short run vs full-day run:

   | conditioning | SSIM (30k run) | SSIM (300k run) |
   | --- | --- | --- |
   | real NAC | 0.762 | 0.883 |
   | **zeroed** | 0.731 | 0.857 |
   | shuffled NAC | 0.726 | 0.854 |

   NAC adds only **~0.03 SSIM** — and that gap **did not grow** when training went
   10× longer (0.031→0.026). So more steps improve denoising but **do not** make the
   model use the condition. For reference: decoding the NAC latent directly (no
   model) = **0.48** SSIM vs AC; AE reconstruction ceiling = **0.86**. SDEdit-from-NAC
   (0.37–0.43) is *worse* than NAC passthrough — the trajectory drifts away from the
   target. So actual NAC→AC translation is not happening.

3. **More training is not the fix (confirmed empirically).** A full-day run (diff2d
   300k vs 30k steps, ~17h) lifted diff2d TEST SSIM 0.098→0.185 but PSNR/NRMSE were
   flat (14.3 dB / 0.196) and voxel R² stayed negative; the conditioning gap above
   stayed ~0.03. **ft3d got worse** (PSNR went negative, −2.2 dB; SSIM 0.012) — the
   3D stage diverges and needs its own clip/latent-scale retune, separate from the
   conditioning issue.

### Recommended next step (model-design decision)
Add **classifier-free guidance**: train the diffusion UNets with NAC-conditioning
dropout (p≈0.1–0.2) and sample with a guidance scale >1 (cond + uncond pass in the
sampler). This directly targets the proven failure (condition is ignored), which
longer training does not. Optionally switch to **v-prediction** or
**zero-terminal-SNR** (and adapt the val proxy) for high-t stability, and debug ft3d
3D divergence separately. Until then the rows in §2 are a baseline/negative result,
not a SOTA entry. Diagnostics that produced these findings: `scripts/_diag_sample*.py`,
`scripts/_diag_cond.py`.

---

## 4. References

- TCIA ACRIN-NSCLC-FDG-PET (ACRIN 6668):
  https://www.cancerimagingarchive.net/collection/acrin-nsclc-fdg-pet/
- Shiri et al., Radiology: Artificial Intelligence 2021 (2D U-Net, image-domain
  direct AC): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8043359/
- Eur Radiol 2024, CycleGAN total-body (n=122), PubMed 38355987 /
  doi:10.1007/s00330-024-10647-1:
  https://doi.org/10.1007/s00330-024-10647-1
- GAN whole-body FDG (pseudo-CT), PMC7246235:
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7246235/
- MADM (2.5D multi-view diffusion), arXiv 2406.08374:
  https://arxiv.org/abs/2406.08374
