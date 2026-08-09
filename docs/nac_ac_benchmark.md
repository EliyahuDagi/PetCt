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
| ~~`max_rel_error`~~ | Foreground max relative error | ~~Lesion SUVmax error~~ **DO NOT USE -- see below** |
| `p95_rel_error` | Relative error of the 95th foreground percentile | High-uptake fidelity |
| `hot_band_rel_error` | Relative error of the mean over the GT p90-p98 band | High-uptake fidelity |
| `voxel_r2` | R^2 of pred vs gt over foreground voxels | Voxel-wise R^2 |
| `reg_slope` / `reg_intercept` | LS slope/intercept of pred on gt (ideal slope=1) | Regression slope |
| `ba_mean_bias` | Mean bias of (pred-gt) over foreground | Bland-Altman mean bias |
| `ba_loa_lower` / `ba_loa_upper` | mean_diff +/- 1.96*SD | Bland-Altman 95% LoA |

All metrics are reported as mean +/- SD across evaluated patients; non-finite
values (R^2/slope on flat regions) are filtered per metric in `_aggregate`.

#### 2026-08-06: `max_rel_error` was never a lesion-SUVmax proxy -- retract it

`normalize_volume` (`src/training/data.py`) percentile-normalizes **and clips** every
volume to `[0,1]`. Two consequences, both measured:

1. **A GT foreground max is always exactly 1.0**, so `max_rel_error` reduces to
   `pred_max - 1.0`: it measured how far the *decoder rang above the valid range*, not
   lesion accuracy. Nothing was clamping the decoded prediction (MONAI's `AutoencoderKL`
   ends in a linear conv), and it reached ~1.80. Adding the clamp -- now the default,
   `--no_clamp_output` to opt out -- takes it from **0.8035 to 0.0000** on the 41-patient
   test set with **no retraining**. It is retained in the panel only for continuity with
   the older rows below.
2. **The clip saturates the top of every volume**: on an ACRIN test patient, 3.26% of
   foreground voxels sit at exactly 1.0, so GT p98 / p99 / p99.9 are *all* 1.0. Any
   high-percentile metric is therefore blind to over-shoot (both GT and a clamped
   prediction read 1.0). `p95_rel_error` (GT fg p95 = 0.946) is the highest percentile
   still below saturation; `hot_band_rel_error` averages the GT p90-p98 band.

**Therefore true lesion-SUVmax fidelity is not measurable in this pipeline at all** --
the preprocessing discards the top ~1% of intensities, which is exactly where lesions
live. That needs a non-clipping normalization, not a better metric. Any SUVmax claim must
wait for it.

---

## 2. Comparison table

Published references are private cohorts in calibrated SUV; "Ours" columns are
public ACRIN 6668 (NSCLC FDG) in normalized intensity -- fill after the WSL/GPU
eval run. See Caveats before reading any row as a head-to-head.

Numbers below were re-verified by a 2026-06-20 literature sweep (adversarial
3-vote verification, 16 primary sources). Method family: **C**=CNN/U-Net,
**G**=GAN/CycleGAN, **D**=diffusion. "AC-only" vs "joint" notes whether the paper
also denoises low-count data (changes the task and the metric scale).

| Method (family) | Cohort | Tracer / region | n (train/test) | SSIM | PSNR (dB) | NRMSE / NMSE | Organ SUV bias | Lesion SUV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CNN U-Net, image-domain joint AC+scatter (C) — Radiology:AI 2021 [1] | private | FDG, whole-body | (institutional) | 0.98 ± 0.01 | 36.3 ± 3.0 | NRMSE 0.21 ± 0.05 (SUV) | −0.8 ± 8.6% (organ)¹ | 0.9 ± 9.2%¹ |
| Dong et al. 2020, 3D patch CycleGAN (G) [4] | private (Emory/GE D690) | FDG, whole-body | 25 LOO + 30 hold-out | -- | 44.3 ± 3.5 (hold-out) / 43.1 ± 4.6 (LOO) | NMSE **0.72% ± 0.34%** (hold-out) | heart +2.1%, liver +2.9%, **lung −17.0% ± 12.0%**² | -- |
| Li (Wenbo) et al., Eur Radiol 2024, CycleGAN + site prior (G) [3] | private (122 subj) | FDG, total-body | n=122 | 0.980 ± 0.041 | 36.92 ± 5.49 | -- | -- | -- |
| Guo et al., Nat Commun 2022, semi-supervised 3D cGAN, domain decomposition (G) [5] | private | FDG, whole-body | -- | -- | -- | -- | -- | -- |
| cDDPM, conditional DDPM direct NAC→AC (D) — JNM 2023 **abstract** [6] | private | FDG, whole-body | -- | -- | 29.1 (vs cGAN 23.9 / CycleGAN 25.5) | NMSE 2.01 (vs 5.88 / 4.02) | -- | -- |
| MADM, 2.5D multi-view averaging diffusion, **joint low-count+AC** (D) — arXiv 2024 [2] | private (Yale, 147) | FDG, whole-body | 120 / 27 | 0.9944 ± 0.0020 | 38.92 ± 1.50 | RMSE 0.0299 ± 0.0088 | -- | lesion SUV err 0.079 ± 0.039 |
| **Ours - diff2d (2D latent diffusion, D)** — 2026-06-12, no CFG | **public ACRIN 6668 (+ Bladder/Normal for AE)** | FDG, **NSCLC (lung)** | 130 / 19 (paired) | 0.185 ± 0.034 | 14.30 ± 0.90 | NRMSE 0.196 ± 0.021 | −0.797 ± 0.071 (`rel_bias`) | 0.141 ± 0.173 (`max_rel_error`) |
| **Ours - diff2d + CFG (D)** — 2026-06-21, 150k, guidance **w=3.0** | same | FDG, **NSCLC (lung)** | 130 / 19 (paired) | **0.362 ± 0.052** | 14.54 ± 0.96 | NRMSE 0.192 ± 0.022 | −0.785 ± 0.047 (`rel_bias`) | 0.169 ± 0.148 (`max_rel_error`) |
| ~~Ours - diff2d "flow BRIDGE" (early 2026-06-26 attempts)~~ **INVALID — were epsilon** | expanded paired set | FDG, mostly lung | 188 / 27 | ~~0.31–0.35~~ | — | — | — | — |
| **Ours - diff2d flow BRIDGE (genuine, D)** — 2026-06-26, 100k, no-concat | **public, expanded paired set** (ACRIN+NSCLC_Radiogenomics+CPTAC+TCGA) | FDG, mostly lung | 188 / 27 (paired) | **0.886** | **25.6** | NRMSE **0.063** | **+0.003** (`rel_bias`) | — |
| Ours - diff2d flow + VGG perceptual (D) — 2026-06-26 ablation | same | FDG, mostly lung | 188 / 27 | 0.887 | 25.6 | 0.063 | +0.012 | 0.16 (`max_rel_error`) |
| **Ours - ft3d flow BRIDGE (3D, from 2D init, D)** — 2026-06-28, *val-only* | same | FDG, mostly lung | 188 / 27 | **0.89** (val) | — | — | ~0 (val) | voxel_r2 **+0.53** (val) — TEST eval pending re-run |
| **Ours - ft3d (3D latent diffusion, D)** | **public ACRIN 6668 (+ Bladder/Normal for AE)** | FDG, **NSCLC (lung)** | 130 / 19 (paired) | 0.012 ± 0.005 | -2.24 ± 0.33 | NRMSE 1.295 ± 0.048 | 3.77 ± 0.76 (`rel_bias`) | 20.2 ± 3.0 (`max_rel_error`) |
| **Ours - diff2d_flow_perc_p (2D flow on fixed AE, D)** — 2026-07-05, s8 | **public, 410-patient paired set** | FDG, mostly lung | 287 / **41** (paired) | 0.9017 ± 0.071 | 25.85 | NRMSE 0.0596 | −0.0112 (`rel_bias`) | 0.044 (`max_rel_error`, per-slice) |
| **Ours - ft3d_flow_p (3D flow on fixed AE, D) — CURRENT BEST** — 2026-07-11, s16 | same 410-patient set | FDG, mostly lung | 287 / **41** (paired) | **0.9227 ± 0.031** | 25.29 ± 1.39 | NRMSE **0.0551** | **+0.0009** (`rel_bias`) | voxel_r2 **+0.689**, slope 0.857 |
| **Ours - ft3d_flow_p, `--clamp_output` (D)** — 2026-08-06, s16, **same checkpoint** | same | FDG, mostly lung | 287 / **41** | **0.9231** | **25.45** | NRMSE **0.0542** | −0.0035 | voxel_r2 **+0.699**, `p95_rel_error` −0.031, `hot_band_rel_error` −0.100 |

> **2026-08-06 — the clamp is honest bookkeeping, not a quality win, and the real remaining
> deficiency is under-dispersion.** The `--clamp_output` row above is the *same* `ft3d_flow_p`
> checkpoint, re-evaluated with the decoded prediction projected into the valid `[0,1]` range.
> Verified by running both arms over the same 41 patients: the **unclamped arm reproduces the
> pre-change numbers exactly** on all nine pre-existing metrics (SSIM 0.9227, PSNR 25.2937,
> voxel_r2 0.6889, …), so the change is behaviour-preserving when disabled. Clamping then buys
> +0.16 dB PSNR, +0.010 voxel_r2, −0.0009 NRMSE, and drives `max_rel_error` 0.80 → 0.00.
>
> But it is **not free**: `reg_slope` drops 0.857 → 0.842 and `hot_band_rel_error` goes
> −0.084 → −0.100, because truncating the over-shooting top end removes dynamic range the
> model was (wrongly) getting credit for. What remains is the genuine, un-attacked problem:
> **the model under-estimates high-uptake tissue by ~10% and compresses dynamic range by
> ~15%** (`reg_slope` has sat at 0.80–0.86 in *every* good run, 2D and 3D, and no experiment
> has ever targeted it). `p95_rel_error` is unchanged by clamping (−0.0308 in both arms),
> which is what makes it the right metric to judge that on.
>
> **The ceiling diagnostic (`scripts/_diag_ceiling_3d.sh`) says where to spend effort.** It
> scores raw NAC, `decode(encode(NAC))` (passthrough), `decode(encode(AC))` (the AE ceiling —
> the best any latent model can reach, since the flow model only emits a latent this same
> frozen AE decodes), and the flow prediction. Full 41-patient TEST split, unclamped
> (the flow row reproduces `eval/ft3d_flow_p/test_s16.json` exactly, which validates the tool):
>
> | reference | ssim | psnr | voxel_r2 | reg_slope | max_rel_error | hot_band_rel_error |
> | --- | --- | --- | --- | --- | --- | --- |
> | raw NAC vs AC (floor) | 0.4315 | 16.66 | −0.286 | 0.363 | — | −0.243 |
> | `decode(encode(NAC))` passthrough | 0.4306 | 16.75 | −0.246 | 0.353 | 0.791 | −0.256 |
> | `decode(encode(AC))` **AE ceiling** | **0.9832** | **32.42** | **0.937** | **0.969** | 0.741 | **−0.016** |
> | flow prediction (`ft3d_flow_p`, s16) | 0.9227 | 25.29 | 0.689 | 0.857 | 0.802 | −0.084 |
>
> Per-metric verdict: SSIM / PSNR / NRMSE / MAE / `voxel_r2` / `reg_slope` /
> `hot_band_rel_error` are all **flow-limited** — real headroom, worth model work (**7.1 dB**
> of PSNR; voxel_r2 0.689 vs 0.937; hot-band under-estimation −8.4% where the AE is only
> −1.6%). `max_rel_error` is **AE-limited**: the AE round-tripping the *ground-truth* AC
> already scores 0.741 vs the flow model's 0.802, so no model change can move it.
>
> Also note the passthrough is barely distinguishable from raw NAC (SSIM 0.4315 → 0.4306),
> so the flow model's 0.9227 is genuinely earned, not an artifact of the AE round trip.
>
> **✅ BREAKTHROUGH (2026-06-26): the genuine flow bridge works.** Once actually run as flow
> (no-concat I2SB-style, after fixing the config bug below), held-out TEST = **SSIM 0.886,
> PSNR 25.6, NRMSE 0.063, rel_bias +0.003, voxel_r2 +0.47, slope 0.82** (27 patients, stable
> across 8/16/32 sampling steps — no drift). That **~doubles NAC-passthrough (0.448)**, beats
> epsilon/CFG (0.362) on every metric, and is the **first model with positive voxel_r2 and
> ~zero SUV-proxy bias** — i.e. genuine quantitative NAC→AC translation, not just structure.
> val (~0.91) and TEST (0.886) now agree, so the honest monitor is trustworthy. Still below the
> literature's ~0.98 SSIM, but those are private SUV-calibrated whole-body cohorts (see Caveats);
> in normalized intensity on the public benchmark this is a strong, reproducible result.
>
> **⚠️ Earlier "flow-bridge" rows (0.31–0.35) were RETRACTED — they were never flow.** A config-load
> bug (PyYAML was not installed in the WSL venv → `_load_config` silently fell back to the
> epsilon `default_config`) meant **every** run ignored its YAML and trained as **epsilon**
> with the small default model. Proof: the saved checkpoints embed `prediction_type=epsilon`,
> `in_channels=16`, `num_channels=[16,32,64]` — the default, not the requested flow config.
> So the earlier "flow" numbers (0.31 / 0.35) were epsilon models, and the "all methods cluster
> at ~0.3 → deep shared bottleneck" inference was an artifact (they were the *same* epsilon
> model). **Fixes:** `pip install pyyaml`; `_load_config` now **raises** instead of silently
> falling back when an explicit `--config` can't be read; MONAI `MetaTensor` is stripped to a
> plain tensor in the flow path (it tripped `torch.compile`). The **first genuine flow-bridge
> run** (real `prediction_type=flow`, `in_channels=8`, `num_channels=[64,128,256]`) is training
> now → `outputs/diff2d_flow2`; TEST verdict pending. The one still-valid carry-over finding:
> the *in-training* val monitor was inflated by full-depth/3-plane sampling — now fixed to
> axial + per-slice. See memory `diffusion-generation-collapse`.

> **CFG result (2026-06-21, 150k diff2d_cfg, held-out TEST, guidance sweep w∈{1,1.5,2,3}):**
> SSIM rises monotonically with guidance — 0.188 → 0.209 → 0.267 → **0.362** (w=1→3) —
> beating the no-CFG baseline (0.185). **But the decisive metrics stay broken at every
> guidance level:** `voxel_r2` −1.9 to −2.2 (worse than predicting the mean), `reg_slope`
> 0.01–0.05 (≈no correlation with true uptake), `rel_bias` ≈ −0.8 (foreground ~80% too
> low), PSNR/NRMSE flat (~14.5 / ~0.19). Even the best SSIM (0.362) is **below
> NAC-passthrough (0.48)** and far from the AE ceiling (0.86). **Conclusion: CFG amplifies
> the structural signal but cannot make the model quantitatively translate NAC→AC** — it is
> a partial, structure-only win, not a fix. This is the empirical motivation for the
> NAC→AC rectified-flow bridge (start the trajectory *from* NAC; see §6 / `diffusion-generation-collapse`).

¹ Organ/lesion SUV-bias for the CNN row is from prior verification of the
whole-body FDG CNN result; the 2026-06-20 sweep re-confirmed that paper's
SSIM/PSNR/NRMSE but did not re-extract its per-organ SUV table — treat the SUV
cells as secondary.
² **Most relevant failure mode for us:** Dong's −17% lung bias was caused by
lung-cancer-patient imbalance (8/25 train vs 10/10 test). ACRIN 6668 is an
**all-NSCLC (lung)** cohort, so a single-disease distribution is a known,
documented risk for region-specific SUV bias — see §3 caveat 5.

> **The "Ours" rows are a NEGATIVE/baseline result, not a SOTA claim** (honest
> held-out TEST numbers, 2026-06-12 full run, clamped DDIM, EMA weights). They are
> far below the references because the diffusion models do **not** yet perform the
> NAC→AC translation (see §5), AND because — critically — **none of the reference
> numbers are head-to-head comparable** to ours (different cohorts, body regions,
> SUV-vs-normalized intensity, and AC-only-vs-joint tasks). See §6 for how to
> compare honestly.

Reference anchors: the **CNN U-Net (Radiology:AI 2021)** is the image-domain
reference; **MADM** is the closest diffusion competitor (but solves a *joint*
denoise+AC task, not AC-only); **Dong 2020** is the most directly relevant
caution because it reports per-organ SUV bias on a partly-lung-cancer cohort.
PSNR is **not comparable across these rows** (~29 dB cDDPM to ~44 dB Dong)
because each uses a different intensity normalization/range.

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
5. **Single-disease cohort → region-specific SUV-bias risk.** ACRIN 6668 is
   *entirely* NSCLC (lung) FDG. Dong et al. 2020 documented a **−17% lung SUV
   bias** that they attribute directly to lung-cancer-patient imbalance between
   train and test. With an all-lung cohort, any anatomical/uptake region that is
   under-represented (or any heterogeneity inside the lung) can produce large,
   organ-specific SUV bias that whole-image SSIM/PSNR will not reveal. Report SUV
   bias *per region*, not just globally.
6. **AC-only vs joint (denoise+AC) task mismatch.** MADM (the closest diffusion
   competitor) solves a *joint* low-count-denoising **and** CT-free AC task on full-
   count Yale data; its very high SSIM (0.9944) and small RMSE reflect that easier
   denoising-dominated regime and a different metric scale. cDDPM is AC-only but is
   a **conference abstract** (preliminary, not peer-reviewed). Match the task
   (AC-only vs joint) before reading any row as comparable.

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

> **Literature corroboration for the ft3d struggle (2026-06-20 sweep).** MADM's own
> ablation shows a **plain 3D diffusion baseline (3D-DDPM, PSNR 37.2) is *beaten* by
> a cGAN (38.3) and a U-Net (38.1)** — i.e. naive end-to-end 3D diffusion is *not*
> automatically superior for 3D PET, which is exactly why MADM uses a **CNN-generated
> prior refined by a 2.5D multi-view (axial/coronal/sagittal) averaging** scheme
> instead of full 3D diffusion. Our collapsing `ft3d` is consistent with that
> finding: a 2.5D multi-view or CNN-prior-warm-started design is a literature-backed
> alternative to fighting full-3D latent diffusion.

---

## 6. How to compare to others (honest positioning) + next steps

**The headline positioning, established by the 2026-06-20 sweep:** *no* published
SOTA NAC→AC method validates on a public paired NAC+AC FDG benchmark — Shiri/CNN,
Dong, Li (Eur Radiol), Guo, cDDPM and MADM **all use private institutional
cohorts** (Yale, Emory/GE Discovery 690, Chinese total-body). So:

1. **Do not make a numeric head-to-head claim.** Absolute SSIM/PSNR/NRMSE are not
   comparable across these papers (different cohorts, body region, SUV-vs-normalized
   intensity, AC-only-vs-joint task). A table cell like "ours 0.18 SSIM vs their
   0.98" is meaningless, not a loss.
2. **Position on the reproducibility gap, not the leaderboard.** The defensible,
   genuinely novel angle is: *a NAC→AC (latent-)diffusion model evaluated on the
   public, reproducible ACRIN 6668 benchmark* — something the cited private-cohort
   papers cannot offer. Frame the contribution as a public, reproducible baseline +
   protocol, then report **within-pipeline deltas** (CFG vs no-CFG, guidance sweep,
   ablations) as the real evidence.
3. **Report the decisive metrics, or it isn't a SOTA conversation.** The field
   treats **organ-level SUV %-bias** and **lesion SUVmax/mean error** as decisive;
   image SSIM/PSNR is necessary-but-not-sufficient (and saturates ~0.98). Our
   `rel_bias`/`max_rel_error`/`voxel_r2`/Bland-Altman are *proxies in normalized
   intensity*. To claim parity on the metrics that matter we need **SUV calibration
   (Bq/mL via injected dose + body weight + decay) and organ/lesion VOIs** — not yet
   tracked. Until then, present clinical-tier numbers explicitly as proxies.
4. **Watch the documented failure modes.** Direct-correction networks are reported
   to (a) blur and lose small low-uptake lesions and (b) **hallucinate pseudo-uptake
   / false positives** in lung/heart/bowel where boundaries are unclear (Radiology:AI
   2021). For a single-disease lung cohort this is the exact risk to test for —
   inspect lesion regions for both *missed* and *invented* uptake, don't rely on
   global metrics.

**Next steps for the comparison (knowledge/eval only — no model changes implied):**
- Adopt and publish a fixed ACRIN 6668 protocol (which studies are paired NAC+AC,
  the by-patient split + seed, the normalization) so the numbers are citable as a
  reproducible public baseline (open question from the sweep).
- Add **SUV calibration + organ/lesion VOIs** so `rel_bias`/`max_rel_error` become
  true SUV %-bias and lesion SUVmax error — the metrics the field decides on.
- When reporting, always state **task (AC-only vs joint denoise+AC)**, body region,
  and intensity domain next to every borrowed number.
- Read MADM (arXiv 2406.08374) in full as the methodological template for the 3D
  stage (CNN prior + 2.5D multi-view averaging).

---

## 4. References

Citations re-verified 2026-06-20 (adversarial 3-vote, 16 primary sources). Bracket
numbers match the §2 table.

- TCIA ACRIN-NSCLC-FDG-PET (ACRIN 6668), the public paired benchmark:
  https://www.cancerimagingarchive.net/collection/acrin-nsclc-fdg-pet/
- **[1]** CNN U-Net, image-domain joint attenuation+scatter correction,
  Radiology: Artificial Intelligence 2021, doi:10.1148/ryai.2020200137:
  https://pubs.rsna.org/doi/full/10.1148/ryai.2020200137 (PMC mirror PMC8043359).
  *Documents the blurring / hallucinated-pseudo-uptake failure modes.*
- **[2]** MADM — 2.5D multi-view averaging diffusion, **joint low-count denoising +
  CT-free AC**, private Yale 147-subj FDG (120/27), arXiv 2406.08374:
  https://arxiv.org/abs/2406.08374 . *Its 3D-DDPM baseline (37.2 dB) is beaten by
  cGAN (38.3) and U-Net (38.1) → plain 3D diffusion is not automatically superior.*
- **[3]** Li (Wenbo) / Huang / Chen et al., Eur Radiol 2024, CycleGAN + site-structure
  prior, total-body (n=122), PubMed 38355987 / doi:10.1007/s00330-024-10647-1:
  https://doi.org/10.1007/s00330-024-10647-1 . *(Lead author Wenbo Li, not "Wang".)*
- **[4]** Dong et al., Phys Med Biol **2020** (online Dec 2019; often cited "2019"),
  3D patch CycleGAN, private Emory/GE Discovery 690, PMID 31869826 / PMC7099429:
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7099429/ . *Reports per-organ SUV bias
  incl. −17% lung bias from lung-cancer-patient imbalance — most relevant caution for
  an all-NSCLC cohort.*
- **[5]** Guo et al., Nature Communications 2022, semi-supervised 3D conditional GAN
  with domain-decomposition (learns the low-frequency anatomy-dependent correction),
  doi:10.1038/s41467-022-33562-9: https://www.nature.com/articles/s41467-022-33562-9
- **[6]** cDDPM (Hu et al.), conditional DDPM direct NAC→AC, **JNM 2023 conference
  abstract (preliminary, not peer-reviewed)**, 64(suppl 1):P386:
  https://jnm.snmjournals.org/content/64/supplement_1/P386
