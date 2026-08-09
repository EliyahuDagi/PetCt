#!/usr/bin/env bash
# ae2d embedding-size ablation: train the 2D AutoencoderKL on the FULL pooled PET
# dataset (all 10 roots, NAC+AC, unpaired fine) at several latent_channels values
# and compare encode->decode reconstruction fidelity. The val rows already written
# to each run's metrics.jsonl contain recon_l1/psnr/ssim/nrmse/mae, so no separate
# eval pass is needed -- scripts/_ae2d_ablation_report.py just reads them back.
#
# Run scripts/_ae2d_precache.sh ONCE first so each run reads cached PET in ~ms.
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=~/petct/.venv/bin/python
CACHE=/mnt/c/DeepTrainingData/PetCT

# --- Tunables (kept identical across every ablation point) -------------------
STEPS=4000        # training steps per latent_channels value
BATCH=32          # slice batch size
VAL_EVERY=250     # validate every N steps (-> several val rows in metrics.jsonl)
VAL_BATCHES=8     # batches per validation pass
SLICE=128         # slice_size; report assumes 128 + default /4 -> 32x32 latent

# Embedding-size grid: MUST match LATENT_CHANNELS_GRID in
# scripts/gen_ae2d_ablation_configs.py (which generates the per-N configs below).
# Single-point check for now; add more values (e.g. 2 4 16) to sweep.
LATENT_CHANNELS=(8)

ROOTS=(
  "/mnt/d/DeepTrainingData/Project/ACRIN 6668"
  "/mnt/d/DeepTrainingData/Project/PetCt/Bladder 13.11.25"
  "/mnt/d/DeepTrainingData/Project/PetCtNormal/PET_CT_NORMAL_2"
  "/mnt/d/DeepTrainingData/Project/NSCLC_Radiogenomics"
  "/mnt/d/DeepTrainingData/Project/TCGA-LUAD"
  "/mnt/d/DeepTrainingData/Project/CPTAC-LUAD"
  "/mnt/d/DeepTrainingData/Project/CPTAC-LSCC"
  "/mnt/d/DeepTrainingData/Project/CPTAC-UCEC"
  "/mnt/d/DeepTrainingData/Project/CPTAC-PDA"
  "/mnt/d/DeepTrainingData/Project/TCGA-THCA"
)

# Regenerate the per-N config YAMLs (idempotent) before training.
echo ">>> GEN_CONFIGS $(date '+%H:%M:%S')"
"$PY" scripts/gen_ae2d_ablation_configs.py
echo ">>> GEN_CONFIGS_DONE rc=$? $(date '+%H:%M:%S')"

echo ">>> ABLATION_START $(date '+%Y-%m-%d %H:%M:%S')"
for N in "${LATENT_CHANNELS[@]}"; do
  CFG="src/training/configs/ablation/ae2d_lc${N}.yaml"
  SAVE="outputs/ae2d_ablation/lc${N}"   # mirrors output_dir in the generated config
  mkdir -p "$SAVE"
  echo ">>> RUN lc=${N} start $(date '+%H:%M:%S')"
  "$PY" -m src.training.train.train_ae2d \
    --config "$CFG" \
    --data_dir "${ROOTS[@]}" \
    --epochs 1 --steps_per_epoch "$STEPS" \
    --val_every "$VAL_EVERY" --val_batches "$VAL_BATCHES" \
    --val_fraction 0.2 --test_fraction 0.1 \
    --batch_size "$BATCH" --slice_size "$SLICE" \
    --modality pet --cache_dir "$CACHE" --prefetch 2 \
    --device cuda --save_dir "$SAVE" \
    > "$SAVE/train.log" 2>&1
  rc=$?
  echo ">>> RUN lc=${N} done rc=$rc $(date '+%H:%M:%S')"
done
echo ">>> ABLATION_DONE $(date '+%Y-%m-%d %H:%M:%S')"

# Tabulate reconstruction fidelity vs embedding size.
echo ">>> REPORT $(date '+%H:%M:%S')"
"$PY" scripts/_ae2d_ablation_report.py
echo ">>> REPORT_DONE rc=$? $(date '+%H:%M:%S')"
echo ">>> DONE $(date '+%Y-%m-%d %H:%M:%S')"
