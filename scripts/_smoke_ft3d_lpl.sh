#!/usr/bin/env bash
# Wiring smoke for the LPL perceptual arm: a handful of real 3D flow steps against the
# frozen ae3d_p AE. Checks (a) the "Perceptual loss enabled ... space=latent" line fires,
# (b) the logged `perceptual` metric is non-zero, (c) its magnitude vs `mse` so
# perceptual_weight can be calibrated before committing to a long run.
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=/root/petct/.venv/bin/python
CFG=${CFG:-src/training/configs/ft3d_flow_lpl.yaml}
SAVE=${SAVE:-outputs/_smoke_ft3d_lpl}
STEPS=${STEPS:-12}
# The SSD PET cache: without it the NAC/AC pairing scan re-reads DICOM headers for every
# ACRIN study (~30-50 min before the first training step).
CACHE=${CACHE:-/mnt/c/DeepTrainingData/PetCT}
ROOTS=(
  "/mnt/d/DeepTrainingData/Project/ACRIN 6668"
  "/mnt/d/DeepTrainingData/Project/NSCLC_Radiogenomics"
)
rm -rf "$SAVE"; mkdir -p "$SAVE"
"$PY" -m src.training.train.train_ft3d \
  --config "$CFG" --save_dir "$SAVE" \
  --data_dir "${ROOTS[@]}" --cache_dir "$CACHE" --prefetch 2 \
  --ae_ckpt outputs/ae3d_p/best.pt \
  --epochs 1 --steps_per_epoch "$STEPS" --val_every 0 --latent_size 64 \
  --device cuda 2>&1 | grep -viE "futurewarn|deprecat|^  @torch"
