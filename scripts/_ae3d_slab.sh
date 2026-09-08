#!/usr/bin/env bash
# In-plane-only ("anisotropic") 3D autoencoder in SLAB mode: the same problem the 2D
# autoencoder solves (128x128 slices -> 32x32x8 latents, 2:1 in-plane compression) but
# the encoder/decoder see the neighbouring slices. Depth is never compressed.
#
#   2D autoencoder (outputs/ae2d_p/best.pt, EMA weights) centre-inflated into a 3D
#   AutoencoderKL with depth stride 1 everywhere -> starts as an EXACT slice-wise copy,
#   then learns to mix depth on random 128x128x16 slabs at native slice spacing.
#
# Patient split: --split_json outputs/ae2d_p/split.json, i.e. EXACTLY the partition the 2D
# autoencoder was trained on, so the only difference between the two autoencoders is 2D vs
# 3D. (Caveat, equal for both: that partition put 26 of the flow chain's 41 test patients in
# TRAIN and 9 in VAL, so both frozen autoencoders have seen 35/41 flow test patients at the
# reconstruction stage. Measured 2026-09-06.)
# Usage (inside WSL):  EPOCHS=20 bash scripts/_ae3d_slab.sh      (RESUME=1 to continue)
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=~/petct/.venv/bin/python
CACHE=${CACHE:-/root/petct_cache}
EPOCHS=${EPOCHS:-20}
SAVE=${SAVE:-outputs/ae3d_slab}
SPLIT=${SPLIT:-outputs/ae2d_p/split.json}
RESUME=${RESUME:-}

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

for f in outputs/ae2d_p/best.pt "$SPLIT" src/training/configs/ae3d_slab.yaml ${RESUME:+$SAVE/last.pt}; do
  if [ ! -f "$f" ]; then echo ">>> MISSING $f -- abort"; exit 2; fi
done

mkdir -p "$SAVE"
if [ -z "$RESUME" ]; then : > "$SAVE/train.log"; fi
echo ">>> STAGE_START ae3d_slab_train${RESUME:+ (resume from $SAVE/last.pt)} $(date '+%Y-%m-%d %H:%M:%S')"
"$PY" -m src.training.train.launcher ae3d \
  --config src/training/configs/ae3d_slab.yaml \
  --data_dir "${ROOTS[@]}" --cache_dir "$CACHE" --device cuda \
  --val_fraction 0.2 --test_fraction 0.1 \
  --split_json "$SPLIT" \
  --epochs "$EPOCHS" --steps_per_epoch 1000 --val_every 2500 --val_batches 8 \
  --ae2d_ckpt outputs/ae2d_p/best.pt \
  --save_dir "$SAVE" --cache_size 128 ${RESUME:+--resume} \
  >> "$SAVE/train.log" 2>&1
rc=$?; echo ">>> STAGE_END ae3d_slab_train rc=$rc $(date '+%Y-%m-%d %H:%M:%S')"
