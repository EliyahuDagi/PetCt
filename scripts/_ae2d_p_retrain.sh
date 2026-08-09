#!/usr/bin/env bash
# STAGE 1 of the perceptual-loss-fix retrain. Retrain the 2D AutoencoderKL on the
# FULL pooled PET dataset with the FIXED VGG perceptual loss (shared/intensity-aware
# normalization + LPIPS-style feature normalization). Writes to a FRESH dir
# (outputs/ae2d_p) so the old AE (outputs/ae2d/best.pt) is preserved for comparison.
#
# Budget mirrors the production ae2d run (100k steps, batch 16, slice 128) so the
# recon metrics are apples-to-apples vs outputs/ae2d. Run _ae2d_precache.sh ONCE first.
# After this finishes, compare with:  python scripts/_ae_p_compare.py
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=~/petct/.venv/bin/python
CACHE=/mnt/c/DeepTrainingData/PetCT
SAVE=outputs/ae2d_p

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

mkdir -p "$SAVE"
echo ">>> STAGE_START ae2d_p_train $(date '+%H:%M:%S')"
"$PY" -m src.training.train.train_ae2d \
  --config src/training/configs/ae2d.yaml \
  --data_dir "${ROOTS[@]}" --cache_dir "$CACHE" --prefetch 2 --loader_workers 8 --patient_reuse 32 \
  --epochs 200 --steps_per_epoch 500 --val_every 2500 --val_batches 8 \
  --val_fraction 0.2 --test_fraction 0.1 \
  --batch_size 16 --slice_size 128 --modality pet \
  --device cuda --save_dir "$SAVE" \
  > "$SAVE/train.log" 2>&1
rc=$?; echo ">>> STAGE_END ae2d_p_train rc=$rc $(date '+%H:%M:%S')"
if [ $rc -ne 0 ]; then echo ">>> ABORT"; exit $rc; fi

echo ">>> COMPARE vs outputs/ae2d $(date '+%H:%M:%S')"
"$PY" scripts/_ae_p_compare.py
echo ">>> DONE $(date '+%H:%M:%S')"
