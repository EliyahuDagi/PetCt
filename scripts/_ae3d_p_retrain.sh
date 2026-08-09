#!/usr/bin/env bash
# STAGE 2 (run only if STAGE 1 / _ae2d_p_retrain.sh improved the AE metrics):
# inflate the FIXED 2D AE (outputs/ae2d_p/best.pt) into a 3D AutoencoderKL and
# fine-tune it on volumes with the fixed perceptual loss. Fresh dir outputs/ae3d_p
# so the old 3D AE (outputs/ae3d/best.pt) is preserved.
#
# Downstream: point diff2d/ft3d (flow or epsilon) at outputs/ae2d_p/best.pt /
# outputs/ae3d_p/best.pt and retrain those on the new latent space.
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=~/petct/.venv/bin/python
CACHE=/mnt/c/DeepTrainingData/PetCT
SAVE=outputs/ae3d_p

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
echo ">>> STAGE_START ae3d_p_train $(date '+%H:%M:%S')"
"$PY" -m src.training.train.train_ae3d \
  --config src/training/configs/ae3d.yaml \
  --data_dir "${ROOTS[@]}" --cache_dir "$CACHE" --prefetch 2 \
  --epochs 100 --steps_per_epoch 500 --val_every 2500 --val_batches 4 \
  --val_fraction 0.2 --test_fraction 0.1 \
  --batch_size 1 --crop_size 64 \
  --ae2d_ckpt outputs/ae2d_p/best.pt \
  --device cuda --save_dir "$SAVE" \
  > "$SAVE/train.log" 2>&1
rc=$?; echo ">>> STAGE_END ae3d_p_train rc=$rc $(date '+%H:%M:%S')"
if [ $rc -ne 0 ]; then echo ">>> ABORT"; exit $rc; fi

echo ">>> COMPARE vs outputs/ae3d $(date '+%H:%M:%S')"
"$PY" scripts/_ae_p_compare.py --old outputs/ae3d/metrics.jsonl --new outputs/ae3d_p/metrics.jsonl --tag ae3d
echo ">>> DONE $(date '+%H:%M:%S')"
