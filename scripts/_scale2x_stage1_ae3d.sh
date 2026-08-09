#!/usr/bin/env bash
# STAGE 1 of the 2x-resolution chain: a 128^3 3D AE ("B2 config").
#
# WHY (measured, scripts/_diag_resolution_loss.py, 8 held-out patients):
#   native -> 64^3  -> native : PSNR 32.52  hot_band -6.1%
#   native -> 128^3 -> native : PSNR 37.79  hot_band -2.4%
# The 64^3 resize alone costs as much as the entire autoencoder (AE ceiling 32.4 dB), and
# it is INVISIBLE to every metric reported so far because they all compare against the
# already-resized 64^3 ground truth. Moving to 128^3 adds ~5.3 dB of headroom and halves the
# resize's hot-tissue damage -- damage no loss function can recover.
#
# KEY DESIGN (--extra_levels 1): one extra downsample level means 128^3 -> 16^3 latent, the
# SAME latent grid as today, so the downstream flow UNet's cost is UNCHANGED. Without it a
# 128^3 crop would give a 32^3 latent: 8x the diffusion activations and O(N^2) attention over
# 32768 tokens, which is not trainable on one 24 GB GPU.
#
# NOT retraining ae2d at 256^2: ae2d_p was trained at 128^2, which exactly matches the
# in-plane resolution of a 128^3 volume -- so it is a BETTER inflation source here than it is
# for today's 64^3 AE (whose 64^2 in-plane silently mismatches it). That saves ~1-2 days and
# removes an existing mismatch. If in-plane detail later proves limiting, ae2d@256^2 +
# diff2d@256^2 is a separate stage.
#
# Runs a short OOM smoke first: 128^3 with 4 levels is the memory-risky part, and finding out
# 20 hours in is not acceptable.
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=/root/petct/.venv/bin/python
CACHE=/mnt/c/DeepTrainingData/PetCT
SAVE=${SAVE:-outputs/ae3d_2x}
CROP=${CROP:-128}
EPOCHS=${EPOCHS:-100}
STEPS=${STEPS:-500}
AE2D=${AE2D:-outputs/ae2d_p/best.pt}

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
[ -f "$AE2D" ] || { echo "MISSING $AE2D" >&2; exit 1; }

run_ae3d () {  # $1=save_dir  $2=epochs  $3=steps_per_epoch  $4=val_every
  "$PY" -m src.training.train.train_ae3d \
    --data_dir "${ROOTS[@]}" --cache_dir "$CACHE" --prefetch 4 --cache_size 24 \
    --save_dir "$1" --epochs "$2" --steps_per_epoch "$3" \
    --val_every "$4" --val_batches 4 \
    --batch_size 1 --crop_size "$CROP" --extra_levels 1 \
    --ae2d_ckpt "$AE2D" --modality pet --device cuda
}

echo "=== OOM smoke: ${CROP}^3, extra_levels 1, batch 1 ==="
if ! run_ae3d "outputs/_smoke_ae3d_2x" 1 20 0 2>&1 | tail -12; then
  echo "SMOKE FAILED -- not starting the long run. Likely OOM at ${CROP}^3." >&2
  exit 1
fi
if grep -qiE "out of memory|OutOfMemory" outputs/_smoke_ae3d_2x/train.log 2>/dev/null; then
  echo "SMOKE HIT OOM at ${CROP}^3 -- reduce crop or add another level before the long run." >&2
  exit 1
fi
echo "=== smoke OK -> full stage-1 run into ${SAVE} ==="
run_ae3d "$SAVE" "$EPOCHS" "$STEPS" 2500 2>&1 | tail -20
echo "=== stage 1 done: ${SAVE} ==="
