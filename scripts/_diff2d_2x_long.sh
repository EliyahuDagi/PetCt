#!/usr/bin/env bash
# Longer stage 3: diff2d_2x at the reference chain's 100k budget.
#
# The first attempt was budgeted 30k and stopped at ~22.5k (best val l1 0.01814). At MATCHED
# steps it was already AHEAD of the reference diff2d_flow_perc_p (0.0241 vs 0.0284 at 20k),
# so the only thing holding it back was the budget -- and the reference kept improving to
# 0.01072 by 100k. Since this checkpoint is the inflation source for the 3D flow stage, an
# undertrained one weakens a 22-28 h run downstream.
#
# FRESH run, not --resume: lr_total_steps was baked at 30k in the first attempt, so resuming
# would train at the cosine floor (~zero LR) and gain almost nothing. A new run sizes the
# schedule to 100k.
#
# Warm-started from the partial run (identical architecture -> full tensor transfer, already
# adapted to the 16-channel latent), not from the old 8-channel model, so none of the 22.5k
# steps already spent are thrown away.
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=/root/petct/.venv/bin/python
CACHE=/mnt/c/DeepTrainingData/PetCT
STEPS=${STEPS:-100000}
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
"$PY" -m src.training.train.train_diff2d \
  --config src/training/configs/diff2d_2x.yaml --save_dir outputs/diff2d_2x_long \
  --data_dir "${ROOTS[@]}" --cache_dir "$CACHE" --prefetch 4 \
  --ae_ckpt outputs/ae2d_2x/best.pt \
  --init_from outputs/diff2d_2x/best.pt \
  --epochs $((STEPS / 500)) --steps_per_epoch 500 --val_every 2500 --val_batches 8 \
  --batch_size 16 --latent_size 128 --device cuda 2>&1 | tail -10
echo "================= diff2d_2x_long done"
