#!/usr/bin/env bash
# FULL 2x CHAIN: ae2d -> ae3d -> flow-2d -> flow-3d, warm-started, reduced budgets.
#
# Configuration is what the measurements support (see configs/ae2d_2x.yaml for the numbers):
#   * 3D at 128^3 -- resizing to 64^3 destroys 6.1% of hot-tissue signal before any model
#     runs; at 128^3 that drops to 2.4%. Measured against NATIVE truth the 128^3 AE cut p95
#     error 0.057 -> 0.025 and hot-band 0.068 -> 0.047.
#   * latent_channels 16 -- the 128^3 AE gave up 7.48 dB to compression (vs 1.74 dB at 64^3),
#     so the latent is now the bottleneck. Doubling channels (not the 16^3 spatial grid)
#     keeps the flow UNet affordable.
#   * ae3d uses --extra_levels 1 so 128^3 still lands a 16^3 latent grid.
#
# WARM STARTS (--init_from = weights only, strict=False; the 8->16 latent convs are
# re-initialised, all other blocks transfer) -- this is why the budgets are ~1/3 of the
# originals. ft3d additionally inflates the freshly trained 2D flow UNet (flow->flow, so the
# in_channels=C input conv transfers; inflating across prediction types silently drops it).
#
# SPLIT: every stage reuses outputs/ft3d_flow_p/split.json. Without it each run derives its
# OWN partition -- same seed, different patients -- which already leaked 17/41 test patients
# between two runs in this project.
#
# Stages are sequential (one GPU) and each aborts the chain on failure, so a broken stage
# never silently poisons the next one's warm start.
set -euo pipefail
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=/root/petct/.venv/bin/python
CACHE=/mnt/c/DeepTrainingData/PetCT
SPLIT=outputs/ft3d_flow_p/split.json

# Reduced budgets (originals: 100k / 50k / 100k / 30k).
AE2D_STEPS=${AE2D_STEPS:-30000}
AE3D_STEPS=${AE3D_STEPS:-25000}
D2D_STEPS=${D2D_STEPS:-30000}
FT3D_STEPS=${FT3D_STEPS:-15000}

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
step () { echo; echo "################ $* ($(date +%H:%M))"; }

# ---------- 1/4  ae2d @128^2, latent 16, warm-started from ae2d_p ----------
step "1/4 ae2d_2x  (${AE2D_STEPS} steps, warm start from ae2d_p)"
"$PY" -m src.training.train.train_ae2d \
  --config src/training/configs/ae2d_2x.yaml --save_dir outputs/ae2d_2x \
  --data_dir "${ROOTS[@]}" --cache_dir "$CACHE" --prefetch 4 \
  --init_from outputs/ae2d_p/best.pt \
  --epochs $((AE2D_STEPS / 500)) --steps_per_epoch 500 --val_every 2500 --val_batches 8 \
  --batch_size 16 --slice_size 128 --modality pet --device cuda 2>&1 | tail -8

# ---------- 2/4  ae3d @128^3, inherits latent 16 from ae2d_2x, +1 level -> 16^3 grid ------
step "2/4 ae3d_2x2 (${AE3D_STEPS} steps, inflate from ae2d_2x, crop 128, extra_levels 1)"
"$PY" -m src.training.train.train_ae3d \
  --save_dir outputs/ae3d_2x2 \
  --data_dir "${ROOTS[@]}" --cache_dir "$CACHE" --prefetch 4 --cache_size 24 \
  --ae2d_ckpt outputs/ae2d_2x/best.pt --extra_levels 1 \
  --epochs $((AE3D_STEPS / 500)) --steps_per_epoch 500 --val_every 2500 --val_batches 4 \
  --batch_size 1 --crop_size 128 --modality pet --device cuda 2>&1 | tail -8

# ---------- 3/4  flow 2D on the new 2D AE ----------
step "3/4 diff2d_2x (${D2D_STEPS} steps, warm start from diff2d_flow_perc_p)"
"$PY" -m src.training.train.train_diff2d \
  --config src/training/configs/diff2d_2x.yaml --save_dir outputs/diff2d_2x \
  --data_dir "${ROOTS[@]}" --cache_dir "$CACHE" --prefetch 4 \
  --ae_ckpt outputs/ae2d_2x/best.pt \
  --init_from outputs/diff2d_flow_perc_p/best.pt \
  --epochs $((D2D_STEPS / 500)) --steps_per_epoch 500 --val_every 2500 --val_batches 8 \
  --batch_size 16 --latent_size 128 --device cuda 2>&1 | tail -8

# ---------- 4/4  flow 3D on the new 3D AE, inflated from the new 2D flow ----------
# Inflates from diff2d_2x_LONG (100k budget), not the original 30k-budgeted diff2d_2x: the
# short run was still improving when it stopped, and an undertrained inflation source would
# handicap this 22-28 h stage from step 0.
step "4/4 ft3d_2x   (${FT3D_STEPS} steps, inflate from diff2d_2x, 128^3)"
"$PY" -m src.training.train.train_ft3d \
  --config src/training/configs/ft3d_2x.yaml --save_dir outputs/ft3d_2x \
  --data_dir "${ROOTS[@]}" --cache_dir "$CACHE" --prefetch 6 --cache_size 24 \
  --ae_ckpt outputs/ae3d_2x2/best.pt --inflate_from outputs/diff2d_2x_long/best.pt \
  --split_json "$SPLIT" \
  --epochs $((FT3D_STEPS / 500)) --steps_per_epoch 500 --val_every 1000 --val_batches 4 \
  --latent_size 128 --device cuda 2>&1 | tail -12

step "chain complete -- evaluating ft3d_2x on the shared test split"
for S in 16 32; do
  "$PY" -m src.training.evaluate --task ft3d \
    --data_dir "${ROOTS[@]}" \
    --ae_ckpt outputs/ae3d_2x2/best.pt --diff_ckpt outputs/ft3d_2x/best.pt \
    --split_json "$SPLIT" --mode test \
    --size 128 --ddim_steps "$S" --spacing linear --max_patients 0 --device cuda \
    --clamp_output --out "outputs/eval/ft3d_2x/test_s${S}.json" 2>&1 | tail -2
done
echo "################ 2x CHAIN DONE"
