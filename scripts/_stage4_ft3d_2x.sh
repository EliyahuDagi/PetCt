#!/usr/bin/env bash
# STAGE 4 ONLY: the 3D flow bridge at 128^3 (the last piece of the 2x chain).
#
# Stages 1-3 are complete:
#   ae2d_2x         128^2, latent 16                -> 44.51 dB (beat ae2d_p by +2.3 dB)
#   ae3d_2x2        128^3, latent 16^3x16, +1 level -> 30.58 dB (+0.68 over the 8ch version)
#   diff2d_2x_long  100k steps                      -> val l1 0.01154, ssim 0.9526, on par
#                   with the reference chain's 2D flow (the 30k run it replaces was
#                   undertrained at 0.01814 / 0.8455)
#
# Inflates from diff2d_2x_long (flow->flow, so the in_channels=C input conv transfers;
# inflating across prediction types would silently drop it and destroy the warm start).
#
# Budget 15k: every long run here -- the 128^3 AE, the slope sweep, stage 3 itself -- has
# plateaued well before its budget, so a larger number is unlikely to pay.
#
# batch 4 x accum 4 was PRE-FLIGHT MEASURED at 128^3 with the frozen AE resident:
# 1x16 = 7.50 s/step / 3.3 GiB, 2x8 = 5.77 s / 5.3 GiB, 4x4 = 5.22 s / 9.3 GiB of 24. No OOM
# at any setting; 4x4 is both fastest and comfortable.
#
# SPLIT: reuses outputs/ft3d_flow_p/split.json so the result is comparable to the current
# best model. Without it each run derives its OWN partition from the same seed -- which has
# already leaked 17/41 test patients between two runs in this project.
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=/root/petct/.venv/bin/python
CACHE=/mnt/c/DeepTrainingData/PetCT
SPLIT=outputs/ft3d_flow_p/split.json
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
for f in outputs/ae3d_2x2/best.pt outputs/diff2d_2x_long/best.pt "$SPLIT"; do
  [ -f "$f" ] || { echo "MISSING prerequisite: $f" >&2; exit 1; }
done

echo "################ stage 4: ft3d_2x (${FT3D_STEPS} steps, 128^3) $(date +%H:%M)"
"$PY" -m src.training.train.train_ft3d \
  --config src/training/configs/ft3d_2x.yaml --save_dir outputs/ft3d_2x \
  --data_dir "${ROOTS[@]}" --cache_dir "$CACHE" --prefetch 6 --cache_size 24 \
  --ae_ckpt outputs/ae3d_2x2/best.pt --inflate_from outputs/diff2d_2x_long/best.pt \
  --split_json "$SPLIT" \
  --epochs $((FT3D_STEPS / 500)) --steps_per_epoch 500 --val_every 1000 --val_batches 4 \
  --latent_size 128 --device cuda 2>&1 | tail -12

echo "################ evaluating ft3d_2x on the shared 41-patient test split $(date +%H:%M)"
for S in 16 32; do
  "$PY" -m src.training.evaluate --task ft3d \
    --data_dir "${ROOTS[@]}" \
    --ae_ckpt outputs/ae3d_2x2/best.pt --diff_ckpt outputs/ft3d_2x/best.pt \
    --split_json "$SPLIT" --mode test \
    --size 128 --ddim_steps "$S" --spacing linear --max_patients 0 --device cuda \
    --clamp_output --out "outputs/eval/ft3d_2x/test_s${S}.json" 2>&1 | tail -2
done
echo "################ STAGE 4 DONE"
