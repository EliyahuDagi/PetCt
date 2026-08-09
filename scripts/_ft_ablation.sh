#!/usr/bin/env bash
# SHORT fine-tune ablation: what fixes the under-dispersion / hot-tissue underestimation?
#
# Every arm starts from the SAME best checkpoint (ft3d_flow_p/best.pt, weights only) and
# reuses the SAME partition (ft3d_flow_p/split.json), so all arms and the baseline share one
# leak-free 41-patient test set and can be compared PAIRED per patient.
#
#   control    nothing changed        -- isolates fine-tuning + batching (MANDATORY baseline)
#   slope      slope+variance penalty -- optimizes the broken statistic directly
#   iw         intensity-weighted L1  -- hot voxels dominate the objective
#   expectile  q=0.7 expectile        -- optimum sits above the conditional mean
#   ushaped    U-shaped tau sampling  -- free; isolated from the failed lpl_rfpp combo
#   lpl        LPL perceptual only    -- isolated from the (1-tau) weighting it was bundled with
#
# Usage:  bash scripts/_ft_ablation.sh [arm ...]        (default: all six, in the order above)
#         STEPS=1500 bash scripts/_ft_ablation.sh slope
#
# Judge with scripts/_cmp_eval_paired.py against outputs/eval/ft3d_flow_p/test_s16_clamped.json
# on psnr / voxel_r2 / reg_slope / hot_band_rel_error -- NOT max_rel_error (AE-limited, ~0
# under the clamp). Expect a TRADE: fixing the slope should cost some PSNR.
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=/root/petct/.venv/bin/python
CACHE=/mnt/c/DeepTrainingData/PetCT
STEPS=${STEPS:-1500}
VAL_EVERY=${VAL_EVERY:-250}
INIT=${INIT:-outputs/ft3d_flow_p/best.pt}
SPLIT=${SPLIT:-outputs/ft3d_flow_p/split.json}
AE=${AE:-outputs/ae3d_p/best.pt}

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
for f in "$INIT" "$SPLIT" "$AE"; do
  [ -e "$f" ] || { echo "MISSING prerequisite: $f" >&2; exit 1; }
done

ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(control slope iw expectile ushaped lpl)

for ARM in "${ARMS[@]}"; do
  CFG="src/training/configs/ft3d_ft_${ARM}.yaml"
  SAVE="outputs/ft3d_ft_${ARM}"
  [ -f "$CFG" ] || { echo "no such arm config: $CFG" >&2; exit 1; }
  echo "================= ARM ${ARM}  (${STEPS} steps, init from ${INIT})"
  mkdir -p "$SAVE"
  "$PY" -m src.training.train.train_ft3d \
    --config "$CFG" --save_dir "$SAVE" \
    --data_dir "${ROOTS[@]}" --cache_dir "$CACHE" --prefetch 6 --cache_size 32 \
    --ae_ckpt "$AE" --init_from "$INIT" --split_json "$SPLIT" \
    --epochs 1 --steps_per_epoch "$STEPS" \
    --val_every "$VAL_EVERY" --val_batches 4 --latent_size 64 \
    --device cuda 2>&1 | tail -12

  # Eval on the SHARED split so every arm is paired against the same 41 patients.
  "$PY" -m src.training.evaluate --task ft3d \
    --data_dir "${ROOTS[@]}" \
    --ae_ckpt "$AE" --diff_ckpt "$SAVE/best.pt" \
    --split_json "$SPLIT" --mode test \
    --size 64 --ddim_steps 16 --spacing linear --max_patients 0 --device cuda \
    --clamp_output --out "outputs/eval/ft3d_ft_${ARM}/test_s16.json" 2>&1 | tail -2
done

echo "================= ablation done"
