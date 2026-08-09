#!/usr/bin/env bash
# A/B arms for the 3D flow bridge, against the ft3d_flow_p pure-MSE baseline.
#
#   ushaped   (i)   RFPP U-shaped tau sampling only        -- costs nothing per step
#   lpl       (ii)  LPL latent perceptual only             -- single variable vs control
#   lpl_rfpp  (iii) LPL + (1-tau) weighting + U-shaped tau -- the full RFPP recipe
#
# Each arm is inflated from the SAME 2D flow checkpoint and uses the SAME frozen ae3d_p AE
# as ft3d_flow_p, so only the config differs. After training, each is evaluated on the
# held-out TEST split at 16 and 32 flow steps.
#
# Usage:  bash scripts/_flow_ft3d_arms.sh [ushaped|lpl|lpl_rfpp ...]      (default: all)
# Judge on: best rollout val l1, psnr, voxel_r2, hot_band_rel_error, p95_rel_error.
# Do NOT judge on max_rel_error -- it is AE-limited and ~0 under the default clamp
# (see docs/nac_ac_benchmark.md §1 and scripts/_diag_ceiling_3d.sh).
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=/root/petct/.venv/bin/python
CACHE=/mnt/c/DeepTrainingData/PetCT
EPOCHS=${EPOCHS:-30}
STEPS=${STEPS:-1000}
INFLATE=${INFLATE:-outputs/diff2d_flow_perc_p/best.pt}
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

for f in "$INFLATE" "$AE"; do
  [ -f "$f" ] || { echo "MISSING prerequisite: $f" >&2; exit 1; }
done

ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(ushaped lpl lpl_rfpp)

for ARM in "${ARMS[@]}"; do
  CFG="src/training/configs/ft3d_flow_${ARM}.yaml"
  SAVE="outputs/ft3d_flow_${ARM}"
  [ -f "$CFG" ] || { echo "no such arm config: $CFG" >&2; exit 1; }
  echo "=============== ARM $ARM  ($CFG -> $SAVE)"
  mkdir -p "$SAVE"
  "$PY" -m src.training.train.train_ft3d \
    --config "$CFG" --save_dir "$SAVE" \
    --data_dir "${ROOTS[@]}" --cache_dir "$CACHE" \
    --prefetch "${PREFETCH:-6}" --cache_size "${CACHE_SIZE:-32}" \
    --ae_ckpt "$AE" --inflate_from "$INFLATE" \
    --epochs "$EPOCHS" --steps_per_epoch "$STEPS" \
    --val_every 250 --val_batches 4 --latent_size 64 \
    --device cuda 2>&1 | tail -20

  for S in 16 32; do
    "$PY" -m src.training.evaluate --task ft3d \
      --data_dir "${ROOTS[@]}" \
      --ae_ckpt "$AE" --diff_ckpt "$SAVE/best.pt" \
      --split_json "$SAVE/split.json" \
      --mode test --val_fraction 0.2 --test_fraction 0.1 --seed 42 \
      --size 64 --ddim_steps "$S" --spacing linear --max_patients 0 --device cuda \
      --out "outputs/eval/ft3d_flow_${ARM}/test_s${S}.json" 2>&1 | tail -3
  done
done

echo "=============== done. Compare against outputs/eval/ft3d_flow_p/test_s16_clamped.json"
