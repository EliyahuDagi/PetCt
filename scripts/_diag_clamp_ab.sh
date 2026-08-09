#!/usr/bin/env bash
# Clamp A/B: the SAME ft3d_flow_p checkpoint on the SAME 41 held-out test patients,
# evaluated with and without the [0,1] output clamp. No retraining is involved, so any
# movement beyond max_rel_error / nrmse / mae is a bug.
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=/root/petct/.venv/bin/python
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
for MODE in clamped unclamped; do
  FLAG="--clamp_output"; [ "$MODE" = unclamped ] && FLAG="--no_clamp_output"
  echo "=== $MODE ($FLAG)"
  "$PY" -m src.training.evaluate --task ft3d \
    --data_dir "${ROOTS[@]}" \
    --ae_ckpt outputs/ae3d_p/best.pt --diff_ckpt outputs/ft3d_flow_p/best.pt \
    --split_json outputs/ft3d_flow_p/split.json \
    --mode test --val_fraction 0.2 --test_fraction 0.1 --seed 42 \
    --size 64 --ddim_steps 16 --spacing linear --max_patients 0 --device cuda \
    $FLAG --out "outputs/eval/ft3d_flow_p/test_s16_${MODE}.json" 2>&1 | tail -3
done
