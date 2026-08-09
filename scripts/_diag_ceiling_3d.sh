#!/usr/bin/env bash
# 3D ceiling / passthrough diagnostic on the held-out TEST split of the current best
# 3D flow model. Answers, per metric, whether the bottleneck is the frozen 3D AE or the
# flow model -- see scripts/_diag_ceiling_3d.py.
# Pass extra args through (e.g. --max_patients 4 for a quick look, --skip_flow).
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

"$PY" scripts/_diag_ceiling_3d.py \
  --data_dir "${ROOTS[@]}" \
  --ae_ckpt outputs/ae3d_p/best.pt \
  --diff_ckpt outputs/ft3d_flow_p/best.pt \
  --split_json outputs/ft3d_flow_p/split.json \
  --mode test --val_fraction 0.2 --test_fraction 0.1 --seed 42 \
  --size 64 --ddim_steps 16 --spacing linear --max_patients 0 \
  --device cuda \
  --out outputs/report/ceiling_3d.json \
  "$@"
