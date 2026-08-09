#!/usr/bin/env bash
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=~/petct/.venv/bin/python
ROOTS=("/mnt/d/DeepTrainingData/Project/ACRIN 6668" "/mnt/d/DeepTrainingData/Project/CPTAC-LSCC" "/mnt/d/DeepTrainingData/Project/CPTAC-LUAD" "/mnt/d/DeepTrainingData/Project/CPTAC-PDA" "/mnt/d/DeepTrainingData/Project/CPTAC-UCEC" "/mnt/d/DeepTrainingData/Project/NSCLC_Radiogenomics" "/mnt/d/DeepTrainingData/Project/PetCt/Bladder 13.11.25" "/mnt/d/DeepTrainingData/Project/PetCtNormal/PET_CT_NORMAL_2" "/mnt/d/DeepTrainingData/Project/TCGA-LUAD" "/mnt/d/DeepTrainingData/Project/TCGA-THCA")
for S in 16 32; do
  echo ">>> eval_s$S $(date '+%H:%M:%S')"
  "$PY" -m src.training.evaluate --task ft3d --data_dir "${ROOTS[@]}" --mode test \
    --val_fraction 0.2 --test_fraction 0.1 --seed 42 \
    --diff_ckpt outputs/ft3d_flow/best.pt --ae_ckpt outputs/ae3d/best.pt --split_json outputs/ft3d_flow/split.json \
    --size 64 --ddim_steps $S --spacing linear --guidance_scale 1.0 --max_patients 0 --device cuda \
    --out outputs/eval/ft3d_flow/test_s$S.json > outputs/eval_ft3d_flow_s$S.log 2>&1
  echo ">>> done s$S rc=$? $(date '+%H:%M:%S')"
done
echo ">>> EVAL_DONE $(date '+%H:%M:%S')"
