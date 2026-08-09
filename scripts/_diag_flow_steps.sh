#!/usr/bin/env bash
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=~/petct/.venv/bin/python
ROOTS=("/mnt/d/DeepTrainingData/Project/ACRIN 6668" "/mnt/d/DeepTrainingData/Project/CPTAC-LSCC" "/mnt/d/DeepTrainingData/Project/CPTAC-LUAD" "/mnt/d/DeepTrainingData/Project/CPTAC-PDA" "/mnt/d/DeepTrainingData/Project/CPTAC-UCEC" "/mnt/d/DeepTrainingData/Project/NSCLC_Radiogenomics" "/mnt/d/DeepTrainingData/Project/PetCt/Bladder 13.11.25" "/mnt/d/DeepTrainingData/Project/PetCtNormal/PET_CT_NORMAL_2" "/mnt/d/DeepTrainingData/Project/TCGA-LUAD" "/mnt/d/DeepTrainingData/Project/TCGA-THCA")
common=(--task diff2d --data_dir "${ROOTS[@]}" --val_fraction 0.2 --test_fraction 0.1 --seed 42 --diff_ckpt outputs/diff2d_flow/best.pt --ae_ckpt outputs/ae2d/best.pt --split_json outputs/diff2d_flow/split.json --size 128 --num_slices 32 --spacing linear --guidance_scale 1.0 --max_patients 0 --device cuda --mode test)
for S in 32 64; do
  echo ">>> TEST @ ddim_steps=$S"
  "$PY" -m src.training.evaluate "${common[@]}" --ddim_steps $S --out outputs/eval/diff2d_flow/test_s$S.json >outputs/_diag_test_s$S.log 2>&1; echo "rc=$?"
done
echo ">>> DIAG_STEPS_DONE"
