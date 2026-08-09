#!/usr/bin/env bash
# Evaluate BOTH the baseline and the lpl_rfpp arm on the 24 patients held out by BOTH
# training runs. Needed because the two runs produced different partitions at the same
# seed (see outputs/eval/_clean24_split.json), so neither run's own test set is leak-free
# for the other model.
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=/root/petct/.venv/bin/python
SPLIT=outputs/eval/_clean24_split.json
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
for RUN in ft3d_flow_p ft3d_flow_lpl_rfpp; do
  echo "=== $RUN on the clean-24 split"
  "$PY" -m src.training.evaluate --task ft3d \
    --data_dir "${ROOTS[@]}" \
    --ae_ckpt outputs/ae3d_p/best.pt --diff_ckpt "outputs/$RUN/best.pt" \
    --split_json "$SPLIT" --mode test \
    --size 64 --ddim_steps 16 --spacing linear --max_patients 0 --device cuda \
    --clamp_output --out "outputs/eval/$RUN/clean24_s16.json" 2>&1 | tail -2
done
