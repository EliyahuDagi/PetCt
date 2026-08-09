#!/usr/bin/env bash
# Ablation 2: 3D flow bridge, warm-started by inflating the 2D FLOW UNet.
# flow->flow inflation (in=8 both) so the input conv inflates (no 2C mismatch).
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=~/petct/.venv/bin/python
CACHE=/mnt/c/DeepTrainingData/PetCT
SAVE=outputs/ft3d_flow
ROOTS=("/mnt/d/DeepTrainingData/Project/ACRIN 6668" "/mnt/d/DeepTrainingData/Project/CPTAC-LSCC" "/mnt/d/DeepTrainingData/Project/CPTAC-LUAD" "/mnt/d/DeepTrainingData/Project/CPTAC-PDA" "/mnt/d/DeepTrainingData/Project/CPTAC-UCEC" "/mnt/d/DeepTrainingData/Project/NSCLC_Radiogenomics" "/mnt/d/DeepTrainingData/Project/PetCt/Bladder 13.11.25" "/mnt/d/DeepTrainingData/Project/PetCtNormal/PET_CT_NORMAL_2" "/mnt/d/DeepTrainingData/Project/TCGA-LUAD" "/mnt/d/DeepTrainingData/Project/TCGA-THCA")

echo ">>> STAGE_START ft3d_flow_train $(date '+%H:%M:%S')"
"$PY" -m src.training.train.launcher ft3d \
  --config src/training/configs/ft3d_flow.yaml \
  --data_dir "${ROOTS[@]}" --cache_dir "$CACHE" --device cuda \
  --val_fraction 0.2 --test_fraction 0.1 \
  --epochs 30 --steps_per_epoch 1000 --val_every 2500 --val_batches 4 \
  --latent_size 64 --ae_ckpt outputs/ae3d/best.pt \
  --inflate_from outputs/diff2d_flow2/best.pt --save_dir "$SAVE" \
  --cache_size 128 \
  > outputs/ft3d_flow.log 2>&1
rc=$?; echo ">>> STAGE_END ft3d_flow_train rc=$rc $(date '+%H:%M:%S')"
if [ $rc -ne 0 ]; then echo ">>> ABORT train"; exit $rc; fi

for S in 16 32; do
  echo ">>> STAGE_START eval_s$S $(date '+%H:%M:%S')"
  "$PY" -m src.training.evaluate --task ft3d \
    --data_dir "${ROOTS[@]}" --mode test \
    --val_fraction 0.2 --test_fraction 0.1 --seed 42 \
    --diff_ckpt "$SAVE/best.pt" --ae_ckpt outputs/ae3d/best.pt --split_json "$SAVE/split.json" \
    --size 64 --ddim_steps $S --spacing linear --guidance_scale 1.0 --max_patients 0 --device cuda \
    --out outputs/eval/ft3d_flow/test_s$S.json > outputs/eval_ft3d_flow_s$S.log 2>&1
  echo ">>> STAGE_END eval_s$S rc=$? $(date '+%H:%M:%S')"
done
echo ">>> PIPELINE_DONE $(date '+%H:%M:%S')"
