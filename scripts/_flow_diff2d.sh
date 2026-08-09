#!/usr/bin/env bash
# Train the NAC->AC rectified-flow BRIDGE (prediction_type=flow) on all paired roots,
# writing to outputs/diff2d_flow (baselines outputs/diff2d, outputs/diff2d_cfg preserved),
# then evaluate on the held-out TEST split. Reuses the existing 2D AE (outputs/ae2d/best.pt).
# Non-paired patients in any root are auto-skipped by the diffusion pairing filter.
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=~/petct/.venv/bin/python
CACHE=/mnt/c/DeepTrainingData/PetCT
SAVE=outputs/diff2d_flow

ROOTS=(
  "/mnt/d/DeepTrainingData/Project/ACRIN 6668"
  "/mnt/d/DeepTrainingData/Project/CPTAC-LSCC"
  "/mnt/d/DeepTrainingData/Project/CPTAC-LUAD"
  "/mnt/d/DeepTrainingData/Project/CPTAC-PDA"
  "/mnt/d/DeepTrainingData/Project/CPTAC-UCEC"
  "/mnt/d/DeepTrainingData/Project/NSCLC_Radiogenomics"
  "/mnt/d/DeepTrainingData/Project/PetCt/Bladder 13.11.25"
  "/mnt/d/DeepTrainingData/Project/PetCtNormal/PET_CT_NORMAL_2"
  "/mnt/d/DeepTrainingData/Project/TCGA-LUAD"
  "/mnt/d/DeepTrainingData/Project/TCGA-THCA"
)

echo ">>> STAGE_START diff2d_flow_train $(date '+%H:%M:%S')"
"$PY" -m src.training.train.launcher diff2d \
  --config src/training/configs/diff2d_flow.yaml \
  --data_dir "${ROOTS[@]}" --cache_dir "$CACHE" --device cuda \
  --val_fraction 0.2 --test_fraction 0.1 \
  --epochs 100 --steps_per_epoch 1000 --val_every 2500 --val_batches 8 \
  --latent_size 128 --ae_ckpt outputs/ae2d/best.pt --save_dir "$SAVE" \
  --cache_size 280 \
  > outputs/diff2d_flow.log 2>&1
rc=$?; echo ">>> STAGE_END diff2d_flow_train rc=$rc $(date '+%H:%M:%S')"
if [ $rc -ne 0 ]; then echo ">>> ABORT train"; exit $rc; fi

echo ">>> STAGE_START eval_flow $(date '+%H:%M:%S')"
"$PY" -m src.training.evaluate --task diff2d \
  --data_dir "${ROOTS[@]}" --mode test \
  --val_fraction 0.2 --test_fraction 0.1 --seed 42 \
  --diff_ckpt "$SAVE/best.pt" --ae_ckpt outputs/ae2d/best.pt --split_json "$SAVE/split.json" \
  --size 128 --num_slices 32 --ddim_steps 16 --spacing linear \
  --guidance_scale 1.0 --max_patients 0 --device cuda \
  --out outputs/eval/diff2d_flow/metrics.json > outputs/eval_diff2d_flow.log 2>&1
echo ">>> STAGE_END eval_flow rc=$? $(date '+%H:%M:%S')"
echo ">>> PIPELINE_DONE $(date '+%H:%M:%S')"
