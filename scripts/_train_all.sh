#!/usr/bin/env bash
# Full NAC->AC pipeline: ae2d -> ae3d -> diff2d -> ft3d, then evaluate diff2d/ft3d
# on the held-out TEST split, all with augmentation + SSD cache + by-patient
# train/val/test split. Run with the WSL torch venv. Each stage logs to
# outputs/<stage>.log; stage markers are echoed to stdout for monitoring.
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt

PY=~/petct/.venv/bin/python
CACHE=/mnt/c/DeepTrainingData/PetCT
DEV=cuda
VF=0.2
TF=0.1

R1="/mnt/d/DeepTrainingData/Project/ACRIN 6668"
R2="/mnt/d/DeepTrainingData/Project/PetCt/Bladder 13.11.25"
R3="/mnt/d/DeepTrainingData/Project/PetCtNormal/PET_CT_NORMAL_2"

run_stage () {  # $1=name ; rest=args
  local name="$1"; shift
  echo ">>> STAGE_START $name $(date '+%H:%M:%S')"
  "$PY" "$@" > "outputs/${name}.log" 2>&1
  local rc=$?
  echo ">>> STAGE_END $name rc=$rc $(date '+%H:%M:%S')"
  if [ $rc -ne 0 ]; then echo ">>> PIPELINE_ABORT at $name"; exit $rc; fi
}

# ---- Stage 1: 2D AutoencoderKL ----  (100k steps)
run_stage ae2d -m src.training.train.launcher ae2d \
  --config src/training/configs/ae2d.yaml \
  --data_dir "$R1" "$R2" "$R3" --cache_dir "$CACHE" --device "$DEV" \
  --val_fraction $VF --test_fraction $TF \
  --epochs 200 --steps_per_epoch 500 --val_every 2000 --val_batches 8 \
  --slice_size 128

# ---- Stage 2: 3D AutoencoderKL (inflated from ae2d) ----  (25k steps)
run_stage ae3d -m src.training.train.launcher ae3d \
  --config src/training/configs/ae3d.yaml \
  --data_dir "$R1" "$R2" "$R3" --cache_dir "$CACHE" --device "$DEV" \
  --val_fraction $VF --test_fraction $TF \
  --epochs 50 --steps_per_epoch 500 --val_every 1000 --val_batches 4 \
  --crop_size 64

# ---- Stage 3: 2D latent diffusion (paired NAC+AC -> ACRIN only) ----  (300k steps)
run_stage diff2d -m src.training.train.launcher diff2d \
  --config src/training/configs/diff2d.yaml \
  --data_dir "$R1" "$R2" "$R3" --cache_dir "$CACHE" --device "$DEV" \
  --val_fraction $VF --test_fraction $TF \
  --epochs 300 --steps_per_epoch 1000 --val_every 3000 --val_batches 8 \
  --latent_size 128

# ---- Stage 4: 3D latent diffusion (inflated from diff2d) ----  (60k steps)
run_stage ft3d -m src.training.train.launcher ft3d \
  --config src/training/configs/ft3d.yaml \
  --data_dir "$R1" "$R2" "$R3" --cache_dir "$CACHE" --device "$DEV" \
  --val_fraction $VF --test_fraction $TF \
  --epochs 75 --steps_per_epoch 800 --val_every 2000 --val_batches 4 \
  --latent_size 64 --inflate_from outputs/diff2d/best.pt

# ---- Evaluation on the held-out TEST split (uses split.json from training) ----
echo ">>> STAGE_START eval_diff2d $(date '+%H:%M:%S')"
"$PY" -m src.training.evaluate --task diff2d \
  --data_dir "$R1" "$R2" "$R3" --mode test \
  --val_fraction $VF --test_fraction $TF --seed 42 \
  --size 128 --num_slices 32 --ddim_steps 25 --spacing karras \
  --max_patients 0 --device "$DEV" \
  --out outputs/eval/diff2d/metrics.json > outputs/eval_diff2d.log 2>&1
echo ">>> STAGE_END eval_diff2d rc=$? $(date '+%H:%M:%S')"

echo ">>> STAGE_START eval_ft3d $(date '+%H:%M:%S')"
"$PY" -m src.training.evaluate --task ft3d \
  --data_dir "$R1" "$R2" "$R3" --mode test \
  --val_fraction $VF --test_fraction $TF --seed 42 \
  --size 64 --ddim_steps 25 --spacing karras \
  --max_patients 0 --device "$DEV" \
  --out outputs/eval/ft3d/metrics.json > outputs/eval_ft3d.log 2>&1
echo ">>> STAGE_END eval_ft3d rc=$? $(date '+%H:%M:%S')"

echo ">>> PIPELINE_DONE $(date '+%H:%M:%S')"
