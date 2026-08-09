#!/usr/bin/env bash
# RESUME the CFG diff2d run from outputs/diff2d_cfg/last.pt (~30k) to 150k steps,
# then run the held-out TEST guidance sweep. Does NOT restart from scratch.
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=~/petct/.venv/bin/python
CACHE=/mnt/c/DeepTrainingData/PetCT
R1="/mnt/d/DeepTrainingData/Project/ACRIN 6668"
R2="/mnt/d/DeepTrainingData/Project/PetCt/Bladder 13.11.25"
R3="/mnt/d/DeepTrainingData/Project/PetCtNormal/PET_CT_NORMAL_2"
SAVE=outputs/diff2d_cfg

echo ">>> STAGE_START diff2d_cfg_resume $(date '+%H:%M:%S')"
"$PY" -m src.training.train.launcher diff2d \
  --config src/training/configs/diff2d.yaml \
  --data_dir "$R1" "$R2" "$R3" --cache_dir "$CACHE" --device cuda \
  --val_fraction 0.2 --test_fraction 0.1 \
  --epochs 150 --steps_per_epoch 1000 --val_every 2500 --val_batches 8 \
  --latent_size 128 --ae_ckpt outputs/ae2d/best.pt --save_dir "$SAVE" \
  --cache_size 130 --resume \
  > outputs/diff2d_cfg.log 2>&1
rc=$?; echo ">>> STAGE_END diff2d_cfg_resume rc=$rc $(date '+%H:%M:%S')"
if [ $rc -ne 0 ]; then echo ">>> ABORT resume"; exit $rc; fi

for G in 1.0 1.5 2.0 3.0; do
  echo ">>> STAGE_START eval_g${G} $(date '+%H:%M:%S')"
  "$PY" -m src.training.evaluate --task diff2d \
    --data_dir "$R1" "$R2" "$R3" --mode test \
    --val_fraction 0.2 --test_fraction 0.1 --seed 42 \
    --diff_ckpt "$SAVE/best.pt" --ae_ckpt outputs/ae2d/best.pt --split_json "$SAVE/split.json" \
    --size 128 --num_slices 32 --ddim_steps 25 --spacing karras --clip_x0 4.0 \
    --guidance_scale "$G" --max_patients 0 --device cuda \
    --out "outputs/eval/diff2d_cfg/g${G}.json" > "outputs/eval_diff2d_cfg_g${G}.log" 2>&1
  echo ">>> STAGE_END eval_g${G} rc=$? $(date '+%H:%M:%S')"
done
echo ">>> PIPELINE_DONE $(date '+%H:%M:%S')"
