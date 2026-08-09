#!/usr/bin/env bash
# Early guidance sweep on the in-progress CFG diff2d checkpoint (~30k steps),
# run concurrently with training. Snapshots best.pt so all 4 runs use one model.
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=~/petct/.venv/bin/python
R1="/mnt/d/DeepTrainingData/Project/ACRIN 6668"
R2="/mnt/d/DeepTrainingData/Project/PetCt/Bladder 13.11.25"
R3="/mnt/d/DeepTrainingData/Project/PetCtNormal/PET_CT_NORMAL_2"
SNAP=outputs/diff2d_cfg/snap_early.pt

cp outputs/diff2d_cfg/best.pt "$SNAP"
echo ">>> snapshot taken $(date '+%H:%M:%S')"

for G in 1.0 1.5 2.0 3.0; do
  echo ">>> SWEEP g=${G} start $(date '+%H:%M:%S')"
  "$PY" -m src.training.evaluate --task diff2d \
    --data_dir "$R1" "$R2" "$R3" --mode test \
    --val_fraction 0.2 --test_fraction 0.1 --seed 42 \
    --diff_ckpt "$SNAP" --ae_ckpt outputs/ae2d/best.pt --split_json outputs/diff2d_cfg/split.json \
    --size 128 --num_slices 24 --ddim_steps 25 --spacing karras --clip_x0 4.0 \
    --guidance_scale "$G" --max_patients 0 --device cuda \
    --out "outputs/eval/diff2d_cfg_early/g${G}.json" > "outputs/eval_early_g${G}.log" 2>&1
  echo ">>> SWEEP g=${G} done rc=$? $(date '+%H:%M:%S')"
done
echo ">>> SWEEP_DONE $(date '+%H:%M:%S')"
