#!/usr/bin/env bash
# Extend the guidance sweep (w=4,5,6) to find the SSIM peak, plus one reconciliation
# eval (final model @ 24 slices, w=1.0) to isolate the slice-count effect vs the
# early 30k result (0.313 @ 24 slices).
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=~/petct/.venv/bin/python
R1="/mnt/d/DeepTrainingData/Project/ACRIN 6668"
R2="/mnt/d/DeepTrainingData/Project/PetCt/Bladder 13.11.25"
R3="/mnt/d/DeepTrainingData/Project/PetCtNormal/PET_CT_NORMAL_2"
SAVE=outputs/diff2d_cfg

for G in 4.0 5.0 6.0; do
  echo ">>> ext w=${G} $(date +%H:%M:%S)"
  "$PY" -m src.training.evaluate --task diff2d --data_dir "$R1" "$R2" "$R3" --mode test \
    --val_fraction 0.2 --test_fraction 0.1 --seed 42 \
    --diff_ckpt "$SAVE/best.pt" --ae_ckpt outputs/ae2d/best.pt --split_json "$SAVE/split.json" \
    --size 128 --num_slices 32 --ddim_steps 25 --spacing karras --clip_x0 4.0 \
    --guidance_scale "$G" --max_patients 0 --device cuda \
    --out "outputs/eval/diff2d_cfg/g${G}.json" > "outputs/eval_ext_g${G}.log" 2>&1
done

echo ">>> reconciliation: final model, 24 slices, w=1.0 $(date +%H:%M:%S)"
"$PY" -m src.training.evaluate --task diff2d --data_dir "$R1" "$R2" "$R3" --mode test \
  --val_fraction 0.2 --test_fraction 0.1 --seed 42 \
  --diff_ckpt "$SAVE/best.pt" --ae_ckpt outputs/ae2d/best.pt --split_json "$SAVE/split.json" \
  --size 128 --num_slices 24 --ddim_steps 25 --spacing karras --clip_x0 4.0 \
  --guidance_scale 1.0 --max_patients 0 --device cuda \
  --out "outputs/eval/diff2d_cfg/recon_final_24slice.json" > outputs/eval_recon_final24.log 2>&1
echo ">>> EXT_DONE $(date +%H:%M:%S)"
