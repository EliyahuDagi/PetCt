#!/usr/bin/env bash
# Slab-mode 3D flow run: the "same problem as 2D, with depth context" chain.
#
#   frozen 2D autoencoder (outputs/ae2d_p/best.pt) applied slice by slice
#   -> 3D flow UNet with in-plane-only strides, centre-inflated from the 2D flow
#      bridge (outputs/diff2d_flow_perc_p/best.pt), trained on 128x128x16 depth slabs
#      at native depth, validated on whole volumes with a depth sliding window.
#
# Uses the SAME patient split as ft3d_flow_p so the 41 test patients are shared with
# the 2D control and the older 3D chain (paired comparison).
# Usage (inside WSL):  EPOCHS=20 bash scripts/_slab_flow.sh
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=~/petct/.venv/bin/python
# Patient cache on the WSL disk (copy of /mnt/c/DeepTrainingData/PetCT). Reading the .npz files
# through the Windows drive bridge (9p) costs ~0.6 s per patient vs ~0.03 s here, and each
# optimiser step loads 4 patients -- on /mnt/c the run was 1.1 s/step instead of 0.4 s.
CACHE=${CACHE:-/root/petct_cache}
EPOCHS=${EPOCHS:-20}
SAVE=${SAVE:-outputs/ft3d_slab_flow}
# RESUME=1 continues from $SAVE/last.pt (model, optimiser, EMA, RNG, step) and APPENDS to train.log
# and metrics.jsonl instead of starting over.  Launch detached (setsid nohup ... &) so the job
# survives the terminal that started it -- the first run died at step 5000 when its parent exited.
RESUME=${RESUME:-}

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

for f in outputs/ae2d_p/best.pt outputs/diff2d_flow_perc_p/best.pt outputs/ft3d_flow_p/split.json ${RESUME:+$SAVE/last.pt}; do
  if [ ! -f "$f" ]; then echo ">>> MISSING $f -- abort"; exit 2; fi
done

mkdir -p "$SAVE"
if [ -z "$RESUME" ]; then : > "$SAVE/train.log"; fi
echo ">>> STAGE_START ft3d_slab_flow_train${RESUME:+ (resume from $SAVE/last.pt)} $(date '+%Y-%m-%d %H:%M:%S')"
"$PY" -m src.training.train.launcher ft3d \
  --config src/training/configs/ft3d_slab_flow.yaml \
  --data_dir "${ROOTS[@]}" --cache_dir "$CACHE" --device cuda \
  --val_fraction 0.2 --test_fraction 0.1 \
  --split_json outputs/ft3d_flow_p/split.json \
  --epochs "$EPOCHS" --steps_per_epoch 1000 --val_every 2500 --val_batches 8 \
  --latent_size 128 \
  --ae2d_ckpt outputs/ae2d_p/best.pt \
  --inflate_from outputs/diff2d_flow_perc_p/best.pt \
  --save_dir "$SAVE" --cache_size 128 ${RESUME:+--resume} \
  >> "$SAVE/train.log" 2>&1
rc=$?; echo ">>> STAGE_END ft3d_slab_flow_train rc=$rc $(date '+%Y-%m-%d %H:%M:%S')"
