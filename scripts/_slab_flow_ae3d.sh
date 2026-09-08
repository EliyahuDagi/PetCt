#!/usr/bin/env bash
# Slab-mode 3D flow run on the IN-PLANE-ONLY 3D AUTOENCODER: the controlled twin of
# scripts/_slab_flow.sh. Everything is identical (UNet, slabs, depth windows, warm start
# from the 2D flow bridge, patient split, step budget) except the frozen autoencoder:
#
#   frozen in-plane-only 3D autoencoder (outputs/ae3d_slab/best.pt; the 2D autoencoder
#   centre-inflated and fine-tuned on 128x128x16 slabs, depth never compressed)
#   -> 3D flow UNet with in-plane-only strides, centre-inflated from the 2D flow
#      bridge (outputs/diff2d_flow_perc_p/best.pt), trained on 128x128x16 depth slabs
#      at native depth, validated on whole volumes with a depth sliding window.
#
# Uses the SAME patient split as ft3d_flow_p so the 41 test patients are shared with
# the 2D control and the 2D-autoencoder slab run (paired comparison).
# Usage (inside WSL):  EPOCHS=20 bash scripts/_slab_flow_ae3d.sh     (RESUME=1 to continue)
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=~/petct/.venv/bin/python
# Patient cache on the WSL disk (copy of /mnt/c/DeepTrainingData/PetCT): ~0.03 s per patient
# instead of ~0.6 s through the Windows drive bridge.
CACHE=${CACHE:-/root/petct_cache}
EPOCHS=${EPOCHS:-20}
SAVE=${SAVE:-outputs/ft3d_slab_flow_ae3d}
AE=${AE:-outputs/ae3d_slab/best.pt}
# RESUME=1 continues from $SAVE/last.pt and APPENDS to train.log / metrics.jsonl. Launch
# detached (setsid nohup ... &) so the job survives the terminal that started it.
RESUME=${RESUME:-}
# INIT_FROM=<3D flow checkpoint>: start a NEW run (fresh optimizer, fresh warmup + cosine
# cycle, step counter at 0) from that checkpoint's EMA weights instead of inflating the 2D
# flow bridge. Used to continue outputs/ft3d_slab_flow_ae3d for another cycle in a new SAVE
# dir (its steps are numbered from 0 again: step k here = original step 20000 + k).
INIT_FROM=${INIT_FROM:-}

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

for f in "$AE" outputs/diff2d_flow_perc_p/best.pt outputs/ft3d_flow_p/split.json ${RESUME:+$SAVE/last.pt} ${INIT_FROM:+$INIT_FROM}; do
  if [ ! -f "$f" ]; then echo ">>> MISSING $f -- abort"; exit 2; fi
done

mkdir -p "$SAVE"
if [ -z "$RESUME" ]; then : > "$SAVE/train.log"; fi
echo ">>> STAGE_START ft3d_slab_flow_ae3d_train${RESUME:+ (resume from $SAVE/last.pt)}${INIT_FROM:+ (init from $INIT_FROM)} $(date '+%Y-%m-%d %H:%M:%S')"
# Warm start: inflate the 2D flow bridge (default) or init from an existing 3D checkpoint.
if [ -n "$INIT_FROM" ]; then WARM=(--init_from "$INIT_FROM"); else WARM=(--inflate_from outputs/diff2d_flow_perc_p/best.pt); fi
"$PY" -m src.training.train.launcher ft3d \
  --config src/training/configs/ft3d_slab_flow_ae3d.yaml \
  --data_dir "${ROOTS[@]}" --cache_dir "$CACHE" --device cuda \
  --val_fraction 0.2 --test_fraction 0.1 \
  --split_json outputs/ft3d_flow_p/split.json \
  --epochs "$EPOCHS" --steps_per_epoch 1000 --val_every 2500 --val_batches 8 \
  --latent_size 128 \
  --ae_ckpt "$AE" \
  "${WARM[@]}" \
  --save_dir "$SAVE" --cache_size 128 ${RESUME:+--resume} \
  >> "$SAVE/train.log" 2>&1
rc=$?; echo ">>> STAGE_END ft3d_slab_flow_ae3d_train rc=$rc $(date '+%Y-%m-%d %H:%M:%S')"
