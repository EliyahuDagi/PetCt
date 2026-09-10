#!/usr/bin/env bash
# Test-set scoring for the slab-mode 3D flow run and its fair 2D control.
#
# Both are scored the same way: whole volume, 128x128 in-plane, native depth, the 41 test
# patients of outputs/ft3d_flow_p/split.json. The 2D control is the 2D flow bridge run
# slice by slice and stacked (evaluate.py --task diff2d --volumetric).
# Usage (inside WSL):  WHICH=control bash scripts/_slab_flow_eval.sh   # or WHICH=slab / WHICH=both
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=~/petct/.venv/bin/python
CACHE=/mnt/c/DeepTrainingData/PetCT
WHICH=${WHICH:-both}
STEPS=${STEPS:-16}
SAVE=${SAVE:-outputs/ft3d_slab_flow}
CKPT=${CKPT:-$SAVE/best.pt}
TAG=${TAG:-best}
# Frozen autoencoder for the slab arm: the 2D one applied slice by slice (default) or an
# in-plane-only 3D one (AE=outputs/ae3d_slab/best.pt for the ft3d_slab_flow_ae3d chain).
AE=${AE:-outputs/ae2d_p/best.pt}
SPLIT=outputs/ft3d_flow_p/split.json
# Results folder; the 2D control JSON is shared, so point OUT at the folder that has it.
OUT=${OUT:-outputs/eval/ft3d_slab_flow}
# DUMP=<dir> also writes each scored patient's prediction and ground truth as .npy, for
# scripts/_diag_false_hot.py to count invented uptake with. Costs ~40 MB per patient, so
# pair it with MAXPAT=<n> when only a look is wanted -- the split order is fixed, so the
# same MAXPAT gives the same patients for every arm and the dumps stay comparable.
DUMP=${DUMP:-}
MAXPAT=${MAXPAT:-0}
mkdir -p "$OUT"

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

if [ "$WHICH" = control ] || [ "$WHICH" = both ]; then
  echo ">>> STAGE_START control_2d_volumetric_s$STEPS $(date '+%Y-%m-%d %H:%M:%S')"
  "$PY" -m src.training.evaluate --task diff2d --volumetric \
    --data_dir "${ROOTS[@]}" --mode test \
    --val_fraction 0.2 --test_fraction 0.1 --seed 42 --split_json "$SPLIT" \
    --diff_ckpt outputs/diff2d_flow_perc_p/best.pt --ae_ckpt outputs/ae2d_p/best.pt \
    --size 128 --ddim_steps "$STEPS" --spacing linear --guidance_scale 1.0 --max_patients 0 --device cuda \
    --out "$OUT/control2d_vol_s$STEPS.json" > "$OUT/control2d_vol_s$STEPS.log" 2>&1
  echo ">>> STAGE_END control rc=$? $(date '+%Y-%m-%d %H:%M:%S')"
fi
if [ "$WHICH" = slab ] || [ "$WHICH" = both ]; then
  echo ">>> STAGE_START slab_${TAG}_s$STEPS $(date '+%Y-%m-%d %H:%M:%S')"
  "$PY" -m src.training.evaluate --task ft3d \
    --data_dir "${ROOTS[@]}" --mode test \
    --val_fraction 0.2 --test_fraction 0.1 --seed 42 --split_json "$SPLIT" \
    --diff_ckpt "$CKPT" --ae_ckpt "$AE" \
    --size 128 --ddim_steps "$STEPS" --spacing linear --guidance_scale 1.0 --max_patients "$MAXPAT" --device cuda \
    ${DUMP:+--save_pred_dir "$DUMP"} \
    --out "$OUT/slab_${TAG}_s$STEPS.json" > "$OUT/slab_${TAG}_s$STEPS.log" 2>&1
  echo ">>> STAGE_END slab rc=$? $(date '+%Y-%m-%d %H:%M:%S')"
  # A capped run scores a different patient set from the shared 2D control, so pairing the
  # two would silently compare different cohorts. Skip it there.
  if [ "$MAXPAT" = "0" ]; then
    "$PY" scripts/_cmp_eval_paired.py "$OUT/control2d_vol_s$STEPS.json" "$OUT/slab_${TAG}_s$STEPS.json" \
      --label-a control2d --label-b "slab_$TAG" | tee "$OUT/paired_${TAG}_s$STEPS.txt"
  else
    echo "MAXPAT=$MAXPAT: capped run, so no pairing against the 41-patient 2D control."
  fi
fi
