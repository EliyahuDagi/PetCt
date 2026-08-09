#!/usr/bin/env bash
# FULL-DATA 3D flow pipeline (NAC-classifier fix + perceptual fix AE track).
#
# Every prior ft3d run reused outputs/ae3d/best.pt -- the Jun-12 3D AE trained on
# the buggy NAC-dropped pool. This chain builds the 3D diffusion on the FIXED,
# full-NAC+AC data instead:
#   Stage 1  ae3d_p   : inflate the fixed 2D AE (outputs/ae2d_p/best.pt, 628-patient
#                        pool, CorrectedImage-first classifier) -> 3D AutoencoderKL.
#   Stage 2  ft3d_flow_p : NAC->AC rectified-flow bridge, warm-started flow->flow from
#                        the 2D flow bridge on the same fixed AE (diff2d_flow_perc_p),
#                        trained on the ae3d_p latents.
#   Stage 3  eval      : test-set NAC->AC at 16 / 32 flow steps.
# Fresh dirs (outputs/ae3d_p, outputs/ft3d_flow_p) preserve the old runs.
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=~/petct/.venv/bin/python
CACHE=/mnt/c/DeepTrainingData/PetCT

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

# --- preflight: required upstream checkpoints must exist ---
for f in outputs/ae2d_p/best.pt outputs/diff2d_flow_perc_p/best.pt; do
  if [ ! -f "$f" ]; then echo ">>> MISSING $f -- abort"; exit 2; fi
done

# ================= Stage 1: ae3d_p (inflate ae2d_p -> 3D AE) =================
SAVE1=outputs/ae3d_p
mkdir -p "$SAVE1"
echo ">>> STAGE_START ae3d_p_train $(date '+%Y-%m-%d %H:%M:%S')"
"$PY" -m src.training.train.train_ae3d \
  --config src/training/configs/ae3d.yaml \
  --data_dir "${ROOTS[@]}" --cache_dir "$CACHE" --prefetch 2 \
  --epochs 100 --steps_per_epoch 500 --val_every 2500 --val_batches 4 \
  --val_fraction 0.2 --test_fraction 0.1 \
  --batch_size 1 --crop_size 64 \
  --ae2d_ckpt outputs/ae2d_p/best.pt \
  --device cuda --save_dir "$SAVE1" \
  > "$SAVE1/train.log" 2>&1
rc=$?; echo ">>> STAGE_END ae3d_p_train rc=$rc $(date '+%Y-%m-%d %H:%M:%S')"
if [ $rc -ne 0 ]; then echo ">>> ABORT after ae3d_p"; exit $rc; fi
"$PY" scripts/_ae_p_compare.py --old outputs/ae3d/metrics.jsonl --new "$SAVE1/metrics.jsonl" --tag ae3d || true

# ============= Stage 2: ft3d flow (inflate 2D flow bridge -> 3D) =============
SAVE2=outputs/ft3d_flow_p
mkdir -p "$SAVE2"
echo ">>> STAGE_START ft3d_flow_p_train $(date '+%Y-%m-%d %H:%M:%S')"
"$PY" -m src.training.train.launcher ft3d \
  --config src/training/configs/ft3d_flow.yaml \
  --data_dir "${ROOTS[@]}" --cache_dir "$CACHE" --device cuda \
  --val_fraction 0.2 --test_fraction 0.1 \
  --epochs 30 --steps_per_epoch 1000 --val_every 2500 --val_batches 4 \
  --latent_size 64 --ae_ckpt "$SAVE1/best.pt" \
  --inflate_from outputs/diff2d_flow_perc_p/best.pt --save_dir "$SAVE2" \
  --cache_size 128 \
  > outputs/ft3d_flow_p.log 2>&1
rc=$?; echo ">>> STAGE_END ft3d_flow_p_train rc=$rc $(date '+%Y-%m-%d %H:%M:%S')"
if [ $rc -ne 0 ]; then echo ">>> ABORT after ft3d train"; exit $rc; fi

# ============================ Stage 3: eval =================================
for S in 16 32; do
  echo ">>> STAGE_START eval_s$S $(date '+%Y-%m-%d %H:%M:%S')"
  "$PY" -m src.training.evaluate --task ft3d \
    --data_dir "${ROOTS[@]}" --mode test \
    --val_fraction 0.2 --test_fraction 0.1 --seed 42 \
    --diff_ckpt "$SAVE2/best.pt" --ae_ckpt "$SAVE1/best.pt" --split_json "$SAVE2/split.json" \
    --size 64 --ddim_steps $S --spacing linear --guidance_scale 1.0 --max_patients 0 --device cuda \
    --out outputs/eval/ft3d_flow_p/test_s$S.json > outputs/eval_ft3d_flow_p_s$S.log 2>&1
  echo ">>> STAGE_END eval_s$S rc=$? $(date '+%Y-%m-%d %H:%M:%S')"
done
echo ">>> PIPELINE_DONE $(date '+%Y-%m-%d %H:%M:%S')"
