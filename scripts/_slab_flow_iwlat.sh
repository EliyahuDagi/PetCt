#!/usr/bin/env bash
# THE FLATNESS EXPERIMENT on the slab chain: fine-tune the finished 3D flow bridge with the
# latent velocity error WEIGHTED BY TRUE UPTAKE, against a MATCHED plain-loss control.
#
#   iw   src/training/configs/ft3d_slab_flow_ae3d_iwlat.yaml  -> outputs/ft3d_slab_flow_ae3d_iwlat
#   ctl  src/training/configs/ft3d_slab_flow_ae3d.yaml        -> outputs/ft3d_slab_flow_ae3d_ctl
#
# Everything else is byte-for-byte identical between the two: the same starting checkpoint,
# the same frozen autoencoder, the same patient split, the same number of steps, the same
# learning rate, the same ramp and cosine decay, the same validation cadence. The ONLY
# difference is which config file is read, and those two files differ only in the weighting
# block (verified by diffing the parsed YAML).
#
# The two runs are in lockstep, not merely matched. Both configs carry seed 42, both start
# their step counter at 0, and the training draws come from a generator seeded once from
# that seed (train_ft3d.py:774), so the two arms see the same patients, the same 16-slice
# crops and the same sampled bridge times in the same order. Building the weight map uses
# no randomness of its own, so the sequences cannot drift apart. Whatever separates the two
# score sheets at the end is the weighting and nothing else.
#
# Why a control is not optional: the last 15000 steps of plain training moved this model by
# +0.213 dB and 0.009 of slope on their own. An arm run without a control would measure
# "more training" and call it "weighting". See memory/ft-ablation-inert.md -- the earlier
# 1500-step 1e-5 recipe was inert, so this runs 6000 steps at the chain's own 5e-5.
#
# Both arms start from the FINISHED chain's averaged weights via --init_from: fresh
# optimizer, fresh 500-step ramp, fresh cosine decay, step counter back at 0. Step 0 of both
# runs is therefore the exact model whose test scores are in docs/PROJECT_REPORT.md.
#
# Usage (inside WSL), one arm at a time -- they share one 24 GB card:
#   setsid nohup bash scripts/_slab_flow_iwlat.sh iw  >> scratchpad/iwlat_iw.log  2>&1 </dev/null &
#   setsid nohup bash scripts/_slab_flow_iwlat.sh ctl >> scratchpad/iwlat_ctl.log 2>&1 </dev/null &
# Env overrides: TRAIN_STEPS (default 6000), INIT (the starting checkpoint), AE, CACHE,
# NORESUME=1 (start over instead of continuing an interrupted arm), EVAL=0 (skip scoring).
# NOT "STEPS": scripts/_slab_flow_eval.sh reads that name as the number of SAMPLING steps,
# so a training budget leaking into it would score the model with 6000-step rollouts.
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=~/petct/.venv/bin/python
CACHE=${CACHE:-/root/petct_cache}
AE=${AE:-outputs/ae3d_slab/best.pt}
# The finished chain: 20000 inflated steps + 20000 continued steps, the arm scored in the
# report. Its averaged weights are the start of both arms here.
INIT=${INIT:-outputs/ft3d_slab_flow_ae3d_cont/last.pt}
TRAIN_STEPS=${TRAIN_STEPS:-6000}
EVAL=${EVAL:-1}
NORESUME=${NORESUME:-}

ARM=${1:-}
case "$ARM" in
  iw)  CFG=src/training/configs/ft3d_slab_flow_ae3d_iwlat.yaml
       SAVE=outputs/ft3d_slab_flow_ae3d_iwlat; TAG=ae3d_iwlat ;;
  ctl) CFG=src/training/configs/ft3d_slab_flow_ae3d.yaml
       SAVE=outputs/ft3d_slab_flow_ae3d_ctl;   TAG=ae3d_ctl ;;
  cmp) # Pair the two finished arms against EACH OTHER -- the comparison the experiment
       # turns on. Each arm is already paired against the 2D chain by its own run, but that
       # only says "better than the old baseline"; this one says "better than the same
       # amount of plain training", which is the question.
       A=outputs/eval/ft3d_slab_flow/slab_ae3d_ctl_s16.json
       B=outputs/eval/ft3d_slab_flow/slab_ae3d_iwlat_s16.json
       for f in "$A" "$B"; do
         if [ ! -f "$f" ]; then echo ">>> MISSING $f -- run both arms first"; exit 2; fi
       done
       "$PY" scripts/_cmp_eval_paired.py "$A" "$B" \
         --label-a matched_control --label-b uptake_weighted \
         | tee outputs/eval/ft3d_slab_flow/paired_iwlat_vs_ctl_s16.txt
       exit 0 ;;
  *)   echo "usage: bash scripts/_slab_flow_iwlat.sh iw|ctl|cmp"; exit 2 ;;
esac

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

for f in "$CFG" "$AE" "$INIT" outputs/ft3d_flow_p/split.json; do
  if [ ! -f "$f" ]; then echo ">>> MISSING $f -- abort"; exit 2; fi
done

# An interrupted arm continues from its own last.pt (keeping its schedule) unless NORESUME=1.
RESUME=
if [ -z "$NORESUME" ] && [ -f "$SAVE/last.pt" ]; then RESUME=1; fi

mkdir -p "$SAVE"
if [ -z "$RESUME" ]; then : > "$SAVE/train.log"; fi
# 500 steps per epoch so validation lands every 500: 12 points over 6000 steps, enough to
# see whether the curves are still moving when the budget runs out.
EPOCHS=$(( (TRAIN_STEPS + 499) / 500 ))
echo ">>> STAGE_START slab_iwlat_$ARM steps=$TRAIN_STEPS init=$INIT${RESUME:+ (resume from $SAVE/last.pt)} $(date '+%Y-%m-%d %H:%M:%S')"
"$PY" -m src.training.train.launcher ft3d \
  --config "$CFG" \
  --data_dir "${ROOTS[@]}" --cache_dir "$CACHE" --device cuda \
  --val_fraction 0.2 --test_fraction 0.1 \
  --split_json outputs/ft3d_flow_p/split.json \
  --epochs "$EPOCHS" --steps_per_epoch 500 --val_every 500 --val_batches 8 \
  --latent_size 128 \
  --ae_ckpt "$AE" \
  --init_from "$INIT" \
  --save_dir "$SAVE" --cache_size 128 ${RESUME:+--resume} \
  >> "$SAVE/train.log" 2>&1
rc=$?; echo ">>> STAGE_END slab_iwlat_${ARM}_train rc=$rc $(date '+%Y-%m-%d %H:%M:%S')"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

# Score the finished arm on the 41 shared test patients and pair it against the 2D chain,
# exactly as every other arm in the report was scored.
if [ "$EVAL" = "1" ]; then
  echo ">>> STAGE_START slab_iwlat_${ARM}_eval $(date '+%Y-%m-%d %H:%M:%S')"
  WHICH=slab STEPS=16 SAVE="$SAVE" CKPT="$SAVE/last.pt" TAG="$TAG" AE="$AE" \
    bash scripts/_slab_flow_eval.sh
  echo ">>> STAGE_END slab_iwlat_${ARM}_eval rc=$? $(date '+%Y-%m-%d %H:%M:%S')"
fi
