#!/usr/bin/env bash
# PET-VALUE weighting of the latent velocity MSE: fine-tune BOTH trained 3D flow models.
#
# The arm (src/training/configs/ft3d_ft_iwlat.yaml, and the 2x twin) weights every position
# of the flow loss by the GT PET value pooled onto the 16^3 LATENT grid -- not on a
# full-resolution decode, so it costs nothing per step -- with an outlier-resistant
# percentile normalizer. Measured before launch (scripts/_diag_intensity_weight.py): the
# hot GT band's share of the gradient goes 2.4% -> 12.1% while air drops 80.6% -> 53.1%.
#
# TWO RUNS BY DEFAULT -- the arm and its matched control on the 64^3 base:
#   iwlat     / ctl     from ft3d_flow_p/best.pt   64^3,  ae3d_p     (the current best model)
# The 2x pair (128^3, from ft3d_2x/best.pt on ae3d_2x2) is OPT-IN, named explicitly:
#   bash scripts/_ft_iwlat.sh 2x_iwlat 2x_ctl
# It is not in the default set because the 2x chain is the weaker model (psnr 21.4 vs 25.4,
# reg_slope 0.665 vs 0.842 at s16) and, at 15k steps, is far likelier undertrained than
# mis-objectived -- so it is the wrong place to read a loss-shaping change.
#
# The controls are NOT optional. The earlier six-arm ablation ran 1500 steps at lr 1e-5 and
# every arm came back within 0.005 reg_slope / 0.02 dB of the baseline -- the fine-tune
# itself did nothing, so nothing was attributable. This runs 4x longer at the baseline's own
# 5e-5, which WILL move the model on its own; the control is what separates that motion from
# the arm's. The controls write to *_ft_ctl5e5 dirs so the 1e-5 ablation results survive.
#
# Every run reuses outputs/ft3d_flow_p/split.json (the partition both bases trained on), so
# all four and both baselines share one leak-free 41-patient test set and compare PAIRED.
#
# Usage:  bash scripts/_ft_iwlat.sh                       # the 64^3 pair
#         bash scripts/_ft_iwlat.sh 2x_iwlat 2x_ctl       # opt into the 128^3 pair
#         STEPS=3000 bash scripts/_ft_iwlat.sh iwlat      # override the step budget
#
# AUTO-RESUME: an arm whose save dir already holds last.pt is RESUMED from it (metrics
# append, the LR schedule fast-forwards), so an interrupted run continues instead of
# starting over. The flip side is the trap: to re-run an arm from scratch, move its save
# dir aside or pass NORESUME=1.
#
# Judge with scripts/_cmp_eval_paired.py against the matching baseline
# (outputs/eval/ft3d_flow_p/test_s16_clamped.json or outputs/eval/ft3d_2x/test_s16.json)
# on reg_slope / hot_band_rel_error / voxel_r2 / psnr. This arm is intensity-MONOTONE, so
# also look at hallucinated uptake -- report_triplets and the UNCLAMPED max_rel_error --
# before calling a slope gain a win.
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=/root/petct/.venv/bin/python
CACHE=/mnt/c/DeepTrainingData/PetCT
SPLIT=${SPLIT:-outputs/ft3d_flow_p/split.json}
LR=${LR:-5e-5}

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
[ -e "$SPLIT" ] || { echo "MISSING prerequisite: $SPLIT" >&2; exit 1; }

# arm -> config, save dir, init checkpoint, AE checkpoint, crop size, default steps
arm_cfg() {
  case "$1" in
    iwlat)    CFG=src/training/configs/ft3d_ft_iwlat.yaml;    SAVE=outputs/ft3d_ft_iwlat
              INIT=outputs/ft3d_flow_p/best.pt; AE=outputs/ae3d_p/best.pt
              SIZE=64;  DEF_STEPS=6000; CSIZE=32 ;;
    ctl)      CFG=src/training/configs/ft3d_ft_control.yaml;  SAVE=outputs/ft3d_ft_ctl5e5
              INIT=outputs/ft3d_flow_p/best.pt; AE=outputs/ae3d_p/best.pt
              SIZE=64;  DEF_STEPS=6000; CSIZE=32 ;;
    2x_iwlat) CFG=src/training/configs/ft3d_2x_ft_iwlat.yaml; SAVE=outputs/ft3d_2x_ft_iwlat
              INIT=outputs/ft3d_2x/best.pt;     AE=outputs/ae3d_2x2/best.pt
              SIZE=128; DEF_STEPS=4000; CSIZE=24 ;;
    2x_ctl)   CFG=src/training/configs/ft3d_2x.yaml;          SAVE=outputs/ft3d_2x_ft_ctl5e5
              INIT=outputs/ft3d_2x/best.pt;     AE=outputs/ae3d_2x2/best.pt
              SIZE=128; DEF_STEPS=4000; CSIZE=24 ;;
    *) echo "unknown arm: $1 (iwlat|ctl|2x_iwlat|2x_ctl)" >&2; exit 1 ;;
  esac
}

ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(iwlat ctl)
NORESUME=${NORESUME:-0}

for ARM in "${ARMS[@]}"; do
  arm_cfg "$ARM"
  N=${STEPS:-$DEF_STEPS}
  for f in "$CFG" "$INIT" "$AE"; do
    [ -e "$f" ] || { echo "MISSING prerequisite: $f" >&2; exit 1; }
  done
  # Continue an interrupted arm rather than restarting it. train_ft3d gives --resume
  # precedence over --init_from, so passing both is safe in either state.
  RESUME=()
  if [ "$NORESUME" != "1" ] && [ -f "$SAVE/last.pt" ]; then
    RESUME=(--resume)
    echo "RESUMING ${ARM} from ${SAVE}/last.pt"
  fi
  echo "================= ARM ${ARM}: ${N} steps @ lr ${LR}, ${SIZE}^3, init ${INIT}  $(date +%H:%M)"
  mkdir -p "$SAVE"
  "$PY" -m src.training.train.train_ft3d \
    --config "$CFG" --save_dir "$SAVE" \
    --data_dir "${ROOTS[@]}" --cache_dir "$CACHE" --prefetch 6 --cache_size "$CSIZE" \
    --ae_ckpt "$AE" --init_from "$INIT" --split_json "$SPLIT" "${RESUME[@]+"${RESUME[@]}"}" \
    --learning_rate "$LR" \
    --epochs $(( (N + 499) / 500 )) --steps_per_epoch 500 \
    --val_every 500 --val_batches 4 --latent_size "$SIZE" \
    --device cuda 2>&1 | tail -12

  # Eval on the SHARED split so every run is paired against the same 41 patients.
  "$PY" -m src.training.evaluate --task ft3d \
    --data_dir "${ROOTS[@]}" \
    --ae_ckpt "$AE" --diff_ckpt "$SAVE/best.pt" \
    --split_json "$SPLIT" --mode test \
    --size "$SIZE" --ddim_steps 16 --spacing linear --max_patients 0 --device cuda \
    --clamp_output --out "outputs/eval/$(basename "$SAVE")/test_s16.json" 2>&1 | tail -2
done

echo "================= iwlat runs done  $(date +%H:%M)"
