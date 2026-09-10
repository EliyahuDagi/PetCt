#!/usr/bin/env bash
# THE FLATNESS EXPERIMENT, end to end, SAFE TO RE-RUN AT ANY POINT.
#
# Runs the uptake-weighted arm, then its matched plain-loss control, scores both on the 41
# test patients, dumps predictions for the invented-uptake check, and pairs the two arms
# against each other. Every stage is skipped if its output already exists, and each arm
# continues from its own last.pt, so re-running this after an interruption picks up where
# it stopped instead of starting over.
#
# WHY THAT MATTERS HERE: this run was killed once at step 4183 of 6000, at 02:15 on
# 2026-09-09, by a Windows Update restart (System event 1074, MoUsoCoreWorker.exe, then a
# second one from TrustedInstaller.exe at 02:21). `setsid nohup` does NOT protect against
# that -- the host reboots, the WSL virtual machine goes with it, and every process inside
# dies. Nothing inside WSL can prevent it. The defence is being able to resume, which the
# trainer supports exactly: the checkpoint carries the optimizer, the learning-rate
# schedule position AND the sampler's random state (train_ft3d.py:1215 capture_rng_state /
# :1242 restore_rng_state), so a resumed arm draws the same patients, crops and bridge
# times it would have drawn had it never stopped. Validation uses a SEPARATE generator
# (train_ft3d.py:1095) and so never perturbs that sequence.
#
# That is what keeps the two arms comparable draw-for-draw even though one of them was
# interrupted and the other was not: the draw sequence is a function of the step count, not
# of how the run was chopped up.
#
# Usage (inside WSL), detached from this terminal:
#   setsid nohup bash scripts/_slab_flow_iwlat_chain.sh >> scratchpad/iwlat_chain.log 2>&1 </dev/null &
# Re-run the identical command after any interruption. Env: TRAIN_STEPS (default 6000),
# FORCE=1 to ignore existing outputs and redo every stage.
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=~/petct/.venv/bin/python
TRAIN_STEPS=${TRAIN_STEPS:-6000}
FORCE=${FORCE:-}
OUT=outputs/eval/ft3d_slab_flow

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

# Steps already completed in an arm's directory, or 0 if it has not started. Read from the
# checkpoint rather than the log so a truncated log cannot fake progress.
ckpt_step() {
  local d="$1"
  if [ ! -f "$d/last.pt" ]; then echo 0; return; fi
  "$PY" - "$d/last.pt" <<'EOF' 2>/dev/null || echo 0
import sys, torch
try:
    s = torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("step", 0)
    print(int(s))
except Exception:
    print(0)
EOF
}

run_arm() {
  local arm="$1" save="$2" tag="$3" dump="$4"
  if [ -z "$FORCE" ] && [ -f "$OUT/slab_${tag}_s16.json" ]; then
    echo ">>> SKIP $arm -- already scored ($OUT/slab_${tag}_s16.json) $(stamp)"
    return 0
  fi
  local done_steps; done_steps=$(ckpt_step "$save")
  echo ">>> ARM_START $arm at step $done_steps of $TRAIN_STEPS $(stamp)"
  if [ "$done_steps" -ge "$TRAIN_STEPS" ]; then
    # Training finished but scoring did not. Score only -- re-entering the trainer at or
    # past its total step count would either do nothing or trip its schedule.
    echo ">>> ARM_TRAIN_ALREADY_DONE $arm ($done_steps steps) $(stamp)"
    WHICH=slab STEPS=16 SAVE="$save" CKPT="$save/last.pt" TAG="$tag" \
      AE=outputs/ae3d_slab/best.pt DUMP="$dump" \
      bash scripts/_slab_flow_eval.sh
    echo ">>> ARM_END $arm rc=$? $(stamp)"
    return 0
  fi
  # The arm script auto-resumes when $save/last.pt exists and runs its own scoring.
  DUMP="$dump" TRAIN_STEPS="$TRAIN_STEPS" bash scripts/_slab_flow_iwlat.sh "$arm"
  echo ">>> ARM_END $arm rc=$? $(stamp)"
}

echo ">>> CHAIN_START $(stamp)"
run_arm iw  outputs/ft3d_slab_flow_ae3d_iwlat ae3d_iwlat outputs/eval/dump_ae3d_iwlat
run_arm ctl outputs/ft3d_slab_flow_ae3d_ctl   ae3d_ctl   outputs/eval/dump_ae3d_ctl

# The comparison the experiment turns on: weighted against the same amount of plain
# training, not against the older report numbers.
if [ -f "$OUT/slab_ae3d_iwlat_s16.json" ] && [ -f "$OUT/slab_ae3d_ctl_s16.json" ]; then
  echo ">>> CHAIN_STEP cmp $(stamp)"
  bash scripts/_slab_flow_iwlat.sh cmp
  # Invented uptake: the weight rises with true intensity, so it can reward painting
  # uptake into cold tissue. No eval metric can see that; this counts it.
  echo ">>> CHAIN_STEP false_hot $(stamp)"
  "$PY" scripts/_diag_false_hot.py --a outputs/eval/dump_ae3d_ctl --b outputs/eval/dump_ae3d_iwlat \
    --label_a matched_control --label_b uptake_weighted \
    | tee "$OUT/false_hot_iwlat_vs_ctl.txt"
else
  echo ">>> CHAIN_INCOMPLETE -- one arm did not finish; re-run this script $(stamp)"
fi
echo ">>> CHAIN_END $(stamp)"
