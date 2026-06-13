#!/usr/bin/env bash
# Complete the SSD pre-cache for the full NAC->AC pipeline root set.
# Idempotent: fresh entries are skipped, so this resumes the partial cache.
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt

PY=~/petct/.venv/bin/python
CACHE=/mnt/c/DeepTrainingData/PetCT

"$PY" -m src.training.precache \
  --data_dir \
    "/mnt/d/DeepTrainingData/Project/ACRIN 6668" \
    "/mnt/d/DeepTrainingData/Project/PetCt/Bladder 13.11.25" \
    "/mnt/d/DeepTrainingData/Project/PetCtNormal/PET_CT_NORMAL_2" \
  --cache_dir "$CACHE" \
  --workers 2
echo "PRECACHE_EXIT=$?"
