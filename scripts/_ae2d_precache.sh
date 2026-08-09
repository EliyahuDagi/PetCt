#!/usr/bin/env bash
# One-time SSD warm of the FULL pooled PET dataset (all 10 roots) before the ae2d
# embedding-size ablation. The raw data lives on a spinning HDD as many small DICOM
# files; decoding one patient is ~5-8 s of random I/O. precache decodes +
# percentile-normalizes each patient's PET ONCE and writes a compact .npz to the
# NVMe SSD, so the repeated ablation training runs read PET volumes back in ~ms.
#
# Run this ONCE before scripts/_ae2d_ablation.sh. It is resumable: fresh, in-version
# cache entries are skipped, so re-running only fills in new/changed patients
# (pass --force inside precache to rebuild everything).
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=~/petct/.venv/bin/python
CACHE=/mnt/c/DeepTrainingData/PetCT

R1="/mnt/d/DeepTrainingData/Project/ACRIN 6668"
R2="/mnt/d/DeepTrainingData/Project/PetCt/Bladder 13.11.25"
R3="/mnt/d/DeepTrainingData/Project/PetCtNormal/PET_CT_NORMAL_2"
R4="/mnt/d/DeepTrainingData/Project/NSCLC_Radiogenomics"
R5="/mnt/d/DeepTrainingData/Project/TCGA-LUAD"
R6="/mnt/d/DeepTrainingData/Project/CPTAC-LUAD"
R7="/mnt/d/DeepTrainingData/Project/CPTAC-LSCC"
R8="/mnt/d/DeepTrainingData/Project/CPTAC-UCEC"
R9="/mnt/d/DeepTrainingData/Project/CPTAC-PDA"
R10="/mnt/d/DeepTrainingData/Project/TCGA-THCA"

echo ">>> PRECACHE_START $(date '+%Y-%m-%d %H:%M:%S')"
"$PY" -m src.training.precache \
  --data_dir "$R1" "$R2" "$R3" "$R4" "$R5" "$R6" "$R7" "$R8" "$R9" "$R10" \
  --cache_dir "$CACHE" --workers 4
rc=$?
echo ">>> PRECACHE_DONE rc=$rc $(date '+%Y-%m-%d %H:%M:%S')"
exit $rc
