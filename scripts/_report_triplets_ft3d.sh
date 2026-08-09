#!/usr/bin/env bash
# Run the held-out TEST set through the best 3D flow model (ft3d_flow_p) and build the
# NAC | predicted-AC | GT-AC triplet HTML report.
#   ae3d_p    : 3D AutoencoderKL on the fixed full-NAC+AC pool (628 patients)
#   ft3d_flow_p: 3D rectified-flow NAC->AC bridge on ae3d_p latents, 64^3, 16 flow steps
# Pass extra args through (e.g. --max_patients 2 --out_dir ... for a smoke run).
set -u
cd /mnt/c/Users/algo/VScodeProjects/PetCt/PetCt
PY=~/petct/.venv/bin/python

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

"$PY" scripts/report_triplets.py --task ft3d \
  --data_dir "${ROOTS[@]}" \
  --diff_ckpt outputs/ft3d_flow_p/best.pt \
  --ae_ckpt outputs/ae3d_p/best.pt \
  --split_json outputs/ft3d_flow_p/split.json \
  --mode test --val_fraction 0.2 --test_fraction 0.1 --seed 42 \
  --size 64 --ddim_steps 16 --spacing linear --max_patients 0 \
  --device cuda \
  --n_axial 64 --n_coronal 64 --n_sagittal 64 \
  --out_dir outputs/report/ft3d_flow_p \
  --label "ft3d_flow_p — 3D rectified-flow bridge" \
  "$@"
