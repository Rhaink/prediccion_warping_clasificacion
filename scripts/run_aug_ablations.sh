#!/bin/bash
# Run all 8 augmentation ablation experiments sequentially
set -e
cd /home/donrobot/Projects/prediccion_warping_clasificacion
source .venv/bin/activate

CONFIGS=(
  "cv_aug_elastic"
  "cv_aug_grid"
  "cv_aug_pixel"
  "cv_aug_mixup"
  "cv_aug_cutmix"
  "cv_aug_elastic_curriculum"
  "cv_aug_grid_curriculum"
  "cv_aug_mixup_curriculum"
)

for cfg in "${CONFIGS[@]}"; do
  echo "=========================================="
  echo "Starting: $cfg at $(date)"
  echo "=========================================="
  python -m src_v2 cross-validate-classifier --config "configs/${cfg}.json"
  echo "Completed: $cfg at $(date)"
  echo ""
done

echo "ALL EXPERIMENTS COMPLETE at $(date)"
