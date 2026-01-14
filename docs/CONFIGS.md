# Config Templates

This document describes the JSON templates under `configs/`.
All templates are flat key/value maps; keys must match CLI argument names.
CLI flags always override config values.

## Landmark Training
- File: configs/landmarks_train_base.json
- Script: scripts/train.py
- Example:
  - python scripts/train.py --config configs/landmarks_train_base.json \
    --seed 123 --split-seed 123 --save-dir checkpoints/... --output-dir outputs/...

## Landmark Ensemble
- File: configs/ensemble_best.json
- Script: scripts/evaluate_ensemble_from_config.py
- Example:
  - python scripts/evaluate_ensemble_from_config.py --config configs/ensemble_best.json

## Classifier (Warped)
- File: configs/classifier_warped_base.json
- Script: scripts/train_classifier.py (wrapper del CLI)
- Example:
  - python -m src_v2 train-classifier --config configs/classifier_warped_base.json
  - python scripts/train_classifier.py --config configs/classifier_warped_base.json

## Classifier (Original)
- File: configs/classifier_original_base.json
- Script: scripts/archive/classification/train_classifier_original.py (legacy)
- Example:
  - python scripts/archive/classification/train_classifier_original.py --config configs/classifier_original_base.json

## Hierarchical Landmark Model
- File: configs/hierarchical_train_base.json
- Script: scripts/train_hierarchical.py
- Example:
  - python scripts/train_hierarchical.py --config configs/hierarchical_train_base.json
