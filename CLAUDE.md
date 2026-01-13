# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a deep learning research project for COVID-19 detection in chest X-rays using a two-stage approach:
1. **Landmark prediction**: ResNet-18 with Coordinate Attention predicts 15 anatomical landmarks on chest X-rays
2. **Geometric normalization**: Piecewise affine warping aligns images to a canonical pose using predicted landmarks
3. **Classification**: Multi-architecture CNN ensemble classifies normalized images (COVID-19 vs Normal vs Viral Pneumonia)

Key finding: Warping improves robustness to perturbations (JPEG compression, blur) and within-domain generalization by 2-6x, but does NOT solve domain shift between different institutions.

## Essential Commands

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (do this first)
pip install -r requirements.txt

# For AMD GPUs (ROCm):
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# For NVIDIA GPUs (CUDA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_classifier.py

# Run with coverage
pytest --cov=src_v2 --cov-report=html

# Run specific test function
pytest tests/test_warp.py::test_piecewise_affine_warp
```

### Landmark Prediction Workflow

```bash
# 1. Train landmark model (reproduces best results)
python -m src_v2 train --data-root data/ \
  --csv-path data/coordenadas/coordenadas_maestro.csv \
  --checkpoint-dir checkpoints_v2 \
  --phase1-epochs 15 --phase2-epochs 100 \
  --coord-attention --deep-head --hidden-dim 768 \
  --clahe --loss wing --seed 123

# 2. Evaluate single model with Test-Time Augmentation
python -m src_v2 evaluate checkpoints_v2/final_model.pt \
  --data-root data/ --csv-path data/coordenadas/coordenadas_maestro.csv \
  --tta --split test --clahe

# 3. Evaluate ensemble (reproduces 3.61 px error)
python -m src_v2 evaluate-ensemble \
  checkpoints/session10/ensemble/seed123/final_model.pt \
  checkpoints/session13/seed321/final_model.pt \
  checkpoints/repro_split111/session14/seed111/final_model.pt \
  checkpoints/repro_split666/session16/seed666/final_model.pt \
  --tta --clahe

# 4. Predict landmarks on a single image
python -m src_v2 predict data/dataset/COVID/COVID-100.png \
  --checkpoint checkpoints_v2/final_model.pt \
  --output outputs/prediction.png --clahe
```

### Classification Workflow

```bash
# 1. Generate warped dataset (96% fill rate - recommended)
python -m src_v2 generate-dataset \
  data/dataset/COVID-19_Radiography_Dataset \
  outputs/warped_lung_best/session_warping \
  --checkpoint checkpoints_v2/final_model.pt \
  --margin 1.05 --splits 0.75,0.125,0.125 --seed 42 \
  --use-full-coverage

# 2. Train classifier on warped images
python -m src_v2 train-classifier outputs/warped_lung_best/session_warping \
  --backbone resnet18 --epochs 50 --batch-size 32

# Or use config file:
python -m src_v2 train-classifier --config configs/classifier_warped_base.json

# 3. Evaluate classifier
python -m src_v2 evaluate-classifier outputs/classifier/best_classifier.pt \
  --data-dir outputs/warped_lung_best/session_warping --split test

# 4. Classify with warping (end-to-end inference)
python -m src_v2 classify image.png --classifier clf.pt \
  --warp --landmark-model checkpoints/final_model.pt
```

### Validation & Analysis

```bash
# Cross-dataset generalization (compare model A on dataset B)
python -m src_v2 cross-evaluate \
  outputs/classifier_original/best.pt \
  outputs/classifier_warped/best.pt \
  --data-a data/dataset/COVID-19_Radiography_Dataset \
  --data-b outputs/full_warped_dataset

# Test robustness (JPEG, blur, noise)
python -m src_v2 test-robustness \
  outputs/classifier/best.pt \
  --data-dir outputs/warped_lung_best/session_warping

# Grad-CAM visualization
python -m src_v2 gradcam --checkpoint outputs/classifier/best.pt \
  --data-dir outputs/warped_lung_best/session_warping/test \
  --output-dir outputs/gradcam_analysis --num-samples 20

# Error analysis with Grad-CAM
python -m src_v2 analyze-errors \
  --checkpoint outputs/classifier/best.pt \
  --data-dir outputs/warped_lung_best/session_warping/test \
  --output-dir outputs/error_analysis --visualize --gradcam
```

### Using Config Files

Config files (in `configs/`) provide defaults that can be overridden by CLI flags:

```bash
# Landmark training with config
python scripts/train.py --config configs/landmarks_train_base.json \
  --seed 123 --save-dir checkpoints/custom_run

# Classifier training with config
python -m src_v2 train-classifier --config configs/classifier_warped_base.json

# Ensemble evaluation with config
python scripts/evaluate_ensemble_from_config.py --config configs/ensemble_best.json
```

## Architecture Overview

### Code Structure

```
src_v2/                      # Main source code (v2 = modern Typer CLI)
├── models/
│   ├── resnet_landmark.py   # ResNet-18 + Coordinate Attention for landmark prediction
│   ├── classifier.py        # Multi-architecture CNN classifier (ResNet, EfficientNet, etc.)
│   ├── losses.py            # Wing Loss, Weighted Wing Loss, Combined Loss
│   └── hierarchical.py      # Hierarchical landmark model (experimental)
├── data/
│   ├── dataset.py           # LandmarkDataset, ClassificationDataset
│   ├── transforms.py        # CLAHE preprocessing, augmentations
│   └── utils.py             # Data utilities
├── training/
│   ├── trainer.py           # LandmarkTrainer (2-phase training)
│   └── callbacks.py         # EarlyStopping, ModelCheckpoint, LRScheduler
├── processing/
│   ├── warp.py              # Piecewise affine warping using Delaunay triangulation
│   └── gpa.py               # Generalized Procrustes Analysis for canonical shape
├── evaluation/
│   └── metrics.py           # Pixel error, classification metrics
├── visualization/
│   ├── gradcam.py           # Grad-CAM heatmaps
│   ├── pfs_analysis.py      # Pulmonary Focus Score analysis
│   └── error_analysis.py    # Classification error analysis
├── utils/
│   └── geometry.py          # Geometric utilities
├── constants.py             # Centralized constants (LANDMARK_NAMES, SYMMETRIC_PAIRS, etc.)
└── cli.py                   # Typer CLI commands

scripts/                     # Utility scripts (some legacy, use CLI when available)
configs/                     # JSON config templates
tests/                       # Unit and integration tests
docs/                        # Experiment documentation and reproduction guides
```

### Key Architecture Concepts

**Landmark Prediction Model (`src_v2/models/resnet_landmark.py`)**:
- **Backbone**: ResNet-18 pretrained on ImageNet, with final FC layer removed
- **Attention**: Coordinate Attention module (CVPR 2021) after backbone - captures spatial dependencies with positional information
- **Head**: Deep regression head with GroupNorm (not BatchNorm), 3 FC layers, dropout 0.3, hidden_dim 768
- **Output**: 30 normalized coordinates [0, 1] representing 15 landmarks (x, y)
- **Loss**: Wing Loss (optimized for facial/anatomical landmarks, handles small errors better than L1/L2)

**Two-Phase Training (`src_v2/training/trainer.py`)**:
1. **Phase 1** (15 epochs): Backbone frozen, train head only (LR=0.001)
2. **Phase 2** (100 epochs): Fine-tune all layers with differential learning rates (backbone LR=0.00002, head LR=0.0002)

**Geometric Warping (`src_v2/processing/warp.py`)**:
- Uses Delaunay triangulation over predicted landmarks to create a mesh
- Applies piecewise affine transformations triangle-by-triangle
- Adds 8 boundary points (4 corners + 4 edge midpoints) for ~99% fill rate (controlled by `--use-full-coverage`)
- Canonical shape computed via Generalized Procrustes Analysis (GPA) on training set

**15 Landmarks Structure** (`src_v2/constants.py`):
The landmarks define the **contour of the lungs**, NOT specific anatomical points:
- **Central axis** (L1, L9, L10, L11, L2): vertical line dividing the image (L1=superior, L2=inferior)
- **Left lung contour** (L12, L3, L5, L7, L14): 5 points on left boundary
- **Right lung contour** (L13, L4, L6, L8, L15): 5 points on right boundary
- **5 symmetric pairs**: (L3-L4, L5-L6, L7-L8, L12-L13, L14-L15)

**Classifier** (`src_v2/models/classifier.py`):
Supports multiple backbones (ResNet-18/50, EfficientNet-B0, DenseNet-121, VGG-16, MobileNetV2).
Best results: ResNet-18 on warped dataset with 96% fill rate → 99.10% accuracy.

## Important Implementation Details

### Data Splits
- Default split: 75% train, 12.5% val, 12.5% test (see `pyproject.toml`)
- Reproducible via `--seed` parameter
- **Critical**: The same seed must be used for landmark training and dataset generation to avoid data leakage

### CLAHE Preprocessing
Always enabled by default (`--clahe`):
- Clip limit: 2.0
- Tile size: 4x4
- Applied before ImageNet normalization
- Essential for consistent results

### Test-Time Augmentation (TTA)
- Enabled via `--tta` flag
- Applies horizontal flip, averages predictions
- Reduces error by ~0.3-0.4 px for landmark prediction

### Fill Rate Trade-off
Warping creates black pixels where no source pixels map:
- **47% fill rate**: Original lung crop (no boundary points) - best robustness to perturbations
- **96% fill rate**: With boundary points, margin=1.05 - RECOMMENDED (best accuracy + good robustness)
- **99% fill rate**: With boundary points, margin=1.15 - lower robustness
See `docs/sesiones/SESION_53_FILL_RATE_TRADEOFF.md` for full analysis.

### Config Override Pattern
Config files provide defaults, CLI flags always override:
```bash
python -m src_v2 train-classifier --config configs/classifier_warped_base.json \
  --epochs 100 --batch-size 64  # These override the config
```

### Model Checkpoints
Checkpoints contain:
- `model_state_dict`: Model weights
- `config`: All hyperparameters (architecture flags, loss type, etc.)
- Architecture is auto-detected during loading via state_dict keys

### GPU Support
- Auto-detection via `--device auto` (default)
- CUDA (NVIDIA): `--device cuda`
- ROCm (AMD): `--device cuda` (PyTorch treats ROCm as CUDA)
- MPS (Apple Silicon): `--device mps`

## Common Pitfalls & Solutions

### Issue: "Model checkpoint missing keys"
**Cause**: Architecture mismatch (e.g., loading a model with `--coord-attention` into one without)
**Solution**: The CLI auto-detects architecture from checkpoint. Don't manually specify `--coord-attention` when evaluating.

### Issue: High pixel error (>10 px) during evaluation
**Cause**: Missing `--clahe` flag
**Solution**: Always use `--clahe` during both training and evaluation

### Issue: Different results on same checkpoint
**Cause**: Different random seeds or TTA not applied consistently
**Solution**: Use `--tta` consistently, fix seed via `--seed`

### Issue: Warped dataset has poor fill rate (<50%)
**Cause**: Missing `--use-full-coverage` flag
**Solution**: Use `--use-full-coverage` in `generate-dataset` command

### Issue: Tests failing with "RuntimeError: DataLoader worker"
**Cause**: DataLoader multiprocessing issues in test environment
**Solution**: Tests set `FORCE_NUM_WORKERS_ZERO=1` env var automatically (see `pyproject.toml`)

### Issue: Low classifier accuracy on external data
**Expected behavior**: This is domain shift, not a bug. The model is trained on one institution's X-rays and generalizes poorly to different equipment/protocols. See Session 55 documentation in README.md for validation on FedCOVIDx dataset (~55% accuracy, near random).

## Reproducibility

**Best Landmark Ensemble (3.61 px error)**:
See `docs/REPRO_ENSEMBLE_3_71.md` for step-by-step reproduction.
Config: `configs/ensemble_best.json`

**Best Classifier (99.10% accuracy)**:
See `docs/REPRO_CLASSIFIER_RESNET18.md` for reproduction.
Config: `configs/classifier_warped_base.json`

**Quickstart Script**:
```bash
bash scripts/quickstart_landmarks.sh
```
Trains landmark model, generates predictions, and saves visualizations.

## Documentation

- `docs/REPRO_ENSEMBLE_3_71.md`: Reproduce landmark ensemble
- `docs/REPRO_CLASSIFIER_RESNET18.md`: Reproduce classifier
- `docs/QUICKSTART_LANDMARKS.md`: Quickstart guide
- `docs/CONFIGS.md`: Config file usage
- `docs/EXPERIMENTS.md`: Validated experiments index
- `docs/FISHER_EXPERIMENT_README.md`: Geometric validation (PCA + Fisher LDA)
- `docs/sesiones/`: Detailed session notes (e.g., Session 53 fill rate analysis)
- `README.md`: Full project overview with results tables

## Dataset Structure

**Landmark annotations** (`data/coordenadas/coordenadas_maestro.csv`):
- 957 manually annotated chest X-rays
- 15 landmarks per image (30 columns: x1, y1, x2, y2, ..., x15, y15)
- Coordinates in pixel space (299x299 original image)

**Image dataset** (`data/dataset/COVID-19_Radiography_Dataset/`):
```
COVID-19_Radiography_Dataset/
├── COVID/images/       # 306 images (32.0%)
├── Normal/images/      # 468 images (48.9%)
└── Viral Pneumonia/images/  # 183 images (19.1%)
```

**Warped dataset output** (generated by `generate-dataset`):
```
session_warping/
├── train/
│   ├── COVID/
│   ├── Normal/
│   └── Viral_Pneumonia/
├── val/
└── test/
```

## Version Information

- **Project version**: 2.1.0 (see `pyproject.toml`)
- **Python**: >=3.9 (tested on 3.12)
- **PyTorch**: >=2.0.0
- **NumPy**: >=2.0.0

CLI command to check version:
```bash
python -m src_v2 version
```

## Key Constants (`src_v2/constants.py`)

Always import constants instead of hardcoding:
```python
from src_v2.constants import (
    NUM_LANDMARKS,           # 15
    NUM_COORDINATES,         # 30
    DEFAULT_IMAGE_SIZE,      # 224
    ORIGINAL_IMAGE_SIZE,     # 299
    SYMMETRIC_PAIRS,         # [(2,3), (4,5), (6,7), (11,12), (13,14)]
    CENTRAL_LANDMARKS,       # [8, 9, 10] (L9, L10, L11)
    CLASSIFIER_CLASSES,      # ['COVID', 'Normal', 'Viral_Pneumonia']
    DEFAULT_WING_OMEGA,      # 10.0
    DEFAULT_WING_EPSILON,    # 2.0
)
```

## Limitations

- Small dataset (957 samples) - external validation recommended
- Domain shift between institutions - accuracy drops to ~55% on external data (FedCOVIDx)
- Warping improves **within-domain** robustness, NOT cross-domain generalization
- Model attention NOT focused on lung regions (PFS ≈ 50%) - robustness comes from geometric normalization
- Manual landmark annotations - inter-annotator variability not quantified
- NOT validated for clinical use - experimental research only
