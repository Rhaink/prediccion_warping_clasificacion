# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

COVID-19 detection system using chest X-rays with anatomical landmark prediction and geometric normalization. Two-stage approach:
1. **Landmark Prediction**: ResNet-18 with Coordinate Attention predicts 15 anatomical landmarks
2. **Geometric Normalization**: Piecewise affine warping aligns images to canonical pose
3. **Classification**: Multi-architecture ensemble classifies normalized images

## Commands

### CLI (Primary Interface)
```bash
# All commands via src_v2 module
python -m src_v2 --help

# Training
python -m src_v2 train --data-root data/ --csv-path data/coordenadas/coordenadas_maestro.csv

# Evaluation
python -m src_v2 evaluate checkpoints/model.pt --tta --clahe
python -m src_v2 evaluate-ensemble model1.pt model2.pt model3.pt model4.pt --tta

# Prediction & Warping
python -m src_v2 predict image.png --checkpoint model.pt
python -m src_v2 warp data/dataset/ outputs/warped/ --checkpoint model.pt

# Classification
python -m src_v2 classify image.png --classifier clf.pt --warp --landmark-model model.pt
python -m src_v2 train-classifier outputs/warped_dataset --backbone resnet18

# Dataset Generation
python -m src_v2 generate-dataset data/ outputs/warped/ --checkpoint model.pt --use-full-coverage
```

### Testing
```bash
pytest                           # Run all tests
pytest tests/test_losses.py      # Single file
pytest -k "test_wing_loss"       # Single test by name
pytest -v --tb=short             # Verbose with short traceback
```

### Package Installation
```bash
pip install -e .                 # Editable install
covid-landmarks --help           # CLI after install
```

## Architecture

### Source Structure
```
src_v2/
├── cli.py              # Typer-based CLI (entry point: python -m src_v2)
├── constants.py        # All domain constants (landmarks, losses, training params)
├── models/
│   ├── resnet_landmark.py  # ResNet-18 + Coordinate Attention backbone
│   ├── classifier.py       # ImageClassifier for COVID classification
│   ├── losses.py           # WingLoss, WeightedWingLoss, CombinedLandmarkLoss
│   └── hierarchical.py     # Hierarchical landmark model (experimental)
├── data/
│   ├── dataset.py      # LandmarkDataset with CLAHE preprocessing
│   └── transforms.py   # Augmentations (flip, rotation, CLAHE)
├── training/
│   ├── trainer.py      # Two-phase training (frozen backbone → fine-tuning)
│   └── callbacks.py    # EarlyStopping, ModelCheckpoint
├── processing/
│   ├── warp.py         # Piecewise affine warping implementation
│   └── gpa.py          # Generalized Procrustes Analysis for canonical shape
├── evaluation/
│   └── metrics.py      # Landmark error, per-category metrics
└── visualization/
    ├── gradcam.py      # Grad-CAM visualizations
    └── pfs_analysis.py # Pulmonary Focus Score analysis
```

### Key Domain Concepts

**15 Anatomical Landmarks** define lung contour (not specific anatomy):
- Vertical axis: L1 (superior) → L9, L10, L11 → L2 (inferior)
- Left contour: L12 → L3 → L5 → L7 → L14
- Right contour: L13 → L4 → L6 → L8 → L15
- 5 symmetric pairs: (L3-L4), (L5-L6), (L7-L8), (L12-L13), (L14-L15)

**Training Pipeline**:
1. Phase 1: Frozen ResNet backbone, train head only (15 epochs)
2. Phase 2: Fine-tune all layers with differential LR (100 epochs)

**Warping Pipeline**:
1. Predict landmarks → 2. Match to canonical shape (from GPA) → 3. Delaunay triangulation → 4. Piecewise affine transform

### Important Constants (from `src_v2/constants.py`)
- `NUM_LANDMARKS = 15`
- `DEFAULT_IMAGE_SIZE = 224`
- `OPTIMAL_MARGIN_SCALE = 1.05` (for warping)
- `CLASSIFIER_CLASSES = ['COVID', 'Normal', 'Viral_Pneumonia']`

## Key Files

- `configs/final_config.json` - Production training configuration
- `data/coordenadas/coordenadas_maestro.csv` - Ground truth landmark annotations
- `GROUND_TRUTH.json` - Validated experimental results (use for visualization scripts)

## Testing Patterns

Fixtures in `tests/conftest.py`:
- `sample_landmarks` - Normalized landmark array (15, 2)
- `untrained_model` / `pretrained_model` - Model fixtures
- `mock_landmark_checkpoint` / `mock_classifier_checkpoint` - Checkpoint fixtures
- `minimal_landmark_dataset` - Minimal dataset for integration tests

## Known Limitations

- **External validation**: Models achieve ~55% on FedCOVIDx (domain shift issue, not a warping problem)
- **PFS ≈ 50%**: Model attention is not specifically focused on lung regions
- Warping improves within-domain robustness but does NOT solve cross-institution generalization
