# Reproducibility Guide

This document provides step-by-step instructions to reproduce the main result: **3.71 px** landmark prediction error using an ensemble of 4 models with Test-Time Augmentation.

## Prerequisites

1. Python 3.10+
2. PyTorch 2.0+ (with CUDA or ROCm support)
3. Project dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

## Required Checkpoints

The 4 model checkpoints needed for the ensemble:

```
checkpoints/
├── session10/
│   └── ensemble/
│       ├── seed123/final_model.pt   # 4.05 px individual
│       └── seed456/final_model.pt   # 4.04 px individual
└── session13/
    ├── seed321/final_model.pt       # ~4.0 px individual
    └── seed789/final_model.pt       # ~4.0 px individual
```

## Reproducing the Result

### Option 1: Using CLI (Recommended)

```bash
# Activate virtual environment
source .venv/bin/activate

# Run ensemble evaluation
python -m src_v2 evaluate-ensemble \
    checkpoints/session10/ensemble/seed123/final_model.pt \
    checkpoints/session10/ensemble/seed456/final_model.pt \
    checkpoints/session13/seed321/final_model.pt \
    checkpoints/session13/seed789/final_model.pt \
    --tta --clahe --split test
```

Expected output:
```
Error promedio: 3.71 px
Error mediana:  3.17 px
Error std:      2.42 px
```

### Option 2: Using Legacy Script

```bash
python scripts/evaluate_ensemble.py --exclude-42
```

Note: The legacy script was designed for the original 3-model ensemble (excluding seed=42). The CLI command above uses the improved 4-model ensemble.

## Configuration Details

### Model Architecture
- Backbone: ResNet-18 (pretrained on ImageNet)
- Attention: Coordinate Attention Module
- Head: Deep regression head with GroupNorm
- Hidden dimension: 768
- Dropout: 0.3

### Preprocessing
- Input size: 299x299 (resized to 224x224)
- CLAHE: enabled (clip_limit=2.0, tile_size=4)
- Normalization: ImageNet mean/std

### Evaluation Settings
- Test split: 10% of data (96 samples), stratified by category
- TTA: Horizontal flip with landmark pair swapping
- Split seed: 42 (fixed for reproducibility)

## Result Comparison

| Configuration | Error (px) | Notes |
|---------------|------------|-------|
| Ensemble 4 models + TTA | **3.71** | Best result |
| Ensemble 2 models + TTA | 3.79 | Seeds 123, 456 only |
| Ensemble 3 models + TTA | 4.50 | Includes seed 42 (degrades) |
| Best individual + TTA | 4.04 | Seed 456 |

## Per-Category Results

| Category | Error (px) | Samples |
|----------|------------|---------|
| Normal | 3.42 | 47 |
| COVID-19 | 3.77 | 31 |
| Viral Pneumonia | 4.40 | 18 |

## Troubleshooting

### CUDA/ROCm Issues
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU evaluation
python -m src_v2 evaluate-ensemble ... --device cpu
```

### Missing Dependencies
```bash
pip install typer tqdm scikit-learn pandas opencv-python-headless
```

### Wrong CLAHE Settings
Ensure CLAHE is enabled and matches training settings:
```bash
--clahe --clahe-clip 2.0 --clahe-tile 4
```

## Training from Scratch

To train new models matching the ensemble configuration:

```bash
for seed in 123 456 321 789; do
    python -m src_v2 train \
        --seed $seed \
        --checkpoint-dir checkpoints/seed${seed} \
        --phase1-epochs 15 --phase2-epochs 100 \
        --coord-attention --deep-head --hidden-dim 768 \
        --clahe --clahe-clip 2.0 --clahe-tile 4 \
        --loss wing
done
```

## Verification

To verify the installation and configuration are correct:

```bash
# Run tests
pytest tests/test_cli.py -v -k ensemble

# Check command help
python -m src_v2 evaluate-ensemble --help

# Quick test with 2 models (should give ~3.79 px)
python -m src_v2 evaluate-ensemble \
    checkpoints/session10/ensemble/seed123/final_model.pt \
    checkpoints/session10/ensemble/seed456/final_model.pt \
    --tta --clahe
```
