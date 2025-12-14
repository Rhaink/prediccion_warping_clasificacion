# COVID-19 Detection via Anatomical Landmarks and Geometric Normalization

Deep learning system for COVID-19 detection in chest X-rays using anatomical landmark prediction and piecewise affine warping for geometric normalization.

## Overview

This project implements a two-stage approach for COVID-19 classification:

1. **Landmark Prediction**: A ResNet-18 model with Coordinate Attention predicts 15 anatomical landmarks on chest X-rays
2. **Geometric Normalization**: Piecewise affine warping aligns images to a canonical pose using predicted landmarks
3. **Classification**: Multi-architecture ensemble classifies normalized images (COVID-19 vs Normal vs Viral Pneumonia)

## Key Results

### Landmark Prediction

| Metric | Value |
|--------|-------|
| Landmark Error (Ensemble 4 models + TTA) | **3.71 px** |
| Landmark Error Std | 2.42 px |
| Median Error | 3.17 px |
| Best Individual Model (TTA) | 4.04 px |

**Per-Category Landmark Performance (Test Split):**
- Normal: 3.42 px
- COVID-19: 3.77 px
- Viral Pneumonia: 4.40 px

### COVID-19 Classification

| Dataset | Accuracy | Fill Rate | Robustness (JPEG Q50) |
|---------|----------|-----------|----------------------|
| Original 100% | 98.84% | 100% | 16.14% |
| Warped 47% | 98.02% | 47% | 0.53% |
| Warped 99% | 98.73% | 99% | 7.34% |
| **Warped 96% (RECOMMENDED)** | **99.10%** | **96%** | **3.06%** |

> **Note:** Warped 96% achieves the best accuracy while maintaining 2.4x better robustness than Warped 99%. See [Session 53 documentation](docs/sesiones/SESION_53_FILL_RATE_TRADEOFF.md) for the fill rate trade-off analysis.

### Robustness to Perturbations (Validated - Sessions 39, 52-53)

| Model | Fill Rate | JPEG Q50 | JPEG Q30 | Blur sigma=1 |
|-------|-----------|----------|----------|--------------|
| Original 100% | 100% | 16.14% | 29.97% | 14.43% |
| Original Cropped 47% | 47% | 2.11% | 7.65% | 7.65% |
| **Warped 47%** | 47% | **0.53%** | **1.32%** | 6.06% |
| Warped 99% | 99% | 7.34% | 16.73% | 11.35% |
| **Warped 96% (RECOMMENDED)** | 96% | 3.06% | 5.28% | **2.43%** |

*Values represent accuracy degradation under perturbations (lower is better).*

**Key findings:**
- **JPEG Q50**: Warped 47% is **30x more robust** than Original (0.53% vs 16.14%)
- **JPEG Q30**: Warped 47% is **23x more robust** than Original (1.32% vs 29.97%)
- **Blur sigma=1**: Warped 96% is **5.9x more robust** than Original (2.43% vs 14.43%)
- **Best overall**: Warped 96% balances **highest accuracy (99.10%)** with strong robustness

### Cross-Dataset Generalization (Validated - Session 39)

| Model | On Original | On Warped 99% | Gap |
|-------|-------------|---------------|-----|
| Original | 98.84% | 91.13% | **7.70%** |
| Warped | 95.57% | 98.73% | **3.17%** |

**Ratio: 2.4x** - The warped model generalizes **2.4x better** than the original.

### Robustness Mechanism (Control Experiment - Session 39)

The control experiment with Original Cropped 47% revealed the causal mechanism:

| Component | Contribution | Evidence |
|-----------|--------------|----------|
| **Information reduction** | ~75% | Original Cropped is 7.6x more robust than Original 100% |
| **Geometric normalization** | ~25% additional | Warped 47% is 4x more robust than Original Cropped 47% |

**Conclusion:** Robustness primarily comes from implicit regularization via reduced fill rate, with additional contribution from geometric normalization.

## Architecture

### Landmark Prediction Model
- **Backbone**: ResNet-18 (pretrained on ImageNet)
- **Attention**: Coordinate Attention Module (CVPR 2021)
- **Head**: Deep regression head with GroupNorm
- **Output**: 15 landmarks (30 normalized coordinates)
- **Loss**: Wing Loss (optimized for facial/anatomical landmarks)

### Training Strategy
- **Phase 1**: Frozen backbone, train head only (15 epochs)
- **Phase 2**: Fine-tuning all layers with differential LR (100 epochs)
- **Ensemble**: 4 models (seeds 123, 456, 321, 789) with Test-Time Augmentation

## Project Structure

```
prediccion_warping_clasificacion/
├── src_v2/                    # Core source code
│   ├── data/                  # Dataset and transforms
│   ├── models/                # Neural network architectures
│   ├── training/              # Training logic and callbacks
│   └── evaluation/            # Metrics and evaluation
├── scripts/                   # Training, evaluation, visualization scripts
│   └── visualization/         # Figure generation for thesis
├── tests/                     # Unit tests
├── configs/                   # Configuration files
├── documentación/             # LaTeX documentation (Spanish)
├── data/                      # Datasets (not included, see below)
├── checkpoints/               # Trained models (not included)
└── outputs/                   # Experiment outputs (not included)
```

## Installation

```bash
# Clone repository
# Reemplaza <usuario> con tu nombre de usuario de GitHub o la URL de tu fork
git clone https://github.com/<usuario>/prediccion_warping_clasificacion.git
cd prediccion_warping_clasificacion

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For AMD GPUs (ROCm):
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# For NVIDIA GPUs (CUDA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Dataset

The project uses the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) with manual landmark annotations.

**Dataset statistics:**
- Total samples: 957
- COVID-19: 306 (32.0%)
- Normal: 468 (48.9%)
- Viral Pneumonia: 183 (19.1%)

**15 Anatomical Landmarks:**
1. Superior mediastinum
2. Inferior mediastinum
3. Left apex
4. Right apex
5. Left hilum
6. Right hilum
7. Left base
8. Right base
9. Superior central
10. Middle central
11. Inferior central
12. Left upper border
13. Right upper border
14. Left costophrenic angle
15. Right costophrenic angle

## Usage

### CLI Commands (src_v2)

The project includes a modern CLI built with Typer:

```bash
# View all available commands
python -m src_v2 --help

# Train landmark prediction model (reproduce best results)
python -m src_v2 train --data-root data/ \
  --csv-path data/coordenadas/coordenadas_maestro.csv \
  --checkpoint-dir checkpoints_v2 \
  --phase1-epochs 15 --phase2-epochs 100 \
  --coord-attention --deep-head --hidden-dim 768 \
  --clahe --clahe-clip 2.0 --clahe-tile 4 \
  --loss wing --seed 123

# Train with different loss function
python -m src_v2 train --loss weighted_wing  # or: combined

# Evaluate model on test set (default: test split, 10% of data)
python -m src_v2 evaluate checkpoints_v2/final_model.pt \
  --data-root data/ --csv-path data/coordenadas/coordenadas_maestro.csv \
  --tta --split test --clahe

# Evaluate on all data without TTA
python -m src_v2 evaluate checkpoints_v2/final_model.pt --split all --no-tta

# Predict landmarks on a single image
python -m src_v2 predict data/dataset/COVID/COVID-100.png \
  --checkpoint checkpoints_v2/final_model.pt \
  --output outputs/prediction.png \
  --clahe

# Apply geometric warping to a dataset
python -m src_v2 warp data/dataset/ outputs/warped/ \
  --checkpoint checkpoints_v2/final_model.pt \
  --clahe

# Evaluate ensemble of 4 models (reproduces 3.71 px)
python -m src_v2 evaluate-ensemble \
  checkpoints/session10/ensemble/seed123/final_model.pt \
  checkpoints/session10/ensemble/seed456/final_model.pt \
  checkpoints/session13/seed321/final_model.pt \
  checkpoints/session13/seed789/final_model.pt \
  --tta --clahe

# Show version
python -m src_v2 version

# --- Classification Commands ---

# Classify a single image (without warping)
python -m src_v2 classify image.png --classifier outputs/classifier/best_classifier.pt

# Classify with geometric normalization (warping)
python -m src_v2 classify image.png --classifier clf.pt \
  --warp --landmark-model checkpoints/final_model.pt

# Classify with ensemble landmarks + TTA (highest accuracy)
python -m src_v2 classify data/test/ --classifier clf.pt \
  --warp --landmark-ensemble m1.pt m2.pt m3.pt m4.pt --tta \
  --output results.json

# Train classifier on warped dataset
python -m src_v2 train-classifier outputs/warped_dataset \
  --backbone resnet18 --epochs 50 --batch-size 32

# Evaluate classifier
python -m src_v2 evaluate-classifier outputs/classifier/best_classifier.pt \
  --data-dir outputs/warped_dataset --split test

# --- Processing Commands ---

# Compute canonical shape using GPA (Generalized Procrustes Analysis)
python -m src_v2 compute-canonical data/coordenadas/coordenadas_maestro.csv \
  --output-dir outputs/shape_analysis --visualize

# Generate warped dataset with train/val/test splits
# --use-full-coverage (default) adds boundary points for ~99% fill rate
python -m src_v2 generate-dataset \
  data/COVID-19_Radiography_Dataset \
  outputs/warped_dataset \
  --checkpoint checkpoints_v2/final_model.pt \
  --margin 1.05 --splits 0.75,0.125,0.125 --seed 42 \
  --use-full-coverage

# --- Analysis & Validation Commands ---

# Cross-evaluation: Compare generalization of two models on two datasets
python -m src_v2 cross-evaluate \
  outputs/classifier_original/best.pt \
  outputs/classifier_warped/best.pt \
  --data-a data/COVID-19_Radiography_Dataset \
  --data-b outputs/full_warped_dataset \
  --output-dir outputs/cross_evaluation

# Evaluate on external binary dataset (FedCOVIDx/Dataset3)
python -m src_v2 evaluate-external \
  outputs/classifier/best.pt \
  --external-data outputs/external_validation/dataset3 \
  --output results.json

# Test robustness to perturbations (JPEG compression, blur, noise)
python -m src_v2 test-robustness \
  outputs/classifier/best.pt \
  --data-dir outputs/warped_dataset \
  --output robustness_results.json

# Compare multiple CNN architectures
python -m src_v2 compare-architectures outputs/warped_dataset \
  --architectures resnet18,efficientnet_b0,densenet121 \
  --epochs 30 --output-dir outputs/arch_comparison

# --- Explainability Commands ---

# Generate Grad-CAM visualizations (single image)
python -m src_v2 gradcam --checkpoint outputs/classifier/best.pt \
  --image test.png --output gradcam.png

# Generate Grad-CAM visualizations (batch mode)
python -m src_v2 gradcam --checkpoint outputs/classifier/best.pt \
  --data-dir outputs/warped_dataset/test \
  --output-dir outputs/gradcam_analysis --num-samples 20

# Analyze classification errors with optional Grad-CAM
python -m src_v2 analyze-errors \
  --checkpoint outputs/classifier/best.pt \
  --data-dir outputs/warped_dataset/test \
  --output-dir outputs/error_analysis \
  --visualize --gradcam

# Analyze Pulmonary Focus Score (PFS) - measures lung region attention
# NOTE: Analysis showed PFS ≈ 0.487 (~50%), indicating the model does NOT
# specifically focus on lung regions. The robustness improvement comes from
# geometric normalization (reduced fill rate), not from forced lung attention.
python -m src_v2 pfs-analysis \
  --checkpoint outputs/classifier/best.pt \
  --data-dir outputs/warped_dataset/test \
  --mask-dir data/COVID-19_Radiography_Dataset \
  --output-dir outputs/pfs_analysis

# Generate approximate lung masks (when segmentation masks unavailable)
python -m src_v2 generate-lung-masks \
  --data-dir outputs/warped_dataset \
  --output-dir outputs/lung_masks \
  --method rectangular --margin 0.15

# --- Optimization Commands ---

# Find optimal warping margin via grid search
python -m src_v2 optimize-margin \
  --data-dir data/COVID-19_Radiography_Dataset \
  --landmarks-csv data/landmarks.csv \
  --margins 1.00,1.05,1.10,1.15,1.20,1.25,1.30 \
  --epochs 10 --output-dir outputs/margin_optimization
```

### Key CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--clahe/--no-clahe` | True | CLAHE contrast enhancement |
| `--clahe-clip` | 2.0 | CLAHE clip limit |
| `--clahe-tile` | 4 | CLAHE tile grid size |
| `--loss` | wing | Loss function: wing, weighted_wing, combined |
| `--coord-attention` | True | Use Coordinate Attention |
| `--deep-head` | True | Use deep regression head |
| `--hidden-dim` | 768 | Hidden layer dimension |
| `--tta` | True | Test-Time Augmentation |
| `--split` | test | Data split: train, val, test, all |
| `--classifier` | - | Classifier checkpoint (for classify) |
| `--warp/--no-warp` | False | Apply geometric normalization |
| `--landmark-model` | - | Single landmark model for warping |
| `--landmark-ensemble` | - | Multiple landmark models for ensemble |
| `--backbone` | resnet18 | Classifier backbone: resnet18, efficientnet_b0 |
| `--use-full-coverage` | True | Add boundary points for ~99% fill rate |

### Legacy Scripts

```bash
# Train landmark prediction model (legacy)
python scripts/train.py

# Train classifier on warped images
python scripts/train_classifier.py

# Evaluate ensemble
python scripts/evaluate_ensemble.py
```

### Inference (Python API)

```python
import torch
from src_v2.models import create_model

# Load model (architecture auto-detected from checkpoint)
checkpoint = torch.load('path/to/checkpoint.pt')
state_dict = checkpoint['model_state_dict']

# Auto-detect architecture from state_dict keys
use_coord_attention = any('coord_attention' in k for k in state_dict.keys())
deep_head = 'head.9.weight' in state_dict
hidden_dim = state_dict['head.5.weight'].shape[0] if deep_head else 256

model = create_model(
    num_landmarks=15,
    pretrained=False,
    use_coord_attention=use_coord_attention,
    deep_head=deep_head,
    hidden_dim=hidden_dim
)
model.load_state_dict(state_dict)

# Predict landmarks
landmarks = model.predict_landmarks(image_tensor)  # Shape: (B, 15, 2)
```

**Note:** The CLI commands (`evaluate`, `predict`, `warp`) automatically detect the model architecture from the checkpoint.

## Preprocessing

- Input size: 299x299 (resized to 224x224)
- CLAHE enhancement (clip_limit=2.0, tile_size=4)
- ImageNet normalization

## Hardware

Tested on:
- AMD Radeon RX 6600 (8GB VRAM)
- PyTorch 2.0+ with ROCm

## Documentation

Full scientific documentation in `documentación/` (Spanish, LaTeX format):
- Data analysis
- Model architecture details
- Loss function derivations
- Training methodology
- Experimental results
- Geometric warping theory

## Limitations and Known Biases

### Dataset Limitations

- **Sample size**: Small dataset (957 samples) - suitable for thesis research, external validation recommended for broader conclusions
- **Demographic information**: Age and gender distribution of patients unknown
- **Equipment variation**: Images acquired with multiple radiological equipment types/brands
- **Geographic origin**: Data from multiple institutions/countries with varying protocols
- **Labeling**: Manual annotations - inter-annotator variability not quantified

### Model Limitations

- **Generalization**: Performance on different X-ray equipment/protocols may vary
- **External validation**: Not validated on independent external datasets beyond thesis scope
- **Demographic bias**: Unknown performance across different demographic groups
- **PFS Analysis**: Pulmonary Focus Score ≈ 50% indicates model attention is NOT specifically focused on lung regions

### Clinical Use Disclaimer

> **WARNING**: This model is experimental and developed for academic research purposes only.
> It is NOT validated for clinical decision-making and should NOT be used in clinical
> settings without proper regulatory approval (FDA, CE marking, etc.) and extensive
> external validation. Any clinical application requires supervision by qualified
> medical professionals.

## License

This project is part of a doctoral thesis. Please contact the author for licensing information.

## Citation

If you use this code, please cite:

```bibtex
@misc{covid19landmarks2024,
  title={COVID-19 Detection via Anatomical Landmarks and Geometric Normalization},
  author={},
  year={2024},
  howpublished={GitHub repository}
}
```

## Acknowledgments

- COVID-19 Radiography Database contributors
- PyTorch team
- Coordinate Attention paper authors
