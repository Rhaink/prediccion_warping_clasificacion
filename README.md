# COVID-19 Detection via Anatomical Landmarks and Geometric Normalization

Deep learning system for COVID-19 detection in chest X-rays using anatomical landmark prediction and piecewise affine warping for geometric normalization.

## Overview

This project implements a two-stage approach for COVID-19 classification:

1. **Landmark Prediction**: A ResNet-18 model with Coordinate Attention predicts 15 anatomical landmarks on chest X-rays
2. **Geometric Normalization**: Piecewise affine warping aligns images to a canonical pose using predicted landmarks
3. **Classification**: Multi-architecture ensemble classifies normalized images (COVID-19 vs Normal vs Viral Pneumonia)

## Key Results

| Metric | Value |
|--------|-------|
| Landmark Error (Ensemble + TTA) | **3.79 px** |
| Landmark Error Std | 2.49 px |
| Best Individual Model | 4.04 px |

### Per-Category Landmark Performance
- Normal: 3.53 px
- COVID-19: 3.83 px
- Viral Pneumonia: 4.42 px

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
- **Ensemble**: 2 models (seeds 123, 456) with Test-Time Augmentation

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
git clone https://github.com/YOUR_USERNAME/prediccion_warping_clasificacion.git
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

### Training

```bash
# Train landmark prediction model
python scripts/train.py

# Train classifier on warped images
python scripts/train_classifier.py

# Evaluate ensemble
python scripts/evaluate_ensemble.py
```

### Inference

```python
from src_v2.models import create_model

# Load model
model = create_model(
    num_landmarks=15,
    pretrained=True,
    use_coord_attention=True,
    deep_head=True
)
model.load_state_dict(torch.load('path/to/checkpoint.pt'))

# Predict landmarks
landmarks = model.predict_landmarks(image_tensor)  # Shape: (B, 15, 2)
```

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
