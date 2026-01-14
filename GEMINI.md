# GEMINI.md - Project Context & Usage Guide

## Project Overview

**Name:** `prediccion_warping_clasificacion`
**Purpose:** A comprehensive pipeline for COVID-19 detection from Chest X-Rays using anatomical landmark detection and geometric normalization (warping).
**Core Logic:**
1.  **Landmark Detection:** Predicts anatomical landmarks on lungs using CNNs (ResNet18 backbone + custom heads).
2.  **Geometric Normalization:** Warps lung images to a canonical shape using Piecewise Affine Transformation to reduce anatomical variance.
3.  **Classification:** Classifies the warped lung images into categories (COVID-19, Normal, Viral Pneumonia).

## Architecture & Structure

### Directory Layout
*   **`src_v2/`**: Core source code.
    *   `cli.py`: Main CLI entry point (`covid-landmarks` / `python -m src_v2`).
    *   `models/`: Model definitions (Landmark Detector, Classifier), losses (WingLoss), and factories.
    *   `data/`: Dataset handling (`LandmarkDataset`, `ImageFolder`), transforms, and loading utilities.
    *   `processing/`: Image processing logic (GPA, warping, CLAHE).
    *   `training/`: Training loops and trainers.
    *   `evaluation/`: Metric calculations and evaluation routines.
*   **`configs/`**: JSON configuration files for reproducible experiments (e.g., `classifier_warped_base.json`).
*   **`scripts/`**: Utility scripts for batch processing, reproduction, and analysis.
*   **`tests/`**: Unit and integration tests (`pytest`).
*   **`docs/`**: Documentation and experiment logs (`REPRO_FULL_PIPELINE.md` is critical).
*   **`data/`, `outputs/`, `checkpoints/`**: Local directories for datasets, model artifacts, and results (git-ignored).

### Key Technologies
*   **Language:** Python 3.9+
*   **Deep Learning:** PyTorch 2.0+, Torchvision
*   **Image Processing:** OpenCV, Pillow, Scikit-image (warp)
*   **Data/Math:** NumPy, Pandas, SciPy
*   **CLI:** Typer
*   **Testing:** PyTest

## Key Workflows & CLI Usage

The project uses a unified CLI accessed via `python -m src_v2 <command>`.

### 1. Landmark Detection
**Training:**
```bash
python -m src_v2 train --config configs/landmarks_train_base.json
# Or with manual flags:
python -m src_v2 train --data-root data/ --csv-path data/coordenadas/coordenadas_maestro.csv --loss wing --clahe
```

**Evaluation:**
```bash
python -m src_v2 evaluate --checkpoint <path/to/model.pt> --split test --tta
```

**Ensemble Evaluation:**
```bash
python -m src_v2 evaluate-ensemble <ckpt1.pt> <ckpt2.pt> ... --tta
```

**Inference (Single Image):**
```bash
python -m src_v2 predict --image xray.png --checkpoint model.pt --output result.png
```

### 2. Geometric Normalization (Warping)
**Generate Canonical Shape (GPA):**
```bash
python -m src_v2 compute-canonical data/coordenadas/coordenadas_maestro.csv --output-dir outputs/shape_analysis
```

**Warp Dataset:**
```bash
python -m src_v2 generate-dataset --config configs/warping_best.json
# Or CLI warping command:
python -m src_v2 warp --input-dir data/images --output-dir warped/ --checkpoint model.pt
```

### 3. Classification
**Train Classifier:**
```bash
python -m src_v2 train-classifier --config configs/classifier_warped_base.json
```

**Evaluate Classifier:**
```bash
python -m src_v2 evaluate-classifier <classifier.pt> --data-dir <warped_data_dir> --split test
```

**Classify Image (End-to-End):**
```bash
python -m src_v2 classify image.png --classifier clf.pt --warp --landmark-model lm.pt
```

## Configuration (JSON)
Experiments are driven by JSON configs in `configs/`. Key parameters:
*   **Landmarks:** `loss_type` (wing/weighted_wing), `hidden_dim`, `use_coord_attention`, `deep_head`.
*   **Warping:** `margin_scale` (e.g., 1.05), `use_clahe`, `clahe_clip`.
*   **Classifier:** `backbone` (resnet18), `lr`, `batch_size`, `use_class_weights`.

## Development Guidelines

1.  **Code Style:** Follow PEP 8. Use `pytest` for testing.
2.  **Conventions:**
    *   Use `pathlib.Path` for file operations.
    *   Logging is configured in `src_v2/__init__.py` and `cli.py`.
    *   `src_v2` is the source of truth; avoid modifying `scripts/` for core logic.
3.  **Reproducibility:** Always set seeds (default 42). Use configuration files for complex runs.
4.  **Testing:** Run `pytest` before committing changes.
    ```bash
    pytest tests/
    ```

## Critical References
*   **`docs/REPRO_FULL_PIPELINE.md`**: The definitive guide for reproducing the current best results.
*   **`docs/EXPERIMENTS.md`**: Log of past experiments and findings.
*   **`FISHER_EXPERIMENT_README.md`**: Details on Fisher Linear Analysis experiments.
