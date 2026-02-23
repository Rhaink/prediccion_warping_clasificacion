# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project for COVID-19 detection from chest X-rays using:
1. Anatomical landmark detection (15 lung contour landmarks)
2. Geometric normalization via piecewise affine warping
3. CNN classification on normalized images

Current validated results (GROUND_TRUTH.json v2.2.0):
- Landmark ensemble error: 3.61 px (on 224x224 images)
- Classifier ensemble+TTA accuracy: 98.26% (5-fold CV weighted soft voting, 47% error reduction over individual baseline)
- Single best classifier (warped_lung_best): 98.05%
- Test set: 1,895 images (COVID=452, Normal=1,274, Viral_Pneumonia=169)

## Development Commands

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# For development with tests:
pip install -e ".[dev]"
```

### Main CLI Commands
The primary interface is `python -m src_v2`. Key commands:

```bash
# 1. Compute canonical shape via GPA
python -m src_v2 compute-canonical data/coordenadas/coordenadas_maestro.csv \
  --output-dir outputs/shape_analysis --visualize

# 2. Generate landmark predictions for entire dataset (cache)
python scripts/predict_landmarks_dataset.py \
  --input-dir data/dataset/COVID-19_Radiography_Dataset \
  --output outputs/landmark_predictions/session_warping/predictions.npz \
  --ensemble-config configs/ensemble_best.json \
  --tta --clahe --clahe-clip 2.0 --clahe-tile 4

# 3. Generate warped dataset (using cached predictions)
python -m src_v2 generate-dataset --config configs/warping_best.json

# 4. Train classifier on warped images
python -m src_v2 train-classifier --config configs/classifier_warped_base.json

# 5. Evaluate classifier
python -m src_v2 evaluate-classifier \
  outputs/classifier_warped_lung_best/best_classifier.pt \
  --data-dir outputs/warped_lung_best/session_warping --split test

# 6. Evaluate landmark ensemble
python scripts/evaluate_ensemble_from_config.py --config configs/ensemble_best.json

# 6b. Generate landmark visualization dataset
python -m src_v2 generate-landmark-visualization-dataset \
  --config configs/landmark_viz_best.json

# Verify alignment with warped dataset
python scripts/verify_landmark_viz_dataset.py
```

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ -v --cov=src_v2 --cov-report=html

# Run specific test file
python -m pytest tests/test_processing.py -v

# Run tests matching pattern
python -m pytest tests/ -k "test_warp" -v
```

### GUI Demo
```bash
# Launch Gradio demo (checks dependencies and models before launching)
python scripts/run_demo.py
python scripts/run_demo.py --share  # Public link
python scripts/run_demo.py --port 8080  # Custom port
```

The GUI (`src_v2/gui/`) provides a Gradio-based interactive demo with:
- Full pipeline visualization (original → landmarks → warped → GradCAM)
- Quick classification mode
- PDF export of results
- Supports both development mode and PyInstaller standalone

### Quick Start Scripts
```bash
# Automated warping pipeline (canonical + predictions + warping)
nohup bash scripts/quickstart_warping.sh > outputs/warping_quickstart.log 2>&1 &

# Landmarks quickstart (train + evaluate + extract)
bash scripts/quickstart_landmarks.sh
```

## Architecture & Pipeline

### Pipeline Flow
1. **Canonical Shape (GPA)**: `src_v2/processing/gpa.py::gpa_iterative()` aligns training landmarks to compute consensus shape via Generalized Procrustes Analysis
2. **Landmark Prediction**: Ensemble of 4 ResNet-18 models predicts 15 (x,y) coordinates per image
3. **Warping**: `src_v2/processing/warp.py::piecewise_affine_warp()` normalizes geometry using Delaunay triangulation
4. **Classification**: 5-fold CV ensemble of ResNet-18 classifiers trained on warped images, combined via weighted soft voting with dual-level TTA

### Key Modules

**src_v2/models/**
- `resnet_landmark.py::ResNet18Landmarks`: Landmark detection model with optional Coordinate Attention
- `classifier.py::ImageClassifier`: Multi-class COVID/Normal/Viral Pneumonia classifier
- `hierarchical.py::HierarchicalLandmarkModel`: Alternative hierarchical approach
- `losses.py`: Wing Loss (landmark regression), Combined Loss (with symmetry penalties)

**src_v2/processing/**
- `gpa.py`: Generalized Procrustes Analysis for canonical shape
  - `gpa_iterative()`: Align and compute mean shape
  - `compute_delaunay_triangulation()`: Generate triangles for warping
- `warp.py`: Piecewise affine warping
  - `piecewise_affine_warp()`: Main warping function using cv2.warpAffine per triangle
  - `scale_landmarks_from_centroid()`: Apply margin expansion
  - `compute_fill_rate()`: Measure non-black pixel percentage

**src_v2/data/**
- `dataset.py::LandmarkDataset`: Loads X-rays with landmarks from CSV
- `transforms.py`: CLAHE, augmentations, TTA (Test-Time Augmentation with horizontal flip)
- `utils.py`: Data splits, normalization utilities

**src_v2/training/**
- `trainer.py::LandmarkTrainer`: Two-phase training (frozen backbone, then fine-tuning)
- `callbacks.py`: Early stopping, model checkpointing

**src_v2/evaluation/**
- `metrics.py`: Pixel error computation, classification metrics
- `ensemble.py`: Classifier ensemble soft/hard voting, TTA, case-level impact analysis
- `quality_assessment.py`: Image quality scoring (BRISQUE/NIQE)
- `error_analysis.py`: Misclassification forensics and error pattern categorization

**src_v2/gui/**
- `app.py`: Gradio interface (3 tabs: full demo, quick classify, about)
- `config.py`: Centralized GUI config (paths, validated metrics, colors, UI text)
- `inference_pipeline.py`: Orchestrates landmark → warp → classify pipeline
- `model_manager.py`: Singleton lazy-loading model manager (GPU/CPU auto-detection)
- `gradcam_utils.py`: GradCAM visualization for model explainability
- `visualizer.py`: Rendering functions for pipeline stage images

### Landmark Structure
15 landmarks define lung contours (NOT specific anatomical points):
- Central axis: L1 (top) → L9 → L10 → L11 → L2 (bottom) at t=[0, 0.25, 0.50, 0.75, 1.0]
- Left lung: L12, L3, L5, L7, L14
- Right lung: L13, L4, L6, L8, L15
- 5 symmetric pairs: (L3,L4), (L5,L6), (L7,L8), (L12,L13), (L14,L15)

Defined in `src_v2/constants.py::SYMMETRIC_PAIRS`, `CENTRAL_LANDMARKS`.

### Configuration System
JSON configs in `configs/`. Key ones:
- `ensemble_best.json`: Best landmark ensemble (3.61 px error)
- `warping_best.json`: Optimal warping parameters (margin=1.05, use_full_coverage=false)
- `classifier_warped_base.json`: Classifier training defaults
- `landmarks_train_base.json`: Landmark model training defaults
- `ensemble_classifier.json`: Classifier ensemble config (5-fold CV weighted soft voting)
- `warping_cleaned.json`: Warping config for cleaned dataset (v1.1)
- `cv_*.json`: Cross-validation training configs for ablation experiments (focal, mining, curriculum, augmentation variants)
- `ensemble_phase10_*.json`: Phase 10 ensemble evaluation configs

Configs avoid CLI flag proliferation and enable reproducibility.

### Critical Implementation Details

**Warping Margin**: Use `margin_scale=1.05` (5% expansion from landmark centroid). Found via grid search in Session 25 `optimize-margin` command. Value stored in `GROUND_TRUTH.json` and `src_v2/constants.py::OPTIMAL_MARGIN_SCALE`.

**CLAHE**: `tile_size=4` performs better than 8 (validated in ensemble experiments). See `src_v2/constants.py::DEFAULT_CLAHE_TILE_SIZE`.

**TTA (Test-Time Augmentation)**: Horizontal flip with `SYMMETRIC_PAIRS` correction. The `src_v2/evaluation/metrics.py::compute_pixel_error()` swaps left/right landmarks when averaging predictions from flipped images.

**Two-Phase Training**: Phase 1 trains only the regression head with frozen backbone (15 epochs, lr=1e-3). Phase 2 fine-tunes entire model (100 epochs, backbone_lr=2e-5, head_lr=2e-4). See `src_v2/training/trainer.py::LandmarkTrainer`.

**Dataset Splits**: Use `split_seed` parameter to ensure reproducible train/val/test splits. Splits are created in `src_v2/data/dataset.py::create_dataloaders()`.

**Coordinate Normalization**: Landmarks are normalized to [0,1] during training and denormalized to image coordinates (224x224) for evaluation. The `detect_architecture_from_checkpoint()` in `src_v2/cli.py` infers model architecture from checkpoint.

## Ground Truth & Validation

`GROUND_TRUTH.json` (v2.2.0) is the source of truth for validated metrics. When modifying experiments or creating visualizations, reference values from this file, not hardcoded numbers.

Key validated values:
- Landmark ensemble best (seeds 123,321,111,666): 3.61 px
- Best individual landmark model (seed456): 4.04 px
- **Classifier ensemble+TTA accuracy: 98.26%** (current best, final_evaluation in GROUND_TRUTH.json)
- Single classifier (warped_lung_best): 98.05%
- 5-fold CV mean accuracy: 98.60% ± 0.26%
- Optimal margin_scale: 1.05
- CLAHE tile_size: 4

Note: `warped_96` (99.10%) is marked **obsolete** in GROUND_TRUTH.json — it used full_coverage warping which is no longer the current approach. The current method is `warped_lung_best`.

## Important Data Flow

**Cached Predictions**: The current pipeline caches landmark predictions for the entire dataset in `.npz` format:
```bash
# Generate cache
python scripts/predict_landmarks_dataset.py --ensemble-config configs/ensemble_best.json ...
# Output: outputs/landmark_predictions/session_warping/predictions.npz

# Use cache for warping (no re-inference)
python -m src_v2 generate-dataset --predictions outputs/landmark_predictions/.../predictions.npz
```

This avoids re-running inference during warping experiments. The `.npz` file contains:
- `predictions`: landmark coordinates
- `image_paths`: corresponding image paths
- Metadata: models, TTA, CLAHE settings, seeds

**Dataset Organization**:
- `data/dataset/COVID-19_Radiography_Dataset/`: Original images (not in repo)
- `checkpoints/`: Trained models (~629 MB after cleanup on 2026-01-20)
  - `session10/ensemble/seed123/final_model.pt` - Ensemble model (critical)
  - `session10/ensemble/seed456/final_model.pt` - Best individual (4.04 px, historical)
  - `session13/seed321/final_model.pt` - Ensemble model (critical)
  - `session13/seed789/final_model.pt` - Historical ensemble (3.71 px)
  - `repro_split111/session14/seed111/final_model.pt` - Ensemble model (critical)
  - `repro_split666/session16/seed666/final_model.pt` - Ensemble model (critical)
  - **NOTE**: Cleanup removed ~133 GB of intermediate checkpoints and non-critical experiments. See `docs/CHECKPOINTS_CLEANUP_REPORT.md` for details. Backup available at `checkpoints_backup_20260120.tar.gz`.
- `outputs/`: Generated artifacts (not in repo)
  - `shape_analysis/`: Canonical shape and triangulation
  - `landmark_predictions/`: Cached predictions
  - `warped_lung_best/`: Warped dataset (current)
  - `classifier_warped_lung_best/`: Trained classifier
  - `classifier_cv/`: 5-fold CV ensemble models and results
  - `data_cleaning/`: Cleaning manifests and reports (v1.1)
  - `ablation_results/`: Training improvement ablation outputs (v1.1)

## Milestone Context

The project uses GSD (Get Stuff Done) workflow with planning in `.planning/`:
- **v1.0** (shipped 2026-02-16): Ensemble+TTA classifier (Phases 1-5)
- **v1.1** (in progress): Data-centric accuracy improvement (Phases 6-10)
  - Phase 7: Data cleaning pipeline (landmark outliers, label noise via cleanlab)
  - Phase 8: Training improvements (focal loss, hard example mining, curriculum learning)
  - Phase 9: Advanced augmentation (medical-specific, MixUp, CutMix, ElasticTransform)
  - Phase 10: Final evaluation with statistical validation (McNemar, bootstrap CI, DeLong AUC)

Constraint: v1.1 keeps ResNet-18 architecture fixed to isolate data quality effects.

## Legacy/Archived Code

Do NOT use these for current work:
- `scripts/generate_warped_dataset.py`: Old session 21 workflow with GT landmarks
- `scripts/generate_full_warped_dataset.py`: Session 25 inline warping without cache
- `scripts/predict.py`: Old prediction wrapper (use CLI or predict_landmarks_dataset.py)
- `scripts/archive/`: Historical experiments and debugging scripts

The current pipeline is documented in `docs/REPRO_FULL_PIPELINE.md`.

## Documentation

Key docs in `docs/`:
- `REPRO_FULL_PIPELINE.md`: Complete reproduction guide
- `REPRO_ENSEMBLE_3_71.md`: Landmark ensemble details
- `QUICKSTART_WARPING.md`: Warping pipeline quick start
- `LANDMARK_VISUALIZATION_DATASET.md`: Generate landmark visualizations on original images
- `REPRO_CLASSIFIER_RESNET18.md`: Classifier training details
- `CONFIGS.md`: Configuration system guide
- `EXPERIMENTS.md`: Experimental results summary
- `CHECKPOINTS_CLEANUP_REPORT.md`: Checkpoint cleanup report (2026-01-20, freed 133 GB)

Session notes and reports are in `docs/sesiones/` and `docs/reportes/`.

## Commit Conventions

Use Conventional Commits format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/modifications
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

Recent commit history shows Spanish descriptive messages are also acceptable, but prefer English Conventional Commits for new work.

## Code Style

- PEP 8 with 100-character line limit
- Type hints for function parameters and return values
- Google-style docstrings for public functions
- Imports: stdlib → third-party → local, alphabetized within groups
- Preserve Spanish names in datasets/labels (e.g., `Viral_Pneumonia`), use English in code

## Testing Notes

- Test files: `test_*.py` in `tests/` directory
- Focus on edge cases for warping, transforms, and GPA
- Environment variable `FORCE_NUM_WORKERS_ZERO=1` for deterministic testing (see `pyproject.toml`)
- Coverage target: `src_v2` module
