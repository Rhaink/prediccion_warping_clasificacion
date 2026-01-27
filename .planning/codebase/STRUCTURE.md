# Codebase Structure

**Analysis Date:** 2026-01-27

## Directory Layout

```
prediccion_warping_clasificacion/
├── src_v2/                         # Main package with all core logic
│   ├── __main__.py                 # Entry point: python -m src_v2
│   ├── __init__.py                 # Package initialization
│   ├── cli.py                      # Primary CLI (900+ lines, 35+ commands)
│   ├── constants.py                # Centralized constants (landmarks, sizes, params)
│   │
│   ├── data/                       # Data loading and transforms
│   │   ├── dataset.py              # LandmarkDataset, data splitting, sample weights
│   │   ├── transforms.py           # Image preprocessing (CLAHE, augmentation, TTA)
│   │   ├── utils.py                # CSV loading, coordinate handling
│   │   └── __init__.py
│   │
│   ├── models/                     # Neural network models
│   │   ├── resnet_landmark.py      # ResNet18Landmarks with CoordinateAttention
│   │   ├── classifier.py           # ImageClassifier (7 architectures: ResNet, EfficientNet, DenseNet, etc.)
│   │   ├── hierarchical.py         # HierarchicalLandmarkModel (alternative approach)
│   │   ├── losses.py               # WingLoss, CombinedLoss with symmetry penalties
│   │   └── __init__.py
│   │
│   ├── processing/                 # Geometric processing
│   │   ├── gpa.py                  # Generalized Procrustes Analysis for canonical shape
│   │   ├── warp.py                 # Piecewise affine warping via Delaunay triangulation
│   │   └── __init__.py
│   │
│   ├── training/                   # Training loop implementations
│   │   ├── trainer.py              # LandmarkTrainer (two-phase: frozen backbone → fine-tuning)
│   │   ├── callbacks.py            # EarlyStopping, ModelCheckpoint, LRScheduler
│   │   └── __init__.py
│   │
│   ├── evaluation/                 # Evaluation metrics
│   │   ├── metrics.py              # Pixel error, TTA evaluation, per-landmark stats
│   │   └── __init__.py
│   │
│   ├── visualization/              # Analysis and visualization tools
│   │   ├── scientific_viz.py       # Paper-quality figure generation
│   │   ├── gradcam.py              # GradCAM heatmap analysis
│   │   ├── feature_extractor.py    # Feature map extraction
│   │   ├── feature_visualizer.py   # Feature visualization
│   │   ├── comparison_viz.py       # Before/after warping comparison
│   │   ├── pfs_analysis.py         # Position-focused slicing analysis
│   │   ├── error_analysis.py       # Error distribution analysis
│   │   ├── plot_failure_cases.py   # Misclassified sample visualization
│   │   ├── plot_roc_curves.py      # ROC curve generation
│   │   ├── diagramming.py          # Architecture diagrams
│   │   ├── utils.py                # Visualization utilities
│   │   └── __init__.py
│   │
│   ├── gui/                        # Gradio web interface
│   │   ├── app.py                  # Gradio interface with 3 tabs (Full demo, Quick view, About)
│   │   ├── inference_pipeline.py   # Full warping + classification pipeline
│   │   ├── model_manager.py        # Model loading and caching
│   │   ├── visualizer.py           # Probability chart visualization
│   │   ├── gradcam_utils.py        # GradCAM integration for UI
│   │   ├── config.py               # UI theming, labels, colors
│   │   ├── CHANGELOG.md            # GUI changelog
│   │   ├── README.md               # GUI documentation
│   │   └── __init__.py
│   │
│   └── utils/                      # Utility functions
│       ├── geometry.py             # Geometric helper functions
│       └── __init__.py
│
├── configs/                        # JSON configuration files for reproducibility
│   ├── ensemble_best.json          # Best landmark ensemble config (3.61 px error)
│   ├── warping_best.json           # Optimal warping params (margin=1.05)
│   ├── classifier_warped_base.json # Classifier training defaults
│   ├── landmarks_train_base.json   # Landmark model training defaults
│   └── *.json                      # Other experiment configs
│
├── scripts/                        # Utility and analysis scripts
│   ├── predict_landmarks_dataset.py # Cache landmark predictions for entire dataset
│   ├── evaluate_ensemble_from_config.py # Evaluate ensemble models
│   ├── extract_dataset_splits.py   # Extract train/val/test splits
│   ├── generate_*.py               # Figure generation for thesis/papers (20+ scripts)
│   ├── create_thesis_figures.py    # Batch thesis figure generation
│   ├── analyze_hospital_marks.py   # Hospital mark analysis
│   ├── gpa_analysis.py             # GPA debugging and analysis
│   ├── build_windows_exe.py        # Windows standalone build
│   ├── visualization/              # Visualization generation scripts
│   │   ├── generate_publication_gradcam_grid.py
│   │   ├── generate_feature_maps_pipeline.py
│   │   └── *.py
│   ├── glass_box_visualizations/   # Explainability visualizations
│   │   └── *.py
│   ├── fisher/                     # Fisher information analysis
│   │   └── studies/
│   ├── archive/                    # Legacy and obsolete scripts (DO NOT USE)
│   │   ├── session*.py
│   │   ├── classification/
│   │   └── invalid_warping/
│   └── quickstart_*.sh             # Automated pipeline scripts
│
├── tests/                          # Test suite (if present)
│   └── test_*.py
│
├── docs/                           # Documentation
│   ├── REPRO_FULL_PIPELINE.md      # Complete reproduction guide
│   ├── REPRO_ENSEMBLE_3_71.md      # Landmark ensemble details
│   ├── QUICKSTART_WARPING.md       # Warping pipeline quick start
│   ├── LANDMARK_VISUALIZATION_DATASET.md
│   ├── REPRO_CLASSIFIER_RESNET18.md # Classifier training guide
│   ├── CONFIGS.md                  # Configuration system guide
│   ├── EXPERIMENTS.md              # Experimental results summary
│   ├── CHECKPOINTS_CLEANUP_REPORT.md # (2026-01-20)
│   ├── sesiones/                   # Session notes
│   └── reportes/                   # Experiment reports
│
├── checkpoints/                    # Trained model checkpoints (not in repo)
│   ├── session10/ensemble/seed123/final_model.pt
│   ├── session13/seed321/final_model.pt
│   ├── repro_split666/session16/seed666/final_model.pt (BEST ENSEMBLE 3.61 px)
│   └── ...
│
├── outputs/                        # Generated artifacts (not in repo)
│   ├── shape_analysis/             # Canonical shape and triangulation
│   ├── landmark_predictions/       # Cached NPZ predictions
│   ├── warped_lung_best/           # Warped dataset
│   └── classifier_warped_lung_best/ # Trained classifier
│
├── data/                           # Data (not in repo)
│   ├── dataset/COVID-19_Radiography_Dataset/ # Original images (299x299)
│   ├── coordenadas/
│   │   └── coordenadas_maestro.csv # Master landmarks CSV
│   └── external_datasets/
│
├── .planning/codebase/             # GSD codebase mapping documents
│   ├── ARCHITECTURE.md
│   ├── STRUCTURE.md
│   ├── CONVENTIONS.md
│   ├── TESTING.md
│   ├── STACK.md
│   ├── INTEGRATIONS.md
│   └── CONCERNS.md
│
├── .venv/                          # Python virtual environment
├── pyproject.toml                  # Project metadata and dependencies
├── requirements.txt                # Pip dependencies
├── CLAUDE.md                       # Claude Code instructions
├── GROUND_TRUTH.json               # Validated metrics source of truth
├── README.md                       # Project overview
└── setup.py / MANIFEST.in          # Package configuration

```

## Directory Purposes

**src_v2/**
- Purpose: Core implementation package
- Contains: All models, training, data, and utility code
- Key files: `cli.py` (main entry point), `constants.py` (centralized config)

**src_v2/data/**
- Purpose: Dataset loading and preprocessing
- Contains: LandmarkDataset, image transforms (CLAHE), TTA, data splitting
- Key files: `dataset.py` (LandmarkDataset with stratified splits), `transforms.py` (augmentation pipeline)

**src_v2/models/**
- Purpose: Neural network architectures
- Contains: Landmark detector (ResNet18 + CoordinateAttention), classifier (7 backbone options), loss functions
- Key files: `resnet_landmark.py` (main architecture), `classifier.py` (multi-class COVID detection), `losses.py` (WingLoss, CombinedLoss)

**src_v2/processing/**
- Purpose: Geometric processing pipeline
- Contains: GPA (canonical shape), piecewise affine warping via Delaunay triangulation
- Key files: `gpa.py` (Generalized Procrustes Analysis), `warp.py` (image warping and boundary handling)

**src_v2/training/**
- Purpose: Model training orchestration
- Contains: Two-phase trainer (frozen backbone → fine-tuning), callbacks (early stopping, checkpointing)
- Key files: `trainer.py` (LandmarkTrainer), `callbacks.py` (training utilities)

**src_v2/evaluation/**
- Purpose: Metric computation
- Contains: Pixel error calculation, TTA evaluation, per-landmark and per-category statistics
- Key files: `metrics.py` (compute_pixel_error, evaluate_model)

**src_v2/visualization/**
- Purpose: Analysis and publication-quality visualization
- Contains: GradCAM, feature extraction, scientific figures, failure case analysis, PFS
- Key files: `scientific_viz.py` (publication figures), `gradcam.py` (attention visualization)

**src_v2/gui/**
- Purpose: Interactive Gradio web interface
- Contains: Full demo (landmarks + warping + classification), quick classification, results visualization
- Key files: `app.py` (main Gradio interface), `inference_pipeline.py` (end-to-end pipeline)

**configs/**
- Purpose: Configuration reproducibility
- Contains: JSON configs for ensemble, warping, classifier, and landmark training
- Rationale: Avoids CLI flag proliferation; enables reproducibility by bundling parameters
- Key files: `ensemble_best.json`, `warping_best.json`

**scripts/**
- Purpose: One-off analysis and data generation
- Contains: Landmark prediction caching, figure generation, experiment analysis
- Use: Run specific analysis or generate outputs; not part of main package
- Key files: `predict_landmarks_dataset.py` (cache predictions), `evaluate_ensemble_from_config.py` (ensemble eval)
- Archive: `scripts/archive/` contains obsolete session scripts; do NOT use for new work

**checkpoints/**
- Purpose: Trained model storage
- Contains: Model weights for landmarks and classifier
- Critical files:
  - `repro_split666/session16/seed666/final_model.pt` (BEST: 3.61 px ensemble error)
  - `session13/seed321/final_model.pt` (ensemble model)
- Note: Intermediate checkpoints cleaned up 2026-01-20 (freed 133 GB)

**outputs/**
- Purpose: Generated artifacts during pipeline execution
- Contains: Cached predictions, warped datasets, trained classifiers
- Not committed to repo; regenerated per experiment

## Key File Locations

**Entry Points:**
- `src_v2/__main__.py`: Python module entry (`python -m src_v2`)
- `src_v2/cli.py`: CLI implementation (35+ commands via Typer)
- `src_v2/gui/app.py`: Gradio web interface (run with `python -m src_v2.gui.app`)

**Configuration:**
- `src_v2/constants.py`: All domain constants (15 landmarks, sizes, default parameters)
- `configs/*.json`: Reproducible experiment configurations
- `GROUND_TRUTH.json`: Validated metrics (source of truth for metrics)

**Core Logic:**
- `src_v2/processing/gpa.py`: Canonical shape computation (GPA)
- `src_v2/processing/warp.py`: Geometric normalization via Delaunay warping
- `src_v2/models/resnet_landmark.py`: Landmark detection model
- `src_v2/models/classifier.py`: COVID classification model
- `src_v2/training/trainer.py`: Two-phase training loop

**Data Handling:**
- `src_v2/data/dataset.py`: LandmarkDataset with stratified splits
- `src_v2/data/transforms.py`: Image preprocessing (CLAHE, TTA, augmentation)
- `src_v2/data/utils.py`: CSV loading and coordinate normalization

**Evaluation:**
- `src_v2/evaluation/metrics.py`: Pixel error, per-landmark stats, TTA evaluation

**Visualization:**
- `src_v2/visualization/scientific_viz.py`: Publication-quality figures
- `src_v2/visualization/gradcam.py`: GradCAM heatmaps
- `scripts/generate_*.py`: Thesis/paper figure generation

**Testing:**
- `tests/` (if present): Test files with pytest

## Naming Conventions

**Files:**
- Source modules: `snake_case.py` (e.g., `resnet_landmark.py`, `piecewise_affine_warp`)
- Entry points: `__main__.py`, `__init__.py`, `cli.py`
- Test files: `test_*.py` (pytest convention)
- Scripts: descriptive names starting with verb (e.g., `predict_landmarks_dataset.py`, `generate_confusion_matrix_cv.py`)
- Config files: `*_best.json` (best params) or `*_base.json` (defaults)

**Directories:**
- Package modules: `snake_case` (e.g., `src_v2`, `data`, `models`)
- Documentation: `UPPERCASE.md` (e.g., `REPRO_FULL_PIPELINE.md`)
- Session records: `sesiones/`, `reportes/`
- Archive: `archive/` (historical code, do NOT use)

**Python Classes:**
- Models: `PascalCase` ending in descriptive suffix (e.g., `ResNet18Landmarks`, `ImageClassifier`, `LandmarkDataset`)
- Losses: `PascalCase` ending in `Loss` (e.g., `WingLoss`, `CombinedLandmarkLoss`)
- Modules: `PascalCase` only for classes, `snake_case` for functions

**Python Functions:**
- Private/internal: Leading underscore `_function_name()`
- Public API: No underscore `compute_pixel_error()`, `piecewise_affine_warp()`
- Callbacks: Named for action `on_epoch_end()`, `early_stopping_check()`

## Where to Add New Code

**New Landmark Detection Feature:**
- Implementation: `src_v2/models/resnet_landmark.py` (add to ResNet18Landmarks)
- Tests: `tests/test_landmark_model.py` (if tests exist)
- Constants: Update `src_v2/constants.py` if adding new architectural parameters
- Config: Create new JSON in `configs/` (e.g., `landmarks_train_new_feature.json`)

**New Classification Architecture:**
- Implementation: `src_v2/models/classifier.py` (add to SUPPORTED_BACKBONES and __init__)
- Testing: Add evaluation command in `src_v2/cli.py`
- CLI: Add new training command or extend `train-classifier` with `--backbone` option

**New Geometric Processing:**
- Implementation: `src_v2/processing/` (create `new_method.py` or extend existing)
- Pipeline integration: Add command to `src_v2/cli.py` (e.g., `@app.command("generate-dataset")`)
- Tests: `tests/test_processing.py`

**New Data Transform:**
- Implementation: `src_v2/data/transforms.py` (add function and register in factories)
- Usage: Update `get_train_transforms()` and `get_val_transforms()`
- CLI: Add `--use-new-transform` flag to relevant commands

**New Visualization:**
- Implementation: `src_v2/visualization/new_viz.py` (for reusable components)
- Scripts: `scripts/generate_new_figures.py` (for one-off generation)
- Integration: Add to `generate_all_visualizations.py` if widely used

**New CLI Command:**
- Location: Add function with `@app.command()` decorator in `src_v2/cli.py`
- Structure: Follow existing command pattern (import, setup, execute, logging)
- Documentation: Include docstring with usage examples
- Utilities: Extract common logic to appropriate module (e.g., `src_v2/data/utils.py`)

**Utility Functions:**
- General helpers: `src_v2/utils/` (geometry.py or new module)
- Data utilities: `src_v2/data/utils.py`
- Model utilities: `src_v2/models/` (create losses.py if needed)
- Evaluation: `src_v2/evaluation/metrics.py`

## Special Directories

**scripts/archive/**
- Purpose: Historical experiments and debugging
- Status: Obsolete; not maintained
- DO NOT USE for new work; reference only for historical context
- Contains: session*.py (numbered experiment runs), classification/, invalid_warping/

**scripts/glass_box_visualizations/**
- Purpose: Explainability and interpretability analysis
- Status: Active research support
- Use: For generating explanation visualizations (GradCAM, feature maps)

**scripts/fisher/**
- Purpose: Fisher information analysis
- Status: Research feature
- Use: Statistical significance testing and uncertainty quantification

**data/ (not committed)**
- Purpose: Training data storage
- Contains: COVID-19 Radiography Dataset (original 299x299 images)
- Landmark coordinates: `data/coordenadas/coordenadas_maestro.csv`
- Size: Large; not in repository

**outputs/ (not committed)**
- Purpose: Intermediate and final results
- Sub-directories auto-created per experiment:
  - `landmark_predictions/`: Cached NPZ files with landmark predictions
  - `warped_*`: Warped image datasets
  - `classifier_*`: Trained classifier checkpoints
  - `shape_analysis/`: Canonical shape and Delaunay triangulation
- Regenerated per pipeline run; not persisted

**checkpoints/ (not committed, but critical)**
- Purpose: Model weights for all trained models
- Critical files: Must preserve best performing models
- Backup: `checkpoints_backup_20260120.tar.gz` available for emergency restore
- Cleanup: 2026-01-20 removed 133 GB of intermediate checkpoints

---

*Structure analysis: 2026-01-27*
