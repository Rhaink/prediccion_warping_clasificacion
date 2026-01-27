# External Integrations

**Analysis Date:** 2026-01-27

## APIs & External Services

**None detected** - This project is a self-contained machine learning research system with no external API integrations. All inference, training, and evaluation runs locally.

## Data Storage

**File Storage:**
- Local filesystem only - No cloud storage integration
  - Input images: `data/dataset/COVID-19_Radiography_Dataset/` subdirectories
  - Training outputs: `outputs/` directory (generated artifacts)
  - Model checkpoints: `checkpoints/` directory (trained models)
  - Configuration: `configs/` directory (JSON-based settings)

**Data Formats:**
- Images: JPEG/PNG radiographs (224x224 or 299x299 pixels)
- Annotations: CSV format (`data/coordenadas/coordenadas_maestro.csv`)
  - Columns: image_name, category, L1_x, L1_y, ... L15_x, L15_y
  - Categories: COVID, Normal, Viral_Pneumonia
- Serialization: NumPy NPZ format for cached landmark predictions
  - Path: `outputs/landmark_predictions/session_warping/predictions.npz`
  - Contains: predictions array, image_paths, metadata (models, TTA settings, seeds)
- Models: PyTorch `.pt` files (stored in `checkpoints/`)
- Configuration: JSON format (stored in `configs/`)
- Ground truth: JSON format (`GROUND_TRUTH.json`)
- Shape analysis: JSON format (GPA canonical shape, Delaunay triangles)

**Database:**
- Not applicable - Project uses file-based storage only
- No database server required
- No ORM in use (direct file I/O and pandas for CSV)

**Caching:**
- Landmark prediction cache: `.npz` files in `outputs/landmark_predictions/`
  - Purpose: Avoid re-running inference during warping experiments
  - Used by: `src_v2/processing/warp.py` in warping pipeline
- Dataset caches: Loaded via pandas DataFrames in memory
- No persistent cache layer (all in-memory during execution)

## Authentication & Identity

**Auth Provider:** Not applicable
- No authentication system required
- No user management
- No API keys or credentials
- Designed for local/institutional research use

**Security Considerations:**
- Models are stored as `.pt` files (PyTorch serialization)
  - Risk: `torch.load()` can execute arbitrary code from untrusted model files
  - Mitigation: Only load models from trusted sources (committed to repo)
  - Implementation: Standard `torch.load()` in `src_v2/cli.py` checkpoint loading

## Monitoring & Observability

**Error Tracking:**
- Not integrated - Uses standard Python logging
- No external error reporting (Sentry, etc.)

**Logging:**
- Framework: Python `logging` module
  - Configuration: `src_v2/cli.py::logging.basicConfig()`
  - Level: INFO by default, DEBUG with `--verbose` flag
  - Loggers: Per-module loggers (e.g., `logger = logging.getLogger(__name__)`)
- Output: Console/stderr by default
- Files: Log files in `outputs/` subdirectories (training, evaluation results)

**Training Metrics:**
- Stored in trainer history dictionaries: `train_loss`, `val_loss`, `train_error_px`, `val_error_px`, `lr`
- Checkpointing: Best model saved during training by `ModelCheckpoint` callback
- No experiment tracking service (no MLflow, Weights & Biases, etc.)

## CI/CD & Deployment

**Hosting:**
- Local execution only - No hosted deployment
- Gradio demo available for local sharing/testing (`scripts/run_demo.py`)
- Optional: PyInstaller standalone Windows distribution

**Execution Modes:**
- **CLI:** `python -m src_v2 [command]` via Typer
- **Demo:** `python scripts/run_demo.py` with optional `--share` flag for temporary public link
- **Standalone:** PyInstaller `.exe` on Windows (portable, no Python installation)

**CI Pipeline:**
- Not detected - No GitHub Actions, GitLab CI, or similar
- Testing available locally: `python -m pytest tests/ -v`
- No automated testing in git hooks or CI/CD pipelines

**Dependency Management:**
- `requirements.txt` - Pinned versions for reproducibility
- `pyproject.toml` - Package metadata and optional dev dependencies
- No lock files (pip freeze output) - Versions specified in requirements

## Environment Configuration

**Required Environment Variables:**
- None required for standard operation
- Optional for deployment:
  - `COVID_DEMO_MODELS_DIR` - Set by `scripts/run_demo.py` for PyInstaller mode
  - `COVID_DEMO_FROZEN` - Flag indicating PyInstaller frozen mode (set by launcher)

**Configuration Files:**
- JSON configs in `configs/` directory:
  - `ensemble_best.json` - Landmark model ensemble configuration
  - `warping_best.json` - Warping/geometric normalization parameters
  - `classifier_warped_base.json` - Classification training settings
  - `landmarks_train_base.json` - Landmark training defaults
  - Other variants for experiments

**Secrets Location:**
- Not applicable - No secrets or credentials in system
- All sensitive parameters (learning rates, hyperparameters) in JSON configs
- Model paths are relative or configurable in CLI

## Webhooks & Callbacks

**Incoming:** None

**Outgoing:** None

**Training Callbacks (Internal):**
- `EarlyStopping` - Monitors validation loss, stops training if no improvement
  - Implementation: `src_v2/training/callbacks.py::EarlyStopping`
- `ModelCheckpoint` - Saves best model during training
  - Implementation: `src_v2/training/callbacks.py::ModelCheckpoint`
- `LRSchedulerCallback` - Applies learning rate scheduling (CosineAnnealingLR)
  - Implementation: `src_v2/training/callbacks.py::LRSchedulerCallback`

## Data Input/Output Pipelines

**Input Pipeline:**
```
CSV annotations → pandas DataFrame → Dataset (image + landmark pairs) → DataLoader → Model
```
- Location: `src_v2/data/dataset.py::create_dataloaders()`
- Supports train/val/test splits with `train_test_split` from sklearn
- Applies transforms: CLAHE, augmentation, normalization

**Landmark Prediction Cache:**
```
Model inference on dataset → NumPy arrays → NPZ serialization → Disk storage
↓ (reused for warping)
Load NPZ → Use predictions with GPA/Delaunay → Warping pipeline
```
- Cache creation: `scripts/predict_landmarks_dataset.py`
- Cache usage: `src_v2/cli.py::warp_command()`

**Warping Pipeline:**
```
Original images → Cached landmarks → GPA alignment → Delaunay triangulation →
Piecewise affine transform → Normalized images → Classification
```
- Locations: `src_v2/processing/gpa.py`, `src_v2/processing/warp.py`

**Output Pipeline:**
```
Training artifacts → checkpoints/ and outputs/
↓
Warped dataset → outputs/warped_*/
↓
Classifier → outputs/classifier_*/
↓
Visualizations → outputs/*/figures/ (PNG, PDF)
```

## Research Data Management

**Dataset Organization:**
- Original images: `data/dataset/COVID-19_Radiography_Dataset/` (not in repo, ~20 GB)
- Landmark annotations: CSV in `data/coordenadas/`
- Warped dataset: `outputs/warped_lung_best/session_warping/` (generated)
- Landmark visualizations: `outputs/landmark_visualizations/` (generated)

**Ground Truth File:**
- `GROUND_TRUTH.json` - Single source of truth for validated metrics
- Contains: Best ensemble error, classifier accuracy, optimal parameters
- Used by: Visualization and comparison scripts to reference validated results

**Checkpoint Management:**
- Critical models in `checkpoints/`:
  - Landmark ensembles: `session10/`, `session13/`, `repro_split*/`
  - Individual best models stored for historical reference
  - Cleanup report: `docs/CHECKPOINTS_CLEANUP_REPORT.md` (freed 133 GB on 2026-01-20)
  - Backup: `checkpoints_backup_20260120.tar.gz`

## Export & Serialization Formats

**Model Export:**
- PyTorch format: `.pt` files (complete state dict)
- No ONNX export (could be added for cross-platform inference)
- No TorchScript export (could be added for C++ deployment)

**Data Export:**
- CSV: Landmark coordinates, annotations
- JSON: Configuration, ground truth, shape analysis
- NPZ: NumPy arrays for cached predictions
- PNG/JPEG: Generated visualizations and warped images
- PDF: Thesis and publication figures

## Third-Party Libraries Usage

**No External Service Dependencies:**
- All computation is local
- No API calls to external services
- No cloud storage integration
- No analytics or telemetry
- Research-focused, not production-oriented system

---

*Integration audit: 2026-01-27*
