# Technology Stack

**Analysis Date:** 2026-01-27

## Languages

**Primary:**
- Python 3.9+ - All source code, scripts, and CLI
- Supported versions: 3.9, 3.10, 3.11 (tested with 3.12.3 in environment)

**Secondary:**
- Bash - Installation and automation scripts (`install.sh`, `run_demo.sh`)
- Batch - Windows deployment scripts (`install.bat`, `run_demo.bat`)
- JSON - Configuration files and serialization
- LaTeX - Documentation and thesis generation

## Runtime

**Environment:**
- Python 3.9+ with virtual environment support
- CUDA/ROCm optional for GPU acceleration (see requirements.txt notes)
- PyInstaller support for standalone Windows distribution (Python 3.12.8 embeddable)

**Package Manager:**
- pip with requirements.txt
- setuptools for package build and installation

**Build System:**
- setuptools 61.0+ with pyproject.toml configuration
- Entry point: `covid-landmarks` CLI via `src_v2.cli:app`

## Frameworks

**Core:**
- PyTorch 2.0.0+ - Deep learning framework for models and training
  - `torch` - Core neural network operations
  - `torchvision` 0.15.0+ - Pre-trained models (ResNet-18), image transforms
  - Models used: ResNet-18 for landmark detection and classification

**CLI & Interface:**
- Typer 0.9.0+ - Type-safe CLI framework
  - Located: `src_v2/cli.py`
  - Provides subcommands for training, evaluation, prediction, warping, etc.

**UI/Demo:**
- Gradio 4.0.0+ - Interactive web interface for COVID detection demo
  - Entry point: `scripts/run_demo.py`
  - Supports local and shareable links (--share flag)
  - Supports PyInstaller frozen distribution mode

**Testing:**
- pytest 7.0.0+ - Test framework
  - Configuration: `pyproject.toml`
  - Test discovery: `tests/test_*.py` pattern
  - Coverage: pytest-cov 4.0.0+
  - Environment: `FORCE_NUM_WORKERS_ZERO=1` for deterministic testing

**Build/Deployment:**
- PyInstaller - Standalone Windows executable generation
  - Configuration: `scripts/build_portable_windows.py`
  - Creates portable distribution with embedded Python 3.12.8

## Key Dependencies

**Critical (Core Pipeline):**
- torch 2.0.0+ - Neural network architecture, training, inference
  - Used in: `src_v2/models/resnet_landmark.py`, `src_v2/models/classifier.py`, `src_v2/training/trainer.py`
- numpy 2.0.0+ - Numerical computations for shape analysis, warping
  - Used in: `src_v2/processing/gpa.py`, `src_v2/processing/warp.py`
- opencv-python 4.8.0+ - Image I/O, warping operations
  - Used in: `src_v2/processing/warp.py` for `cv2.warpAffine()`
  - Also: image preprocessing, CLAHE histogram equalization

**Scientific Computing:**
- scipy 1.10.0+ - Spatial algorithms
  - `scipy.spatial.Delaunay` - Triangulation for piecewise affine warping
  - Used in: `src_v2/processing/gpa.py`, `src_v2/processing/warp.py`
- pandas 2.0.0+ - Data manipulation and CSV loading
  - Used in: `src_v2/data/dataset.py` for landmark coordinate CSVs
  - Used in: Data preprocessing and split creation

**Machine Learning:**
- scikit-learn 1.3.0+ - Model evaluation and metrics
  - `sklearn.model_selection.train_test_split` - Dataset splitting
  - `sklearn.metrics` - ROC curves, classification metrics
  - Used in: `src_v2/evaluation/metrics.py`, `src_v2/data/dataset.py`

**Visualization:**
- matplotlib 3.7.0+ - Plotting and scientific figures
  - Used in: `src_v2/visualization/` modules for landmark visualization, ROC curves, confusion matrices
  - Supports publication-quality figures
- seaborn 0.12.0+ - Statistical data visualization
  - Used in: `src_v2/visualization/` for heatmaps and styled plots
- Pillow 10.0.0+ - Image processing and manipulation
  - Used in: Loading/saving images as PIL Image objects in dataset pipeline

**Utilities:**
- tqdm 4.65.0+ - Progress bars for training/evaluation loops
  - Used in: `src_v2/training/trainer.py`, `src_v2/visualization/` modules

## Configuration

**Environment:**
- Runtime configuration via JSON files in `configs/`:
  - `ensemble_best.json` - Landmark ensemble settings
  - `warping_best.json` - Geometric normalization parameters
  - `classifier_warped_base.json` - Classification training defaults
  - `landmarks_train_base.json` - Landmark training defaults
  - Additional: `hierarchical_train_base.json`, classifier variants

- Ground truth validation: `GROUND_TRUTH.json` (source of truth for metrics)
  - Contains validated results: ensemble error (3.61 px), classifier accuracy (99.10%)
  - Margin scale optimization: 1.05 (from grid search)
  - CLAHE settings: tile_size=4

**Build Configuration:**
- `pyproject.toml` - Package metadata, dependencies, build system, pytest/coverage config
- `MANIFEST.in` - Additional files included in distribution

**Development:**
- `.gitignore` - Git exclusions (outputs, checkpoints, cache)
- `.claude` directory - Claude Code session metadata (not tracked)

## Platform Requirements

**Development:**
- Python 3.9+ interpreter
- pip package manager
- Virtual environment support
- CUDA/ROCm optional (for GPU training)
- 8GB+ RAM recommended (16GB+ for GPU)
- GPU: NVIDIA CUDA 11.8+ or AMD ROCm 6.0+ (optional but recommended)

**Production/Deployment:**
- **Linux/macOS:** Python 3.9+ installation
- **Windows:**
  - Option 1: Python 3.9+ installation
  - Option 2: Portable standalone executable (PyInstaller, ~800MB)
    - Includes embedded Python 3.12.8
    - Single-file distribution, no Python installation required
    - Pre-packaged models and dependencies

**Data Requirements:**
- Input images: JPEG/PNG radiographs (224x224 or 299x299 pixels)
- Landmark annotations: CSV format with 15 (x,y) coordinate pairs per image
- Dataset organization: `data/dataset/COVID-19_Radiography_Dataset/` subdirectories

## Hardware Recommendations

**Minimum:**
- CPU-only: 4-core processor, 8GB RAM, 10GB disk
- Training time: ~4-6 hours per model on CPU

**Recommended:**
- GPU: NVIDIA GPU with 8GB+ VRAM (e.g., RTX 3060, RTX 4060)
- RAM: 16GB system memory
- Disk: 50GB (for models, outputs, cached predictions)
- Training time: ~30-60 minutes per model with GPU

## Dependency Installation

Standard installation:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Development (with testing tools):
```bash
pip install -e ".[dev]"
```

GPU-specific (CUDA 12.1):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

GPU-specific (ROCm 6.0):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
pip install -r requirements.txt
```

---

*Stack analysis: 2026-01-27*
