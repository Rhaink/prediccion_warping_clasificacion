# Architecture

**Analysis Date:** 2026-01-27

## Pattern Overview

**Overall:** Two-stage geometrically-normalized CNN classification pipeline

**Key Characteristics:**
- **Stage 1 (Landmarks):** ResNet-18 ensemble predicts 15 lung contour landmarks via regression
- **Stage 2 (Normalization):** Piecewise affine warping normalizes geometry using Delaunay triangulation
- **Stage 3 (Classification):** ResNet-18 classifier trained on warped images achieves 99.10% accuracy
- **Caching Strategy:** Landmark predictions cached as NPZ files to avoid re-inference during warping experiments
- **Two-Phase Training:** Landmark models trained with frozen backbone (Phase 1) then fine-tuning (Phase 2)

## Layers

**Input Layer:**
- Purpose: Accept chest X-ray images (299x299 original, resized to 224x224 for processing)
- Location: `src_v2/data/dataset.py::LandmarkDataset`
- Contains: Image loading (PIL), coordinate parsing from CSV
- Depends on: `data/coordenadas/coordenadas_maestro.csv` master landmarks file
- Used by: Training pipeline, prediction pipeline

**Preprocessing Layer:**
- Purpose: Image enhancement and augmentation
- Location: `src_v2/data/transforms.py`
- Contains:
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) - enabled by default (clip=2.0, tile=4)
  - Random rotations, flips for augmentation
  - Test-Time Augmentation (TTA) - horizontal flip with symmetric landmark pair swapping
  - Normalization to ImageNet mean/std
- Depends on: `cv2`, `torchvision.transforms`
- Used by: LandmarkDataset, classifier inference

**Landmark Detection Layer:**
- Purpose: Predict 15 lung contour landmark coordinates
- Location: `src_v2/models/resnet_landmark.py::ResNet18Landmarks`
- Architecture:
  - Backbone: ResNet-18 pretrained (ImageNet)
  - Optional: CoordinateAttention module after layer3 (CVPR 2021)
  - Head: FC layers with optional GroupNorm and dropout
    - Output: 30 coordinates (15 landmarks × 2) in [0,1] normalized space
  - Loss: WingLoss (wing regression loss) or CombinedLoss (with symmetry penalties)
- Configuration: `use_coord_attention=True`, `deep_head=True`, `hidden_dim=768`
- Training: Two-phase
  - Phase 1 (15 epochs): Frozen backbone, train only regression head (lr=1e-3)
  - Phase 2 (100 epochs): Fine-tune entire model (backbone_lr=2e-5, head_lr=2e-4)
- Validation: Pixel error (Euclidean distance), per-landmark stats, TTA evaluation
- Best Result: 3.61 px ensemble error (repro_split666/session16/seed666)
- Depends on: `torch`, `torchvision.models`
- Used by: Warping pipeline, GUI inference

**Warping Layer (Geometric Normalization):**
- Purpose: Transform images to canonical shape using predicted landmarks
- Location: `src_v2/processing/warp.py`
- Pipeline:
  1. Compute canonical shape: Generalized Procrustes Analysis (GPA) on training landmarks
  2. Predict landmarks: Ensemble of landmark models on input image
  3. Build Delaunay triangulation: Connect landmarks to form triangles in both shapes
  4. Warp per triangle: Apply affine transformation within each triangle using cv2.warpAffine
  5. Compose output: Blend warped image into canvas
- Key Parameters:
  - `margin_scale=1.05` (5% expansion from landmark centroid) - optimized via grid search
  - Boundary handling: Add image boundary points to Delaunay for edge coverage
  - Fill validation: Reject triangles with <30% coverage (degenerate cases)
- Output: Geometrically normalized 224x224 image with consistent landmark configuration
- Depends on: `scipy.spatial.Delaunay`, `cv2.warpAffine`, GPA result
- Used by: Classifier training, inference pipeline

**Canonical Shape Layer:**
- Purpose: Compute mean shape for warping normalization
- Location: `src_v2/processing/gpa.py`
- Algorithm: Generalized Procrustes Analysis
  - Center: Remove translation (center at origin)
  - Scale: Normalize to unit Frobenius norm
  - Rotate: Iteratively align to consensus via SVD (Procrustes problem)
- Output: Canonical landmark coordinates + Delaunay triangulation mesh
- Cached in: `outputs/shape_analysis/canonical_shape.npz`
- Used by: Warp layer during inference
- Source: `src_v2/cli.py::compute-canonical` command

**Classification Layer:**
- Purpose: COVID-19 / Normal / Viral Pneumonia classification on normalized images
- Location: `src_v2/models/classifier.py::ImageClassifier`
- Supported Backbones: ResNet-18, ResNet-50, EfficientNet-B0, DenseNet-121, AlexNet, VGG-16, MobileNetV2
- Training: Standard supervised classification (CrossEntropyLoss)
- Input: Warped images from Stage 2
- Output: 3-class probability distribution
- Best Configuration: ResNet-18 on warped_96 achieves 99.10% accuracy
- Training: 200 epochs, early stopping patience=20
- Depends on: `torchvision.models`
- Used by: Clinical decision support, GUI

**Output Layer:**
- Purpose: Present final classification with confidence and explanation
- Location: `src_v2/gui/app.py` (Gradio interface) or CLI output
- Contains:
  - Predicted class (COVID-19, Normal, Viral Pneumonia)
  - Per-class probabilities
  - GradCAM visualization (if requested)
  - Landmark visualization (if debug enabled)
- Used by: End users via web interface or CLI

## Data Flow

**Training Pipeline (Landmarks):**

1. **Load Data** (`src_v2/data/dataset.py::LandmarkDataset`)
   - Read images from `data/dataset/COVID-19_Radiography_Dataset/`
   - Load landmarks from `coordenadas_maestro.csv`
   - Normalize landmarks to [0, 1] relative to original image size (299×299)
   - Create stratified train/val/test splits (75%/15%/10%, seed=42)

2. **Preprocess** (`src_v2/data/transforms.py`)
   - Apply CLAHE if `use_clahe=True` (default: clip=2.0, tile=4)
   - Resize to 224×224
   - Random augmentations (rotation, flip) for training
   - Normalize to ImageNet mean/std
   - Denormalize landmarks from [0, 1] to [0, 224] pixel coordinates

3. **Train Model** (`src_v2/training/trainer.py::LandmarkTrainer`)
   - Phase 1: Frozen backbone (15 epochs, lr=1e-3)
     - Optimize: WingLoss (wing regression with smooth approximation)
     - Monitor: Val loss for early stopping (patience=5)
   - Phase 2: Fine-tune entire model (100 epochs, differentiated LR)
     - Backbone LR: 2e-5 (very conservative)
     - Head LR: 2e-4 (faster adaptation)
     - Monitor: Val loss (patience=10)

4. **Evaluate** (`src_v2/evaluation/metrics.py`)
   - Compute pixel error per landmark (Euclidean distance in 224×224 space)
   - Optional: TTA (horizontal flip + symmetry pair swap)
   - Report: Mean, median, std per landmark; per-category stats
   - Validation: Compare to GROUND_TRUTH.json

**Warping Pipeline (Normalization):**

1. **Compute Canonical Shape** (`src_v2/processing/gpa.py::gpa_iterative`)
   - GPA on training set landmarks
   - Output: Mean shape + Delaunay triangulation
   - Saved to: `outputs/shape_analysis/`

2. **Predict Landmarks** (`src_v2/cli.py::predict-landmarks` or cached `.npz`)
   - Load trained landmark model ensemble
   - Optional: TTA with horizontal flip
   - Cache predictions in NPZ format:
     - `predictions`: (N, 15, 2) landmark coordinates
     - `image_paths`: Corresponding image file paths
     - Metadata: Model seeds, TTA flag, CLAHE settings

3. **Generate Warped Dataset** (`src_v2/cli.py::generate-dataset`)
   - Load cached predictions + canonical shape + Delaunay triangulation
   - For each image:
     - Scale landmarks from centroid by `margin_scale=1.05`
     - Build source/target point pairs via Delaunay triangles
     - Warp per triangle using cv2.warpAffine
     - Compose into output canvas
   - Output: `outputs/warped_lung_best/` (categorical subdirs)

**Classification Pipeline:**

1. **Train Classifier** (`src_v2/cli.py::train-classifier`)
   - Load warped dataset from `outputs/warped_lung_best/`
   - Split: train/val/test with stratified random state
   - Architecture: ImageClassifier (ResNet-18 by default)
   - Loss: CrossEntropyLoss
   - Augmentation: Random crop, color jitter, flip (standard augmentation)
   - Training: 200 epochs, early stopping (patience=20)
   - Output: `outputs/classifier_warped_lung_best/best_classifier.pt`

2. **Evaluate Classifier** (`src_v2/cli.py::evaluate-classifier`)
   - Load trained classifier + test set
   - Compute: Accuracy, precision, recall, F1-score, confusion matrix
   - Optional: Per-category breakdown, cross-validation

**Inference Pipeline (End-to-End):**

1. Load input image (JPEG, PNG)
2. **Landmark Detection:** ResNet18Landmarks ensemble → 15 coordinates
3. **Warping:** Piecewise affine using cached canonical shape
4. **Classification:** ImageClassifier → class probabilities
5. **Visualization:** GradCAM (optional), landmark overlay (optional)
6. **Output:** Predicted class + confidence + visualization

**State Management:**

- **Checkpoint System:** Models saved in `checkpoints/` after each epoch
  - Format: PyTorch `.pt` file with full state_dict
  - Metadata: Epoch, loss, architecture params (for auto-detection)
  - Critical files: `repro_split666/session16/seed666/final_model.pt` (best ensemble)

- **Configuration System:** JSON configs in `configs/`
  - `ensemble_best.json`: Best ensemble hyperparameters (3.61 px)
  - `warping_best.json`: Optimal warping parameters (margin_scale=1.05)
  - `classifier_warped_base.json`: Classifier defaults
  - Enables reproducibility without CLI flags

- **Cached Predictions:** `.npz` files in `outputs/landmark_predictions/`
  - Avoids re-inference during warping experiments
  - Format: numpy arrays + metadata

- **Canonical Shape Cache:** `outputs/shape_analysis/canonical_shape.npz`
  - Delaunay triangulation (edges)
  - Mean landmark coordinates
  - Valid triangle list (non-degenerate)

## Key Abstractions

**Generalized Procrustes Analysis (GPA):**
- Purpose: Compute canonical (mean) shape from set of landmark configurations
- Location: `src_v2/processing/gpa.py`
- Functions:
  - `center_shape()`: Remove translation
  - `scale_shape()`: Normalize to unit Frobenius norm
  - `optimal_rotation_matrix()`: SVD-based Procrustes rotation
  - `gpa_iterative()`: Full GPA iteration (alignment + consensus)
- Output: Canonical coordinates + Delaunay triangulation
- Critical for: Warping normalization pipeline

**Piecewise Affine Warping:**
- Purpose: Transform image to canonical geometry using Delaunay triangulation
- Location: `src_v2/processing/warp.py`
- Functions:
  - `piecewise_affine_warp()`: Main warping function
  - `scale_landmarks_from_centroid()`: Expand landmarks by margin factor
  - `add_boundary_points()`: Add image boundary to Delaunay for coverage
  - `compute_fill_rate()`: Validate triangle coverage
- Input: Image + predicted landmarks + canonical shape + Delaunay mesh
- Output: Warped 224×224 image with consistent geometry
- Critical parameters: `margin_scale=1.05` (5% expansion)

**Landmark Regression Model:**
- Purpose: Predict 15 lung contour landmarks
- Location: `src_v2/models/resnet_landmark.py::ResNet18Landmarks`
- Architecture Pattern:
  ```
  Backbone (ResNet-18, pretrained)
  └── Optional: CoordinateAttention (CVPR 2021)
      └── Head (FC layers, GroupNorm, dropout)
          └── Output: 30 coordinates in [0, 1]
  ```
- Training: Two-phase with frozen backbone → fine-tuning
- Loss: WingLoss (robust to outliers) or CombinedLoss (with symmetry penalties)
- Ensemble: Multiple models with different seeds, averaged predictions

**Coordinate Attention Module:**
- Purpose: Capture spatial relationships in heatmaps
- Location: `src_v2/models/resnet_landmark.py::CoordinateAttention`
- Pattern: Channel attention split into height and width branches
- Application: After ResNet-18 layer3, before fully connected head
- Impact: Improves spatial accuracy for landmark prediction

**Classification Model:**
- Purpose: COVID-19 vs Normal vs Viral Pneumonia classification
- Location: `src_v2/models/classifier.py::ImageClassifier`
- Architecture: Transfer learning (7 backbone options)
  - Backbone: Pretrained on ImageNet (ResNet, EfficientNet, DenseNet, etc.)
  - Head: Dropout → Linear(num_features → 3 classes)
- Input: Warped 224×224 images (geometrically normalized)
- Output: 3-class logits → softmax probabilities
- Training: Standard supervised learning (CrossEntropyLoss)

**Loss Functions:**
- Location: `src_v2/models/losses.py`
- **WingLoss:** Robust regression loss for landmarks
  - Smooth near zero, linear away
  - Formula: w*ln(1 + |x|/epsilon) for |x| < omega; else w*(|x| - omega) + w*ln(omega/epsilon)
  - Parameters: omega=DEFAULT_WING_OMEGA, epsilon=DEFAULT_WING_EPSILON
  - Use: Landmark training (default)
- **CombinedLoss:** WingLoss + symmetry penalty
  - Enforces symmetric landmark pairs (L3↔L4, L5↔L6, L7↔L8, L12↔L13, L14↔L15)
  - Penalty: Weight × MSE between paired landmarks
  - Use: Optional, when symmetry is important
- **CrossEntropyLoss:** Classification loss (torch.nn.CrossEntropyLoss)
  - Use: Classifier training

**Two-Phase Training Pattern:**
- Location: `src_v2/training/trainer.py::LandmarkTrainer`
- Phase 1 (Frozen Backbone):
  - Freeze: All ResNet-18 layers
  - Train: Only FC head with high learning rate
  - Duration: 15 epochs
  - Learning Rate: 1e-3
  - Rationale: Initialize head on pre-trained backbone features
- Phase 2 (Fine-Tuning):
  - Unfreeze: All layers
  - Train: Entire model with differentiated learning rates
  - Backbone LR: 2e-5 (conservative - preserve ImageNet features)
  - Head LR: 2e-4 (faster - adapt to domain)
  - Duration: 100 epochs
  - Rationale: Gradual adaptation with backbone preservation
- Early Stopping: Per-phase (Phase1 patience=5, Phase2 patience=10)

**Data Splitting Strategy:**
- Location: `src_v2/data/dataset.py::create_dataloaders`
- Method: Stratified random split by category
- Split Seed: Fixed at 42 (for reproducibility across all experiments)
- Model Seed: Varies (affects initialization only, not splits)
- Ratio: train=75%, val=15%, test=10%
- Fallback: Non-stratified split if category counts too small
- Purpose: Ensure category balance across splits

**Test-Time Augmentation (TTA):**
- Location: `src_v2/evaluation/metrics.py::compute_pixel_error` (with `use_tta=True`)
- Method: Horizontal flip + symmetric pair swap
- Symmetric Pairs: (L3↔L4), (L5↔L6), (L7↔L8), (L12↔L13), (L14↔L15)
- Ensemble: Average predictions from original and flipped images
- Impact: Marginal improvement in robustness (1-2%)
- Use: Optional in evaluation and inference

**Cached Prediction System:**
- Location: `scripts/predict_landmarks_dataset.py`
- Format: NumPy `.npz` archive containing:
  - `predictions`: (N, 15, 2) float32 landmark coordinates
  - `image_paths`: (N,) object array with image file paths
  - Metadata: Model seeds, TTA flag, CLAHE parameters
- Purpose: Avoid re-running inference during warping experiments
- Benefit: 10-100× speedup for dataset processing
- Used by: `generate-dataset` command (reads `.npz`, no re-inference)

## Entry Points

**CLI Entry Point:**
- Location: `src_v2/__main__.py`
- Triggers: `python -m src_v2 [command] [options]`
- Responsibilities:
  - Load CLI from `src_v2.cli:main`
  - Route to appropriate command handler
  - Set up logging with optional verbose mode

**Primary Commands (src_v2/cli.py):**

`train` - Landmark model training
- Location: `src_v2/cli.py::train()`
- Triggers: `python -m src_v2 train --data-root data/ --csv-path ...`
- Responsibilities:
  - Load dataset (train/val/test splits)
  - Create model with specified architecture
  - Execute two-phase training
  - Save checkpoints and metrics

`evaluate` - Landmark model evaluation
- Location: `src_v2/cli.py::evaluate()`
- Triggers: `python -m src_v2 evaluate checkpoint.pt`
- Responsibilities:
  - Load model and dataset
  - Compute pixel error metrics
  - Optional TTA evaluation
  - Output JSON results

`predict` - Single image landmark prediction
- Location: `src_v2/cli.py::predict()`
- Triggers: `python -m src_v2 predict image.png --checkpoint model.pt`
- Responsibilities:
  - Load image and model
  - Predict 15 landmarks
  - Optional visualization overlay

`compute-canonical` - GPA canonical shape
- Location: `src_v2/cli.py::compute-canonical()`
- Triggers: `python -m src_v2 compute-canonical coordenadas_maestro.csv --output-dir outputs/shape_analysis`
- Responsibilities:
  - Load training landmarks
  - Run GPA iteration
  - Compute Delaunay triangulation
  - Save canonical shape to NPZ

`generate-dataset` - Warped dataset generation
- Location: `src_v2/cli.py::generate-dataset()`
- Triggers: `python -m src_v2 generate-dataset --config configs/warping_best.json`
- Responsibilities:
  - Load canonical shape
  - Load cached landmark predictions (or predict if missing)
  - Warp each image using Delaunay
  - Output categorical directory structure

`train-classifier` - COVID classification training
- Location: `src_v2/cli.py::train-classifier()`
- Triggers: `python -m src_v2 train-classifier --config configs/classifier_warped_base.json`
- Responsibilities:
  - Load warped dataset
  - Create classifier model
  - Standard supervised training
  - Save best model

`evaluate-classifier` - Classifier evaluation
- Location: `src_v2/cli.py::evaluate-classifier()`
- Triggers: `python -m src_v2 evaluate-classifier model.pt --data-dir warped_dataset/`
- Responsibilities:
  - Load classifier and test set
  - Compute classification metrics
  - Output confusion matrix, per-class stats

**GUI Entry Point:**
- Location: `src_v2/gui/app.py`
- Triggers: `python -m src_v2.gui.app` (or via `run_demo.sh`)
- Responsibilities:
  - Launch Gradio interface
  - Provide 3 tabs: Full demo, Quick view, About
  - Handle image upload and inference
  - Display results with GradCAM overlay

## Error Handling

**Strategy:** Explicit error checking with early exit + informative logging

**Patterns:**

1. **File Existence:** Check before use
   ```python
   if not Path(checkpoint).exists():
       logger.error("Checkpoint not found: %s", checkpoint)
       raise typer.Exit(code=1)
   ```

2. **Device Fallback:** Auto-detect GPU/MPS/CPU
   ```python
   if device == "auto":
       if torch.cuda.is_available():
           return torch.device("cuda")
       elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
           return torch.device("mps")
       else:
           return torch.device("cpu")
   ```

3. **Multiprocessing Sandbox Fix:** Graceful fallback to single-threaded
   ```python
   if os.environ.get("FORCE_NUM_WORKERS_ZERO"):
       return 0  # Single worker for restricted environments
   ```

4. **Data Splits:** Stratified with fallback to non-stratified
   ```python
   try:
       return train_test_split(..., stratify=dataframe[stratify_col])
   except ValueError as exc:
       logger.warning("Fallback to non-stratified split (%s)", exc)
       return train_test_split(..., stratify=None)
   ```

5. **Checkpoint Loading:** Auto-detect architecture from weights
   ```python
   arch_params = detect_architecture_from_checkpoint(state_dict)
   # Detects: use_coord_attention, deep_head, hidden_dim
   model = create_model(**arch_params)
   ```

**Validation:** GROUND_TRUTH.json reference values
- Landmark ensemble: 3.61 px error (Session 16, seed 666)
- Classifier accuracy: 99.10% (warped_96 config)
- Compare results to validate pipeline correctness

## Cross-Cutting Concerns

**Logging:**
- Framework: Python `logging` module
- Levels: DEBUG (verbose flag), INFO (default), WARNING (issues), ERROR (failures)
- Configuration: Set in `src_v2/cli.py::verbose_callback()` (eager flag processing)
- Format: `"%(asctime)s - %(name)s - %(levelname)s - %(message)s"`
- Usage: Log progress, hyperparameters, metrics, errors

**Validation:**
- Landmark coordinates: Normalized to [0, 1] during training, denormalized to pixels in evaluation
- Image sizes: Support multiple sizes (original 299, training 224) with proper scaling
- Splits: Stratified by category; seed fixed for reproducibility
- Metrics: Compare to GROUND_TRUTH.json for correctness

**Authentication:** Not applicable (research code, no external services)

**Performance:**
- Landmark inference: ~100ms per image (ResNet-18 on GPU)
- Warping: ~50ms per image (Delaunay + cv2.warpAffine)
- Classification: ~50ms per image (ResNet-18 on GPU)
- Caching: Landmark NPZ caches avoid re-inference (10× speedup for batch processing)

**Reproducibility:**
- Seed Management:
  - Dataset split seed: Fixed at 42 (for consistent train/val/test)
  - Model init seed: Varies per experiment (affects weight initialization)
  - NumPy/PyTorch: Set in `train()` command (random.seed, np.random.seed, torch.manual_seed)
- Configuration: JSON configs bundle all hyperparameters (no hardcoded values)
- Version Control: `GROUND_TRUTH.json` stores validated metrics for each experiment

**Scalability:**
- Batch Processing: DataLoader with configurable batch size and num_workers
- GPU Acceleration: Auto-detect device (CUDA, MPS, CPU)
- Distributed Training: Not implemented (single-GPU design)
- Memory: Optimize via batch size tuning and gradient accumulation (optional)

---

*Architecture analysis: 2026-01-27*
