# Architecture Research

**Domain:** Ensemble + TTA Evaluation for Medical Image Classification
**Researched:** 2026-01-27
**Confidence:** HIGH

## Standard Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     Evaluation Orchestrator                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Config     │  │  Checkpoint  │  │   Dataset    │           │
│  │   Loader     │  │   Locator    │  │   Loader     │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                  │                  │                   │
├─────────┴──────────────────┴──────────────────┴───────────────────┤
│                      Inference Pipeline                           │
├──────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                   Model Ensemble Manager                   │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │
│  │  │ Model 1  │  │ Model 2  │  │ Model 3  │  │ Model N  │  │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │   │
│  └───────┼──────────────┼──────────────┼──────────────┼────────┘   │
│          │              │              │              │            │
│  ┌───────┴──────────────┴──────────────┴──────────────┴────────┐  │
│  │                 TTA Augmentation Engine                      │  │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │  │
│  │   │  Original   │  │    Flip     │  │   Future    │        │  │
│  │   │   Image     │  │  + Correct  │  │ Augmentations│        │  │
│  │   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │  │
│  └──────────┼─────────────────┼─────────────────┼──────────────┘  │
│             │                 │                 │                 │
├─────────────┴─────────────────┴─────────────────┴─────────────────┤
│                     Aggregation Layer                             │
├──────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐         ┌────────────────┐                   │
│  │  Soft Voting   │         │  Prediction    │                   │
│  │  (Probability  │   →     │  Integration   │                   │
│  │   Averaging)   │         │  & Validation  │                   │
│  └────────┬───────┘         └────────┬───────┘                   │
├───────────┴──────────────────────────┴───────────────────────────┤
│                    Analysis & Metrics Layer                       │
├──────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │Accuracy/ │  │Per-Class │  │Confusion │  │Confidence│         │
│  │F1/Recall │  │ Metrics  │  │  Matrix  │  │ Analysis │         │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘         │
├──────────────────────────────────────────────────────────────────┤
│                    Visualization Layer                            │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Report Generation, Plots, Comparison Tables             │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **Config Loader** | Parse evaluation parameters (fold paths, TTA settings, metrics) | JSON/YAML parser with validation schema |
| **Checkpoint Locator** | Find and validate all N model checkpoint files | Path resolution + existence checks |
| **Dataset Loader** | Load test images with preprocessing pipeline | PyTorch DataLoader with transforms |
| **Model Ensemble Manager** | Load N models into memory, manage device placement | List of model instances on GPU/CPU |
| **TTA Augmentation Engine** | Apply transformations, handle symmetric correction | Per-image augmentation with landmark-aware flipping |
| **Soft Voting** | Average probability distributions across models+TTA | torch.mean over stacked predictions |
| **Prediction Integration** | Combine ensemble+TTA outputs, apply argmax | Probability aggregation + final decision |
| **Analysis & Metrics** | Compute accuracy, F1, confusion matrix, per-class stats | scikit-learn metrics + custom COVID domain metrics |
| **Visualization** | Generate plots, tables, comparison charts | matplotlib/seaborn, match thesis style |

## Recommended Project Structure

```
src_v2/
├── evaluation/
│   ├── ensemble.py           # EnsembleEvaluator class
│   ├── metrics.py            # Existing metrics utilities (reuse)
│   └── visualization.py      # Plot generation for ensemble results
├── models/
│   ├── classifier.py         # Existing ImageClassifier (reuse)
│   └── ensemble_wrapper.py   # Multi-model container
├── cli.py                    # Add evaluate-classifier-ensemble command
└── data/
    └── dataset.py            # Existing dataloader (reuse)

scripts/
└── evaluate_classifier_ensemble.py   # Standalone script for ad-hoc runs

configs/
└── classifier_ensemble_eval.json     # Ensemble evaluation config
```

### Structure Rationale

- **src_v2/evaluation/ensemble.py:** Central orchestrator for ensemble+TTA evaluation. Encapsulates loading models, iterating dataset, aggregating predictions, and computing metrics. Separates concerns from CLI.

- **src_v2/models/ensemble_wrapper.py:** Lightweight container holding list of N classifiers. Provides unified interface for batch prediction across all models. Handles device management and memory optimization.

- **src_v2/cli.py:** Thin CLI layer that parses arguments and delegates to EnsembleEvaluator. Keeps command-line interface consistent with existing patterns (`evaluate-ensemble` for landmarks).

- **configs/classifier_ensemble_eval.json:** Declarative configuration for reproducibility. Stores fold checkpoint paths, TTA settings, batch size, device, output directory. Avoids hardcoding in code.

## Architectural Patterns

### Pattern 1: Model Pool with Lazy Loading

**What:** Load models into memory pool, but support lazy loading if memory-constrained. For 5x ResNet-18 models (~44 MB each), full loading is feasible. For larger ensembles, load one at a time.

**When to use:** When ensemble size N > 10 or using larger backbones (ResNet-50, DenseNet-121).

**Trade-offs:**
- Pros: Memory efficient for large ensembles
- Cons: Slower inference (repeated disk I/O), more complex code

**Example:**
```python
class EnsembleWrapper:
    def __init__(self, checkpoint_paths, lazy_load=False):
        self.checkpoint_paths = checkpoint_paths
        self.lazy_load = lazy_load
        self.models = None if lazy_load else self._load_all_models()

    def predict(self, batch):
        if self.lazy_load:
            # Load each model, predict, unload
            predictions = []
            for ckpt in self.checkpoint_paths:
                model = load_model(ckpt)
                pred = model(batch)
                predictions.append(pred)
                del model  # Free memory
            return torch.stack(predictions)
        else:
            # Use pre-loaded models
            return torch.stack([m(batch) for m in self.models])
```

### Pattern 2: TTA as Transform Pipeline

**What:** Treat TTA augmentations as a composable pipeline. Each augmentation generates a view, all views pass through model(s), results are aggregated.

**When to use:** When supporting multiple TTA strategies beyond horizontal flip (rotation, scale, intensity).

**Trade-offs:**
- Pros: Extensible to new augmentations, clean separation of concerns
- Cons: More overhead for simple flip-only TTA

**Example:**
```python
class TTAStrategy:
    def __init__(self, augmentations=['horizontal_flip']):
        self.augmentations = augmentations

    def apply(self, image, model):
        """Apply TTA and return averaged predictions."""
        predictions = []

        # Original
        predictions.append(model(image))

        # Horizontal flip (with symmetric correction for landmarks if needed)
        if 'horizontal_flip' in self.augmentations:
            flipped = torch.flip(image, dims=[3])
            pred_flipped = model(flipped)
            predictions.append(pred_flipped)

        # Future: rotation, scale, etc.

        return torch.stack(predictions).mean(dim=0)
```

### Pattern 3: Stratified Batch Evaluation

**What:** Evaluate ensemble by stratifying test set by class. Compute per-class metrics before global aggregation. Essential for imbalanced medical datasets.

**When to use:** Always for medical imaging (COVID/Normal/Viral_Pneumonia have different distributions).

**Trade-offs:**
- Pros: Reveals class-specific performance, catches bias
- Cons: Slightly more bookkeeping

**Example:**
```python
def evaluate_stratified(ensemble, dataloader, class_names):
    """Evaluate with per-class tracking."""
    predictions_by_class = {cls: [] for cls in class_names}
    targets_by_class = {cls: [] for cls in class_names}

    for batch_images, batch_labels in dataloader:
        preds = ensemble.predict(batch_images)  # (B, C) probabilities

        for pred, label in zip(preds, batch_labels):
            cls_name = class_names[label]
            predictions_by_class[cls_name].append(pred)
            targets_by_class[cls_name].append(label)

    # Compute metrics per class
    results = {}
    for cls in class_names:
        preds = torch.stack(predictions_by_class[cls])
        targets = torch.tensor(targets_by_class[cls])
        results[cls] = compute_metrics(preds, targets)

    return results
```

## Data Flow

### Inference Flow

```
Test Dataset (1,895 images)
    ↓
DataLoader (batch_size=32)
    ↓
For each batch:
    ↓
┌───────────────────────────────────────┐
│ Original Image (B, 3, 224, 224)       │
├───────────────────────────────────────┤
│                                       │
│  ┌─────────────┐  ┌─────────────┐    │
│  │  Original   │  │   Flipped   │    │
│  │    View     │  │    View     │    │
│  └──────┬──────┘  └──────┬──────┘    │
│         │                 │           │
│    For each model in ensemble:       │
│         │                 │           │
│    ┌────▼────┐       ┌────▼────┐     │
│    │ Model i │       │ Model i │     │
│    │ Predict │       │ Predict │     │
│    └────┬────┘       └────┬────┘     │
│         │                 │           │
│    Pred_orig_i       Pred_flip_i     │
│         │                 │           │
│         └────────┬────────┘           │
│                  │                    │
│            Mean(TTA views)            │
│                  │                    │
│             Pred_i (B, C)             │
└──────────────────┼────────────────────┘
                   │
         Stack predictions from all models
                   │
                   ▼
        Pred_ensemble (N, B, C)
                   │
                   ▼
           Mean(ensemble axis)
                   │
                   ▼
         Final_pred (B, C) probabilities
                   │
                   ▼
              Argmax → Class labels
                   │
                   ▼
         Compare with ground truth
                   │
                   ▼
        Accumulate metrics (TP, FP, FN, TN)
```

### Configuration Flow

```
User CLI Command
    ↓
evaluate-classifier-ensemble --config ensemble_eval.json
    ↓
Load JSON config
    ↓
Parse: checkpoint_paths, tta_enabled, batch_size, device
    ↓
Pass to EnsembleEvaluator(config)
    ↓
EnsembleEvaluator.run()
    ↓
Generate report + visualizations → outputs/
```

### Key Data Flows

1. **Model Loading Flow:** Config → Checkpoint paths → Load each model → Validate architecture → Place on device → Store in model pool

2. **Batch Inference Flow:** Image batch → Apply TTA transforms → Pass through N models → Stack predictions (N, B, C) → Average over N → Average over TTA → Final probabilities

3. **Metrics Flow:** Predictions (N_samples, C) + Labels (N_samples,) → Argmax → Accuracy, F1, Recall, Precision → Per-class breakdown → Confusion matrix → Save to JSON + CSV

4. **Visualization Flow:** Metrics dict → Generate confusion matrix heatmap → Per-class bar charts → Confidence histograms → Save PNG/PDF matching thesis style

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 5-10 models | Load all models in memory, evaluate in parallel if multi-GPU |
| 10-50 models | Use lazy loading pattern, process sequentially per model |
| 50+ models | Distributed evaluation with model sharding across nodes |

### Scaling Priorities

1. **First bottleneck:** GPU memory for large ensembles. Fix: Lazy loading + CPU offloading for inactive models.

2. **Second bottleneck:** Inference time for TTA (2x slowdown per augmentation). Fix: Batch TTA views together, use DataParallel for multi-GPU.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Existing metrics.py | Import and reuse `compute_pixel_error`, `evaluate_model` | Already handles batch processing, device management |
| Existing classifier.py | Use `create_classifier()` factory function | Handles checkpoint loading, architecture detection |
| Existing dataset.py | Reuse `LandmarkDataset` + transforms | May need classifier-specific transform pipeline |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| CLI ↔ EnsembleEvaluator | Config dict (JSON-like) | CLI parses args → dict → EnsembleEvaluator |
| EnsembleEvaluator ↔ EnsembleWrapper | Batch tensors (B, 3, H, W) | Evaluator passes batch, Wrapper returns (B, C) probabilities |
| EnsembleWrapper ↔ Individual Models | Batch tensors | Wrapper iterates models, stacks outputs |
| EnsembleEvaluator ↔ Visualization | Metrics dict | Evaluator computes metrics, Visualization renders plots |

## Anti-Patterns

### Anti-Pattern 1: Hardcoding Checkpoint Paths

**What people do:** Embed fold checkpoint paths directly in script:
```python
models = [
    'checkpoints/fold_01/best.pt',
    'checkpoints/fold_02/best.pt',
    # ...
]
```

**Why it's wrong:** Not reproducible, brittle to directory structure changes, can't compare different ensembles.

**Do this instead:** Use JSON config:
```json
{
  "ensemble_name": "5fold_resnet18",
  "checkpoints": [
    "checkpoints/fold_01/best.pt",
    "checkpoints/fold_02/best.pt"
  ],
  "tta": true
}
```

### Anti-Pattern 2: Re-implementing TTA Logic per Model Type

**What people do:** Write separate TTA code for classifier vs landmark models:
```python
# TTA for landmarks
def predict_landmarks_tta(model, img):
    pred1 = model(img)
    pred2 = model(flip(img))
    pred2 = correct_symmetric_pairs(pred2)
    return (pred1 + pred2) / 2

# TTA for classifier (duplicated logic!)
def predict_classifier_tta(model, img):
    pred1 = model(img)
    pred2 = model(flip(img))
    return (pred1 + pred2) / 2
```

**Why it's wrong:** Code duplication, inconsistent behavior, harder to maintain.

**Do this instead:** Use unified TTA abstraction with optional post-processing:
```python
class TTAEngine:
    def __init__(self, augmentations, post_process_fn=None):
        self.augmentations = augmentations
        self.post_process_fn = post_process_fn

    def predict(self, model, image):
        preds = [model(image)]
        for aug in self.augmentations:
            aug_img = aug.apply(image)
            aug_pred = model(aug_img)
            if self.post_process_fn:
                aug_pred = self.post_process_fn(aug_pred, aug)
            preds.append(aug_pred)
        return torch.stack(preds).mean(dim=0)
```

### Anti-Pattern 3: Mixing Training and Evaluation Code

**What people do:** Put evaluation logic inside training scripts, leading to:
```python
# train_classifier.py
if args.evaluate:
    # 200 lines of evaluation code mixed with training
    evaluate_model(...)
```

**Why it's wrong:** Violates single responsibility principle, can't evaluate without training dependencies, harder to test.

**Do this instead:** Separate evaluation into dedicated module/script:
```
src_v2/evaluation/ensemble.py  # Pure evaluation logic
scripts/evaluate_ensemble.py   # CLI wrapper
```

## Error Handling and Validation Strategies

### Critical Validation Points

1. **Checkpoint Loading:**
   - Validate file exists before loading
   - Check architecture compatibility (all models same backbone + num_classes)
   - Verify model.eval() mode is set
   - Catch corrupted checkpoint errors gracefully

2. **TTA Correctness:**
   - Verify flip dimension (horizontal=dim[3] for NCHW format)
   - For landmarks: validate symmetric pair indices exist
   - For classifiers: ensure no post-processing needed (flip-invariant classes)

3. **Prediction Shape Consistency:**
   - Assert all models output same shape (B, num_classes)
   - Validate batch size matches between image and label tensors
   - Check probability distribution sums to 1.0 (after softmax)

4. **Metric Computation:**
   - Handle empty classes gracefully (e.g., if test set missing a class)
   - Verify confusion matrix dimensions match num_classes
   - Check for NaN/Inf in metrics (division by zero)

### Error Handling Pattern

```python
class EnsembleEvaluator:
    def load_models(self, checkpoint_paths):
        """Load models with robust error handling."""
        models = []
        for i, ckpt_path in enumerate(checkpoint_paths):
            try:
                if not Path(ckpt_path).exists():
                    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

                model = create_classifier(checkpoint=ckpt_path, device=self.device)
                model.eval()

                # Validate architecture
                if i == 0:
                    self.reference_arch = model.backbone_name
                    self.num_classes = model.num_classes
                else:
                    if model.backbone_name != self.reference_arch:
                        raise ValueError(f"Architecture mismatch: {model.backbone_name} != {self.reference_arch}")
                    if model.num_classes != self.num_classes:
                        raise ValueError(f"Num classes mismatch: {model.num_classes} != {self.num_classes}")

                models.append(model)
                logger.info(f"Loaded model {i+1}/{len(checkpoint_paths)}: {ckpt_path}")

            except Exception as e:
                logger.error(f"Failed to load model {i+1}: {e}")
                raise

        return models
```

## Build Order and Dependencies

### Phase 1: Core Infrastructure (Dependencies: None)
- `src_v2/models/ensemble_wrapper.py`: Container for multiple models
- `configs/classifier_ensemble_eval.json`: Configuration schema

### Phase 2: Evaluation Logic (Dependencies: Phase 1 + existing src_v2/evaluation/metrics.py)
- `src_v2/evaluation/ensemble.py`: EnsembleEvaluator class
- Reuse existing `src_v2/evaluation/metrics.py` for base metrics

### Phase 3: CLI Integration (Dependencies: Phase 1 + 2)
- Add `evaluate-classifier-ensemble` command to `src_v2/cli.py`
- Standalone script `scripts/evaluate_classifier_ensemble.py`

### Phase 4: Visualization (Dependencies: Phase 2)
- `src_v2/evaluation/visualization.py`: Plot generation
- Match thesis style (fonts, colors, layout)

### Phase 5: Analysis Tools (Dependencies: Phase 2 + 4)
- Comparison scripts (ensemble vs individual models)
- Statistical significance testing (McNemar's test, bootstrap CI)

## Comparison: Training vs Inference Architecture

| Aspect | Training Time | Inference Time (This Project) |
|--------|---------------|-------------------------------|
| Model lifecycle | Create → Train → Save checkpoint | Load checkpoint → Freeze |
| Data augmentation | Random (flip, rotate, crop) | Deterministic TTA (flip only) |
| Loss computation | Cross-entropy, backprop | No loss, forward-only |
| Batch processing | Variable batch size, gradient accumulation | Fixed batch size, no gradients |
| Metrics | Training loss, validation accuracy per epoch | Final test set metrics only |
| Ensemble | Not applicable (train individual models) | Load N models, aggregate predictions |
| Output | Checkpoint files (.pt) | Metrics JSON, plots, reports |

**Key Distinction:** This project focuses on **inference-only evaluation**. No model weights are updated. No training loops. No gradient computation. This simplifies architecture significantly compared to training pipelines.

## Sources

Research findings based on:
- [Advanced Multi-architecture Deep Learning Framework for BIRADS-Based Mammographic Image Retrieval](https://link.springer.com/article/10.1007/s10278-025-01770-6)
- [Medical Image Segmentation: A Comprehensive Review of Deep Learning-Based Methods - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12115501/)
- [A Large Scale Benchmark for Test Time Adaptation Methods in Medical Image Segmentation](https://arxiv.org/html/2512.02497v1)
- [S³-TTA: Scale-Style Selection for Test-Time Augmentation in Biomedical Image Segmentation](https://arxiv.org/html/2310.16783v1)
- [Understanding Test-Time Augmentation](https://arxiv.org/html/2402.06892v1)
- [Ensemble-PyTorch: A unified ensemble framework](https://github.com/TorchEnsemble-Community/Ensemble-Pytorch)
- [VotingClassifier — scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)
- [Breast Cancer Detection with Ensemble Models](https://github.com/richardcornall/Tensorflow-Pytorch-Ensemble-Machine-Learning-Model-for-Breast-Cancer-Detection-)

---
*Architecture research for: Ensemble + TTA Evaluation for Medical Image Classification*
*Researched: 2026-01-27*
