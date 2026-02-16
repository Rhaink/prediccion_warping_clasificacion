# Phase 3: TTA Integration - Research

**Researched:** 2026-01-27
**Domain:** Test-Time Augmentation for Medical Image Classification
**Confidence:** HIGH

## Summary

Test-time augmentation (TTA) for chest X-ray classification is a well-established technique that improves model robustness by averaging predictions across transformed inputs. For conservative medical imaging applications, horizontal flip is the standard augmentation - anatomically valid (lungs are bilaterally symmetric) and proven effective in recent studies showing 0.25-0.51% AUC improvements.

The existing codebase already implements TTA for landmark prediction (`src_v2/evaluation/metrics.py::predict_with_tta`), providing validated infrastructure including horizontal flip with `torch.flip(images, dims=[3])` and simple averaging. This infrastructure can be adapted directly for classifier TTA with minimal modifications.

**Primary recommendation:** Implement dual-level TTA (model-level + ensemble-level) using simple averaging. For statistical validation, use McNemar's test via `mlxtend.evaluate.mcnemar` to verify improvement significance. Track per-class performance with `sklearn.metrics.classification_report` for full traceability.

## Standard Stack

The established libraries/tools for TTA in PyTorch medical imaging:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.x | Tensor operations, horizontal flip | Native `torch.flip(dims=[3])` for horizontal flip |
| scikit-learn | 1.8.0+ | Metrics (confusion matrix, classification report) | Industry standard for ML metrics, stable API |
| NumPy | Latest | Array operations | Required by scikit-learn for metric computation |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| mlxtend | Latest | McNemar's test | Statistical significance testing of classifier improvements |
| ttach | 0.0.3 | Optional TTA wrapper library | Alternative if custom implementation not needed (Phase 3 uses custom) |
| torchmetrics | Latest | PyTorch-native confusion matrix | Alternative to scikit-learn if staying in PyTorch ecosystem |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom TTA implementation | ttach.ClassificationTTAWrapper | ttach adds dependency but simplifies code. Custom gives full control for dual-level TTA |
| scikit-learn metrics | torchmetrics | torchmetrics stays in PyTorch tensors, scikit-learn more mature/stable for reporting |
| Simple average | Weighted average | Weighted adds complexity without clear benefit for symmetric geometric transforms |

**Installation:**
```bash
# Core (already in project)
pip install torch torchvision scikit-learn numpy

# For statistical testing (new dependency)
pip install mlxtend
```

## Architecture Patterns

### Recommended Project Structure
```
src_v2/
├── evaluation/
│   ├── metrics.py           # Extend with classifier TTA functions
│   ├── ensemble.py          # Add TTA logic to ensemble_inference()
│   └── statistical.py       # NEW: McNemar's test, case-level analysis
├── models/
│   └── classifier.py        # Already has predict_proba() for soft voting
└── cli.py                   # Add --tta/--no-tta flags
```

### Pattern 1: Dual-Level TTA (Model + Ensemble)
**What:** Apply TTA at both individual model level AND ensemble level
**When to use:** When using ensemble of models (5 CV folds in this project)
**Example:**
```python
# Source: Based on src_v2/evaluation/metrics.py::predict_with_tta (landmark TTA)
import torch

@torch.no_grad()
def predict_with_tta_classifier(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Model-level TTA: average predictions from original + flipped images.

    Args:
        model: Classifier model
        images: Batch (B, 3, H, W)
        device: Device

    Returns:
        Averaged probabilities (B, num_classes)
    """
    model.eval()
    images = images.to(device)

    # Original prediction
    logits_orig = model(images)
    probs_orig = torch.softmax(logits_orig, dim=1)

    # Flipped prediction
    images_flipped = torch.flip(images, dims=[3])  # Horizontal flip (W dimension)
    logits_flip = model(images_flipped)
    probs_flip = torch.softmax(logits_flip, dim=1)

    # Simple average (no symmetry correction needed for class labels)
    return (probs_orig + probs_flip) / 2
```

**Pipeline:**
```
For each image in test set:
  For each of 5 CV fold models:
    1. Forward pass: original image → logits → softmax → prob_orig
    2. Forward pass: flipped image → logits → softmax → prob_flip
    3. Model-level TTA: (prob_orig + prob_flip) / 2 → model_tta_prob

  Ensemble-level averaging: mean([model1_tta, ..., model5_tta]) → final_prob
  Prediction: argmax(final_prob) → predicted_class
```

### Pattern 2: JSON-Driven Configuration
**What:** Top-level `use_tta` parameter in config, CLI override capability
**When to use:** Allowing runtime control without code changes
**Example:**
```json
// configs/ensemble_classifier.json
{
  "use_tta": true,  // Default: enabled
  "checkpoint_paths": [...],
  "data_dir": "outputs/warped_lung_best/session_warping",
  "split": "test"
}
```

```python
# CLI override pattern (Click framework already in src_v2/cli.py)
import click

@click.command()
@click.option('--config', required=True, type=click.Path(exists=True))
@click.option('--tta/--no-tta', default=None, help='Override config use_tta setting')
def evaluate_ensemble_classifier(config: str, tta: Optional[bool]):
    cfg = json.load(open(config))
    use_tta = tta if tta is not None else cfg.get('use_tta', True)
    # ... evaluation logic
```

### Pattern 3: Delta Metrics with Dual Evaluation
**What:** Run evaluation twice internally (with/without TTA) and report delta
**When to use:** Demonstrating TTA improvement transparently
**Example:**
```python
# Source: Pattern from medical imaging research best practices
def evaluate_with_tta_comparison(models, dataloader, device):
    """Evaluate both with and without TTA, compute delta metrics."""

    # Baseline: no TTA
    results_baseline = ensemble_inference(
        models, dataloader, device, use_tta=False
    )

    # With TTA
    results_tta = ensemble_inference(
        models, dataloader, device, use_tta=True
    )

    # Delta metrics
    delta = {
        'accuracy_delta': results_tta['accuracy'] - results_baseline['accuracy'],
        'f1_macro_delta': results_tta['f1_macro'] - results_baseline['f1_macro'],
        'per_class_delta': {
            cls: results_tta['per_class'][cls]['f1'] - results_baseline['per_class'][cls]['f1']
            for cls in ['COVID', 'Normal', 'Viral_Pneumonia']
        }
    }

    return {
        'baseline_no_tta': results_baseline,
        'with_tta': results_tta,
        'improvement': delta
    }
```

### Pattern 4: Case-Level Impact Tracking
**What:** Per-image tracking of TTA effect (helped/hurt/neutral)
**When to use:** Understanding when TTA improves vs degrades predictions
**Example:**
```python
# Categorize TTA impact per image
def categorize_tta_impact(pred_baseline, pred_tta, ground_truth):
    """
    Returns: 'helped', 'hurt', or 'neutral'
    """
    baseline_correct = (pred_baseline == ground_truth)
    tta_correct = (pred_tta == ground_truth)

    if not baseline_correct and tta_correct:
        return 'helped'  # Baseline wrong → TTA correct
    elif baseline_correct and not tta_correct:
        return 'hurt'    # Baseline correct → TTA wrong
    else:
        return 'neutral' # Both correct OR both wrong
```

### Anti-Patterns to Avoid
- **Symmetry correction for class labels:** Unlike landmarks (L/R swap), disease classes are symmetric. Don't swap labels after flip.
- **Weighted aggregation without validation:** Weighted TTA (e.g., learned weights) adds complexity. Simple average is standard for geometric transforms.
- **Excessive augmentations:** Rotation, scaling, brightness jitter are NOT conservative for medical imaging. Stick to horizontal flip only.
- **Single-level TTA:** Applying TTA only at ensemble level misses per-model variance reduction benefits.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| McNemar's statistical test | Custom chi-squared implementation | `mlxtend.evaluate.mcnemar()` | Handles small sample corrections (exact test if cell < 25), provides both statistic and p-value |
| Confusion matrix visualization | Custom matplotlib plotting | `sklearn.metrics.confusion_matrix()` + `ConfusionMatrixDisplay` | Normalized/unnormalized options, standard format for papers |
| Per-class metrics (precision/recall/F1) | Manual TP/FP/FN counting | `sklearn.metrics.classification_report(output_dict=True)` | Handles class imbalance, supports multiple averaging modes |
| Prediction stability metric (KL divergence) | Custom KL computation | `torch.nn.functional.kl_div()` with log-softmax | Numerically stable, handles edge cases (log(0)) |

**Key insight:** Medical imaging research demands reproducibility. Using established libraries (scikit-learn, mlxtend) ensures results are comparable to published literature and prevents implementation bugs in statistical tests.

## Common Pitfalls

### Pitfall 1: Class Label Confusion with Symmetry
**What goes wrong:** Developers apply landmark-style symmetry correction (swapping L/R indices) to classifier predictions
**Why it happens:** Codebase has `SYMMETRIC_PAIRS` constant and `_flip_landmarks_horizontal()` function, tempting to reuse
**How to avoid:**
- Class labels (COVID, Normal, Viral_Pneumonia) are anatomically symmetric - a flipped COVID X-ray is still COVID
- NO label swapping after horizontal flip for classification
- Only flip image tensor, directly average probabilities
**Warning signs:** Seeing `SYMMETRIC_PAIRS` imports in classifier TTA code, accuracy drops after TTA implementation

### Pitfall 2: Forgetting Softmax Before Averaging
**What goes wrong:** Averaging logits instead of probabilities leads to incorrect ensemble predictions
**Why it happens:** Model forward() returns logits, easy to forget softmax conversion
**How to avoid:**
```python
# WRONG: averaging logits
logits1 = model(img)
logits2 = model(flip(img))
avg_logits = (logits1 + logits2) / 2
pred = avg_logits.argmax(dim=1)  # Incorrect!

# CORRECT: averaging probabilities
probs1 = torch.softmax(model(img), dim=1)
probs2 = torch.softmax(model(flip(img)), dim=1)
avg_probs = (probs1 + probs2) / 2
pred = avg_probs.argmax(dim=1)  # Correct
```
**Warning signs:** TTA hurts accuracy instead of helping, probabilities don't sum to 1.0

### Pitfall 3: McNemar's Test Misapplication
**What goes wrong:** Using paired t-test or simple accuracy comparison instead of McNemar's test
**Why it happens:** McNemar's test is less familiar than t-tests
**How to avoid:**
- McNemar's test is for PAIRED binary outcomes (same images, two different models/configurations)
- Requires 2x2 contingency table: [[both_correct, baseline_correct_tta_wrong], [baseline_wrong_tta_correct, both_wrong]]
- Use `mlxtend.evaluate.mcnemar(contingency_table, corrected=True)`
- Check if any cell < 25 → uses exact binomial test automatically
**Warning signs:** p-values don't match published medical imaging papers, test doesn't handle paired nature of data

### Pitfall 4: Config Override Precedence Bugs
**What goes wrong:** CLI flag doesn't override config value, or override logic is backward
**Why it happens:** Python's `None` vs `False` confusion, incorrect default handling
**How to avoid:**
```python
# WRONG: False CLI flag gets ignored
use_tta = tta or config['use_tta']  # --no-tta (False) falls back to config

# CORRECT: Explicit None check for overrides
use_tta = tta if tta is not None else config.get('use_tta', True)
```
**Warning signs:** `--no-tta` flag has no effect, tests passing with wrong TTA state

### Pitfall 5: Incomplete Output Preservation
**What goes wrong:** Saving only final ensemble prediction, losing per-model and intermediate predictions
**Why it happens:** Minimizing output file size, not anticipating debugging needs
**How to avoid:**
- Save original predictions, flipped predictions, TTA-averaged predictions per model
- Save ensemble final prediction
- Save case-level impact (helped/hurt/neutral)
- Structure as nested JSON for traceability
**Warning signs:** Can't debug why ensemble disagrees with individual models, can't analyze which images benefit from TTA

## Code Examples

Verified patterns from official sources:

### Horizontal Flip (PyTorch Official)
```python
# Source: https://docs.pytorch.org/docs/stable/generated/torch.flip.html
import torch

# Image tensor: (B, C, H, W) where W is width dimension (dim=3)
images = torch.randn(8, 3, 224, 224)
images_flipped = torch.flip(images, dims=[3])  # Flip along width (horizontal)

# Verify flip
assert images[0, :, :, 0].allclose(images_flipped[0, :, :, -1])  # Left ↔ Right
```

### Classification Report (scikit-learn 1.8.0)
```python
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
from sklearn.metrics import classification_report
import numpy as np

y_true = np.array([0, 1, 2, 0, 1, 2])  # Ground truth
y_pred = np.array([0, 2, 2, 0, 1, 1])  # Predictions
class_names = ['COVID', 'Normal', 'Viral_Pneumonia']

# Dictionary output for programmatic access
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0  # Handle undefined metrics (no samples for class)
)

# Access per-class metrics
covid_f1 = report['COVID']['f1-score']
overall_accuracy = report['accuracy']
macro_avg_f1 = report['macro avg']['f1-score']
```

### McNemar's Test (mlxtend)
```python
# Source: https://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/
from mlxtend.evaluate import mcnemar
import numpy as np

# Contingency table: [[both_correct, baseline_correct],
#                     [tta_correct, both_wrong]]
# Example: baseline correct on 10 images TTA missed, TTA correct on 15 baseline missed
contingency = np.array([[1850, 10], [15, 20]])

# McNemar's test with continuity correction
chi2, p_value = mcnemar(contingency, corrected=True)

if p_value < 0.05:
    print(f"TTA improvement is statistically significant (p={p_value:.4f})")
else:
    print(f"TTA improvement not significant (p={p_value:.4f})")
```

### Confusion Matrix (scikit-learn 1.8.0)
```python
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 2, 0, 1, 1]
class_names = ['COVID', 'Normal', 'Viral_Pneumonia']

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display with labels
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', values_format='d')
plt.title('Ensemble Predictions with TTA')
plt.savefig('outputs/confusion_matrix_tta.png')
```

### KL Divergence for Prediction Stability (PyTorch)
```python
# Source: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html
import torch
import torch.nn.functional as F

# Predictions from original and flipped images
probs_orig = torch.tensor([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])  # (B, num_classes)
probs_flip = torch.tensor([[0.65, 0.25, 0.1], [0.35, 0.45, 0.2]])

# KL divergence: KL(P_orig || P_flip)
# NOTE: kl_div expects log-probabilities as input for numerical stability
kl_div = F.kl_div(
    probs_flip.log(),  # Input must be log-probabilities
    probs_orig,         # Target is regular probabilities
    reduction='batchmean'  # Aligns with mathematical definition
)

# Interpret: high KL = unstable predictions under flip
print(f"Prediction stability (KL divergence): {kl_div.item():.4f}")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single augmentation (flip only) | Multi-augmentation TTA (rotation, scale, crop) | 2024-2025 | Complex TTA NOT recommended for medical imaging - breaks anatomical validity |
| Hard-coded TTA in model code | Library wrappers (ttach, tta.pytorch) | 2020-2021 | Libraries simplify but custom still valid for research control |
| Simple averaging | Generative model-based TTA (stable diffusion) | 2025-2026 | TTGA shows 0.1-2.3% improvements but computationally expensive, not production-ready |
| Equal weights | Learned aggregation weights | Ongoing | No clear consensus - simple average still standard for geometric transforms |

**Deprecated/outdated:**
- **ttach 0.0.3 last update:** Library appears unmaintained (last PyPI release 2020), but core functionality still works. Consider monitoring for alternatives if issues arise.
- **Rotation/scaling for chest X-rays:** Recent medical imaging research emphasizes conservative augmentations. Rotation breaks anatomical orientation (lungs don't rotate in real imaging).

## Open Questions

Things that couldn't be fully resolved:

1. **Class-Specific TTA Benefit**
   - What we know: Horizontal flip improves overall accuracy by 0.25-0.51% in recent chest X-ray studies
   - What's unclear: Whether COVID/Normal/Viral_Pneumonia classes benefit equally from TTA
   - Recommendation: Implement per-class delta tracking in Phase 3, analyze in thesis discussion

2. **Flip Stability as Quality Metric**
   - What we know: KL divergence can quantify prediction stability under flip (code example provided)
   - What's unclear: What threshold indicates "unstable" predictions? Is high KL divergence a failure mode indicator?
   - Recommendation: Compute KL divergence for all test images, analyze distribution, identify outliers for visual inspection

3. **Optimal Ensemble Size for TTA**
   - What we know: Project uses 5 CV fold models (from Phase 2), literature shows 3-7 is common
   - What's unclear: Does TTA benefit scale differently with ensemble size (e.g., 3 models + TTA vs 7 models no TTA)?
   - Recommendation: Out of scope for Phase 3. Document as future work if ensemble size changes.

4. **Statistical Significance Threshold**
   - What we know: McNemar's test provides p-value, standard threshold is p < 0.05
   - What's unclear: Medical imaging may require stricter thresholds (e.g., p < 0.01) for clinical claims
   - Recommendation: Report p-value, use p < 0.05 for "statistically significant", note in thesis that clinical deployment may require stricter validation

## Sources

### Primary (HIGH confidence)
- PyTorch Official Documentation: `torch.flip`, `torch.nn.functional.kl_div`, `torch.softmax` (https://docs.pytorch.org/docs/stable/)
- scikit-learn 1.8.0: `confusion_matrix`, `classification_report`, `precision_recall_fscore_support` (https://scikit-learn.org/stable/)
- mlxtend Documentation: `mcnemar` test implementation (https://rasbt.github.io/mlxtend/)
- Existing codebase: `src_v2/evaluation/metrics.py::predict_with_tta` (validated TTA implementation for landmarks)

### Secondary (MEDIUM confidence)
- [Test-Time Generative Augmentation for Medical Image Segmentation](https://arxiv.org/html/2406.17608v1) - 2025 research showing TTGA improvements (0.1-2.3% DSC gains)
- [Improving Medical Image Segmentation Using Test-Time Augmentation with MedSAM](https://www.mdpi.com/2227-7390/12/24/4003) - MedSAM TTA integration
- [Enhancing Multi-Label Chest X-Ray Classification](https://www.mdpi.com/2306-5354/12/6/593) - Horizontal flip TTA showing 0.25-0.51% AUC improvement
- [Weighted Average Ensemble Deep Learning for Brain Tumor Classification](https://www.mdpi.com/2075-4418/13/7/1320) - Weighted ensemble achieving 98% accuracy
- [ttach GitHub Repository](https://github.com/qubvel/ttach) - PyTorch TTA library (unmaintained but functional)

### Tertiary (LOW confidence)
- Web search results on TTA best practices (no single authoritative source, synthesized from multiple blog posts)
- StackOverflow discussions on KL divergence thresholds (no consensus found)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - PyTorch and scikit-learn are established, versions verified from official docs
- Architecture: HIGH - Dual-level TTA pattern validated in existing landmark code, directly adaptable
- Pitfalls: MEDIUM - Based on code review and common ML mistakes, not exhaustive testing
- McNemar's test: HIGH - mlxtend implementation verified, standard in medical imaging literature
- Class-specific TTA benefit: LOW - Requires empirical validation in Phase 3

**Research date:** 2026-01-27
**Valid until:** 60 days (stable domain - TTA methodology evolves slowly, core patterns unlikely to change before Phase 3 completion)
