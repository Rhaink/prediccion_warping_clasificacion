# Phase 5: Final Test Evaluation - Research

**Researched:** 2026-02-16
**Domain:** Final thesis evaluation protocol with deterministic reproducibility, duplicate handling, and methodology documentation
**Confidence:** HIGH

## Summary

Phase 5 performs final validation of the ensemble+TTA configuration on the complete test set (1,895 images) with rigorous checks to prove methodological integrity for thesis defense. The core task is executing deterministic evaluation twice to prove reproducibility, handling the 1 known test duplicate transparently, and generating a comprehensive methodology summary in Spanish documenting the full pipeline.

The project has established patterns for all necessary components: deterministic evaluation via PyTorch inference mode with fixed seeds, MD5-based duplicate detection (Phase 1 audit), JSON-based metrics persistence matching GROUND_TRUTH.json structure, and LaTeX/Markdown generation for thesis integration (Phase 4). The final evaluation extends these patterns with reproducibility verification (double-run assertion) and dual-dataset evaluation (original vs cleaned).

**Primary recommendation:** Create standalone evaluation script that runs ensemble+TTA evaluation twice consecutively, asserts bit-identical outputs (JSON string comparison or metrics hash), handles duplicates via on-the-fly filtering during data loading, updates GROUND_TRUTH.json with new `final_evaluation` section, and generates Spanish methodology summary as `docs/METODOLOGIA_COMPLETA.md` for thesis appendix inclusion.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Results package:**
- Claude decides JSON structure (extend existing vs separate file) based on existing patterns
- Claude decides whether to reference Phase 4 figures or regenerate based on validity
- GROUND_TRUTH.json must be updated with a `final_evaluation` section as the canonical source of truth
- Generate a standalone Markdown methodology summary covering the full pipeline (data prep, landmarks, warping, ensemble, TTA, final evaluation) in Spanish

**Validation checks:**
- Re-verify test set isolation programmatically as part of the final evaluation script (file hashes, path comparisons)
- If accuracy falls outside expected range (+0.5 to +1.0pp over 97.68%): warn but continue — the actual number is what it is
- Claude decides strictness level for class count verification (COVID=452, Normal=1274, Viral_Pneumonia=169)
- Run evaluation twice and assert identical outputs to prove deterministic reproducibility

**Reporting format:**
- All human-readable output in Spanish (labels, headers, narrative text)
- Claude decides console output verbosity based on existing CLI patterns
- Methodology summary as standalone Markdown (.md), full pipeline scope, in Spanish

**Duplicate handling:**
- Run evaluation on BOTH original (1,895) and cleaned (duplicates removed) test sets
- Claude decides whether to remove or replace the 1 test duplicate (methodologically sound approach)
- Report both results equally in the thesis with a note about the known duplicates
- Claude decides implementation: on-the-fly cleanup vs separate pre-step

### Claude's Discretion

- JSON output file structure and naming
- Whether to regenerate Phase 4 figures or reference existing
- Class count verification strictness (assert vs warn)
- Console output verbosity level
- Duplicate removal implementation approach (on-the-fly vs pre-step)
- Test duplicate handling (remove vs replace)

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

## Standard Stack

### Core Libraries

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.4.1+rocm6.0 | Model inference with deterministic evaluation | Project dependency, proven evaluation pipeline |
| NumPy | 1.24+ | Metrics computation, array comparison | Universal, required for bit-identical comparison |
| json | stdlib | Metrics persistence, GROUND_TRUTH updates | Standard serialization, existing pattern |
| pathlib | stdlib | Path handling, file operations | Python 3 standard, consistent with codebase |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| hashlib | stdlib | MD5 file hashing for duplicate detection | Duplicate verification (Phase 1 pattern) |
| scikit-learn | 1.7.2 | Classification metrics (accuracy, F1, confusion matrix) | Existing evaluation pattern |
| pandas | 2.0+ | CSV loading for duplicate filtering | Data manipulation, existing pattern |

**Installation:**
```bash
# All libraries already installed in project environment
# No additional dependencies required
```

## Architecture Patterns

### Recommended Project Structure

Based on existing codebase patterns:

```
scripts/
├── evaluate_final_ensemble_tta.py          # Main evaluation script
└── generate_methodology_summary.py         # Methodology doc generator

outputs/classifier_cv/
├── final_evaluation_original.json          # Results on 1,895 images
├── final_evaluation_cleaned.json           # Results on 1,894 images (duplicate removed)
└── final_evaluation_reproducibility.json   # Comparison hash proving identity

docs/
└── METODOLOGIA_COMPLETA.md                 # Spanish methodology summary

GROUND_TRUTH.json                           # Updated with final_evaluation section
```

### Pattern 1: Deterministic Evaluation with Reproducibility Check

**What:** Run evaluation twice, assert outputs are bit-identical to prove determinism
**When to use:** Final validation when methodological rigor is critical for thesis defense
**Example:**

```python
# Source: Existing evaluation pattern + Phase 1 verification approach
import json
import hashlib
import torch

def compute_metrics_hash(metrics: dict) -> str:
    """Compute deterministic hash of metrics dict."""
    # Sort keys for deterministic serialization
    metrics_str = json.dumps(metrics, sort_keys=True)
    return hashlib.sha256(metrics_str.encode()).hexdigest()

def evaluate_with_reproducibility_check(model, dataloader, device, run_label):
    """Run evaluation twice and verify identity."""
    print(f"\n--- {run_label} ---")

    # First run
    print("Ejecución 1...")
    metrics_1 = run_evaluation(model, dataloader, device)
    hash_1 = compute_metrics_hash(metrics_1)

    # Second run
    print("Ejecución 2...")
    metrics_2 = run_evaluation(model, dataloader, device)
    hash_2 = compute_metrics_hash(metrics_2)

    # Verify identity
    if hash_1 == hash_2:
        print(f"✓ Reproducibilidad verificada (hash: {hash_1[:16]}...)")
        return metrics_1, True
    else:
        print(f"✗ ADVERTENCIA: Resultados no idénticos")
        print(f"  Hash 1: {hash_1}")
        print(f"  Hash 2: {hash_2}")
        return metrics_1, False

@torch.no_grad()
def run_evaluation(model, dataloader, device):
    """Single evaluation run with deterministic inference."""
    model.eval()
    # ... standard evaluation loop ...
    return metrics
```

### Pattern 2: On-the-Fly Duplicate Filtering

**What:** Filter duplicates during data loading without modifying original dataset
**When to use:** Transparent duplicate handling preserving original data
**Example:**

```python
# Source: Phase 1 audit duplicate detection + project data loading patterns
import hashlib
from pathlib import Path

def load_test_set_with_duplicate_filtering(data_dir: Path, filter_duplicates: bool = False):
    """
    Load test set with optional duplicate filtering.

    Args:
        data_dir: Path to warped dataset (e.g., outputs/warped_lung_best/session_warping)
        filter_duplicates: If True, exclude known duplicate

    Returns:
        Dataset with 1,895 samples (original) or 1,894 (cleaned)
    """
    # Known duplicate from Phase 1 audit
    KNOWN_DUPLICATE = "test/Normal/Normal-817_warped.png"

    # Load dataset
    dataset = ImageFolder(data_dir / "test")

    if not filter_duplicates:
        return dataset

    # Filter duplicate
    filtered_samples = [
        sample for sample in dataset.samples
        if KNOWN_DUPLICATE not in str(sample[0])
    ]

    dataset.samples = filtered_samples
    dataset.targets = [s[1] for s in filtered_samples]

    print(f"Duplicados filtrados: {len(dataset.samples)} muestras restantes (1 removida)")
    return dataset
```

### Pattern 3: GROUND_TRUTH.json Update

**What:** Extend GROUND_TRUTH.json with final_evaluation section following existing structure
**When to use:** Persist validated final metrics as canonical source of truth
**Example:**

```python
# Source: Existing GROUND_TRUTH.json structure
def update_ground_truth_with_final_evaluation(metrics_original, metrics_cleaned):
    """Add final_evaluation section to GROUND_TRUTH.json."""

    ground_truth_path = Path("GROUND_TRUTH.json")

    with open(ground_truth_path) as f:
        gt = json.load(f)

    # Add final_evaluation section
    gt["classification"]["final_evaluation"] = {
        "description": "Final test set evaluation with ensemble+TTA (Phase 5)",
        "timestamp": datetime.now().astimezone().isoformat(),
        "validated": "2026-02-16",
        "original_test_set": {
            "test_set_size": 1895,
            "duplicates_present": 1,
            "accuracy": metrics_original["accuracy"],
            "f1_macro": metrics_original["f1_macro"],
            "f1_weighted": metrics_original["f1_weighted"],
            "confusion_matrix": metrics_original["confusion_matrix"],
            "per_class": metrics_original["per_class"]
        },
        "cleaned_test_set": {
            "test_set_size": 1894,
            "duplicates_removed": 1,
            "accuracy": metrics_cleaned["accuracy"],
            "f1_macro": metrics_cleaned["f1_macro"],
            "f1_weighted": metrics_cleaned["f1_weighted"],
            "confusion_matrix": metrics_cleaned["confusion_matrix"],
            "per_class": metrics_cleaned["per_class"]
        },
        "reproducibility_verified": True,
        "baseline_improvement_pp": (metrics_original["accuracy"] - 0.9768) * 100,
        "expected_range_pp": [0.5, 1.0],
        "note": "Both results reported equally; duplicate noted for transparency"
    }

    # Update metadata
    gt["_metadata"]["last_updated"] = datetime.now().date().isoformat()
    gt["_metadata"]["version"] = "2.2.0"  # Increment version

    # Write back
    with open(ground_truth_path, "w") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)
```

### Pattern 4: Test Set Isolation Verification

**What:** Programmatic check that test images were never seen during training
**When to use:** Final validation proving methodological integrity
**Example:**

```python
# Source: Phase 1 audit methodology
def verify_test_set_isolation(cv_dir: Path, test_data_dir: Path):
    """
    Verify test set was isolated from training.

    Checks:
    1. Training history files contain no test metrics
    2. Test evaluation timestamp > model save timestamp
    3. No test image paths in training metadata
    """
    print("\n=== VERIFICACIÓN DE AISLAMIENTO DEL CONJUNTO DE PRUEBA ===")

    checks_passed = 0
    checks_total = 0

    # Check 1: No test metrics in training history
    for fold in range(1, 6):
        history_path = cv_dir / f"fold_{fold:02d}" / "training_history.json"
        with open(history_path) as f:
            history = json.load(f)

        # Verify only train/val metrics present
        metrics = history["history"][0].keys()
        has_test_metrics = any("test" in m for m in metrics)

        checks_total += 1
        if not has_test_metrics:
            checks_passed += 1
        else:
            print(f"  ✗ Fold {fold}: Métricas de prueba encontradas en historial")

    # Check 2: Temporal separation
    # ... (similar to Phase 1 audit) ...

    print(f"\nVerificación: {checks_passed}/{checks_total} controles aprobados")
    return checks_passed == checks_total
```

### Anti-Patterns to Avoid

- **Modifying original test set files:** Never delete or overwrite test images; use on-the-fly filtering to preserve original data
- **Hardcoded metrics:** Always load from existing JSON files (ensemble_test_results_tta.json, GROUND_TRUTH.json) rather than hardcoding baseline values
- **Non-deterministic evaluation:** Avoid any randomness in evaluation (dropout disabled via model.eval(), no data shuffling, fixed PyTorch seed)
- **Silent duplicate handling:** Always report both original and cleaned results explicitly; transparency is critical for thesis defense

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Classification metrics | Custom accuracy/F1 calculation | scikit-learn classification_report | Handles edge cases (zero division), standard format, widely validated |
| File hashing | Custom checksum function | hashlib.md5() or hashlib.sha256() | Cryptographically sound, optimized C implementation, stdlib |
| JSON comparison | Manual dict comparison | json.dumps(sort_keys=True) + hash | Deterministic serialization, handles nested structures |
| Model checkpointing | Custom save/load | torch.save/torch.load | Handles CUDA/CPU mapping, versioning, metadata preservation |

**Key insight:** The evaluation pipeline is well-established in PyTorch. Custom implementations introduce bugs (e.g., forgetting to call model.eval(), incorrect metric averaging). Use proven libraries and existing codebase patterns.

## Common Pitfalls

### Pitfall 1: Non-Deterministic Evaluation Due to Dropout/BatchNorm

**What goes wrong:** Even with torch.manual_seed(), results vary between runs if model.eval() is not called
**Why it happens:** Dropout and BatchNorm behave differently in train vs eval mode; forgetting model.eval() uses training behavior (random dropout)
**How to avoid:** Always call model.eval() before inference AND wrap with @torch.no_grad() decorator
**Warning signs:** Accuracy varies by ~0.1-0.5% between identical runs, predictions differ for same input

```python
# BAD: Missing model.eval()
@torch.no_grad()
def evaluate(model, dataloader):
    # model.eval() MISSING!
    for images, labels in dataloader:
        outputs = model(images)  # Uses training-mode dropout

# GOOD: Explicit eval mode
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()  # Critical: disables dropout/BatchNorm randomness
    for images, labels in dataloader:
        outputs = model(images)
```

### Pitfall 2: JSON Serialization Issues with NumPy/PyTorch Types

**What goes wrong:** json.dump() fails with TypeError when metrics contain numpy.int64 or torch.Tensor
**Why it happens:** JSON standard only supports native Python types (int, float, str, list, dict)
**How to avoid:** Convert all metrics to native Python types before serialization using .item() for scalars, .tolist() for arrays
**Warning signs:** TypeError: Object of type int64 is not JSON serializable

```python
# BAD: Direct serialization of NumPy types
confusion_matrix = np.array([[441, 10, 1], ...])
metrics = {"confusion_matrix": confusion_matrix}  # Will fail
json.dump(metrics, f)  # TypeError!

# GOOD: Convert to native Python types
metrics = {
    "confusion_matrix": confusion_matrix.tolist(),  # List of lists
    "accuracy": float(accuracy),  # Python float
    "sample_count": int(sample_count)  # Python int
}
json.dump(metrics, f)  # Works
```

### Pitfall 3: Incomplete Test Set Isolation Verification

**What goes wrong:** Claiming test set isolation based only on timestamps, missing actual code inspection
**Why it happens:** Timestamps prove temporal separation but not logical isolation (code could have loaded test data during training)
**How to avoid:** Multi-pronged verification: (1) training history logs, (2) timestamps, (3) config files, (4) git history, (5) programmatic path checks
**Warning signs:** Reviewer asks "How do you KNOW test data wasn't accessed?" and only answer is "We checked timestamps"

```python
# INCOMPLETE: Only timestamp check
def verify_isolation():
    model_time = get_file_mtime("model.pt")
    test_time = get_file_mtime("test_results.json")
    return test_time > model_time  # Not enough!

# COMPLETE: Multi-pronged verification
def verify_isolation_comprehensive():
    checks = {
        "no_test_metrics_in_history": verify_no_test_metrics(),
        "temporal_separation": verify_timestamps(),
        "no_test_in_configs": verify_configs(),
        "no_test_in_git_history": verify_git_history()
    }
    return all(checks.values()), checks
```

### Pitfall 4: Ambiguous Duplicate Handling in Reporting

**What goes wrong:** Reporting single accuracy value without clarifying if duplicates were included/excluded, leading to thesis defense questions
**Why it happens:** Treating duplicate removal as internal cleanup step rather than methodological decision requiring transparency
**How to avoid:** Report BOTH results explicitly (original and cleaned) with clear labeling, include duplicate count in metadata
**Warning signs:** Thesis reviewer asks "Did you include or exclude duplicates?" and answer is uncertain

```python
# BAD: Ambiguous reporting
final_accuracy = 0.9826  # Which dataset? Original or cleaned?

# GOOD: Explicit dual reporting
results = {
    "original_test_set": {
        "samples": 1895,
        "duplicates": 1,
        "accuracy": 0.9826,
        "note": "Includes known duplicate for transparency"
    },
    "cleaned_test_set": {
        "samples": 1894,
        "duplicates_removed": 1,
        "accuracy": 0.9825,  # May differ slightly
        "note": "Duplicate removed for methodological purity"
    },
    "recommendation": "Report both; note duplicate in thesis methodology"
}
```

## Code Examples

Verified patterns from existing codebase:

### Loading Ensemble Models for Evaluation

```python
# Source: Phase 2 ensemble implementation + existing CLI evaluation pattern
from pathlib import Path
import torch
from src_v2.models.classifier import ImageClassifier

def load_ensemble_models(cv_dir: Path, device: torch.device):
    """
    Load all 5 cross-validation models for ensemble evaluation.

    Args:
        cv_dir: Path to outputs/classifier_cv/
        device: torch device (cuda/cpu)

    Returns:
        List of (model, validation_f1) tuples
    """
    models = []

    for fold in range(1, 6):
        checkpoint_path = cv_dir / f"fold_{fold:02d}" / "best_classifier.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint missing: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create model
        model = ImageClassifier(
            backbone=checkpoint.get("backbone", "resnet18"),
            num_classes=len(checkpoint["class_names"])
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()  # Critical: disable dropout/BatchNorm randomness

        # Get validation F1 for weighting
        val_f1 = checkpoint.get("best_val_f1", 1.0)

        models.append((model, val_f1))
        print(f"Fold {fold}: Val F1-macro = {val_f1:.4f}")

    return models
```

### Soft Voting with TTA

```python
# Source: Phase 3 TTA implementation + Phase 2 ensemble voting
import torch
import torch.nn.functional as F

@torch.no_grad()
def ensemble_predict_with_tta(models, images, device):
    """
    Ensemble prediction with test-time augmentation.

    Args:
        models: List of (model, weight) tuples
        images: Batch of images (B, C, H, W)
        device: torch device

    Returns:
        Combined probabilities (B, num_classes)
    """
    all_probs = []
    weights = []

    for model, val_f1 in models:
        # Original prediction
        logits_orig = model(images)
        probs_orig = F.softmax(logits_orig, dim=1)

        # Flipped prediction (TTA)
        images_flipped = torch.flip(images, dims=[3])  # Horizontal flip
        logits_flip = model(images_flipped)
        probs_flip = F.softmax(logits_flip, dim=1)

        # Average TTA
        probs_avg = (probs_orig + probs_flip) / 2

        all_probs.append(probs_avg)
        weights.append(val_f1)

    # Stack and weight
    probs_stacked = torch.stack(all_probs, dim=0)  # (5, B, 3)
    weights_tensor = torch.tensor(weights, device=device)
    weights_norm = weights_tensor / weights_tensor.sum()

    # Weighted average
    weights_expanded = weights_norm.view(-1, 1, 1)  # (5, 1, 1)
    ensemble_probs = (probs_stacked * weights_expanded).sum(dim=0)  # (B, 3)

    return ensemble_probs
```

### Computing Classification Metrics

```python
# Source: Existing evaluate-classifier CLI command
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def compute_classification_metrics(all_preds, all_labels, class_names):
    """
    Compute comprehensive classification metrics.

    Args:
        all_preds: List or array of predicted class indices
        all_labels: List or array of true class indices
        class_names: List of class names (e.g., ["COVID", "Normal", "Viral_Pneumonia"])

    Returns:
        Dict with accuracy, F1 scores, confusion matrix, per-class metrics
    """
    # Convert to numpy if needed
    preds = np.array(all_preds)
    labels = np.array(all_labels)

    # Overall metrics
    accuracy = (preds == labels).mean()

    # Per-class metrics
    report = classification_report(
        labels, preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(labels, preds)

    return {
        "accuracy": float(accuracy),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "f1_weighted": float(report["weighted avg"]["f1-score"]),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1-score": float(report[name]["f1-score"]),
                "support": int(report[name]["support"])
            }
            for name in class_names
        }
    }
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single model evaluation | Ensemble + TTA evaluation | Phase 2-3 (2026-01) | +0.58pp accuracy improvement over baseline |
| Manual reproducibility claims | Programmatic double-run verification | Phase 5 (2026-02) | Proves determinism objectively, thesis-defensible |
| Ignore dataset duplicates | Transparent dual reporting (original + cleaned) | Phase 1-5 (2026-01-02) | Methodological integrity, reviewer confidence |
| Informal methodology notes | Standalone Spanish methodology doc | Phase 5 (2026-02) | Thesis appendix-ready, complete pipeline coverage |

**Deprecated/outdated:**
- Manual accuracy verification: Superseded by automated reproducibility checks (double-run + hash comparison)
- Single-dataset reporting: Now requires dual reporting (original + cleaned) for transparency

## Open Questions

1. **Class Count Verification Strictness**
   - What we know: Expected counts are COVID=452, Normal=1274, Viral_Pneumonia=169 (from Phase 1 audit)
   - What's unclear: Should mismatch trigger hard assertion (exit) or warning (continue)?
   - Recommendation: Use hard assertion (raises error if mismatch) for final evaluation — any count deviation indicates data corruption or loading error requiring investigation

2. **Phase 4 Figure Reuse vs Regeneration**
   - What we know: Phase 4 generated confusion matrices for baseline vs ensemble+TTA comparison
   - What's unclear: Are Phase 4 figures based on original (1,895) or would need regeneration for cleaned (1,894)?
   - Recommendation: Reference existing Phase 4 figures IF they used original test set; regeneration not needed unless cleaned-only reporting is required (unlikely given dual reporting approach)

3. **Expected Range Handling**
   - What we know: Expected improvement is +0.5 to +1.0pp over 97.68% baseline
   - What's unclear: What happens if actual improvement is +0.4pp or +1.2pp?
   - Recommendation: Warn (not error) if outside range, log expectation vs actual, continue with evaluation — the actual number is what matters, range is guidance not hard constraint

## Sources

### Primary (HIGH confidence)

- **GROUND_TRUTH.json** - Baseline metrics (97.68% accuracy, validation F1 weights), expected ranges, existing structure
- **outputs/classifier_cv/ensemble_test_results_tta.json** - Phase 3 TTA results (98.26% accuracy, case-level analysis)
- **src_v2/cli.py** - Deterministic seed setting pattern (lines 360-366), evaluation command patterns
- **src_v2/evaluation/metrics.py** - Existing evaluation functions (evaluate_model, compute_pixel_error)
- **scripts/generate_confusion_matrices_comparison.py** - Phase 4 data loading pattern, matplotlib configuration
- **scripts/verify_dataset_splits.py** - MD5 hashing for duplicate detection (lines 25-31)

### Secondary (MEDIUM confidence)

- **.planning/phases/01-pre-implementation-audit/AUDIT_REPORT.md** - Duplicate detection findings (1 test, 8 val duplicates), verification methodology
- **configs/ensemble_classifier.json** - Expected sample counts, class names, baseline accuracy reference
- **.planning/phases/04-analysis-visualization/04-01-SUMMARY.md** - LaTeX generation patterns, Spanish labeling conventions
- **.planning/phases/02-ensemble-core/02-RESEARCH.md** - Weighted soft voting pattern, ensemble architecture

### Tertiary (LOW confidence)

- None - all findings verified with primary sources (code, configs, existing results)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already in use, no new dependencies
- Architecture patterns: HIGH - Extends proven Phase 1-4 patterns (evaluation, duplicate detection, JSON persistence, Spanish docs)
- Pitfalls: HIGH - Based on actual bugs encountered in Phases 1-4 (numpy JSON serialization, model.eval() missing)
- Reproducibility protocol: MEDIUM-HIGH - Double-run + hash comparison pattern validated in research literature but not yet implemented in this codebase

**Research date:** 2026-02-16
**Valid until:** 30 days (stable evaluation methodology, unlikely to change)
