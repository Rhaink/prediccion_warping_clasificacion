# Phase 3: TTA Integration - Implementation Context

## Phase Goal
Add test-time augmentation (horizontal flip) to the ensemble classifier to improve accuracy beyond the 98.10% baseline, using conservative augmentation appropriate for medical radiographs.

**Scope:** TTA capability within existing classifier pipeline. NOT redesigning classifier or adding new augmentation types.

---

## 1. Symmetry Correction Approach

### Decision: Minimal Symmetry Logic
Since class labels (COVID, Normal, Viral Pneumonia) are anatomically symmetric, **skip explicit symmetry correction** for classification TTA. Unlike landmark prediction where L/R swaps are critical, there's no left-lung vs right-lung class distinction.

**Implementation:**
- No SYMMETRIC_PAIRS import or mapping logic needed
- Horizontal flip is applied at image level only
- Predictions combine directly without label swapping

**Rationale:** Class labels don't encode anatomical sidedness. A flipped COVID X-ray is still COVID. This is fundamentally different from landmarks where L3↔L4 swaps are required.

**Validation:** Unit tests only
- Test that `flip_horizontal(image) → model → predictions` produces stable results
- Verify that `(pred_orig + pred_flip) / 2` doesn't degrade accuracy on known-symmetric cases
- No visual inspection needed (classes are not spatially mapped)

**Future extensibility:** If later phases add sided features (e.g., "left-lung opacity" class), revisit symmetry mapping.

---

## 2. TTA Configuration & Control

### Config Structure
**File:** `configs/ensemble_classifier.json`

**New parameter (top-level):**
```json
{
  "use_tta": true,
  "ensemble_config": "configs/ensemble_best.json",
  ...
}
```

**Default:** `use_tta: true` (TTA enabled by default)
- Phase 3 goal is TTA integration → should be standard behavior
- Users can disable for faster inference if needed

### CLI Override
**Command:** `python -m src_v2 evaluate-ensemble-classifier`

**New flags:**
- `--tta` / `--no-tta`: Override config's `use_tta` setting
- Config provides default, CLI provides runtime control

**Example:**
```bash
# Use config default (TTA enabled)
python -m src_v2 evaluate-ensemble-classifier --config configs/ensemble_classifier.json

# Explicitly disable TTA (for baseline comparison)
python -m src_v2 evaluate-ensemble-classifier --config configs/ensemble_classifier.json --no-tta

# Explicitly enable (redundant if config has use_tta: true)
python -m src_v2 evaluate-ensemble-classifier --config configs/ensemble_classifier.json --tta
```

**No additional parameters:** No `tta_aggregation`, `flip_threshold`, or augmentation params. Aggregation method is hardcoded (simple average). Phase scope is horizontal flip only.

---

## 3. Prediction Aggregation

### Aggregation Method: Simple Average
**Formula:** `final_pred = (pred_original + pred_flipped) / 2`

**Level:** Both model-level AND ensemble-level TTA
1. **Model-level TTA:** Each of the 5 CV fold models averages its own original+flip predictions
2. **Ensemble-level TTA:** The 5 model-level TTA predictions are then ensemble-averaged

**Pipeline:**
```
Image → [Model1, Model2, Model3, Model4, Model5]
       ↓
For each model:
  - Forward pass: original image → pred_orig
  - Forward pass: flipped image → pred_flip
  - TTA average: (pred_orig + pred_flip) / 2 → model_tta_pred

Ensemble average: mean([model1_tta, model2_tta, ..., model5_tta]) → final_pred
```

**Disagreement handling:** Proceed with average
- If `argmax(pred_orig) ≠ argmax(pred_flip)`, still compute mean
- Disagreement is informative (model uncertainty), not an error
- No warnings or flags (see case-level tracking in metrics section)

### Output Preservation
**Save intermediate predictions** for full traceability:

**Output JSON structure:**
```json
{
  "image_path": "COVID/image123.png",
  "ground_truth": "COVID",
  "predictions": {
    "model_1": {
      "original": [0.85, 0.10, 0.05],  // [COVID, Normal, Viral]
      "flipped": [0.82, 0.12, 0.06],
      "tta_averaged": [0.835, 0.11, 0.055]
    },
    "model_2": {...},
    ...
    "model_5": {...},
    "ensemble_final": [0.841, 0.108, 0.051]
  },
  "predicted_class": "COVID",
  "confidence": 0.841
}
```

**Why preserve intermediates:**
- Debug per-model behavior (e.g., Model 3 disagrees with others)
- Analyze flip stability (large `|pred_orig - pred_flip|` indicates uncertainty)
- Thesis documentation requires transparency in ensemble decisions

---

## 4. Performance & Metrics Tracking

### Metrics to Compute
**Per-class breakdown (COVID, Normal, Viral Pneumonia):**
- Accuracy
- Precision
- Recall
- F1-score

**Output format:**
```json
{
  "overall_accuracy": 0.9852,
  "per_class_metrics": {
    "COVID": {"precision": 0.99, "recall": 0.98, "f1": 0.985},
    "Normal": {"precision": 0.97, "recall": 0.99, "f1": 0.980},
    "Viral Pneumonia": {"precision": 0.99, "recall": 0.98, "f1": 0.985}
  },
  "confusion_matrix": [[...], [...], [...]]
}
```

### TTA Improvement Comparison
**Include delta metrics** showing TTA improvement over baseline:

**Output structure:**
```json
{
  "baseline_no_tta": {
    "overall_accuracy": 0.9810,
    "per_class_metrics": {...}
  },
  "with_tta": {
    "overall_accuracy": 0.9852,
    "per_class_metrics": {...}
  },
  "improvement": {
    "accuracy_delta": +0.0042,
    "per_class_f1_delta": {
      "COVID": +0.003,
      "Normal": +0.005,
      "Viral Pneumonia": +0.004
    }
  }
}
```

**How to compute:**
- Run evaluation twice internally: once with TTA, once without
- Both runs use same ensemble, same test set
- Report both results + delta in single output JSON

**Benefit:** Clear thesis reporting without manual runs. Shows exactly where TTA helps.

### Timing Breakdown
**Detailed performance metrics:**

**Output:**
```json
{
  "timing": {
    "total_images": 1895,
    "without_tta": {
      "total_time_sec": 23.4,
      "time_per_image_ms": 12.3
    },
    "with_tta": {
      "total_time_sec": 45.7,
      "time_per_image_ms": 24.1
    },
    "overhead": {
      "absolute_sec": +22.3,
      "relative_factor": 1.95
    }
  }
}
```

**What to measure:**
- Total evaluation time (wall clock)
- Per-image average (total_time / num_images)
- TTA overhead (both absolute and relative)

**Purpose:** Deployment planning. Shows inference cost of TTA for production use.

### Case-Level Analysis
**Save per-image TTA impact:**

**Output structure:**
```json
{
  "case_level_analysis": [
    {
      "image_path": "COVID/img001.png",
      "ground_truth": "COVID",
      "baseline_prediction": "Viral Pneumonia",  // Error
      "tta_prediction": "COVID",                  // Correct
      "tta_impact": "helped"
    },
    {
      "image_path": "Normal/img123.png",
      "ground_truth": "Normal",
      "baseline_prediction": "Normal",
      "tta_prediction": "Normal",
      "tta_impact": "neutral"
    },
    {
      "image_path": "Viral/img456.png",
      "ground_truth": "Viral Pneumonia",
      "baseline_prediction": "Viral Pneumonia",  // Correct
      "tta_prediction": "COVID",                  // Error
      "tta_impact": "hurt"
    }
  ],
  "summary": {
    "tta_helped": 12,
    "tta_hurt": 3,
    "tta_neutral": 1880
  }
}
```

**Impact categories:**
- **Helped:** Baseline wrong, TTA correct
- **Hurt:** Baseline correct, TTA wrong
- **Neutral:** Both correct OR both wrong (same class)

**Purpose:**
- Understand TTA failure modes (when does it hurt?)
- Identify images for visual inspection (high disagreement cases)
- Thesis discussion: "TTA improved 12 misclassifications but introduced 3 new errors"

---

## Implementation Checklist

### Code Changes
- [ ] Add `use_tta: true` to `configs/ensemble_classifier.json`
- [ ] Implement horizontal flip transform in `src_v2/data/transforms.py`
- [ ] Add TTA logic to ensemble evaluation (dual forward passes per model)
- [ ] Implement simple averaging: `(pred_orig + pred_flip) / 2`
- [ ] Add CLI flags `--tta` / `--no-tta` to override config
- [ ] Extend output JSON with intermediate predictions structure
- [ ] Add baseline comparison (run evaluation twice: with/without TTA)
- [ ] Implement timing measurement (before/after TTA)
- [ ] Implement case-level impact tracking (helped/hurt/neutral)

### Testing
- [ ] Unit test: flip + predict returns stable probabilities
- [ ] Unit test: averaging logic (pred_orig + pred_flip) / 2
- [ ] Integration test: full pipeline with TTA enabled
- [ ] Validate output JSON structure matches specification

### Documentation
- [ ] Update `GROUND_TRUTH.json` with TTA results after validation
- [ ] Update `REPRO_FULL_PIPELINE.md` with TTA usage
- [ ] Add TTA section to thesis methods chapter

---

## Constraints & Boundaries

**In Scope:**
- Horizontal flip TTA only
- Simple average aggregation (hardcoded)
- Both model-level and ensemble-level TTA
- Detailed metrics and case-level tracking

**Out of Scope (Future Phases):**
- Other augmentations (rotation, scaling, brightness)
- Learned aggregation weights
- Selective TTA (applying only to uncertain cases)
- TTA for landmark prediction (already implemented in Phase 2)

**Phase Success Criteria:**
- Ensemble accuracy > 98.10% baseline (with TTA)
- Inference time overhead measured and documented
- Per-class metrics show TTA benefit for COVID/Normal/Viral
- Output JSON includes full traceability (intermediate predictions + case-level impact)

---

## Open Questions for Research Phase

1. **Statistical significance:** How to test if TTA improvement is statistically significant (McNemar's test)?
2. **Flip stability metric:** Should we define a metric for prediction stability under flip (e.g., KL divergence between pred_orig and pred_flip)?
3. **Class-specific TTA:** Do COVID/Normal/Viral benefit equally from TTA, or should we analyze per-class improvement?

**Researcher:** Investigate TTA best practices for medical imaging (papers, benchmarks). Check if simple average is standard or if weighted methods are preferred.

**Planner:** Design tasks to implement both-level TTA (model + ensemble) with full output preservation.
