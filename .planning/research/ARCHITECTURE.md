# Architecture Integration: Data-Centric Improvements

**Researched:** 2026-02-16
**Current:** v1.0 pipeline, 98.26% ensemble accuracy
**Goal:** Integrate data-centric improvements without architectural refactoring

## Integration Points

### 1. Data Cleaning — Before Warping
- **Landmark quality filtering** (NEW: `src_v2/data/quality_checks.py`)
  - Outlier landmarks (>3σ from canonical), degenerate triangles, low fill rate
  - Modify `generate_dataset()` to filter before warping
- **Near-duplicate detection** (add to quality_checks.py)
  - Perceptual hash-based, run before splits

### 2. Advanced Augmentation — During Training
- **MixUp/CutMix** (NEW: `src_v2/data/batch_augmentations.py`)
  - Batch-level in training loop (NOT per-sample transforms)
  - Modify `classifier_trainer.py` to support mixed loss
- Config-driven: `augmentation.advanced.mixup.enabled`, etc.

### 3. Label Noise Detection — After CV Training
- **Cleanlab integration** (NEW: `src_v2/data/label_cleaning.py`)
  - Modify training to save `validation_predictions.npz` per fold
  - 5-fold CV provides natural out-of-sample predictions
  - CLI: `python -m src_v2 detect-label-noise --cv-dir outputs/classifier_cv`

### 4. Error Forensics — After Evaluation
- **Misclassification analysis** (NEW: `src_v2/evaluation/error_forensics.py`)
  - Extract misclassified samples + confidence + margin
  - Categorize: high-confidence errors → label noise; low-margin → augmentation needed
  - Generate visualization grids

### 5. Preprocessing Optimization
- CLAHE tuning (currently tile=4, clip=2.0)
- Normalization strategy comparison

## Pipeline Order (11 Steps)

```
1. Raw dataset → duplicate detection
2. Landmark prediction (cached NPZ)
3. Landmark quality check (NEW)
4. Warping → warped images
5. Dataset splits (stratified)
6. Training + augmentation (basic + MixUp/CutMix NEW)
7. 5-fold CV → save validation predictions (NEW)
8. Label noise detection (NEW)
9. Ensemble evaluation + TTA
10. Error forensics (NEW)
11. Iterative refinement (feedback loop)
```

## Modified vs New Components

### Modified
| File | Modification |
|------|-------------|
| `src_v2/cli.py::generate_dataset()` | Quality check before warping |
| `scripts/train_classifier_cv.py` | Save validation predictions per fold |
| `scripts/evaluate_final_ensemble_tta.py` | Call error forensics |
| `configs/*.json` | Add feature flags |
| `src_v2/training/classifier_trainer.py` | Batch augmentation support |

### New
| File | Purpose |
|------|---------|
| `src_v2/data/quality_checks.py` | Landmark quality, duplicates |
| `src_v2/data/batch_augmentations.py` | MixUp, CutMix |
| `src_v2/data/label_cleaning.py` | Cleanlab integration |
| `src_v2/evaluation/error_forensics.py` | Error analysis pipeline |

## Key ADRs
1. **Clean before warp** — avoid wasting computation on invalid samples
2. **Label noise after CV** — 5-fold naturally provides out-of-sample predictions
3. **Batch augmentations in loop** — require batch-level ops + mixed loss
4. **Config-driven flags** — reproducibility, consistency, no CLI flag proliferation

## Build Order
| Phase | Duration | Focus |
|-------|----------|-------|
| 1 | 2-3 days | Error forensics + data quality audit |
| 2 | 3-4 days | Data cleaning pipeline |
| 3 | 3-4 days | Advanced augmentation + focal loss |
| 4 | 4-5 days | Label noise detection |
| 5 | 3-4 days | Re-training + evaluation |
| 6 | 2-3 days | Documentation + validation |
