# COVID-19 Detection Ensemble Enhancement

## What This Is

A thesis project enhancement that implements an ensemble classifier with Test-Time Augmentation (TTA) to improve COVID-19 detection accuracy from chest X-rays. Building on an existing geometric normalization pipeline (landmark detection + warping), this adds ensemble evaluation of 5 cross-validation models to achieve ~98.2-98.7% test accuracy, up from the current 97.68% individual model average.

## Core Value

Maximize test set accuracy using existing cross-validation models while preserving methodological integrity (no test set contamination). The ensemble must demonstrate reproducible improvement with rigorous validation for thesis reporting.

## Requirements

### Validated

<!-- Existing system capabilities (already implemented and working) -->

- ✓ Landmark detection ensemble (3.61 px error on 15 lung contour landmarks) — existing
- ✓ Piecewise affine warping for geometric normalization via GPA — existing
- ✓ 5 ResNet-18 classifiers trained with k-fold cross-validation (k=5) — existing
- ✓ Individual model evaluation on fixed test set (97.68% ± 0.16% accuracy) — existing
- ✓ Complete data pipeline (landmarks → warping → classification) — existing
- ✓ JSON configuration system for reproducibility — existing
- ✓ Cached landmark predictions in NPZ format — existing
- ✓ Visualization scripts for matrices and figures — existing

### Active

<!-- Current scope: ensemble implementation for thesis -->

- [ ] Ensemble evaluation script with soft voting (average probabilities from 5 CV models)
- [ ] Conservative Test-Time Augmentation (horizontal flip + safe radiograph augmentations)
- [ ] Evaluation on complete test set (1,895 images: COVID-19=452, Normal=1,274, Viral_Pneumonia=169)
- [ ] Detailed improvement analysis (where ensemble corrects errors, why it works, which cases benefit)
- [ ] Confusion matrices following existing visual style (locate style from .tex references)
- [ ] Comprehensive results JSON with metrics, per-class breakdown, comparison to individual models
- [ ] Validation checks (no data leakage, expected gain +0.5-1.0 points, 1895 samples total)

### Out of Scope

- Training new models — use existing 5 CV models only
- Hard voting ensemble — soft voting superior, stick to probability averaging
- Aggressive/destructive augmentations — preserve medical semantic integrity
- Threshold optimization using test set — methodological violation
- Automatic LaTeX chapter updates — user updates manually after validation
- Ablation studies (ensemble-only, TTA-only) — focus on best combined result
- Additional experimental architectures — thesis deadline constraints

## Context

**Thesis Stage:** Writing phase with recent correction to methodology (switched from reporting validation accuracy to test accuracy). Need improved test accuracy for final results chapter.

**Existing Results:**
- 5 models evaluated individually on test set: 97.52% - 97.94% accuracy
- Best individual model: Fold 5 (97.94%)
- Average: 97.68% ± 0.16%
- F1-Macro: 96.47% ± 0.27%

**Expected Improvement:**
- Ensemble (soft voting): +0.3 to +0.8 points
- TTA (conservative): +0.2 to +0.5 points
- Combined target: 98.2% - 98.7% accuracy

**Visualization Requirements:**
- Must match existing figure style in Chapter 5 (locate scripts via .tex file references)
- Primary: Confusion matrices
- Secondary: Metric comparisons (if needed for analysis)

**Augmentation Strategy:**
- Horizontal flip (safe for symmetric lung anatomy)
- Small rotations (±5°) if validated safe
- Avoid: Vertical flip, aggressive crops, extreme color jitter
- Rationale: Preserve radiological diagnostic features

## Constraints

- **Methodological**: NEVER use test set for hyperparameter optimization — only final evaluation
- **Medical**: Only apply augmentations that preserve diagnostic semantic content (lungs, pathology patterns)
- **Visual**: Maintain consistency with existing thesis figures (fonts, colors, layout)
- **Technical**: Use existing model checkpoints in `outputs/classifier_cv/fold_01-05/best_classifier.pt`
- **Timeline**: Not urgent but reasonable completion (weeks not months) — thesis writing in progress

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Soft voting over hard voting | Probability averaging captures model confidence, superior to majority vote | — Pending |
| Conservative TTA (flip horizontal focus) | Radiographs are medical images; preserve diagnostic features | — Pending |
| 5 CV models ensemble | Diversity from different data partitions adds complementary information | — Pending |
| Test set used only for final evaluation | Methodological rigor for thesis validity | — Pending |
| User updates LaTeX manually | Allows validation of results before committing to thesis document | — Pending |

---
*Last updated: 2026-01-27 after initialization*
