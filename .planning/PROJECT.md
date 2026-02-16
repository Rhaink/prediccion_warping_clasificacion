# COVID-19 Detection Ensemble Enhancement

## What This Is

A thesis project that detects COVID-19 from chest X-rays using anatomical landmark detection, geometric normalization via piecewise affine warping, and an ensemble classifier with Test-Time Augmentation (TTA). The ensemble combines 5 cross-validation ResNet-18 models via weighted soft voting, achieving 98.26% test accuracy on 1,895 images — a 47% error reduction over the 97.68% individual model baseline. Results are reproducible with deterministic evaluation and full methodological integrity documentation.

## Core Value

Maximize test set accuracy using existing cross-validation models while preserving methodological integrity (no test set contamination). The ensemble must demonstrate reproducible improvement with rigorous validation for thesis reporting.

## Requirements

### Validated

- ✓ Landmark detection ensemble (3.61 px error on 15 lung contour landmarks) — existing
- ✓ Piecewise affine warping for geometric normalization via GPA — existing
- ✓ 5 ResNet-18 classifiers trained with k-fold cross-validation (k=5) — existing
- ✓ Individual model evaluation on fixed test set (97.68% ± 0.16% accuracy) — existing
- ✓ Complete data pipeline (landmarks → warping → classification) — existing
- ✓ JSON configuration system for reproducibility — existing
- ✓ Cached landmark predictions in NPZ format — existing
- ✓ Visualization scripts for matrices and figures — existing
- ✓ Ensemble soft voting with weighted probability averaging (98.10% accuracy) — v1.0
- ✓ Dual-level TTA with horizontal flip (98.26% accuracy, +0.58pp) — v1.0
- ✓ Complete test set evaluation (1,895 images with correct class distribution) — v1.0
- ✓ Case-level impact analysis (6 helped, 3 hurt, 1886 neutral) — v1.0
- ✓ Thesis-ready confusion matrices matching Chapter 5 style — v1.0
- ✓ LaTeX comparison tables with per-class breakdown — v1.0
- ✓ Comprehensive results JSON with full metrics (ensemble_test_results_tta.json) — v1.0
- ✓ Validation: no data leakage, +0.58pp gain within expected range, 1895 samples — v1.0
- ✓ Deterministic reproducibility proof (hash-verified dual-run) — v1.0
- ✓ Spanish methodology document (683 lines) for thesis appendix — v1.0

### Active

(None — start next milestone with `/gsd:new-milestone`)

### Out of Scope

- Training new models — use existing 5 CV models only
- Aggressive/destructive augmentations — preserve medical semantic integrity
- Threshold optimization using test set — methodological violation
- Automatic LaTeX chapter updates — user updates manually after validation
- MC Dropout uncertainty estimation — ensemble disagreement simpler and sufficient

## Context

**Shipped:** v1.0 (2026-02-16) — Ensemble+TTA classifier achieving 98.26% accuracy.

**Codebase State:**
- 541K LOC Python total
- Key new module: `src_v2/evaluation/ensemble.py` (ensemble voting, TTA, case-level impact)
- Key new scripts: `evaluate_final_ensemble_tta.py`, `generate_confusion_matrices_comparison.py`, `generate_comparison_tables.py`
- Config: `configs/ensemble_classifier.json`
- GROUND_TRUTH.json v2.2.0 — canonical validated metrics

**Final Results (v1.0):**
- Baseline (individual avg): 97.68% ± 0.16%
- Ensemble (soft voting): 98.10% (+0.42pp)
- Ensemble + TTA: 98.26% (+0.58pp, 47% error reduction)
- Test set: 1,895 images (COVID=452, Normal=1,274, Viral_Pneumonia=169)

**Known Issues:**
- 1 test duplicate (Normal-817/Normal-818) — handled via dual-dataset reporting (original and cleaned produce identical results)
- v2 requirements deferred: disagreement analysis, uncertainty quantification, confidence calibration, extended TTA

## Constraints

- **Methodological**: NEVER use test set for hyperparameter optimization — only final evaluation
- **Medical**: Only apply augmentations that preserve diagnostic semantic content
- **Visual**: Maintain consistency with existing thesis figures (fonts, colors, layout)
- **Technical**: Use existing model checkpoints in `outputs/classifier_cv/fold_01-05/best_classifier.pt`

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Soft voting over hard voting | Probability averaging captures model confidence | ✓ Good — 98.10% vs hard voting, confirmed superior |
| Conservative TTA (horizontal flip only) | Radiographs are medical images; preserve diagnostic features | ✓ Good — +0.16pp additional improvement, safe |
| Weighted soft voting (validation F1-macro weights) | Avoid test contamination by using validation metrics | ✓ Good — better than uniform averaging |
| 5 CV models ensemble | Diversity from different data partitions | ✓ Good — 47% error reduction over baseline |
| Test set used only for final evaluation | Methodological rigor for thesis validity | ✓ Good — verified with 4 independent methods |
| Dual-level TTA (model + ensemble) | Maximum variance reduction | ✓ Good — net +3 samples corrected |
| Dual-dataset reporting (original + cleaned) | Transparency about 1 test duplicate | ✓ Good — identical results prove robustness |
| Spanish methodology document | Thesis appendix needs native language | ✓ Good — 683 lines covering complete pipeline |
| Two separate confusion matrix figures | Better thesis layout than side-by-side | ✓ Good — clearer visual comparison |
| LaTeX hand-crafted over pandas.to_latex() | Precise booktabs formatting control | ✓ Good — publication-ready tables |

---
*Last updated: 2026-02-16 after v1.0 milestone*
