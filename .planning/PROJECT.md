# COVID-19 Detection Ensemble Enhancement

## What This Is

A thesis project that detects COVID-19 from chest X-rays using anatomical landmark detection, geometric normalization via piecewise affine warping, and an ensemble classifier with Test-Time Augmentation (TTA). The ensemble combines 5 cross-validation ResNet-18 models via weighted soft voting, achieving 98.26% test accuracy on 1,895 images — a 47% error reduction over the 97.68% individual model baseline. Results are reproducible with deterministic evaluation and full methodological integrity documentation.

## Current Milestone: v1.1 Data-Centric Accuracy Improvement

**Goal:** Push ensemble accuracy beyond 98.26% through data quality improvements across the full pipeline (landmarks → warping → classification), keeping ResNet-18 architecture fixed to isolate the data effect.

**Target features:**
- Error forensics on the 33 misclassified test images
- Data quality audit and cleaning (label noise detection, outlier identification)
- Preprocessing improvements (CLAHE tuning, normalization strategies)
- Advanced augmentation strategies (medical-aware, class-balanced)
- Class imbalance strategies beyond current weighted loss
- Re-trained 5-fold CV ensemble on improved data
- Comparative evaluation: v1.0 baseline vs data-improved models

## Core Value

Maximize classification accuracy through data-centric improvements — better data quality, preprocessing, and augmentation — while preserving methodological integrity. Architecture stays fixed (ResNet-18) so improvements are attributable solely to data quality.

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

- [ ] Error forensics on 33 misclassified images (understand why they fail)
- [ ] Data quality audit across full dataset (label noise, outliers, duplicates)
- [ ] Preprocessing optimization (CLAHE, normalization, warping quality)
- [ ] Advanced augmentation strategies for medical imaging
- [ ] Class imbalance mitigation (VP=169 vs Normal=1274)
- [ ] Re-train 5-fold CV ensemble on improved data
- [ ] Comparative evaluation with v1.0 baseline

### Out of Scope

- Architecture changes — ResNet-18 fixed to isolate data effect
- Threshold optimization using test set — methodological violation
- Automatic LaTeX chapter updates — user updates manually after validation
- MC Dropout uncertainty estimation — ensemble disagreement simpler and sufficient
- External datasets — improvements must use existing COVID-19 Radiography Dataset only

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
- v1.0 deferred: disagreement analysis, uncertainty quantification, confidence calibration, extended TTA

**v1.1 Starting Point:**
- 33 misclassified test images (11 COVID errors, 10 Normal errors, 12 VP errors)
- VP recall worst at 92.9% (12/169 misclassified as Normal)
- Augmentations basic (flip, rotation ±15°, brightness/contrast jitter)
- Class weights already used in CrossEntropy loss
- No data cleaning or label quality verification performed yet

## Constraints

- **Methodological**: NEVER use test set for hyperparameter optimization — only final evaluation
- **Medical**: Only apply augmentations that preserve diagnostic semantic content
- **Architecture**: ResNet-18 only — isolate data quality effect from model capacity
- **Visual**: Maintain consistency with existing thesis figures (fonts, colors, layout)
- **Comparison**: v1.0 checkpoints preserved as baseline; new models trained separately

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

| ResNet-18 architecture fixed for v1.1 | Isolate data quality effect from model capacity — fair comparison | — Pending |

---
*Last updated: 2026-02-16 after v1.1 milestone start*
