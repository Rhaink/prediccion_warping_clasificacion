# Requirements: COVID-19 Detection Ensemble Enhancement

**Defined:** 2026-01-27
**Core Value:** Maximize test set accuracy using existing cross-validation models while preserving methodological integrity (no test set contamination)

## v1 Requirements

Requirements for ensemble+TTA implementation. Each maps to roadmap phases.

### Ensemble Core

- [ ] **ENSEMBLE-01**: Load 5 CV models from `outputs/classifier_cv/fold_01-05/best_classifier.pt`
- [ ] **ENSEMBLE-02**: Implement soft voting (average probabilities from 5 models)
- [ ] **ENSEMBLE-03**: Implement hard voting (majority vote) for baseline comparison
- [ ] **ENSEMBLE-04**: Evaluate on complete test set (1,895 images: COVID=452, Normal=1,274, Viral_Pneumonia=169)

### Test-Time Augmentation

- [ ] **TTA-01**: Apply horizontal flip with symmetry correction (validated in landmark pipeline)
- [ ] **TTA-02**: JSON-driven configuration (enable/disable TTA, augmentation parameters)

### Metrics & Evaluation

- [ ] **METRICS-01**: Compute accuracy, F1-macro, F1-weighted for ensemble
- [ ] **METRICS-02**: Compute individual metrics per model (all 5 CV folds)
- [ ] **METRICS-03**: Per-class breakdown (COVID/Normal/Viral_Pneumonia)
- [ ] **METRICS-04**: Generate confusion matrix for ensemble predictions
- [ ] **METRICS-05**: Compare ensemble vs individual average vs best individual model

### Output & Visualization

- [ ] **OUTPUT-01**: Generate `outputs/classifier_cv/ensemble_test_results.json` with complete metrics
- [ ] **OUTPUT-02**: Create confusion matrix visualization matching Chapter 5 thesis style
- [ ] **OUTPUT-03**: Auto-locate existing visualization scripts from .tex file references

### Validation

- [ ] **VALID-01**: Verify total sample count equals 1,895 images
- [ ] **VALID-02**: Verify improvement gain (+0.5 to +1.0 points over 97.68% baseline)
- [ ] **VALID-03**: Document test set used only for final evaluation (no hyperparameter tuning)

## v2 Requirements

Deferred to future work or thesis extensions.

### Advanced Analysis

- **ANALYSIS-01**: Disagreement analysis (identify where/why models differ)
- **ANALYSIS-02**: Uncertainty quantification (ensemble variance per prediction)
- **ANALYSIS-03**: Confidence calibration (temperature scaling, ECE metrics)
- **ANALYSIS-04**: Per-sample confidence scores for clinical routing

### Extended TTA

- **TTA-EXT-01**: Small rotation augmentations (±5°) validated on medical images
- **TTA-EXT-02**: Brightness/contrast jitter (±5-10%) if medically safe
- **TTA-EXT-03**: TTA sample size optimization (N augmentations per image)

### Rigorous Validation

- **RIGOR-01**: Pre-implementation audit (verify ensemble not cherry-picked on test)
- **RIGOR-02**: Ablation studies (model-only, +TTA, +ensemble, +both)
- **RIGOR-03**: CLAIM 2024 compliance checklist (44 items for medical AI)
- **RIGOR-04**: Confidence intervals on all reported metrics
- **RIGOR-05**: External validation on FedCOVIDx dataset

## Out of Scope

Explicitly excluded to prevent scope creep and maintain methodological rigor.

| Feature | Reason |
|---------|--------|
| Training new ensemble models | Already have 5 trained CV models; defeats "quick ensemble" timeline |
| Test set optimization | Test contamination risk; methodological violation for thesis |
| Aggressive TTA (rotation >15°, vertical flip, cutout) | Destroys diagnostic features in medical images; unsafe for radiographs |
| Weighted voting by validation accuracy | Overfits to validation split; complexity without proven benefit |
| Real-time inference optimization | Premature for research phase; thesis focuses on accuracy not speed |
| MC Dropout uncertainty estimation | Ensemble disagreement simpler and sufficient for thesis scope |
| Automatic LaTeX chapter updates | User validates results first, then manually updates thesis document |
| Hard voting as primary method | Soft voting superior for calibrated models; hard voting only for comparison |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENSEMBLE-01 | TBD | Pending |
| ENSEMBLE-02 | TBD | Pending |
| ENSEMBLE-03 | TBD | Pending |
| ENSEMBLE-04 | TBD | Pending |
| TTA-01 | TBD | Pending |
| TTA-02 | TBD | Pending |
| METRICS-01 | TBD | Pending |
| METRICS-02 | TBD | Pending |
| METRICS-03 | TBD | Pending |
| METRICS-04 | TBD | Pending |
| METRICS-05 | TBD | Pending |
| OUTPUT-01 | TBD | Pending |
| OUTPUT-02 | TBD | Pending |
| OUTPUT-03 | TBD | Pending |
| VALID-01 | TBD | Pending |
| VALID-02 | TBD | Pending |
| VALID-03 | TBD | Pending |

**Coverage:**
- v1 requirements: 14 total
- Mapped to phases: 0 (pending roadmap creation)
- Unmapped: 14 ⚠️

---
*Requirements defined: 2026-01-27*
*Last updated: 2026-01-27 after initial definition*
