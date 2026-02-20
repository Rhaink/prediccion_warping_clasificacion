# Requirements: COVID-19 Detection Data-Centric Improvement

**Defined:** 2026-02-16
**Core Value:** Maximize classification accuracy through data-centric improvements while preserving methodological integrity

## v1.1 Requirements

Requirements for data-centric accuracy improvement. Each maps to roadmap phases.

### Error Analysis

- [ ] **ERR-01**: Analyst can visually inspect all misclassified test images with true/predicted labels and confidence scores
- [ ] **ERR-02**: System detects duplicate and near-duplicate images across the full dataset
- [ ] **ERR-03**: System computes no-reference image quality scores (BRISQUE/NIQE) for all images
- [ ] **ERR-04**: Error forensics report categorizes errors as high-confidence (suspect label noise) vs low-margin (hard examples)

### Data Cleaning

- [ ] **CLN-01**: System filters images with outlier landmarks (>3σ from canonical shape) before warping
- [ ] **CLN-02**: System detects potential label noise using cleanlab confident learning on 5-fold CV predictions
- [ ] **CLN-03**: Flagged samples undergo manual review with documented accept/reject decisions
- [ ] **CLN-04**: Data cleaning manifest (JSON) documents every excluded/corrected sample with reasoning

### Training

- [ ] **TRN-01**: Classifier uses focal loss (γ=2.0) instead of weighted CrossEntropy
- [ ] **TRN-02**: Training pipeline supports hard example mining (oversampling of frequently misclassified samples)
- [ ] **TRN-03**: Training supports curriculum learning (easy→hard schedule based on loss)
- [ ] **TRN-04**: New 5-fold CV ensemble trained on improved data with same ResNet-18 architecture

### Augmentation

- [x] **AUG-01**: Training uses medical-specific augmentations via albumentations (ElasticTransform, GridDistortion)
- [x] **AUG-02**: Training supports batch-level MixUp and CutMix augmentation
- [x] **AUG-03**: Each augmentation strategy tested individually (ablation study) with validation metrics

### Evaluation

- [ ] **EVL-01**: Comparative evaluation: v1.0 baseline vs data-improved ensemble on same test set
- [ ] **EVL-02**: Case-level impact analysis (helped vs hurt vs neutral per sample)
- [ ] **EVL-03**: McNemar's paired statistical test validates improvement significance
- [ ] **EVL-04**: Confidence intervals reported for all accuracy claims
- [ ] **EVL-05**: Regression guardrail: abort if >5 new errors introduced vs baseline

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Deferred from v1.0

- **DEF-01**: Ensemble disagreement analysis (which models disagree on which samples)
- **DEF-02**: Uncertainty quantification via ensemble variance
- **DEF-03**: Confidence calibration (temperature scaling)
- **DEF-04**: Extended TTA beyond horizontal flip

### Deferred from v1.1

- **DEF-05**: AutoAugment search for optimal augmentation policy
- **DEF-06**: Full dataset re-annotation by domain expert
- **DEF-07**: GAN-based synthetic data augmentation

## Out of Scope

| Feature | Reason |
|---------|--------|
| Architecture changes (beyond ResNet-18) | Fixed to isolate data-centric effect |
| External datasets | Improvements must use existing COVID-19 Radiography Dataset only |
| Threshold optimization using test set | Methodological violation |
| SMOTE-based oversampling | Creates unrealistic X-ray interpolations |
| Aggressive geometric augmentation (>20° rotation) | Destroys anatomical realism in chest X-rays |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ERR-01 | Phase 6 | Pending |
| ERR-02 | Phase 6 | Pending |
| ERR-03 | Phase 6 | Pending |
| ERR-04 | Phase 6 | Pending |
| CLN-01 | Phase 7 | Pending |
| CLN-02 | Phase 7 | Pending |
| CLN-03 | Phase 7 | Pending |
| CLN-04 | Phase 7 | Pending |
| TRN-01 | Phase 8 | Pending |
| TRN-02 | Phase 8 | Pending |
| TRN-03 | Phase 8 | Pending |
| TRN-04 | Phase 8 | Pending |
| AUG-01 | Phase 9 | Complete |
| AUG-02 | Phase 9 | Complete |
| AUG-03 | Phase 9 | Complete |
| EVL-01 | Phase 10 | Pending |
| EVL-02 | Phase 10 | Pending |
| EVL-03 | Phase 10 | Pending |
| EVL-04 | Phase 10 | Pending |
| EVL-05 | Phase 10 | Pending |

**Coverage:**
- v1.1 requirements: 20 total
- Mapped to phases: 20/20 (100%)
- Unmapped: 0

**Coverage by phase:**
- Phase 6: 4 requirements (ERR-01 through ERR-04)
- Phase 7: 4 requirements (CLN-01 through CLN-04)
- Phase 8: 4 requirements (TRN-01 through TRN-04)
- Phase 9: 3 requirements (AUG-01 through AUG-03)
- Phase 10: 5 requirements (EVL-01 through EVL-05)

---
*Requirements defined: 2026-02-16*
*Last updated: 2026-02-16 after roadmap creation*
