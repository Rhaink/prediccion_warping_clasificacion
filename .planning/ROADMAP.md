# Roadmap: COVID-19 Detection Ensemble Enhancement

## Overview

This roadmap delivers an ensemble classifier with Test-Time Augmentation (TTA) to improve COVID-19 detection accuracy from chest X-rays. Starting with validation of existing system integrity, the project implements soft/hard voting across 5 cross-validation models, adds conservative TTA with horizontal flip, and produces thesis-ready visualizations with comprehensive metrics. The final phase evaluates the complete test set (1,895 images) with rigorous validation to demonstrate reproducible improvement while maintaining methodological integrity.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3, 4, 5): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Pre-Implementation Audit** - Validate test set integrity and document current state
- [x] **Phase 2: Ensemble Core** - Implement model loading and soft/hard voting with baseline metrics
- [ ] **Phase 3: TTA Integration** - Add test-time augmentation with conservative horizontal flip
- [ ] **Phase 4: Analysis & Visualization** - Generate comparison metrics and thesis-ready confusion matrices
- [ ] **Phase 5: Final Test Evaluation** - Evaluate complete test set with validation checks

## Phase Details

### Phase 1: Pre-Implementation Audit
**Goal**: Verify test set integrity and establish baseline methodology documentation
**Depends on**: Nothing (first phase)
**Requirements**: VALID-01, VALID-02, VALID-03
**Success Criteria** (what must be TRUE):
  1. Test dataset contains exactly 1,895 images with correct class distribution (COVID=452, Normal=1,274, Viral_Pneumonia=169)
  2. Current baseline accuracy (97.68% average) is confirmed as test set performance, not validation
  3. Documentation exists proving test set was never used for model selection or hyperparameter tuning
  4. Ensemble model seeds (123, 321, 111, 666) selection methodology is documented
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md - Data integrity verification and methodology audit
- [x] 01-02-PLAN.md - Baseline re-evaluation and final audit report

### Phase 2: Ensemble Core
**Goal**: Load 5 CV models and implement soft/hard voting with per-model baseline metrics
**Depends on**: Phase 1
**Requirements**: ENSEMBLE-01, ENSEMBLE-02, ENSEMBLE-03, METRICS-01, METRICS-02, OUTPUT-01
**Success Criteria** (what must be TRUE):
  1. Script successfully loads 5 models from outputs/classifier_cv/fold_01-05/best_classifier.pt
  2. Soft voting (probability averaging) produces ensemble predictions for any input batch
  3. Hard voting (majority vote) produces baseline comparison predictions
  4. Individual metrics computed for all 5 CV fold models (accuracy, F1-macro, F1-weighted)
  5. Ensemble aggregated metrics computed (accuracy, F1-macro, F1-weighted)
  6. Results JSON file generated at outputs/classifier_cv/ensemble_test_results.json with complete metrics breakdown
**Plans**: 2 plans

Plans:
- [x] 02-01-PLAN.md - Implement ensemble module and CLI command
- [x] 02-02-PLAN.md - Create configuration and execute evaluation

### Phase 3: TTA Integration
**Goal**: Add conservative test-time augmentation with horizontal flip and expand metrics
**Depends on**: Phase 2
**Requirements**: TTA-01, TTA-02, METRICS-03, METRICS-04
**Success Criteria** (what must be TRUE):
  1. Horizontal flip TTA implemented (no symmetry correction needed - class labels are anatomically symmetric)
  2. TTA configuration controlled via JSON (use_tta parameter) with CLI override (--tta/--no-tta)
  3. Per-class breakdown computed for all predictions (COVID/Normal/Viral_Pneumonia accuracy, precision, recall, F1)
  4. Case-level impact tracking categorizes each sample as helped/hurt/neutral by TTA
  5. Ensemble+TTA improves over ensemble-only baseline (measurable accuracy delta)
**Plans**: 2 plans

Plans:
- [x] 03-01-PLAN.md - Implement TTA prediction functions and CLI integration
- [ ] 03-02-PLAN.md - Execute evaluation with case-level tracking and update GROUND_TRUTH

### Phase 4: Analysis & Visualization
**Goal**: Generate comparison analysis and thesis-ready confusion matrix visualizations
**Depends on**: Phase 3
**Requirements**: METRICS-05, OUTPUT-02, OUTPUT-03
**Success Criteria** (what must be TRUE):
  1. Comparison metrics computed showing ensemble vs individual average vs best individual model (quantified improvement)
  2. Confusion matrix visualization generated matching Chapter 5 thesis style (fonts, colors, layout consistency)
  3. Existing visualization scripts located from .tex file references or codebase patterns identified
  4. Visualizations are publication-ready with proper labels, legends, and thesis formatting
**Plans**: TBD

Plans:
- [ ] 04-01: TBD

### Phase 5: Final Test Evaluation
**Goal**: Evaluate final ensemble+TTA configuration on complete test set with validation checks
**Depends on**: Phase 4
**Requirements**: ENSEMBLE-04, VALID-01, VALID-02, VALID-03
**Success Criteria** (what must be TRUE):
  1. Ensemble+TTA evaluated on complete test set with all 1,895 images (verified sample count)
  2. Final accuracy improvement falls within expected range (+0.5 to +1.0 points over 97.68% baseline)
  3. Final results documented with confirmation that test set was used only for final evaluation (no optimization)
  4. Results package ready for thesis inclusion (JSON metrics, confusion matrices, methodology documentation)
**Plans**: TBD

Plans:
- [ ] 05-01: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Pre-Implementation Audit | 2/2 | Complete | 2026-01-27 |
| 2. Ensemble Core | 2/2 | Complete | 2026-01-27 |
| 3. TTA Integration | 1/2 | In progress | - |
| 4. Analysis & Visualization | 0/TBD | Not started | - |
| 5. Final Test Evaluation | 0/TBD | Not started | - |
