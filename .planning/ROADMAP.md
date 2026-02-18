# Roadmap: COVID-19 Detection Ensemble Enhancement

## Milestones

- ✅ **v1.0 Ensemble+TTA** - Phases 1-5 (shipped 2026-02-16)
- 🚧 **v1.1 Data-Centric Accuracy Improvement** - Phases 6-10 (in progress)

## Phases

<details>
<summary>✅ v1.0 Ensemble+TTA (Phases 1-5) - SHIPPED 2026-02-16</summary>

**Milestone Goal:** Ensemble+TTA classifier achieving 98.26% test accuracy, a 47% error reduction over the 97.68% individual model baseline.

- [x] Phase 1: Pre-Implementation Audit (2/2 plans) - completed 2026-01-27
- [x] Phase 2: Ensemble Core (2/2 plans) - completed 2026-01-27
- [x] Phase 3: TTA Integration (3/3 plans) - completed 2026-01-27
- [x] Phase 4: Analysis & Visualization (2/2 plans) - completed 2026-02-16
- [x] Phase 5: Final Test Evaluation (2/2 plans) - completed 2026-02-16

**Delivered:**
- Validated test set integrity with 4 independent isolation methods
- Implemented weighted soft voting ensemble (5 CV models) achieving 98.10% accuracy (+0.42pp)
- Integrated dual-level TTA (horizontal flip) achieving 98.26% accuracy (+0.58pp total)
- Generated thesis-ready confusion matrices and LaTeX comparison tables
- Created comprehensive 683-line Spanish methodology document for thesis appendix
- Deterministic reproducibility proof (hash-verified dual-run)

</details>

### 🚧 v1.1 Data-Centric Accuracy Improvement (In Progress)

**Milestone Goal:** Push ensemble accuracy beyond 98.26% through data quality improvements across the full pipeline (landmarks → warping → classification), keeping ResNet-18 architecture fixed to isolate the data effect.

- [ ] **Phase 6: Error Forensics & Data Quality Audit** - Understand the 33 misclassified images and assess dataset quality
- [x] **Phase 7: Data Cleaning Pipeline** - Filter outliers, detect label noise, and document cleaning decisions (completed 2026-02-17)
- [x] **Phase 8: Training Improvements** - Implement focal loss, hard example mining, and curriculum learning (completed 2026-02-17)
- [ ] **Phase 9: Advanced Augmentation** - Medical-specific augmentations with ablation studies
- [ ] **Phase 10: Final Evaluation & Statistical Validation** - Comparative evaluation with statistical testing

## Phase Details

### Phase 6: Error Forensics & Data Quality Audit
**Goal**: Understand failure modes of the current 98.26% model and identify data quality issues that can be fixed
**Depends on**: Phase 5 (v1.0 complete)
**Requirements**: ERR-01, ERR-02, ERR-03, ERR-04
**Success Criteria** (what must be TRUE):
  1. Analyst can visually inspect all 33 misclassified test images with predicted/true labels and confidence scores
  2. Error patterns are categorized (high-confidence errors suggest label noise, low-margin errors suggest hard examples)
  3. Duplicate and near-duplicate images are detected across the full dataset with documented pairs
  4. Image quality scores (BRISQUE/NIQE) are computed for all images to identify low-quality samples
  5. Error forensics report exists documenting root causes and recommended fixes for each error category
**Plans**: 3 plans

Plans:
- [ ] 06-01-PLAN.md — Error analysis core and pipeline trace visualization (ERR-01, ERR-04)
- [ ] 06-02-PLAN.md — Duplicate detection and image quality assessment (ERR-02, ERR-03)
- [ ] 06-03-PLAN.md — Interactive notebook and Spanish forensics report

### Phase 7: Data Cleaning Pipeline
**Goal**: Remove or correct data quality issues identified in Phase 6 before re-training models
**Depends on**: Phase 6
**Requirements**: CLN-01, CLN-02, CLN-03, CLN-04
**Success Criteria** (what must be TRUE):
  1. Images with outlier landmarks (>3 sigma from canonical shape) are filtered before warping
  2. Potential label noise is detected using cleanlab confident learning on 5-fold CV predictions
  3. All flagged samples undergo manual review with documented accept/reject decisions
  4. Data cleaning manifest (JSON) documents every excluded/corrected sample with reasoning
  5. Cleaned dataset is ready for re-training with full traceability to original dataset
**Plans**: 3 plans

Plans:
- [ ] 07-01-PLAN.md — Landmark outlier detection and cross-split duplicate resolution (CLN-01)
- [ ] 07-02-PLAN.md — OOF probability extraction and cleanlab label noise detection (CLN-02)
- [ ] 07-03-PLAN.md — Manifest assembly, review notebook, and CLI exclude-list integration (CLN-03, CLN-04)

### Phase 8: Training Improvements
**Goal**: Re-train 5-fold CV ensemble with focal loss, hard example mining, and curriculum learning on cleaned data
**Depends on**: Phase 7
**Requirements**: TRN-01, TRN-02, TRN-03, TRN-04
**Success Criteria** (what must be TRUE):
  1. Classifier uses focal loss (gamma=2.0) instead of weighted CrossEntropy to address class imbalance
  2. Training pipeline supports hard example mining by oversampling frequently misclassified samples
  3. Training supports curriculum learning with easy-to-hard schedule based on loss
  4. New 5-fold CV ensemble is trained on improved data with same ResNet-18 architecture
  5. Viral Pneumonia recall improves above baseline 92.9% (current worst class)
**Plans**: 3 plans

Plans:
- [ ] 08-01-PLAN.md — Re-warp cleaned dataset, implement FocalLoss/mining/curriculum, create ablation configs (TRN-01, TRN-02, TRN-03, TRN-04)
- [ ] 08-02-PLAN.md — Run 4 individual ablations: cleaned baseline + focal + mining + curriculum (TRN-01, TRN-02, TRN-03, TRN-04)
- [ ] 08-03-PLAN.md — Run combined model fine-tuned from best ablation + comparison analysis (TRN-04)

### Phase 9: Advanced Augmentation
**Goal**: Integrate medical-specific augmentations that preserve anatomical validity and improve generalization
**Depends on**: Phase 8
**Requirements**: AUG-01, AUG-02, AUG-03
**Success Criteria** (what must be TRUE):
  1. Training uses medical-specific augmentations via albumentations (ElasticTransform, GridDistortion with anatomical constraints)
  2. Training supports batch-level MixUp and CutMix augmentation
  3. Each augmentation strategy is tested individually with ablation study results documented
  4. Visual validation confirms augmented samples preserve diagnostic features
  5. Augmentation parameters are conservative and medically appropriate (validated against literature)
**Plans**: 3 plans

Plans:
- [ ] 09-01-PLAN.md — Augmentation infrastructure, preview script, visual gate checkpoint (AUG-01, AUG-02, AUG-03)
- [ ] 09-02-PLAN.md — Run individual and curriculum-combined augmentation ablations (AUG-01, AUG-02, AUG-03)
- [ ] 09-03-PLAN.md — Comparison analysis script with dual baselines (AUG-03)

### Phase 10: Final Evaluation & Statistical Validation
**Goal**: Validate that data-centric improvements produce statistically significant accuracy gains over v1.0 baseline
**Depends on**: Phase 9
**Requirements**: EVL-01, EVL-02, EVL-03, EVL-04, EVL-05
**Success Criteria** (what must be TRUE):
  1. Comparative evaluation between v1.0 baseline and data-improved ensemble is complete on same test set
  2. Case-level impact analysis documents which samples were helped vs hurt vs neutral
  3. McNemar's paired statistical test confirms improvement is statistically significant (p < 0.05)
  4. Confidence intervals are reported for all accuracy claims
  5. Regression guardrail passes: fewer than 5 new errors introduced compared to baseline
**Plans**: TBD

Plans:
- [ ] 10-01: TBD
- [ ] 10-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 6 → 7 → 8 → 9 → 10

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Pre-Implementation Audit | v1.0 | 2/2 | Complete | 2026-01-27 |
| 2. Ensemble Core | v1.0 | 2/2 | Complete | 2026-01-27 |
| 3. TTA Integration | v1.0 | 3/3 | Complete | 2026-01-27 |
| 4. Analysis & Visualization | v1.0 | 2/2 | Complete | 2026-02-16 |
| 5. Final Test Evaluation | v1.0 | 2/2 | Complete | 2026-02-16 |
| 6. Error Forensics | v1.1 | 0/3 | Planned | - |
| 7. Data Cleaning | v1.1 | Complete    | 2026-02-17 | - |
| 8. Training Improvements | v1.1 | 3/3 | Complete | 2026-02-17 |
| 9. Advanced Augmentation | 1/3 | In Progress|  | - |
| 10. Final Evaluation | v1.1 | 0/2 | Not started | - |

---
*Last updated: 2026-02-17 after Phase 8 execution complete*
