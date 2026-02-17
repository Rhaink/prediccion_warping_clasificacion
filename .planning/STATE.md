# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-16)

**Core value:** Maximize classification accuracy through data-centric improvements while preserving methodological integrity
**Current focus:** Phase 8 - Training Improvements

## Current Position

Phase: 8 of 10 (Training Improvements)
Plan: 1 of 3 complete
Status: Phase 8 Plan 1 complete, ready for Plan 2 (ablation CV execution)
Last activity: 2026-02-17 - Completed 08-01-PLAN.md (warped cleaned dataset, FocalLoss, technique CLI flags, 5 ablation configs)

Progress: [████████░░] 80% (7 phases complete + Phase 8 in progress)

## Performance Metrics

**Velocity:**
- Total plans completed: 18 (11 from v1.0 + 7 from v1.1)
- Average duration: 7 min (v1.1 only, tracked going forward)
- Total execution time: 21 days (v1.0 milestone) + 59 min (v1.1 so far)

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Pre-Implementation Audit | 2 | v1.0 | - |
| 2. Ensemble Core | 2 | v1.0 | - |
| 3. TTA Integration | 3 | v1.0 | - |
| 4. Analysis & Visualization | 2 | v1.0 | - |
| 5. Final Test Evaluation | 2 | v1.0 | - |
| 6. Error Forensics & Data Quality | 3 of 3 | 34 min | 11 min |
| 7. Data Cleaning Pipeline | 3 of 3 | 16 min | ~5 min |
| 8. Training Improvements | 1 of 3 | 4 min | - |

**Recent Trend:**
- v1.0 shipped successfully with 98.26% accuracy achieved
- v1.1 Phase 6 Plan 1 complete: Error analysis core built in 5 minutes
- v1.1 Phase 6 Plan 2 complete: Duplicate detection & quality audit in 19 minutes
- v1.1 Phase 6 Plan 3 complete: Interactive notebook & forensics report in 10 minutes
- v1.1 Phase 6 COMPLETE: All 3 plans executed successfully
- v1.1 Phase 7 Plan 1 complete: Landmark outlier detection and duplicate resolution in 8 minutes
- v1.1 Phase 7 Plan 2 complete: OOF extraction and cleanlab label noise detection in 3 minutes
- v1.1 Phase 7 Plan 3 complete: Manifest assembly + CLI --exclude-list in 5 minutes
- v1.1 Phase 7 COMPLETE: All 3 plans executed, manifest approved (432 excluded, 2.9%)
- v1.1 Phase 8 Plan 1 complete: FocalLoss + hard mining + curriculum in CLI, cleaned dataset warped, 5 configs created in 4 minutes

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- **v1.0**: ResNet-18 architecture fixed for v1.1 - Isolate data quality effect from model capacity, enable fair comparison
- **v1.0**: Soft voting over hard voting - Probability averaging captures model confidence (98.10% achieved)
- **v1.0**: Conservative TTA (horizontal flip only) - Preserve diagnostic features in medical images (+0.16pp additional improvement)
- **v1.0**: Test set used only for final evaluation - Methodological rigor for thesis validity (verified with 4 independent methods)
- **06-01**: Use confidence x fold agreement matrix for error categorization - Captures both model certainty and ensemble consensus (2026-02-16)
- **06-02**: Use pyiqa instead of pybrisque - Modern library with scikit-learn 0.26 compatibility and GPU acceleration (2026-02-16)
- **06-03**: Accept .ipynb gitignored - Notebook is a generated artifact, scripts are tracked (2026-02-16)
- **07-01**: Same-class duplicate resolution keeps alphabetically first image_name for determinism and reproducibility (2026-02-17)
- **07-01**: Cross-class pairs (6,026 of 17,312) require excluding both images due to label ambiguity (2026-02-17)
- **07-02**: Temperature scaling T=2.0 applied (94.2% of OOF samples had max_prob > 0.99 -> overconfident) (2026-02-17)
- **07-02**: All 34 cleanlab label issues had self_confidence < 0.05 -> all auto_excluded, no manual_review tier needed (2026-02-17)
- **08-01**: Use standard CE (val_criterion) for validation across all ablations to ensure comparable F1 metrics regardless of training loss (2026-02-17)
- **08-01**: Curriculum stages sorted per-class (not globally) to preserve class balance at each stage (2026-02-17)
- **08-01**: OOF difficulty anchored at 95th percentile for mining weight clipping to avoid extreme overweighting (2026-02-17)
- **08-01**: All new technique flags default to False for full backward compatibility (2026-02-17)

### Pending Todos

None yet.

### Blockers/Concerns

**Phase 8 Status:**
- ~~Focal loss implementation~~ ✓ Complete in 08-01: FocalLoss class in losses.py
- ~~Hard example mining~~ ✓ Complete in 08-01: OOF-based WeightedRandomSampler in CLI
- ~~Curriculum learning~~ ✓ Complete in 08-01: Class-balanced stages with difficulty ordering
- ~~Cleaned dataset warping~~ ✓ Complete in 08-01: 14,721 images at outputs/warped_cleaned/session_warping
- ~~Ablation configs~~ ✓ Complete in 08-01: 5 configs created and validated

**Phase 8 Plan 2 Ready:**
- 5 ablation experiments ready to run: cv_cleaned_baseline, cv_focal, cv_mining, cv_curriculum, cv_combined
- Each requires ~50 epochs x 5 folds of training (GPU time)
- Results will determine which technique (if any) improves over baseline

**Downstream Concerns:**
- Statistical significance: At 1895 test samples, each corrected sample is worth only +0.05pp
- Regression risk: Could break 1862 correct predictions while fixing 33 errors

## Session Continuity

Last session: 2026-02-17 (Phase 8 Plan 1 complete)
Stopped at: Completed 08-01-PLAN.md (infrastructure for training improvements)
Resume file: .planning/phases/08-training-improvements/08-01-SUMMARY.md

Next: Execute Phase 8 Plan 2 (run ablation CV experiments)

---
*Last updated: 2026-02-17*
