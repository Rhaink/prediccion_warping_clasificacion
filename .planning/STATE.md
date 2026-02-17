# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-16)

**Core value:** Maximize classification accuracy through data-centric improvements while preserving methodological integrity
**Current focus:** Phase 7 - Data Cleaning Pipeline

## Current Position

Phase: 7 of 10 (Data Cleaning Pipeline)
Plan: 2 of 3 complete (07-01, 07-02 done)
Status: Phase 7 in progress — OOF extraction and cleanlab label noise detection complete
Last activity: 2026-02-17 - Completed 07-02-PLAN.md (OOF extraction and cleanlab label noise detection)

Progress: [███████░░░] 65% (6 phases + 2 plans complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 16 (11 from v1.0 + 5 from v1.1)
- Average duration: 9 min (v1.1 only, tracked going forward)
- Total execution time: 21 days (v1.0 milestone) + 45 min (v1.1 so far)

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Pre-Implementation Audit | 2 | v1.0 | - |
| 2. Ensemble Core | 2 | v1.0 | - |
| 3. TTA Integration | 3 | v1.0 | - |
| 4. Analysis & Visualization | 2 | v1.0 | - |
| 5. Final Test Evaluation | 2 | v1.0 | - |
| 6. Error Forensics & Data Quality | 3 of 3 | 34 min | 11 min |
| 7. Data Cleaning Pipeline | 2 of 3 | 11 min | 5.5 min |

**Recent Trend:**
- v1.0 shipped successfully with 98.26% accuracy achieved
- v1.1 Phase 6 Plan 1 complete: Error analysis core built in 5 minutes
- v1.1 Phase 6 Plan 2 complete: Duplicate detection & quality audit in 19 minutes
- v1.1 Phase 6 Plan 3 complete: Interactive notebook & forensics report in 10 minutes
- v1.1 Phase 6 COMPLETE: All 3 plans executed successfully
- v1.1 Phase 7 Plan 1 complete: Landmark outlier detection and duplicate resolution in 8 minutes
- v1.1 Phase 7 Plan 2 complete: OOF extraction and cleanlab label noise detection in 3 minutes

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- **v1.0**: ResNet-18 architecture fixed for v1.1 - Isolate data quality effect from model capacity, enable fair comparison
- **v1.0**: Soft voting over hard voting - Probability averaging captures model confidence (98.10% achieved)
- **v1.0**: Conservative TTA (horizontal flip only) - Preserve diagnostic features in medical images (+0.16pp additional improvement)
- **v1.0**: Test set used only for final evaluation - Methodological rigor for thesis validity (verified with 4 independent methods)
- **06-01**: Use confidence x fold agreement matrix for error categorization - Captures both model certainty and ensemble consensus (2026-02-16)
- **06-01**: Placeholder confidence values acceptable for v1 - Per-sample probabilities not in ensemble JSON, can refine if needed for Phase 07-08 (2026-02-16)
- **06-02**: Use pyiqa instead of pybrisque - Modern library with scikit-learn 0.26 compatibility and GPU acceleration (2026-02-16)
- **06-02**: Skip CNN verification for initial duplicate audit - PHash with threshold=3 sufficient for conservative detection (2026-02-16)
- **06-03**: Accept .ipynb gitignored - Notebook is a generated artifact, scripts are tracked (2026-02-16)
- **07-01**: Same-class duplicate resolution keeps alphabetically first image_name for determinism and reproducibility (2026-02-17)
- **07-01**: Cross-class pairs (6,026 of 17,312) require excluding both images due to label ambiguity (2026-02-17)
- **07-01**: 5,018 unique images excluded from 17,312 pairs — ~33% of dataset flagged for cross-split exclusion (2026-02-17)
- **07-02**: Temperature scaling T=2.0 applied (94.2% of OOF samples had max_prob > 0.99 -> overconfident) (2026-02-17)
- **07-02**: All 34 cleanlab label issues had self_confidence < 0.05 -> all auto_excluded, no manual_review tier needed (2026-02-17)

### Pending Todos

None yet.

### Blockers/Concerns

**Phase 7 Readiness:**
- ~~Cross-split duplicate resolution~~ ✓ Complete in 07-01: 5,018 exclusions produced
- ~~Landmark outlier detection~~ ✓ Complete in 07-01: 463 flagged (3.06%)
- ~~cleanlab label noise detection~~ ✓ Complete in 07-02: 34 auto_excluded issues identified
- **CONCERN**: 5,018 cross-split exclusions is ~33% of 15,153 images — manifest assembly (Plan 03) must handle overlap between exclusion categories carefully
- Manifest assembly (Plan 03) ready to proceed: all three data sources (landmark outliers, cross-split duplicates, label noise) collected

**Phase 8 Readiness:**
- Focal loss implementation needs testing before CV training
- Hard example mining requires tracking misclassification history during CV
- Curriculum learning scheduler needs validation set-based difficulty scoring

**Downstream Concerns:**
- Statistical significance: At 1895 test samples, each corrected sample is worth only +0.05pp
- Test set contamination risk during data cleaning must be prevented
- Regression risk: Could break 1862 correct predictions while fixing 33 errors
- **PARTIALLY RESOLVED**: Cross-split leakage: 5,018 images identified for exclusion; manifest assembly will determine final set

## Session Continuity

Last session: 2026-02-17 (Phase 7 Plan 2)
Stopped at: Completed 07-02-PLAN.md (OOF extraction and cleanlab label noise detection)
Resume file: .planning/phases/07-data-cleaning-pipeline/07-02-SUMMARY.md

Next: Phase 7 Plan 03 (data cleaning manifest assembly)

---
*Last updated: 2026-02-17*
