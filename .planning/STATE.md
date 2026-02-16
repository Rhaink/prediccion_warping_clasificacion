# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-16)

**Core value:** Maximize classification accuracy through data-centric improvements while preserving methodological integrity
**Current focus:** Phase 6 - Error Forensics & Data Quality Audit

## Current Position

Phase: 6 of 10 (Error Forensics & Data Quality Audit)
Plan: 2 of 3 in current phase
Status: In progress
Last activity: 2026-02-16 - Completed 06-02-PLAN.md (duplicate detection & quality assessment)

Progress: [█████░░░░░] 50% (5 of 10 phases complete from v1.0, Phase 6 Plan 2 of 3 complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 13 (11 from v1.0 + 2 from v1.1)
- Average duration: 12 min (v1.1 only, tracked going forward)
- Total execution time: 21 days (v1.0 milestone) + 24 min (v1.1 so far)

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Pre-Implementation Audit | 2 | v1.0 | - |
| 2. Ensemble Core | 2 | v1.0 | - |
| 3. TTA Integration | 3 | v1.0 | - |
| 4. Analysis & Visualization | 2 | v1.0 | - |
| 5. Final Test Evaluation | 2 | v1.0 | - |
| 6. Error Forensics & Data Quality | 2 of 3 | 24 min | 12 min |

**Recent Trend:**
- v1.0 shipped successfully with 98.26% accuracy achieved
- v1.1 Phase 6 Plan 1 complete: Error analysis core built in 5 minutes
- v1.1 Phase 6 Plan 2 complete: Duplicate detection & quality audit in 19 minutes

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
- **06-02**: Use pyiqa instead of pybrisque - Modern library with scikit-image 0.26 compatibility and GPU acceleration (2026-02-16)
- **06-02**: Skip CNN verification for initial duplicate audit - PHash with threshold=3 sufficient for conservative detection (2026-02-16)

### Pending Todos

None yet.

### Blockers/Concerns

**Phase 6 Readiness:**
- ~~Need pyiqa library for image quality assessment (BRISQUE/NIQE metrics)~~ ✓ Installed in 06-02
- Must ensure error forensics strictly post-hoc to avoid test set contamination ✓ Maintained
- ~~Error pattern categorization requires careful definition (high-confidence vs low-margin thresholds)~~ ✓ Defined in 06-01
- **CRITICAL NEW CONCERN**: 17,312 cross-split duplicates found in warped dataset - may invalidate test results

**Phase 7 Readiness:**
- Need cleanlab library for label noise detection
- Manual review process for flagged samples needs workflow definition
- Data cleaning manifest format needs specification

**Phase 8 Readiness:**
- Focal loss implementation needs testing before CV training
- Hard example mining requires tracking misclassification history during CV
- Curriculum learning scheduler needs validation set-based difficulty scoring

**Downstream Concerns:**
- Statistical significance: At 1895 test samples, each corrected sample is worth only +0.05pp
- Test set contamination risk during data cleaning must be prevented
- Regression risk: Could break 1862 correct predictions while fixing 33 errors
- **NEW**: Warping-induced convergence (42,175 pairs) may have created artificial similarity - need to investigate
- **NEW**: Cross-split leakage may invalidate accuracy claims - needs resolution before thesis

## Session Continuity

Last session: 2026-02-16 (Phase 6 Plan 2 execution)
Stopped at: Completed 06-02-PLAN.md (duplicate detection & quality assessment)
Resume file: .planning/phases/06-error-forensics-data-quality-audit/06-02-SUMMARY.md

Next: Plan 06-03 (forensics report + interactive notebook)

---
*Last updated: 2026-02-16*
