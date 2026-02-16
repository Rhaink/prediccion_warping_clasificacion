---
phase: 04-analysis-visualization
plan: 01
subsystem: visualization
tags: [matplotlib, seaborn, confusion-matrix, thesis-figures, metrics-comparison]

# Dependency graph
requires:
  - phase: 03-tta-integration
    provides: ensemble_test_results_tta.json with case-level analysis and metrics
provides:
  - Confusion matrix comparison script generating thesis-ready figures
  - Structured comparison_metrics.json with full baseline → ensemble → TTA pipeline
  - 300 DPI PNG figures with Spanish labels and dual annotations
affects: [thesis-writing, results-presentation, 04-02-quantitative-analysis]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Dual annotation confusion matrices (counts + percentages)"
    - "Structured JSON metrics for reproducible comparisons"
    - "Matplotlib rcParams standardization across thesis figures"

key-files:
  created:
    - scripts/generate_confusion_matrices_comparison.py
    - outputs/classifier_cv/confusion_matrix_baseline.png
    - outputs/classifier_cv/confusion_matrix_ensemble_tta.png
    - outputs/classifier_cv/comparison_metrics.json
  modified: []

key-decisions:
  - "Generate two separate confusion matrices (baseline aggregated vs ensemble+TTA) instead of side-by-side subplots for clarity"
  - "Use row-wise normalization for percentages to show class-specific performance"
  - "Include case-level impact (helped/hurt/neutral) in comparison_metrics.json"

patterns-established:
  - "Pattern 1: All thesis confusion matrices use dual annotations (raw counts + percentages) for maximum information density"
  - "Pattern 2: Baseline represents 5-fold aggregated matrix (total_evaluations = 9,475) not single-model average"
  - "Pattern 3: comparison_metrics.json structure enables programmatic extraction of improvement deltas"

# Metrics
duration: 2min
completed: 2026-02-16
---

# Phase 04 Plan 01: Confusion Matrices Comparison Summary

**Thesis-ready confusion matrix figures comparing baseline (97.68% ± 0.16%) vs ensemble+TTA (98.26%) with structured metrics JSON capturing +0.58pp total improvement**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-16T10:20:07Z
- **Completed:** 2026-02-16T10:22:33Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Generated two separate 300 DPI confusion matrix PNGs with Spanish labels and dual annotations
- Created structured comparison_metrics.json containing baseline, ensemble (no TTA), ensemble+TTA, and all deltas
- Baseline confusion matrix aggregates 5 folds (9,475 total evaluations) with mean ± std metrics
- Ensemble+TTA confusion matrix shows single evaluation (1,895 samples) with case-level impact summary
- All metrics sourced from existing JSON files (no hardcoded values)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create confusion matrix comparison script and generate all outputs** - `f6946993` (feat)

## Files Created/Modified
- `scripts/generate_confusion_matrices_comparison.py` - Script generating both confusion matrices and comparison JSON from fold test results and ensemble results
- `outputs/classifier_cv/confusion_matrix_baseline.png` - Baseline aggregated 5-fold confusion matrix (gitignored, reproducible)
- `outputs/classifier_cv/confusion_matrix_ensemble_tta.png` - Ensemble+TTA confusion matrix (gitignored, reproducible)
- `outputs/classifier_cv/comparison_metrics.json` - Structured metrics with deltas (gitignored, reproducible)

## Decisions Made

**1. Two separate figures instead of subplots:** Generates `confusion_matrix_baseline.png` and `confusion_matrix_ensemble_tta.png` as separate files rather than side-by-side subplots. Rationale: Easier to integrate into thesis LaTeX independently, allows for different caption styles.

**2. Baseline aggregation approach:** Baseline confusion matrix is the SUM of 5 fold confusion matrices (total_evaluations = 9,475), not the average. Percentages are row-normalized from the aggregated matrix. Rationale: Matches methodology in existing `generate_confusion_matrix_cv.py`, preserves actual evaluation counts.

**3. Case-level impact in JSON:** Included `case_level_impact` from `ensemble_test_results_tta.json` in the comparison_metrics.json output (helped: 6, hurt: 3, neutral: 1886). Rationale: Essential for understanding TTA's granular effect on individual predictions.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed numpy int64 JSON serialization error**
- **Found during:** Task 1 (comparison_metrics.json generation)
- **Issue:** `best_fold` variable was numpy.int64, causing TypeError during json.dump()
- **Fix:** Cast to Python int: `best_fold = int(best_fold_idx + 1)`
- **Files modified:** scripts/generate_confusion_matrices_comparison.py
- **Verification:** JSON serialization successful, file created with valid structure
- **Committed in:** f6946993 (Task 1 commit)

**2. [Rule 1 - Bug] Replaced deprecated datetime.utcnow()**
- **Found during:** Task 1 (timestamp generation)
- **Issue:** DeprecationWarning for datetime.utcnow(), scheduled for removal in future Python
- **Fix:** Changed to `datetime.now().astimezone().replace(microsecond=0).isoformat()`
- **Files modified:** scripts/generate_confusion_matrices_comparison.py
- **Verification:** Timestamp generated without warnings, ISO format with timezone
- **Committed in:** f6946993 (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both fixes necessary for correct execution (JSON serialization, deprecation warnings). No scope changes.

## Issues Encountered

**outputs/ directory gitignored:** Cannot commit PNG and JSON files to git. Solution: Script is committed and reproducible; outputs can be regenerated on demand. This is consistent with project convention (outputs/ in .gitignore).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Confusion matrix figures and structured comparison metrics ready for thesis integration
- comparison_metrics.json provides programmatic access to all deltas for Phase 04 Plan 02 (quantitative analysis tables)
- Baseline accuracy (97.68%) and Ensemble+TTA accuracy (98.26%) validated against GROUND_TRUTH.json expectations
- Ready for additional visualization tasks in Phase 04

## Self-Check: PASSED

All claimed files and commits verified:
- ✓ scripts/generate_confusion_matrices_comparison.py
- ✓ outputs/classifier_cv/confusion_matrix_baseline.png
- ✓ outputs/classifier_cv/confusion_matrix_ensemble_tta.png
- ✓ outputs/classifier_cv/comparison_metrics.json
- ✓ Commit f6946993 exists

---
*Phase: 04-analysis-visualization*
*Completed: 2026-02-16*
