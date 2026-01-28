---
phase: 03-tta-integration
plan: 02
subsystem: evaluation
tags: [tta, metrics, case-level-analysis, delta-computation, ensemble, validation]

# Dependency graph
requires:
  - phase: 03-tta-integration
    plan: 01
    provides: TTA infrastructure with dual-level averaging
  - phase: 02-ensemble-core
    plan: 02
    provides: Baseline ensemble metrics (98.10% accuracy)
provides:
  - Case-level TTA impact tracking (helped/hurt/neutral categorization)
  - Delta metrics computation (TTA vs baseline)
  - Validated TTA improvement (+0.16pp accuracy)
  - Full TTA evaluation results with 1,895 test samples
affects: [thesis-results-chapter, final-evaluation, deployment-decision]

# Tech tracking
tech-stack:
  added: []
  patterns: [case-level-impact-analysis, delta-computation, impact-categorization]

key-files:
  created:
    - outputs/classifier_cv/ensemble_test_results_tta.json
    - outputs/classifier_cv/ensemble_test_results_no_tta.json
    - outputs/classifier_cv/ensemble_predictions_tta.csv
  modified:
    - src_v2/evaluation/ensemble.py
    - GROUND_TRUTH.json

key-decisions:
  - "TTA improves accuracy by +0.16pp (98.10% → 98.26%)"
  - "TTA helped 6 samples, hurt 3 samples (net +3)"
  - "Per-class impact varies: COVID +0.44%, Normal +0.12%, Viral -0.28%"
  - "TTA meets validation criteria (>= baseline, no degradation)"

patterns-established:
  - "Case-level impact categorization: helped (wrong→correct), hurt (correct→wrong), neutral (same outcome)"
  - "Delta metrics track improvement across overall and per-class metrics"
  - "Validation workflow: TTA evaluation, baseline comparison, case-level analysis, GROUND_TRUTH update"

# Metrics
duration: 11 min
completed: 2026-01-28
---

# Phase 3 Plan 2: TTA Evaluation Summary

**Case-level impact tracking, delta metrics, and validated TTA improvement over 98.10% baseline**

## Performance

- **Duration:** 11 min
- **Started:** 2026-01-28T03:57:07Z
- **Completed:** 2026-01-28T04:09:03Z
- **Tasks:** 2
- **Files modified:** 2 (src_v2/evaluation/ensemble.py, GROUND_TRUTH.json)
- **Files created:** 3 (results JSONs and predictions CSV in outputs/)

## Accomplishments

- Implemented `categorize_tta_impact` for per-sample helped/hurt/neutral tracking
- Implemented `compute_tta_delta_metrics` for TTA vs baseline comparison
- Executed full TTA evaluation on 1,895 test samples with **98.26% accuracy**
- Compared against baseline (no-TTA) with **98.10% accuracy**
- Validated **+0.16pp improvement** (+3 net samples corrected)
- Updated GROUND_TRUTH.json with validated TTA metrics and case-level impact

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement case-level impact and delta metrics** - `bcce29e8` (feat)
2. **Task 2: Execute TTA evaluation and update GROUND_TRUTH** - `0ffb420c` (feat)

## Files Created/Modified

- `src_v2/evaluation/ensemble.py` - Added `categorize_tta_impact` and `compute_tta_delta_metrics` functions
- `GROUND_TRUTH.json` - Updated `classifier_ensemble_cv` section with `with_tta` metrics
- `outputs/classifier_cv/ensemble_test_results_tta.json` - Full TTA evaluation results
- `outputs/classifier_cv/ensemble_test_results_no_tta.json` - Baseline (no-TTA) comparison
- `outputs/classifier_cv/ensemble_predictions_tta.csv` - Per-sample TTA predictions

## Decisions Made

1. **TTA validated as improvement**: TTA achieved 98.26% accuracy (+0.16pp over baseline 98.10%), meeting the "no degradation" requirement and providing modest but consistent improvement.

2. **Case-level impact is net positive**: Out of 1,895 samples, TTA helped 6 cases (baseline wrong → TTA correct) and hurt 3 cases (baseline correct → TTA wrong), yielding net +3 samples improvement.

3. **Per-class impact varies**: COVID class benefited most (+0.44% F1), Normal improved slightly (+0.12% F1), while Viral_Pneumonia degraded slightly (-0.28% F1). This suggests TTA is most effective for the largest class (COVID).

4. **GROUND_TRUTH.json structure redesign**: Changed `ensemble_soft_voting` to `baseline_no_tta` and added `with_tta` sibling section for clearer baseline/TTA comparison. This makes the progression from Phase 2 → Phase 3 explicit.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all tasks completed successfully with passing verification tests.

## User Setup Required

None - evaluation ran on existing CV models and test set.

## Next Phase Readiness

- **Phase 3 complete**: TTA integration done, improvement validated
- **Phase 4 ready**: Documented results available for thesis writing
- **Phase 5 ready**: Final evaluation can use TTA-enabled ensemble
- **Key result for thesis**: +0.16pp accuracy improvement from TTA (small but consistent gain)
- **No blockers**: All success criteria met

## Validation Results

**Test Set:** 1,895 samples (verified)

**Baseline (No TTA):**
- Accuracy: 98.10%
- F1-macro: 0.9703

**With TTA:**
- Accuracy: 98.26%
- F1-macro: 0.9712
- Improvement: +0.16pp accuracy, +0.09pp F1-macro

**Case-Level Impact:**
- Helped: 6 samples (baseline wrong → TTA correct)
- Hurt: 3 samples (baseline correct → TTA wrong)
- Neutral: 1,886 samples (same outcome)
- Net improvement: +3 samples

**Per-Class F1 Delta:**
- COVID: +0.44% (most benefit)
- Normal: +0.12% (slight benefit)
- Viral_Pneumonia: -0.28% (slight degradation)

**Validation Checks:**
- ✓ TTA accuracy >= baseline (98.26% >= 98.10%)
- ✓ Test set size = 1,895 samples
- ✓ Results JSON includes `tta_enabled: true`
- ✓ GROUND_TRUTH.json updated with validated metrics
- ✓ Case-level impact documented
- ✓ Delta metrics computed and saved

---
*Phase: 03-tta-integration*
*Completed: 2026-01-28*
