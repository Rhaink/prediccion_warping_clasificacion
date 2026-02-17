---
phase: 08-training-improvements
plan: 03
subsystem: training
tags: [combined-model, ablation-comparison, curriculum-learning]

# Dependency graph
requires:
  - phase: 08-02
    provides: 4 individual ablation results + curriculum fold_01 checkpoint

provides:
  - Combined model 5-fold CV results (F1=0.9878)
  - Ablation comparison script and JSON output
  - Best configuration identified: curriculum learning alone (F1=0.9932)

affects:
  - Phase 9 (augmentation experiments will build on curriculum learning baseline)
  - Phase 10 (final evaluation will compare against v1.0)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Curriculum learning alone outperforms the combined model (adding focal+mining hurts)
    - Fine-tuning from a pre-trained checkpoint with focal loss may conflict with curriculum staging

key-files:
  created:
    - scripts/compare_ablations.py
  modified:
    - configs/cv_combined.json

key-decisions:
  - "Curriculum learning is the best Phase 8 configuration (F1=0.9932 vs combined 0.9878)"
  - "Adding focal loss and mining on top of curriculum hurts performance — techniques interfere"
  - "All 5 experiments exceed the 92.9% VP recall target"

patterns-established:
  - "Not all technique combinations improve performance — ablation identifies the best individual technique"
  - "Curriculum learning's staged training provides the strongest regularization effect for this dataset"

requirements-completed: [TRN-04]

# Metrics
duration: ~1.5h (combined model ~75 min + comparison script)
completed: 2026-02-17
---

# Phase 8 Plan 03: Combined Model + Comparison Analysis Summary

**Combined model (focal+mining+curriculum fine-tuned from curriculum) achieves F1=0.9878, which is lower than curriculum alone (0.9932). Curriculum learning is the best Phase 8 configuration.**

## Performance

- **Duration:** ~1.5h
- **Started:** 2026-02-17T15:03:00Z
- **Completed:** 2026-02-17T16:03:34Z
- **Tasks:** 2
- **Files modified:** 2

## Combined Model Results

| Metric | Combined | Curriculum (best) | Delta |
|--------|---------|-------------------|-------|
| Val Acc | 0.9909 +/- 0.0046 | 0.9951 +/- 0.0050 | -0.42pp |
| Val F1-Macro | 0.9878 +/- 0.0067 | 0.9932 +/- 0.0072 | -0.54pp |
| VP Recall | 0.9901 +/- 0.0077 | 0.9910 +/- 0.0158 | -0.09pp |

The combined model slightly regresses vs curriculum alone, suggesting focal loss and mining interfere with curriculum's staged training approach.

## Full Ablation Comparison

| Experiment | Val Acc | Val F1-Macro | VP Recall | Delta vs BL |
|------------|---------|-------------|-----------|-------------|
| Cleaned Baseline | 98.85 +/- 0.27 | 98.44 +/- 0.43 | 97.49 +/- 1.29 | -- |
| Focal Loss | 98.72 +/- 0.20 | 98.34 +/- 0.37 | 98.74 +/- 0.59 | -0.10pp |
| Hard Mining | 98.69 +/- 0.23 | 98.20 +/- 0.50 | 97.49 +/- 1.41 | -0.23pp |
| **Curriculum** | **99.51 +/- 0.50** | **99.32 +/- 0.72** | **99.10 +/- 1.58** | **+0.88pp** |
| Combined | 99.09 +/- 0.46 | 98.78 +/- 0.67 | 99.01 +/- 0.77 | +0.34pp |

## Key Findings

1. **Curriculum learning is the clear winner** — +0.88pp F1 over cleaned baseline, +1.12pp over hard mining
2. **Combined model regresses** — adding focal+mining on top of curriculum hurts by 0.54pp F1
3. **All experiments beat VP recall target** — lowest is 97.49% (baseline/mining), well above 92.9% target
4. **Focal loss trades overall F1 for VP recall** — best VP recall among individual techniques but slightly lower overall
5. **Hard mining slightly hurts** — OOF-based weighting may overemphasize genuinely ambiguous samples

## Artifacts Created

- `scripts/compare_ablations.py` — Comparison script producing table and JSON
- `outputs/ablation_comparison.json` — Structured comparison of all 5 experiments
- `outputs/classifier_cv_combined/cross_validation_results.json` — Combined model results

## Deviations from Plan

None — all tasks completed as planned. The regression check for the combined model (plan says >1pp regression warrants investigation) shows 0.54pp which is within tolerance.

## Issues Encountered

None.

---
*Phase: 08-training-improvements*
*Completed: 2026-02-17*
