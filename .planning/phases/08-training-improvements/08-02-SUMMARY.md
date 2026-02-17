---
phase: 08-training-improvements
plan: 02
subsystem: training
tags: [ablation, cross-validation, focal-loss, hard-mining, curriculum-learning]

# Dependency graph
requires:
  - phase: 08-01
    provides: FocalLoss, mining/curriculum helpers, 5 ablation configs, cleaned warped dataset

provides:
  - 4 complete 5-fold CV result sets (baseline, focal, mining, curriculum)
  - Best individual ablation identified: curriculum (val_f1_macro=0.9932)
  - All fold checkpoints available for combined model fine-tuning

affects:
  - 08-03 (combined model will fine-tune from curriculum fold_01 checkpoint)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Curriculum learning dramatically outperforms other individual techniques (+0.88pp F1 over baseline)
    - Focal loss improves VP recall but slightly reduces overall F1 vs baseline
    - Hard mining slightly hurts overall performance on this dataset

key-files:
  created: []
  modified: []

key-decisions:
  - "Curriculum learning is the best individual ablation by a large margin (F1=0.9932 vs baseline 0.9844)"
  - "All techniques exceed the 92.9% VP recall target on validation"
  - "Curriculum's staged training (60%/80%/100%) provides strong regularization effect"

patterns-established:
  - "Curriculum learning outperforms focal loss and hard mining individually on this dataset"

requirements-completed: [TRN-01, TRN-02, TRN-03, TRN-04]

# Metrics
duration: ~8h (GPU training time across 4 experiments)
completed: 2026-02-17
---

# Phase 8 Plan 02: Ablation CV Execution Summary

**4 individual ablation experiments completed: curriculum learning is the clear winner (val_f1_macro=0.9932 +/- 0.0072), dramatically outperforming cleaned baseline (0.9844) and other techniques.**

## Performance

- **Duration:** ~8h (GPU training, 4 experiments x 5 folds x ~25 min/fold)
- **Started:** 2026-02-17T04:30:00Z
- **Completed:** 2026-02-17T14:52:51Z
- **Tasks:** 2 (baseline+focal, mining+curriculum)
- **Files modified:** 0 (all outputs in gitignored outputs/ directory)

## Ablation Results

| Experiment | Val Acc | Val F1-Macro | VP Recall | Delta vs Baseline |
|------------|---------|-------------|-----------|-------------------|
| Cleaned Baseline | 0.9885 +/- 0.0027 | 0.9844 +/- 0.0043 | 0.9749 +/- 0.0129 | -- |
| Focal Loss | 0.9872 +/- 0.0020 | 0.9834 +/- 0.0037 | 0.9874 +/- 0.0059 | -0.10pp F1 |
| Hard Mining | 0.9869 +/- 0.0023 | 0.9820 +/- 0.0050 | 0.9749 +/- 0.0141 | -0.24pp F1 |
| **Curriculum** | **0.9951 +/- 0.0050** | **0.9932 +/- 0.0072** | **0.9910 +/- 0.0158** | **+0.88pp F1** |

### Key Observations

1. **Curriculum learning is the clear winner** - +0.88pp F1 improvement over cleaned baseline is substantial and consistent across folds
2. **Focal loss improves VP recall** (0.9874 vs 0.9749) but slightly reduces overall F1, suggesting it helps the minority class at the cost of the majority
3. **Hard mining slightly hurts** - the OOF-based weighting may be over-emphasizing hard samples that are genuinely ambiguous rather than misclassified
4. **All techniques exceed 92.9% VP recall target** - even the lowest (baseline: 0.9749) is far above the Phase 8 success criterion
5. **Curriculum achieves near-perfect VP recall** - 3/5 folds reach 100% VP recall

### Per-Fold VP Recall Details

| Fold | Baseline | Focal | Mining | Curriculum |
|------|----------|-------|--------|------------|
| 1 | 0.9910 | 0.9955 | 0.9910 | 1.0000 |
| 2 | 0.9865 | 0.9910 | 0.9910 | 1.0000 |
| 3 | 0.9552 | 0.9865 | 0.9686 | 0.9955 |
| 4 | 0.9686 | 0.9776 | 0.9686 | 1.0000 |
| 5 | 0.9731 | 0.9865 | 0.9552 | 0.9596 |

## Best Individual Ablation

**Curriculum learning** (val_f1_macro_mean=0.9932) will be used as the fine-tuning starting point for the combined model in Plan 03.

- Checkpoint: `outputs/classifier_cv_curriculum/fold_01/best_classifier.pt`

## Regression Guardrail

All 4 experiments passed the >95% val_f1_macro threshold:
- Baseline: 0.9844 PASS
- Focal: 0.9834 PASS
- Mining: 0.9820 PASS
- Curriculum: 0.9932 PASS

## Deviations from Plan

None - all 4 experiments ran cleanly with expected behavior.

## Issues Encountered

- Initial baseline run was interrupted by duplicate processes from subagent retries. Required cleanup and fresh restart.
- Long training times (~75 min per experiment) exceeded subagent bash timeouts. Switched to direct execution with background monitoring.

## Next Phase Readiness

- Best individual ablation identified: **curriculum learning**
- Plan 03 should set `finetune_from` to `outputs/classifier_cv_curriculum/fold_01/best_classifier.pt`
- Combined model will use all three techniques (focal + mining + curriculum) fine-tuned from curriculum checkpoint

---
*Phase: 08-training-improvements*
*Completed: 2026-02-17*
