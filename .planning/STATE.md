# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-16)

**Core value:** Maximize classification accuracy through data-centric improvements while preserving methodological integrity
**Current focus:** Phase 10 Plan 2 paused at checkpoint:human-verify (Task 3). All automated tasks complete — user must verify figures, LaTeX tables, and case-level analysis before Plan 2 is marked complete.

## Current Position

Phase: 10 of 10 (Final Evaluation & Statistical Validation) — IN PROGRESS
Plan: 2 of 2 (paused at checkpoint:human-verify Task 3)
Status: Phase 10 Plan 2 Tasks 1-2 complete. 8 figures, 3 LaTeX tables, and case-level image grids generated. Awaiting human verification.
Last activity: 2026-02-20 - Completed Phase 10 Plan 2 Tasks 1-2 (generate_figures_phase10.py + generate_case_analysis_phase10.py)

Progress: [█████████░] 95% (9 phases complete + 1/2 phase 10 plans done)

## Performance Metrics

**Velocity:**
- Total plans completed: 21 (11 from v1.0 + 10 from v1.1)
- Average duration: 7 min (v1.1 code plans only, excluding GPU training time)
- Total execution time: 21 days (v1.0 milestone) + ~10h (v1.1 so far including GPU training)

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
| 8. Training Improvements | 3 of 3 | ~10h | ~3.3h (GPU) |
| 9. Advanced Augmentation | 3 of 3 | ~16h 13min | - |

**Recent Trend:**
- v1.1 Phase 8 Plan 1 complete: FocalLoss + hard mining + curriculum in CLI, cleaned dataset warped, 5 configs created in 4 minutes
- v1.1 Phase 8 Plan 2 complete: 4 ablation experiments trained (~8h GPU). Curriculum learning wins (F1=0.9932)
- v1.1 Phase 8 Plan 3 complete: Combined model (F1=0.9878) underperforms curriculum alone. Comparison script created.
- v1.1 Phase 8 COMPLETE: All 3 plans executed. Best config: curriculum learning (F1=0.9932, VP recall=99.1%)
- v1.1 Phase 9 Plan 1 complete: Albumentations infrastructure built. Visual gate passed. 5 augmentations approved (elastic alpha=20, grid, pixel, mixup, cutmix). 8 ablation configs ready.
- v1.1 Phase 9 Plan 2 complete: All 8 ablation experiments trained (~16h GPU). New best: elastic+curriculum (F1=0.9971, VP Recall=100%, surpasses curriculum-alone by +0.39pp).
- v1.1 Phase 10 Plan 1 complete: All 4 model ensembles evaluated on test set. Critical finding: validation gains did NOT transfer to test accuracy. v1.0 remains best (98.26%). McNemar not significant after Holm-Bonferroni (elastic+curriculum p=0.0339 raw, not corrected).
| Phase 10 P01 | 9 | 1 tasks | 5 files |
| Phase 10 P02 | 5 | 2 tasks | 2 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- **08-01**: Use standard CE (val_criterion) for validation across all ablations to ensure comparable F1 metrics regardless of training loss (2026-02-17)
- **08-01**: Curriculum stages sorted per-class (not globally) to preserve class balance at each stage (2026-02-17)
- **08-01**: OOF difficulty anchored at 95th percentile for mining weight clipping to avoid extreme overweighting (2026-02-17)
- **08-01**: All new technique flags default to False for full backward compatibility (2026-02-17)
- **08-02**: Curriculum learning is the best individual ablation (F1=0.9932, +0.88pp over baseline) (2026-02-17)
- **08-03**: Combined model (focal+mining+curriculum) regresses vs curriculum alone (F1=0.9878 vs 0.9932) — techniques interfere (2026-02-17)
- [Phase 09-01]: border_mode=0 with fill=0.0 required for all albumentations spatial transforms — warped images have black background
- [Phase 09-01]: SSIM FAIL for PixelAug (0.57) is expected — SSIM penalizes intensity changes; visual review is the actual gate for pixel augmentations
- [Phase 09-01]: ElasticTransform alpha=1 is essentially a no-op (SSIM=1.0); alpha=20 provides visible but medically safe deformation (SSIM=0.9974)
- [Phase 09-01]: Visual gate approved: elastic(alpha=20), grid, pixel, MixUp, CutMix APPROVED; elastic(alpha=1) REJECTED as no-op. Elastic configs updated to alpha=20.
- [Phase 09-02]: elastic_curriculum is new best (F1=0.9971, VP Recall=100%) — elastic deformation + curriculum ordering synergizes better than either alone (2026-02-19)
- [Phase 09-02]: Individual augmentations alone do not improve over baseline on warped normalized dataset — geometric normalization reduces geometric variance, making aug less impactful without curriculum (2026-02-19)
- [Phase 09-02]: Curriculum is the dominant technique — all 3 curriculum-combined experiments outperform standalone counterparts (2026-02-19)
- [Phase 09-02]: Spatial augmentations (elastic, grid) combine better with curriculum than batch-mixing (mixup, cutmix) (2026-02-19)
- [Phase 09-03]: elastic+curriculum is the final recommended config for Phase 10 (F1=0.9971, VP Recall=100%) — dual-baseline comparison confirms +0.39pp improvement over curriculum alone (2026-02-20)
- [Phase 09-03]: Individual augmentations alone universally fail to improve over curriculum-alone; only grid marginally beats cleaned baseline (+0.07pp) (2026-02-20)
- [Phase 10-01]: Sanity check tolerance set to 0.01pp because V1_BASELINE_ACCURACY=0.9826 is rounded; actual value is 0.9825858
- [Phase 10-01]: Critical finding: improved models do NOT exceed v1.0 on test set (val F1=0.9971 elastic+curriculum did not transfer to test accuracy)
- [Phase 10-01]: Regression guardrail is soft report (not abort) — all 3 comparisons failed (b=8,14,14 vs threshold=5), script continues per plan
- [Phase 10-02]: Confusion matrices reconstructed from precision/recall/support since raw CM not stored in phase10_final_report.json
- [Phase 10-02]: Case analysis cross-reference with Phase 6 forensics: 4 improved cases were in original 33 error set, 0 regressed

### Key Phase 8 Results

| Experiment | Val F1-Macro | VP Recall | vs Baseline |
|------------|-------------|-----------|-------------|
| Cleaned Baseline | 0.9844 | 97.49% | -- |
| Focal Loss | 0.9834 | 98.74% | -0.10pp |
| Hard Mining | 0.9820 | 97.49% | -0.23pp |
| **Curriculum** | **0.9932** | **99.10%** | **+0.88pp** |
| Combined | 0.9878 | 99.01% | +0.34pp |

### Key Phase 9 Results (Plan 02)

| Experiment | Val F1-Macro | VP Recall | vs Baseline |
|------------|-------------|-----------|-------------|
| elastic | 0.9840 | 97.85% | -0.04pp |
| grid | 0.9851 | 97.76% | +0.07pp |
| pixel | 0.9818 | 97.58% | -0.26pp |
| mixup | 0.9814 | 98.65% | -0.30pp |
| cutmix | 0.9774 | 98.03% | -0.70pp |
| **elastic+curriculum** | **0.9971** | **100.00%** | **+1.27pp** |
| grid+curriculum | 0.9961 | 99.64% | +1.17pp |
| mixup+curriculum | 0.9880 | 98.57% | +0.36pp |

### Key Phase 10 Results (Plan 01)

| Model | Acc(TTA) | F1-macro | VP Recall | Delta(pp) |
|-------|----------|----------|-----------|-----------|
| v1.0 Baseline | 0.9826 | 0.9712 | 0.9290 | +0.000 |
| Cleaned Baseline | 0.9810 | 0.9703 | 0.9290 | -0.158 |
| Curriculum | 0.9778 | 0.9660 | 0.9231 | -0.475 |
| Elastic+Curriculum | 0.9773 | 0.9668 | 0.9290 | -0.528 |

McNemar p-values: cleaned=0.5791, curriculum=0.0665, elastic_curriculum=0.0339 (none significant after Holm-Bonferroni)

### Pending Todos

None.

### Blockers/Concerns

**Phase 8 Status:** COMPLETE
- All 5 experiments completed with 5 folds each
- VP recall target (>92.9%) met by all experiments
- Curriculum learning is the recommended configuration for downstream phases

**Downstream Concerns:**
- Statistical significance: At 1895 test samples, each corrected sample is worth only +0.05pp
- elastic_curriculum VP Recall=100% on validation is extraordinary — verify it holds on held-out test set in Plan 03
- Spatial augmentations + curriculum is the winning pattern; batch-mixing (mixup, cutmix) does not combine well with curriculum

## Session Continuity

Last session: 2026-02-20 (Phase 10 Plan 2 Tasks 1-2 complete, paused at human-verify)
Stopped at: Paused at checkpoint:human-verify — Task 3 of 10-02-PLAN.md
Resume file: .planning/phases/10-final-evaluation-statistical-validation/10-02-SUMMARY.md

Next: User must verify outputs/phase10/figures/ and outputs/phase10/case_analysis/ then type "approved"

---
*Last updated: 2026-02-20*
