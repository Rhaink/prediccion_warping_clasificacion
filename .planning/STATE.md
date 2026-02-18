# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-16)

**Core value:** Maximize classification accuracy through data-centric improvements while preserving methodological integrity
**Current focus:** Phase 8 complete — ready for Phase 9

## Current Position

Phase: 8 of 10 (Training Improvements) — COMPLETE
Plan: 3 of 3 complete
Status: Phase 8 complete. Curriculum learning identified as best technique (F1=0.9932).
Last activity: 2026-02-17 - Completed all 3 plans of Phase 8

Progress: [████████░░] 80% (8 phases complete, 2 remaining)

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

**Recent Trend:**
- v1.1 Phase 8 Plan 1 complete: FocalLoss + hard mining + curriculum in CLI, cleaned dataset warped, 5 configs created in 4 minutes
- v1.1 Phase 8 Plan 2 complete: 4 ablation experiments trained (~8h GPU). Curriculum learning wins (F1=0.9932)
- v1.1 Phase 8 Plan 3 complete: Combined model (F1=0.9878) underperforms curriculum alone. Comparison script created.
- v1.1 Phase 8 COMPLETE: All 3 plans executed. Best config: curriculum learning (F1=0.9932, VP recall=99.1%)
| Phase 09-advanced-augmentation P01 | 6 | 2 tasks | 11 files |

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

### Key Phase 8 Results

| Experiment | Val F1-Macro | VP Recall | vs Baseline |
|------------|-------------|-----------|-------------|
| Cleaned Baseline | 0.9844 | 97.49% | -- |
| Focal Loss | 0.9834 | 98.74% | -0.10pp |
| Hard Mining | 0.9820 | 97.49% | -0.23pp |
| **Curriculum** | **0.9932** | **99.10%** | **+0.88pp** |
| Combined | 0.9878 | 99.01% | +0.34pp |

### Pending Todos

None.

### Blockers/Concerns

**Phase 8 Status:** COMPLETE
- All 5 experiments completed with 5 folds each
- VP recall target (>92.9%) met by all experiments
- Curriculum learning is the recommended configuration for downstream phases

**Downstream Concerns:**
- Statistical significance: At 1895 test samples, each corrected sample is worth only +0.05pp
- Regression risk: Could break 1862 correct predictions while fixing 33 errors
- Curriculum + other techniques interfere — keep techniques separate or test pairwise

## Session Continuity

Last session: 2026-02-17 (Phase 8 complete)
Stopped at: All 3 plans of Phase 8 completed
Resume file: .planning/phases/08-training-improvements/08-03-SUMMARY.md

Next: Plan Phase 9 (Advanced Augmentation) or verify Phase 8

---
*Last updated: 2026-02-17*
