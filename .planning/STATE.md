# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-27)

**Core value:** Maximize test set accuracy using existing cross-validation models while preserving methodological integrity (no test set contamination)
**Current focus:** TTA Integration

## Current Position

Phase: 3 of 5 (TTA Integration)
Plan: 3 of 3 in current phase
Status: Phase complete
Last activity: 2026-01-27 — Completed 03-03-PLAN.md (Gap closure: case-level impact wiring)

Progress: [██████░░░░] 60%

## Performance Metrics

**Velocity:**
- Total plans completed: 7
- Average duration: 7 min
- Total execution time: 0.92 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-pre-implementation-audit | 2 | 8 min | 4 min |
| 02-ensemble-core | 2 | 10 min | 5 min |
| 03-tta-integration | 3 | 34 min | 11 min |

**Recent Trend:**
- Last 5 plans: 02-02 (7m), 03-01 (15m), 03-02 (11m), 03-03 (8m)
- Trend: Phase 3 maintains complexity (11m avg) - gap closure plan faster than initial infrastructure

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Case-level impact fully wired: CLI now tracks helped/hurt/neutral outcomes, enables TTA value assessment (IMPLEMENTED - 03-03)
- TTA validated at 98.26% accuracy: +0.16pp improvement over baseline, net +3 samples corrected (VALIDATED - 03-02)
- TTA per-class impact varies: COVID +0.44%, Normal +0.12%, Viral -0.28% F1 delta (VALIDATED - 03-02)
- Case-level tracking implemented: 6 helped, 3 hurt, 1886 neutral samples (IMPLEMENTED - 03-02)
- Dual-level TTA averaging: Applied at both model-level and ensemble-level for maximum variance reduction (IMPLEMENTED - 03-01)
- No symmetry correction for classifier TTA: Class labels are anatomically symmetric, unlike landmarks (IMPLEMENTED - 03-01)
- Soft voting over hard voting: Probability averaging captures model confidence, superior to majority vote (VALIDATED - 02-02)
- 5 CV models ensemble: Diversity from different data partitions adds complementary information (VALIDATED - 02-02)
- Ensemble achieves 98.10% accuracy: +0.42pp improvement over baseline, 47% error reduction (VALIDATED - 02-02)
- Validation F1-macro weighting: Avoids test contamination by using validation metrics as weights (VALIDATED - 02-01)
- Test set used only for final evaluation: Methodological rigor for thesis validity (VALIDATED - 01-01)
- Baseline metrics fully verified: 97.68% accuracy confirmed with exact match (difference = 0.000000) (VALIDATED - 01-02)
- Methodology validated through 4 independent methods: Test set properly isolated (git, logs, timestamps, configs) (VALIDATED - 01-02)

### Pending Todos

None yet.

### Blockers/Concerns

**From Phase 01 (Pre-Implementation Audit):**
1. Data cleanup required for Phase 5 - Must remove 9 duplicate images (1 test, 8 val) before final evaluation
2. Impact assessment pending - Re-evaluate on cleaned test set to confirm 97.68% baseline holds
3. Root cause known - Original COVID-19 Radiography Dataset had sequential duplicate images (e.g., Normal-817/818)

**From Phase 02 (Ensemble Core):**
1. TTA target established - Ensemble baseline 98.10% accuracy to beat in Phase 3
2. Configuration pattern validated - Can extend ensemble_classifier.json with TTA flags
3. Device mismatch bug fixed - torch.ones must match model device (CPU vs CUDA)

**From Phase 03 (TTA Integration) - COMPLETE:**
1. TTA improvement validated - 98.26% accuracy (+0.16pp over 98.10% baseline)
2. Case-level impact fully wired - Functions integrated into CLI, output JSON includes analysis
3. Case-level metrics validated - 6 helped, 3 hurt, net +3 samples improvement (matches ground truth)
4. Per-class impact documented - COVID benefits most (+0.44% F1), Viral degrades slightly (-0.28% F1)
5. GROUND_TRUTH.json updated - with_tta section contains validated metrics

## Session Continuity

Last session: 2026-01-27 22:46 UTC (plan execution)
Stopped at: Completed 03-03-PLAN.md - Gap closure (case-level impact wiring)
Resume file: None
Next: Phase 3 verification pending - Ready for verifier to validate must_haves
