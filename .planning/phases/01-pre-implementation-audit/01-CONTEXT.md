# Phase 1: Pre-Implementation Audit - Context

**Gathered:** 2026-01-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Validate test set integrity and establish baseline methodology documentation before implementing ensemble+TTA. This phase verifies that existing system state is methodologically sound (no test contamination, accurate baseline metrics, proper model selection protocol). Audit findings must pass all checks to proceed to Phase 2.

</domain>

<decisions>
## Implementation Decisions

### Audit scope & evidence requirements

- **Rigor level:** Standard audit (basic verification + no test images in train splits + check logs for test usage)
- **Evidence locations:** Config files, git history, training logs
- **Data leakage verification:** Hash images to verify no duplicates across train/test splits
- **Success threshold:** Must pass all checks - any integrity issue blocks Phase 2 and requires investigation

### Documentation format & depth

- **Output location:** Markdown in `.planning/` (AUDIT_REPORT.md for workflow tracking)
- **Detail level:** Technical reference (detailed commands, file paths, verification steps for reproducibility)
- **Include recommendations:** Yes, actionable fixes for each issue found
- **Structure:** Hybrid format (summary checklist + detailed sections for each check)

### Seed selection methodology

- **Selection method:** Validation-based (trained multiple seeds, selected best on validation set)
- **Existing evidence:** None yet - need to create documentation during audit
- **If proof missing:** Recreate selection (re-evaluate all seeds on validation, document retroactively)
- **Documentation requirement:** Formal protocol (document exact criteria: "Selected top 4 by val F1-macro" or similar)

### Baseline verification approach

- **Verification method:** Both approaches (check files first, re-run if inconsistencies found)
- **Files to check:**
  - `outputs/classifier_cv/fold_*/test_results.json` (individual fold test results)
  - `outputs/classifier_cv/cross_validation_test_results.json` (aggregated CV test metrics)
  - Config files in `configs/` (verify test split configuration)
- **Re-evaluation approach:** Use existing checkpoints (load fold_01-05/best_classifier.pt, re-evaluate on test)
- **Tolerance:** Reasonable variance (97.68% ± 0.10% to allow for minor stochasticity)

### Claude's Discretion

- Exact commands for hashing images (md5sum vs sha256)
- Git history search depth (how many commits back to check)
- Formatting details for audit report sections
- Prioritization of checks if some are more critical than others

</decisions>

<specifics>
## Specific Ideas

- Research findings flagged 5 CRITICAL pitfalls: test contamination, data leakage, unsafe augmentations, inflated metrics, ensemble cherry-picking
- Project already experienced one critical mistake (reporting validation instead of test accuracy), demonstrating vulnerability
- Current baseline: 97.68% ± 0.16% average across 5 CV folds (range: 97.52% - 97.94%)
- Ensemble seeds: {123, 321, 111, 666} - selection methodology must be documented
- Test set composition: 1,895 images (COVID=452, Normal=1,274, Viral_Pneumonia=169)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-pre-implementation-audit*
*Context gathered: 2026-01-27*
