# Phase 10: Final Evaluation & Statistical Validation - Context

**Gathered:** 2026-02-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Validate that data-centric improvements (data cleaning, curriculum learning, elastic augmentation) produce statistically significant accuracy gains over the v1.0 baseline (98.26%) on the held-out test set. Primary metric is test accuracy; F1 analyzed for accuracy vs sensitivity tradeoffs. No new training strategies or augmentations — this phase is evaluation only.

</domain>

<decisions>
## Implementation Decisions

### Comparison scope
- Full 4-model pipeline comparison: v1.0 baseline → cleaned baseline → curriculum → elastic+curriculum
- Traces improvement at each data-centric stage
- Each model compared against v1.0 baseline (3 comparisons, not pairwise)
- Re-train cleaned baseline and curriculum models to ensure consistency (elastic+curriculum uses existing checkpoints from Phase 9)
- Report both with and without TTA (horizontal flip) to isolate TTA's contribution from data-centric improvements

### Report deliverables
- JSON data files for reproducibility + LaTeX tables and figures for thesis
- All text in Spanish (labels, headers, captions, report narrative)
- Full figure suite: confusion matrices (one per model), per-class bar charts (accuracy/recall/F1 across models), and waterfall chart showing cumulative accuracy gain at each pipeline stage
- Full per-class metrics table: precision, recall, F1 for COVID, Normal, Viral Pneumonia — for each of the 4 models

### Case-level analysis
- Image grids showing actual X-ray images where predictions changed (correct→wrong and wrong→correct), with labels and confidence scores
- Useful for thesis discussion of what the model learned

### Statistical rigor
- Full test suite: McNemar's test (paired), bootstrap confidence intervals (95% CI), DeLong's test for AUC comparison
- 3 comparisons: cleaned vs v1.0, curriculum vs v1.0, elastic+curriculum vs v1.0

### Claude's Discretion
- Case-level categorization scheme (3 categories vs 5 with confidence changes)
- Regression guardrail behavior (hard gate vs soft report)
- Whether to cross-reference Phase 6 error forensics (original 33 misclassified images)
- Ensemble strategy (5-fold soft voting vs also reporting single-fold)
- Multiple comparison correction method (Bonferroni, Holm-Bonferroni, or none)
- Bootstrap iteration count (1,000 vs 10,000)

</decisions>

<specifics>
## Specific Ideas

- "La métrica más importante es el accuracy en test, aunque sí es importante que analicemos el F1 para ver la interacción entre accuracy y sensibilidad"
- Waterfall chart should visually tell the story: v1.0 (98.26%) → each improvement stage → final accuracy

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 10-final-evaluation-statistical-validation*
*Context gathered: 2026-02-20*
