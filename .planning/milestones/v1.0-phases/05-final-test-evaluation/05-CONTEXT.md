# Phase 5: Final Test Evaluation - Context

**Gathered:** 2026-02-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Evaluate the final ensemble+TTA configuration on the complete test set (1,895 images) with rigorous validation checks. Produce a thesis-ready results package including metrics, methodology summary, and documentation of methodological integrity. Run on both original and cleaned (duplicate-free) datasets for transparency.

</domain>

<decisions>
## Implementation Decisions

### Results package
- Claude decides JSON structure (extend existing vs separate file) based on existing patterns
- Claude decides whether to reference Phase 4 figures or regenerate based on validity
- GROUND_TRUTH.json must be updated with a `final_evaluation` section as the canonical source of truth
- Generate a standalone Markdown methodology summary covering the full pipeline (data prep, landmarks, warping, ensemble, TTA, final evaluation) in Spanish

### Validation checks
- Re-verify test set isolation programmatically as part of the final evaluation script (file hashes, path comparisons)
- If accuracy falls outside expected range (+0.5 to +1.0pp over 97.68%): warn but continue — the actual number is what it is
- Claude decides strictness level for class count verification (COVID=452, Normal=1274, Viral_Pneumonia=169)
- Run evaluation twice and assert identical outputs to prove deterministic reproducibility

### Reporting format
- All human-readable output in Spanish (labels, headers, narrative text)
- Claude decides console output verbosity based on existing CLI patterns
- Methodology summary as standalone Markdown (.md), full pipeline scope, in Spanish

### Duplicate handling
- Run evaluation on BOTH original (1,895) and cleaned (duplicates removed) test sets
- Claude decides whether to remove or replace the 1 test duplicate (methodologically sound approach)
- Report both results equally in the thesis with a note about the known duplicates
- Claude decides implementation: on-the-fly cleanup vs separate pre-step

### Claude's Discretion
- JSON output file structure and naming
- Whether to regenerate Phase 4 figures or reference existing
- Class count verification strictness (assert vs warn)
- Console output verbosity level
- Duplicate removal implementation approach (on-the-fly vs pre-step)
- Test duplicate handling (remove vs replace)

</decisions>

<specifics>
## Specific Ideas

- Double-run reproducibility check: run twice, assert bit-identical results to prove determinism
- Methodology summary should be useful as a thesis appendix reference — covers the complete pipeline end-to-end
- Both dataset variants (original and cleaned) presented equally — no preference for one over the other
- Spanish language for all human-facing outputs, matching thesis language

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-final-test-evaluation*
*Context gathered: 2026-02-16*
