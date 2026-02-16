# Phase 6: Error Forensics & Data Quality Audit - Context

**Gathered:** 2026-02-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Understand failure modes of the current 98.26% ensemble model (33 misclassified test images) and assess dataset quality through duplicate detection and image quality scoring. This phase produces diagnostic analysis and reports — it does NOT clean data or retrain models (those are Phase 7 and Phase 8).

</domain>

<decisions>
## Implementation Decisions

### Error visualization
- Show both original X-ray AND warped version side-by-side for each misclassified image
- Include landmark detection overlay on original images to check if bad landmarks caused errors
- Produce both static image grids (thesis-ready) AND interactive Jupyter notebook for exploration
- Static outputs at two scales: compact overview grid (all 33 in one figure) + detailed per-sample figures for appendix
- Per-sample detailed figures use a "pipeline trace" layout: original → landmarks overlay → warped → classification result in a single row

### Error categorization
- Trace errors through the full pipeline to identify WHERE failure originated: bad landmarks → bad warp → misclassification vs good warp but ambiguous image vs possible label noise
- Include recoverability assessment: tag each error as fixable (label noise, bad landmark), partially fixable (hard example, better training may help), or inherent (genuinely ambiguous)

### Duplicate detection
- Full dataset scope: cross-split (train/val/test leakage), within-split, AND cross-class (potential label errors)
- Run duplicate detection on both original images and warped images — warping could make different images converge or same images diverge
- Both stages provide different diagnostic signals

### Claude's Discretion
- Error visualization grouping strategy (by confusion pair, by confidence, or hybrid — pick most informative)
- Metadata shown per image (minimal vs full context with fold agreement and probability bar charts)
- Whether to include "nearest correct neighbor" comparison for each misclassified image
- Error categorization thresholds (confidence-based, fold agreement, or combined matrix)
- Whether to also analyze CV validation set errors beyond the 33 test errors for larger sample
- Duplicate detection similarity threshold and metric choice (SSIM, perceptual hash, feature embeddings)
- Handling of discovered train/test leakage (quantify impact vs document only — recommend based on severity)
- Report detail level per error (summary per category vs every error documented — balance thoroughness with readability)
- Image quality score scope (full dataset distribution vs errors + control sample — balance compute cost vs diagnostic value)

</decisions>

<specifics>
## Specific Ideas

- Pipeline trace visualization is key: original → landmarks → warped → classification result shows the full chain for each error
- Report must be in Spanish — ready for thesis inclusion, consistent with v1.0 methodology appendix
- Both Jupyter notebook (reproducible, interactive exploration) AND markdown report (thesis summary) should be produced
- Dual-stage duplicate detection (original + warped) captures both dataset issues and pipeline-induced similarity

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 06-error-forensics-data-quality-audit*
*Context gathered: 2026-02-16*
