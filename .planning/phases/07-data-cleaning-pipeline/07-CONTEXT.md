# Phase 7: Data Cleaning Pipeline - Context

**Gathered:** 2026-02-17
**Status:** Ready for planning

<domain>
## Phase Boundary

Remove or correct data quality issues identified in Phase 6 before re-training models. This includes: filtering landmark outliers before warping, detecting label noise with cleanlab, manual review of flagged samples, and producing a cleaning manifest with full traceability. The cleaned dataset feeds into Phase 8 (re-training).

</domain>

<decisions>
## Implementation Decisions

### Cross-split duplicate resolution
- Clean at the **original image level** (before warping) to address root cause — re-warp after cleaning
- Exclude uncertain/noisy samples rather than correcting labels
- **Prioritize data quality** over dataset size — acceptable to lose 10-15% of training data if it improves signal
- Report **both** original v1.0 accuracy (98.26%) AND corrected baseline after removing leakers, for transparent comparison in thesis

### Outlier landmark filtering
- Use **>3 sigma** threshold for automatic removal (auto-keep anything between 2-3 sigma, no manual review for borderline)
- Use **combined metric**: flag if overall Procrustes distance >3 sigma OR any single landmark >4 sigma — catches both globally distorted shapes and localized landmark errors

### Claude's Discretion
- **Duplicate resolution strategy**: Claude picks between removing from train only vs re-splitting, based on methodological rigor and v1.0 comparability
- **pHash threshold**: Claude adjusts from Phase 6's threshold=3 based on the distance distribution found during the audit
- **Review tool format**: Claude picks the review interface (notebook, HTML report, or CLI) based on what integrates best with existing Phase 6 tooling
- **Cleanlab target**: Claude decides whether to run cleanlab on warped or original images based on practicality (existing 5-fold predictions available for warped)
- **Auto-exclude threshold**: Claude sets the cleanlab confidence threshold for automatic exclusion vs manual review
- **Manifest scope**: Claude decides whether manifest documents only removed samples or full audit trail (all flagged, kept or removed)
- **Post-cleaning flow**: Claude decides whether to auto-re-warp or stop for user review of manifest before proceeding

</decisions>

<specifics>
## Specific Ideas

- User wants the leakage impact shown transparently: original accuracy alongside corrected baseline
- Cleaning happens upstream (originals), not downstream (warped) — root cause approach
- Conservative on borderline landmarks: only remove clear outliers (>3 sigma), don't waste effort reviewing 2-3 sigma range
- Quality-first philosophy: better to have a smaller, cleaner dataset than preserve noisy samples

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 07-data-cleaning-pipeline*
*Context gathered: 2026-02-17*
