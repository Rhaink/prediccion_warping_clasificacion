---
phase: 07-data-cleaning-pipeline
plan: 03
subsystem: data-quality
tags: [manifest, cleaning, cli-integration, review-notebook, cross-split-false-positive]

# Dependency graph
requires:
  - phase: 07-data-cleaning-pipeline plan 01
    provides: landmark_outliers.csv (463 flagged) and cross_split_exclusions.csv (5,018 — later invalidated)
  - phase: 07-data-cleaning-pipeline plan 02
    provides: cleanlab_issues.csv (34 auto-excluded) and oof_probabilities.npz
provides:
  - outputs/data_cleaning/cleaning_manifest.json — 15,153-entry manifest with 432 exclusions (2.9%)
  - notebooks/data_cleaning_review.ipynb — interactive review tool for flagged samples
  - src_v2/cli.py --exclude-list parameter on generate-dataset command
affects:
  - 08-training-improvements — re-warping with --exclude-list uses this manifest

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Cleaning manifest merges multiple quality signals with priority-based decision logic"
    - "--exclude-list loads manifest and skips excluded images by stem name match in warping loop"
    - "Cross-split duplicate detection on warped images produces false positives due to geometric normalization"

key-files:
  created:
    - scripts/generate_cleaning_manifest.py
    - notebooks/data_cleaning_review.ipynb
  modified:
    - src_v2/cli.py

key-decisions:
  - "Cross-split duplicate exclusions removed: pHash on warped images is invalid because warping homogenizes lung geometry, making different patients appear similar (verified 20/20 false positives)"
  - "Final exclusion set: 432 images (2.9%) from landmark outliers (399) + cleanlab auto-excluded (33)"
  - "No manual review items: all 34 cleanlab issues had self_confidence < 0.05 → auto-excluded"
  - "--exclude-list parameter is backward-compatible: no effect when not provided"

patterns-established:
  - "Never run duplicate detection on geometrically normalized (warped) images — use originals"
  - "Manifest schema_version v1: entries array with per-image flags and decisions"

requirements-completed: [CLN-03, CLN-04]

# Metrics
duration: 5min
completed: 2026-02-17
---

# Phase 7 Plan 03: Manifest Assembly, Review Notebook & CLI Integration Summary

**Cleaning manifest with 432 exclusions (2.9%) after invalidating 4,146 warping-induced false positive cross-split duplicates; CLI --exclude-list parameter for re-warping cleaned dataset**

## Performance

- **Duration:** 5 min (+ human review checkpoint)
- **Started:** 2026-02-17
- **Completed:** 2026-02-17
- **Tasks:** 3 (2 auto + 1 checkpoint)
- **Files modified:** 3

## Accomplishments

- Created cleaning manifest merging landmark outliers + cleanlab label noise signals into 15,153-entry JSON with full audit trail
- Discovered and proved that cross-split duplicate exclusions (4,146 images, 29.5%) were false positives from running pHash on warped images — verified 20/20 sampled pairs had original pHash distance 14-30 (threshold ≤3)
- Final manifest: 432 exclusions (2.9%), 0 review pending, 14,721 kept, 0 test set excluded
- Added --exclude-list parameter to generate-dataset CLI command with backward compatibility
- Created interactive Jupyter notebook for visual review of flagged samples

## Task Commits

1. **Task 1: Manifest assembly + review notebook** - `2c54a8ff` (feat)
2. **Task 2: CLI --exclude-list integration** - `e3915da5` (feat)
3. **Fix: Remove cross-split false positives** - `d3dbf3d9` (fix)

## Files Created/Modified

- `scripts/generate_cleaning_manifest.py` — Merges quality signals with --skip-cross-split flag
- `notebooks/data_cleaning_review.ipynb` — Interactive review tool (gitignored)
- `src_v2/cli.py` — Added --exclude-list parameter to generate-dataset command

## Key Results

| Metric | Value |
|--------|-------|
| Total images | 15,153 |
| Excluded | 432 (2.9%) |
| Landmark outlier exclusions | 399 |
| Cleanlab auto-excluded | 33 |
| Cross-split false positives removed | 4,146 |
| Review pending | 0 |
| Test set excluded | 0 |

## Decisions Made

- **Cross-split duplicates invalidated**: Phase 6 pHash detection was run on warped images. Warping normalizes lung geometry to canonical shape, making different patients appear similar. 20/20 sampled pairs confirmed as false positives (original pHash distance 14-30, threshold ≤3). Exclusions removed from manifest.
- **No manual review tier**: All 34 cleanlab issues had self_confidence < 0.05, falling into auto_excluded. Zero items in manual_review range (0.05-0.40).

## Deviations from Plan

- Plan expected cross-split exclusions to be valid. Investigation during human checkpoint revealed they were warping artifacts. Added --skip-cross-split flag and regenerated manifest.

## Issues Encountered

- Subagent permission denial (known issue) — executed directly in main conversation.
- Cross-split false positive discovery — resolved by skipping cross-split exclusions after verification.

## Self-Check: PASSED

| Check | Result |
|-------|--------|
| scripts/generate_cleaning_manifest.py | FOUND |
| notebooks/data_cleaning_review.ipynb | FOUND |
| src_v2/cli.py --exclude-list | VERIFIED |
| outputs/data_cleaning/cleaning_manifest.json | FOUND |
| Manifest entries = 15,153 | VERIFIED |
| Test set excluded = 0 | VERIFIED |
| Commit 2c54a8ff | FOUND |
| Commit e3915da5 | FOUND |
| Commit d3dbf3d9 | FOUND |

---
*Phase: 07-data-cleaning-pipeline*
*Completed: 2026-02-17*
