---
phase: 07-data-cleaning-pipeline
plan: 01
subsystem: data
tags: [landmark-detection, procrustes, MAD, outlier-detection, duplicate-resolution, data-cleaning]

# Dependency graph
requires:
  - phase: 06-error-forensics-data-quality-audit
    provides: cross_split_leakage.csv with 17,312 duplicate pairs and warped filename keys
  - phase: 05-final-test-evaluation
    provides: outputs/landmark_predictions/session_warping/predictions.npz (15,153 images)
  - phase: 04-analysis-visualization
    provides: outputs/shape_analysis/canonical_shape_gpa.json (canonical shape)
provides:
  - outputs/data_cleaning/landmark_outliers.csv — 15,153-row per-image Procrustes + per-landmark sigma scores with flag status
  - outputs/data_cleaning/cross_split_exclusions.csv — 5,018 original image names to exclude from cross-split duplicates
  - outputs/data_cleaning/duplicate_resolution_summary.json — statistics on 17,312 pairs processed
  - scripts/run_landmark_outlier_detection.py — reusable outlier detection with configurable sigma thresholds
  - scripts/run_duplicate_resolution.py — reusable duplicate resolution with full reverse-mapping validation
affects:
  - 07-03-manifest-assembly — consumes both CSVs as inputs for cleaning manifest
  - 07-02 — label noise detection (cleanlab) may intersect with outlier list

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Robust statistics (MAD-based): use median + 1.4826*MAD instead of mean+std to avoid outlier inflation of threshold
    - Combined flagging criterion: Procrustes >3 sigma OR per-landmark >4 sigma (OR not AND, catches different failure modes)
    - Reverse-mapping validation: always validate 100% of warped paths map to images.csv before proceeding

key-files:
  created:
    - scripts/run_landmark_outlier_detection.py
    - scripts/run_duplicate_resolution.py
    - outputs/data_cleaning/landmark_outliers.csv
    - outputs/data_cleaning/cross_split_exclusions.csv
    - outputs/data_cleaning/duplicate_resolution_summary.json
  modified: []

key-decisions:
  - "Canonical shape key is 'canonical_shape_normalized' (not 'canonical_shape') in canonical_shape_gpa.json"
  - "Same-class duplicate resolution: keep alphabetically first image_name, exclude the other (deterministic, reproducible)"
  - "Cross-class duplicate resolution: exclude both images due to label ambiguity"
  - "5,018 unique images flagged for exclusion from 17,312 pairs — many images appear in multiple pairs"

patterns-established:
  - "Outlier detection: center+scale+align (full Procrustes) before computing per-landmark deviations"
  - "Robust sigma: sigma_equiv = 1.4826 * MAD converts MAD to Gaussian-equivalent sigma"
  - "images.csv warped_path key format: '{split}/{category}/{warped_filename}'"

requirements-completed: [CLN-01]

# Metrics
duration: 8min
completed: 2026-02-17
---

# Phase 7 Plan 01: Landmark Outlier Detection & Duplicate Resolution Summary

**MAD-based Procrustes outlier detection (463 flagged, 3.06%) and cross-split duplicate reverse-mapping (5,018 excluded from 17,312 pairs) producing CSV-format exclusion lists for manifest assembly**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-17T06:37:09Z
- **Completed:** 2026-02-17T06:38:54Z
- **Tasks:** 1
- **Files modified:** 2 scripts created, 3 outputs generated

## Accomplishments

- Landmark outlier detection using robust MAD-based statistics: 463 images flagged (3.06%), safely within expected 0.1-10% range per Pitfall 4 guidance
- Cross-split duplicate resolution with 100% reverse-mapping success (0 unmapped of 17,312 pairs), producing 5,018 unique exclusions
- Both outputs (landmark_outliers.csv, cross_split_exclusions.csv) are CSV-format ready for manifest assembly in Plan 03

## Task Commits

1. **Task 1: Implement landmark outlier detection and cross-split duplicate resolution** - `03a9bcb6` (feat)

**Plan metadata:** (see final commit below)

## Files Created/Modified

- `scripts/run_landmark_outlier_detection.py` - Detects landmark outliers via Procrustes distance and per-landmark deviation with robust MAD statistics; outputs 15,153-row CSV with sigma scores
- `scripts/run_duplicate_resolution.py` - Reverse-maps warped duplicate pairs to original image names via images.csv; outputs exclusion CSV and summary JSON
- `outputs/data_cleaning/landmark_outliers.csv` - 15,153 rows: image_name, category, procrustes_distance, max_landmark_deviation, procrustes_z, max_landmark_z, flagged, flag_reason
- `outputs/data_cleaning/cross_split_exclusions.csv` - 5,018 rows: image_name, category, reason, duplicate_partner, duplicate_partner_category
- `outputs/data_cleaning/duplicate_resolution_summary.json` - Statistics: 17,312 pairs, 11,286 same-class, 6,026 cross-class, 5,018 unique excluded

## Key Results

| Metric | Value |
|--------|-------|
| Total images evaluated | 15,153 |
| Landmark outliers flagged | 463 (3.06%) |
| Procrustes-only flags | 242 |
| Per-landmark-only flags | 0 |
| Both criteria flags | 221 |
| Procrustes sigma (MAD equiv) | 0.0456 |
| Per-landmark sigma (MAD equiv) | 0.0212 |
| Cross-split pairs processed | 17,312 |
| Same-class pairs | 11,286 |
| Cross-class pairs | 6,026 |
| Unmapped pairs | 0 (100% mapped) |
| Unique exclusions | 5,018 |
| Same-class exclusions | 3,475 |
| Cross-class exclusions | 1,543 |

## Decisions Made

- Canonical shape JSON key is `canonical_shape_normalized` (unit-normalised, range ~[-0.25, 0.25]) not `canonical_shape` — discovered from inspecting the file during execution
- Same-class duplicate resolution keeps alphabetically first image_name for determinism and reproducibility across runs
- Cross-class pairs (6,026 of 17,312) require excluding both images due to label ambiguity — these are flagged `cross_class_duplicate`
- Note: 5,018 unique images excluded from 17,312 pairs because many images appear as one half of multiple duplicate pairs

## Deviations from Plan

None — plan executed exactly as written. The only minor adaptation was discovering the actual JSON key name (`canonical_shape_normalized`) by inspecting the file before writing the script, which was straightforward exploration rather than a code deviation.

## Issues Encountered

None — both scripts ran cleanly on first execution, all 17,312 warped paths mapped successfully.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `outputs/data_cleaning/landmark_outliers.csv` ready for Plan 03 manifest assembly
- `outputs/data_cleaning/cross_split_exclusions.csv` ready for Plan 03 manifest assembly
- Plan 02 (label noise detection with cleanlab) can run independently in parallel
- Note: 5,018 cross-split exclusions is a large fraction of the 15,153-image dataset (~33%) — manifest assembly should decide how to handle images that appear in multiple exclusion categories

## Self-Check: PASSED

All files verified present on disk. Commit 03a9bcb6 confirmed in git log.

| Check | Result |
|-------|--------|
| scripts/run_landmark_outlier_detection.py | FOUND |
| scripts/run_duplicate_resolution.py | FOUND |
| outputs/data_cleaning/landmark_outliers.csv | FOUND |
| outputs/data_cleaning/cross_split_exclusions.csv | FOUND |
| outputs/data_cleaning/duplicate_resolution_summary.json | FOUND |
| .planning/phases/07-data-cleaning-pipeline/07-01-SUMMARY.md | FOUND |
| Commit 03a9bcb6 | FOUND |

---
*Phase: 07-data-cleaning-pipeline*
*Completed: 2026-02-17*
