---
phase: 08-training-improvements
plan: 01
subsystem: training
tags: [focal-loss, hard-mining, curriculum-learning, cross-validation, data-cleaning, warping]

# Dependency graph
requires:
  - phase: 07-data-cleaning-pipeline
    provides: cleaning_manifest.json (432 excluded images) and oof_probabilities.npz

provides:
  - FocalLoss class in src_v2/models/losses.py
  - Extended cross_validate_classifier with focal/mining/curriculum technique flags
  - Freshly warped cleaned dataset at outputs/warped_cleaned/session_warping (14,721 images)
  - 5 ablation config files ready for experiment execution

affects:
  - 08-02 (ablation CV execution will use these configs and data)
  - 08-03 (combined model fine-tuning will reference configs)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - FocalLoss as drop-in replacement for CrossEntropyLoss (same interface, gamma parameter)
    - val_criterion (standard CE) always used for validation to keep metrics comparable across ablations
    - WeightedRandomSampler replaces shuffle=True when hard mining is active (mutually exclusive)
    - Curriculum stage transitions at 0%, 33%, 66% of total epochs with class-balanced subsets
    - Config-driven technique activation (all new flags default to False for backward compatibility)

key-files:
  created:
    - configs/warping_cleaned.json
    - configs/cv_cleaned_baseline.json
    - configs/cv_focal.json
    - configs/cv_mining.json
    - configs/cv_curriculum.json
    - configs/cv_combined.json
  modified:
    - src_v2/models/losses.py
    - src_v2/cli.py

key-decisions:
  - "Use standard CE (val_criterion) for validation across all ablations to ensure comparable F1 metrics regardless of training loss choice"
  - "Curriculum stages sorted per-class (not globally) to preserve class balance at each stage"
  - "OOF difficulty computed as per-sample CE loss from oof_probabilities.npz; 95th percentile anchors mining weight scaling"
  - "All new technique flags default to False ensuring full backward compatibility with existing CV usage"
  - "Auto-fixed Rule 1 bug: exclude_list not loaded from config in generate-dataset override_param section"

patterns-established:
  - "Ablation config pattern: one JSON per technique, all pointing to cleaned dataset, technique flags explicit"
  - "Separation of train criterion (can be focal) vs val criterion (always CE) for fair comparison"

requirements-completed: [TRN-01, TRN-02, TRN-03, TRN-04]

# Metrics
duration: 4min
completed: 2026-02-17
---

# Phase 8 Plan 01: Training Improvements Infrastructure Summary

**FocalLoss, OOF-based hard mining, and curriculum learning added to cross_validate_classifier; cleaned warped dataset generated (14,721 images, 432 excluded); 5 ablation configs ready for execution.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-17T03:08:48Z
- **Completed:** 2026-02-17T03:16:40Z
- **Tasks:** 3
- **Files modified:** 8

## Accomplishments

- Warped the cleaned dataset (432 Phase 7 exclusions applied): 14,721 images across train=10,987 / val=1,839 / test=1,895, class distribution COVID/Normal/Viral_Pneumonia maintained
- Implemented FocalLoss class (multi-class, gamma-weighted, class-weight-compatible) as drop-in replacement for CrossEntropyLoss
- Extended cross_validate_classifier with 8 new config-driven flags: use_focal_loss, focal_gamma, use_hard_mining, oof_path, mining_max_ratio, use_curriculum, curriculum_fractions, finetune_from
- Added three helper functions: load_oof_difficulty(), build_sampling_weights(), build_curriculum_stages()
- Created 5 ablation config files covering the full factorial of techniques (baseline, focal, mining, curriculum, combined)

## Task Commits

Each task was committed atomically:

1. **Task 1: Re-warp cleaned dataset** - `77409e08` (feat)
2. **Task 2: FocalLoss + CLI extensions** - `31654ac7` (feat)
3. **Task 3: 5 ablation config files** - `2fcbbccc` (feat)

## Files Created/Modified

- `src_v2/models/losses.py` - Added FocalLoss class and torch.nn.functional import
- `src_v2/cli.py` - Extended cross_validate_classifier with technique flags, helper functions, and bug fix for exclude_list config loading
- `configs/warping_cleaned.json` - Warping config with exclude_list pointing to cleaning_manifest.json
- `configs/cv_cleaned_baseline.json` - Baseline ablation (no techniques, just cleaned data)
- `configs/cv_focal.json` - Focal loss ablation (gamma=2.0)
- `configs/cv_mining.json` - Hard mining ablation (max_ratio=3.0)
- `configs/cv_curriculum.json` - Curriculum learning ablation (fractions=0.60,0.80,1.0)
- `configs/cv_combined.json` - All three techniques combined

## Decisions Made

- Use standard CE (`val_criterion`) for validation across all ablations to keep F1 metrics comparable regardless of training loss choice — focal loss is only used during training forward pass
- Sort curriculum stages per-class (not globally) to preserve class balance at each stage (essential for imbalanced COVID/Normal/Viral dataset)
- OOF difficulty anchored at 95th percentile for mining weight clipping — avoids extreme overweighting of outlier samples
- All new flags default to False for full backward compatibility

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed exclude_list not loaded from config in generate-dataset**
- **Found during:** Task 1 (Re-warp cleaned dataset)
- **Issue:** The config override_param section in generate-dataset did not include `exclude_list`, so `--config configs/warping_cleaned.json` would silently ignore the exclude_list key
- **Fix:** Added `exclude_list = override_param("exclude_list", exclude_list, "exclude_list", ("--exclude-list",))` to the config override block
- **Files modified:** src_v2/cli.py
- **Verification:** Warping ran with 432 exclusions correctly applied (confirmed in log output)
- **Committed in:** 77409e08 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 bug)
**Impact on plan:** Bug fix was essential for the warping config to work correctly. Without it, exclude_list would be silently ignored and the cleaned dataset would not apply exclusions.

## Issues Encountered

None — all tasks executed cleanly after the Rule 1 bug fix.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- All 5 ablation configs are ready for execution with `python -m src_v2 cross-validate-classifier --config configs/{name}.json`
- Cleaned warped dataset is at outputs/warped_cleaned/session_warping (14,721 images)
- FocalLoss, mining, and curriculum are implemented and tested
- Plan 02 (ablation execution) can begin immediately

---
*Phase: 08-training-improvements*
*Completed: 2026-02-17*

## Self-Check: PASSED

- configs/warping_cleaned.json: FOUND
- configs/cv_cleaned_baseline.json: FOUND
- configs/cv_focal.json: FOUND
- configs/cv_mining.json: FOUND
- configs/cv_curriculum.json: FOUND
- configs/cv_combined.json: FOUND
- .planning/phases/08-training-improvements/08-01-SUMMARY.md: FOUND
- Commit 77409e08: FOUND
- Commit 31654ac7: FOUND
- Commit 2fcbbccc: FOUND
