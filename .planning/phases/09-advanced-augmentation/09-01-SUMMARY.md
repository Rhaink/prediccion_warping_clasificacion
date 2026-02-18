---
phase: 09-advanced-augmentation
plan: 01
subsystem: training
tags: [albumentations, elastic-transform, grid-distortion, mixup, cutmix, augmentation, pytorch, torchvision]

# Dependency graph
requires:
  - phase: 08-training-improvements
    provides: cross_validate_classifier with curriculum/focal/mining flags and config system
provides:
  - AlbumentationsWrapper class bridging albumentations to torchvision Compose pipeline
  - get_classifier_transforms() with elastic, grid distortion, pixel augmentation flags
  - MixUp/CutMix batch-level application in cross_validate_classifier training loop
  - preview_augmentations.py with SSIM-annotated visual grids for all augmentation types
  - 8 ablation config files with user-approved parameters (elastic alpha=20, not alpha=1)
  - User-approved augmentation list: elastic(alpha=20), grid, pixel, mixup, cutmix — alpha=1 rejected
affects: [09-02-ablation-training, augmentation-experiments]

# Tech tracking
tech-stack:
  added: [albumentations>=2.0.0]
  patterns:
    - AlbumentationsWrapper PIL->numpy->albumentations->PIL bridge pattern for torchvision integration
    - MixUp/CutMix soft-label handling with argmax fallback for training accuracy tracking
    - SSIM-based augmentation validation with status thresholds (PASS/REVIEW/FAIL)
    - Config-first augmentation flags defaulting to False for backward compatibility

key-files:
  created:
    - scripts/preview_augmentations.py
    - configs/cv_aug_elastic.json
    - configs/cv_aug_grid.json
    - configs/cv_aug_pixel.json
    - configs/cv_aug_mixup.json
    - configs/cv_aug_cutmix.json
    - configs/cv_aug_elastic_curriculum.json
    - configs/cv_aug_grid_curriculum.json
    - configs/cv_aug_mixup_curriculum.json
  modified:
    - requirements.txt
    - src_v2/models/classifier.py
    - src_v2/cli.py

key-decisions:
  - "border_mode=0 (BORDER_CONSTANT) with fill=0.0 required for all spatial albumentations transforms — warped images have black background that must be preserved"
  - "MixUp takes precedence over CutMix if both flags set simultaneously (conflict resolved with warning)"
  - "F.cross_entropy accepts soft labels (B, C) in PyTorch >= 2.0 — FocalLoss is compatible without modification"
  - "SSIM FAIL status for PixelAug (0.57) is expected — SSIM measures intensity structure, but pixel brightness changes lower it artificially; visual review is the actual gate"
  - "User REJECTED ElasticTransform alpha=1 (SSIM=1.0, essentially no-op) and APPROVED alpha=20 (SSIM=0.9974, visible smooth deformation)"
  - "cv_aug_elastic.json and cv_aug_elastic_curriculum.json updated to elastic_alpha=20.0 after user review"
  - "All new augmentation flags default to False for full backward compatibility per Phase 8 pattern"

patterns-established:
  - "Augmentation insertion point: AFTER Resize, BEFORE RandomHorizontalFlip in train transform chain"
  - "MixUp/CutMix applied at batch level in training loop, NEVER in eval/validation path"
  - "Preview script forces p=1.0 for visibility — ablation configs use p=0.5 for actual training"

requirements-completed: [AUG-01, AUG-02, AUG-03]

# Metrics
duration: 8min
completed: 2026-02-18
---

# Phase 09 Plan 01: Augmentation Infrastructure Summary

**Albumentations augmentation pipeline (elastic/grid/pixel/MixUp/CutMix) with SSIM-validated visual gate, user-approved parameters, and 8 ablation configs ready for training**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-18T10:33:10Z
- **Completed:** 2026-02-18T10:41:00Z
- **Tasks:** 3 of 3 complete
- **Files modified:** 13

## Accomplishments

- Implemented `AlbumentationsWrapper` (PIL->numpy->albumentations->PIL bridge) and extended `get_classifier_transforms()` with 7 new augmentation parameters
- Integrated MixUp/CutMix batch-level mixing in `cross_validate_classifier` training loop with soft-label accuracy handling
- Created preview script with SSIM metrics and visual grids for all 6 augmentation types
- Generated 8 ablation config files (5 standalone + 3 curriculum-combined), all updated with user-approved parameters
- Visual gate checkpoint completed: user approved 5 augmentation types, rejected alpha=1 elastic (no-op)

## SSIM Preview Results and User Decisions

| Augmentation | Mean SSIM | Auto-Status | User Decision |
|---|---|---|---|
| ElasticTransform(alpha=1, sigma=30) | 1.0000 | PASS | **REJECTED** — no-op, invisible effect |
| ElasticTransform(alpha=20, sigma=30) | 0.9974 | PASS | **APPROVED** — visible smooth deformation |
| GridDistortion(distort=0.1) | 0.8043 | FAIL (threshold 0.95) | **APPROVED** — clinically acceptable distortion |
| PixelAug (brightness+noise) | 0.5665 | FAIL (threshold 0.90) | **APPROVED** — SSIM drop expected for pixel aug |
| MixUp(alpha=0.4) | N/A | INFO | **APPROVED** — cross-class regularization acceptable |
| CutMix(alpha=0.2) | N/A | INFO | **APPROVED** — patch replacement acceptable |

## Task Commits

1. **Task 1: Install albumentations, implement augmentation transforms, and extend CLI** - `b16dcfdb` (feat)
2. **Task 2: Create preview script and ablation config files** - `7b0fb4b8` (feat)
3. **Task 3: Visual validation gate — record decisions and update elastic configs** - `[this commit]` (chore)

## Files Created/Modified

- `requirements.txt` - Added `albumentations>=2.0.0`
- `src_v2/models/classifier.py` - Added `AlbumentationsWrapper`, extended `get_classifier_transforms()` with 7 augmentation params
- `src_v2/cli.py` - Added 12 new augmentation config keys, MixUp/CutMix init, batch-level mixing in training loop
- `scripts/preview_augmentations.py` - SSIM-annotated visual validation gate script
- `configs/cv_aug_elastic.json` - Elastic ablation config (updated to alpha=20 after user review)
- `configs/cv_aug_grid.json` - Grid distortion ablation config
- `configs/cv_aug_pixel.json` - Pixel augmentation ablation config
- `configs/cv_aug_mixup.json` - MixUp ablation config
- `configs/cv_aug_cutmix.json` - CutMix ablation config
- `configs/cv_aug_elastic_curriculum.json` - Elastic + curriculum config (updated to alpha=20)
- `configs/cv_aug_grid_curriculum.json` - Grid + curriculum combined config
- `configs/cv_aug_mixup_curriculum.json` - MixUp + curriculum combined config

## Decisions Made

- `border_mode=0` (BORDER_CONSTANT) with `fill=0.0` for all spatial albumentations — preserves black background of warped images
- MixUp takes precedence over CutMix if both flags set simultaneously
- `F.cross_entropy` in `FocalLoss` already accepts soft labels (B, C) in PyTorch >= 2.0 — no FocalLoss changes needed
- SSIM FAIL status for PixelAug is expected: SSIM penalizes intensity changes even when structure is preserved
- User rejected ElasticTransform alpha=1 (SSIM=1.0, no-op); approved alpha=20 (SSIM=0.9974)
- Both elastic configs updated to `elastic_alpha: 20.0` to reflect user decision
- All new flags default to False for backward compatibility

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed KeyError in print_summary_table for mixing augmentation results**
- **Found during:** Task 2 (verification run of preview script)
- **Issue:** `preview_mixup()` and `preview_cutmix()` returned dicts with key `ssim` but `print_summary_table()` expected `mean_ssim` — causing KeyError crash after printing spatial aug results
- **Fix:** Standardized return dicts to use `mean_ssim` and `std_ssim` keys; removed redundant post-assignment in `main()`; changed `print_summary_table` to use `.get()` for safe access
- **Files modified:** `scripts/preview_augmentations.py`
- **Verification:** Script ran to completion, all 6 rows printed in summary table
- **Committed in:** `7b0fb4b8` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug)
**Impact on plan:** Bug fix necessary for script to complete. No scope creep.

## Issues Encountered

- albumentations was already installed (2.0.8) — only needed to add to requirements.txt
- SSIM "FAIL" for GridDistortion (0.80) and PixelAug (0.57) are metric artifacts resolved by visual review: both approved by user

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Augmentation infrastructure complete and backward-compatible
- User decisions recorded: 5 augmentation types approved, alpha=1 elastic rejected
- Both elastic configs updated to `elastic_alpha=20.0`
- 8 ablation configs ready for Plan 02 training
- Plan 02 should run: elastic(alpha=20), grid, pixel, mixup, cutmix — plus the 3 curriculum-combined variants for the winning standalone augmentations

---
*Phase: 09-advanced-augmentation*
*Completed: 2026-02-18*
