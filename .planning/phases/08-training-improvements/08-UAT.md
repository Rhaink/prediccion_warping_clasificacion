---
status: complete
phase: 08-training-improvements
source: 08-01-SUMMARY.md, 08-02-SUMMARY.md, 08-03-SUMMARY.md
started: 2026-02-18T12:00:00Z
updated: 2026-02-18T12:15:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Cleaned warped dataset generated with exclusions
expected: outputs/warped_cleaned/session_warping contains ~14,721 images across 3 classes with train/val/test splits. 432 Phase 7 exclusions applied.
result: pass

### 2. FocalLoss implementation in losses.py
expected: src_v2/models/losses.py contains a FocalLoss class with gamma parameter, compatible with class weights, usable as drop-in replacement for CrossEntropyLoss.
result: pass

### 3. CLI technique flags in cross-validate-classifier
expected: Running `python -m src_v2 cross-validate-classifier --help` shows new flags for focal loss, hard mining, and curriculum learning (or these are config-driven from JSON configs).
result: pass

### 4. Five ablation config files exist and are valid
expected: configs/ contains cv_cleaned_baseline.json, cv_focal.json, cv_mining.json, cv_curriculum.json, cv_combined.json — all valid JSON with appropriate technique flags set.
result: pass

### 5. Four individual ablation CV results completed
expected: outputs/ contains cross_validation_results.json for each of the 4 individual experiments (baseline, focal, mining, curriculum) with 5-fold metrics.
result: pass

### 6. Curriculum learning is best ablation (F1=0.9932)
expected: Curriculum learning achieves val_f1_macro ~0.9932 (+0.88pp over cleaned baseline 0.9844), clearly outperforming focal and mining.
result: pass

### 7. All experiments exceed VP recall target (>92.9%)
expected: All 5 experiments (baseline, focal, mining, curriculum, combined) achieve Viral Pneumonia recall well above the 92.9% Phase 8 success criterion.
result: pass

### 8. Combined model trained and compared
expected: outputs/classifier_cv_combined/ contains cross_validation_results.json showing F1=0.9878, confirming combined model underperforms curriculum alone.
result: pass

### 9. Ablation comparison script and output
expected: scripts/compare_ablations.py exists and outputs/ablation_comparison.json contains structured comparison of all 5 experiments.
result: pass

## Summary

total: 9
passed: 9
issues: 0
pending: 0
skipped: 0

## Gaps

[none yet]
