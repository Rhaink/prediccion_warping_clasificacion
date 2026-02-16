# Phase 1: Pre-Implementation Audit - Research Findings

**Phase Goal:** Verify test set integrity and establish baseline methodology documentation before implementing ensemble+TTA.

**Research Date:** 2026-01-27
**Status:** Complete - Ready for planning

---

## Executive Summary

This phase is CRITICAL for methodological validity. The project already experienced one major mistake (reporting validation accuracy instead of test accuracy), demonstrating vulnerability to test contamination. A thorough audit must be performed before proceeding with ensemble+TTA implementation.

**Key Finding:** The current system state appears methodologically sound based on initial investigation:
- Test set properly isolated (1,895 images fixed by seed=42)
- CV training performed on train+val only (13,258 images)
- Test evaluation completed separately after training
- Baseline metrics confirmed: 97.68% ± 0.16% on test set

However, formal documentation and verification are still required.

---

## 1. Test Set Integrity Verification

### 1.1 Current Test Set Configuration

**Location:** `outputs/warped_lung_best/session_warping/test/`

**Composition (from dataset_summary.json):**
- Total images: 1,895
- COVID: 452 images
- Normal: 1,274 images
- Viral_Pneumonia: 169 images
- Fill rate: 47.08% (consistent with train/val)

**Split Configuration:**
- Seed: 42 (fixed across all experiments)
- Split methodology: 75% train, 12.5% val, 12.5% test (stratified)
- Source: `src_v2/cli.py` generate-dataset command with seed parameter

### 1.2 Data Leakage Risk Assessment

**Primary Risk:** Images appearing in both train and test sets due to:
1. Duplicate images in source dataset
2. Inconsistent seed usage across experiments
3. Manual file manipulation

**Verification Strategy:**
```bash
# Hash all images to detect duplicates
find outputs/warped_lung_best/session_warping/train -name "*.png" -exec md5sum {} \; > train_hashes.txt
find outputs/warped_lung_best/session_warping/test -name "*.png" -exec md5sum {} \; > test_hashes.txt
comm -12 <(sort train_hashes.txt) <(sort test_hashes.txt)  # Should be empty
```

**Evidence Locations:**
- Dataset generation: `outputs/warped_lung_best/session_warping/dataset_summary.json`
- Split metadata: Check for `images.csv` in test/ directory listing original filenames
- Git history: `git log --all --grep="test\|split" -- outputs/` to trace test set creation

### 1.3 Test Set Usage History

**Files to Audit:**
1. Training logs: `outputs/classifier_cv/fold_*/training_history.json`
2. Config files: `configs/classifier_*.json`
3. Git commits: Search for "test" mentions in commits after 2026-01-15
4. Script execution: Check bash history or nohup logs for test evaluations

**Red Flags to Check:**
- Early stopping based on test metrics (should use validation only)
- Hyperparameter sweeps mentioning test set
- Model selection based on test performance
- Test set in DataLoader during training phase

**Current Evidence (from CV results):**
- `cross_validation_results.json`: Contains only validation metrics (no test)
- `cross_validation_test_results.json`: Created 2026-01-27 (recently, separate evaluation)
- `fold_*/test_results.json`: All dated 2026-01-27 (after training completed)

**Preliminary Assessment:** ✅ Test set appears to have been evaluated AFTER training completion.

---

## 2. Baseline Methodology Documentation

### 2.1 Current Baseline Metrics (Test Set)

**Source:** `outputs/classifier_cv/cross_validation_test_results.json`

**Aggregate Performance:**
```
Accuracy:     97.68% ± 0.16%  (range: 97.52% - 97.94%)
F1-Macro:     96.47% ± 0.27%
F1-Weighted:  97.67% ± 0.16%

Test set size: 1,895 images
Evaluations: 5 folds × 1,895 = 9,475 predictions
```

**Per-Fold Breakdown:**
| Fold | Accuracy | F1-Macro | F1-Weighted |
|------|----------|----------|-------------|
| 1    | 97.52%   | 96.09%   | 97.51%      |
| 2    | 97.78%   | 96.68%   | 97.78%      |
| 3    | 97.52%   | 96.32%   | 97.52%      |
| 4    | 97.63%   | 96.39%   | 97.62%      |
| 5    | 97.94%   | 96.85%   | 97.93%      |

**Best Model:** Fold 5 (97.94% accuracy)

**Variance Analysis:**
- Standard deviation < 0.2% indicates stable training
- Low variance suggests models are not overfitting
- Consistent performance across folds validates methodology

### 2.2 Per-Class Performance (Test Set Aggregate)

**Source:** Aggregated confusion matrix from `cross_validation_test_results.json`

**Aggregated Confusion Matrix (9,475 predictions):**
```
              Predicted
              COVID  Normal  Viral
Actual COVID  2,189    64      7
       Normal   53   6,282    35
       Viral     1     60     784
```

**Per-Class Metrics (Mean ± Std):**
| Class           | Precision      | Recall         | F1-Score       | Support |
|-----------------|----------------|----------------|----------------|---------|
| COVID-19        | 97.59% ± 0.25% | 96.86% ± 0.26% | 97.22% ± 0.16% | 452×5   |
| Normal          | 98.06% ± 0.06% | 98.62% ± 0.21% | 98.34% ± 0.13% | 1,274×5 |
| Viral_Pneumonia | 94.93% ± 1.08% | 92.78% ± 0.69% | 93.84% ± 0.60% | 169×5   |

**Class Imbalance Impact:**
- Viral_Pneumonia has lowest performance (smallest class, 169 samples)
- Normal class most stable (largest class, 1,274 samples)
- COVID-19 intermediate performance (452 samples)

### 2.3 Training Configuration

**Cross-Validation Setup:**
- K-Folds: 5 (stratified)
- Training data: Train + Val combined (13,258 images total)
- Test holdout: Fixed 1,895 images (never used for training/validation)
- Seed: 42 (consistent across all experiments)

**Model Configuration (from configs/classifier_warped_base.json):**
```json
{
  "backbone": "resnet18",
  "epochs": 50,
  "batch_size": 32,
  "lr": 0.0001,
  "patience": 10,
  "use_class_weights": true,
  "seed": 42
}
```

**Training Protocol:**
- Optimizer: Adam (lr=1e-4)
- Loss: CrossEntropyLoss with class weights
- Early stopping: Patience 10 epochs on validation F1-macro
- Data augmentation: Standard (flip, rotation, affine, color jitter)
- Preprocessing: Grayscale + CLAHE (clip=2.0, tile=4)

### 2.4 Data Pipeline Verification

**Warped Dataset Generation:**
- Source: `outputs/landmark_predictions/session_warping/predictions.npz`
- Landmark ensemble: 4 models (seeds 123, 321, 111, 666)
- Landmark error: 3.61 px on 224×224 images
- TTA: Yes (horizontal flip with symmetric pairs correction)
- CLAHE: Yes (clip=2.0, tile=4)
- Margin scale: 1.05 (5% expansion from landmark centroid)

**Dataset Statistics:**
- Total images: 15,153 (from 15,153 original images in COVID-19 Radiography Dataset)
- Train: 11,364 images (75%)
- Val: 1,894 images (12.5%)
- Test: 1,895 images (12.5%)
- Fill rate: ~47% (piecewise affine warping with margin=1.05)

**Critical Note:** All images use predicted landmarks (not ground truth), ensuring deployment realism.

---

## 3. Ensemble Model Selection Methodology

### 3.1 Current Ensemble Configuration

**Landmark Ensemble (Reference):**
- Configuration: `configs/ensemble_best.json`
- Models: 4 models (seeds 123, 321, 111, 666)
- Selection methodology: Grid search over seed combinations
- Performance: 3.61 px error (best among 15 combinations tested)
- Documentation: `GROUND_TRUTH.json` lines 54-65

**Classifier Ensemble (To Be Implemented):**
- Models: 5 CV folds (fold_01 through fold_05)
- Selection methodology: **NOT YET DOCUMENTED**
- Current status: All 5 folds trained, individually evaluated on test set

### 3.2 Missing Documentation

**Questions to Answer:**
1. Why were these 5 specific folds chosen? (Answer: Standard 5-fold CV protocol)
2. Were other K values tested (k=3, k=7, k=10)? (Unknown - needs investigation)
3. Was there model selection based on validation or test performance?
4. What was the seed selection process for CV splits?

**Evidence to Gather:**
- Git history: `git log --all --grep="CV\|cross.validation\|fold"`
- Commit `fb062906`: "feat: complete scientific validation with gpu-accelerated k-fold cv"
- Training scripts: Check for CV implementation in `scripts/` or `src_v2/cli.py`
- Experiment logs: Look for sweep outputs or CV parameter searches

### 3.3 Validation-Based Selection Protocol

**Expected Methodology (Standard Practice):**
1. Train 5 models with k-fold CV on train+val (13,258 images)
2. Each model evaluated on its validation fold during training
3. Best checkpoint selected per fold based on validation F1-macro
4. All 5 models kept for ensemble (no cherry-picking)
5. Final ensemble evaluation on test set (never used during training)

**Verification Steps:**
```bash
# Check training logs for model selection criteria
jq '.best_val_f1' outputs/classifier_cv/fold_*/results.json

# Verify early stopping used validation (not test)
grep -r "early_stop\|patience" outputs/classifier_cv/fold_*/training_history.json

# Confirm test evaluation happened after training
stat outputs/classifier_cv/fold_*/best_classifier.pt
stat outputs/classifier_cv/fold_*/test_results.json
# test_results.json should be dated AFTER best_classifier.pt
```

### 3.4 Retroactive Documentation Plan

**If Evidence Missing:**
Create formal protocol document: `.planning/phases/01-pre-implementation-audit/SEED_SELECTION_PROTOCOL.md`

**Content:**
- CV configuration: k=5, stratified, seed=42
- Model selection: Best validation F1-macro per fold
- No hyperparameter tuning on test set
- All 5 models included in ensemble (no post-hoc selection)
- Justification: Diversity from different data partitions adds complementary information

---

## 4. Implementation Technical Details

### 4.1 File Locations

**Test Set:**
```
outputs/warped_lung_best/session_warping/test/
├── COVID/           (452 images)
├── Normal/          (1,274 images)
├── Viral_Pneumonia/ (169 images)
├── images.csv       (metadata)
└── landmarks.json   (predicted landmarks)
```

**CV Model Checkpoints:**
```
outputs/classifier_cv/fold_01/best_classifier.pt  (44.8 MB)
outputs/classifier_cv/fold_02/best_classifier.pt
outputs/classifier_cv/fold_03/best_classifier.pt
outputs/classifier_cv/fold_04/best_classifier.pt
outputs/classifier_cv/fold_05/best_classifier.pt  (Best: 97.94% test accuracy)
```

**Result Files:**
```
outputs/classifier_cv/cross_validation_results.json       (Validation metrics)
outputs/classifier_cv/cross_validation_test_results.json  (Test metrics aggregate)
outputs/classifier_cv/fold_*/test_results.json            (Per-fold test metrics)
outputs/classifier_cv/fold_*/training_history.json        (Training curves)
outputs/classifier_cv/fold_*/results.json                 (Validation metrics)
```

### 4.2 Code Architecture

**Model Loading:**
```python
# From src_v2/models/classifier.py
from src_v2.models import ImageClassifier

# Load from checkpoint
model = ImageClassifier.load_from_checkpoint(checkpoint_path)
model.eval()
model.to(device)
```

**Dataset Structure:**
```python
# Test set uses ImageFolder structure (torchvision)
from torchvision import datasets, transforms

test_dataset = datasets.ImageFolder(
    root='outputs/warped_lung_best/session_warping/test',
    transform=transforms.Compose([...])
)
# Classes: ['COVID', 'Normal', 'Viral_Pneumonia'] (alphabetical order)
```

**Transforms (Classifier):**
```python
# From src_v2/models/classifier.py::get_classifier_transforms
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],  # ImageNet stats
    std=[0.229, 0.224, 0.225]
)

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
```

### 4.3 Verification Commands

**Check Image Hashing (Data Leakage):**
```bash
cd outputs/warped_lung_best/session_warping
find train -name "*.png" -exec md5sum {} \; | cut -d' ' -f1 | sort > /tmp/train_hashes.txt
find test -name "*.png" -exec md5sum {} \; | cut -d' ' -f1 | sort > /tmp/test_hashes.txt
comm -12 /tmp/train_hashes.txt /tmp/test_hashes.txt  # Should output nothing
wc -l /tmp/train_hashes.txt /tmp/test_hashes.txt    # Should be 11364 and 1895
```

**Verify Test Set Never Modified:**
```bash
# Check git history for test directory
git log --all --follow -- outputs/warped_lung_best/session_warping/test/

# Check file timestamps (should all be from same session)
find outputs/warped_lung_best/session_warping/test -name "*.png" -printf "%T+ %p\n" | sort | head -5
find outputs/warped_lung_best/session_warping/test -name "*.png" -printf "%T+ %p\n" | sort | tail -5
# All timestamps should be identical (batch generation)
```

**Confirm Test Evaluation After Training:**
```bash
# Compare timestamps
for fold in {01..05}; do
    echo "=== Fold $fold ==="
    stat -c '%y %n' outputs/classifier_cv/fold_$fold/best_classifier.pt
    stat -c '%y %n' outputs/classifier_cv/fold_$fold/test_results.json
    echo ""
done
# test_results.json should be created AFTER best_classifier.pt
```

**Audit Training Logs for Test Usage:**
```bash
# Search for "test" in training history (should only be validation)
for fold in {01..05}; do
    echo "=== Fold $fold ==="
    jq 'keys' outputs/classifier_cv/fold_$fold/training_history.json | grep -i test
done
# Should return nothing (only train/val metrics during training)
```

---

## 5. Risk Assessment

### 5.1 Critical Risks (Must Address)

**RISK 1: Data Leakage (Test Contamination)**
- **Severity:** CRITICAL
- **Probability:** Low (preliminary evidence suggests clean split)
- **Mitigation:** Hash-based verification, git history audit
- **Blocker:** Yes - any leakage invalidates all results

**RISK 2: Test Set Used for Model Selection**
- **Severity:** HIGH
- **Probability:** Low (test_results.json dated after training)
- **Mitigation:** Audit logs, verify early stopping used validation
- **Blocker:** Yes - violates scientific methodology

**RISK 3: Inconsistent Seed Usage**
- **Severity:** MEDIUM
- **Probability:** Low (seed=42 appears consistent)
- **Mitigation:** Verify all splits use same seed
- **Blocker:** No - can document retroactively if needed

### 5.2 Moderate Risks (Should Address)

**RISK 4: Missing Seed Selection Documentation**
- **Severity:** MEDIUM
- **Probability:** High (no formal documentation found)
- **Mitigation:** Create retroactive protocol document
- **Blocker:** No - can document methodology retroactively

**RISK 5: Unclear Hyperparameter Selection**
- **Severity:** LOW
- **Probability:** Medium (configs appear to use defaults)
- **Mitigation:** Document that defaults were used
- **Blocker:** No - standard practice to use pretrained defaults

### 5.3 Green Flags (Positive Evidence)

✅ **Test Metrics Dated After Training:** All `test_results.json` files created 2026-01-27, suggesting proper isolation

✅ **Separate CV Results Files:** Distinct files for validation (`cross_validation_results.json`) and test (`cross_validation_test_results.json`)

✅ **Consistent Seed:** seed=42 appears in configs and dataset metadata

✅ **Low Variance:** Std < 0.2% suggests proper methodology (no cherry-picking)

✅ **Fixed Dataset:** `dataset_summary.json` documents exact split sizes

---

## 6. Audit Checklist (For Planning Phase)

### 6.1 Data Integrity Checks

- [ ] **VALID-01:** Verify test set contains exactly 1,895 images (COVID=452, Normal=1,274, Viral_Pneumonia=169)
- [ ] Hash all images to confirm no duplicates between train and test
- [ ] Check file timestamps to verify test set created in single batch
- [ ] Audit git history for any test directory modifications
- [ ] Verify seed=42 used consistently across all experiments

### 6.2 Baseline Verification

- [ ] **VALID-02:** Confirm 97.68% ± 0.16% is test set accuracy (not validation)
- [ ] Re-evaluate all 5 folds on test set to confirm metrics
- [ ] Compare reported metrics with re-evaluation (tolerance: ±0.10%)
- [ ] Verify per-class metrics match aggregated confusion matrix
- [ ] Check F1-macro calculation methodology

### 6.3 Methodology Documentation

- [ ] **VALID-03:** Document test set used only for final evaluation
- [ ] Audit training logs to confirm no early stopping on test metrics
- [ ] Verify hyperparameters not tuned using test set
- [ ] Check for any sweeps or grid searches mentioning test evaluation
- [ ] Document CV fold creation methodology

### 6.4 Seed Selection Protocol

- [ ] Document why k=5 was chosen (vs k=3, k=7, k=10)
- [ ] Verify all 5 models selected based on validation (not test)
- [ ] Create formal protocol document if missing
- [ ] Document ensemble composition: all 5 folds (no cherry-picking)
- [ ] Justify decision: diversity from data partitions

---

## 7. Expected Findings & Recommendations

### 7.1 Best Case Scenario

**All checks pass:**
1. No data leakage detected
2. Test set never used for training decisions
3. Baseline metrics confirmed accurate
4. Clear methodology documented

**Recommendation:** Proceed to Phase 2 immediately with confidence.

### 7.2 Issues Found (Minor)

**Typical issues:**
1. Missing documentation (but methodology correct)
2. Informal seed selection (but no cherry-picking)
3. Unclear hyperparameter choices (but reasonable defaults used)

**Recommendation:** Create retroactive documentation, then proceed to Phase 2.

### 7.3 Issues Found (Major)

**Critical problems:**
1. Data leakage detected
2. Test set used for model selection
3. Inconsistent seeds across experiments
4. Cherry-picking models based on test performance

**Recommendation:** STOP. Fix issues before proceeding. May require:
- Re-generating dataset with proper splits
- Re-training models with isolated test set
- Updating all reported baselines

---

## 8. References & Documentation

### 8.1 Existing Project Documentation

**Methodology:**
- `docs/REPRO_FULL_PIPELINE.md` - Full pipeline documentation
- `docs/PROMPT_CONTINUACION_ENSEMBLE.md` - Ensemble implementation context
- `docs/ESTRATEGIAS_MEJORA_TEST_ACCURACY.md` - Improvement strategies
- `docs/Tesis/README_CV_VERSION.md` - CV version documentation

**Ground Truth:**
- `GROUND_TRUTH.json` - Validated metrics reference
- `outputs/warped_lung_best/session_warping/dataset_summary.json` - Dataset metadata

**Results:**
- `outputs/classifier_cv/cross_validation_test_results.json` - Test metrics aggregate
- `outputs/classifier_cv/fold_*/test_results.json` - Per-fold test results

### 8.2 Similar Project Audits (Reference)

**Landmark Model Selection:**
- Documented in `GROUND_TRUTH.json` lines 54-65
- Ensemble seed selection via grid search (15 combinations)
- Best combination: seeds 123, 321, 111, 666
- Evaluation on test set: 3.61 px error
- Methodology: Validation-based selection, test only for final metric

**Lesson Learned:** Formal documentation prevents future confusion and builds trust.

### 8.3 Scientific Standards

**Best Practices (ML in Medical Imaging):**
1. Fixed test holdout (never used for training/validation)
2. Cross-validation on train+val only
3. Model selection based on validation metrics
4. Test evaluation performed once at the end
5. Report test metrics with confidence intervals
6. Document all data splits and seeds

**Common Pitfalls (Research Flagged 5 Critical Issues):**
1. Test contamination (data leakage)
2. Multiple evaluations on test (p-hacking)
3. Model selection based on test performance
4. Unsafe augmentations changing semantics
5. Ensemble cherry-picking (reporting best subset)

**Project's History:** Already fell into pitfall #2 (reported validation as test accuracy), demonstrating need for rigorous audit.

---

## 9. Tools & Scripts Required

### 9.1 Audit Utilities

**Data Leakage Checker:**
```bash
#!/bin/bash
# check_data_leakage.sh
echo "Checking for duplicate images between train and test..."
find outputs/warped_lung_best/session_warping/train -name "*.png" -exec md5sum {} \; | cut -d' ' -f1 | sort > /tmp/train.txt
find outputs/warped_lung_best/session_warping/test -name "*.png" -exec md5sum {} \; | cut -d' ' -f1 | sort > /tmp/test.txt
duplicates=$(comm -12 /tmp/train.txt /tmp/test.txt | wc -l)
if [ $duplicates -eq 0 ]; then
    echo "✓ No data leakage detected"
else
    echo "✗ CRITICAL: Found $duplicates duplicate images!"
fi
```

**Baseline Re-evaluator:**
```python
#!/usr/bin/env python3
# re_evaluate_cv_folds.py
"""Re-evaluate all 5 CV folds on test set to verify reported metrics."""
import json
from pathlib import Path
import torch
from src_v2.models import ImageClassifier
from src_v2.evaluation.metrics import evaluate_classifier

cv_dir = Path("outputs/classifier_cv")
test_dir = Path("outputs/warped_lung_best/session_warping/test")

for fold in range(1, 6):
    checkpoint = cv_dir / f"fold_{fold:02d}" / "best_classifier.pt"
    model = ImageClassifier.load_from_checkpoint(checkpoint)
    metrics = evaluate_classifier(model, test_dir)

    # Compare with reported metrics
    reported = json.load(open(cv_dir / f"fold_{fold:02d}" / "test_results.json"))
    accuracy_diff = abs(metrics['accuracy'] - reported['metrics']['accuracy'])

    print(f"Fold {fold}: accuracy={metrics['accuracy']:.4f} (diff={accuracy_diff:.4f})")
    assert accuracy_diff < 0.001, f"Fold {fold} metrics mismatch!"
```

**Git History Auditor:**
```bash
#!/bin/bash
# audit_git_history.sh
echo "Auditing git history for test set usage..."
git log --all --grep="test" --oneline | head -20
git log --all --follow -- outputs/warped_lung_best/session_warping/test/
git log --all --grep="CV\|fold\|cross.validation" --oneline | head -20
```

### 9.2 Documentation Generators

**Seed Selection Protocol Generator:**
```markdown
# Template: SEED_SELECTION_PROTOCOL.md

## Cross-Validation Configuration
- K-Folds: 5 (stratified by class)
- Train+Val: 13,258 images
- Test holdout: 1,895 images (fixed, never used for training)
- Seed: 42 (consistent across all experiments)

## Model Selection Criteria
- Per-fold: Best validation F1-macro (patience=10)
- Ensemble: All 5 folds included (no cherry-picking)
- No hyperparameter tuning on test set

## Validation Protocol
1. Train 5 models with stratified k-fold CV
2. Each model saves best checkpoint based on validation F1-macro
3. After all training completes, evaluate each model on test set
4. Report mean ± std across 5 folds
5. Use all 5 models for ensemble (no post-hoc selection)

## Rationale
- Diversity: Different data partitions capture complementary patterns
- Stability: Low variance (σ < 0.2%) indicates robust methodology
- Standard: k=5 is industry standard for medical imaging
```

---

## 10. Success Criteria (Planning Phase Inputs)

### 10.1 All Checks Must Pass

1. ✅ Test set contains exactly 1,895 images with correct distribution
2. ✅ No data leakage detected (zero duplicate hashes)
3. ✅ Baseline 97.68% ± 0.16% confirmed on test set (tolerance: ±0.10%)
4. ✅ Test set never used for model selection or hyperparameter tuning
5. ✅ Seed selection methodology documented (validation-based)

### 10.2 Deliverables

**Required Outputs:**
1. `AUDIT_REPORT.md` - Comprehensive audit results with all verifications
2. `SEED_SELECTION_PROTOCOL.md` - Formal model selection documentation (if missing)
3. `BASELINE_VERIFICATION.json` - Re-evaluation results confirming metrics
4. `DATA_INTEGRITY_CHECK.txt` - Hash-based leakage verification results

**Format:** Technical reference with detailed commands and reproducible steps.

### 10.3 Go/No-Go Decision

**Proceed to Phase 2 if:**
- All integrity checks pass
- Baseline metrics verified
- Methodology documented
- No critical issues found

**STOP and Fix if:**
- Data leakage detected
- Test contamination confirmed
- Metrics cannot be reproduced
- Methodology unclear or invalid

---

## 11. Notes for Planning Agent

### 11.1 Estimated Effort

**Audit Execution Time:**
- Data integrity checks: 30 minutes
- Baseline re-evaluation: 1-2 hours (GPU inference)
- Git history audit: 30 minutes
- Documentation creation: 1-2 hours
- **Total: 3-5 hours**

**Complexity:** LOW-MEDIUM
- Mostly verification and documentation
- No new model training required
- Clear pass/fail criteria

### 11.2 Prerequisites

**Required:**
- Access to `outputs/classifier_cv/` directory with all checkpoints
- Access to `outputs/warped_lung_best/session_warping/test/` dataset
- GPU for re-evaluation (can use CPU but slower)
- Git repository access for history audit

**Optional:**
- Historical logs or bash history for additional context
- Author interviews for methodology clarification

### 11.3 Risk Mitigation

**High-Risk Items:**
1. Re-evaluation may find metric discrepancies → Document variance tolerance (±0.10%)
2. Missing documentation → Create retroactive protocol (acceptable if methodology correct)
3. Unclear seed selection → Assume standard CV practice, document retroactively

**Low-Risk Items:**
- Data leakage (preliminary evidence clean)
- Test contamination (separate files suggest proper isolation)

### 11.4 Dependencies

**Blocks Phase 2:** Yes - must complete audit before implementing ensemble+TTA

**Blocks Phase 3:** Yes (transitive via Phase 2)

**Critical Path:** This is the first phase - sets foundation for entire project

---

## 12. Recommendations

### 12.1 Immediate Actions

1. **Execute Data Integrity Checks** (HIGH PRIORITY)
   - Run hash-based leakage verification
   - Verify image counts match documented split
   - Check file timestamps for consistency

2. **Re-evaluate Baseline** (HIGH PRIORITY)
   - Load all 5 fold checkpoints
   - Re-run evaluation on test set
   - Confirm metrics within tolerance (±0.10%)

3. **Audit Git History** (MEDIUM PRIORITY)
   - Search for test set modifications
   - Verify training happened before test evaluation
   - Document timeline of experiments

4. **Create Missing Documentation** (MEDIUM PRIORITY)
   - Formalize seed selection protocol
   - Document CV configuration
   - Explain ensemble composition rationale

### 12.2 Future Improvements

**For Thesis Defense:**
- Include audit report in appendix (demonstrates rigor)
- Reference data integrity checks in methodology section
- Highlight proactive verification as project strength

**For Future Projects:**
- Automate audit checks in CI/CD pipeline
- Create data leakage detection scripts
- Require formal protocols before training

### 12.3 Context for Ensemble Implementation

**Why This Audit Matters:**
- Ensemble will amplify existing issues (garbage in, garbage out)
- TTA will compound any test contamination
- Baseline must be rock-solid before claiming improvement
- Thesis defense will scrutinize methodology rigorously

**Expected Outcome:**
- Clean audit → Proceed with confidence
- Minor issues → Document and fix, then proceed
- Major issues → Pause, retrain if necessary

---

## Appendix A: File Structure Reference

```
outputs/classifier_cv/
├── fold_01/
│   ├── best_classifier.pt        (44.8 MB, ResNet-18 weights)
│   ├── test_results.json         (Test metrics, created 2026-01-27)
│   ├── results.json              (Validation metrics)
│   └── training_history.json     (Loss curves, epoch-by-epoch)
├── fold_02/ ... fold_05/         (Same structure)
├── cross_validation_results.json       (Validation aggregate)
└── cross_validation_test_results.json  (Test aggregate)

outputs/warped_lung_best/session_warping/
├── dataset_summary.json          (Split metadata, seed=42)
├── train/                        (11,364 images)
│   ├── COVID/
│   ├── Normal/
│   └── Viral_Pneumonia/
├── val/                          (1,894 images)
└── test/                         (1,895 images, AUDIT TARGET)
    ├── COVID/ (452)
    ├── Normal/ (1,274)
    ├── Viral_Pneumonia/ (169)
    ├── images.csv
    └── landmarks.json

configs/
├── classifier_warped_base.json   (Training config)
└── ensemble_best.json            (Landmark ensemble reference)

GROUND_TRUTH.json                 (Validated metrics source of truth)
```

---

## Appendix B: Validation vs Test Clarification

**Recent Project History:**
- Initially reported: 98.60% ± 0.26% (VALIDATION metrics)
- Corrected to: 97.68% ± 0.16% (TEST metrics)
- Difference: +0.92 points (validation overestimates)

**Why Validation is Higher:**
1. Validation set used for early stopping (slight overfitting)
2. Smaller validation set (1,894) vs test (1,895) - statistical variance
3. Random variation in class distribution across splits

**Current Status:**
- All project docs updated to use test metrics
- Figures regenerated with correct test results
- LaTeX chapter corrected (2026-01-27)

**Audit Importance:** Ensure no confusion between validation and test in future work.

---

**END OF RESEARCH FINDINGS**

**Status:** Ready for planning phase
**Next Step:** Create detailed audit plan with specific verification steps, commands, and success criteria
**Estimated Time to Execute Plan:** 3-5 hours
**Blocker for Phase 2:** Yes - must pass audit before implementing ensemble+TTA
