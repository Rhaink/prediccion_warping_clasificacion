# Pitfalls: Data-Centric Improvements at 98%+ Accuracy

**Researched:** 2026-02-16
**Context:** 98.26% accuracy, 33 errors, 1895 test images, thesis project

## P1: Statistical Significance of Small Improvements

**Risk:** At 98.26% (33 errors), each corrected sample = +0.05pp. Improvements of 0.2-0.5pp may not be statistically significant on 1895 samples.

**Impact:** HIGH — thesis claims must be defensible.

**Prevention:**
- Use McNemar's test to compare baseline vs improved (paired test on same samples)
- Report confidence intervals, not just point estimates
- Consider that fixing 5 samples = +0.26pp may not reach p<0.05
- Document effect size alongside statistical significance

**Phase:** Final evaluation

---

## P2: Test Set Contamination During Data Cleaning

**Risk:** Label noise detection or error forensics may use information from test set, violating methodological integrity.

**Impact:** CRITICAL — invalidates all results.

**Prevention:**
- Label noise detection uses ONLY train/validation predictions (from CV)
- Error forensics on test set is post-hoc analysis only — never feeds back to training
- Implement `verify_test_set_isolation()` check
- Document data flow explicitly

**Phase:** All phases — verify throughout

---

## P3: Label Noise Detection False Positives

**Risk:** Cleanlab may flag hard-but-correctly-labeled samples as "noisy." Removing genuine hard examples reduces model's ability to handle ambiguous cases.

**Impact:** MEDIUM-HIGH — could degrade VP recall further.

**Prevention:**
- NEVER auto-remove flagged samples. Always manual review
- Start with `action: "report_only"`, not `action: "remove"`
- Cross-reference with error forensics (is the sample also misclassified?)
- Set conservative confidence threshold (flag top 2-3%, not 10%)
- Test with and without removal (ablation study)

**Phase:** Label noise detection

---

## P4: Augmentation Destroying Diagnostic Features

**Risk:** Aggressive augmentations can create non-physiological chest X-rays that confuse rather than help.

**Impact:** MEDIUM — could reduce accuracy.

**Specific pitfalls:**
- **Horizontal flip on warped images**: Heart appears on wrong side. Current pipeline uses flip — REVIEW whether this is appropriate post-warping
- **Rotation >15°**: Non-physiological for chest X-rays. Clinical range is ±5°
- **Elastic deformations too aggressive**: Can distort lung boundaries beyond physiological range
- **Color jitter on medical images**: X-ray intensity has diagnostic meaning

**Prevention:**
- Test each augmentation individually (ablation)
- Compare train accuracy (should decrease with augmentation) vs val accuracy (should increase)
- Visual inspection of augmented samples
- MixUp alpha=0.2 (conservative for medical)

**Phase:** Advanced augmentation

---

## P5: Class Imbalance Overcorrection

**Risk:** Aggressive oversampling or high focal loss gamma can bias model toward minority class, degrading majority class accuracy. Net accuracy could decrease.

**Impact:** MEDIUM — VP has only 169 test samples.

**Prevention:**
- Start with moderate focal loss gamma=2.0, not extreme values
- Monitor per-class metrics, not just overall accuracy
- Set acceptable tradeoff: VP recall must increase without >0.5pp Normal accuracy drop
- Use validation set to tune, never test set

**Phase:** Focal loss + sampling

---

## P6: Overfitting to Validation Set via Iterative Cleaning

**Risk:** Multiple iterations of error forensics → label cleaning → retraining can implicitly overfit to the validation set.

**Impact:** MEDIUM — inflated validation metrics, disappointing test results.

**Prevention:**
- Limit to 2-3 cleaning cycles maximum
- Track validation performance trend (diminishing returns = stop)
- Final evaluation only once on test set
- Document each iteration's decisions

**Phase:** Integration + iteration

---

## P7: COVID-19 Radiography Dataset Known Issues

**Risk:** Public dataset has documented quality issues that may explain some of the 33 errors.

**Known issues from literature:**
- Labels derived from NLP processing of radiology reports (not expert annotation)
- Possible mislabeling between VP and Normal (exactly our confusion pattern)
- 1 known duplicate in test set (Normal-817/Normal-818, already documented)
- Images from heterogeneous sources with different equipment/protocols

**Prevention:**
- Research this dataset specifically during error forensics
- Compare error patterns with published literature
- Document dataset limitations in thesis methodology section

**Phase:** Error forensics + documentation

---

## P8: Reproducibility of Data-Dependent Pipeline

**Risk:** Data cleaning decisions are subjective. Different runs of label noise detection may flag different samples. Results become non-reproducible.

**Impact:** MEDIUM — thesis requires reproducibility.

**Prevention:**
- Document every sample excluded/corrected with reasoning
- Save cleanlab thresholds and flagged samples as JSON artifacts
- Use deterministic seeds throughout
- Provide `data_cleaning_manifest.json` as reproducibility artifact
- Run dual verification (same as v1.0 approach)

**Phase:** All phases

---

## P9: Destroying What Already Works

**Risk:** In pursuit of fixing 33 errors, changes could break the 1862 correct classifications.

**Impact:** HIGH — regression is worse than no improvement.

**Prevention:**
- Always compare against v1.0 baseline (same test set, same metrics)
- Track case-level changes: how many helped vs hurt vs neutral
- Set regression threshold: abort if >5 new errors introduced
- Preserve v1.0 checkpoints as baseline (never overwrite)

**Phase:** Every evaluation

---

## Summary: Pitfall Risk Matrix

| # | Pitfall | Likelihood | Impact | Phase to Address |
|---|---------|-----------|--------|-----------------|
| P1 | Statistical significance | HIGH | HIGH | Final evaluation |
| P2 | Test set contamination | LOW | CRITICAL | All phases |
| P3 | Label noise false positives | MEDIUM | MEDIUM-HIGH | Label cleaning |
| P4 | Augmentation destroying features | MEDIUM | MEDIUM | Augmentation |
| P5 | Class imbalance overcorrection | MEDIUM | MEDIUM | Focal loss |
| P6 | Validation overfitting | LOW-MED | MEDIUM | Integration |
| P7 | Dataset known issues | HIGH | LOW-MED | Forensics |
| P8 | Reproducibility | MEDIUM | MEDIUM | All phases |
| P9 | Breaking what works | LOW-MED | HIGH | Every evaluation |
