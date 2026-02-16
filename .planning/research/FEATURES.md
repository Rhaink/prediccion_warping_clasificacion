# Feature Landscape: Data-Centric Accuracy Improvements

**Domain:** Medical image classification (COVID-19 chest X-ray)
**Researched:** 2026-02-16
**Context:** Ensemble at 98.26%, 33 errors, VP recall 92.9% (worst)

## Table Stakes

| Feature | Complexity | Notes |
|---------|------------|-------|
| **Error forensics on 33 misclassified images** | Low | Visual inspection, pattern identification, categorize errors |
| **Basic data quality checks** | Low | Duplicates, corrupt images, statistical profiling |
| **Augmentation policy tuning** | Low-Med | Current is basic (flip, ±15° rotation, jitter). Room for improvement |
| **Class imbalance handling** | Medium | Already using weighted CE. Focal loss + sampling could improve VP recall |

## Differentiators

| Feature | Value | Complexity | Expected Impact |
|---------|-------|------------|-----------------|
| **Label noise detection (cleanlab)** | COVID-19 public datasets known for label errors | Medium | 0.5-1.5% if noise present |
| **Focal loss** | Emphasize hard examples + minority classes | Low-Med | 0.3-0.5% + VP recall +3-5% |
| **Hard example mining** | Oversample frequently misclassified samples | Medium | VP recall +3-8% |
| **Curriculum learning** | Easy→hard training schedule | Medium | 0.3-0.5% on hard classes |
| **Medical-specific augmentation** | Elastic deforms, GridDistortion (anatomically valid) | Med-High | 0.2-0.5% |
| **Confidence calibration** | Temperature scaling for reliable uncertainty | Low | Better error identification |
| **Outlier/anomaly detection** | UMAP + isolation forest on feature embeddings | Medium | Identify 1-5% dataset issues |

## Anti-Features (DO NOT build)

| Anti-Feature | Why Avoid |
|--------------|-----------|
| SMOTE-based oversampling | Creates unrealistic X-ray interpolations |
| Aggressive geometric aug (>20° rotation) | Destroys anatomical realism |
| AutoAugment search | Computationally expensive, non-generalizable |
| Full dataset re-annotation | 21K images, expensive. Use automated detection first |
| Architecture changes | ResNet-18 fixed — isolate data effect |
| GAN training from scratch | Complex, unnecessary at this stage |

## Error Pattern Analysis

**33 errors breakdown:**
- 12 VP→Normal (36%) — VP recall 92.9%, worst class
- 10 COVID→Normal (30%)
- 6 Normal→VP (18%)
- 4 Normal→COVID (12%)
- 1 COVID→VP (3%)

**VP→Normal root causes:**
1. Class imbalance (VP=169 vs Normal=1274, 7.5x)
2. Visual similarity (diffuse interstitial pattern)
3. Possible label noise (public dataset, NLP-processed labels)
4. Insufficient VP representation during training

**Targeted solutions:** Focal loss + hard mining on VP → most impactful per-effort

## Estimated Cumulative Impact

| Phase | Techniques | Expected Gain | Cumulative |
|-------|-----------|---------------|------------|
| Baseline | Current ensemble + TTA | 98.26% | 98.26% |
| Phase 1 | Error forensics + focal loss | +0.3-0.5% | 98.56-98.76% |
| Phase 2 | Label cleaning | +0.5-1.5% | 99.06-99.26% |
| Phase 3 | Advanced aug + hard mining | +0.2-0.5% | 99.26-99.76% |

**Notes:** Gains not strictly additive (diminishing returns). Label cleaning highest variance. Realistic target: 99.0-99.5% accuracy, VP recall >95%.

## Feature Dependencies

```
Error Forensics → Hard Example Mining
Error Forensics → Curriculum Learning
Data Quality Checks → Label Noise Detection
Data Quality Checks → Outlier Detection
Focal Loss → Hard Example Mining (combined effect)
5-fold CV → Label Noise Detection (provides out-of-sample predictions)
```
