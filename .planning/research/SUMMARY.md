# Project Research Summary

**Project:** Data-Centric Improvements for COVID-19 Chest X-ray Classification
**Domain:** Medical image classification (deep learning research)
**Researched:** 2026-02-16
**Confidence:** HIGH

## Executive Summary

This project targets the final 1.74% accuracy gap in a COVID-19 chest X-ray classifier that has reached 98.26% accuracy with 33 remaining errors. Research shows this is a classic data-centric AI problem where the model architecture (ResNet-18 ensemble with geometric normalization) is already optimal, but data quality, labeling accuracy, and class imbalance handling have room for improvement. Expert consensus indicates that at 98%+ accuracy, further gains come exclusively from data improvements, not architectural changes.

The recommended approach follows a systematic data-centric methodology: (1) error forensics to understand failure modes, (2) automated label noise detection using confident learning (cleanlab), (3) advanced augmentation strategies that preserve anatomical validity, and (4) focal loss with hard example mining to address the critical Viral Pneumonia recall issue (92.9%, worst class). This targets the root causes: 36% of errors are VP misclassifications, likely driven by 7.5x class imbalance and potential label noise in the public dataset.

The key risk is statistical significance: at 1895 test samples with 33 errors, each corrected sample is worth only +0.05pp, making claimed improvements difficult to defend without proper statistical testing. Secondary risks include test set contamination during data cleaning (methodologically catastrophic), and breaking the 1862 correct predictions while fixing 33 errors. Mitigation requires strict train/test isolation, McNemar's paired statistical testing, and regression guardrails that abort if >5 new errors are introduced.

## Key Findings

### Recommended Stack

Four new libraries extend the existing PyTorch pipeline without architectural changes. cleanlab (>=2.7.1) provides industry-standard label noise detection via confident learning, requiring only soft predictions from 5-fold cross-validation. albumentations (>=2.0.8) replaces torchvision transforms with medical-grade augmentation including ElasticTransform and GridDistortion that preserve anatomical validity. pyiqa (>=0.1.13) enables no-reference image quality assessment (BRISQUE, NIQE) to identify low-quality samples. torchsampler (>=0.1.2) provides automatic minority class oversampling. Focal loss will be implemented directly in PyTorch without external dependencies.

**Core technologies:**
- **cleanlab** (>=2.7.1): Label noise detection via confident learning — model-agnostic, provable guarantees, works seamlessly with PyTorch via sklearn wrappers
- **albumentations** (>=2.0.8): Medical-grade augmentation pipeline — 100+ transforms, medical-specific operations (ElasticTransform, GridDistortion), faster than torchvision
- **pyiqa** (>=0.1.13): No-reference image quality metrics — 38+ metrics (BRISQUE, NIQE), GPU-accelerated, PyTorch-native integration
- **Focal loss** (custom): Address class imbalance and hard examples — simple PyTorch implementation, no dependency bloat

**Rejected alternatives:** SMOTE creates unrealistic X-ray interpolations, MONAI adds complexity without clear benefit, AlbumentationsX has incompatible AGPL-3.0 licensing for academic use.

### Expected Features

Error analysis reveals a clear pattern: 12 of 33 errors (36%) are Viral Pneumonia misclassified as Normal, driven by 7.5x class imbalance (VP=169 vs Normal=1274 test samples). The COVID-19 Radiography Dataset has documented label quality issues (NLP-derived labels, not expert annotation), making label noise detection high-value. Current augmentation is basic (flip, ±15° rotation, color jitter) with room for medical-specific improvements.

**Must have (table stakes):**
- Error forensics on 33 misclassified images — visual inspection, pattern identification, root cause categorization
- Basic data quality checks — duplicates (1 known pair), corrupt images, statistical profiling
- Augmentation policy tuning — current is basic, medical literature shows elastic deformations improve generalization
- Class imbalance handling — focal loss + sampling to improve VP recall (92.9%, worst class)

**Should have (competitive):**
- Label noise detection (cleanlab) — COVID-19 public datasets have known labeling errors, expected +0.5-1.5% if noise present
- Hard example mining — oversample frequently misclassified samples, expected VP recall +3-8%
- Medical-specific augmentation — ElasticTransform, GridDistortion with anatomical constraints, expected +0.2-0.5%
- Confidence calibration — temperature scaling for better error identification and uncertainty quantification

**Defer (v2+):**
- AutoAugment search — computationally expensive, non-generalizable
- Full dataset re-annotation — 21K images, expensive, use automated detection first
- Architecture changes — ResNet-18 is fixed to isolate data-centric effects
- GAN-based augmentation — complex, unnecessary at this stage

**Estimated cumulative impact:** Baseline 98.26% → Phase 1 (error forensics + focal loss) 98.56-98.76% → Phase 2 (label cleaning) 99.06-99.26% → Phase 3 (advanced aug + hard mining) 99.26-99.76%. Realistic target: 99.0-99.5% accuracy, VP recall >95%.

### Architecture Approach

The existing pipeline architecture requires minimal structural changes. New components integrate at four strategic points: (1) data quality checks before warping to avoid wasting computation on invalid samples, (2) advanced augmentation during training via batch-level operations (MixUp/CutMix), (3) label noise detection after 5-fold CV using out-of-sample predictions, and (4) error forensics after evaluation to drive iterative refinement. The pipeline remains config-driven to avoid CLI flag proliferation and ensure reproducibility.

**Major components:**
1. **Data cleaning module** (`src_v2/data/quality_checks.py`) — landmark quality filtering (outliers >3σ, degenerate triangles), near-duplicate detection via perceptual hashing, integrated before warping step
2. **Label cleaning module** (`src_v2/data/label_cleaning.py`) — cleanlab integration using 5-fold CV predictions, outputs flagged samples for manual review, never auto-removes
3. **Advanced augmentation** (`src_v2/data/batch_augmentations.py`) — MixUp/CutMix at batch level in training loop, requires mixed loss implementation in classifier trainer
4. **Error forensics** (`src_v2/evaluation/error_forensics.py`) — misclassification analysis with confidence + margin, categorizes high-confidence errors (label noise) vs low-margin errors (augmentation needed), generates visualization grids

**Pipeline flow (11 steps):** Raw dataset → duplicate detection → landmark prediction (cached NPZ) → landmark quality check (NEW) → warping → dataset splits → training + augmentation → 5-fold CV + save predictions (NEW) → label noise detection (NEW) → ensemble eval + TTA → error forensics (NEW) → iterative refinement.

### Critical Pitfalls

1. **Statistical significance at 98%+ accuracy (P1, HIGH impact)** — Each corrected sample = +0.05pp. Claimed improvements of 0.2-0.5pp may not reach p<0.05 on 1895 test samples. Mitigation: use McNemar's paired test, report confidence intervals, document effect sizes alongside significance, consider that fixing 5 samples may not be statistically defensible.

2. **Test set contamination during data cleaning (P2, CRITICAL impact)** — Label noise detection or error forensics using test set information invalidates all results. Mitigation: label noise uses ONLY train/val predictions from CV, error forensics on test set is post-hoc analysis only (never feeds back to training), implement `verify_test_set_isolation()` check, document data flow explicitly.

3. **Label noise false positives (P3, MEDIUM-HIGH impact)** — cleanlab may flag hard-but-correctly-labeled samples. Removing genuine hard examples degrades model's ability to handle ambiguous cases, potentially worsening VP recall. Mitigation: NEVER auto-remove, start with `action: "report_only"`, manual review all flagged samples, set conservative threshold (flag top 2-3%), run ablation study with/without removal.

4. **Augmentation destroying diagnostic features (P4, MEDIUM impact)** — Aggressive augmentations create non-physiological X-rays. Specific risks: horizontal flip on warped images (heart on wrong side), rotation >15° (non-physiological for chest X-rays), excessive elastic deformation. Mitigation: test each augmentation individually (ablation), visual inspection, conservative MixUp alpha=0.2, compare train accuracy (should decrease) vs val accuracy (should increase).

5. **Breaking what already works (P9, HIGH impact)** — In pursuit of fixing 33 errors, changes could break 1862 correct classifications. Mitigation: always compare against v1.0 baseline, track case-level changes (helped vs hurt vs neutral), abort if >5 new errors introduced, preserve v1.0 checkpoints.

## Implications for Roadmap

Based on research, suggested phase structure follows a data-centric methodology with strict isolation between phases to prevent test set contamination and enable statistical validation.

### Phase 1: Error Forensics & Data Quality Audit
**Rationale:** Must understand failure modes before attempting fixes. This phase provides the diagnostic foundation for all subsequent improvements and can be done with zero risk of test set contamination if properly scoped.
**Delivers:** Misclassification analysis report, data quality report (duplicates, corrupt images, outliers), categorized error patterns (label noise vs hard examples vs augmentation gaps)
**Addresses:** Error forensics (table stakes), basic data quality checks (table stakes)
**Avoids:** P2 (test set contamination) by keeping analysis strictly post-hoc, P7 (dataset known issues) by researching COVID-19 Radiography Dataset literature
**Duration:** 2-3 days

### Phase 2: Data Cleaning Pipeline
**Rationale:** Must clean data before attempting model improvements. Landmark quality filtering prevents wasting computation on invalid samples. Duplicate removal and outlier detection are low-risk, high-value interventions.
**Delivers:** `quality_checks.py` module, cleaned dataset manifest, pre-warping quality gates
**Addresses:** Basic data quality checks (table stakes)
**Avoids:** P2 (test set contamination) by applying cleaning before train/test split
**Uses:** pyiqa for image quality assessment
**Duration:** 3-4 days

### Phase 3: Focal Loss & Class Imbalance
**Rationale:** VP recall (92.9%) is the clear weakness. Focal loss + hard example mining directly targets the 12 VP→Normal errors (36% of total). This is a surgical intervention with clear success metrics.
**Delivers:** Focal loss implementation, ImbalancedDatasetSampler integration, per-class metric tracking, ablation study results
**Addresses:** Class imbalance handling (table stakes), focal loss (differentiator), hard example mining (differentiator)
**Avoids:** P5 (class imbalance overcorrection) by starting with moderate gamma=2.0 and monitoring per-class metrics, P9 (breaking what works) via regression guardrails
**Uses:** torchsampler
**Duration:** 3-4 days

### Phase 4: Label Noise Detection
**Rationale:** COVID-19 Radiography Dataset has documented label quality issues. This is the highest-value intervention (+0.5-1.5% expected) but also highest risk due to false positives. Requires 5-fold CV infrastructure.
**Delivers:** `label_cleaning.py` module, 5-fold CV training script modifications, cleanlab flagged samples report, manual review documentation
**Addresses:** Label noise detection (differentiator)
**Avoids:** P2 (test set contamination) by using only train/val predictions, P3 (false positives) by requiring manual review and conservative thresholds, P8 (reproducibility) by documenting all decisions
**Uses:** cleanlab
**Implements:** Label cleaning module from ARCHITECTURE.md
**Duration:** 4-5 days

### Phase 5: Advanced Augmentation
**Rationale:** After data cleaning and class imbalance fixes, augmentation can fill remaining gaps. Medical-specific transforms (ElasticTransform, GridDistortion) preserve anatomical validity while improving generalization.
**Delivers:** albumentations integration, MixUp/CutMix batch augmentations, augmentation ablation study, visual validation of augmented samples
**Addresses:** Augmentation policy tuning (table stakes), medical-specific augmentation (differentiator)
**Avoids:** P4 (destroying diagnostic features) via conservative parameters, anatomical constraints, and visual inspection
**Uses:** albumentations
**Implements:** Batch augmentation module from ARCHITECTURE.md
**Duration:** 3-4 days

### Phase 6: Final Evaluation & Documentation
**Rationale:** Thesis requires rigorous statistical validation. Cannot claim improvement without proper statistical testing and comprehensive documentation.
**Delivers:** McNemar's test results, confidence intervals, effect size analysis, regression analysis (case-level changes), data cleaning manifest, reproducibility artifacts, thesis methodology section
**Addresses:** All features integrated, final validation
**Avoids:** P1 (statistical significance) via proper testing, P6 (validation overfitting) by limiting cleaning cycles to 2-3 maximum, P8 (reproducibility) via comprehensive documentation
**Duration:** 2-3 days

### Phase Ordering Rationale

- **Forensics before fixes:** Cannot optimize what you don't understand. Error analysis drives all subsequent decisions.
- **Cleaning before training:** Prevents wasted computation on invalid data. Landmark quality check integrated before warping step per ARCHITECTURE.md.
- **Imbalance before augmentation:** VP recall is the clear bottleneck. Focal loss is lower-risk than aggressive augmentation.
- **Label noise middle-phase:** Requires 5-fold CV infrastructure (time-consuming) but provides highest expected gain. Must come after basic cleaning to avoid flagging samples that are actually corrupt/low-quality.
- **Augmentation late-phase:** Fills gaps after data quality and class distribution are addressed. Requires careful validation to avoid destroying diagnostic features.
- **Evaluation final:** Statistical testing only makes sense after all improvements are integrated. Prevents validation set overfitting from iterative tuning.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 4 (Label Noise Detection):** Complex integration with CV infrastructure, cleanlab API specifics, threshold tuning strategies, false positive management
- **Phase 5 (Advanced Augmentation):** Medical imaging domain knowledge (what transforms are anatomically valid?), albumentations API for medical use cases, MixUp/CutMix loss implementation

Phases with standard patterns (skip research-phase):
- **Phase 1 (Error Forensics):** Standard evaluation metrics, well-documented visualization techniques
- **Phase 2 (Data Cleaning):** Basic data quality checks, duplicate detection, outlier analysis
- **Phase 3 (Focal Loss):** Focal loss is well-documented, PyTorch implementation straightforward, torchsampler has simple API
- **Phase 6 (Final Evaluation):** McNemar's test is standard statistical method, documentation is process-driven

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | cleanlab, albumentations, pyiqa are industry standards with excellent documentation. Focal loss implementation is straightforward PyTorch. All libraries actively maintained. |
| Features | HIGH | Error analysis is concrete (33 specific images). VP recall issue is well-characterized. Expected gains are based on published medical imaging literature and data-centric AI research. |
| Architecture | HIGH | Integration points are clearly defined. Modifications are minimal and surgical. Pipeline flow is well-understood (current v1.0 is 98.26% validated). Config-driven approach proven. |
| Pitfalls | HIGH | Statistical significance issues are well-documented in ML literature. Test set contamination is standard methodology concern. Dataset-specific issues (COVID-19 Radiography) are published. |

**Overall confidence:** HIGH

Research is grounded in concrete baseline (98.26%, 33 errors, 1895 test samples), validated components (cleanlab, albumentations), and published medical imaging best practices. The data-centric approach is appropriate for the 98%+ accuracy regime where architectural improvements show diminishing returns.

### Gaps to Address

- **Augmentation anatomical validity:** During Phase 5 planning, need to research medical imaging literature for acceptable transform parameters (rotation range, elastic deformation limits). Current research identifies the risk (P4) but doesn't provide specific parameter ranges.
  - Handle during Phase 5: Literature review + radiologist consultation if available, start conservative and ablate.

- **cleanlab threshold calibration:** Research identifies the need for conservative thresholds (flag top 2-3%) but doesn't provide dataset-specific calibration strategy.
  - Handle during Phase 4: Start with cleanlab defaults, use validation set to tune, cross-reference with error forensics, require manual review.

- **Statistical power analysis:** Research identifies significance issues (P1) but doesn't calculate minimum detectable effect size for 1895 samples.
  - Handle during Phase 6: Pre-compute power analysis before claiming improvements, consider stratified analysis (per-class) if overall test lacks power.

- **Horizontal flip on warped images:** Research flags this (P4) but doesn't definitively resolve whether flip is appropriate post-warping given heart laterality.
  - Handle during Phase 5: Test flip ablation on validation set, visual inspection of flipped warped images, consider removing flip if it degrades accuracy.

## Sources

### Primary (HIGH confidence)
- cleanlab documentation (https://docs.cleanlab.ai/) — label noise detection methodology, API usage, confident learning theory
- albumentations documentation (https://albumentations.ai/docs/) — medical imaging transforms, augmentation strategies, anatomical constraints
- pyiqa documentation (https://github.com/chaofengc/IQA-PyTorch) — no-reference quality metrics, BRISQUE/NIQE usage for medical images
- COVID-19 Radiography Dataset paper — documented label quality issues, NLP-derived labels, heterogeneous sources
- ResearchRabbit literature graph — data-centric AI at 98%+ accuracy, medical image augmentation best practices, focal loss for class imbalance

### Secondary (MEDIUM confidence)
- Medical imaging forums (Stack Overflow, Reddit r/MachineLearning) — community consensus on augmentation parameters, chest X-ray specific constraints
- PyTorch discussion forums — focal loss implementations, MixUp/CutMix integration patterns
- cleanlab GitHub issues — threshold tuning strategies, false positive management, medical imaging use cases

### Tertiary (LOW confidence)
- Specific expected gain percentages (+0.5-1.5% for label cleaning) — inferred from data-centric AI literature, not specific to this dataset/domain

---
*Research completed: 2026-02-16*
*Ready for roadmap: yes*
