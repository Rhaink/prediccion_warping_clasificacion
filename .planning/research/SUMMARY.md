# Project Research Summary

**Project:** Ensemble Learning + Test-Time Augmentation for COVID-19 Classification
**Domain:** Medical Image Analysis (Chest X-Ray Classification)
**Researched:** 2026-01-27
**Confidence:** HIGH

## Executive Summary

This thesis project enhances an existing COVID-19 chest X-ray classification system by adding ensemble learning and test-time augmentation (TTA) capabilities to 5 trained ResNet-18 models from cross-validation. The research reveals three critical insights: (1) Use PyTorch-native ensemble implementation rather than heavyweight frameworks for faster integration (2-3 days vs weeks), (2) Restrict TTA to medical-safe augmentations (horizontal flip only, validated with landmark symmetry correction), and (3) Treat this as EVALUATION-FOCUSED work, not training new models.

The recommended approach prioritizes thesis timeline and methodological rigor. PyTorch native soft voting with ttach for TTA (2 new dependencies) integrates into existing CLI patterns within 2-3 days. Expected accuracy gain is modest (+0.5-1.0pp, from 98% baseline), so thesis contribution should emphasize uncertainty quantification via ensemble disagreement analysis. The architecture follows proven inference-only evaluation patterns: load N models, aggregate predictions, compute metrics, report with confidence intervals.

Critical risk mitigation centers on test set integrity. Research uncovered 15 pitfalls, with 5 CRITICAL issues that invalidate thesis results if mishandled: test set contamination via ensemble selection, data leakage from patient-level splitting, unsafe medical augmentations destroying diagnostic features, inflated metrics reporting (already occurred once in project), and reproducibility failures. The roadmap MUST include a Phase 0 audit to verify current ensemble wasn't cherry-picked on test data and establish strict validation protocols before any new code.

## Key Findings

### Recommended Stack

The minimal ensemble+TTA stack adds only 2 dependencies to the existing PyTorch codebase: `ttach==0.0.3` for test-time augmentation and `torchmetrics>=1.8.2` for standardized classification metrics. Research strongly recommends AGAINST ensemble frameworks (Ensemble-PyTorch, MONAI) as overkill for soft voting, which requires only 50 lines of native PyTorch (`torch.stack().mean()`). This approach integrates in 2-3 days vs weeks for frameworks.

**Core technologies:**
- **PyTorch native ensemble**: Average probabilities via `torch.stack().mean()` — simpler and faster than frameworks, no new dependencies
- **ttach 0.0.3**: Lightweight TTA library with horizontal flip support — medical imaging standard, proven in Kaggle competitions (alternative: custom implementation in 100 lines)
- **torchmetrics 1.8.2+**: Official metrics library with multi-class support — handles accuracy, F1, AUROC, confusion matrices with proper device handling, replaces manual computation

**Critical stack decisions:**
- PyTorch 2.10.0+ (current: 2.0+) includes improved numerical debugging but not required for ensemble
- Avoid albumentations (maintenance mode 2024-2025) unless advanced augmentations needed
- Avoid Neptune.ai (shutting down March 2026) — use JSON logging for thesis
- ttach is unmaintained since 2020 but stable; backup plan: custom TTA (100 lines)

### Expected Features

Research identifies 8 table stakes features users expect in medical imaging ensemble evaluation, 10 differentiators for thesis contributions, and 7 anti-features that seem good but create problems (notably: training ensemble from scratch, test set optimization, aggressive TTA, real-time inference optimization).

**Must have (table stakes):**
- Soft voting (probability averaging) — standard ensemble baseline, medical imaging expects probability outputs
- Hard voting (majority vote) — simplest baseline for comparison
- Per-model metrics (accuracy, F1, confusion matrix) — validate ensemble adds value
- Ensemble aggregated metrics — overall performance with per-class breakdown (COVID/Normal/Viral_Pneumonia)
- TTA with horizontal flip — already validated in landmark pipeline at 3.61px error, reuse infrastructure
- Config-based ensemble definition — follow project pattern (ensemble_best.json)
- Reproducible evaluation — fixed test split, deterministic aggregation, seed tracking
- Confusion matrix visualization — per-model and ensemble

**Should have (competitive):**
- Confidence calibration (temperature scaling) — medical papers increasingly expect this, 2-3% ECE improvement
- Expected Calibration Error (ECE) — quantify reliability of probabilities
- Disagreement analysis — identify WHERE models fail differently, key thesis insight
- Uncertainty quantification — entropy/variance of ensemble predictions flags cases needing review
- Per-sample confidence scores — enables confidence-based routing to radiologist
- Model diversity metrics — pairwise agreement, Kappa statistics validates ensemble composition
- Comparative TTA analysis — beyond flip (rotation, scaling, brightness), medical-safe only
- Export analysis reports — automated thesis-ready outputs (JSON + markdown)

**Defer (v2+ or anti-features):**
- Training ensemble from scratch — already have 5 trained CV models, defeats "quick ensemble" goal
- Test set optimization — TEST CONTAMINATION, invalidates thesis
- Aggressive TTA (rotation >15°, cropping, color inversion) — medical unsafe, creates artifacts
- Weighted voting by validation accuracy — overfits to validation split, complexity without benefit
- Real-time inference optimization — premature for thesis research
- MC Dropout uncertainty — ensemble disagreement is simpler and sufficient

### Architecture Approach

The standard architecture for ensemble+TTA evaluation follows an inference-only pattern with no gradient computation or weight updates. Five layers: (1) Evaluation Orchestrator loads configs/checkpoints/dataset, (2) Inference Pipeline manages N models with TTA engine, (3) Aggregation Layer performs soft voting and prediction validation, (4) Analysis Layer computes metrics with per-class breakdown, (5) Visualization Layer generates thesis-ready reports and plots.

**Major components:**
1. **EnsembleEvaluator** (src_v2/evaluation/ensemble.py) — Central orchestrator loading models, iterating dataset, aggregating predictions, computing metrics. Separates concerns from CLI.
2. **EnsembleWrapper** (src_v2/models/ensemble_wrapper.py) — Lightweight container holding N classifiers with unified interface for batch prediction. Handles device management, supports lazy loading for memory efficiency.
3. **TTA Engine** — Applies medical-safe transformations (horizontal flip with landmark-aware correction), averages predictions across augmentations. Reuses existing flip logic from landmark pipeline (SYMMETRIC_PAIRS).
4. **Metrics Module** (extend src_v2/evaluation/metrics.py) — Computes accuracy, F1, AUROC, confusion matrix using torchmetrics. Includes per-class breakdown essential for imbalanced medical data.
5. **CLI Integration** (src_v2/cli.py) — Add `evaluate-classifier-ensemble` command following existing pattern. Config-driven to avoid hardcoded paths.

**Key architectural patterns:**
- **Model Pool with Lazy Loading**: Load 5x ResNet-18 (~44MB each) fully into memory (feasible). For ensembles >10 models, lazy load one at a time to conserve memory.
- **TTA as Transform Pipeline**: Composable augmentation pipeline where each augmentation generates a view, all views pass through models, results aggregate.
- **Stratified Batch Evaluation**: Evaluate by class (COVID/Normal/Viral_Pneumonia) before global aggregation. Essential for imbalanced medical datasets.

**Data flow:** Test dataset → DataLoader → For each batch: [Original + Flipped views] → For each model: predict both views → Mean(TTA) per model → Stack(N models) → Mean(ensemble) → Argmax → Compare with ground truth → Accumulate metrics.

### Critical Pitfalls

Research identified 15 pitfalls with 5 CRITICAL severity issues that invalidate thesis results. The project already experienced one critical mistake (reporting validation accuracy instead of test), demonstrating vulnerability to these patterns.

1. **Test Set Contamination via Ensemble Selection** — Using test set to select which models to include in ensemble inflates accuracy by 5-30%. Project has ensemble_best.json with seeds {123, 321, 111, 666} — MUST verify these were selected on validation, not test. Prevention: Freeze test set, use validation for all tuning, document decision trail.

2. **Data Leakage via Improper Splitting** — Patient images in both train and test sets inflate accuracy by 29-55%. COVID-19_Radiography_Dataset structure must be verified for patient IDs or multiple views per patient. Prevention: Split at patient level if IDs exist, document methodology, apply augmentation AFTER splitting.

3. **Unsafe Medical Augmentations** — Horizontal flip places heart on RIGHT side (medically impossible), yet improves accuracy via spurious correlations. Thesis reviewers will question validity. Prevention: For chest X-rays, SAFE = small rotations (-5° to +5°), CLAHE, slight scaling. UNSAFE = horizontal flip without landmark correction, vertical flip, large rotations (>10°), cutout. Current project's flip+SYMMETRIC_PAIRS correction is acceptable but requires medical justification in thesis.

4. **Inflated Metrics Reporting** — Project already caught reporting validation instead of test accuracy once. Other variants: peak performance instead of early-stopped model, no confidence intervals, cherry-picking best run. Prevention: Standardize reporting (always test set with label, include std/CI, report worst/best/mean across seeds), separate files for val_results.json and test_results.json.

5. **Ensemble Overfitting via Model Cherry-Picking** — Training 10+ models with different seeds, evaluating on test, selecting "best 4" is indirect test optimization. Current ensemble (seeds {123, 321, 111, 666}) achieving 3.61px error MUST be verified not selected by test performance. Prevention: Pre-specify ensemble strategy on validation set, freeze composition, evaluate once on test. Alternative: Use ALL trained models (no selection bias).

**Additional HIGH severity pitfalls:**
- **CLAIM 2024 Non-Compliance** — Medical AI requires 44-item reporting checklist. Missing items (confidence intervals, train/val/test methodology, limitations discussion) may cause thesis rejection.
- **Reproducibility Failures** — Missing random seeds, library versions, GPU model. Only 5/44 Alzheimer's studies met basic reproducibility criteria. Changing single seed can inflate performance 2-fold.
- **Overfitting by Observer** — Iteratively adjusting methods while observing test performance is subtle data leakage that produces better-than-random results even on synthetic data.

## Implications for Roadmap

Based on research, the roadmap should have 6 phases with strict ordering to prevent test set contamination and ensure thesis validity. Phases 0-1 are CRITICAL pre-implementation work before writing any new code.

### Phase 0: Pre-Implementation Audit (BLOCKING)
**Rationale:** Research uncovered project already made one critical mistake (validation vs test accuracy reporting). Must verify current state before adding ensemble+TTA or risk compounding errors.
**Delivers:** Verified test set integrity, documented ensemble selection methodology, confirmed all reported metrics are test-based
**Addresses:** Pitfalls 1, 2, 5, 6, 11, 15 (test contamination, data leakage, cherry-picking, inflated metrics, reproducibility, overfitting by observer)
**Duration:** 1-2 days
**Verification checklist:**
- Audit experiment logs: Was test set ever loaded during model selection?
- Verify GROUND_TRUTH.json metrics are test-based, not validation
- Check dataset for patient IDs or multiple views per patient (data leakage risk)
- Document how ensemble seeds {123, 321, 111, 666} were selected
- Verify current 98.05% classifier accuracy is test set performance
- Check for patient-level splitting if IDs exist in filenames

### Phase 1: Validation Strategy & Baseline Measurement
**Rationale:** Establish strict validation protocols BEFORE implementation to prevent test contamination. Measure expected improvement based on literature (ResNet18 + TTA: ~1-3% gain).
**Delivers:** Pre-registered analysis plan, validation set reserved for tuning, baseline single-model performance measured
**Addresses:** Pitfalls 1, 7 (test contamination prevention, model complexity trade-off)
**Uses:** Existing trained models, GROUND_TRUTH.json for baseline
**Duration:** 1 day
**Outputs:**
- analysis_plan.md: Pre-specified ensemble config, TTA parameters, metrics before experiments
- Baseline measurement: Single model accuracy, expected TTA gain, improvement target
- Decision log template for tracking validation-based choices

### Phase 2: Core Ensemble Implementation (MVP)
**Rationale:** Implement table stakes features for thesis baseline chapter. Minimal stack (PyTorch native + ttach + torchmetrics), no frameworks.
**Delivers:** Config-based ensemble loading, soft/hard voting, per-model and ensemble metrics, TTA with horizontal flip, reproducible evaluation
**Addresses:** Features: Table stakes (soft voting, hard voting, per-model metrics, ensemble metrics, TTA, config loading, reproducibility)
**Uses:** PyTorch native, ttach, torchmetrics (STACK.md recommendations)
**Implements:** EnsembleEvaluator, EnsembleWrapper, TTA Engine (ARCHITECTURE.md components)
**Duration:** 2-3 days
**Verification:** Single model vs ensemble accuracy on validation set, TTA improves >0.3%

### Phase 3: TTA Safety Validation & Optimization
**Rationale:** Medical imaging requires validating augmentations preserve diagnostic features. Literature shows optimal TTA N=20-40, but current N=2 (flip only) may be suboptimal.
**Delivers:** Visualized augmented samples (verify anatomy preserved), TTA parameter tuning on validation (N samples, transformations), expanded safe augmentation set if beneficial
**Addresses:** Pitfalls 3, 4, 8, 13 (unsafe augmentations, TTA sample size, TTA dropout exclusion, fixed transformation sets)
**Uses:** Medical-safe augmentation guidelines (PITFALLS.md), ablation methodology
**Duration:** 2-3 days
**Experiments on validation set:**
- Test N = {2, 5, 10, 20} TTA samples, plot accuracy vs N
- Test expanded transformations: flip + rotation (-5°, 0°, +5°) = N=6
- Visualize 50 augmented samples, verify lung pathology visible
- Document medical justification for chosen augmentations

### Phase 4: Ablation Studies & Uncertainty Quantification
**Rationale:** Thesis contribution requires isolating ensemble vs TTA vs combined improvements. Uncertainty quantification (ensemble disagreement) strengthens thesis if accuracy gain is modest.
**Delivers:** Ablation results (single model, +TTA only, +ensemble only, +both), disagreement analysis, per-sample confidence scores, model diversity metrics
**Addresses:** Pitfalls 10, 14 (missing ablations, prediction variance not used)
**Uses:** Statistical testing (McNemar's, paired t-test), variance metrics
**Duration:** 2-3 days
**Systematic ablation on validation:**
- Single model (seed 42): X%
- Single model + TTA: X + Δ_TTA %
- Ensemble (4 models, no TTA): X + Δ_ensemble %
- Ensemble + TTA: X + Δ_ensemble+TTA %
- Compute prediction variance across 4 models × 2 TTA = 8 predictions
- Stratify by variance (low/medium/high), measure accuracy per stratum

### Phase 5: Final Test Evaluation & External Validation
**Rationale:** Evaluate final configuration on test set ONCE after all decisions made on validation. External validation on fedcovidx dataset (already available) tests generalizability.
**Delivers:** Test set results with confidence intervals, external dataset results, statistical significance testing, comparison with literature baselines
**Addresses:** Pitfalls 9 (dataset dependence), Pitfall 1 final prevention (test set used once)
**Uses:** Bootstrap CI, McNemar's test, external dataset3_fedcovidx
**Duration:** 1 day
**Strict protocol:**
- Load frozen analysis_plan.md from Phase 1
- Evaluate final ensemble+TTA config on test set ONCE
- Report with confidence intervals (std across ensemble models or bootstrap)
- Test on external dataset (expected: accuracy drops from 98% to 53-57% due to domain shift)
- Document generalizability limitations in thesis discussion

### Phase 6: Thesis Writing & CLAIM Compliance
**Rationale:** Medical AI research requires CLAIM 2024 checklist (44 items) for thesis acceptance. Reproducibility package ensures results can be verified.
**Delivers:** CLAIM-compliant thesis methodology section, reproducibility package (requirements.txt, all seeds documented, complete commands), limitations discussion, thesis-ready figures/tables
**Addresses:** Pitfalls 11, 12 (reproducibility, CLAIM compliance)
**Uses:** CLAIM 2024 checklist (https://pubs.rsna.org/doi/full/10.1148/ryai.240300), thesis style guidelines
**Duration:** Ongoing during Phase 4-5, 2-3 days final
**CLAIM critical items:**
- Item 10a: Document TTA augmentations (horizontal flip + landmark correction)
- Item 10b: Specify TTA is test-time only (not training augmentation)
- Item 12: Describe ensemble methodology (soft voting, N=4 models)
- Item 16: Report confidence intervals (not just point estimates)
- Item 18: Document train/val/test split with seed
- Item 42: Discuss limitations (dataset dependence, external validation failure)

### Phase Ordering Rationale

The strict Phase 0 → Phase 1 → Phase 2-4 → Phase 5 → Phase 6 ordering prevents test set contamination while ensuring thesis validity:

1. **Phase 0 blocks all other work**: Cannot proceed until verifying current ensemble wasn't cherry-picked on test data. Research shows 29-55% accuracy inflation from data leakage.

2. **Phase 1 establishes validation protocol**: Pre-registration prevents "overfitting by observer" where iterative tuning on test data creates subtle leakage. Analysis plan locks decisions before experiments.

3. **Phases 2-4 use validation set exclusively**: All tuning (TTA parameters, augmentation selection, ablation studies) happens on validation data. Test set never loaded.

4. **Phase 5 evaluates test set ONCE**: After all decisions frozen, final config evaluated once on test. External validation tests generalizability but won't change methods.

5. **Phase 6 runs concurrently with Phase 4-5**: Documentation and CLAIM compliance preparation while waiting for final results.

This ordering directly addresses the project's demonstrated vulnerability (already reported validation instead of test once) and prevents the 5 CRITICAL pitfalls that invalidate thesis results.

### Research Flags

**Phases likely needing deeper research:**
- **Phase 3 (TTA Optimization)**: If expanding beyond horizontal flip, need medical literature review on safe augmentation parameters for chest X-rays. Current flip+SYMMETRIC_PAIRS is validated; rotation/scaling need radiologist consultation.
- **Phase 6 (CLAIM Compliance)**: First time applying CLAIM 2024 checklist (44 items). Budget time to review each item against thesis draft.

**Phases with standard patterns (skip research-phase):**
- **Phase 2 (Core Ensemble)**: Well-documented PyTorch patterns. Native soft voting is 50 lines, ttach has examples, torchmetrics has medical imaging tutorials.
- **Phase 4 (Ablation)**: Standard ML evaluation pattern. Statistical testing (McNemar's, paired t-test) is well-established.
- **Phase 5 (Test Evaluation)**: Straightforward inference on held-out set. External validation may show poor results (expected: 53-57% on fedcovidx) but this is feature not bug (demonstrates domain shift).

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | PyTorch native ensemble is proven pattern in existing codebase (landmark ensemble). ttach is medical imaging standard. Only 2 new dependencies, both stable. |
| Features | HIGH | Table stakes features validated by medical imaging literature (soft voting, per-class metrics, TTA). Anti-features confirmed via pitfall research (test optimization, unsafe augmentations). MVP scope is conservative. |
| Architecture | HIGH | Inference-only evaluation follows standard ML patterns. No training loops, no gradient computation. Existing src_v2/evaluation/ structure supports extension. Component responsibilities are clear. |
| Pitfalls | HIGH | Research synthesized 30+ peer-reviewed sources (2020-2026). Project already experienced one critical pitfall (validation vs test reporting), validating research relevance. CLAIM 2024 guidelines are authoritative. |

**Overall confidence:** HIGH

Research is comprehensive for thesis-scoped work with clear constraints (evaluation-only, 2-3 week timeline, existing trained models). Stack recommendations are minimal and proven. Pitfalls research is critical-severity-focused with concrete prevention strategies. The main uncertainty is whether ensemble+TTA improvement will be substantial enough for thesis contribution (literature suggests +0.5-1.0pp from 98% baseline), but uncertainty quantification provides fallback contribution.

### Gaps to Address

**Gap: Expected accuracy improvement may be modest**
- Literature suggests ResNet18 + TTA yields 1-3% gain, but current baseline is already 98.05%
- **Handling**: Frame thesis contribution as uncertainty quantification (ensemble disagreement analysis) + methodological rigor (CLAIM compliance) rather than purely accuracy improvement
- Validate fallback contribution during Phase 4 (compute prediction variance, stratify by confidence)

**Gap: Horizontal flip medical validity**
- Research shows flip places heart on wrong side (medically impossible) yet can improve accuracy
- Current project uses flip+SYMMETRIC_PAIRS landmark correction, which is safe for landmark geometry
- **Handling**: Document medical justification in thesis (bilateral lung symmetry assumption), cite existing landmark validation (3.61px error with flip TTA), consider radiologist consultation if thesis committee questions validity

**Gap: External validation expected to fail**
- Current project shows 98% internal accuracy but 53-57% external (fedcovidx)
- Ensemble+TTA will NOT fix domain shift
- **Handling**: Set correct expectations in Phase 1 analysis plan, frame external validation failure as demonstration of dataset dependence (honest limitation discussion), not as method failure

**Gap: TTA sample size (N=2) may be suboptimal**
- Literature cites optimal N=20-40 for medical imaging, current project uses N=2 (flip only)
- **Handling**: Phase 3 validation on validation set to expand transformations if safe (rotation ±5°, scaling ±5%) or confirm N=2 is sufficient. Document decision rationale in thesis.

**Gap: Ensemble seed selection transparency**
- ensemble_best.json uses seeds {123, 321, 111, 666} achieving 3.61px landmark error
- **Handling**: Phase 0 audit MUST document how these seeds were selected. If test-based, re-run selection on validation. If validation-based, document in analysis_plan.md.

**Gap: Reproducibility verification**
- Project has configs but may be missing complete environment specification
- **Handling**: Phase 6 creates requirements.txt with exact versions (pip freeze), documents GPU/CUDA, adds complete command examples to thesis appendix

## Sources

### Stack Research
**Primary sources (HIGH confidence):**
- PyTorch 2.10.0+ official documentation — FlexAttention, numerical debugging features
- ttach library (qubvel/ttach on GitHub) — Medical imaging TTA standard, Kaggle competition proven
- torchmetrics v1.8.2 (Lightning.ai) — Official metrics library, actively maintained (Jan 2026)
- Ensemble learning research: "An Analysis on Ensemble Learning optimized Medical Image Classification" (2024, arXiv)
- TTA research: "A Large Scale Benchmark for Test Time Adaptation Methods in Medical Image Segmentation" (Dec 2024, arXiv)

**Secondary sources (MEDIUM confidence):**
- Neptune.ai shutdown announcement (March 2026) — verified via multiple sources, avoid for new projects
- Albumentations maintenance mode (2024-2025) — library entered maintenance, stable but minimal updates
- MONAI framework — confirmed overkill for 2D classification (designed for 3D segmentation workflows)

### Features Research
**Primary sources (HIGH confidence):**
- Medical imaging ensemble papers (2024-2025): Soft voting standard, calibration increasingly expected
- TTA in medical imaging: Horizontal flip safe for bilateral symmetry, rotation/color transforms risky
- Classification vs regression ensemble: Calibration critical for classification probabilities
- CLAIM guidelines influence: Per-class metrics, confidence intervals, limitations discussion mandatory

**Feature validation from project context:**
- Landmark ensemble (ensemble_best.json) provides proven template for classifier ensemble
- TTA flip+SYMMETRIC_PAIRS validated at 3.61px error, pattern reusable for classification
- Existing CLI patterns (evaluate-ensemble) inform classifier-ensemble command design

### Architecture Research
**Primary sources (HIGH confidence):**
- Ensemble-PyTorch documentation — confirms soft voting is trivial (10 lines), framework overkill
- VotingClassifier (scikit-learn) — standard pattern, but PyTorch-native is simpler for this use case
- Medical image evaluation architecture papers (2024-2026): Inference-only pattern is standard
- TTA implementation patterns: Transform pipeline vs custom implementation trade-offs

**Project codebase validation:**
- src_v2/evaluation/metrics.py already handles batch processing, device management
- src_v2/models/classifier.py uses create_classifier() factory pattern for checkpoint loading
- src_v2/constants.py::SYMMETRIC_PAIRS provides flip correction logic
- Existing architecture supports evaluation extension with minimal refactoring

### Pitfalls Research
**Primary sources (HIGH confidence):**
- Data leakage studies: "Effect of data leakage in brain MRI classification" (Nature 2021) — 29-55% accuracy inflation
- CLAIM 2024 guidelines (PMC 2024) — 44-item checklist, mandatory for medical imaging AI
- Reproducibility studies: "Checklist for Reproducibility of Deep Learning in Medical Imaging" (PMC 2024)
- Safe augmentation research: "Investigating Image Augmentation for Classification of Chest X-Ray Images" (IEEE 2021)
- Test set contamination: Multiple medical imaging papers documenting 5-30% inflation from improper validation

**Project-specific evidence:**
- Project already made one critical mistake: reporting validation accuracy instead of test
- External validation shows 98% internal vs 53-57% external (dataset dependence confirmed)
- ensemble_best.json with seeds {123, 321, 111, 666} requires verification of selection methodology
- GROUND_TRUTH.json exists but needs verification metrics are test-based

**Critical pitfall confirmation:**
- 15 pitfalls identified, 5 CRITICAL severity confirmed via peer-reviewed sources (2020-2026)
- Only 5/44 Alzheimer's studies met basic reproducibility criteria (MDPI 2024)
- Medical imaging papers increasingly rejected for CLAIM non-compliance (EQUATOR Network 2024)

### Aggregated Source Count
- Medical imaging ensemble/TTA research: 25+ papers (2020-2026)
- Data leakage and validation studies: 10+ papers (2020-2024)
- Reporting guidelines (CLAIM 2024): Official checklist + 5 implementation guides
- Library documentation: PyTorch, torchmetrics, ttach official docs
- Project codebase: CLAUDE.md, GROUND_TRUTH.json, existing src_v2/ structure

---
*Research completed: 2026-01-27*
*Ready for roadmap: yes*
*Next step: Use this summary to structure roadmap phases in PROJECT.md, with special attention to Phase 0 blocking audit*
