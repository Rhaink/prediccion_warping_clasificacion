# Pitfalls Research

**Domain:** Ensemble + Test-Time Augmentation for Medical Imaging Classification (COVID-19 Chest X-Ray)
**Researched:** 2026-01-27
**Confidence:** HIGH

## Critical Pitfalls

### Pitfall 1: Test Set Contamination via Ensemble/Hyperparameter Selection

**What goes wrong:**
Using the test set to select which models to include in the ensemble, optimize ensemble weights, or tune TTA parameters. This inflates reported accuracy by 5-30% in medical imaging studies. The project already corrected one instance (reporting val accuracy instead of test), showing vulnerability to this pattern.

**Why it happens:**
- Iterative ensemble refinement feels like "validation" not "tuning"
- Evaluating multiple ensemble combinations on test set to "pick the best"
- Using test metrics to decide TTA parameters (flip yes/no, number of augmentations)
- Comparing ensemble vs single model performance on same test set without correction

**How to avoid:**
1. **Freeze test set completely** - use ONLY for final evaluation after all decisions made
2. Use validation set (or nested cross-validation) for:
   - Ensemble model selection (which seeds to include)
   - Ensemble weighting schemes (uniform vs weighted averaging)
   - TTA parameter selection (augmentation types, number of samples)
3. Document decision trail: "Ensemble config chosen on validation accuracy X%, then evaluated once on test"
4. If comparing multiple ensemble configurations, use Bonferroni correction or holdout final test set

**Warning signs:**
- Multiple test evaluation runs in logs/notebooks
- Test accuracy mentioned in intermediate experiment notes
- "Let's try adding model X to ensemble and see test performance"
- No validation set strategy documented
- Ensemble config files with timestamps suggesting iteration

**Phase to address:**
Phase 1 (Validation Strategy Design) - Define strict validation protocol before any coding

**Severity:** CRITICAL - Thesis will be rejected if discovered

---

### Pitfall 2: Data Leakage via Improper Train/Val/Test Splitting

**What goes wrong:**
Medical imaging data often has multiple slices/views from the same patient. If patient A's images appear in both training and test sets, accuracy inflates by 29-55% (brain MRI study). The current project uses `split_seed` for reproducibility but must verify subject-level splitting.

**Why it happens:**
- Image-level random split seems "fair" but ignores patient correlation
- Dataset may have multiple acquisitions from same patient with different filenames
- Temporal ordering (patient visits over time) creates leakage if not handled
- Data augmentation applied before splitting (augmented + original in different sets)

**How to avoid:**
1. **Verify current dataset structure**: Check if COVID-19_Radiography_Dataset has patient IDs or multiple images per patient
2. If patient IDs exist: split at patient level, not image level
3. If temporal data: strict temporal split (train on older, test on newer)
4. **Document splitting methodology** in thesis methodology section
5. Apply augmentation AFTER splitting, never before
6. Use stratified split to maintain class balance per set

**Warning signs:**
- No patient/subject ID in dataset metadata
- Unusually high test accuracy compared to validation
- Performance drops sharply on external datasets (53-57% in current project vs 98%+ internal - this could indicate overfitting to internal data structure)
- CSV files with similar filenames in train/test (e.g., "patient_001_view1.jpg" and "patient_001_view2.jpg")

**Phase to address:**
Phase 0 (Pre-Implementation Audit) - Verify dataset structure before any ensemble work

**Severity:** CRITICAL - Invalidates all results

---

### Pitfall 3: Unsafe Medical Augmentations Destroying Diagnostic Features

**What goes wrong:**
Horizontal flip of chest X-rays places the heart on the RIGHT side (medically impossible), creating non-physiological training data. Yet research shows it can improve accuracy (p=0.001), suggesting the model learns spurious correlations. This is especially dangerous for thesis defense - reviewers will question methodological validity.

**Why it happens:**
- Standard computer vision augmentations (flip, extreme rotation, color jitter) copied without medical validation
- "It improves accuracy" justification without checking if improvement is meaningful
- Not consulting domain experts (radiologists) about safe augmentations
- Assuming geometric invariance (rotation) applies to anatomical images

**How to avoid:**
1. **For chest X-rays, SAFE augmentations:**
   - Small rotations (-5° to +5°) - anatomically plausible patient positioning
   - Slight scaling/translation - mimics acquisition variation
   - CLAHE/contrast adjustment - mimics equipment differences
   - Gaussian noise (small) - mimics sensor noise

2. **UNSAFE augmentations (AVOID):**
   - Horizontal flip (creates anatomically impossible images)
   - Vertical flip (nonsensical)
   - Large rotations (>10°)
   - Color jitter/hue changes on grayscale medical images
   - Equalize/Invert (shown to HARM performance in literature)
   - Cutout/random erasing (may remove pathology)

3. **If using horizontal flip for TTA:**
   - **MUST** correct landmark symmetry (swap L/R pairs) - project already does this via SYMMETRIC_PAIRS
   - Document medical justification (e.g., "bilateral symmetry assumption")
   - Compare with/without flip on validation to verify it's not learning artifacts
   - Consult radiologist if possible

4. **Validation approach:**
   - Ablation study: measure performance with/without each augmentation
   - Visualize augmented samples and verify diagnostic features preserved
   - Document rationale for each augmentation in thesis

**Warning signs:**
- Copying augmentation code from natural image classification
- No medical literature citations for augmentation choices
- Extreme augmentation parameters (rotation >15°, cutout >20%)
- Accuracy improves but F1/precision drops (suggests spurious learning)
- Grad-CAM shows model attending to non-lung regions after TTA

**Phase to address:**
Phase 2 (TTA Safety Validation) - Before ensemble implementation, validate each augmentation

**Severity:** CRITICAL - Methodological validity for thesis

---

### Pitfall 4: TTA Sample Size Not Optimized (Too Few = Noisy, Too Many = Diminishing Returns)

**What goes wrong:**
Using too few TTA samples (e.g., N=2 for just flip) gives noisy predictions with high variance. Using too many (N>50) wastes compute with no accuracy gain. Research shows optimal N=20-40 for medical imaging CNNs, with accuracy plateauing beyond that.

**Why it happens:**
- Default to N=2 (original + flip) without validation
- "More is better" assumption leading to N=100+
- Not measuring marginal benefit per additional augmentation
- Confusing TTA sample count with ensemble size

**How to avoid:**
1. **Empirically determine optimal N on validation set:**
   - Test N = {1, 2, 5, 10, 20, 30, 40, 50} on validation set
   - Plot accuracy vs N, find plateau point
   - Balance accuracy gain vs inference time

2. **Current project uses TTA with horizontal flip:**
   - N=2 (original + flip)
   - Verify this is sufficient or if additional augmentations needed
   - Document decision in thesis ("N=2 chosen as accuracy plateaued beyond flip augmentation")

3. **For new TTA implementations:**
   - Start with N=20 as literature-recommended baseline
   - Tune on validation set only
   - Report both single-model and TTA performance for comparison

**Warning signs:**
- No documented rationale for TTA sample count
- Using same N as another paper without validation
- TTA inference time >> training time
- High variance in repeated evaluations with same TTA config
- Marginal accuracy gain <0.1% when doubling N

**Phase to address:**
Phase 3 (TTA Optimization) - After basic ensemble working, tune TTA parameters

**Severity:** MEDIUM - Affects reproducibility and efficiency, not validity

---

### Pitfall 5: Ensemble Overfitting to Test Set via Model Cherry-Picking

**What goes wrong:**
Training 10+ models with different seeds, evaluating all on test set, then selecting the "best 4" for ensemble. This is indirect test set optimization. The current project has ensemble_best.json with seeds {123, 321, 111, 666} achieving 3.61px error - must verify these were NOT selected by test performance.

**Why it happens:**
- Treating seed variation as "randomness" not hyperparameter tuning
- Incremental ensemble building: "Let's add seed X and check test accuracy"
- Not recognizing that model selection IS a form of optimization
- Ensemble paper comparisons show "best ensemble" without documenting selection process

**How to avoid:**
1. **Pre-specify ensemble strategy on validation set:**
   - Train N models with different seeds (on train set)
   - Evaluate on validation set
   - Select ensemble based on validation diversity/accuracy
   - Freeze ensemble composition
   - Evaluate once on test set

2. **Alternative: Use ALL trained models (no selection):**
   - Train K models (e.g., K=5 with seeds 42, 123, 456, 789, 321)
   - Ensemble all K models (no cherry-picking)
   - This avoids selection bias

3. **For current project:**
   - **VERIFY**: Were seeds {123, 321, 111, 666} chosen by validation or test performance?
   - If test: Re-run ensemble selection on validation set only
   - Document selection process transparently in thesis

4. **Ensemble diversity metrics (use on validation):**
   - Disagreement rate between models
   - Correlation of errors
   - Complementary strengths per class

**Warning signs:**
- Ensemble config files with different model combinations and timestamps
- "Best ensemble" without documented selection methodology
- Suspiciously high test accuracy for ensemble vs individual models (ensemble should improve ~1-3%, not 10%)
- No validation metrics in ensemble selection documentation
- Model seeds appear hand-picked rather than systematic

**Phase to address:**
Phase 0 (Pre-Implementation Audit) - Verify current ensemble wasn't cherry-picked on test

**Severity:** CRITICAL - Common mistake in ensemble papers, thesis reviewers will scrutinize

---

### Pitfall 6: Reporting Inflated Metrics (Val Instead of Test, Peak Instead of Final)

**What goes wrong:**
The project already caught this once: reporting validation accuracy instead of test accuracy. Other variants: reporting best epoch (peak performance) instead of early-stopped model, reporting accuracy without confidence intervals, cherry-picking best run across multiple seeds.

**Why it happens:**
- Val and test metrics both available, grabbing wrong one
- Best checkpoint vs final checkpoint confusion
- Multiple experimental runs with different random seeds, reporting maximum
- Pressure to show "good results" for thesis

**How to avoid:**
1. **Standardize reporting protocol:**
   - Always report test set performance (with clear label)
   - Report final early-stopped model, not peak validation
   - Include confidence intervals (std across seeds or bootstrap)
   - Report worst/best/mean across seeds for transparency

2. **Current project validated metrics (from GROUND_TRUTH.json):**
   - Ensemble: 3.61 ± 2.48 px (test set)
   - Classifier: 98.05% accuracy (test set)
   - VERIFY these are test, not validation

3. **Thesis reporting checklist:**
   - [ ] Explicitly state "test set" in all result tables
   - [ ] Include standard deviation or confidence intervals
   - [ ] Report model selection criterion (e.g., "best validation accuracy")
   - [ ] Distinguish between validation metrics (for tuning) and test metrics (for reporting)
   - [ ] Show learning curves (train/val) to demonstrate no overfitting

4. **Add to evaluation scripts:**
   - Print clear labels: "VALIDATION METRICS" vs "TEST METRICS"
   - Save separate files: `val_results.json` and `test_results.json`
   - Add assertion checks to prevent accidentally using test during training

**Warning signs:**
- Results "too good to be true" compared to literature
- No error bars or confidence intervals
- Metrics unlabeled as train/val/test
- Best epoch performance reported instead of early-stopped
- Suspiciously round numbers (99.0% suggests rounding/cherry-picking)

**Phase to address:**
Phase 0 (Pre-Implementation Audit) - Verify current reported metrics are correct

**Severity:** CRITICAL - Thesis integrity issue

---

### Pitfall 7: Not Accounting for Model Complexity vs TTA Benefit Trade-off

**What goes wrong:**
Research shows complex models (deep ResNets, Transformers) benefit LESS from TTA than simple models. Adding ensemble+TTA to already-sophisticated ResNet18 with CoordAttention may yield minimal improvement (<1%), making thesis contribution weak.

**Why it happens:**
- Assuming TTA always helps regardless of base model
- Not measuring baseline improvement potential
- Adding ensemble+TTA because "it's standard practice"
- Complex model already captures invariances that TTA would provide

**How to avoid:**
1. **Baseline measurement:**
   - Single model performance: X%
   - Expected TTA gain for ResNet18: ~1-3% (from literature)
   - If gain <1%, question whether TTA adds value

2. **For current project (ResNet18 + CoordAttention):**
   - Check if model already learns flip invariance (unlikely for medical images)
   - Measure improvement: single model vs ensemble vs ensemble+TTA
   - Document marginal contribution of each component

3. **Thesis framing:**
   - If improvement is small: emphasize robustness/uncertainty quantification
   - If improvement is large: investigate why (suggests base model underfit)
   - Compare with simpler baseline (ResNet18 without CoordAttention)

4. **Alternative value propositions if accuracy gain is minimal:**
   - Uncertainty quantification via prediction variance
   - Robustness to perturbations (current project has strong robustness results)
   - Confidence calibration
   - Out-of-distribution detection

**Warning signs:**
- Ensemble+TTA improves <0.5% over single model
- Validation accuracy already >98% (ceiling effect)
- TTA variance is high (predictions unstable)
- Literature reports TTA works well for simpler models (VGG) but not ResNets

**Phase to address:**
Phase 1 (Baseline Validation) - Measure expected improvement before implementation

**Severity:** MEDIUM - Affects thesis contribution, not validity

---

### Pitfall 8: TTA Dropout (TTD) Harming Performance Instead of Helping

**What goes wrong:**
Test-Time Dropout (activating dropout during inference for uncertainty estimation) introduces uncontrolled stochastic perturbations that can DECREASE segmentation/classification accuracy. Recent research (2024) warns TTD may undermine performance and yield unstable outputs.

**Why it happens:**
- Confusing TTA (geometric/photometric augmentations) with TTD (stochastic dropout)
- Following uncertainty quantification papers without validating on own data
- Assuming randomness = better ensemble diversity
- Not measuring TTD impact on accuracy

**How to avoid:**
1. **Do NOT use Test-Time Dropout** unless explicitly validating it improves results
2. **If uncertainty quantification is needed:**
   - Use TTA variance (prediction disagreement across augmentations)
   - Use ensemble disagreement (prediction variance across models)
   - Use Monte Carlo Dropout ONLY IF validated on validation set

3. **For current project:**
   - Stick to geometric TTA (horizontal flip + landmark correction)
   - Ensemble diversity comes from different training seeds, not dropout randomness
   - Measure uncertainty as prediction variance across {ensemble models} × {TTA augmentations}

4. **Validation protocol if considering TTD:**
   - Measure accuracy: no dropout vs TTD with N={5,10,20} samples
   - If TTD decreases accuracy >0.5%, reject it
   - Document decision in thesis

**Warning signs:**
- Inconsistent predictions on same image across runs (with fixed seed)
- Accuracy drops when adding "uncertainty estimation"
- High prediction variance without corresponding accuracy gain
- Papers cited are from segmentation, not classification (TTD effects differ)

**Phase to address:**
Phase 2 (TTA Implementation) - Explicitly exclude TTD from design

**Severity:** MEDIUM - Can harm performance if implemented

---

### Pitfall 9: Dataset Dependence Not Acknowledged (Claiming Universal Improvement)

**What goes wrong:**
TTA performance varies significantly across datasets. Datasets with homogeneous instances benefit LESS from TTA than heterogeneous ones. The current project's external validation (53-57% vs 98% internal) suggests strong dataset dependence, yet ensemble+TTA might be claimed as "universal improvement."

**Why it happens:**
- Validating only on internal test set (same distribution as training)
- Not testing on external datasets (different hospitals, equipment, populations)
- Claiming "ensemble+TTA improves COVID detection" when it only improves on this specific dataset
- Generalizing from single-dataset experiments

**How to avoid:**
1. **Acknowledge dataset limitations in thesis:**
   - "Ensemble+TTA improves accuracy from X% to Y% on COVID-19_Radiography_Dataset"
   - NOT "Ensemble+TTA improves COVID-19 detection" (implies universal)

2. **External validation (if feasible):**
   - Test on dataset3_fedcovidx (already available in project)
   - If ensemble+TTA helps on external data: strong contribution
   - If no improvement: acknowledge limitation, discuss domain adaptation

3. **Current project context:**
   - Internal test: 98.05% accuracy
   - External test: 53-57% (all models fail due to domain shift)
   - **Ensemble+TTA will NOT fix domain shift** - set correct expectations

4. **Thesis discussion section:**
   - Limitations: "Results specific to this dataset/acquisition protocol"
   - Future work: "Validate on multi-center datasets, explore domain adaptation"
   - Compare ensemble+TTA improvement internally (controlled) vs externally (generalization)

**Warning signs:**
- No external validation attempted
- Overclaiming in abstract/conclusions ("improves COVID detection" vs "improves accuracy on Dataset X")
- Not discussing generalizability limitations
- Ignoring current project's 53% external accuracy failure

**Phase to address:**
Phase 5 (External Validation) - After internal validation complete

**Severity:** MEDIUM - Affects thesis claims strength, not methodology

---

### Pitfall 10: Not Ablating Ensemble vs TTA vs Ensemble+TTA Contributions

**What goes wrong:**
Reporting only final "ensemble+TTA" accuracy without isolating contributions of (1) ensemble alone, (2) TTA alone, (3) their combination. This prevents understanding which component drives improvement and makes thesis contribution unclear.

**Why it happens:**
- Implementing both components together from the start
- Assuming combined approach is obviously best
- Not considering that ensemble and TTA might be redundant (both provide similar diversity)
- Skipping ablation studies to save time

**How to avoid:**
1. **Systematic ablation on validation set:**
   ```
   Single model (seed 42):               X%
   Single model + TTA:                   X + Δ_TTA %
   Ensemble (4 models, no TTA):          X + Δ_ensemble %
   Ensemble + TTA:                       X + Δ_ensemble+TTA %
   ```

2. **Key questions to answer:**
   - Is Δ_ensemble+TTA > Δ_TTA + Δ_ensemble? (Synergy or redundancy?)
   - Does TTA provide same benefit to ensemble as to single model?
   - Is improvement statistically significant?

3. **For current project:**
   - Baseline: Single ResNet18 (from GROUND_TRUTH.json, find single model accuracy)
   - Ensemble only: Average 4 models without TTA
   - TTA only: Best single model with TTA
   - Ensemble+TTA: 4 models with TTA (target configuration)

4. **Statistical testing:**
   - Use McNemar's test or paired t-test to compare configurations
   - Report p-values for claimed improvements
   - Consider multiple comparison correction (Bonferroni)

**Warning signs:**
- Only final combined results reported
- No intermediate baselines in results section
- Unable to answer "which component contributes more?"
- Claiming "ensemble+TTA improves X%" without showing single model baseline

**Phase to address:**
Phase 4 (Ablation Studies) - After ensemble+TTA implementation

**Severity:** MEDIUM - Affects thesis contribution clarity

---

### Pitfall 11: Reproducibility Failures Due to Missing Random Seeds/Environment Details

**What goes wrong:**
Results cannot be reproduced because random seeds, library versions, GPU models, or threading settings are not documented. Research shows that changing a single random seed can inflate model performance by up to 2-fold. Only 5 out of 44 Alzheimer's studies met basic reproducibility criteria.

**Why it happens:**
- Assuming code + data is sufficient for reproducibility
- Not realizing PyTorch/CUDA non-determinism affects results
- Using default random seeds without documenting them
- Not tracking library versions (pip freeze)
- GPU model/CUDA version differences causing numerical variations

**How to avoid:**
1. **Document ALL random seeds used:**
   - Training seed (for weight initialization, data shuffling)
   - Split seed (for train/val/test division) - project uses `split_seed` ✓
   - Augmentation seed (for random transformations)
   - Example: Current project uses seeds {123, 321, 111, 666} for ensemble - VERIFY these are all documented

2. **Environment specification:**
   - Create `requirements.txt` with exact versions (pip freeze)
   - Document CUDA version, GPU model
   - Document PyTorch deterministic settings (torch.backends.cudnn.deterministic = True)
   - Add environment info to thesis methodology appendix

3. **Reproducibility checklist for thesis:**
   - [ ] All random seeds documented in config files
   - [ ] requirements.txt with exact library versions
   - [ ] GPU/CUDA version documented
   - [ ] Deterministic mode enabled for PyTorch (if possible)
   - [ ] Complete command-line examples in thesis appendix
   - [ ] Checkpoints and configs archived with DOI (Zenodo/Figshare)

4. **Current project status:**
   - GOOD: Uses config files (ensemble_best.json, warping_best.json)
   - GOOD: Documents split_seed in code
   - VERIFY: Are all training seeds in configs?
   - ADD: requirements.txt with exact versions
   - ADD: Document GPU model in thesis methodology

**Warning signs:**
- "Works on my machine" but collaborators can't reproduce
- Results vary when re-running same script
- No requirements.txt or outdated versions
- Seeds hardcoded in scripts instead of configs
- Missing GPU/CUDA documentation in thesis

**Phase to address:**
Phase 0 (Pre-Implementation Audit) + Phase 6 (Final Reproducibility Verification)

**Severity:** CRITICAL - Thesis may not be accepted without reproducibility

---

### Pitfall 12: Non-Compliance with CLAIM Reporting Guidelines (2024 Update)

**What goes wrong:**
Medical imaging AI research has mandatory reporting guidelines (CLAIM - Checklist for Artificial Intelligence in Medical Imaging). Thesis that doesn't follow CLAIM may be rejected or require major revisions. The 2024 update has 44 items that must be addressed.

**Why it happens:**
- Not knowing CLAIM guidelines exist
- Treating medical imaging like general computer vision
- Skipping literature review of reporting standards
- Not consulting recent papers to see reporting norms

**How to avoid:**
1. **Follow CLAIM 2024 checklist (44 items):**
   - Study population and setting
   - Data preprocessing and augmentation (MUST document)
   - Model architecture details
   - Training procedures (hyperparameters, early stopping)
   - Evaluation metrics and statistical tests
   - Reference standard (avoid "ground truth" - use "reference standard")
   - External testing (prefer "external testing" over "external validation")

2. **Critical CLAIM items for ensemble+TTA thesis:**
   - Item 10a: Describe data augmentation (TTA) in detail
   - Item 10b: Document when augmentation is applied (train vs test time)
   - Item 12: Describe ensemble methodology explicitly
   - Item 16: Report confidence intervals (not just point estimates)
   - Item 18: Describe train/val/test split methodology
   - Item 20: Report metrics on independent test set
   - Item 42: Discuss limitations (dataset dependence, generalizability)

3. **For current project:**
   - VERIFY: TTA augmentations documented in methodology (horizontal flip + landmark correction)
   - ADD: Explicit statement that TTA is test-time only (not training augmentation)
   - ADD: Ensemble averaging method (uniform vs weighted)
   - ADD: Confidence intervals to all reported metrics (use std across ensemble models)
   - VERIFY: "Test set" vs "validation set" clearly distinguished in all results

4. **Access CLAIM checklist:**
   - Download from: https://pubs.rsna.org/doi/full/10.1148/ryai.240300
   - Use as thesis writing checklist
   - Include CLAIM compliance statement in thesis methodology

**Warning signs:**
- No reporting checklist consulted during thesis writing
- Metrics reported without confidence intervals
- Train/val/test split not clearly described
- No discussion of limitations or generalizability
- Using "ground truth" instead of "reference standard"
- No external validation or limitation acknowledgment

**Phase to address:**
Phase 6 (Thesis Writing & Compliance) - After all experiments complete

**Severity:** HIGH - Affects thesis acceptance and publication

---

### Pitfall 13: Fixed Transformation Sets Limiting TTA Effectiveness

**What goes wrong:**
Using only horizontal flip for TTA (N=2) may be suboptimal. Recent research (2024) shows that adaptive, diverse transformation sets perform better than fixed minimal sets. However, for medical images, there's a tension between diversity and safety.

**Why it happens:**
- Copying TTA from landmark model (which uses flip for symmetry correction)
- Assuming more augmentations = unsafe for medical images
- Not exploring safe geometric transformations (small rotations, scaling)
- Following papers that only use flip without validating alternatives

**How to avoid:**
1. **Expand safe TTA transformations on validation set:**
   - Current: Horizontal flip only (N=2)
   - Test: Flip + small rotation (-5°, 0°, +5°) = N=6
   - Test: Flip + small scaling (0.95x, 1.0x, 1.05x) = N=6
   - Test: Flip + rotation + scaling = N=18
   - Measure accuracy vs N on validation set

2. **Safe transformation parameters for chest X-rays:**
   - Rotation: -5° to +5° (patient positioning variation)
   - Scaling: 0.95x to 1.05x (distance variation)
   - Translation: ±5 pixels (acquisition alignment)
   - Brightness/contrast: ±10% (equipment variation)
   - ALL must preserve diagnostic features

3. **Validation protocol:**
   - Ablate each transformation individually on validation set
   - Measure: accuracy, confidence calibration, inference time
   - Visualize: Ensure lung pathology remains visible
   - Compare: Fixed set (flip only) vs diverse set (flip+rotation+scale)

4. **For current project:**
   - BASELINE: Horizontal flip only (N=2, existing)
   - EXPERIMENT: Add rotation ±5° (N=6)
   - MEASURE: Improvement on validation set
   - If improvement <0.5%: Stick with N=2 for simplicity
   - If improvement >1%: Use expanded set, document in thesis

**Warning signs:**
- Using N=2 without justification
- Not testing alternative transformations
- TTA performance plateaus immediately (suggests more diversity could help)
- Literature cites N=20-40 optimal, but project uses N=2

**Phase to address:**
Phase 3 (TTA Optimization) - After basic TTA implementation

**Severity:** MEDIUM - Affects performance ceiling, not validity

---

### Pitfall 14: Prediction Variance Not Used for Uncertainty Quantification

**What goes wrong:**
Ensemble+TTA generates multiple predictions per image (4 models × 2 TTA = 8 predictions for current project), but variance is not analyzed. This misses a key thesis contribution: uncertainty quantification for clinical decision support. High variance = low confidence.

**Why it happens:**
- Focusing only on average prediction (final class)
- Not computing or reporting prediction variance
- Treating ensemble as "better accuracy" not "uncertainty estimation"
- Not connecting to clinical utility (flagging uncertain cases)

**How to avoid:**
1. **Compute prediction variance metrics:**
   - Per-image variance: std of 8 predictions (4 models × 2 TTA)
   - Predictive entropy: -Σ p_i log(p_i) across ensemble predictions
   - Disagreement rate: % of ensemble predictions that differ from majority
   - Confidence: max(softmax probability) across ensemble

2. **Clinical utility analysis:**
   - Stratify test set by prediction variance (low/medium/high)
   - Measure accuracy per stratum (expect: high variance = lower accuracy)
   - Identify threshold: "If variance > X, flag for radiologist review"
   - Report: "Ensemble flags 10% of cases as uncertain, achieving 99.5% accuracy on remaining 90%"

3. **For current project:**
   - COMPUTE: Variance across 4 models without TTA
   - COMPUTE: Variance across 4 models with TTA (8 predictions total)
   - COMPARE: Does TTA increase or decrease variance?
   - ANALYZE: Correlation between variance and correctness
   - ADD to thesis: Uncertainty quantification as secondary contribution

4. **Visualization for thesis:**
   - Plot: Prediction variance vs accuracy (scatter plot)
   - Example: Show high-variance misclassification vs low-variance correct prediction
   - Confusion matrix with variance overlay

**Warning signs:**
- Only reporting average accuracy, no variance analysis
- No discussion of uncertainty quantification in thesis outline
- Not using prediction variance for any downstream application
- Missing opportunity to strengthen thesis contribution

**Phase to address:**
Phase 4 (Analysis & Uncertainty Quantification) - After ensemble+TTA evaluation

**Severity:** MEDIUM - Missed opportunity for stronger thesis contribution

---

### Pitfall 15: Overfitting by Observer (Iterative Method Adjustment Based on Test Performance)

**What goes wrong:**
Researchers iteratively adjust methods (ensemble selection, TTA parameters, preprocessing) while observing cross-validation or test performance, effectively including the test set in the validation process. This is a subtle form of data leakage that can produce better-than-random results even on randomly generated data.

**Why it happens:**
- Running experiment → checking test accuracy → tweaking → re-running (feels like iteration, is actually leakage)
- Using cross-validation results to guide method selection across many iterations
- "Just one more experiment" syndrome with test set visibility
- Not pre-specifying analysis plan before seeing data

**How to avoid:**
1. **Pre-registration approach:**
   - Write detailed analysis plan BEFORE running experiments (Phase 1)
   - Specify: ensemble size, model seeds, TTA parameters, evaluation metrics
   - Lock test set: only evaluate final configuration once
   - Document any deviations from plan in thesis

2. **Locked validation data:**
   - Use separate validation set for all tuning decisions
   - Keep test set completely hidden until final evaluation
   - Alternatively: Use nested cross-validation (outer loop for testing, inner loop for validation)

3. **Decision audit trail:**
   - Log ALL experiment runs with timestamps
   - For each decision (e.g., "use 4 models"), document validation metric that justified it
   - Show thesis committee: "We selected 4 models based on validation accuracy X%, then evaluated once on test achieving Y%"

4. **For current project:**
   - AUDIT: Review experiment history to check if test set was used iteratively
   - VERIFY: Ensemble seeds {123, 321, 111, 666} were selected BEFORE test evaluation
   - If test was used: Re-split data, re-run final config on new test set, report both results

**Warning signs:**
- Many experiment runs with slightly different configurations
- Test accuracy in intermediate experiment logs
- Decision rationale not documented (suggests ad-hoc tuning)
- Test metrics "too good" compared to validation (suggests overfitting by observer)
- Cannot explain why specific ensemble configuration was chosen

**Phase to address:**
Phase 0 (Pre-Implementation Audit) - Verify current methods weren't overfit to test

**Severity:** CRITICAL - Subtle but invalidates results if present

---

## Thesis Defense Preparation: Common Questions to Anticipate

Questions reviewers will ask if pitfalls are not addressed.

| Question | What It Reveals | How to Answer Proactively |
|----------|----------------|---------------------------|
| "How did you select which models to include in your ensemble?" | Testing for ensemble cherry-picking | Document validation-based selection: "We trained 5 models with seeds X, evaluated on validation set, selected top 4 by diversity metrics" |
| "Why did you use horizontal flip for TTA on chest X-rays?" | Testing for medical validity | Justify with symmetry assumption + landmark correction: "Bilateral lung symmetry allows flip with L/R landmark swapping; validated safe via radiologist consultation" |
| "What was your train/validation/test split methodology?" | Testing for data leakage | Describe patient-level (if IDs exist) or image-level split with stratification: "80/10/10 split with split_seed=42, stratified by class, no patient overlap" |
| "How much does ensemble contribute vs TTA individually?" | Testing for ablation completeness | Show ablation table: "Single model: 97.5%, +TTA: 98.1%, +Ensemble: 98.3%, +Both: 98.8% (synergistic improvement)" |
| "Are these results reproducible?" | Testing for documentation quality | Provide reproducibility package: "All seeds documented in configs, requirements.txt with exact versions, trained on RTX 3090 with CUDA 11.8" |
| "Did you use the test set for any tuning decisions?" | Testing for test set contamination | Pre-empt with decision log: "All hyperparameters and ensemble selection performed on validation set; test set evaluated once for final results" |
| "What are confidence intervals on your reported accuracy?" | Testing for statistical rigor | Report with error bars: "98.05% ± 0.34% (95% CI via bootstrap, N=1000)" or "98.05% ± 0.25% (std across 4 ensemble models)" |
| "How does your method generalize to other datasets?" | Testing for overclaiming | Acknowledge limitations: "Evaluated on external fedcovidx dataset: 55% accuracy due to domain shift; future work includes domain adaptation" |
| "Why not use more sophisticated TTA (e.g., N=50 augmentations)?" | Testing for design rationale | Justify with validation: "Tested N={2,5,10,20,50} on validation; accuracy plateaued at N=6, selected N=2 for efficiency with minimal loss" |
| "How do you handle uncertainty in predictions?" | Testing for clinical utility | Present variance analysis: "Ensemble prediction variance identifies 10% uncertain cases; flagging these for review achieves 99.2% accuracy on remaining 90%" |

**Pro tip:** Add a "Limitations and Threats to Validity" section to thesis that proactively addresses these questions before reviewers ask.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Using single train/val/test split instead of k-fold CV | Faster experiments, simpler code | Results may be split-dependent, harder to defend in thesis | Only if dataset is large (>10k samples) AND single split is pre-specified |
| Hardcoding ensemble model paths instead of config file | Quick testing | Non-reproducible, difficult to document in thesis | NEVER - already have config system |
| Skipping confidence intervals/error bars | Cleaner result tables | Cannot assess statistical significance, weak thesis defense | NEVER for thesis |
| Using validation set for both model selection AND final comparison | Saves test set data | Optimistic bias, thesis reviewers will question | NEVER - use nested CV or separate holdout |
| Applying same CLAHE to all images without per-image adaptation | Consistent preprocessing | May not handle acquisition variation (e.g., external datasets) | Acceptable for controlled single-center study |
| Implementing TTA without measuring variance/uncertainty | Simpler code, faster inference | Misses key thesis contribution (uncertainty quantification) | Only if thesis focuses purely on accuracy |

---

## Integration Gotchas

Common mistakes when connecting to external services/data.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| External datasets (fedcovidx) | Assuming same preprocessing works (CLAHE clip/tile) | Validate preprocessing on external data, may need different parameters or no CLAHE |
| Pretrained models (ResNet18) | Using ImageNet weights without validating benefit | Compare random init vs pretrained on validation set (medical images differ from ImageNet) |
| Landmark predictions (cached .npz) | Assuming predictions valid for all experiments | Verify predictions match current model ensemble, regenerate if ensemble changes |
| CLAHE preprocessing | Applying same clip_limit=2.0 to all images | Current project validated this value; document rationale if changing for ensemble experiments |
| Git checkpoints | Overwriting "best_model.pt" across experiments | Use descriptive names (ensemble_best_20260111.pt) to prevent accidental deletion |

---

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Loading all augmented images into memory | Out of memory errors during TTA | Use on-the-fly augmentation, process batches | N_models × N_TTA × batch_size exceeds RAM (~4 models × 20 TTA × 32 batch = ~2.5k images) |
| Sequential ensemble inference | Slow evaluation (4 models × TTA each serially) | Parallelize model inference if GPUs available | When ensemble size >4 or TTA samples >10 |
| Storing all TTA predictions before averaging | Memory usage scales with N_TTA | Stream predictions, running average | N_TTA > 50 |
| Regenerating warped dataset for each classifier experiment | Wasted compute, inconsistent data | Cache warped images (already done in project), reuse across experiments | Warping takes >30min (already mitigated) |
| Re-running landmark prediction for each experiment | Hours of GPU time | Use cached predictions.npz (already done in project) | Dataset size >10k images |

---

## Security Mistakes

Domain-specific security issues beyond general web security.

| Mistake | Risk | Prevention |
|---------|------|------------|
| Committing patient data to git | HIPAA/GDPR violation, thesis rejected | Add `data/dataset/` to .gitignore, use synthetic data for demos |
| Logging image paths with patient IDs | De-anonymization risk | Strip patient IDs in logs, use anonymized identifiers |
| Sharing trained models without data usage agreement | License violation, legal issues | Check COVID-19_Radiography_Dataset license, add disclaimer to model sharing |
| Including identifiable images in thesis figures | Privacy violation | Use only de-identified images, blur metadata, obtain consent if needed |
| Storing predictions with patient identifiers | Data breach risk | Use image hashes or anonymized IDs in prediction caches |

---

## UX Pitfalls

Common user experience mistakes in this domain (thesis readers, code users).

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Not documenting which split_seed was used | Cannot reproduce results | Add split_seed to all config files and result logs |
| Reporting accuracy without specifying which test set | Confusion: val or test? internal or external? | Always label: "Test set (internal, N=X)" or "External validation (fedcovidx)" |
| Mixing Spanish and English in results | Thesis reviewers confused by class names | Keep class names as-is (COVID, Viral_Pneumonia) but translate all analysis text to English |
| No visualization of ensemble predictions | Hard to debug failures | Add ensemble prediction visualization (e.g., heatmap of model disagreement) |
| Complex CLI commands without examples | Users cannot reproduce | Add examples to CLAUDE.md (already done well) and thesis appendix |

---

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Ensemble implementation:** Often missing ablation study comparing ensemble-only vs TTA-only vs combined — verify contributions are isolated
- [ ] **TTA implementation:** Often missing validation of augmentation safety (visualize augmented samples) — verify chest anatomy preserved
- [ ] **Test set evaluation:** Often missing verification that test set was never used for tuning — audit experiment logs/notebooks
- [ ] **Confidence intervals:** Often missing error bars on reported metrics — add std or bootstrap CI to all key results
- [ ] **External validation:** Often missing completely — test on fedcovidx or acknowledge limitation in thesis discussion
- [ ] **Statistical significance:** Often missing p-values for claimed improvements — add McNemar's test or paired t-test
- [ ] **Reproducibility:** Often missing exact commands to reproduce results — verify all steps documented in CLAUDE.md
- [ ] **Dataset split verification:** Often missing patient-level split validation — check for patient ID patterns in filenames
- [ ] **Ensemble model selection justification:** Often missing documentation of why these 4 seeds — verify selection was validation-based
- [ ] **TTA parameter justification:** Often missing ablation of N_TTA samples — validate N=2 is sufficient on validation set
- [ ] **Random seed documentation:** Often missing training seeds, split seeds, augmentation seeds — document ALL seeds in configs
- [ ] **CLAIM compliance:** Often missing proper reporting checklist adherence — verify thesis follows CLAIM 2024 guidelines (44 items)
- [ ] **Uncertainty quantification:** Often missing prediction variance analysis — compute and report ensemble disagreement metrics
- [ ] **Pre-registration/analysis plan:** Often missing documented method decisions before seeing test results — create decision audit trail
- [ ] **Library versions:** Often missing requirements.txt with exact versions — run pip freeze and commit to repo

---

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Test set contamination (used for ensemble selection) | HIGH | Re-split data with new random seed, re-train all models, re-evaluate ensemble on new test set; report original results as "preliminary" in thesis |
| Data leakage (patient images in train+test) | HIGH | Identify patient IDs, re-split at patient level, re-train all models; compare leaked vs clean results in thesis discussion |
| Unsafe augmentation used | MEDIUM | Remove unsafe augmentation, re-train ensemble on validation set, evaluate on test; report ablation showing harm in thesis |
| Cherry-picked ensemble on test performance | HIGH | Re-run ensemble selection on validation set only, freeze config, re-evaluate on test once; acknowledge methodology error if results differ significantly |
| Reported validation instead of test | LOW | Re-evaluate on correct test set, update all result tables/figures; acknowledge correction in thesis if already shared results |
| TTA sample size not optimized | LOW | Run ablation on validation set (N=2,5,10,20), select optimal N, re-evaluate ensemble; minimal impact expected |
| Missing ablation studies | MEDIUM | Train single-model and ensemble-only baselines, evaluate on same test set, add ablation section to thesis results |
| No external validation | MEDIUM | Evaluate final ensemble+TTA on fedcovidx (already available), acknowledge limitations in discussion if results poor |
| No confidence intervals | LOW | Bootstrap resample test set or report std across ensemble models, update result tables |
| Irreproducible results (missing seeds/configs) | MEDIUM | Document all current configs/seeds in GROUND_TRUTH.json (already done), add reproducibility section to thesis methodology |
| Non-compliance with CLAIM | MEDIUM | Download CLAIM 2024 checklist, systematically address all 44 items in thesis revision, add compliance statement |
| Fixed transformation set suboptimal | LOW | Expand TTA to include safe rotations/scaling on validation set, re-evaluate final ensemble on test once |
| Prediction variance not analyzed | LOW | Compute variance metrics on existing test predictions (no re-training needed), add uncertainty quantification section |
| Overfitting by observer | HIGH | Document all experiment iterations with timestamps, verify decisions used validation not test, re-run final config on fresh test split if needed |

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Test set contamination | Phase 0 (Pre-Implementation Audit) | Audit experiment logs, verify test set never loaded during development |
| Data leakage | Phase 0 (Pre-Implementation Audit) | Check dataset for patient IDs, verify split is image-level (or patient-level if IDs exist) |
| Unsafe augmentations | Phase 2 (TTA Safety Validation) | Visualize 50 augmented samples, verify lung anatomy preserved, consult radiologist if possible |
| TTA sample size | Phase 3 (TTA Optimization) | Plot accuracy vs N_TTA on validation set, verify plateau, document optimal N |
| Ensemble cherry-picking | Phase 0 (Pre-Implementation Audit) | Verify current ensemble seeds were selected on validation, not test; document selection rationale |
| Inflated metrics | Phase 0 (Pre-Implementation Audit) | Verify GROUND_TRUTH.json metrics are test-based, re-evaluate if unclear |
| Model complexity trade-off | Phase 1 (Baseline Validation) | Measure single model accuracy, expected TTA gain from literature, set improvement target |
| TTA Dropout | Phase 2 (TTA Implementation) | Explicitly design TTA as geometric augmentations only, exclude dropout |
| Dataset dependence | Phase 5 (External Validation) | Evaluate on fedcovidx, report generalization gap, discuss in thesis limitations |
| Missing ablations | Phase 4 (Ablation Studies) | Train single-model baseline, ensemble-only, TTA-only, combined; report all four results |
| Reproducibility failures | Phase 0 (Audit) + Phase 6 (Final Verification) | Document all seeds, create requirements.txt, verify reproduction on clean environment |
| CLAIM non-compliance | Phase 6 (Thesis Writing) | Download CLAIM 2024 checklist, verify all 44 items addressed in thesis draft |
| Fixed transformation sets | Phase 3 (TTA Optimization) | Test expanded safe transformations on validation, select optimal set |
| Prediction variance not used | Phase 4 (Analysis) | Compute variance metrics, analyze correlation with correctness, add to thesis results |
| Overfitting by observer | Phase 0 (Pre-Implementation) + Phase 1 (Pre-Registration) | Create analysis plan before experiments, audit decision trail, verify validation-based selection |

---

## Sources

### Medical Imaging TTA & Ensemble Research
- [Test-Time Generative Augmentation for Medical Image Segmentation](https://www.sciencedirect.com/science/article/abs/pii/S1361841525004487)
- [Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with convolutional neural networks](https://pmc.ncbi.nlm.nih.gov/articles/PMC6783308/)
- [Test-Time Generative Augmentation for Medical Image Segmentation (arXiv)](https://arxiv.org/html/2406.17608v1)
- [Improving Medical Image Segmentation Using Test-Time Augmentation with MedSAM](https://www.mdpi.com/2227-7390/12/24/4003)
- [Understanding Test-Time Augmentation](https://arxiv.org/html/2402.06892v1)

### Data Leakage in Medical Imaging
- [Inflation of test accuracy due to data leakage in deep learning-based classification of OCT images](https://www.nature.com/articles/s41597-022-01618-6)
- [Effect of data leakage in brain MRI classification using 2D convolutional neural networks](https://www.nature.com/articles/s41598-021-01681-w)
- [Data Leakage in Deep Learning for Alzheimer's Disease Diagnosis: A Scoping Review](https://www.mdpi.com/2075-4418/15/18/2348)
- [Effect of data leakage in brain MRI classification (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8604922/)

### Safe Augmentations for Chest X-Rays
- [A Review of Recent Advances in Deep Learning Models for Chest Disease Detection Using Radiography](https://pmc.ncbi.nlm.nih.gov/articles/PMC9818166/)
- [Differential Data Augmentation Techniques for Medical Imaging Classification Tasks](https://pmc.ncbi.nlm.nih.gov/articles/PMC5977656/)
- [The Effectiveness of Image Augmentation in Deep Learning Networks for Detecting COVID-19](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2021.629134/full)
- [Investigating Image Augmentation for Classification of Chest X-Ray Images](https://ieeexplore.ieee.org/iel7/10008224/10008233/10008268.pdf)

### Ensemble & Validation Best Practices
- [A Guide to Cross-Validation for Artificial Intelligence in Medical Imaging](https://pmc.ncbi.nlm.nih.gov/articles/PMC10388213/)
- [Using Ultrasound Image Augmentation and Ensemble Predictions to Prevent Machine-Learning Model Overfitting](https://pubmed.ncbi.nlm.nih.gov/36766522/)
- [A systematic literature review: exploring the challenges of ensemble model for medical imaging](https://link.springer.com/article/10.1186/s12880-025-01667-4)
- [Nested Cross-Validation for Machine Learning with Python](https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/)

### Medical Imaging Reporting Guidelines (2024)
- [Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update](https://pmc.ncbi.nlm.nih.gov/articles/PMC11304031/)
- [Checklist for Artificial Intelligence in Medical Imaging (CLAIM): A Guide for Authors and Reviewers](https://pmc.ncbi.nlm.nih.gov/articles/PMC8017414/)
- [CLAIM 2024 Update | EQUATOR Network](https://www.equator-network.org/reporting-guidelines/checklist-for-artificial-intelligence-in-medical-imaging-claim-a-guide-for-authors-and-reviewers/)
- [Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update | Radiology: Artificial Intelligence](https://pubs.rsna.org/doi/full/10.1148/ryai.240300)
- [Reporting checklists as compulsory supplements to AI manuscript submissions](https://dirjournal.org/articles/doi/dir.2024.242849)
- [Ten quick tips for computational analysis of medical images](https://pmc.ncbi.nlm.nih.gov/articles/PMC9815662/)

### Reproducibility in Medical Imaging Deep Learning
- [Checklist for Reproducibility of Deep Learning in Medical Imaging](https://pmc.ncbi.nlm.nih.gov/articles/PMC11300409/)
- [Reproducibility in Machine Learning for Medical Imaging - NCBI Bookshelf](https://www.ncbi.nlm.nih.gov/books/NBK597469/)
- [Variability and reproducibility in deep learning for medical image segmentation](https://www.nature.com/articles/s41598-020-69920-0)
- [Challenges to the Reproducibility of Machine Learning Models in Health Care](https://pmc.ncbi.nlm.nih.gov/articles/PMC7335677/)

### Evaluation Metrics and Validation Pitfalls
- [Evaluation metrics in medical imaging AI: fundamentals, pitfalls, misapplications, and recommendations](https://www.sciencedirect.com/science/article/pii/S3050577125000283)
- [Machine learning for medical imaging: methodological failures and recommendations for the future](https://pmc.ncbi.nlm.nih.gov/articles/PMC9005663/)
- [Data Analysis Strategies in Medical Imaging](https://pmc.ncbi.nlm.nih.gov/articles/PMC6082690/)

---

*Pitfalls research for: Ensemble + TTA for COVID-19 Chest X-Ray Classification (Thesis Project)*
*Researched: 2026-01-27*
*Confidence: HIGH - Based on 30+ peer-reviewed sources from 2020-2026*

---

## Key Takeaways for Roadmap Planning

**CRITICAL (Must address in Phase 0 - Pre-Implementation Audit):**
1. Verify test set was never used for ensemble/TTA selection decisions
2. Check dataset for patient-level data leakage
3. Audit current ensemble seed selection (validation vs test)
4. Verify GROUND_TRUTH.json metrics are test-based, not validation
5. Document all random seeds and create analysis plan before experiments

**HIGH PRIORITY (Must address before thesis submission):**
6. Follow CLAIM 2024 guidelines (44 items) in thesis writing
7. Add confidence intervals to all reported metrics
8. Create reproducibility package (seeds, configs, requirements.txt)
9. Perform ablation studies (single model, ensemble-only, TTA-only, combined)
10. Validate TTA augmentation safety (visualize samples, verify anatomy preserved)

**MEDIUM PRIORITY (Strengthen thesis contribution):**
11. Optimize TTA transformation set on validation (expand beyond flip if beneficial)
12. Compute and analyze prediction variance for uncertainty quantification
13. Evaluate on external dataset (fedcovidx) and discuss generalizability
14. Compare expected vs actual TTA improvement (literature baseline)

**DOCUMENTATION (Throughout all phases):**
15. Create decision audit trail showing validation-based choices
16. Document all deviations from pre-specified analysis plan
17. Add reproducibility section to thesis methodology
18. Include CLAIM compliance statement in thesis

---

*Next action: Use this document to inform roadmap phase planning, ensuring each critical pitfall is addressed in appropriate phase with clear verification criteria*
