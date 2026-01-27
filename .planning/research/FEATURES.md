# Feature Research: Ensemble Learning + TTA for Medical Image Classification

**Domain:** COVID-19 Chest X-ray Classification
**Researched:** 2026-01-27
**Confidence:** HIGH

**Context:** Adding ensemble learning + TTA capabilities to existing COVID-19 classifier (5 trained ResNet-18 models from cross-validation). This is EVALUATION-FOCUSED, not training new models. Need rigorous analysis for thesis research.

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Soft Voting (Probability Averaging)** | Standard ensemble method, medical imaging expects probability outputs | LOW | Average predicted probabilities across models, argmax for final prediction. Already partially implemented in landmark ensemble. |
| **Hard Voting (Majority Vote)** | Simplest ensemble baseline, expected for comparison | LOW | Each model votes for single class, majority wins. Fallback if soft voting fails. |
| **Per-Model Metrics** | Need to understand individual model performance before ensemble | LOW | Accuracy, F1, confusion matrix per model. Essential for validating ensemble adds value. |
| **Ensemble Aggregated Metrics** | Overall ensemble performance (accuracy, F1, AUC) | LOW | Standard classification metrics on ensemble predictions. Table stakes for any ensemble. |
| **Per-Class Metrics** | Medical imaging ALWAYS needs per-class analysis (class imbalance) | MEDIUM | Sensitivity, specificity, precision, recall per class (COVID/Normal/Viral_Pneumonia). Critical for clinical relevance. |
| **Confusion Matrix Visualization** | Visual understanding of misclassifications | LOW | Both per-model and ensemble. Already expected in medical AI papers. |
| **Test-Time Augmentation (Horizontal Flip)** | Already implemented in landmark pipeline, users expect consistency | LOW | Flip + symmetric landmark correction already validated at 3.61 px error. Reuse existing TTA infrastructure. |
| **Config-Based Ensemble Definition** | Follow project pattern (ensemble_best.json for landmarks) | LOW | JSON config with model paths, voting strategy, TTA flags. Consistency with existing tooling. |
| **Reproducible Evaluation** | Research requirement - must be reproducible | MEDIUM | Fixed test split, deterministic aggregation, version tracking. Seed management already implemented. |

### Differentiators (Competitive Advantage)

Features that set the product apart. Not required, but valuable for thesis research.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Confidence Calibration (Temperature Scaling)** | Reliable probability estimates crucial for clinical deployment | MEDIUM | Post-hoc temperature scaling on validation set. Medical imaging papers increasingly expect this. Research shows 2-3% ECE improvement. |
| **Expected Calibration Error (ECE)** | Quantify reliability of probability predictions | MEDIUM | Bins predicted probabilities, measures alignment with true frequencies. Standard calibration metric. Dependency: requires validation set holdout. |
| **Disagreement Analysis** | Identify WHERE models fail differently - key thesis insight | HIGH | Per-sample disagreement scores, subgroup analysis by class/error type. Shows ensemble value beyond accuracy. Novel contribution. |
| **Uncertainty Quantification** | Entropy/variance of ensemble predictions | MEDIUM | High disagreement = high uncertainty. Flags cases needing expert review. Clinical safety feature. |
| **Per-Sample Confidence Scores** | Individual prediction reliability | MEDIUM | Max probability or ensemble agreement. Enables confidence-based routing (e.g., low confidence → radiologist review). |
| **Stratified Analysis by Image Characteristics** | Understand where ensemble helps most | HIGH | Analyze by difficulty (original vs. warped), by class, by landmark error bins. Deep thesis analysis. Dependency: requires metadata tracking. |
| **Model Diversity Metrics** | Quantify why ensemble works | MEDIUM | Pairwise agreement, Kappa statistics, oracle accuracy (best model per sample). Validates ensemble composition. Thesis contribution. |
| **Comparative TTA Analysis** | Beyond flip: rotation, scaling, brightness (medical-safe only) | HIGH | Current landmark TTA uses flip only. Evaluate additional safe augmentations. Research contribution. Avoid destructive transforms. |
| **Visualization: Agreement Heatmaps** | Show spatial agreement across models | MEDIUM | Which regions cause disagreement? Overlay on GradCAM. Research figure for paper. |
| **Export Analysis Reports** | Automated thesis-ready outputs | LOW | JSON + markdown reports, LaTeX-ready tables. Saves manual formatting time. |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems in medical imaging context.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Training Ensemble from Scratch** | "Better" ensemble via joint optimization | Already have 5 trained models from CV. Training ensemble is expensive, time-consuming, defeats "quick ensemble" goal. Research focus is evaluation. | Use existing CV models. If needed, fine-tune individual models separately. |
| **Test Set Optimization** | "Tune hyperparameters on test set" | TEST CONTAMINATION. Invalidates all results for thesis. Medical imaging requires strict train/val/test separation. | Reserve validation split for temperature calibration. Test set is NEVER touched for tuning. |
| **Aggressive TTA (Rotation >15°, Cropping, Color Inversion)** | "More augmentation = better" | Medical radiographs have fixed orientation. Severe rotation unrealistic. Color inversion meaningless in grayscale. Warping already normalizes geometry. | Stick to horizontal flip (validated), mild brightness/contrast (±10%). Medical-safe augmentations only. |
| **Weighted Voting by Validation Accuracy** | "Weight better models more" | Overfits to validation split. Models from same CV are similar performance. Complexity without proven benefit in balanced CV. | Use simple averaging. If weighting needed, use ECE-based weights (calibration-aware). |
| **Real-Time Inference Optimization** | "Must be fast for clinical deployment" | This is research/thesis work, not production. Premature optimization. Ensemble inference is inherently slower. | Focus on analysis quality. Document inference time but don't optimize yet. |
| **Ensemble Confidence via Dropout** | "MC Dropout for uncertainty" | Already have ensemble uncertainty from 5 models. MC Dropout adds complexity, training dependency, no clear benefit over ensemble disagreement. | Use ensemble variance/entropy. Simpler, no retraining needed. |
| **Complex Voting Schemes (Weighted Average, Mix Voting)** | "More sophisticated = better" | Introduces hyperparameters needing validation-set tuning. Overfitting risk. Limited data for tuning (thesis dataset not huge). | Start with soft/hard voting. Add weighted voting ONLY if clear underperformance in subgroup analysis. |

---

## Feature Dependencies

```
[Soft Voting]
    └──requires──> [Per-Model Inference]
                       └──requires──> [Config Loader]

[Temperature Scaling] ──requires──> [Validation Split Holdout]
                      └──requires──> [Soft Voting]
                      └──enables──> [ECE Metric]

[Disagreement Analysis] ──requires──> [Per-Sample Predictions]
                        └──enhances──> [Uncertainty Quantification]
                        └──enables──> [Stratified Analysis]

[TTA] ──requires──> [Symmetric Transform Handling]
      └──enhances──> [All Voting Methods]

[Calibration (Temp Scaling)] ──conflicts──> [Test Set Tuning]
                             └──requires──> [Validation Set]

[Model Diversity Metrics] ──requires──> [Per-Model Predictions]
                          └──enhances──> [Disagreement Analysis]

[Stratified Analysis] ──requires──> [Metadata Tracking]
                      └──requires──> [Disagreement Analysis]
```

### Dependency Notes

- **Soft Voting requires Per-Model Inference:** Each model must produce class probabilities, not just argmax. Requires model.eval() + softmax activation.
- **Temperature Scaling requires Validation Split:** Post-hoc calibration needs held-out validation data. CANNOT use test set. Current setup: train/val/test splits exist from CV, can use val fold for calibration.
- **Disagreement Analysis enhances Uncertainty Quantification:** High inter-model disagreement = high epistemic uncertainty. Natural pairing.
- **TTA requires Symmetric Transform Handling:** Landmark flip logic (SYMMETRIC_PAIRS) is already implemented. Classification TTA simpler (no coordinate swapping), but maintain consistency.
- **Calibration conflicts with Test Set Tuning:** Temperature parameter MUST be learned on validation set. Test set is evaluation-only. Strict boundary.
- **Stratified Analysis requires Metadata Tracking:** Need to associate predictions with image paths, classes, landmark errors, warping fill rates. Requires passing metadata through pipeline.

---

## MVP Definition

### Launch With (v1.0 - Thesis Baseline)

Minimum viable product for thesis evaluation chapter.

- [x] **Config-Based Model Loading** — JSON config lists 5 CV model paths. Follows landmark ensemble pattern.
- [x] **Per-Model Inference** — Load each model, run on test set, save predictions. Validate individual model quality.
- [x] **Soft Voting Ensemble** — Average probabilities, argmax for final prediction. Standard baseline.
- [x] **Hard Voting Ensemble** — Majority vote per sample. Comparison to soft voting.
- [x] **Per-Model Metrics** — Accuracy, F1 (macro/weighted), confusion matrix per model. Validate ensemble candidates.
- [x] **Ensemble Metrics** — Aggregated accuracy, F1, AUC, confusion matrix. Core thesis result.
- [x] **Per-Class Breakdown** — Sensitivity, specificity, precision, recall for COVID/Normal/Viral_Pneumonia. Clinical relevance.
- [x] **TTA (Horizontal Flip)** — Apply same flip logic as landmark pipeline. Consistency + validated performance boost.
- [x] **Reproducibility Infrastructure** — Fixed test split, deterministic aggregation, seed tracking. Research hygiene.
- [x] **CLI Integration** — `python -m src_v2 evaluate-classifier-ensemble` command. Consistent with existing CLI.
- [x] **Basic Visualization** — Confusion matrices (per-model + ensemble), bar charts (accuracy comparison). Thesis figures.
- [x] **JSON Export** — Structured results for analysis scripts. Machine-readable output.

**Rationale:** These features answer the core thesis question: "Does ensemble improve classification over single models?" Provides baseline for more advanced analysis.

**Estimated Effort:** 2-3 days development + 1 day validation.

### Add After Validation (v1.1 - Confidence Analysis)

Features to add once v1.0 shows ensemble benefit.

- [ ] **Temperature Scaling** — Post-hoc calibration on validation set. Trigger: if v1.0 shows >5% ECE or overconfident predictions.
- [ ] **ECE Metric** — Quantify calibration quality. Trigger: after temperature scaling implemented.
- [ ] **Per-Sample Confidence Scores** — Max probability or ensemble agreement per sample. Trigger: for case study analysis (identify low-confidence samples).
- [ ] **Disagreement Analysis (Basic)** — Disagreement rate per sample, mean disagreement per class. Trigger: if ensemble accuracy gain >2% (shows models fail differently).
- [ ] **Model Diversity Metrics** — Pairwise accuracy agreement, Kappa statistics. Trigger: to explain why ensemble works.
- [ ] **Markdown Report Generation** — Auto-generate thesis-ready analysis sections. Trigger: for draft chapter writeup.

**Rationale:** These features explain WHY ensemble works and quantify prediction reliability. Add value for thesis discussion section.

**Estimated Effort:** 3-4 days development.

### Future Consideration (v2.0+ - Deep Analysis)

Features to defer until core results validated and thesis outline clear.

- [ ] **Stratified Analysis by Image Characteristics** — Ensemble benefit by warping quality, landmark error bins, class difficulty. Why defer: requires significant metadata infrastructure, analysis time. Add if reviewers request deeper subgroup analysis.
- [ ] **Advanced TTA (Beyond Flip)** — Rotation ±5°, brightness ±10%, contrast ±10%. Why defer: flip TTA likely sufficient (landmark results suggest ~0.1-0.2 px improvement). Add if flip TTA shows no benefit for classification.
- [ ] **Visualization: Agreement Heatmaps** — Spatial agreement analysis with GradCAM overlay. Why defer: high complexity, requires GradCAM integration, unclear thesis value vs. effort. Add if case study analysis needs visual explanation of disagreement.
- [ ] **Uncertainty-Based Sample Selection** — Identify high-uncertainty cases for expert review. Why defer: research contribution, not baseline. Add if extending thesis to human-AI collaboration chapter.
- [ ] **Weighted Voting (ECE-Based Weights)** — Weight models by calibration quality, not just accuracy. Why defer: adds complexity, needs validation split for tuning. Add only if soft voting underperforms and clear heterogeneity in model calibration.
- [ ] **Per-Landmark Error Correlation** — Does ensemble classification benefit correlate with landmark prediction quality? Why defer: requires cross-referencing landmark errors, complex analysis. Add if connecting geometric normalization quality to classification performance.

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Soft Voting | HIGH | LOW | P1 |
| Hard Voting | HIGH | LOW | P1 |
| Per-Model Metrics | HIGH | LOW | P1 |
| Ensemble Metrics | HIGH | LOW | P1 |
| Per-Class Breakdown | HIGH | MEDIUM | P1 |
| TTA (Flip) | HIGH | LOW | P1 |
| Config Loading | HIGH | LOW | P1 |
| Confusion Matrix Viz | HIGH | LOW | P1 |
| Reproducibility | HIGH | MEDIUM | P1 |
| Temperature Scaling | MEDIUM | MEDIUM | P2 |
| ECE Metric | MEDIUM | MEDIUM | P2 |
| Disagreement Analysis (Basic) | MEDIUM | MEDIUM | P2 |
| Model Diversity Metrics | MEDIUM | MEDIUM | P2 |
| Per-Sample Confidence | MEDIUM | LOW | P2 |
| Markdown Report | MEDIUM | LOW | P2 |
| Stratified Analysis | MEDIUM | HIGH | P3 |
| Advanced TTA | LOW | HIGH | P3 |
| Agreement Heatmaps | LOW | HIGH | P3 |
| Weighted Voting (ECE) | LOW | MEDIUM | P3 |
| Uncertainty Routing | LOW | HIGH | P3 |

**Priority key:**
- P1: Must have for launch (v1.0 - Thesis Baseline)
- P2: Should have, add when possible (v1.1 - Confidence Analysis)
- P3: Nice to have, future consideration (v2.0+ - Deep Analysis)

---

## Competitor Feature Analysis

| Feature | Landmark Ensemble (Current) | Classification Ensemble (Proposed) | Notes |
|---------|-------------------------------|-------------------------------------|-------|
| Model Aggregation | Average coordinates (30D continuous) | Soft/hard voting (3-class categorical) | Classification needs probability handling |
| TTA Support | Flip + symmetric pair swapping | Flip (no coordinate logic needed) | Simpler for classification |
| Config-Based | Yes (`ensemble_best.json`) | Yes (follow same pattern) | Consistency is key |
| Calibration | No (regression task) | Yes (classification needs calibration) | Classification probabilities must be calibrated |
| Per-Class Metrics | By category (COVID/Normal/VP) | By class (same categories) | Direct parallel |
| Disagreement Analysis | Per-landmark error variance | Per-sample prediction disagreement | Different domains, same concept |
| CLI Integration | `evaluate-ensemble` command | `evaluate-classifier-ensemble` | Parallel command structure |
| Metadata Tracking | Yes (image paths, categories) | Yes (reuse existing) | Shared infrastructure |

**Key Insight:** Landmark ensemble provides proven template. Classification ensemble should follow same patterns (config-driven, TTA support, CLI integration) but add classification-specific features (calibration, soft/hard voting, per-class analysis).

---

## Implementation Notes

### Medical Imaging Safety Considerations

1. **Test Set Integrity:** NEVER optimize on test set. Use validation split for temperature scaling, all hyperparameter tuning.
2. **Augmentation Safety:** Medical radiographs have clinical meaning in orientation, contrast. Avoid: extreme rotations (>15°), color inversion, aggressive crops that remove anatomy.
3. **Class Imbalance:** COVID-19_Radiography_Dataset has class imbalance. Always report per-class metrics, not just overall accuracy.
4. **Clinical Interpretability:** Provide confidence scores, disagreement flags. Black-box ensemble without uncertainty quantification is not clinically deployable.
5. **Reproducibility:** Thesis research requires exact reproduction. Log seeds, model versions, config hashes.

### Training vs. Evaluation Distinction

**CRITICAL:** This milestone is EVALUATION-ONLY. We are NOT training new models. We have 5 ResNet-18 models from cross-validation (warped_lung_best dataset). Goal: combine their predictions intelligently.

**Avoid:**
- Model retraining, fine-tuning, architecture changes
- Joint ensemble training, distillation, stacking (requires training)
- Hyperparameter optimization that requires retraining models

**Focus:**
- Inference-time aggregation (soft/hard voting)
- Post-hoc calibration (no model weights change)
- Analysis of existing predictions (metrics, visualization)

### Existing Codebase Leverage

**Reuse from landmark ensemble:**
- `src_v2/evaluation/metrics.py::predict_with_tta()` — TTA flip logic
- `src_v2/evaluation/metrics.py::compute_error_per_category()` — Per-category aggregation pattern
- `src_v2/constants.py::SYMMETRIC_PAIRS` — Flip symmetry (if needed for landmark-aware analysis)
- `configs/ensemble_best.json` — Config pattern for model lists

**Extend for classification:**
- Add `evaluate_classifier_ensemble()` in `src_v2/evaluation/metrics.py`
- Add temperature scaling in new `src_v2/evaluation/calibration.py`
- Add disagreement metrics in `src_v2/evaluation/ensemble_analysis.py`
- Add CLI command `evaluate-classifier-ensemble` in `src_v2/cli.py`

---

## Sources

### Ensemble Methods Research:
- [An ensemble approach for classification of tympanic membrane conditions using soft voting classifier](https://link.springer.com/article/10.1007/s11042-024-18631-z) (2024)
- [Classifier Ensemble for Efficient Uncertainty Calibration](https://arxiv.org/html/2501.10089v1) (2025)
- [Early Detection of Retinopathy of Prematurity Using Voting Classifier-Based Ensemble Deep Learning Models](https://link.springer.com/article/10.1007/s44196-025-00847-y) (2025)
- [Implementation of Ensemble Machine Learning with Voting Classifier for Tuberculosis Detection](https://jeeemi.org/index.php/jeeemi/article/view/472) (2024)

### Test-Time Augmentation Research:
- [A Large Scale Benchmark for Test Time Adaptation Methods in Medical Image Segmentation](https://arxiv.org/html/2512.02497v1) (2024)
- [BayTTA: Uncertainty-aware medical image classification with optimized test-time augmentation using Bayesian model averaging](https://ui.adsabs.harvard.edu/abs/2024arXiv240617640S/abstract) (2024)
- [Test-Time Generative Augmentation for Medical Image Segmentation](https://arxiv.org/html/2406.17608v1) (2024)
- [Improving Tuberculosis Detection in Chest X-Ray Images Through Transfer Learning](https://xmed.jmir.org/2025/1/e66029) (2025)

### Explainability and Visualization:
- [Development of an ensemble CNN model with explainable AI for gastrointestinal cancer classification](https://pubmed.ncbi.nlm.nih.gov/38917159/) (2024)
- [Personalized health monitoring using explainable AI](https://www.nature.com/articles/s41598-025-15867-z) (2025)
- [Meta-Learning-Based Ensemble Model for Explainable Alzheimer's Disease Diagnosis](https://pmc.ncbi.nlm.nih.gov/articles/PMC12248535/) (2025)
- [Towards Transparent Diabetes Prediction: Combining AutoML and Explainable AI](https://www.mdpi.com/2078-2489/16/1/7) (2024)

### Confidence Calibration Research:
- [Calibration techniques for node classification using graph neural networks on medical image data](https://proceedings.mlr.press/v227/vos24a.html) (2024)
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) (2017, foundational)
- [Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation](https://www.researchgate.net/publication/342616648_Confidence_Calibration_and_Predictive_Uncertainty_Estimation_for_Deep_Medical_Image_Segmentation)

### Disagreement Analysis Research:
- [AI-clinician collaboration via disagreement prediction](https://pmc.ncbi.nlm.nih.gov/articles/PMC10591030/) (2023)
- [Human–AI collectives most accurately diagnose clinical vignettes](https://www.pnas.org/doi/10.1073/pnas.2426153122) (2025)
- [Data-driven framework for identifying patient subgroups where AI may underperform](https://www.nature.com/articles/s41746-024-01275-6) (2024)
- [Study reveals why AI models that analyze medical images can be biased](https://news.mit.edu/2024/study-reveals-why-ai-analyzed-medical-images-can-be-biased-0628) (2024)

---

*Feature research for: COVID-19 Classification Ensemble + TTA*
*Researched: 2026-01-27*
*Researcher: GSD Project Research Agent*
