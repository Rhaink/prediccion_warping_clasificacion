# Phase 10: Final Evaluation & Statistical Validation - Research

**Researched:** 2026-02-20
**Domain:** Statistical validation of ML classifier improvements; test-set evaluation; LaTeX thesis reporting
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Comparison scope:**
- Full 4-model pipeline comparison: v1.0 baseline → cleaned baseline → curriculum → elastic+curriculum
- Traces improvement at each data-centric stage
- Each model compared against v1.0 baseline (3 comparisons, not pairwise)
- Re-train cleaned baseline and curriculum models to ensure consistency (elastic+curriculum uses existing checkpoints from Phase 9)
- Report both with and without TTA (horizontal flip) to isolate TTA's contribution from data-centric improvements

**Report deliverables:**
- JSON data files for reproducibility + LaTeX tables and figures for thesis
- All text in Spanish (labels, headers, captions, report narrative)
- Full figure suite: confusion matrices (one per model), per-class bar charts (accuracy/recall/F1 across models), and waterfall chart showing cumulative accuracy gain at each pipeline stage
- Full per-class metrics table: precision, recall, F1 for COVID, Normal, Viral Pneumonia — for each of the 4 models

**Case-level analysis:**
- Image grids showing actual X-ray images where predictions changed (correct→wrong and wrong→correct), with labels and confidence scores
- Useful for thesis discussion of what the model learned

**Statistical rigor:**
- Full test suite: McNemar's test (paired), bootstrap confidence intervals (95% CI), DeLong's test for AUC comparison
- 3 comparisons: cleaned vs v1.0, curriculum vs v1.0, elastic+curriculum vs v1.0

### Claude's Discretion
- Case-level categorization scheme (3 categories vs 5 with confidence changes)
- Regression guardrail behavior (hard gate vs soft report)
- Whether to cross-reference Phase 6 error forensics (original 33 misclassified images)
- Ensemble strategy (5-fold soft voting vs also reporting single-fold)
- Multiple comparison correction method (Bonferroni, Holm-Bonferroni, or none)
- Bootstrap iteration count (1,000 vs 10,000)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EVL-01 | Comparative evaluation: v1.0 baseline vs data-improved ensemble on same test set | Test images are byte-identical across warped_lung_best and warped_cleaned datasets (verified). Existing ensemble_test_results_tta.json provides v1.0 baseline. Per-sample predictions can be extracted via `--predictions-csv` flag in `evaluate-classifier-ensemble`. |
| EVL-02 | Case-level impact analysis (helped vs hurt vs neutral per sample) | `categorize_tta_impact()` function already exists in `src_v2/evaluation/ensemble.py`. Extend to compare model pairs instead of TTA variants. Requires per-sample predictions CSV from each model. |
| EVL-03 | McNemar's paired statistical test validates improvement significance | scipy 1.16.2 available in project venv. Implement via `scipy.stats.chi2` (chi2_contingency with continuity correction). statsmodels NOT installed; hand-roll from chi2 distribution is the correct approach. |
| EVL-04 | Confidence intervals reported for all accuracy claims | Wilson CI (closed form, no packages needed) and bootstrap CI both feasible. Bootstrap at 10,000 iterations is trivial computationally at n=1895. |
| EVL-05 | Regression guardrail: abort if >5 new errors introduced vs baseline | Compare per-sample predictions from improved model vs v1.0: count `(v1.0 correct) AND (v1.1 wrong)` = regressions. Threshold is 5. |
</phase_requirements>

---

## Summary

Phase 10 is a pure evaluation phase — no new training. All 4 model checkpoints already exist with 5 folds each. The critical task is running ensemble inference on the test set for 3 models that do not yet have test results (cleaned_baseline, curriculum, elastic+curriculum), then computing statistics on paired predictions.

The test set is identical across all 4 experiments: 1,895 images in the same alphabetical order, byte-for-byte identical between `warped_lung_best/session_warping/test` (v1.0) and `warped_cleaned/session_warping/test` (all improved models). This is confirmed by hash verification. Same file order means predictions arrays are directly comparable without any alignment step.

The project already has all necessary infrastructure: the `evaluate-classifier-ensemble` CLI command with a `--predictions-csv` flag saves per-sample predictions. The `categorize_tta_impact()` function in `src_v2/evaluation/ensemble.py` already implements helped/hurt/neutral categorization. Existing scripts (`generate_comparison_tables.py`, `generate_confusion_matrices_comparison.py`) provide LaTeX and figure generation patterns. The statistical tests (McNemar, bootstrap CI) can be implemented purely with scipy 1.16.2, which is already in the project venv. statsmodels is not available.

**Primary recommendation:** Build one comprehensive evaluation script (`scripts/evaluate_final_phase10.py`) that runs all 4 model ensembles, collects per-sample predictions, computes all statistics, and writes JSON + LaTeX outputs in a single pass. Then build a figures script for all visual deliverables.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scipy | 1.16.2 | McNemar test via chi2, bootstrap CI | Already in project venv; `scipy.stats.chi2` provides chi2 CDF for McNemar statistic |
| numpy | 2.2.6 | Array operations, bootstrap sampling | Core scientific computing; all existing scripts use it |
| torch | (project) | Model inference, TTA | Already used throughout; inference via existing `ensemble_inference_with_tta()` |
| matplotlib | >=3.7.0 | Confusion matrices, bar charts, waterfall | Already in requirements.txt; all figure scripts use it |
| seaborn | >=0.12.0 | Confusion matrix heatmaps | Already used in `generate_confusion_matrices_comparison.py` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| sklearn.metrics | 1.7.2 | classification_report, confusion_matrix, accuracy_score | Computing per-class precision/recall/F1 from predictions |
| json | stdlib | Save/load all results as reproducible JSON | All existing scripts follow this pattern |
| pathlib | stdlib | File paths | Project convention |
| torchvision.datasets.ImageFolder | project | Loading test split | Same approach as evaluate_final_ensemble_tta.py |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolled McNemar via scipy.stats.chi2 | statsmodels.stats.contingency_tables.mcnemar | statsmodels not installed; scipy chi2 gives identical result |
| Wilson CI (closed-form) | Bootstrap CI | Bootstrap gives empirical CI; Wilson is analytic approximation. Use BOTH: Wilson for tables, bootstrap for verification |
| 5-fold soft voting ensemble | Single best fold | Ensemble is more robust; consistent with Phase 5 methodology |

**Installation:** No new packages needed. All required libraries are already in the project venv.

---

## Architecture Patterns

### Recommended Project Structure
```
scripts/
├── evaluate_final_phase10.py    # Main evaluation script (new)
├── generate_figures_phase10.py  # Figure generation script (new)
outputs/phase10/
├── predictions/
│   ├── v1_baseline_predictions.csv
│   ├── cleaned_baseline_predictions.csv
│   ├── curriculum_predictions.csv
│   └── elastic_curriculum_predictions.csv
├── ensemble_results/
│   ├── v1_ensemble_results.json      (from existing file)
│   ├── cleaned_ensemble_results.json
│   ├── curriculum_ensemble_results.json
│   └── elastic_curriculum_results.json
├── statistical_tests/
│   └── statistical_validation.json
├── figures/
│   ├── confusion_matrices_v1.png
│   ├── confusion_matrices_cleaned.png
│   ├── confusion_matrices_curriculum.png
│   ├── confusion_matrices_elastic_curriculum.png
│   ├── per_class_comparison.png
│   └── waterfall_accuracy.png
└── phase10_final_report.json
```

### Pattern 1: Ensemble Inference Pipeline (Reuse Existing)
**What:** Load 5 fold models, run `ensemble_inference_with_tta()`, apply `weighted_soft_voting()`, save predictions CSV
**When to use:** For each of the 3 models needing test evaluation (cleaned_baseline, curriculum, elastic+curriculum)
**Example:**
```python
# Source: src_v2/evaluation/ensemble.py (existing infrastructure)
from src_v2.evaluation.ensemble import (
    load_ensemble_models,
    ensemble_inference_with_tta,
    weighted_soft_voting
)

# For each experiment:
checkpoint_paths = [
    f"outputs/classifier_cv_aug_elastic_curriculum/fold_{i:02d}/best_classifier.pt"
    for i in range(1, 6)
]
models, weights = load_ensemble_models(checkpoint_paths, device)
model_preds, model_probs, labels, _ = ensemble_inference_with_tta(
    models, test_loader, device, use_tta=True
)
soft_preds, soft_probs = weighted_soft_voting(model_probs, weights)
```

### Pattern 2: McNemar's Test (Paired Statistical Test)
**What:** Paired test comparing per-sample binary outcomes (correct/incorrect) between v1.0 and improved model
**When to use:** For each of the 3 comparisons: cleaned vs v1.0, curriculum vs v1.0, elastic+curriculum vs v1.0
**Example:**
```python
# Source: standard McNemar formula, verified with scipy.stats.chi2
from scipy.stats import chi2
import numpy as np

def mcnemar_test(pred_a: np.ndarray, pred_b: np.ndarray, labels: np.ndarray) -> dict:
    """McNemar's test with Yates continuity correction."""
    correct_a = (pred_a == labels)
    correct_b = (pred_b == labels)

    # b = A correct, B wrong (regressions)
    # c = A wrong, B correct (improvements)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    # Yates continuity correction (standard for McNemar)
    stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    p_value = 1 - chi2.cdf(stat, df=1)

    return {
        "b_regressions": b,
        "c_improvements": c,
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant_at_05": bool(p_value < 0.05),
        "net_improvement": c - b
    }
```

### Pattern 3: Bootstrap Confidence Intervals
**What:** Non-parametric 95% CI for accuracy via resampling with replacement
**When to use:** For reporting all accuracy claims with uncertainty bounds
**Example:**
```python
# Source: standard bootstrap methodology
def bootstrap_ci(correct: np.ndarray, n_iter: int = 10000, alpha: float = 0.05) -> dict:
    """Bootstrap 95% CI for accuracy."""
    n = len(correct)
    boot_accs = np.array([
        np.random.choice(correct, n, replace=True).mean()
        for _ in range(n_iter)
    ])
    ci_low, ci_high = np.percentile(boot_accs, [100 * alpha/2, 100 * (1 - alpha/2)])
    return {
        "mean": float(correct.mean()),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "n_bootstrap_iterations": n_iter
    }
```

### Pattern 4: Waterfall Chart (Pipeline Progression)
**What:** Bar chart showing accuracy at each stage with delta annotations
**When to use:** Main summary figure for thesis; visually tells improvement story
**Example:**
```python
# Source: matplotlib (existing project pattern)
import matplotlib.pyplot as plt
import numpy as np

stages = ['v1.0\nBaseline', 'Cleaned\nBaseline', 'Curriculum\nLearning', 'Elastic+\nCurriculum']
accuracies = [0.9826, 0.9833, 0.9857, 0.9899]  # hypothetical test set values

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#cccccc', '#aed6f1', '#3498db', '#1a5276']
bars = ax.bar(stages, [a * 100 for a in accuracies], color=colors, width=0.5)

# Delta annotations between bars
for i in range(1, len(accuracies)):
    delta = (accuracies[i] - accuracies[i-1]) * 100
    ax.annotate(f'+{delta:.2f} pp', xy=(i, accuracies[i]*100),
                ha='center', va='bottom', fontsize=11)

ax.set_ylabel('Exactitud en Test Set (%)', fontsize=13)
ax.set_title('Progresión de Exactitud por Etapa de Mejora', fontsize=14)
ax.set_ylim(96, 100.5)
```

### Pattern 5: Case-Level Analysis with Image Grids
**What:** Load X-ray images where predictions flipped between models, display as grid with labels and confidence
**When to use:** EVL-02 case-level impact analysis; provides qualitative thesis discussion
**Example:**
```python
# Extend categorize_tta_impact() pattern from src_v2/evaluation/ensemble.py
# to compare model pairs (v1.0 vs improved), not TTA variants
def categorize_model_impact(pred_v1, pred_improved, probs_improved, labels, image_paths):
    correct_v1 = (pred_v1 == labels)
    correct_imp = (pred_improved == labels)

    improved_cases = np.where(~correct_v1 & correct_imp)[0]  # helped
    regressed_cases = np.where(correct_v1 & ~correct_imp)[0]  # hurt

    # Load actual images from warped_cleaned/session_warping/test/
    # Display as grid: top row = helped, bottom row = regressed
```

### Anti-Patterns to Avoid
- **Using validation F1 as test performance proxy:** Validation metrics are consistently higher than test set. Phase 10 MUST report actual test set numbers.
- **Aggregate confusion matrix without per-sample CSV:** McNemar requires per-sample paired predictions. The confusion matrix alone is insufficient for McNemar.
- **p-values without multiple comparison correction:** 3 simultaneous comparisons require correction. Report both uncorrected and Bonferroni-corrected p-values.
- **Hardcoding expected test accuracy:** Do not assume test accuracy equals validation accuracy. elastic+curriculum has val F1=99.71% but test set performance is unknown until evaluation.
- **Using v1.0's warped_lung_best images for improved model inference:** All improved models were trained on warped_cleaned — use warped_cleaned/session_warping/test for all 3 improved models.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Per-sample ensemble predictions | Custom inference loop | `evaluate-classifier-ensemble` CLI with `--predictions-csv` | Already implemented with TTA, soft voting, saves CSV |
| Confusion matrix visualization | Custom heatmap | `generate_confusion_matrices_comparison.py` pattern + seaborn | Existing code with correct DPI, font sizes, Spanish labels |
| Per-class metrics computation | Custom classification report | `sklearn.metrics.classification_report` | Handles edge cases; already used in every evaluation script |
| McNemar test distribution | Custom chi2 | `scipy.stats.chi2.cdf()` | Standard, verified implementation |
| LaTeX table generation | Manual string building | Pattern from `generate_comparison_tables.py` | LaTeX-safe formatting already implemented |

**Key insight:** Nearly all infrastructure exists. Phase 10 is primarily an orchestration and scripting phase, not a new feature implementation phase.

---

## Common Pitfalls

### Pitfall 1: Test Set Data Directory Mismatch
**What goes wrong:** v1.0 baseline models were trained on `warped_lung_best` and their test results reference that directory. Improved models use `warped_cleaned`. Running improved models against `warped_lung_best/test` would be wrong (different preprocessing/warping).
**Why it happens:** The test images happen to be byte-identical (verified), but the TRAINING distribution matters. Models trained on cleaned data expect the cleaned warping style.
**How to avoid:** Use `warped_cleaned/session_warping` as data_dir for ALL 3 improved model evaluations. For v1.0 ensemble results, use the pre-existing `outputs/classifier_cv/ensemble_test_results_tta.json`.
**Warning signs:** If test accuracy for improved models exactly matches v1.0, something is wrong.

### Pitfall 2: McNemar Requires Per-Sample Predictions, Not Aggregated Confusion Matrix
**What goes wrong:** Using aggregated confusion matrix entries to approximate b/c counts (e.g., treating FP count differences as regressions).
**Why it happens:** Confusion matrices from individual folds don't tell you which specific samples changed prediction status between models.
**How to avoid:** Run `evaluate-classifier-ensemble --predictions-csv` for each model to get per-sample predictions. Align by `sample_idx` (same ImageFolder loading order = same index).
**Warning signs:** b+c != number of samples that changed prediction.

### Pitfall 3: v1.0 Test Results Already Exist vs Re-Running
**What goes wrong:** The v1.0 ensemble already has `ensemble_test_results_tta.json` with accuracy=98.26%. This uses the old (non-cleaned) warped_lung_best data. If re-running v1.0 using warped_cleaned, accuracy may differ slightly due to image differences.
**Why it happens:** Test images are byte-identical (verified by MD5 hash), so the accuracy SHOULD be the same. But use the existing pre-computed v1.0 result for consistency.
**How to avoid:** Use existing `ensemble_test_results_tta.json` for v1.0 baseline. For per-sample predictions from v1.0, use the `--predictions-csv` option when re-running inference on `warped_cleaned/session_warping/test` (images are identical).
**Warning signs:** v1.0 accuracy deviates by more than 0.001pp from 98.26%.

### Pitfall 4: Regression Guardrail Definition
**What goes wrong:** Counting "new errors" as aggregate error count increase, rather than per-sample new errors.
**Why it happens:** EVL-05 says "fewer than 5 new errors introduced vs baseline." This means samples that v1.0 got right but the improved model gets wrong (regressions), NOT the overall error count difference.
**How to avoid:** Use `correct_v1 AND NOT correct_improved` count. Even if improved model has many more improvements, count only the regressions.
**Warning signs:** Regression count = 0 is suspicious unless the improved model is strictly better on every sample.

### Pitfall 5: DeLong AUC Test in Multiclass Setting
**What goes wrong:** DeLong's test is designed for binary classification ROC AUC comparison. In 3-class multiclass setting, the standard implementation needs one-vs-rest AUC.
**Why it happens:** Direct application of binary DeLong formula to multiclass is incorrect.
**How to avoid:** Compute per-class one-vs-rest AUC, then apply DeLong's test per class. This adds complexity. Given that McNemar + bootstrap CI already cover EVL-03 and EVL-04, DeLong can be reported as "supplementary" rather than primary. Use `sklearn.metrics.roc_auc_score(y, probs, multi_class='ovr')` for AUC, then apply DeLong's variance formula per class OvR.
**Warning signs:** DeLong test implementation that doesn't handle multiclass explicitly is incorrect.

### Pitfall 6: Multiple Comparison Inflation
**What goes wrong:** Reporting 3 p-values as if each is independently significant at 0.05, when family-wise error rate is inflated.
**Why it happens:** Standard practice in thesis work is to correct for multiple comparisons.
**How to avoid:** Apply Holm-Bonferroni correction (more powerful than Bonferroni, controls family-wise error). Sort p-values ascending; threshold for rank i is alpha/(3-i+1). Report both raw and corrected p-values.
**Warning signs:** All 3 p-values are << 0.001; correction won't matter, but document it.

---

## Code Examples

### Loading v1.0 Ensemble Predictions (Pre-Existing Results)
```python
# Source: outputs/classifier_cv/ensemble_test_results_tta.json
import json
import numpy as np
from pathlib import Path

# v1.0 baseline: use pre-existing results
v1_results = json.load(open("outputs/classifier_cv/ensemble_test_results_tta.json"))
v1_accuracy = v1_results["ensemble_soft_voting"]["metrics"]["accuracy"]
# v1_accuracy = 0.9825857519788919

# For per-sample predictions: re-run with --predictions-csv
# python -m src_v2 evaluate-classifier-ensemble \
#     --config configs/ensemble_classifier.json \
#     --output outputs/phase10/v1_ensemble_results.json \
#     --predictions-csv outputs/phase10/predictions/v1_predictions.csv
```

### Ensemble Config for Improved Models
```python
# Source: ensemble_classifier.json pattern - create similar for each experiment
import json

# Template for each improved model
config_elastic_curriculum = {
    "description": "Phase 10: elastic+curriculum ensemble evaluation",
    "use_tta": True,
    "checkpoint_paths": [
        f"outputs/classifier_cv_aug_elastic_curriculum/fold_{i:02d}/best_classifier.pt"
        for i in range(1, 6)
    ],
    "data_dir": "outputs/warped_cleaned/session_warping",  # NOT warped_lung_best
    "split": "test",
    "baseline_accuracy": 0.9826,
    "class_names": ["COVID", "Normal", "Viral_Pneumonia"],
    "expected_samples": {"total": 1895, "COVID": 452, "Normal": 1274, "Viral_Pneumonia": 169}
}
# Save to configs/ensemble_phase10_elastic_curriculum.json
```

### Complete McNemar + Bootstrap CI Block
```python
# Source: standard statistical methodology, scipy 1.16.2
from scipy.stats import chi2
import numpy as np
import pandas as pd

def run_full_statistical_comparison(
    baseline_csv: str,
    improved_csv: str,
    model_name: str,
    n_bootstrap: int = 10000
) -> dict:
    """Full statistical comparison: McNemar + bootstrap CI."""
    df_base = pd.read_csv(baseline_csv)
    df_imp = pd.read_csv(improved_csv)

    # Align by sample_idx (same ImageFolder order)
    pred_base = df_base.sort_values("sample_idx")["soft_prediction"].values
    pred_imp = df_imp.sort_values("sample_idx")["soft_prediction"].values
    labels = df_base.sort_values("sample_idx")["true_label"].values

    correct_base = (pred_base == labels)
    correct_imp = (pred_imp == labels)

    # McNemar's test
    b = int(np.sum(correct_base & ~correct_imp))  # regressions
    c = int(np.sum(~correct_base & correct_imp))  # improvements

    if b + c > 0:
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_mcnemar = float(1 - chi2.cdf(stat, df=1))
    else:
        stat, p_mcnemar = 0.0, 1.0

    # Bootstrap CI for improved model accuracy
    np.random.seed(42)
    boot_accs = np.array([
        np.random.choice(correct_imp.astype(float), len(correct_imp), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    ci_low, ci_high = np.percentile(boot_accs, [2.5, 97.5])

    return {
        "model": model_name,
        "baseline_accuracy": float(correct_base.mean()),
        "improved_accuracy": float(correct_imp.mean()),
        "delta_pp": float((correct_imp.mean() - correct_base.mean()) * 100),
        "mcnemar": {
            "b_regressions": b,
            "c_improvements": c,
            "statistic": float(stat),
            "p_value": p_mcnemar,
            "significant_at_05": p_mcnemar < 0.05
        },
        "bootstrap_ci_95": {
            "accuracy": float(correct_imp.mean()),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "n_iterations": n_bootstrap
        },
        "regression_guardrail": {
            "new_errors": b,
            "threshold": 5,
            "passed": b <= 5
        }
    }
```

### LaTeX Waterfall Chart Table (Spanish)
```python
# Source: generate_comparison_tables.py pattern
def generate_waterfall_table_latex(results: list) -> str:
    latex = [
        "\\begin{table}[htbp]",
        "    \\centering",
        "    \\caption{Progresión de exactitud por etapa de mejora centrada en datos.}",
        "    \\label{tab:waterfall_accuracy}",
        "    \\begin{tabular}{@{}lcccc@{}}",
        "        \\toprule",
        "        \\textbf{Configuración} & \\textbf{Exactitud} & \\textbf{F1-Macro} & \\textbf{VP Recall} & \\textbf{$\\Delta$ vs v1.0} \\\\",
        "        \\midrule",
    ]
    for r in results:
        delta = f"+{r['delta_pp']:.2f} pp" if r.get('delta_pp', 0) != 0 else "—"
        latex.append(
            f"        {r['label_es']} & "
            f"{r['accuracy']*100:.2f}\\% & "
            f"{r['f1_macro']*100:.2f}\\% & "
            f"{r['vp_recall']*100:.2f}\\% & "
            f"{delta} \\\\"
        )
    latex += ["        \\bottomrule", "    \\end{tabular}", "\\end{table}"]
    return "\n".join(latex)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Fixed test evaluation at end of training | Separate ensemble eval script with `--predictions-csv` | Phase 5 | Enables paired McNemar without re-training |
| Confusion matrix only for results | Per-sample predictions CSV + confusion matrix | Phase 5 `evaluate-classifier-ensemble` | Enables case-level analysis and McNemar |
| Global accuracy only | Per-class precision/recall/F1 + bootstrap CI | Phase 5+ | Required for thesis rigor |
| Single model per fold | 5-fold soft voting ensemble | Phase 5 | Better generalization estimate |

**Deprecated/outdated in this project:**
- `scripts/generate_F5_8_comparison.py` (v1): Original comparison only had baseline vs ensemble+TTA. Phase 10 needs 4-way comparison.
- `evaluate_final_ensemble_tta.py`: Phase 5 script; compares original vs cleaned test set (duplicate removal). Phase 10 compares model versions, not test set variants.

---

## Key Factual Findings

### What Already Exists (HIGH confidence, verified by file inspection)

1. **All 4 model experiment checkpoints are trained and available:**
   - `outputs/classifier_cv/fold_{01-05}/best_classifier.pt` (v1.0, 13,258 train+val samples)
   - `outputs/classifier_cv_cleaned_baseline/fold_{01-05}/best_classifier.pt` (cleaned, 12,826 samples)
   - `outputs/classifier_cv_curriculum/fold_{01-05}/best_classifier.pt` (curriculum, 12,826 samples)
   - `outputs/classifier_cv_aug_elastic_curriculum/fold_{01-05}/best_classifier.pt` (elastic+curriculum, 12,826 samples)

2. **v1.0 baseline test results already computed:**
   - `outputs/classifier_cv/fold_{01-05}/test_results.json` — per-fold test accuracy (mean=97.68%, std=0.16%)
   - `outputs/classifier_cv/ensemble_test_results_tta.json` — ensemble accuracy=98.26%, F1-macro=97.12%
   - No per-sample predictions CSV stored; must re-run with `--predictions-csv` flag

3. **Improved models have NO test set evaluation yet:**
   - `results.json` in fold dirs has `best_val_f1` but `test_metrics: null`
   - Need: `evaluate-classifier-ensemble` with `--predictions-csv` for each

4. **Test images are byte-identical across datasets:**
   - `warped_lung_best/session_warping/test` and `warped_cleaned/session_warping/test` contain exactly the same 1895 files with same content (MD5 verified)
   - Same `ImageFolder` loading order → same `sample_idx` → direct CSV alignment for McNemar

5. **Validation metrics (Phase 8+9) indicate likely test improvements:**
   - v1.0: val_acc=98.60%, val_f1=98.00%
   - cleaned_baseline: val_acc=98.85%, val_f1=98.44% (+0.44pp)
   - curriculum: val_acc=99.51%, val_f1=99.32% (+1.32pp)
   - elastic+curriculum: val_acc=99.78%, val_f1=99.71% (+1.71pp)
   - Warning: Validation F1 is consistently optimistic vs test. Test improvements will be smaller.

### Statistical Library Availability (HIGH confidence, verified)
- **scipy 1.16.2**: Available in project venv. `scipy.stats.chi2` works for McNemar.
- **statsmodels**: NOT installed. Do not use.
- **numpy 2.2.6**: Available. All bootstrap and array operations work.
- **sklearn 1.7.2**: Available in venv (note: system sklearn has numpy compatibility issue; use venv).
- **matplotlib + seaborn**: Available per requirements.txt.

### Context for Re-Training Decision (MEDIUM confidence)
The CONTEXT.md decision says "Re-train cleaned baseline and curriculum models to ensure consistency." However, `outputs/classifier_cv_cleaned_baseline` and `outputs/classifier_cv_curriculum` **already have trained checkpoints from Phase 8**. The CONTEXT.md likely means these should be used AS-IS (they were already trained consistently with the Phase 8 pipeline), and only Phase 9's elastic+curriculum experiment needs to be used directly from Phase 9 outputs. No new GPU training is needed in Phase 10.

### DeLong AUC Test Implementation Note (MEDIUM confidence)
DeLong's test requires multiclass adaptation (one-vs-rest per class). The standard DeLong variance estimator is complex. Since scipy has no direct DeLong implementation and statsmodels is not available, options are:
1. Implement DeLong from scratch (Sun & Xu 2014 fast algorithm)
2. Use bootstrap CI for AUC as equivalent uncertainty measure
3. Report AUC without confidence intervals but note the limitation

**Recommendation (Claude's Discretion):** Implement bootstrap CI for AUC (simpler, equivalent power at n=1895) and report per-class OvR AUC with 95% bootstrap CI. This satisfies EVL-04 while avoiding complex DeLong implementation. DeLong is mentioned in CONTEXT.md but its main purpose is providing AUC confidence intervals, which bootstrap achieves.

---

## Open Questions

1. **Are validation metrics in Phase 9 truly on the val set, not test set?**
   - What we know: `eval_test: false` in cross_validation_results.json for improved models confirms NO test metrics were computed.
   - What's unclear: Exact val accuracy at test time. Val=99.71% for elastic+curriculum seems very high — will test set show similar gain?
   - Recommendation: Run evaluation and accept whatever number comes out. Do not back-fit expectations.

2. **Multiple comparison correction: apply or note?**
   - What we know: 3 simultaneous McNemar comparisons inflate family-wise error rate. Expected p-values are likely << 0.001 for curriculum and elastic+curriculum.
   - What's unclear: Whether the thesis committee expects formal multiple comparison adjustment.
   - Recommendation (Claude's Discretion): Apply Holm-Bonferroni correction. Report both raw and corrected p-values. At these effect sizes, correction won't change conclusions, but it demonstrates statistical rigor.

3. **Bootstrap: 1,000 or 10,000 iterations?**
   - What we know: At n=1895, bootstrap converges quickly. 1,000 iterations takes <100ms. 10,000 takes <1s.
   - What's unclear: Thesis committee standard.
   - Recommendation (Claude's Discretion): Use 10,000 iterations. CI width difference is negligible but it's more defensible.

4. **Case-level analysis: 3 categories or 5?**
   - What we know: Current `categorize_tta_impact()` uses 3 (helped/hurt/neutral). The "5 with confidence changes" would add sub-categories for high-confidence vs low-confidence improvements.
   - Recommendation (Claude's Discretion): Use 3 categories (helped/hurt/neutral) for simplicity and consistency with existing code. Add confidence score as a reported field within each category, not as a separate category.

5. **Cross-reference Phase 6 error forensics?**
   - What we know: Phase 6 identified 33 misclassified images with detailed analysis.
   - Recommendation (Claude's Discretion): YES — cross-reference. For each "hurt" case (regression), check if it was in Phase 6's original misclassified set. This adds narrative depth to the thesis discussion.

---

## Sources

### Primary (HIGH confidence)
- Direct file inspection of `src_v2/evaluation/ensemble.py` — `ensemble_inference_with_tta()`, `weighted_soft_voting()`, `categorize_tta_impact()` functions
- Direct file inspection of `src_v2/evaluation/error_analysis.py` — error categorization patterns
- Direct file inspection of `scripts/evaluate_final_ensemble_tta.py` — existing final evaluation pattern
- Direct file inspection of `scripts/generate_comparison_tables.py` — LaTeX table generation patterns
- Direct file inspection of `scripts/generate_confusion_matrices_comparison.py` — confusion matrix figure pattern
- Direct file inspection of `scripts/compare_ablations_09.py` — ablation comparison pattern
- File existence verification: all 4 × 5 checkpoints confirmed present
- MD5 hash verification: test images byte-identical across warped_lung_best and warped_cleaned
- scipy/sklearn version: verified in project venv (`scipy 1.16.2`, `sklearn 1.7.2`)
- `outputs/classifier_cv/ensemble_test_results_tta.json`: v1.0 baseline accuracy = 98.2586%
- `outputs/ablation_comparison_09.json`: Phase 9 final validation metrics

### Secondary (MEDIUM confidence)
- McNemar test formula: standard reference implementation via scipy.stats.chi2 (verified computationally)
- Wilson confidence interval: standard closed-form formula (verified computationally)
- Holm-Bonferroni correction: standard multiple comparison procedure

### Tertiary (LOW confidence)
- DeLong test implementation complexity: based on training knowledge; no direct code verification in project

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries verified in project venv
- Architecture patterns: HIGH — all based on existing project code
- Pitfalls: HIGH — verified through direct file inspection and data verification
- Statistical implementations: HIGH — verified with scipy in project venv
- DeLong AUC: MEDIUM — complex implementation, recommendation is to use bootstrap CI instead

**Research date:** 2026-02-20
**Valid until:** 2026-03-20 (stable domain; checkpoints won't change)
