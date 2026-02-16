# Phase 4: Analysis & Visualization - Research

**Researched:** 2026-02-16
**Domain:** Scientific visualization (matplotlib/seaborn) + LaTeX table generation + metrics comparison
**Confidence:** HIGH

## Summary

Phase 4 consumes existing evaluation results from Phases 2-3 to generate thesis-ready comparison analysis and confusion matrices. The primary task is producing publication-quality figures and LaTeX tables showing the improvement from baseline individual models (97.68%) to ensemble+TTA (98.26%). The project has established visualization conventions in existing scripts (300 DPI, DejaVu Sans font, Spanish labels, Blues colormap), a complete JSON data pipeline for metrics, and LaTeX integration via booktabs package.

**Primary recommendation:** Extend existing `generate_confusion_matrix_cv.py` and `generate_F5_8_comparison_cv.py` patterns to create new comparison scripts. Use pandas DataFrame.to_latex() for table generation and maintain established visual style (300 DPI, DejaVu Sans, Spanish labels, Blues colormap).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Thesis visual style:**
- All labels in Spanish (axis titles, figure titles, annotations)
- Must match existing Chapter 5 figure style — locate existing visualization scripts or .tex references to determine current style conventions

**Comparison story:**
- Main comparison: baseline (97.68% best individual) vs final ensemble+TTA (98.26%)
- Include both overall metrics AND per-class breakdown (COVID-19, Normal, Neumonía Viral)
- Presentation format: tables only (no bar charts)
- Highlight TTA per-class impact as a finding: COVID benefits most (+0.44% F1), Viral degrades slightly (-0.28% F1)
- Generate LaTeX-ready .tex table file for direct \input{} in thesis

**Confusion matrix format:**
- Two separate confusion matrices: baseline (best individual model) vs final (ensemble+TTA)
- Each as its own figure (separate figures, not subfigures a/b)
- Cell values show both raw counts and normalized percentages (e.g., "450\n99.6%")
- Class labels in Spanish: COVID-19, Normal, Neumonía Viral

**Output deliverables:**
- Figure format: PNG high-resolution (300+ DPI)
- Output directory: outputs/classifier_cv/ (alongside existing ensemble results)
- Save structured comparison_metrics.json with all computed deltas and breakdowns
- Generate LaTeX .tex table file ready for thesis inclusion

### Claude's Discretion

- Exact font family and size for figures
- Color palette choice (Blues, Viridis, or other academic palette)
- Figure dimensions and aspect ratio
- Table column layout and formatting
- Any additional supporting figures that enhance the comparison story

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

## Standard Stack

### Core Visualization Libraries

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| matplotlib | 3.8+ | Base plotting framework | Industry standard for scientific visualization, complete control over figure elements |
| seaborn | 0.13+ | Statistical visualization | High-level interface for confusion matrices, built on matplotlib, excellent colormaps |
| numpy | 1.24+ | Numerical operations | Required for matrix operations, aggregations, percentage calculations |
| pandas | 2.0+ | Data manipulation & LaTeX export | DataFrame.to_latex() is the standard method for scientific table generation |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json | stdlib | Load evaluation results | Reading existing metrics from outputs/classifier_cv/*.json |
| pathlib | stdlib | Path manipulation | Standard Python 3 path handling |
| scikit-learn | 1.4+ | Confusion matrix utilities | sklearn.metrics.confusion_matrix if needed, though project uses custom aggregation |

**Installation:**
```bash
# Already installed in project environment
# Verify with: pip list | grep -E "matplotlib|seaborn|pandas"
```

## Architecture Patterns

### Recommended Project Structure

Based on existing codebase patterns:

```
scripts/
├── generate_confusion_matrix_baseline.py      # Baseline individual model matrix
├── generate_confusion_matrix_ensemble_tta.py  # Final ensemble+TTA matrix
├── generate_comparison_table.py               # LaTeX table generation
└── generate_comparison_metrics.py             # JSON metrics aggregation

outputs/classifier_cv/
├── comparison_metrics.json                    # Structured metrics output
├── comparison_table.tex                       # LaTeX table for thesis
├── confusion_matrix_baseline.png              # 300 DPI figure
└── confusion_matrix_ensemble_tta.png          # 300 DPI figure
```

### Pattern 1: Load Existing Evaluation Results

**What:** Parse pre-computed metrics from Phase 2 (baseline) and Phase 3 (TTA) JSON files.

**When to use:** First step in any comparison script — no re-evaluation, only consume existing data.

**Example:**
```python
# Source: scripts/generate_confusion_matrix_cv.py lines 16-105
import json
from pathlib import Path

def load_baseline_metrics(cv_dir: Path) -> dict:
    """Load individual model metrics from fold test results."""
    fold_metrics = []

    for fold in range(1, 6):
        fold_path = cv_dir / f"fold_{fold:02d}" / "test_results.json"
        with open(fold_path) as f:
            results = json.load(f)
        fold_metrics.append(results["metrics"])

    # Calculate mean and std for baseline
    accuracies = [m["accuracy"] for m in fold_metrics]
    return {
        "accuracy_mean": np.mean(accuracies),
        "accuracy_std": np.std(accuracies),
        # ... per-class metrics
    }

def load_ensemble_tta_metrics(cv_dir: Path) -> dict:
    """Load ensemble+TTA metrics from Phase 3."""
    tta_path = cv_dir / "ensemble_test_results_tta.json"
    with open(tta_path) as f:
        return json.load(f)["ensemble_soft_voting"]
```

### Pattern 2: Spanish Labels Configuration

**What:** Dictionary-based i18n pattern for consistent Spanish labeling.

**When to use:** All visualization scripts — maintains thesis language consistency.

**Example:**
```python
# Source: scripts/generate_confusion_matrix_cv.py lines 256-281
LABELS_ES = {
    "colorbar": "Porcentaje (%)",
    "xlabel": "Predicción",
    "ylabel": "Categoría Real",
    "title_baseline": "Matriz de Confusión - Modelo Individual (Baseline)",
    "title_ensemble": "Matriz de Confusión - Ensemble + TTA (Final)",
    "class_names": {
        "COVID": "COVID-19",
        "Normal": "Normal",
        "Viral_Pneumonia": "Neumonía Viral",
    },
}
```

### Pattern 3: Matplotlib rcParams Configuration

**What:** Global style configuration applied at script start for consistent academic appearance.

**When to use:** Beginning of every visualization script — ensures all figures match thesis style.

**Example:**
```python
# Source: scripts/generate_F5_3_single_panel_fixed.py lines 16-22
# Source: scripts/generate_F5_8_comparison_cv.py lines 21-34
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 18  # Base text
plt.rcParams['axes.labelsize'] = 20  # Axis labels
plt.rcParams['axes.titlesize'] = 22  # Subplot titles
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1
```

### Pattern 4: Confusion Matrix with Dual Annotations

**What:** Display both raw counts and percentages in each cell using seaborn heatmap with custom text annotations.

**When to use:** Confusion matrix generation — provides complete information for readers.

**Example:**
```python
# Source: scripts/generate_confusion_matrix_cv.py lines 108-215
import seaborn as sns

def plot_confusion_matrix(cm: np.ndarray, class_names: list, title: str,
                          output_path: Path, accuracy: float):
    fig, ax = plt.subplots(figsize=(12, 9))

    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create heatmap (no default annotations)
    sns.heatmap(
        cm_percent,
        annot=False,  # We'll add custom annotations
        cmap='Blues',
        cbar_kws={'label': 'Porcentaje (%)'},
        ax=ax,
        vmin=0,
        vmax=100
    )

    # Annotate with counts and percentages
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = cm[i, j]
            percent = cm_percent[i, j]

            # Color and weight
            text_color = 'white' if percent > 50 else 'black'
            weight = 'bold' if i == j else 'normal'

            # Dual format: "value\n(percent%)"
            text = f'{value}\n({percent:.1f}%)'
            ax.text(j + 0.5, i + 0.5, text,
                   ha='center', va='center',
                   color=text_color,
                   fontsize=15,
                   weight=weight)

    # Configure axes
    ax.set_xlabel('Predicción', fontsize=17, fontweight='bold')
    ax.set_ylabel('Categoría Real', fontsize=17, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
```

### Pattern 5: LaTeX Table Generation with Pandas

**What:** Use pandas DataFrame.to_latex() with booktabs formatting for professional tables.

**When to use:** Generating comparison tables for direct inclusion in thesis via \input{}.

**Example:**
```python
# Source: pandas documentation + project needs
import pandas as pd

def generate_comparison_table(baseline: dict, ensemble_tta: dict,
                              output_path: Path):
    """Generate LaTeX table comparing baseline vs ensemble+TTA."""

    # Build DataFrame
    data = {
        'Métrica': [
            'Exactitud (Accuracy)',
            'F1-Score Macro',
            'F1-Score Ponderado'
        ],
        'Baseline Individual': [
            f"{baseline['accuracy']*100:.2f}\\% $\\pm$ {baseline['accuracy_std']*100:.2f}\\%",
            f"{baseline['f1_macro']*100:.2f}\\% $\\pm$ {baseline['f1_macro_std']*100:.2f}\\%",
            f"{baseline['f1_weighted']*100:.2f}\\% $\\pm$ {baseline['f1_weighted_std']*100:.2f}\\%"
        ],
        'Ensemble + TTA': [
            f"{ensemble_tta['accuracy']*100:.2f}\\%",
            f"{ensemble_tta['f1_macro']*100:.2f}\\%",
            f"{ensemble_tta['f1_weighted']*100:.2f}\\%"
        ],
        'Mejora': [
            f"+{(ensemble_tta['accuracy'] - baseline['accuracy'])*100:.2f}pp",
            f"+{(ensemble_tta['f1_macro'] - baseline['f1_macro'])*100:.2f}pp",
            f"+{(ensemble_tta['f1_weighted'] - baseline['f1_weighted'])*100:.2f}pp"
        ]
    }

    df = pd.DataFrame(data)

    # Generate LaTeX with booktabs
    latex_str = df.to_latex(
        index=False,
        escape=False,  # Don't escape LaTeX commands
        column_format='lccc',
        caption='Comparación de rendimiento: Baseline vs Ensemble+TTA',
        label='tab:comparison_ensemble_tta',
        position='htbp'
    )

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_str)

    print(f"LaTeX table saved: {output_path}")
```

### Anti-Patterns to Avoid

- **Hardcoded metrics:** Don't copy numbers from terminal output. Always load from JSON files to prevent transcription errors.
- **Mixing evaluation with visualization:** Don't re-run model evaluation in visualization scripts. Phase 4 consumes pre-computed results only.
- **Inconsistent DPI:** Don't use different DPI settings across figures. Standardize on 300 DPI for all thesis figures.
- **English labels:** Don't forget to translate. User explicitly requires Spanish labels throughout.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LaTeX table formatting | Manual string concatenation with alignment | pandas.DataFrame.to_latex() | Handles escaping, booktabs formatting, column alignment, captions, labels automatically |
| Confusion matrix heatmaps | Custom matplotlib pcolormesh | seaborn.heatmap() | Built-in colorbar, annotations, normalization; well-tested for academic use |
| Metric aggregation from folds | Custom loops and averaging | numpy.mean(), numpy.std() | Vectorized operations, numerically stable, less error-prone |
| Figure size and DPI management | Per-plot savefig arguments | rcParams global configuration | Consistent style across all figures; change once, apply everywhere |
| Percentage formatting in LaTeX | String concatenation | f-strings with escaped backslashes | Readable, maintainable, handles LaTeX special characters correctly |

**Key insight:** Scientific visualization has mature tooling. pandas.to_latex() alone eliminates 80% of LaTeX table headaches (booktabs compatibility, column alignment, escaping). Seaborn abstracts matplotlib complexity while maintaining full customization. Don't reinvent confusion matrix plotting — the existing codebase pattern (seaborn heatmap + custom text annotations) is battle-tested.

## Common Pitfalls

### Pitfall 1: Confusion Matrix Normalization Axis

**What goes wrong:** Normalizing confusion matrix by columns instead of rows produces incorrect percentages. Each row should sum to 100% (all predictions for a true class), not each column.

**Why it happens:** Numpy broadcasting confusion — `axis=1` for row-wise normalization is counterintuitive.

**How to avoid:**
```python
# CORRECT: Normalize by rows (axis=1)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# WRONG: Normalizing by columns
cm_percent = cm.astype('float') / cm.sum(axis=0) * 100  # Don't do this
```

**Warning signs:** Column percentages don't sum to 100%; diagonal values aren't the largest in each row.

### Pitfall 2: LaTeX Special Characters in Metrics

**What goes wrong:** Pandas to_latex() auto-escapes underscores, but manually constructed strings (e.g., "F1_Score") break LaTeX compilation with undefined control sequence errors.

**Why it happens:** LaTeX interprets `_` as subscript command; needs `\_` in text mode.

**How to avoid:** Use escape=False in to_latex() and manually control escaping OR let pandas handle everything:
```python
# OPTION 1: Let pandas escape (default escape=True)
df = pd.DataFrame({'Métrica': ['F1-Score Macro']})  # Use - not _
latex = df.to_latex(index=False)

# OPTION 2: Manual control with escape=False
df = pd.DataFrame({'Métrica': ['F1\\_Score Macro']})  # Manually escape
latex = df.to_latex(index=False, escape=False)
```

**Warning signs:** LaTeX compilation fails with "Undefined control sequence"; underscores, percent signs, or ampersands in output.

### Pitfall 3: Baseline Metrics Source Confusion

**What goes wrong:** Using fold validation metrics instead of fold test metrics for baseline comparison, producing misleading improvements.

**Why it happens:** Multiple JSON files in outputs/classifier_cv/ — validation metrics vs test metrics files.

**How to avoid:** Always load from `fold_XX/test_results.json`, NOT `fold_XX/results.json`:
```python
# CORRECT: Test set metrics
fold_path = cv_dir / f"fold_{fold:02d}" / "test_results.json"

# WRONG: Validation metrics (not comparable to ensemble test results)
fold_path = cv_dir / f"fold_{fold:02d}" / "results.json"  # Don't use this
```

**Warning signs:** Baseline accuracy is >98.5% (suspiciously high); improvement is negative.

### Pitfall 4: Aggregating Instead of Selecting for Best Individual

**What goes wrong:** Comparing ensemble against the mean of 5 individual models instead of the best individual model, understating the ensemble's value.

**Why it happens:** The term "baseline" is ambiguous — could mean average or best single model.

**How to avoid:** User specifies "baseline (97.68% best individual)" in context. This is the max accuracy across folds, not the mean:
```python
# CORRECT: Best individual model
fold_accuracies = [fold['test_metrics']['accuracy'] for fold in per_fold_metrics]
baseline_best = max(fold_accuracies)  # 0.9794 (Fold 5)

# WRONG: Average of individuals (understates ensemble benefit)
baseline_mean = np.mean(fold_accuracies)  # 0.9768
```

**Warning signs:** Baseline value doesn't match user-specified 97.68%; double-check GROUND_TRUTH.json.

### Pitfall 5: Figure Background and Facecolor Mismatch

**What goes wrong:** Saving figures with transparent backgrounds causes issues when inserted into LaTeX documents; text becomes unreadable against thesis page color.

**Why it happens:** Default matplotlib savefig() uses transparent facecolor.

**How to avoid:** Explicitly set white background in savefig:
```python
# CORRECT: White background for LaTeX
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

# WRONG: Transparent background
plt.savefig(output_path, dpi=300, bbox_inches='tight')  # May be transparent
```

**Warning signs:** Figures look fine in isolation but have wrong background when inserted in thesis PDF.

## Code Examples

Verified patterns from existing codebase:

### Aggregate Fold Confusion Matrices

```python
# Source: scripts/generate_confusion_matrix_cv.py lines 16-105
import json
import numpy as np
from pathlib import Path

def aggregate_fold_confusion_matrices(cv_dir: Path) -> tuple:
    """Aggregate confusion matrices from 5-fold CV test results."""
    confusion_matrices = []
    accuracies = []
    f1_macros = []

    for fold in range(1, 6):
        fold_path = cv_dir / f"fold_{fold:02d}" / "test_results.json"

        with open(fold_path) as f:
            results = json.load(f)

        cm = np.array(results["confusion_matrix"])
        confusion_matrices.append(cm)
        accuracies.append(results["metrics"]["accuracy"])
        f1_macros.append(results["metrics"]["f1_macro"])

    # Aggregate: sum matrices, average metrics
    aggregated_cm = np.sum(confusion_matrices, axis=0)
    accuracy_mean = np.mean(accuracies)
    accuracy_std = np.std(accuracies)
    f1_macro_mean = np.mean(f1_macros)
    f1_macro_std = np.std(f1_macros)

    return aggregated_cm, accuracy_mean, accuracy_std, f1_macro_mean, f1_macro_std
```

### Generate Comparison JSON

```python
# Pattern for structured metrics output
import json
from pathlib import Path

def generate_comparison_metrics(baseline: dict, ensemble_tta: dict,
                                output_path: Path):
    """Generate structured JSON with all comparison metrics and deltas."""

    comparison = {
        "description": "Baseline vs Ensemble+TTA comparison",
        "timestamp": "2026-02-16",
        "baseline": {
            "source": "fold test results (best individual)",
            "accuracy": baseline["accuracy_mean"],
            "accuracy_std": baseline["accuracy_std"],
            "f1_macro": baseline["f1_macro_mean"],
            "f1_macro_std": baseline["f1_macro_std"],
            "best_fold": 5,
            "best_fold_accuracy": 0.9794
        },
        "ensemble_tta": {
            "source": "ensemble_test_results_tta.json",
            "accuracy": ensemble_tta["accuracy"],
            "f1_macro": ensemble_tta["f1_macro"],
            "f1_weighted": ensemble_tta["f1_weighted"]
        },
        "improvement": {
            "accuracy_delta": ensemble_tta["accuracy"] - baseline["accuracy_mean"],
            "f1_macro_delta": ensemble_tta["f1_macro"] - baseline["f1_macro_mean"],
            "accuracy_delta_pp": (ensemble_tta["accuracy"] - baseline["accuracy_mean"]) * 100
        },
        "per_class_f1_delta": ensemble_tta.get("tta_delta_metrics", {}).get("per_class_f1_delta", {}),
        "class_names": ["COVID-19", "Normal", "Neumonía Viral"]
    }

    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print(f"Comparison metrics saved: {output_path}")
```

### Per-Class Breakdown Table

```python
# Pattern for per-class metrics LaTeX table
import pandas as pd

def generate_per_class_table(baseline: dict, ensemble_tta: dict,
                             output_path: Path):
    """Generate LaTeX table with per-class F1-score breakdown."""

    # Extract per-class metrics
    baseline_per_class = baseline["per_class_metrics"]
    ensemble_per_class = ensemble_tta["per_class"]

    # Build DataFrame
    data = {
        'Clase': ['COVID-19', 'Normal', 'Neumonía Viral'],
        'Baseline F1': [
            f"{baseline_per_class['COVID']['f1-score']*100:.2f}\\%",
            f"{baseline_per_class['Normal']['f1-score']*100:.2f}\\%",
            f"{baseline_per_class['Viral_Pneumonia']['f1-score']*100:.2f}\\%"
        ],
        'Ensemble+TTA F1': [
            f"{ensemble_per_class['COVID']['f1-score']*100:.2f}\\%",
            f"{ensemble_per_class['Normal']['f1-score']*100:.2f}\\%",
            f"{ensemble_per_class['Viral_Pneumonia']['f1-score']*100:.2f}\\%"
        ],
        'Delta F1': [
            f"+{(ensemble_per_class['COVID']['f1-score'] - baseline_per_class['COVID']['f1-score'])*100:.2f}pp",
            f"+{(ensemble_per_class['Normal']['f1-score'] - baseline_per_class['Normal']['f1-score'])*100:.2f}pp",
            f"{(ensemble_per_class['Viral_Pneumonia']['f1-score'] - baseline_per_class['Viral_Pneumonia']['f1-score'])*100:.2f}pp"
        ]
    }

    df = pd.DataFrame(data)

    latex_str = df.to_latex(
        index=False,
        escape=False,
        column_format='lccc',
        caption='Impacto de TTA por clase: Mejora en F1-Score',
        label='tab:tta_per_class_impact',
        position='htbp'
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_str)
```

## State of the Art

| Area | Current Approach | Best Practice 2026 | Notes |
|------|------------------|-------------------|-------|
| Confusion Matrix Display | Seaborn heatmap + custom annotations | Same + scikit-learn ConfusionMatrixDisplay | Seaborn offers more control; sklearn is convenient but less customizable |
| LaTeX Table Generation | pandas.DataFrame.to_latex() | Same + booktabs package | Unchanged; pandas integration is standard |
| Figure DPI | 300 DPI PNG | 300 DPI PNG or vector (PDF/SVG) | Vector formats preferred for LaTeX but PNGs work fine at 300 DPI |
| Font Family | DejaVu Sans | Computer Modern or Latin Modern | Project uses DejaVu Sans; LaTeX native fonts (CMU, LM) offer better integration but require font setup |
| Color Palettes | Blues (sequential) | Colorblind-safe palettes (viridis, cividis) | Blues is acceptable for confusion matrices; consider viridis for accessibility |

**Key findings:**
- **Seaborn vs scikit-learn:** Project correctly uses seaborn.heatmap() with custom annotations. Scikit-learn's ConfusionMatrixDisplay (added 0.22+) is convenient but offers less control over text formatting and dual annotations.
- **Font choice:** DejaVu Sans is widely available and renders well. LaTeX-native fonts (Computer Modern, Latin Modern) offer tighter integration but require matplotlib configuration (`plt.rcParams['text.usetex'] = True`). Project's current choice is pragmatic.
- **Table formatting:** booktabs package (already in thesis setup/settings.tex) is the gold standard for LaTeX tables. pandas.to_latex() supports it natively.

## Open Questions

1. **Should we generate a single combined comparison figure or keep matrices separate?**
   - What we know: User specified "separate figures, not subfigures a/b"
   - What's unclear: Whether a third comparison figure (side-by-side or difference heatmap) would enhance clarity
   - Recommendation: Follow user specification (separate figures) initially; offer side-by-side comparison as optional enhancement

2. **How to handle the "baseline best individual" selection?**
   - What we know: GROUND_TRUTH.json lists 97.68% as baseline_individual_mean, but user context says "best individual"
   - What's unclear: Whether baseline is mean (97.68%) or max (97.94%, Fold 5)
   - Recommendation: Clarify with user; GROUND_TRUTH suggests mean, but context mentions "best individual"

3. **Should per-class confusion matrices be generated?**
   - What we know: User wants per-class F1 breakdown in tables
   - What's unclear: Whether per-class confusion matrices (3 separate 2x2 matrices per model for one-vs-rest) add value
   - Recommendation: Tables only per user specification; per-class matrices are overkill

## Sources

### Primary (HIGH confidence)

- Existing project scripts:
  - `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_confusion_matrix_cv.py` - Established confusion matrix pattern
  - `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_F5_8_comparison_cv.py` - Multi-figure comparison layout
  - `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_F5_3_single_panel_fixed.py` - rcParams style configuration

- Project data files:
  - `/home/donrobot/Projects/prediccion_warping_clasificacion/outputs/classifier_cv/ensemble_test_results_no_tta.json` - Phase 2 baseline metrics
  - `/home/donrobot/Projects/prediccion_warping_clasificacion/outputs/classifier_cv/ensemble_test_results_tta.json` - Phase 3 TTA metrics
  - `/home/donrobot/Projects/prediccion_warping_clasificacion/GROUND_TRUTH.json` - Validated metrics reference

- Thesis style configuration:
  - `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/setup/settings.tex` - booktabs, caption, font settings
  - `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/capitulo5/5_3_resultados_clasificacion_CV.tex` - Existing figure references and table formatting

### Secondary (MEDIUM confidence)

- [Pandas DataFrame.to_latex documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_latex.html) - Official pandas 3.0+ docs
- [Scikit-learn ConfusionMatrixDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html) - Official sklearn docs
- [Seaborn heatmap documentation](https://seaborn.pydata.org/generated/seaborn.heatmap.html) - Official seaborn 0.13+ docs

### Tertiary (LOW confidence, context only)

- [Confusion Matrix Visualization Tips](https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea) - Community best practices
- [Less confusing confusion matrices with Seaborn](https://blog.ddavo.me/posts/tutorials/confusing-confusion-matrices-seaborn/) - Tutorial on normalization
- [How to Construct a Confusion Matrix in LaTeX](https://copyprogramming.com/howto/how-to-construct-a-confusion-matrix-in-latex) - LaTeX integration patterns

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already in project, versions confirmed via imports
- Architecture: HIGH - Existing scripts provide complete patterns (confusion matrix, tables, rcParams)
- Per-class analysis: HIGH - JSON structure in ensemble_test_results_tta.json contains per_class_f1_delta
- LaTeX integration: HIGH - pandas.to_latex() well-documented, booktabs already in thesis settings

**Research date:** 2026-02-16
**Valid until:** 60 days (stable domain: matplotlib/seaborn APIs, pandas LaTeX export)

**Key validation sources:**
- Project codebase: Complete existing patterns for confusion matrices, LaTeX tables, Spanish labels
- GROUND_TRUTH.json: Validated metrics (baseline 97.68%, ensemble+TTA 98.26%)
- Existing thesis .tex files: Established table structure, booktabs usage, Spanish terminology
