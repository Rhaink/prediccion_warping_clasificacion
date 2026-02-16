---
phase: 04-analysis-visualization
plan: 02
subsystem: analysis-tools
tags: [latex, tables, thesis, visualization]
dependency_graph:
  requires:
    - outputs/classifier_cv/fold_*/test_results.json
    - outputs/classifier_cv/ensemble_test_results_tta.json
    - outputs/classifier_cv/ensemble_test_results_no_tta.json
  provides:
    - scripts/generate_comparison_tables.py
    - outputs/classifier_cv/comparison_tables.tex (gitignored)
  affects:
    - thesis Chapter 5 tables
tech_stack:
  added: []
  patterns: [latex-generation, json-data-loading, statistical-formatting]
key_files:
  created:
    - scripts/generate_comparison_tables.py
    - outputs/classifier_cv/comparison_tables.tex
  modified: []
decisions:
  - decision: "Use hand-crafted LaTeX string formatting instead of pandas.to_latex() for full control over booktabs style"
    rationale: "Existing thesis tables use hand-crafted LaTeX matching specific formatting conventions"
    alternatives: ["pandas.to_latex() with custom formatters"]
  - decision: "Output .tex file to outputs/classifier_cv/ (gitignored) instead of docs/Tesis/"
    rationale: "Follows pattern of other generated artifacts in outputs directory, source script is version controlled"
    alternatives: ["Output directly to docs/Tesis/", "Output to both locations"]
  - decision: "Impact classification uses ±0.1pp threshold for Mejora/Leve degradación/Neutro"
    rationale: "Reasonable threshold for per-class F1 changes given scale of improvements observed"
    alternatives: ["±0.05pp threshold", "±0.2pp threshold"]
metrics:
  duration_minutes: 2
  tasks_completed: 1
  files_created: 2
  lines_of_code: 397
  completed_at: "2026-02-16T10:22:01Z"
---

# Phase 04 Plan 02: LaTeX Comparison Tables Generation Summary

**One-liner:** Script generating two booktabs-formatted LaTeX tables comparing baseline (97.68%) vs ensemble+TTA (98.26%) with per-class F1 breakdown and case-level impact for direct thesis inclusion.

## Objective Achieved

Generated LaTeX-ready comparison tables for direct thesis Chapter 5 inclusion via `\input{}`. Tables show baseline vs ensemble+TTA improvement with per-class breakdown, using booktabs formatting and Spanish labels matching existing thesis style.

## Tasks Completed

### Task 1: Create LaTeX table generation script and produce .tex output
**Status:** ✅ Complete
**Commit:** 1bde6735
**Files:** `scripts/generate_comparison_tables.py`, `outputs/classifier_cv/comparison_tables.tex`

**Implementation:**
- Created Python script with argparse CLI accepting `--cv-dir` and `--output` arguments
- Implemented data loading functions:
  - `load_fold_results()`: Loads 5 fold test_results.json files, computes mean/std for accuracy, F1-macro, F1-weighted, and per-class F1
  - `load_ensemble_results()`: Loads ensemble_test_results_tta.json with case-level analysis and TTA delta metrics
- Implemented LaTeX generation functions:
  - `generate_table1_overall_comparison()`: Creates overall comparison table (Baseline vs Ensemble+TTA) with accuracy, F1-macro, F1-weighted rows
  - `generate_table2_per_class_impact()`: Creates per-class F1 breakdown with TTA deltas and impact classification
- Implemented formatting helpers:
  - `format_percentage()`: Formats floats as LaTeX percentages (97.68\%)
  - `format_delta()`: Formats deltas with sign and pp units (+0.58 pp, using $-$ for negatives)
  - `classify_impact()`: Classifies TTA impact as Mejora/Neutro/Leve degradación based on ±0.1pp threshold
- Generated LaTeX with proper booktabs formatting: `@{}lccc@{}` column format, `\toprule`, `\midrule`, `\bottomrule`
- All labels in Spanish matching thesis style
- Values use `\textbf{}` for emphasis on ensemble+TTA results
- LaTeX thousands separator `{,}` for numbers like 1,895
- File header comments with generation instructions

**Table 1 Contents (Overall Comparison):**
- Baseline: 97.68% ± 0.16% accuracy, 96.47% ± 0.27% F1-macro, 97.67% ± 0.16% F1-weighted
- Ensemble+TTA: 98.26% accuracy, 97.12% F1-macro, 98.25% F1-weighted
- Improvements: +0.58pp accuracy, +0.66pp F1-macro, +0.58pp F1-weighted
- Sample counts: 1,895 × 5 vs 1,895

**Table 2 Contents (Per-class TTA Impact):**
- COVID: 97.22% → 98.33% (+1.10pp, Mejora)
- Normal: 98.34% → 98.75% (+0.41pp, Mejora)
- Neumonía Viral: 93.84% → 94.29% (+0.46pp, Mejora)
- Case-level: 6 helped, 3 hurt, 1,886 neutral (net: +3)

**Verification:**
- ✅ comparison_tables.tex exists with 1.7 KB size
- ✅ Contains exactly 2 `\begin{table}` and 2 `\end{table}` tags
- ✅ Uses booktabs commands (toprule, midrule, bottomrule) throughout
- ✅ Metric values match validated data: baseline 97.68%, ensemble+TTA 98.26%
- ✅ Per-class F1 deltas show all positive improvements (COVID highest at +1.10pp)
- ✅ Script runs without errors, prints detailed summary

## Deviations from Plan

None - plan executed exactly as written.

## Verification Results

All verification criteria met:
1. ✅ comparison_tables.tex exists and is valid LaTeX (contains 2 table begin/end pairs)
2. ✅ Contains exactly 2 tables (overall comparison + per-class)
3. ✅ Uses booktabs commands (toprule, midrule, bottomrule)
4. ✅ Metric values match baseline (97.68%) and ensemble+TTA (98.26%)
5. ✅ Per-class TTA deltas show all positive improvements (matches validated findings)
6. ✅ Script runs without errors, generates correct output

## Key Decisions

1. **Hand-crafted LaTeX formatting**: Used direct string building instead of pandas.to_latex() for full control over booktabs style matching existing thesis tables in `5_3_resultados_clasificacion_CV.tex`

2. **Output location**: Generated .tex file in `outputs/classifier_cv/` (gitignored) following pattern of other generated artifacts. Source script `generate_comparison_tables.py` is version controlled.

3. **Impact classification threshold**: Used ±0.1pp threshold to classify per-class F1 changes as Mejora/Neutro/Leve degradación, appropriate for the scale of improvements observed.

4. **No pandas dependency**: Avoided pandas for table generation to maintain direct control over LaTeX formatting and reduce dependencies for a simple table generation task.

## Technical Highlights

- **Data aggregation**: Computed baseline statistics from 5 independent test_results.json files with proper mean/std calculations
- **LaTeX formatting precision**: Used `$\pm$`, `$\Delta$`, `$-$` for mathematical symbols, `\textbf{}` for emphasis, `{,}` for thousands separators
- **Robust path handling**: Uses pathlib with absolute/relative path resolution, creates output directories automatically
- **Comprehensive output**: Script prints detailed summary with all metrics, deltas, and case-level impact for verification

## Files Changed

### Created
- `scripts/generate_comparison_tables.py` (397 lines): Main script with data loading, LaTeX generation, and CLI
- `outputs/classifier_cv/comparison_tables.tex` (40 lines): Generated LaTeX file with two tables (gitignored)

### Modified
- None

## Next Steps

1. Plan 04-01 (if not completed): Generate other visualization artifacts for Chapter 5
2. Phase 04 verification: Validate all must_haves for analysis-visualization phase
3. Thesis integration: Use `\input{outputs/classifier_cv/comparison_tables.tex}` in Chapter 5 LaTeX source

## Self-Check: PASSED

### Created Files
```bash
$ ls -lh scripts/generate_comparison_tables.py
-rwxrwxr-x 1 donrobot donrobot 13K Feb 16 04:21 scripts/generate_comparison_tables.py
FOUND: scripts/generate_comparison_tables.py

$ ls -lh outputs/classifier_cv/comparison_tables.tex
-rw-rw-r-- 1 donrobot donrobot 1.7K Feb 16 04:21 outputs/classifier_cv/comparison_tables.tex
FOUND: outputs/classifier_cv/comparison_tables.tex
```

### Commits
```bash
$ git log --oneline --all | grep 1bde6735
1bde6735 feat(04-02): generate LaTeX comparison tables for thesis
FOUND: 1bde6735
```

### Content Verification
```bash
$ grep -c '\\begin{table}' outputs/classifier_cv/comparison_tables.tex
2

$ grep -c '\\end{table}' outputs/classifier_cv/comparison_tables.tex
2

$ grep '97.68\|98.26' outputs/classifier_cv/comparison_tables.tex | wc -l
2
```

All files created, commit exists, and content verified. Self-check PASSED.
