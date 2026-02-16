# Phase 4: Analysis & Visualization - Context

**Gathered:** 2026-02-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Generate comparison metrics and thesis-ready confusion matrix visualizations showing improvement from ensemble+TTA over the baseline individual model. Produce publication-ready figures and structured data for thesis Chapter 5. No new model training or evaluation — this phase consumes existing results from Phases 2-3.

</domain>

<decisions>
## Implementation Decisions

### Thesis visual style
- Font family: Claude's discretion — pick something academic and clean
- Color palette: Claude's discretion — pick an appropriate academic palette
- All labels in Spanish (axis titles, figure titles, annotations)
- Must match existing Chapter 5 figure style — locate existing visualization scripts or .tex references to determine current style conventions

### Comparison story
- Main comparison: baseline (97.68% best individual) vs final ensemble+TTA (98.26%)
- Include both overall metrics AND per-class breakdown (COVID-19, Normal, Neumonía Viral)
- Presentation format: tables only (no bar charts)
- Highlight TTA per-class impact as a finding: COVID benefits most (+0.44% F1), Viral degrades slightly (-0.28% F1)
- Generate LaTeX-ready .tex table file for direct \input{} in thesis

### Confusion matrix format
- Two separate confusion matrices: baseline (best individual model) vs final (ensemble+TTA)
- Each as its own figure (separate figures, not subfigures a/b)
- Cell values show both raw counts and normalized percentages (e.g., "450\n99.6%")
- Class labels in Spanish: COVID-19, Normal, Neumonía Viral

### Output deliverables
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

</decisions>

<specifics>
## Specific Ideas

- Match existing Chapter 5 figures — researcher should locate .tex file references and existing visualization code to determine current style
- Spanish labels throughout: "Matriz de Confusión", "Clase Real", "Clase Predicha", etc.
- TTA per-class impact is an interesting finding worth highlighting — COVID benefits most from horizontal flip augmentation
- Keep it clean and academic — tables over charts, data over decoration

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-analysis-visualization*
*Context gathered: 2026-02-16*
