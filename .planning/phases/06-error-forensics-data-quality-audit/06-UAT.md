---
status: complete
phase: 06-error-forensics-data-quality-audit
source: [06-01-SUMMARY.md, 06-02-SUMMARY.md, 06-03-SUMMARY.md]
started: 2026-02-16T15:30:00Z
updated: 2026-02-16T23:00:00Z
---

## Current Test

[testing complete - all gaps resolved]

## Tests

### 1. Overview grid shows all 33 misclassified images
expected: File `outputs/error_forensics/error_visualizations/overview_grid_all_33.png` exists and shows a grid of all 33 misclassified test images with true/predicted labels visible. Open it and confirm it looks reasonable as a thesis figure.
result: pass (fixed)
note: "All 33 images visible with real confidence scores. Fixed 3 bugs: (1) sample_idx mapping used images.csv row index instead of ImageFolder ordering, (2) original path missing /images/ subdirectory, (3) placeholder confidence replaced with real ensemble probabilities."

### 2. Per-sample pipeline traces exist for all 33 errors
expected: Directory `outputs/error_forensics/error_visualizations/per_sample_detailed/` contains 33 PNG files. Open one (e.g., sample_375_pipeline.png) and confirm it shows a 4-panel layout: original X-ray, landmarks overlay, warped image, classification result.
result: pass (fixed)
note: "All 4 panels render correctly: original X-ray loaded, landmarks overlay positioned correctly, warped image visible, probability bars show real ensemble probabilities."

### 3. Error analysis JSON contains correct metadata
expected: Run `python3 -c "import json; d=json.load(open('outputs/error_forensics/error_analysis_results.json')); print(f'Errors: {d[\"metadata\"][\"total_errors\"]}, Accuracy: {d[\"metadata\"][\"accuracy\"]}%')"` and confirm it shows 33 errors at 98.26% accuracy.
result: pass

### 4. Duplicate detection found cross-split leakage
expected: Run `python3 -c "import json; d=json.load(open('outputs/error_forensics/duplicates/duplicate_analysis_summary.json')); print(f'Original pairs: {d[\"original\"][\"total_pairs\"]}, Warped cross-split: {d[\"warped\"][\"cross_split\"]}')"` and confirm warped cross-split > 0 (critical finding about data leakage).
result: pass

### 5. BRISQUE quality scores computed for all images
expected: Run `python3 -c "import pandas as pd; df=pd.read_csv('outputs/error_forensics/quality_scores/all_images_quality.csv'); print(f'Images scored: {len(df)}, Mean BRISQUE: {df[\"brisque_score\"].mean():.2f}')"` and confirm all ~15K images are scored with reasonable BRISQUE values (lower is better, expect mean around 20-30).
result: pass

### 6. Spanish forensics report is complete and thesis-ready
expected: Open `outputs/error_forensics/report_error_forensics.md` and confirm it contains: (1) Resumen Ejecutivo, (2) Analisis de Errores, (3) Deteccion de Duplicados, (4) Evaluacion de Calidad, (5) Recomendaciones para Fase 7. All numbers should be populated (no placeholders).
result: pass

### 7. Interactive notebook is valid and loadable
expected: Run `python3 -c "import json; nb=json.load(open('notebooks/error_forensics_interactive.ipynb')); print(f'Cells: {len(nb[\"cells\"])}, Code cells: {sum(1 for c in nb[\"cells\"] if c[\"cell_type\"]==\"code\")}')"` and confirm 12 cells (6 code + 6 markdown).
result: pass

### 8. Confusion pair distribution matches expected pattern
expected: The error analysis shows Viral_Pneumonia->Normal as dominant error (12 cases), followed by COVID->Normal (10 cases). This matches the known difficulty of distinguishing Normal from disease classes. Confirm this pattern makes clinical sense.
result: pass

## Summary

total: 8
passed: 8
issues: 0
pending: 0
skipped: 0

## Gaps

- truth: "All 33 misclassified images visible in overview grid with real confidence scores"
  status: resolved
  reason: "Fixed 3 root causes: (1) sample_idx-to-image mapping was using images.csv row index but predictions CSV uses ImageFolder ordering (sorted by class then filename), (2) original image path missing /images/ subdirectory, (3) real probabilities extracted from 5-fold ensemble with TTA."
  severity: major
  test: 1
  root_cause: "Three bugs: (a) images.csv ordering != ImageFolder ordering (predictions use ImageFolder), (b) dataset structure has {class}/images/{file}.png not {class}/{file}.png, (c) no per-sample probabilities saved to disk."
  fix_commits: "pending commit"
- truth: "Pipeline trace shows 4-panel layout: original X-ray, landmarks overlay, warped image, classification result"
  status: resolved
  reason: "Both visualization issues were caused by the same 3 data-loading bugs above. Once correct images and landmarks were loaded, the visualization code (already fixed in fdf52826) rendered correctly."
  severity: major
  test: 2
  root_cause: "Same root causes as Gap 1 - wrong image mapping led to black panels and misplaced landmarks."
  fix_commits: "pending commit"

## Known Limitations

- failure_origin is "bad_warp" for all 33 errors because the fill_rate threshold (0.85) is too high for warped images that naturally only cover the lung region (~47% fill). This is a pre-existing design issue in trace_pipeline_failures(), not a UAT bug.
