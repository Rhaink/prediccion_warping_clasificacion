---
phase: 02-ensemble-core
plan: 01
type: execute
status: complete
subsystem: evaluation
tags: [ensemble, voting, cli, pytorch]

dependency-graph:
  requires: [01-02]
  provides: [ensemble_evaluation_infrastructure]
  affects: [02-02, 02-03]

tech-stack:
  added: []
  patterns: [weighted_soft_voting, hard_voting, factory_pattern]

key-files:
  created:
    - src_v2/evaluation/ensemble.py
  modified:
    - src_v2/cli.py

decisions:
  - id: use_validation_f1_weights
    what: Use validation F1-macro from results.json as ensemble weights
    why: Avoids test set contamination, proven approach from research
    alternatives: [equal_weights, test_based_weights]
    impact: Maintains methodological integrity while optimizing ensemble
  - id: compute_both_voting_methods
    what: Compute both soft and hard voting for comparison
    why: Soft voting primary method, hard voting provides baseline reference
    alternatives: [soft_only]
    impact: Enables validation that soft voting superior in results

metrics:
  duration: 3 min
  completed: 2026-01-28
---

# Phase 2 Plan 1: Ensemble Evaluation Infrastructure Summary

**One-liner:** Implemented ensemble evaluation infrastructure with weighted soft voting and CLI command for 5-fold CV model evaluation

## What Was Built

Created core ensemble machinery for combining predictions from 5 cross-validation models:

**1. Ensemble Evaluation Module (`src_v2/evaluation/ensemble.py`)**
- `load_ensemble_models()`: Load 5 checkpoints with validation F1 weight extraction from `results.json`
- `weighted_soft_voting()`: Probability averaging using normalized validation F1-macro weights
- `hard_voting()`: Majority vote baseline using Counter for deterministic tie-breaking
- `ensemble_inference()`: Batch processing loop with tqdm progress bar
- `validate_ensemble_setup()`: Pre-flight sanity checks (architecture match, eval mode, sample count, probability sum)

**2. CLI Command (`src_v2/cli.py`)**
- New `evaluate-classifier-ensemble` command following existing `evaluate-classifier` pattern
- Config-based execution (JSON with checkpoint_paths, data_dir, baseline)
- Comprehensive output schema with per-fold and ensemble metrics
- Formatted summary table with baseline comparison
- Optional CSV export for per-sample predictions

## How It Works

**Ensemble Evaluation Flow:**
1. Load config JSON with 5 checkpoint paths and test data directory
2. Load all models using `create_classifier()` factory, extract validation F1-macro from `fold_*/results.json`
3. Run sanity checks (architecture match, eval mode, sample count = 1,895, probability axioms)
4. Batch inference: iterate test set once, collect predictions + probabilities from all models
5. Compute soft voting: weighted probability averaging with normalized F1 weights
6. Compute hard voting: majority vote per sample using Counter
7. Calculate metrics: accuracy, F1-macro, F1-weighted, confusion matrix, per-class
8. Output JSON with full schema + print formatted summary table

**Key Implementation Details:**
- Validation weights read from `Path(checkpoint_path).parent / "results.json"` → `best_val_f1` field
- Soft voting uses `torch.einsum('mni,m->ni', probs_stacked, weights_normalized)` for efficiency
- Hard voting breaks ties deterministically by taking lowest class index
- Sanity checks validate probability sum = 1.0 with `atol=1e-5` tolerance
- Architecture verification prevents silent failures from model mismatch

## Deviations from Plan

None - plan executed exactly as written.

## Technical Decisions

**Decision 1: Use validation F1-macro as ensemble weights**
- Extracted from `results.json` in each fold directory
- Avoids test set contamination (weights fixed from validation)
- Normalized to sum to 1.0 before probability averaging
- Alternative: equal weights (simpler but ignores model quality)

**Decision 2: Compute both soft and hard voting**
- Soft voting primary method (captures model confidence)
- Hard voting computed for comparison (demonstrates soft voting superiority)
- Both included in output JSON for analysis
- Alternative: soft voting only (would save minimal computation)

**Decision 3: Follow existing CLI patterns**
- Matches `evaluate-classifier` command structure (~150 lines)
- Config-based execution reduces CLI flag proliferation
- Reuses `get_device()`, `get_classifier_transforms()`, `ImageFolder` patterns
- Alternative: all parameters via CLI flags (less maintainable)

## Files Changed

**Created:**
- `src_v2/evaluation/ensemble.py` (260 lines): Ensemble voting logic with 5 exported functions

**Modified:**
- `src_v2/cli.py`: Added `evaluate-classifier-ensemble` command (292 lines inserted after line 2602)

## Verification

**All success criteria met:**
- Module importable: All 5 functions import without errors
- CLI command visible: Appears in `python -m src_v2 --help`
- Help displays correctly: Shows all options (--config, --output, --device, --batch-size, --predictions-csv)
- No syntax errors: `py_compile` passes for both files
- Code follows conventions: Type hints, Google-style docstrings, PEP 8

**Testing performed:**
- Import verification: `from src_v2.evaluation.ensemble import *`
- CLI help rendering: `python -m src_v2 evaluate-classifier-ensemble --help`
- Syntax validation: `python -m py_compile src_v2/evaluation/ensemble.py src_v2/cli.py`

## Next Phase Readiness

**Ready for Phase 2 Plan 2 (Config Creation):**
- Ensemble infrastructure complete and tested (imports work)
- CLI command operational (help displays correctly)
- Ready to create `configs/ensemble_classifier.json` with checkpoint paths
- No blockers identified

**Dependencies provided:**
- `load_ensemble_models()`: Used by Plan 2 for config validation
- `evaluate-classifier-ensemble` command: Will execute config in Plan 2
- Output schema established: Plans 3-5 can reference JSON structure

**Known limitations:**
- Baseline ensemble only (no TTA yet - deferred to Phase 3)
- Expected sample count hardcoded to 1,895 (Phase 1 value)
- No test execution yet (requires config creation in Plan 2)

## Task Breakdown

| Task | Description | Commit | Files | Status |
|------|-------------|--------|-------|--------|
| 1 | Create ensemble evaluation module | 19b4e91 | src_v2/evaluation/ensemble.py | Complete |
| 2 | Add evaluate-classifier-ensemble CLI command | a0784f1 | src_v2/cli.py | Complete |

**Total commits:** 2 atomic commits (one per task)
**Execution time:** 3 minutes
**Lines added:** 552 (260 module + 292 CLI)

## Key Learnings

**What worked well:**
- Research phase identified exact results.json structure (best_val_f1 field location confirmed)
- Following existing CLI patterns (evaluate-classifier) reduced implementation friction
- Factory pattern (`create_classifier`) simplified model loading
- Sanity checks provide early error detection

**What could be improved:**
- Config validation could be more robust (e.g., verify all checkpoints have same backbone before loading)
- Hard voting tie-breaking could be explicit in docstring (currently "deterministic" but mechanism unclear until reading code)
- Progress bar granularity (per-batch vs per-model) not documented

**Avoided pitfalls:**
- Data leakage: Weights read ONLY from validation results, never test metrics
- Architecture mismatch: Verified backbone_name consistency in `load_ensemble_models()`
- Probability violations: Validated sum = 1.0 in `validate_ensemble_setup()`
- Sample count mismatch: Assert test set size = 1,895 (Phase 1 verified value)

## Handoff Notes

**For Plan 02-02 (Config Creation):**
- Config schema established: `checkpoint_paths` (list), `data_dir` (str), `baseline_accuracy` (float), `baseline_std` (float)
- Checkpoint paths: `outputs/classifier_cv/fold_{01-05}/best_classifier.pt`
- Test data: `outputs/warped_lung_best/session_warping/test/`
- Baseline from Phase 1: 97.68% ± 0.16%

**For Plan 02-03 (Baseline Evaluation):**
- Output schema: `ensemble_test_results.json` with per_fold_metrics, ensemble_soft_voting, ensemble_hard_voting, comparison
- Expected metrics: soft voting accuracy, F1-macro, F1-weighted, confusion matrix, per-class
- Comparison fields: baseline_mean, baseline_std, ensemble_soft_delta, ensemble_hard_delta

**For future sessions:**
- Ensemble infrastructure reusable for TTA experiments (Phase 3)
- Output JSON format standardized for Phase 5 validation checks
- CLI command pattern established for future ensemble variations
