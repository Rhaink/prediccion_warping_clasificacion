# Phase 2: Ensemble Core - Context

**Gathered:** 2026-01-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement ensemble evaluation infrastructure to load 5 cross-validation models and compute baseline metrics using soft/hard voting. This phase establishes the foundation for ensemble predictions without Test-Time Augmentation (TTA). TTA integration is Phase 3.

Scope: Model loading, voting algorithms, metric computation, baseline comparison, JSON output generation. Does NOT include: TTA, visualization, final test evaluation.

</domain>

<decisions>
## Implementation Decisions

### Code Structure & Integration
- Extend existing CLI via `src_v2/cli.py` with new `evaluate-classifier-ensemble` command (follows existing evaluation patterns)
- Ensemble logic lives in new `src_v2/evaluation/ensemble.py` module (clean separation, matches evaluation domain)
- Reuse existing `create_classifier()` factory pattern from `src_v2/models/classifier.py` for model loading consistency
- Configuration via JSON config file following project pattern: `configs/ensemble_classifier.json` with model paths, voting method, weights

### Voting Implementation
- **Load all 5 models into memory at once** (~220MB for 5x ResNet-18) for faster inference
- **Weighted soft voting by validation F1**: Better models get more influence based on validation F1-macro performance
  - Read weights from existing `outputs/classifier_cv/fold_*/val_results.json` files
  - Aggregate probabilities: `weighted_mean(probs, weights=validation_F1s)`
- **Both soft and hard voting**: Compute both methods, include both in output JSON for comparison
  - Soft voting: weighted probability averaging (primary method)
  - Hard voting: majority vote (baseline comparison)

### Output Format & Reporting
- **Output JSON schema includes:**
  - Per-fold metrics: individual accuracy, F1-macro, F1-weighted for each of 5 folds
  - Ensemble metrics: aggregated metrics for soft voting AND hard voting
  - Per-class breakdown: COVID/Normal/Viral_Pneumonia precision, recall, F1 for ensemble
  - Metadata: timestamp, model paths, voting weights, configuration used
- **Comparison with baseline**: Include delta showing ensemble improvement over 97.68% baseline
- **File location**: `outputs/classifier_cv/ensemble_test_results.json` (as specified in requirements)
- **Human-readable summary**: CLI prints formatted table with key metrics after evaluation (not just JSON)

### Validation & Verification
- **Sanity checks performed:**
  - Model architecture match: verify all 5 models have same architecture before ensembling
  - Probability sum check: assert probabilities sum to 1.0 for each prediction
  - Sample count verification: confirm evaluation runs on exact 1,895 test images
  - Output range validation: check predictions are valid class indices [0, 1, 2]
- **Baseline verification**: Compare individual fold results with existing `test_results.json` to verify reproducibility
- **Error handling**: Fail fast on first error (missing checkpoint, architecture mismatch, data loading failure)
- **Debug output**: Generate per-sample predictions CSV (`predictions.csv`) with columns: `image_path, true_label, fold_1_pred, fold_2_pred, fold_3_pred, fold_4_pred, fold_5_pred, ensemble_soft_pred, ensemble_hard_pred`

### Claude's Discretion
- Exact JSON schema field names and nesting structure
- Progress bar/logging format during evaluation
- DataLoader batch size and num_workers optimization
- Precision of floating-point metrics in output (e.g., 4 vs 6 decimal places)

</decisions>

<specifics>
## Specific Ideas

- Follow existing evaluation patterns in `src_v2/cli.py` (see `evaluate-classifier` command for structure)
- Validation F1 weights are read from existing results files - no manual specification needed
- Ensemble evaluation should feel like running individual fold evaluation, just with combined output
- The 97.68% ± 0.16% baseline from Phase 1 is the comparison anchor

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-ensemble-core*
*Context gathered: 2026-01-27*
