# Phase 8: Training Improvements - Context

**Gathered:** 2026-02-17
**Status:** Ready for planning

<domain>
## Phase Boundary

Re-train 5-fold CV ensemble on Phase 7 cleaned data with three training techniques: focal loss, hard example mining, and curriculum learning. ResNet-18 architecture stays fixed to isolate the data/training effect. Target: improve Viral Pneumonia recall above baseline 92.9%.

</domain>

<decisions>
## Implementation Decisions

### Technique composition & ablation
- Individual ablation: test each technique separately (focal loss, hard example mining, curriculum learning) before combining
- Cleaned-data baseline first: train v1.0 pipeline on cleaned data (no new techniques) to isolate data cleaning effect
- Ablation order: Claude's discretion based on implementation simplicity and dependencies
- If a technique hurts performance: keep it and tune hyperparameters before dropping — these are established techniques
- Final combined model: fine-tune from the best individual ablation checkpoint (not retrain from scratch)
- Each ablation run is a full 5-fold CV for rigorous comparability with v1.0
- Allow light hyperparameter tuning per technique (lr, epochs) if clearly needed

### Data preparation
- Re-warp the full dataset from scratch with Phase 7 exclude-list applied upfront (fresh warped dataset)
- Phase 7 excluded samples (432 in manifest) are strictly excluded from all training — never used as hard examples

### Evaluation strategy
- Validation-only evaluation during ablation — test set reserved for Phase 10 final evaluation
- Metrics, reporting format, and regression guardrail thresholds: Claude's discretion

### Claude's Discretion
- Ablation technique order (focal loss, mining, or curriculum first)
- Epoch count per ablation run (v1.0 used 15 frozen + 100 fine-tune)
- Two-phase training structure — may be adapted per technique if beneficial
- Checkpoint strategy (best-only vs best+last per fold)
- Hard example definition method (OOF loss, misclassification count, or confidence margin)
- Hard example oversampling ratio
- When to start mining during training (after warmup or from epoch 1)
- Static vs dynamic hard example set
- Whether to try OHEM as fallback if basic mining fails
- Class-aware vs class-agnostic hard example weighting
- Hard example reporting (percentile vs absolute count)
- Curriculum difficulty metric (OOF loss, confidence margin, etc.)
- Curriculum schedule type (linear ramp, step-based, or loss-triggered)
- Starting fraction of dataset for curriculum
- Class balance maintenance during curriculum stages
- Curriculum application scope (fine-tuning only vs both phases)
- Interaction between curriculum and mining
- Logging granularity for curriculum statistics
- Curriculum learning approach (research best fit from literature)
- Evaluation metrics set, reporting format, regression thresholds

</decisions>

<specifics>
## Specific Ideas

- v1.0 baseline is 98.26% test accuracy with weighted soft voting ensemble + horizontal flip TTA
- Viral Pneumonia recall at 92.9% is the weakest class — primary improvement target
- Phase 7 OOF data (probabilities with T=2.0 temperature scaling) is available for difficulty scoring
- 33 misclassified test images and 34 cleanlab label-noise samples already identified
- No GPU time constraint — run as many experiments as needed

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 08-training-improvements*
*Context gathered: 2026-02-17*
