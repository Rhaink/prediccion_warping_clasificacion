# Codebase Concerns

**Analysis Date:** 2026-01-27

## Tech Debt

**Monolithic CLI Module:**
- Issue: `src_v2/cli.py` contains 10,520 lines with 40+ commands, mixing parameter parsing, model training, inference, visualization, and evaluation logic all in one file
- Files: `src_v2/cli.py`
- Impact: Difficult to test individual commands, high cyclomatic complexity, hard to maintain and extend. Commands depend on shared state and repeated parameter definitions
- Fix approach: Refactor into command submodules in `src_v2/commands/` directory with shared utilities and parameter definitions. Each command gets its own file (~200-300 lines max)

**Global State in GUI Config:**
- Issue: `src_v2/gui/config.py::EXAMPLES` is modified by `populate_examples()` function using `global` keyword
- Files: `src_v2/gui/config.py:415`, `src_v2/gui/config.py:413-429`
- Impact: Non-deterministic initialization, potential race conditions in multi-instance scenarios, unpredictable behavior if examples directory missing
- Fix approach: Implement lazy initialization with property pattern or store examples list in singleton factory instead of module-level global

**Broad Exception Handling with Silent Failures:**
- Issue: Multiple locations catch bare `Exception` and silently `pass`, particularly in serialization/JSON conversion logic (`src_v2/cli.py:773-787`) and warping functions
- Files: `src_v2/cli.py:773-787`, `src_v2/processing/warp.py:297-299`, `src_v2/cli.py:4686`, `src_v2/cli.py:6109`, `src_v2/cli.py:9561`
- Impact: Swallows conversion errors, incomplete checkpoint serialization, warped image data loss, obscures root causes during debugging
- Fix approach: Catch specific exceptions (ValueError, TypeError, etc.), log before passing, or raise with context using `raise ... from e`

## Known Bugs

**Delaunay Triangulation Instability with Collinear Landmarks:**
- Symptoms: Occasional crashes or degenerate triangles when landmarks are nearly collinear or form a very narrow configuration
- Files: `src_v2/processing/warp.py:268-269`, `src_v2/processing/warp.py:364`, `src_v2/processing/warp.py:371`
- Trigger: Input images where predicted landmarks are tightly bunched (e.g., due to poor model performance or edge cases)
- Workaround: `piecewise_affine_warp()` skips triangles with area < 1e-6, but Delaunay itself can fail if points are badly conditioned
- Fix approach: Add point perturbation before Delaunay if collinearity detected, or validate point distribution before triangulation

**Image Size Mismatch Warnings Not Fatal:**
- Symptoms: Inconsistent normalization if actual image dimensions don't match `ORIGINAL_IMAGE_SIZE` (299x299). Landmark coordinates normalized using actual size but expected to be in [0,1] for 299x299
- Files: `src_v2/data/dataset.py:124-131`
- Trigger: Dataset contains images with non-standard dimensions; warning emitted once per dataset
- Workaround: Code falls back to actual image size for normalization
- Fix approach: Enforce strict image size check at dataset loading, reject or resize non-conforming images before any processing

**Model Architecture Detection Fallback Chain:**
- Symptoms: `detect_architecture_from_checkpoint()` uses multiple fallback layers (checking for specific layer keys) which may misidentify architecture if checkpoint state_dict structure is unusual
- Files: `src_v2/cli.py:154-198`
- Trigger: Checkpoints from different training sessions with variable head architectures
- Workaround: Falls back to hidden_dim=256 if expected keys missing
- Fix approach: Store architecture metadata in checkpoint alongside state_dict, return explicit error if architecture cannot be definitively detected

## Security Considerations

**Arbitrary File Path Inputs:**
- Risk: CLI commands accept file paths directly without validation; potential for path traversal or accessing unauthorized files
- Files: `src_v2/cli.py` (many commands), `src_v2/gui/inference_pipeline.py:46-79`
- Current mitigation: Path validation exists in `validate_image()` for GUI (checks file exists, format, size), but CLI commands have minimal checks
- Recommendations: Add `Path.resolve()` and canonicalization checks, restrict file access to whitelisted directories, validate all user-supplied paths at entry point

**Model Checkpoint Loading Without Validation:**
- Risk: `torch.load()` with `weights_only=False` allows arbitrary Python code execution during unpickling
- Files: `src_v2/models/classifier.py:241`, `src_v2/gui/model_manager.py` (checkpoint loading)
- Current mitigation: None documented
- Recommendations: Use `weights_only=True` if possible (requires recent PyTorch), validate checkpoint provenance, isolate model loading in restricted environment

**No Input Sanitization in Web GUI:**
- Risk: Gradio app (`src_v2/gui/app.py`) accepts image uploads with minimal validation beyond file extension and size checks
- Files: `src_v2/gui/app.py`, `src_v2/gui/inference_pipeline.py`
- Current mitigation: Basic format validation in `validate_image()`, but no content inspection
- Recommendations: Add image integrity checks (magic bytes), reject suspicious content, implement rate limiting, add audit logging

## Performance Bottlenecks

**Numpy Warning Suppression and Copy Operations:**
- Problem: Frequent `np.copy()` calls in data loading and warping, especially in triangle-by-triangle warping loops. Each triangle warp creates intermediate arrays
- Files: `src_v2/processing/warp.py:217`, `src_v2/processing/warp.py:330`, `src_v2/processing/warp.py:430` (copy operations in loops)
- Cause: Defensive copying for safety, but per-triangle overhead compounds with 20-30 triangles per image
- Improvement path: Pre-allocate output arrays, use in-place operations where safe, batch warp similar-sized triangles, consider Cython/numba for tight loops

**DataLoader num_workers Fallback:**
- Problem: Dynamic num_workers selection (`src_v2/cli.py:107-151`) falls back to 0 on multiprocessing issues, silently degrading performance
- Files: `src_v2/cli.py:107-151`
- Cause: Sandbox/platform restrictions, fork context availability
- Improvement path: Log warning when falling back, provide user override via command-line flag, cache worker count decision

**Repeated GPA Computation for Large Datasets:**
- Problem: `gpa_iterative()` computes canonical shape from all training landmarks every session, even if cached result exists
- Files: `src_v2/processing/gpa.py:138-248`, `src_v2/cli.py` (compute-canonical command)
- Cause: No caching mechanism between runs
- Improvement path: Check for canonical shape cache file before computing, store with metadata (seed, data version)

**Inefficient Triangulation Recomputation in Warping:**
- Problem: When `use_full_coverage=True`, Delaunay is recomputed for each image (full triangulation on 23 points = significant overhead)
- Files: `src_v2/processing/warp.py:264-270`
- Cause: Extended landmarks (original 15 + 8 boundary points) need triangulation per image
- Improvement path: Cache triangulation for canonical shape, reuse for all images in batch, only recompute if input landmark config changes

## Fragile Areas

**Warping Pipeline Alignment:**
- Files: `src_v2/processing/warp.py`, `src_v2/gui/inference_pipeline.py`, `src_v2/data/transforms.py`
- Why fragile: Multiple coordinate systems (image pixels, normalized [0,1], canonical shape) with conversions happening in several places. Mismatch between landmark normalization in `transforms.py` and warping assumptions in `warp.py` could cause misalignment
- Safe modification: Always verify coordinate system explicitly (add assertions), test with synthetic grid overlay, include coordinate dumps in debug output
- Test coverage: No direct tests for coordinate transformation consistency; warping verification relies on visual inspection

**Ensemble Landmark Prediction with TTA:**
- Files: `src_v2/evaluation/metrics.py` (TTA + symmetric pair handling), `src_v2/cli.py` (ensemble averaging)
- Why fragile: Test-Time Augmentation averages predictions including horizontal flips. Symmetric pair correction (`SYMMETRIC_PAIRS`) is applied in `compute_pixel_error()` but logic could be duplicated or forgotten in other aggregation paths
- Safe modification: Extract TTA+averaging into reusable function, add validation that ensemble output shape matches single output
- Test coverage: No unit tests for TTA symmetric pair averaging

**Category Weight Balancing:**
- Files: `src_v2/data/dataset.py:27-50` (compute_sample_weights), `src_v2/data/dataset.py:281-302` (WeightedRandomSampler usage)
- Why fragile: Category weights (`DEFAULT_CATEGORY_WEIGHTS`) are baked into defaults but easily forgotten when manual splits are created. WeightedRandomSampler requires consistent weight ordering
- Safe modification: Always validate that weight keys match category names, add assertions on sampler initialization, test with unbalanced dataset
- Test coverage: Dataset splitting is not covered by tests

**CLI Parameter Propagation:**
- Files: `src_v2/cli.py` (parameter parsing), `src_v2/data/dataset.py`, `src_v2/training/trainer.py`
- Why fragile: 40+ CLI commands pass parameters through multiple layers; missing parameter in one layer silently uses default from next layer, leading to non-deterministic behavior
- Safe modification: Create parameter dataclass for each command, validate all required params before execution, log final parameter values before running
- Test coverage: No tests for CLI parameter propagation

## Scaling Limits

**GPU Memory Usage in Ensemble Inference:**
- Current capacity: Batch size 16 on 8GB GPU with single model; ensemble of 3-4 models runs sequentially to avoid OOM
- Limit: Concurrent model loading for ensemble inference not feasible on modest GPUs; batch size scales inversely with model count
- Scaling path: Implement model quantization (int8), use gradient checkpointing if training, support model sharding across devices

**Storage for Cached Predictions:**
- Current capacity: Landmark predictions cached in `.npz` format (~100 MB for 5000 images)
- Limit: No cleanup mechanism; cache grows unbounded if multiple prediction sessions run
- Scaling path: Implement cache versioning, add TTL for cached files, provide cleanup command

**Large Dataset Training Memory:**
- Current capacity: Train/val/test splits use standard DataLoader with num_workers=4, pin_memory=True
- Limit: Memory usage grows with num_workers; on 16GB system, safe maximum ~8 workers before swap
- Scaling path: Profile actual memory usage, add adaptive num_workers based on available RAM, implement gradient accumulation

## Dependencies at Risk

**NumPy >= 2.0.0 Compatibility:**
- Risk: `pyproject.toml` specifies `numpy>=2.0.0`, which has breaking changes in type handling and operations. Code using numpy 1.x idioms may fail silently
- Impact: Type casting, random number generation, indexing behavior changed between versions
- Migration plan: Run comprehensive tests with numpy 2.x, update deprecated patterns (e.g., `np.bool_` → `np.bool`), validate with CI

**PyTorch 2.0+ Dynamic Shapes:**
- Risk: Code relies on static shape assumptions; PyTorch 2.0 compiled graphs may optimize incorrectly for dynamic shapes
- Impact: Warping code assumes (H, W, C) structure; if image dimensions vary, compiled code could be inefficient or incorrect
- Migration plan: Test with `torch.compile()`, add shape assertions, validate with dynamic input sizes

**Scipy Delaunay API Changes:**
- Risk: `scipy.spatial.Delaunay` API has changed between versions; input validation and output format may differ
- Impact: Triangulation could fail silently or produce unexpected results with newer scipy
- Migration plan: Pin scipy version if needed, test with range of scipy versions, add explicit version check on import

## Missing Critical Features

**No Checkpoint Backup/Recovery Mechanism:**
- Problem: Training checkpoints are saved but no automated backup or recovery; disk full or corruption loses weeks of training
- Blocks: Reproducibility, long-running experiments, production deployment
- Recommendation: Implement checkpoint shadowing (backup to secondary location), version control for checkpoints, checksum validation

**No Cross-Validation Support:**
- Problem: Current train/val/test split is fixed (80/10/10); k-fold cross-validation not implemented
- Blocks: Robust model comparison, parameter tuning without overfitting to single validation split
- Recommendation: Add k-fold command, store fold assignments persistently, aggregate metrics across folds

**No Configuration Validation Schema:**
- Problem: JSON configs (`configs/*.json`) lack validation; typos or invalid values silently use defaults
- Blocks: Configuration reproducibility, catching user errors early
- Recommendation: Implement pydantic schemas for each config type, validate on load, provide detailed error messages

**No Experiment Tracking/Logging:**
- Problem: Results scattered across command outputs and files; no centralized experiment metadata (hyperparams, metrics, timestamp, environment)
- Blocks: Systematic comparison, reproduction of specific runs
- Recommendation: Integrate MLflow or custom experiment logger, save run config + metrics + environment info

## Test Coverage Gaps

**Warping Coordinate System Transformation:**
- What's not tested: End-to-end coordinate transformation (image → normalized → canonical → warped). No validation that output coordinates align with input shape
- Files: `src_v2/processing/warp.py`, `src_v2/data/transforms.py`
- Risk: Systematic coordinate offset could be undetected across entire pipeline
- Priority: High (affects all downstream tasks)

**Ensemble TTA Symmetric Pair Handling:**
- What's not tested: Correctness of symmetric pair swapping when averaging predictions from flipped images
- Files: `src_v2/evaluation/metrics.py:compute_pixel_error()` (symmetric pair logic)
- Risk: Incorrect landmark averaging could degrade ensemble accuracy silently
- Priority: High (core functionality)

**Data Loading Robustness:**
- What's not tested: Missing images, corrupted files, size mismatches, invalid landmarks. Error handling in `LandmarkDataset.__getitem__()` has edge cases
- Files: `src_v2/data/dataset.py:91-145`
- Risk: Pipeline crashes mid-training on bad data
- Priority: Medium (edge case handling)

**CLI Parameter Propagation:**
- What's not tested: Parameters passed through multiple function calls without validation; missing params silently use defaults
- Files: `src_v2/cli.py` (entire file)
- Risk: Non-deterministic behavior, hard to debug
- Priority: Medium (development friction)

**Model Checkpoint Compatibility:**
- What's not tested: Loading checkpoints from different training sessions with varying architectures (coord_attention yes/no, deep_head yes/no)
- Files: `src_v2/cli.py:154-198` (architecture detection)
- Risk: Architecture mismatch undetected until forward pass fails
- Priority: Medium (deployment risk)

---

*Concerns audit: 2026-01-27*
