---
phase: 03-tta-integration
verified: 2026-01-28T04:52:16Z
status: passed
score: 5/5 must-haves verified
re_verification:
  previous_status: gaps_found
  previous_score: 4/5
  gaps_closed:
    - "Case-level impact tracking categorizes each sample as helped/hurt/neutral"
  gaps_remaining: []
  regressions: []
---

# Phase 3: TTA Integration Verification Report

**Phase Goal:** Add conservative test-time augmentation with horizontal flip and expand metrics
**Verified:** 2026-01-28T04:52:16Z
**Status:** passed
**Re-verification:** Yes — gap closure verification after plan 03-03

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Horizontal flip TTA implemented (no symmetry correction needed) | ✓ VERIFIED | `torch.flip(images, dims=[3])` at line 234 of ensemble.py, no symmetry mapping (anatomically symmetric classes) |
| 2 | TTA configuration controlled via JSON with CLI override | ✓ VERIFIED | `use_tta: true` in config, `--tta/--no-tta` flag in CLI (line 2631-2635), resolution logic at line 2692 |
| 3 | Per-class breakdown computed for all predictions | ✓ VERIFIED | COVID/Normal/Viral_Pneumonia metrics in output JSON with precision/recall/F1 for each class |
| 4 | Case-level impact tracking categorizes each sample as helped/hurt/neutral | ✓ VERIFIED | Functions imported (line 2672-2673), called (lines 2811, 2837), results in output JSON |
| 5 | Ensemble+TTA improves over ensemble-only baseline | ✓ VERIFIED | TTA: 98.26% vs Baseline: 98.10% = +0.16pp improvement (measured from actual results files) |

**Score:** 5/5 truths verified (gap closed)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src_v2/evaluation/ensemble.py` | TTA prediction functions | ✓ VERIFIED | `predict_with_tta_classifier` (lines 203-241), `ensemble_inference_with_tta` (lines 245-316) |
| `configs/ensemble_classifier.json` | `use_tta` parameter | ✓ VERIFIED | Line 3: `"use_tta": true` |
| `src_v2/cli.py` | `--tta/--no-tta` flags | ✓ VERIFIED | Lines 2631-2635: `tta: Optional[bool]` with `--tta/--no-tta` |
| `src_v2/evaluation/ensemble.py` | `categorize_tta_impact` function | ✓ VERIFIED | Exists (lines 380-427) AND imported/called in CLI |
| `src_v2/evaluation/ensemble.py` | `compute_tta_delta_metrics` function | ✓ VERIFIED | Exists (lines 430-453) AND imported/called in CLI |
| `outputs/classifier_cv/ensemble_test_results_tta.json` | Full TTA evaluation results | ✓ VERIFIED | File exists with tta_enabled=true, 1895 samples, per-class metrics |
| `GROUND_TRUTH.json` | Validated TTA metrics | ✓ VERIFIED | Has `with_tta` section with accuracy/F1/case_level_impact (values match code output) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| CLI config | ensemble.py TTA functions | `ensemble_inference_with_tta` call | ✓ WIRED | Line 2747: function called with `use_tta` parameter |
| Config JSON | CLI TTA setting | `use_tta` parameter read | ✓ WIRED | Line 2692: `config_data.get("use_tta", True)` |
| CLI flag | Config override | `tta if tta is not None else config` | ✓ WIRED | Line 2692: proper override logic |
| TTA functions | torch.flip | Horizontal flip call | ✓ WIRED | Line 234: `torch.flip(images, dims=[3])` |
| Ensemble inference | TTA details | Return value | ✓ WIRED | Returns `tta_details` dict with original/flipped probs |
| **CLI** | **categorize_tta_impact** | **Function import + call** | **✓ WIRED** | Imported line 2672, called line 2811 with baseline/TTA preds |
| **CLI** | **compute_tta_delta_metrics** | **Function import + call** | **✓ WIRED** | Imported line 2673, called line 2837 with metrics dicts |
| **CLI** | **Output JSON** | **case_level_analysis field** | **✓ WIRED** | Lines 2891-2894: conditionally added when use_tta=True |
| **CLI** | **Output JSON** | **tta_delta_metrics field** | **✓ WIRED** | Line 2895: delta_metrics added to output_data |

### Requirements Coverage

**Phase 3 Success Criteria (from ROADMAP.md):**

| Criterion | Status | Evidence/Blocker |
|-----------|--------|------------------|
| 1. Horizontal flip TTA implemented | ✓ SATISFIED | torch.flip at line 234, dual-level averaging (model + ensemble) |
| 2. TTA config via JSON with CLI override | ✓ SATISFIED | use_tta in config, --tta/--no-tta flags, proper resolution logic |
| 3. Per-class breakdown computed | ✓ SATISFIED | COVID/Normal/Viral_Pneumonia metrics in all output files |
| 4. Case-level impact tracking (helped/hurt/neutral) | ✓ SATISFIED | Functions called, baseline computed, results in output JSON |
| 5. Ensemble+TTA improves over baseline | ✓ SATISFIED | +0.16pp improvement (98.10% → 98.26%) validated |

**Mapped Requirements (from REQUIREMENTS.md):**

| Requirement | Status | Evidence |
|-------------|--------|----------|
| TTA-01: Horizontal flip with symmetry correction | ✓ SATISFIED | Horizontal flip implemented (no correction needed for symmetric class labels) |
| TTA-02: JSON-driven configuration | ✓ SATISFIED | `use_tta` parameter in config + CLI override |
| METRICS-03: Per-class breakdown | ✓ SATISFIED | COVID/Normal/Viral_Pneumonia precision/recall/F1 in output |
| METRICS-04: Confusion matrix | ✓ SATISFIED | Confusion matrix in output JSON |

### Anti-Patterns Found

No blockers or warnings. Previous dead code issue (orphaned functions) has been resolved.

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | - | - | - | All functions properly wired |

## Re-verification Summary

**Previous Gap (from initial verification):**

Truth #4: "Case-level impact tracking categorizes each sample as helped/hurt/neutral" was marked as PARTIAL because:
- ✓ Functions `categorize_tta_impact` and `compute_tta_delta_metrics` existed
- ✗ Functions were NOT imported or called in CLI
- ✗ Output JSON missing `case_level_analysis` and `tta_delta_metrics` fields

**Gap Closure (Plan 03-03):**

1. **Imports added:** Lines 2672-2673 in cli.py
   ```python
   categorize_tta_impact,
   compute_tta_delta_metrics,
   ```

2. **Baseline computation added:** Lines 2755-2761
   - Second inference pass without TTA when use_tta=True
   - Required for helped/hurt/neutral categorization
   - Note: Doubles inference time when TTA enabled

3. **Functions called:** Lines 2811 and 2837
   - `categorize_tta_impact()` called with baseline vs TTA predictions
   - `compute_tta_delta_metrics()` called with baseline vs TTA metrics
   - Results logged and stored

4. **Output JSON updated:** Lines 2891-2895
   - `case_level_analysis` includes summary counts + net_improvement
   - `tta_delta_metrics` includes accuracy/F1 deltas per-class
   - Conditionally added only when use_tta=True

**Verification of Gap Closure:**

✓ **Import verification:**
```bash
$ grep -n "categorize_tta_impact\|compute_tta_delta" src_v2/cli.py
2672:        categorize_tta_impact,
2673:        compute_tta_delta_metrics,
2811:        case_level = categorize_tta_impact(
2837:        delta_metrics = compute_tta_delta_metrics(
```

✓ **Output JSON verification:**
```json
{
  "case_level_analysis": {
    "summary": {
      "helped": 6,
      "hurt": 3,
      "neutral": 1886
    },
    "net_improvement": 3
  },
  "tta_delta_metrics": {
    "accuracy_delta": 0.0016,
    "f1_macro_delta": 0.0009,
    "per_class_f1_delta": {
      "COVID": 0.0044,
      "Normal": 0.0012,
      "Viral_Pneumonia": -0.0028
    }
  }
}
```

✓ **Values match GROUND_TRUTH.json:** Case-level counts (6/3/1886) and deltas match exactly

✓ **No regressions:** All previously passing truths (#1-3, #5) remain verified

## Detailed Findings

### Truth 1: Horizontal flip TTA implemented ✓

**Evidence:**
```python
# src_v2/evaluation/ensemble.py:234
images_flipped = torch.flip(images, dims=[3])
```

**Verification:**
- Flip dimension `[3]` is width (correct for horizontal flip)
- No symmetry correction applied (anatomically symmetric classes)
- Softmax applied BEFORE averaging (not on logits)
- Simple average: `(probs_orig + probs_flip) / 2`

**Regression check:** ✓ No changes, still verified

### Truth 2: TTA config via JSON with CLI override ✓

**Config parameter:**
```json
// configs/ensemble_classifier.json:3
"use_tta": true
```

**CLI flags:**
```python
# src_v2/cli.py:2631-2635
tta: Optional[bool] = typer.Option(
    None,
    "--tta/--no-tta",
    help="Enable/disable TTA (overrides config use_tta if provided)"
)
```

**Regression check:** ✓ No changes, still verified

### Truth 3: Per-class breakdown computed ✓

**Evidence from outputs/classifier_cv/ensemble_test_results_tta.json:**
```json
"per_class": {
  "COVID": {
    "precision": 0.9910, "recall": 0.9757, "f1-score": 0.9833
  },
  "Normal": {
    "precision": 0.9829, "recall": 0.9922, "f1-score": 0.9875
  },
  "Viral_Pneumonia": {
    "precision": 0.9573, "recall": 0.9290, "f1-score": 0.9429
  }
}
```

**Regression check:** ✓ No changes, still verified

### Truth 4: Case-level impact tracking ✓ [GAP CLOSED]

**Previous status:** ✗ PARTIAL (functions existed but not called)

**Current status:** ✓ VERIFIED (fully wired)

**Evidence:**

1. **Baseline computation (required for comparison):**
```python
# src_v2/cli.py:2755-2761
if use_tta:
    logger.info("Computing baseline (no-TTA) predictions for case-level comparison...")
    logger.info("NOTE: This runs a second inference pass without TTA (doubles inference time)")
    baseline_model_preds, baseline_model_probs, _, _ = ensemble_inference_with_tta(
        models, test_loader, torch_device, use_tta=False
    )
    baseline_soft_preds, _ = weighted_soft_voting(baseline_model_probs, weights)
    baseline_soft_preds_np = baseline_soft_preds.numpy()
```

2. **Case-level categorization call:**
```python
# src_v2/cli.py:2811-2815
case_level = categorize_tta_impact(
    pred_baseline=baseline_soft_preds_np,
    pred_tta=soft_preds_np,
    ground_truth=labels
)
```

3. **Delta metrics call:**
```python
# src_v2/cli.py:2837-2841
delta_metrics = compute_tta_delta_metrics(
    baseline_metrics=baseline_metrics_dict,
    tta_metrics=tta_metrics_dict,
    class_names=class_names,
)
```

4. **Output JSON inclusion:**
```python
# src_v2/cli.py:2891-2895
if use_tta:
    output_data["case_level_analysis"] = {
        "summary": case_level["summary"],
        "net_improvement": case_level["summary"]["helped"] - case_level["summary"]["hurt"],
    }
    output_data["tta_delta_metrics"] = delta_metrics
```

**Verification results:**
- Helped: 6 samples (baseline wrong, TTA correct)
- Hurt: 3 samples (baseline correct, TTA wrong)
- Neutral: 1,886 samples (same outcome)
- Net improvement: +3 samples
- Accuracy delta: +0.0016 (+0.16pp)
- F1-Macro delta: +0.0009

**Gap closure confirmed:** All missing pieces now present and functioning.

### Truth 5: Ensemble+TTA improves over baseline ✓

**Measured results:**
- Baseline (no TTA): 98.10% accuracy
- With TTA: 98.26% accuracy
- **Improvement: +0.16pp** ✓

**Per-class deltas:**
- COVID: +0.44% F1 (helped most)
- Normal: +0.12% F1 (helped)
- Viral_Pneumonia: -0.28% F1 (slight degradation)

**Test set:** 1,895 samples (verified)

**Regression check:** ✓ No changes, still verified

## Technical Verification

### TTA Implementation Quality ✓

**Dual-level averaging:** ✓ Correct
1. Model-level: Each of 5 models averages orig+flip
2. Ensemble-level: 5 model TTA predictions are ensemble-averaged

**Flip correctness:** ✓ Verified
```python
torch.flip(images, dims=[3])  # Width dimension, correct for horizontal flip
```

**No symmetry correction:** ✓ Correct decision
- Class labels (COVID/Normal/Viral_Pneumonia) are anatomically symmetric
- Unlike landmarks where L3↔L4 swap needed, classes don't encode sidedness

**Probability averaging:** ✓ Correct
```python
probs_orig = torch.softmax(logits_orig, dim=1)
probs_flip = torch.softmax(logits_flip, dim=1)
tta_probs = (probs_orig + probs_flip) / 2
```

### Case-Level Impact Implementation ✓ [NEW]

**Baseline inference approach:** ✓ Correct
- When TTA enabled, runs TWO full inference passes:
  1. TTA pass (original + horizontal flip averaged)
  2. Baseline pass (original only, no TTA)
- Approximately doubles inference time when --tta is used
- Required for per-sample helped/hurt/neutral comparison

**Categorization logic:** ✓ Verified
```python
# categorize_tta_impact logic
helped = baseline wrong AND TTA correct
hurt = baseline correct AND TTA wrong
neutral = baseline and TTA have same outcome (both right or both wrong)
```

**Delta computation:** ✓ Verified
- Accuracy delta: TTA accuracy - baseline accuracy
- F1 delta: Per-class F1 differences
- All deltas match GROUND_TRUTH.json values

### Configuration System ✓

**Config structure:** ✓ Proper
- Top-level `use_tta` parameter
- Checkpoint paths for 5 CV folds
- Expected sample counts for validation

**CLI override pattern:** ✓ Robust
```python
tta: Optional[bool] = None  # None means "use config default"
use_tta = tta if tta is not None else config_data.get("use_tta", True)
```

### Output Completeness ✓ [IMPROVED]

**Present in outputs:**
- ✓ `tta_enabled` field
- ✓ Test set size (1,895)
- ✓ Per-fold metrics
- ✓ Ensemble soft voting metrics
- ✓ Per-class breakdown (COVID/Normal/Viral_Pneumonia)
- ✓ Confusion matrix
- ✓ **case_level_analysis** (summary + net_improvement) [NEW]
- ✓ **tta_delta_metrics** (accuracy/F1 deltas) [NEW]

**Nothing missing:** All planned outputs now present.

## Phase Success Evaluation

**5 success criteria from ROADMAP.md:**

1. ✓ Horizontal flip TTA implemented
2. ✓ TTA config via JSON with CLI override  
3. ✓ Per-class breakdown computed
4. ✓ Case-level impact tracking (functions wired and working)
5. ✓ Ensemble+TTA improves baseline

**5 of 5 criteria fully met.**

**Phase goal ACHIEVED:** TTA improves accuracy (+0.16pp), per-class metrics are computed, and case-level analysis is complete and integrated.

---

_Verified: 2026-01-28T04:52:16Z_
_Verifier: Claude (gsd-verifier)_
_Re-verification: Gap closure confirmed after plan 03-03_
