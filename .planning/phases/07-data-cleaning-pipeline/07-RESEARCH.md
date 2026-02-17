# Phase 7: Data Cleaning Pipeline - Research

**Researched:** 2026-02-17
**Domain:** Data quality remediation for medical imaging ML pipeline (landmark outlier filtering, duplicate deduplication, label noise detection, cleaning manifest)
**Confidence:** HIGH

## Summary

Phase 7 removes or corrects data quality issues discovered in Phase 6 before re-training. The work breaks into four distinct tracks: (1) outlier landmark filtering via Procrustes distance on the predicted landmark cache, (2) cross-split duplicate resolution at the original image level, (3) label noise detection via cleanlab using out-of-fold probabilities from the 5-fold classifier checkpoints, and (4) a structured JSON cleaning manifest with full traceability.

All required infrastructure already exists in the codebase. The landmark cache (`outputs/landmark_predictions/session_warping/predictions.npz`) contains 15,153 predicted shapes. The 5-fold classifier checkpoints exist in `outputs/classifier_cv/fold_{01-05}/best_classifier.pt`. The duplicate audit from Phase 6 produced `outputs/error_forensics/duplicates/cross_split_leakage.csv` with 17,312 cross-split pairs. The primary new work is: (a) adding outlier detection logic, (b) running cleanlab inference to gather out-of-fold probs (they were not saved during Phase 5/6 training), (c) building the manifest schema, and (d) adding an `--exclude` flag to `generate-dataset` for re-warping.

**Primary recommendation:** Build Phase 7 as two scripts + one new CLI parameter. Script 1 produces the cleaning manifest (auditing + decisions). Script 2 re-warps the cleaned dataset using existing `generate-dataset` with a new `--exclude-list` flag pointing to the manifest. The review tool should be a Jupyter notebook (consistent with Phase 6 tooling), with auto-exclude for high-confidence issues and a manual-review section for borderline cases.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Cross-split duplicate resolution:**
- Clean at the **original image level** (before warping) to address root cause — re-warp after cleaning
- Exclude uncertain/noisy samples rather than correcting labels
- **Prioritize data quality** over dataset size — acceptable to lose 10-15% of training data if it improves signal
- Report **both** original v1.0 accuracy (98.26%) AND corrected baseline after removing leakers, for transparent comparison in thesis

**Outlier landmark filtering:**
- Use **>3 sigma** threshold for automatic removal (auto-keep anything between 2-3 sigma, no manual review for borderline)
- Use **combined metric**: flag if overall Procrustes distance >3 sigma OR any single landmark >4 sigma — catches both globally distorted shapes and localized landmark errors

### Claude's Discretion
- **Duplicate resolution strategy**: Claude picks between removing from train only vs re-splitting, based on methodological rigor and v1.0 comparability
- **pHash threshold**: Claude adjusts from Phase 6's threshold=3 based on the distance distribution found during the audit
- **Review tool format**: Claude picks the review interface (notebook, HTML report, or CLI) based on what integrates best with existing Phase 6 tooling
- **Cleanlab target**: Claude decides whether to run cleanlab on warped or original images based on practicality (existing 5-fold predictions available for warped)
- **Auto-exclude threshold**: Claude sets the cleanlab confidence threshold for automatic exclusion vs manual review
- **Manifest scope**: Claude decides whether manifest documents only removed samples or full audit trail (all flagged, kept or removed)
- **Post-cleaning flow**: Claude decides whether to auto-re-warp or stop for user review of manifest before proceeding

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CLN-01 | System filters images with outlier landmarks (>3σ from canonical shape) before warping | GPA module (`src_v2/processing/gpa.py`) already computes `procrustes_distance()`. Combined metric: overall Procrustes >3σ OR per-landmark deviation >4σ. Canonical shape is at `outputs/shape_analysis/canonical_shape_gpa.json`. Predicted landmarks are in NPZ cache (15,153 shapes). |
| CLN-02 | System detects potential label noise using cleanlab confident learning on 5-fold CV predictions | cleanlab 2.9.0 is installable (not yet in requirements.txt). Requires `pred_probs` of shape `(N, K)` as out-of-fold probabilities. All 5 fold checkpoints exist (`outputs/classifier_cv/fold_{01-05}/best_classifier.pt`) but per-sample probs were NOT saved — must re-run inference on each fold's validation split to build OOF matrix. Decision: run cleanlab on **warped images** (matching the fold checkpoints' training distribution). |
| CLN-03 | Flagged samples undergo manual review with documented accept/reject decisions | Notebook-based review tool (consistent with Phase 6 interactive notebook). Auto-exclude high-confidence issues; manual review queue for borderline. Outputs decisions back to manifest. |
| CLN-04 | Data cleaning manifest (JSON) documents every excluded/corrected sample with reasoning | Manifest schema: list of entries per excluded/flagged image with fields: `image_name`, `category`, `reason` (enum), `flags` (list of triggered conditions), `decision` (`excluded`|`kept`|`review_pending`), `confidence`, `metadata`. Full audit trail — documents kept samples too. |
</phase_requirements>

## Claude's Discretion Recommendations

Based on the codebase investigation and Phase 6 findings, here are the recommended choices for Claude's discretion items:

### Duplicate Resolution Strategy: Re-split vs Remove from Train Only

**Recommendation: Full re-split from originals, removing identified duplicates.**

Rationale:
- Phase 6 found 17,312 cross-split duplicate pairs in the **warped** dataset (identified via PHash threshold=3 on warped images)
- The user decision is to clean at the **original image level** and re-warp — so we're not working with warped filenames as exclusions; we're going back to original filenames
- The cross_split_leakage.csv uses warped filenames (e.g., `train/Viral_Pneumonia/Viral Pneumonia-287_warped.png`) — we need to reverse-map to original image names for the manifest
- A full re-split ensures no methodological taint: the clean dataset has a fresh train/val/test assignment without any known leakers
- For comparability with v1.0: since v1.0 used seed=42 splits and the leakers are detected, we can report v1.0 accuracy on the "dirty split" as historical context, then re-train on the clean re-split

### pHash Threshold Adjustment

**Recommendation: Keep threshold=3 (Phase 6's conservative setting) for original image duplicate detection, but verify by checking the distance distribution histogram.**

Rationale:
- Phase 6 found 13,394 pairs in the original dataset (all within-class, no cross-split/cross-class issues) with threshold=3
- Cross-split leakage was only detected in the **warped** dataset (42,175 pairs), suggesting the warping is creating synthetic similarity
- For original image deduplication, the 13,394 within-class pairs at threshold=3 are the cleanup target — these are genuine duplicates that inflate the effective dataset size
- A threshold of 3 on original X-rays is conservative and appropriate given the high baseline similarity of chest X-rays

### Review Tool Format: Notebook

**Recommendation: Jupyter notebook** (consistent with Phase 6's `notebooks/error_forensics_interactive.ipynb`).

Rationale:
- Phase 6 already built a successful interactive notebook pattern with ipywidgets
- The review workflow involves viewing images + making accept/reject decisions — notebooks support this natively
- Alternatively, an HTML report is sufficient for the auto-excluded items (just a table); the notebook focuses on borderline/manual-review items

### Cleanlab Target: Warped Images

**Recommendation: Run cleanlab on **warped images** using the existing 5-fold classifier checkpoints.**

Rationale:
- The 5 fold checkpoints (`outputs/classifier_cv/fold_{01-05}/best_classifier.pt`) were trained on warped images
- These models have domain-appropriate feature spaces for warped chest X-rays
- The out-of-fold probability matrix can be reconstructed by running each fold model on its validation split
- Running cleanlab on original images would require either retraining new classifiers (expensive) or using the landmark models (wrong task)
- The warped dataset has 15,153 images across train/val/test — need OOF probs for all train+val samples

### Auto-Exclude Threshold for Cleanlab

**Recommendation: Auto-exclude if cleanlab `self_confidence` < 0.05 (very high confidence of label error). Flag for manual review if 0.05 ≤ self_confidence < 0.40.**

Rationale:
- At self_confidence < 0.05, the model ensemble unanimously and confidently predicted a different class — strong evidence of label error or severe image quality issue
- The 0.40 manual review boundary is generous enough to capture borderline cases without overwhelming the reviewer
- Given Phase 6 found only 33 misclassified test images (1.7%), the number of cleanlab-flagged items should be manageable for manual review

### Manifest Scope: Full Audit Trail

**Recommendation: Manifest documents ALL flagged samples (excluded AND kept), not just removed ones.**

Rationale:
- Full traceability is essential for thesis documentation and reproducibility
- The manifest serves as the single source of truth linking Phase 7 cleaning to Phase 8 training
- Documenting "kept after review" samples prevents re-investigation if someone asks why a borderline sample was retained

### Post-Cleaning Flow: Stop for Review Before Re-warping

**Recommendation: Stop after manifest generation and have the user review it before triggering re-warp.**

Rationale:
- The manifest defines what gets excluded — user should confirm the exclusion list looks reasonable
- Re-warping is fast (~0.6 minutes for 15,153 images, per dataset_summary.json) so the delay is negligible
- This creates a clean checkpoint: "manifest approved" → "re-warp triggered" → Phase 8 ready

## Standard Stack

### Core (Already in Project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | 2.1.2 | Procrustes distance computation, sigma thresholds | Foundation for landmark array operations |
| scipy | 1.16.2 | `procrustes_distance()` helper via `gpa.py` | Already used for GPA; `scipy.spatial` also available |
| pandas | 2.3.2 | Manifest as DataFrame, CSV I/O, cross_split_leakage.csv | Standard for tabular cleaning operations |
| torch | 2.4.1+rocm6.0 | Fold model inference for OOF probabilities | Existing ResNet-18 classifiers trained on warped images |
| torchvision | 0.19.1+rocm6.0 | ImageFolder loading for fold inference | Used in existing classifier evaluation |
| imagededup | 0.3.3.post2 | Duplicate detection (already used in Phase 6) | `src_v2/utils/duplicates.py` exists |
| pyiqa | 0.1.14.1 | Already installed; not needed for CLN-01/02/03/04 | Phase 6 quality assessment |
| matplotlib | existing | Visualization in review notebook | Phase 6 already used for pipeline traces |
| ipywidgets | existing | Interactive review notebook | Phase 6 used for error_forensics_interactive.ipynb |

### New Dependencies Needed
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| cleanlab | 2.9.0 | Confident learning for label noise detection (CLN-02) | Run `find_label_issues(labels, pred_probs)` |

**Installation:**
```bash
pip install cleanlab==2.9.0
# Add to requirements.txt
```

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| cleanlab | Manual confidence thresholding | cleanlab uses the full confident joint matrix; manual thresholding misses inter-class calibration |
| Jupyter notebook review | CLI interactive prompt | Notebook is far better for image inspection; CLI is acceptable for text-only review |
| Procrustes distance (GPA-based) | Per-landmark pixel deviation only | Combined metric catches both global shape distortion and localized errors |

## Architecture Patterns

### Recommended Project Structure
```
src_v2/
├── processing/
│   ├── gpa.py                   # EXISTING: procrustes_distance(), gpa_iterative()
│   └── outlier_detection.py     # NEW: landmark outlier detection + per-landmark sigma
├── data/
│   └── cleaning.py              # NEW: OOF probability extraction, cleanlab wrapper
└── cli.py                       # MODIFY: add --exclude-list to generate-dataset

scripts/
├── run_landmark_outlier_detection.py   # NEW: CLN-01 script
├── run_label_noise_detection.py        # NEW: CLN-02 script
└── generate_cleaning_manifest.py       # NEW: CLN-03/04 manifest assembly

notebooks/
└── data_cleaning_review.ipynb          # NEW: manual review tool (gitignored)

outputs/
└── data_cleaning/
    ├── landmark_outliers.csv            # Procrustes + per-landmark deviations
    ├── oof_probabilities.npz            # Out-of-fold probs for all train+val images
    ├── cleanlab_issues.csv              # Cleanlab output: issue score per sample
    ├── cross_split_exclusions.csv       # Original-level duplicates to exclude
    ├── cleaning_manifest.json           # CLN-04: full audit trail
    └── report_data_cleaning.md         # Summary report for thesis
```

### Pattern 1: Landmark Outlier Detection (CLN-01)

**What:** Load predicted landmarks from NPZ cache, compute Procrustes distance to canonical shape for each image, flag if >3 sigma overall OR any single landmark >4 sigma.

**Key implementation notes:**
- The NPZ has `landmarks` of shape `(15153, 15, 2)` and `image_names` + `categories`
- The canonical shape is at `outputs/shape_analysis/canonical_shape_gpa.json` — load as `(15, 2)` array
- `procrustes_distance()` in `src_v2/processing/gpa.py` computes a scale-invariant distance — this is appropriate since we want to detect shape distortion regardless of image scale
- Per-landmark deviation: after aligning each predicted shape to canonical via GPA, compute per-landmark Euclidean distance in normalized coordinates; flag if >4 sigma of per-landmark distribution

```python
# Source: src_v2/processing/gpa.py (existing) + new outlier logic
import numpy as np
from src_v2.processing.gpa import gpa_iterative, procrustes_distance

def detect_landmark_outliers(
    landmarks: np.ndarray,  # (N, 15, 2)
    image_names: np.ndarray,
    categories: np.ndarray,
    canonical_shape: np.ndarray,  # (15, 2)
    procrustes_sigma_threshold: float = 3.0,
    per_landmark_sigma_threshold: float = 4.0,
) -> pd.DataFrame:
    """
    Flag images whose predicted landmarks deviate from canonical shape.

    Combined metric: flag if:
      - procrustes_distance(landmarks_i, canonical) > mean + 3*std (overall shape)
      OR
      - max(per_landmark_deviation_i) > mean + 4*std (localized error)

    Returns DataFrame with columns:
      image_name, category, procrustes_distance, max_landmark_deviation,
      procrustes_sigma, max_landmark_sigma, flagged, flag_reason
    """
    N = len(landmarks)

    # Compute Procrustes distances for all shapes
    proc_dists = np.array([
        procrustes_distance(landmarks[i], canonical_shape)
        for i in range(N)
    ])

    # Align all shapes to canonical for per-landmark deviation
    # Run GPA including canonical as reference, get aligned shapes
    _, aligned_shapes, _ = gpa_iterative(landmarks, max_iterations=100)

    # Align canonical too
    from src_v2.processing.gpa import center_shape, scale_shape, align_shape
    canon_c, _ = center_shape(canonical_shape)
    canon_s, _ = scale_shape(canon_c)

    # Per-landmark deviation: Euclidean distance aligned_i - canonical
    per_lm_deviations = np.array([
        np.linalg.norm(aligned_shapes[i] - canon_s, axis=1).max()
        for i in range(N)
    ])  # (N,) — max deviation across all 15 landmarks for each image

    # Compute statistics
    proc_mean, proc_std = proc_dists.mean(), proc_dists.std()
    plm_mean, plm_std = per_lm_deviations.mean(), per_lm_deviations.std()

    proc_sigmas = (proc_dists - proc_mean) / proc_std
    plm_sigmas = (per_lm_deviations - plm_mean) / plm_std

    # Combined flagging
    flag_procrustes = proc_sigmas > procrustes_sigma_threshold
    flag_per_landmark = plm_sigmas > per_landmark_sigma_threshold
    flagged = flag_procrustes | flag_per_landmark

    flag_reasons = []
    for fp, fl in zip(flag_procrustes, flag_per_landmark):
        if fp and fl:
            flag_reasons.append("procrustes_and_per_landmark")
        elif fp:
            flag_reasons.append("procrustes_only")
        elif fl:
            flag_reasons.append("per_landmark_only")
        else:
            flag_reasons.append(None)

    return pd.DataFrame({
        'image_name': image_names,
        'category': categories,
        'procrustes_distance': proc_dists,
        'max_landmark_deviation': per_lm_deviations,
        'procrustes_sigma': proc_sigmas,
        'max_landmark_sigma': plm_sigmas,
        'flagged': flagged,
        'flag_reason': flag_reasons,
    })
```

### Pattern 2: Out-of-Fold Probability Extraction (CLN-02 prerequisite)

**What:** For each of the 5 folds, run the fold model on its validation split to get `pred_probs`. Assemble into a full `(N_train+N_val, 3)` OOF matrix that cleanlab requires.

**Key implementation notes:**
- Fold checkpoints: `outputs/classifier_cv/fold_{01-05}/best_classifier.pt`
- The fold models were trained with `seed=42` splits on the **warped** dataset at `outputs/warped_lung_best/session_warping/`
- Cross-validation used `sklearn.model_selection.train_test_split` with `random_state=42` — we need to reconstruct the same splits to know which images are in each fold's validation set
- The fold training used `torchvision.datasets.ImageFolder` ordering (sorted by class then filename alphabetically) — this is documented in Phase 6 UAT as the critical mapping bug
- OOF probs cover train+val but NOT test (test set stays untouched for final evaluation)
- Store OOF as NPZ: `{'image_names': ..., 'categories': ..., 'pred_probs': (N, 3), 'true_labels': ...}`

```python
# Source: existing src_v2 evaluation patterns + cleanlab API
import torch
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
from src_v2.models.classifier import ImageClassifier

def extract_oof_probabilities(
    warped_dataset_dir: str,  # outputs/warped_lung_best/session_warping
    fold_checkpoint_dirs: list,  # [fold_01/best_classifier.pt, ...]
    output_npz: str,
    batch_size: int = 32,
    device: str = 'cuda',
) -> np.ndarray:
    """
    Extract out-of-fold probabilities for cleanlab.

    For each fold k:
      1. Load fold_k checkpoint
      2. Determine which images were in fold_k's validation split
         (reconstruct the same 80/20 split from ImageFolder ordering)
      3. Run inference on validation images → pred_probs_k
      4. Assemble into full OOF matrix

    Returns:
        pred_probs: (N_train_val, 3) float array
        image_names: (N_train_val,) string array
        true_labels: (N_train_val,) int array
    """
    # Key challenge: reconstruct fold splits
    # The 5-fold CV used sklearn KFold or train_test_split on the ImageFolder ordering
    # Check cross_validation_results.json to understand split methodology
    ...

    # Load model - existing ImageClassifier (ResNet-18)
    model = ImageClassifier(num_classes=3, backbone='resnet18')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Run inference with softmax probabilities
    with torch.no_grad():
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()  # (batch, 3)
    ...
```

**CRITICAL INVESTIGATION NEEDED:** The fold split reconstruction. Check `outputs/classifier_cv/cross_validation_results.json` to understand exactly how the 5-fold split indices were defined. This determines whether we can accurately reconstruct which images belong to each validation fold.

### Pattern 3: Cleanlab Label Issue Detection (CLN-02)

**What:** Given OOF probabilities `(N, 3)` and integer labels `(N,)`, call `cleanlab.filter.find_label_issues()`.

```python
# Source: cleanlab 2.9.0 API (verified via docs.cleanlab.ai)
from cleanlab.filter import find_label_issues

def detect_label_noise(
    labels: np.ndarray,       # (N,) integer labels 0/1/2
    pred_probs: np.ndarray,   # (N, 3) OOF probabilities
    auto_exclude_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Detect label issues using confident learning.

    Args:
        labels: True integer labels (0=COVID, 1=Normal, 2=Viral_Pneumonia)
        pred_probs: Out-of-fold predicted probabilities (N, 3)
        auto_exclude_threshold: self_confidence below this → auto-exclude

    Returns:
        DataFrame with columns: image_name, true_label, self_confidence,
                                label_issue, decision, cleanlab_rank
    """
    # Get label issues sorted by self_confidence (ascending = most likely issue first)
    label_issue_indices = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by='self_confidence',
        filter_by='confident_learning',  # Standard confident learning method
    )

    # Build full result for all N samples
    n = len(labels)
    self_confidence = pred_probs[np.arange(n), labels]  # Model confidence in given label

    # Determine decisions
    decisions = []
    for i in range(n):
        if i in set(label_issue_indices):
            rank = list(label_issue_indices).index(i)
            if self_confidence[i] < auto_exclude_threshold:
                decisions.append('auto_excluded')
            else:
                decisions.append('manual_review')
        else:
            decisions.append('kept')

    return pd.DataFrame({
        'label_issue_idx': range(n),
        'true_label': labels,
        'self_confidence': self_confidence,
        'is_label_issue': [i in set(label_issue_indices) for i in range(n)],
        'decision': decisions,
    })
```

### Pattern 4: Cross-Split Duplicate Resolution (Locked Decision)

**What:** Map warped filename cross-split pairs → original image names → exclusion set. Strategy: full re-split from originals after removing duplicates.

**Key implementation notes:**
- `cross_split_leakage.csv` has warped filenames like `train/Viral_Pneumonia/Viral Pneumonia-287_warped.png`
- Reverse-map to original: strip `_warped` suffix and `{split}/` prefix → `Viral Pneumonia-287`
- The original dataset at `data/dataset/COVID-19_Radiography_Dataset/{class}/images/{name}.png` has these names
- For cross-split duplicates (same class, different splits): remove ALL occurrences and let re-split determine their new assignment with fresh seed
- For cross-split cross-class (different class AND different split, 6,026 pairs): these are the most critical — exclude from dataset entirely (label error candidates)
- The 13,394 within-class original duplicates (Phase 6 finding: all 13,394 pairs in original were within-class) need deduplication — keep one, exclude duplicates

```python
def resolve_cross_split_duplicates(
    cross_split_csv: str,  # outputs/error_forensics/duplicates/cross_split_leakage.csv
    strategy: str = 'full_resplit',  # 'full_resplit' or 'remove_from_train'
) -> set:
    """
    Map warped cross-split duplicates to original image names for exclusion.

    Strategy 'full_resplit':
      - For cross-split (same class): keep one instance (the one appearing earlier
        in sorted order), exclude its duplicates. Re-split will distribute kept images.
      - For cross-split cross-class: exclude ALL instances (both images). Label
        ambiguity makes these unreliable.
    """
    df = pd.read_csv(cross_split_csv)

    exclude_set = set()
    for _, row in df.iterrows():
        # Reverse-map to original name
        img1_name = _warped_to_original(row['img1'])
        img2_name = _warped_to_original(row['img2'])

        if row['class1'] != row['class2']:
            # Cross-class: exclude both (label ambiguity)
            exclude_set.add((img1_name, row['class1']))
            exclude_set.add((img2_name, row['class2']))
        else:
            # Same class: exclude one (the duplicate)
            # Keep the one with lower number (arbitrary but deterministic)
            exclude_set.add((img2_name, row['class2']))

    return exclude_set
```

### Pattern 5: Cleaning Manifest Schema (CLN-04)

**What:** JSON manifest documenting every decision for full traceability.

```json
{
  "schema_version": "v1",
  "generated_at": "2026-02-17T...",
  "summary": {
    "total_images_original": 15153,
    "total_excluded": 0,
    "excluded_landmark_outlier": 0,
    "excluded_cross_split_duplicate": 0,
    "excluded_cross_class_duplicate": 0,
    "excluded_label_noise": 0,
    "excluded_manual_review": 0,
    "total_kept": 0,
    "estimated_accuracy_impact": "..."
  },
  "thresholds": {
    "landmark_procrustes_sigma": 3.0,
    "landmark_per_landmark_sigma": 4.0,
    "cleanlab_auto_exclude_self_confidence": 0.05,
    "cleanlab_manual_review_self_confidence": 0.40,
    "phash_threshold": 3
  },
  "entries": [
    {
      "image_name": "COVID-287",
      "category": "COVID",
      "split_v1": "train",
      "decision": "excluded",
      "reasons": ["cross_split_duplicate"],
      "flags": {
        "procrustes_sigma": 1.2,
        "max_landmark_sigma": 0.8,
        "is_landmark_outlier": false,
        "cleanlab_self_confidence": 0.85,
        "is_cleanlab_issue": false,
        "is_cross_split_duplicate": true,
        "duplicate_partner": "COVID-1170",
        "duplicate_partner_split": "val"
      },
      "review_notes": null
    }
  ]
}
```

### Pattern 6: Exclusion List in generate-dataset (Re-warp)

**What:** Add `--exclude-list` parameter to existing `generate-dataset` CLI command that reads the cleaning manifest and skips excluded images during warping.

```python
# In src_v2/cli.py, add parameter to generate_dataset():
exclude_list: Optional[str] = typer.Option(
    None,
    "--exclude-list",
    help="Path to cleaning manifest JSON; excluded images are skipped during warping"
)

# In the generate_dataset() function body, before the warping loop:
excluded_images = set()
if exclude_list:
    with open(exclude_list) as f:
        manifest = json.load(f)
    excluded_images = {
        (entry['image_name'], entry['category'])
        for entry in manifest['entries']
        if entry['decision'] == 'excluded'
    }
    logger.info(f"Excluding {len(excluded_images)} images from manifest")
```

### Anti-Patterns to Avoid

- **Cleaning on warped images, then re-warping:** The locked decision is to clean at the original image level. Don't try to delete warped files and re-use the rest.
- **Using all 15,153 warped images for cleanlab without excluding test set:** Test set images must stay out of the OOF matrix — only train+val (13,258 images) go into cleanlab.
- **Re-running landmark inference:** The NPZ cache has all 15,153 predicted shapes already. Use it directly rather than re-running ensemble inference (4 models × ~15K images = slow).
- **Modifying `cross_validation_results.json` split indices:** Read only, never modify. The OOF reconstruction must exactly match the original training splits.
- **Auto-excluding all cleanlab-flagged items:** Cleanlab's default `filter_by='prune_by_noise_rate'` may flag ~1-3% of data. With 13,258 samples that's ~130-400 items. Auto-exclude only the very high confidence ones (self_confidence < 0.05).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Confident learning for label noise | Custom confidence thresholding | `cleanlab.filter.find_label_issues()` | Confident joint matrix handles multi-class calibration; hand-rolled thresholds miss inter-class contamination |
| Procrustes alignment | Custom SVD alignment | `src_v2/processing/gpa.py::procrustes_distance()` | Already validated, handles reflection correction |
| Duplicate detection | New similarity metric | Existing `src_v2/utils/duplicates.py` + Phase 6 CSV | Phase 6 already ran detection; reuse outputs |
| OOF probability matrix | Full 5-fold retrain | Inference-only on existing checkpoints | Checkpoints exist; only need forward pass on val splits |
| Manifest I/O | Custom binary format | Standard JSON | Human-readable, versionable, easily audited |

**Key insight:** The entire data cleaning toolkit is assembling existing components (GPA module, Phase 6 duplicate CSVs, fold checkpoints) with minimal new code. The hard part is accurately reconstructing fold splits for OOF extraction, not implementing algorithms from scratch.

## Common Pitfalls

### Pitfall 1: Fold Split Reconstruction Mismatch

**What goes wrong:** When reconstructing which images were in each fold's validation set, the reconstruction doesn't exactly match the original training splits, leading to OOF probabilities being computed on wrong images. Cleanlab receives corrupted data → bad label noise detection.

**Why it happens:** The 5-fold CV implementation used `sklearn.model_selection.KFold` or `train_test_split` — the exact method and seed must be reproduced. The `cross_validation_results.json` documents the fold structure but may not preserve image-level assignment.

**How to avoid:**
1. Read `outputs/classifier_cv/cross_validation_results.json` to find the exact fold split methodology
2. Verify reconstruction by checking that fold k's validation F1 from inference matches the stored `val_metrics.f1_macro` in `fold_k/results.json`
3. If reconstruction is impossible, re-run 5-fold CV from scratch on the warped dataset (slow but correct)

**Warning signs:** Reconstructed OOF predictions give >2% higher accuracy than the stored fold metrics.

### Pitfall 2: Test Set Contamination in Cleanlab

**What goes wrong:** Test set images are included in the OOF probability matrix. Cleanlab flags test images as label noise. Excluding them from the re-warped dataset changes the test set → can't compare with v1.0 baseline.

**Why it happens:** The warped dataset has 15,153 total images (train: 11,364 + val: 1,894 + test: 1,895). Cleanlab should only see train+val = 13,258.

**How to avoid:**
- Read `dataset_summary.json` to get exact split counts
- Filter OOF extraction to train+val images only
- Keep test set completely untouched throughout Phase 7

**Warning signs:** Cleanlab output CSV has 15,153 rows instead of ~13,258.

### Pitfall 3: Warped Filename → Original Filename Mapping Errors

**What goes wrong:** The `cross_split_leakage.csv` has warped filenames like `train/Viral_Pneumonia/Viral Pneumonia-287_warped.png`. Stripping `_warped` suffix doesn't always give the correct original name if the original had spaces or special characters.

**Why it happens:** The warping script appends `_warped` to the stem and may have normalized other naming conventions.

**How to avoid:**
- Use `images.csv` from the warped dataset (if it exists) as the definitive mapping from warped filename → original image name
- Verify that the reverse-mapped name actually exists in the original dataset directory
- Check `outputs/warped_lung_best/session_warping/train/images.csv` for the authoritative mapping

**Warning signs:** >5% of warped filenames fail to map to an original file.

### Pitfall 4: Sigma Threshold Computed on Outlier-Contaminated Distribution

**What goes wrong:** When computing mean + k*std for Procrustes distances, extreme outliers inflate the standard deviation, making the threshold too permissive and missing moderate outliers.

**Why it happens:** The sigma-based threshold assumes a roughly normal distribution; heavy outliers skew both mean and std.

**How to avoid:**
- Use robust statistics for threshold computation: median + k * MAD (Median Absolute Deviation) rather than mean + k * std
- OR compute mean/std on the central 95% of the distribution, then apply to all data
- Validate by visualizing the Procrustes distance distribution histogram

**Warning signs:** The 3-sigma threshold captures >10% of images (suggests highly contaminated distribution) or <0.1% (suggests the threshold is too loose).

### Pitfall 5: cleanlab pred_probs Not Calibrated (Out-of-Distribution Confidence)

**What goes wrong:** The fold classifiers produce very high-confidence predictions (typical for deep neural networks without temperature scaling). Cleanlab's thresholds are calibrated for well-calibrated probabilities; overconfident models may miss label noise.

**Why it happens:** ResNet-18 classifiers without temperature scaling produce overconfident softmax outputs, especially at 99.10% accuracy.

**How to avoid:**
- Check prediction confidence distribution before running cleanlab
- If >90% of samples have max_prob > 0.99, apply temperature scaling (T=1.5-3) to soften probabilities
- Report whether temperature scaling was applied in the manifest

**Warning signs:** Cleanlab flags 0 issues despite known data quality problems.

### Pitfall 6: Re-Warp Changes Split Seed, Breaking v1.0 Comparison

**What goes wrong:** The re-warped cleaned dataset uses a different seed or different total image count, so train/val/test splits are completely different from v1.0. Comparison of v1.0 accuracy (98.26%) with v1.1 accuracy is confounded by split changes.

**Why it happens:** The locked decision is to clean at the original level and re-warp. Re-warping recreates splits from scratch, inevitably producing different assignments.

**How to avoid:**
- Accept that splits WILL change (this is methodologically correct — the v1.0 splits were contaminated)
- Report clearly: "v1.0 used contaminated splits with seed=42; v1.1 uses clean re-split with seed=42_clean"
- The v1.0 baseline accuracy (98.26%) reported in thesis should be labeled "on contaminated test set"
- The v1.1 baseline is reported after re-training on the cleaned, re-split dataset

## Code Examples

Verified patterns from official sources and existing project code:

### Load Canonical Shape from JSON

```python
# Source: src_v2/cli.py pattern for canonical shape loading
import json
import numpy as np

def load_canonical_shape(canonical_json_path: str) -> np.ndarray:
    """Load canonical shape from GPA output JSON."""
    with open(canonical_json_path) as f:
        data = json.load(f)
    # Stored as list of [x, y] pairs
    return np.array(data['canonical_shape'])  # (15, 2)
```

### Load Landmark Predictions from NPZ

```python
# Source: NPZ structure verified by inspection
# Keys: image_paths, image_names, categories, landmarks, metadata_json
import numpy as np

def load_landmark_predictions(npz_path: str):
    """Load landmark predictions cache."""
    data = np.load(npz_path, allow_pickle=True)
    return {
        'landmarks': data['landmarks'],       # (N, 15, 2)
        'image_names': data['image_names'],   # (N,)
        'categories': data['categories'],     # (N,)
    }
```

### Cleanlab find_label_issues (Verified API)

```python
# Source: docs.cleanlab.ai/stable (cleanlab 2.9.0)
# pred_probs: (N, K) out-of-fold probabilities
# labels: (N,) integer labels
from cleanlab.filter import find_label_issues

# Returns indices of label issues, ranked by self_confidence (most likely first)
issue_indices = find_label_issues(
    labels=labels,
    pred_probs=pred_probs,
    return_indices_ranked_by='self_confidence',
    filter_by='confident_learning',
)

# Get per-sample self_confidence (model's probability of the given label)
self_confidence = pred_probs[np.arange(len(labels)), labels]
```

### Check cross_validation_results.json Fold Structure

```python
# Source: actual file structure verified
import json

with open('outputs/classifier_cv/cross_validation_results.json') as f:
    cv_results = json.load(f)
print('Keys:', list(cv_results.keys()))
# Need to check if fold image assignments are stored
```

### Existing Classifier Loading Pattern

```python
# Source: src_v2 evaluate-classifier command in cli.py
import torch
from src_v2.models.classifier import ImageClassifier

def load_fold_classifier(checkpoint_path: str, device: str = 'cuda') -> ImageClassifier:
    """Load a trained fold classifier from checkpoint."""
    model = ImageClassifier(num_classes=3, backbone='resnet18')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    return model
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual label inspection | Confident learning (cleanlab) | ~2021 | Systematic, scalable detection of noisy labels using model predictions |
| Fixed sigma thresholds | Robust statistics (MAD) for outlier detection | ~2020 | Less sensitive to contamination of the reference distribution |
| Delete-only manifest | Full audit trail JSON | ~2022 | Reproducibility and thesis documentation |
| Single-stage cleaning | Pipeline-ordered cleaning (landmarks → duplicates → label noise) | Current best practice | Each stage inputs feed subsequent stages; avoids double-counting exclusions |

## Open Questions

1. **Fold Split Reconstruction**
   - What we know: 5-fold checkpoints exist. `cross_validation_results.json` exists.
   - What's unclear: Does `cross_validation_results.json` store per-image fold assignments? If not, OOF extraction requires re-running cross-validation from the ImageFolder ordering.
   - Recommendation: Read `cross_validation_results.json` first. If fold-image mapping is stored, use it. If not, reconstruct from `datasets.ImageFolder` with same directory + seed=42 KFold splitting. Verify via F1 match check.
   - **Must verify before planning CLN-02 tasks.**

2. **Magnitude of Landmark Outliers**
   - What we know: The GPA test run on 957 randomly-generated shapes gave mean≈1.40, std≈0.13. Real predicted shapes may have different distribution.
   - What's unclear: How many of the 15,153 predicted landmark sets will fall >3 sigma? If it's >5% (750+ images), the threshold may need adjustment.
   - Recommendation: Run the Procrustes distance computation on the full 15,153 NPZ as the first task; inspect the distribution before committing to thresholds.

3. **Warped Filename → Original Filename Mapping**
   - What we know: The warped filenames are `{image_name}_warped.png`. The `cross_split_leakage.csv` uses full relative paths.
   - What's unclear: Whether the `images.csv` files in the warped dataset directory provide the authoritative mapping.
   - Recommendation: Check `outputs/warped_lung_best/session_warping/train/images.csv` before building the reverse-mapping logic. Use this as the ground truth mapping, not string manipulation.

4. **cleanlab on 13,258 Warped Images vs 957 GT-Labeled Images**
   - What we know: The 5-fold classifiers were trained on warped images. GT landmarks cover only 957 images (manual labels). The warped dataset covers 15,153 images (ensemble predicted landmarks).
   - What's unclear: Whether cleanlab should run on the full train+val (13,258 warped) or only the GT-labeled subset (957 images, but the classifier was trained on all of them).
   - Recommendation: Run cleanlab on the full train+val set (13,258 images). The GT landmark subset is irrelevant to label noise detection — cleanlab detects label (COVID/Normal/Viral_Pneumonia) noise, not landmark quality issues.

## Sources

### Primary (HIGH confidence)
- Verified codebase: `src_v2/processing/gpa.py` — `procrustes_distance()` API and `gpa_iterative()` function signatures
- Verified codebase: `outputs/landmark_predictions/session_warping/predictions.npz` — NPZ structure: `landmarks (15153, 15, 2)`, `image_names`, `categories`, `metadata_json`
- Verified codebase: `outputs/classifier_cv/fold_{01-05}/best_classifier.pt` — all 5 fold checkpoints exist
- Verified codebase: `outputs/error_forensics/duplicates/cross_split_leakage.csv` — 17,312 rows, columns: img1, img2, hash_distance, cnn_similarity, split1, split2, class1, class2
- Verified codebase: `outputs/warped_lung_best/session_warping/dataset_summary.json` — train:11,364 / val:1,894 / test:1,895
- Verified codebase: `src_v2/utils/duplicates.py` — `detect_duplicates()`, `classify_duplicate_types()`, `compare_original_vs_warped()` all exist
- [cleanlab 2.9.0 `find_label_issues` API](https://docs.cleanlab.ai/stable/cleanlab/filter.html) — verified: `pred_probs (N, K)`, `labels (N,)`, `return_indices_ranked_by='self_confidence'`
- Verified package: `cleanlab==2.9.0` installable (all dependencies already met: numpy, scikit-learn, tqdm, pandas)

### Secondary (MEDIUM confidence)
- Phase 6 SUMMARY.md (06-02): `imagededup` PHash threshold=3 detected 13,394 original pairs (all within-class), 42,175 warped pairs (17,312 cross-split)
- Phase 6 UAT finding: warped filenames use `{image_name}_warped.png` format; ImageFolder ordering is sorted by class then filename
- Phase 6 06-02-SUMMARY.md: CNN verification was skipped; PHash-only detection may have false positives but conservative threshold minimizes this

### Tertiary (LOW confidence)
- Assumption that `cross_validation_results.json` does NOT store per-image fold assignments (based on key inspection showing only aggregate metrics) — **must verify in planning**
- Temperature scaling recommendation for cleanlab — depends on actual probability distribution observed during OOF extraction

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries verified as installed or installable; no new risky dependencies
- Architecture (landmark outlier detection): HIGH — GPA module verified, NPZ structure confirmed
- Architecture (cleanlab OOF): MEDIUM — cleanlab API verified but fold split reconstruction method needs confirmation
- Architecture (duplicate resolution): HIGH — CSV structure confirmed, reverse-mapping logic is straightforward
- Architecture (manifest): HIGH — JSON schema is a design decision, no external constraints
- Pitfalls: HIGH — documented based on actual codebase investigation and Phase 6 lessons learned

**Research date:** 2026-02-17
**Valid until:** ~60 days (stable domain; cleanlab and scipy are mature; primary risk is fold split reconstruction which needs one-time investigation)
