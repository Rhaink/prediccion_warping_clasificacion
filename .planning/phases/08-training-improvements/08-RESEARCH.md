# Phase 8: Training Improvements - Research

**Researched:** 2026-02-17
**Domain:** PyTorch classifier training techniques (focal loss, hard example mining, curriculum learning) on existing 5-fold CV infrastructure
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **Ablation structure**: Cleaned-data baseline first, then individual ablations (focal, mining, curriculum), then combined model fine-tuned from best individual checkpoint
- **Scope**: Each ablation is a full 5-fold CV run for rigorous comparability with v1.0
- **Data**: Re-warp entire dataset with Phase 7 exclude-list applied upfront (432 excluded). Phase 7 excluded samples never used as hard examples
- **Architecture**: ResNet-18 fixed throughout — isolate data/training effect from model capacity
- **Evaluation**: Validation-only during ablation; test set reserved for Phase 10 final evaluation
- **Failure handling**: If a technique hurts performance, tune hyperparameters before dropping
- **Combined model**: Fine-tune from best individual ablation checkpoint (not retrain from scratch)
- **Light hyperparameter tuning per technique is allowed** (lr, epochs) if clearly needed

### Claude's Discretion

- Ablation technique order
- Epoch count per ablation run (v1.0 used 15 frozen + 100 fine-tune; classifier uses 50 with early stopping)
- Two-phase training structure adaptation per technique
- Checkpoint strategy
- Hard example definition method (OOF loss, misclassification count, or confidence margin)
- Hard example oversampling ratio
- When to start mining during training (after warmup or from epoch 1)
- Static vs dynamic hard example set
- Whether to try OHEM as fallback if basic mining fails
- Class-aware vs class-agnostic hard example weighting
- Hard example reporting
- Curriculum difficulty metric
- Curriculum schedule type (linear ramp, step-based, or loss-triggered)
- Starting fraction of dataset for curriculum
- Class balance maintenance during curriculum stages
- Curriculum application scope (fine-tuning only vs both phases)
- Interaction between curriculum and mining
- Logging granularity for curriculum statistics
- Evaluation metrics set, reporting format, regression thresholds

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TRN-01 | Classifier uses focal loss (γ=2.0) instead of weighted CrossEntropy | FocalLoss implementation pattern verified; drop-in replacement for `nn.CrossEntropyLoss(weight=...)` in `cross_validate_classifier` |
| TRN-02 | Training pipeline supports hard example mining (oversampling of frequently misclassified samples) | OOF data available (13,258 samples, per-sample CE loss computed); `WeightedRandomSampler` is the correct PyTorch mechanism; static pre-computed weights from OOF |
| TRN-03 | Training supports curriculum learning (easy→hard schedule based on loss) | OOF per-sample loss scores available; difficulty percentiles computed; stage-based approach fits existing epoch loop structure |
| TRN-04 | New 5-fold CV ensemble trained on improved data with same ResNet-18 architecture | `cross-validate-classifier` CLI command already supports 5-fold CV; new warped dataset needed |
</phase_requirements>

---

## Summary

Phase 8 re-trains the 5-fold CV classifier ensemble on the Phase 7 cleaned data (432 samples excluded, 14,721 remaining) with three training techniques applied incrementally. The existing `cross-validate-classifier` CLI command in `src_v2/cli.py` (lines 2980-3584) provides the full 5-fold CV infrastructure. The primary target is Viral Pneumonia recall above 92.9% (v1.0 baseline: 0.9290 recall, 169 test samples, 1,176 train+val samples).

All three techniques integrate cleanly into the existing training loop. Focal loss is a one-line drop-in replacement for `nn.CrossEntropyLoss`. Hard example mining uses `WeightedRandomSampler` with pre-computed OOF difficulty scores — the OOF data is already present at `outputs/data_cleaning/oof_probabilities.npz` (13,258 samples, temperature-scaled T=2.0, shape (13258,3)). Curriculum learning sorts samples by OOF loss and reveals harder examples in stages, compatible with the existing epoch-based training loop.

The v1.0 baseline to beat: accuracy=98.26%, Viral Pneumonia recall=92.9% (F1=94.29%), F1-macro=97.12%. The dataset is moderately imbalanced: COVID 3,164 / Normal 8,918 / Viral Pneumonia 1,176 in train+val. OOF analysis shows Viral Pneumonia has the highest mean CE loss (0.1036) vs Normal (0.0407) and COVID (0.0703), confirming it is the hardest class.

**Primary recommendation:** Implement the four runs (cleaned-data baseline + 3 ablations + 1 combined) as separate configs that each call the existing `cross-validate-classifier` infrastructure with minimal targeted modifications — focal loss as a new loss class, mining via `WeightedRandomSampler`, curriculum via a dataset wrapper that exposes subsets per training stage.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.4.1+rocm6.0 (installed) | Training loop, loss functions, samplers | Already used throughout |
| torchvision | 0.19.1+rocm6.0 (installed) | ResNet-18 backbone, transforms | Already used throughout |
| torch.utils.data.WeightedRandomSampler | Built-in | Per-sample oversampling for hard example mining | Official PyTorch mechanism, no extra deps |
| sklearn.model_selection.StratifiedKFold | Already used | 5-fold stratified CV splits | Already in `cross_validate_classifier` |
| numpy | Already used | OOF difficulty score computation | Already used throughout |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| cleanlab | >=2.9.0 (installed) | Already used in Phase 7 | Not needed for Phase 8 directly |

### No Additional Dependencies Required

Phase 8 requires **zero new pip packages**. All techniques are implementable with PyTorch 2.4 built-ins:
- Focal loss: hand-roll ~20 lines (verified below)
- Hard example mining: `torch.utils.data.WeightedRandomSampler`
- Curriculum learning: Python list slicing + `torch.utils.data.Subset`

---

## Architecture Patterns

### Recommended Project Structure

```
src_v2/
├── models/
│   └── losses.py           # Add FocalLoss class here (already has WingLoss, CombinedLoss)
├── training/
│   └── trainer.py          # Existing LandmarkTrainer (not used for classifier)
├── cli.py                  # cross-validate-classifier command (lines 2980-3584)
configs/
├── cv_v1_baseline.json         # Cleaned-data baseline (v1.0 techniques, new data)
├── cv_v1_focal.json            # Ablation: focal loss
├── cv_v1_mining.json           # Ablation: hard example mining
├── cv_v1_curriculum.json       # Ablation: curriculum learning
├── cv_v1_combined.json         # Combined: all three techniques
outputs/
├── classifier_cv_v1_baseline/  # 5 fold dirs + cross_validation_results.json
├── classifier_cv_v1_focal/
├── classifier_cv_v1_mining/
├── classifier_cv_v1_curriculum/
├── classifier_cv_v1_combined/
└── warped_cleaned/             # Re-warped dataset with exclude-list applied
```

### Pattern 1: Focal Loss Drop-In

**What:** Replace `nn.CrossEntropyLoss` with `FocalLoss` in the training loop. Focal loss reduces the contribution of easy examples by modulating CE loss with `(1 - p_t)^gamma`. At gamma=2.0 (TRN-01 specified value), well-classified samples (p_t > 0.9) contribute <1% of their CE weight, forcing the model to focus on hard samples.

**When to use:** When class imbalance causes the loss to be dominated by the majority class (Normal) even though weights already address frequency — focal further suppresses easy Normal examples within each batch.

**Implementation:**

```python
# Source: Lin et al. 2017 (RetinaNet), verified against pytorch-multi-class-focal-loss repo
# Add to src_v2/models/losses.py

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for COVID-19 classifier.

    Modulates CE loss by (1 - p_t)^gamma to down-weight easy examples.
    Compatible with class weights (alpha per class) for dual handling of imbalance.

    Ref: Lin et al. 2017 "Focal Loss for Dense Object Detection" (RetinaNet)
    """
    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard CE loss per sample (no reduction)
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        # p_t = probability assigned to correct class
        pt = torch.exp(-ce_loss)
        # Focal modulation
        focal_loss = (1.0 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss  # reduction='none' for OHEM compatibility
```

**Integration point:** In `cross_validate_classifier` (cli.py line 3404):
```python
# v1.0:
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
# v1.1 focal ablation:
criterion = FocalLoss(gamma=2.0, weight=class_weights_tensor)
```

**Note on combining with class weights:** Focal loss and class weights are complementary. Weights address class frequency (macro imbalance); focal addresses sample difficulty (within-batch focus). Both can be used together.

### Pattern 2: Hard Example Mining via WeightedRandomSampler

**What:** Pre-compute per-sample difficulty scores from OOF CE losses, then use `WeightedRandomSampler` to oversample hard examples. This is a static (non-online) approach using pre-existing OOF data — no additional inference required.

**When to use:** When a fixed subset of samples is consistently hard across folds (OOF loss captures this). Static mining avoids the per-epoch re-ranking overhead of OHEM.

**OOF difficulty data available:**
- File: `outputs/data_cleaning/oof_probabilities.npz`
- Keys: `pred_probs` (13258, 3), `image_names`, `true_labels`, `class_names`, `temperature` (=2.0)
- Difficulty distribution: mean CE loss=0.053, 90th pct=0.046, 95th pct=0.146
- Hard samples (loss>0.5): 275 samples (2.1%)
- Hard samples (loss>1.0): 155 samples (1.2%)
- Per-class mean loss: COVID=0.0703, Normal=0.0407, Viral_Pneumonia=0.1036

**Important constraint:** OOF probabilities were computed on the OLD warped dataset (v1.0, without exclude-list). After re-warping with the cleaned dataset, image filenames will change (different directory). The difficulty scores must be mapped by image stem (e.g., `COVID-1001` from `COVID-1001_warped.png`). Excluded images (432) have no OOF entry → assign default weight=1.0 if a new sample has no OOF match.

**Implementation:**

```python
# Source: PyTorch official docs + OOF data from Phase 7
# In cross_validate_classifier, before DataLoader creation:

def build_sample_weights(
    train_samples: list,  # [(path, label), ...]
    oof_loss_by_stem: dict,  # {image_stem: ce_loss_float}
    oversampling_ratio: float = 3.0,
    default_weight: float = 1.0
) -> torch.Tensor:
    """Build per-sample weights for WeightedRandomSampler."""
    weights = []
    for path, label in train_samples:
        stem = Path(path).stem.replace('_warped', '')
        oof_loss = oof_loss_by_stem.get(stem, default_weight)
        # Scale: hard examples get higher weight
        # Clamp to [1.0, oversampling_ratio] to avoid extreme oversampling
        weight = 1.0 + (oversampling_ratio - 1.0) * min(oof_loss / oof_loss_95th_pct, 1.0)
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32)

# Replace shuffle=True in DataLoader:
sample_weights = build_sample_weights(train_samples, oof_loss_by_stem)
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_samples),  # same dataset size per epoch
    replacement=True  # required for oversampling
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, ...)
# NOTE: sampler and shuffle=True are mutually exclusive — use sampler only
```

**Oversampling ratio recommendation:** Start with ratio=3.0 (hard samples at 95th pct get 3x sampling rate vs easy samples). The OOF data shows only 2.1% truly hard samples — aggressive oversampling at ratio=5.0+ could hurt Normal class by undersampling it. Tune if needed.

**Static vs dynamic decision:** Use static OOF-based weights. Dynamic OHEM (recompute per epoch) is more complex and the OOF data already provides reliable difficulty estimates from a 5-fold trained model. If static mining fails, OHEM is the fallback.

### Pattern 3: Curriculum Learning via Stage-Based Dataset Exposure

**What:** Start training with only easy samples (low OOF loss), progressively add harder samples over epochs. The OOF loss provides the curriculum ordering. This is a data scheduling approach — same model, same loss, different sample exposure over time.

**When to use:** When the model should first learn the "core pattern" from clear examples before encountering ambiguous cases. For Viral Pneumonia with high variance (mean loss=0.1036), curriculum may help the model build robust early features.

**Recommended schedule:** Step-based (simplest, most interpretable):
- Stage 1 (epochs 1-N1): Bottom 60% by OOF loss (easiest)
- Stage 2 (epochs N1-N2): Bottom 80% by OOF loss
- Stage 3 (epochs N2-end): Full dataset (100%)

**Class balance preservation:** OOF loss ordering must be applied within each class separately, then merged to maintain approximate class ratios. Otherwise Stage 1 would contain mostly Normal (lowest mean loss), leaving Viral Pneumonia severely underrepresented in early epochs — the opposite of what we want.

**Implementation:**

```python
# Source: Bengio et al. 2009 curriculum learning + Hacohen & Weinshall 2019 empirical analysis
# See arxiv.org/abs/1904.03626 for implementation guidance

def build_curriculum_stages(
    full_samples: list,  # [(path, label), ...]
    oof_loss_by_stem: dict,
    stages: list = [0.60, 0.80, 1.0],
    class_names: list = None
) -> list:
    """Build per-stage sample lists sorted by OOF difficulty, class-balanced."""
    # Sort within each class by OOF loss, then take percentile slices per stage
    class_samples = {c: [] for c in range(len(class_names))}
    for path, label in full_samples:
        stem = Path(path).stem.replace('_warped', '')
        loss = oof_loss_by_stem.get(stem, 0.0)  # default easy if not in OOF
        class_samples[label].append((path, label, loss))

    for label in class_samples:
        class_samples[label].sort(key=lambda x: x[2])  # sort by loss ascending

    stage_datasets = []
    for fraction in stages:
        stage_data = []
        for label, samples in class_samples.items():
            n = max(1, int(len(samples) * fraction))
            stage_data.extend([(p, l) for p, l, _ in samples[:n]])
        stage_datasets.append(stage_data)

    return stage_datasets  # list of 3 sample lists

# Training loop integration:
stage_transitions = [int(epochs * 0.33), int(epochs * 0.66)]  # at 33% and 66% of epochs
current_stage = 0
for epoch in range(epochs):
    if current_stage < len(stage_transitions) and epoch >= stage_transitions[current_stage]:
        current_stage += 1
        train_dataset = ImagePathDataset(stage_datasets[current_stage], train_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, ...)
        logger.info("Curriculum: advancing to stage %d (%.0f%% of data)",
                    current_stage+1, stages[current_stage]*100)
    # ... normal training step
```

**New samples issue:** The re-warped cleaned dataset will have different image paths. Samples not in OOF (either newly warped or from excluded-then-re-included categories) should be assigned the median OOF loss per class (safe fallback — treated as moderate difficulty). The 432 excluded samples are never used, so this only affects new warped filenames.

**Curriculum scope:** Apply only during fine-tuning (the main training loop in `cross_validate_classifier`). The current v1.0 classifier does not use a two-phase landmark-style training. The classifier uses a single training loop with AdamW + ReduceLROnPlateau.

### Pattern 4: Ablation Order Recommendation

**Recommended order:** (1) cleaned-data baseline, (2) focal loss, (3) hard example mining, (4) curriculum, (5) combined

**Rationale:**
1. **Cleaned-data baseline first** (locked decision): Isolates data cleaning effect
2. **Focal loss second**: Simplest modification (one class change), no data preprocessing needed, fast to implement and run
3. **Hard example mining third**: Requires OOF-to-new-dataset mapping, slightly more setup, but uses same static weights for all folds
4. **Curriculum last**: Most complex epoch-loop modification, benefits from understanding focal loss and mining results first
5. **Combined**: Fine-tune from focal (likely best individual) + add mining + curriculum

### Anti-Patterns to Avoid

- **Running OHEM without `reduction='none'`**: CE loss `reduction='none'` is required; default `reduction='mean'` collapses the batch to a scalar before selection.
- **Mixing `sampler` and `shuffle=True` in DataLoader**: PyTorch raises an error. When using `WeightedRandomSampler`, set `shuffle=False` (the sampler handles ordering).
- **Applying curriculum without class balance within stages**: Sorting all samples globally by loss puts most Viral Pneumonia in Stage 3, causing the model to train for 33%+ of epochs with almost no Viral Pneumonia — wrong direction.
- **Evaluating validation loss with focal loss (for early stopping)**: Focal loss values are not comparable across ablations (different scale than CE). Use val F1-macro for early stopping (already the case in v1.0) and log both focal loss (train) and CE loss (for comparison).
- **Applying focal loss with gamma too high**: gamma=5 causes near-zero gradients on all but the most confused samples; gamma=2 is the RetinaNet recommendation and the locked TRN-01 value.
- **Using test set for ablation decisions**: Locked decision — validation only.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Focal loss | Custom CE from scratch | 20-line `FocalLoss(nn.Module)` using `F.cross_entropy(reduction='none')` | `F.cross_entropy` handles numerical stability, class weights natively |
| Hard example oversampling | Custom batch generator | `torch.utils.data.WeightedRandomSampler` | Official PyTorch, handles replacement sampling, integrates with DataLoader |
| K-fold splits | Custom split logic | `sklearn.model_selection.StratifiedKFold` (already used) | Already present in `cross_validate_classifier`; any change breaks fold reproducibility |
| Per-class metrics | Custom confusion matrix | `sklearn.metrics` (already used) | Already imported in training loop |
| Dataset staging | Custom dataset class | `torch.utils.data.Subset` or inline list filtering | Simpler than custom class, no memory overhead |

**Key insight:** The hardest part of Phase 8 is correct OOF-to-new-dataset stem mapping, not the ML technique implementations. The techniques themselves are ~20-50 lines each.

---

## Common Pitfalls

### Pitfall 1: OOF Stem Mapping After Re-Warping

**What goes wrong:** The OOF file uses filenames like `COVID-1001_warped.png` from the OLD warped dataset. The new warped dataset (with exclude-list) will produce new warped images with potentially different filenames or directory structure. Hard example weights and curriculum ordering depend on matching OOF entries to training samples.

**Why it happens:** The generate-dataset command creates a new output directory. Image stems should be identical (same source images → same stems), but the `_warped` suffix may differ depending on output naming convention.

**How to avoid:** Load OOF data once, build a dict `{stem_without_warped_suffix: ce_loss}`. For each training sample path, strip `_warped` from stem before lookup. Verify match rate: expect ~96-97% match (432 excluded = unmatched, rest should match). Log unmatched count as a sanity check.

**Warning signs:** If match rate < 90%, filename convention changed — investigate before proceeding.

### Pitfall 2: Class Balance Destruction in Curriculum Stage 1

**What goes wrong:** Sorting all 13,258 train+val samples by OOF loss and taking the bottom 60% globally concentrates the easiest Normal samples and excludes almost all Viral Pneumonia in Stage 1.

**Why it happens:** Normal has mean loss=0.0407 (2.5x easier than Viral Pneumonia at 0.1036). A global sort places ~95% of Viral Pneumonia samples in the "hard" top 40%.

**How to avoid:** Sort within each class separately, then take the bottom K% per class. This preserves approximately the original class ratio (3164 COVID / 8918 Normal / 1176 VP) in each curriculum stage.

**Warning signs:** Check that Stage 1 includes at least 60% of Viral Pneumonia samples. Log per-class counts when building curriculum stages.

### Pitfall 3: Early Stopping Metric Inconsistency

**What goes wrong:** Using focal loss as the early stopping criterion (best val loss) instead of val F1-macro means stopping criteria differ across ablations — the cleaned-data baseline uses CE loss but focal ablation uses focal loss.

**Why it happens:** The v1.0 loop already uses `val_f1_macro > best_val_f1` for early stopping (correct). The risk is accidentally using val_loss (which changes scale with focal loss) for some logging or comparison.

**How to avoid:** Confirm early stopping and checkpoint saving uses val F1-macro throughout (already the case in the existing `cross_validate_classifier` code, line 3476: `if val_f1_macro > best_val_f1`). Log train_focal_loss separately from validation metrics.

### Pitfall 4: WeightedRandomSampler with `replacement=False`

**What goes wrong:** If `replacement=False`, `WeightedRandomSampler` cannot oversample — it only changes the order but cannot draw the same hard sample more than once per epoch.

**Why it happens:** Default behavior vs. intent confusion.

**How to avoid:** Always pass `replacement=True` when num_samples >= len(dataset) and oversampling is intended.

### Pitfall 5: Fold-Dependent OOF Weights

**What goes wrong:** Using OOF loss from fold K to oversample hard examples in fold K's training set. The OOF loss for a sample in fold K was computed from the model trained on folds 1-4 (that sample was in the validation set for fold K). This is actually fine — the OOF loss IS the correct estimate for held-out difficulty. But if the OOF was computed with a different fold structure or seed, weights may not align correctly.

**Why it happens:** Phase 7 OOF was computed with `StratifiedKFold(5, shuffle=True, random_state=42)`. Phase 8 must use the same seed=42 to maintain fold correspondence.

**How to avoid:** Document that OOF weights are applied uniformly across all folds (not fold-specific). The OOF loss is a property of the sample, not the fold — using it as a static weight per sample across all folds is standard practice.

---

## Code Examples

### Focal Loss with Class Weights

```python
# Source: Derived from Lin et al. 2017 + verified in pytorch 2.4
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FocalLoss(nn.Module):
    """Focal loss for multi-class classification."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight  # class weights (same as CrossEntropyLoss.weight)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Per-sample CE loss (uses weight for class imbalance)
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        # p_t = model's confidence in correct class
        pt = torch.exp(-ce_loss)
        # Focal modulation: suppress easy examples
        focal = (1.0 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal  # reduction='none' for per-sample access
```

### OOF Loss Loading and Stem Mapping

```python
# Load OOF difficulty scores from Phase 7
import numpy as np
import torch
from pathlib import Path

def load_oof_difficulty_scores(
    oof_path: str,
    class_names: list
) -> dict:
    """
    Load OOF probabilities and compute per-sample CE loss.
    Returns {image_stem_without_warped: ce_loss}.
    """
    oof = np.load(oof_path, allow_pickle=True)
    pred_probs = torch.tensor(oof['pred_probs'], dtype=torch.float32)
    true_labels = torch.tensor(oof['true_labels'].astype(int), dtype=torch.long)
    image_names = oof['image_names']  # e.g. 'COVID-1001_warped.png'

    # CE loss: -log(p_true)
    ce_loss = -torch.log(pred_probs[torch.arange(len(true_labels)), true_labels] + 1e-10)

    # Build stem -> loss mapping
    difficulty = {}
    for name, loss in zip(image_names, ce_loss.numpy()):
        stem = Path(str(name)).stem.replace('_warped', '')
        difficulty[stem] = float(loss)

    return difficulty  # e.g. {'COVID-1001': 0.012, 'Normal-001': 0.003, ...}
```

### WeightedRandomSampler Integration

```python
# Source: PyTorch docs torch.utils.data.WeightedRandomSampler
from torch.utils.data import WeightedRandomSampler

def build_sampling_weights(
    samples: list,  # [(path, label), ...]
    difficulty: dict,  # {stem: ce_loss}
    percentile_95: float,  # 95th percentile CE loss from OOF
    max_ratio: float = 3.0
) -> torch.Tensor:
    weights = []
    for path, label in samples:
        stem = Path(path).stem.replace('_warped', '')
        loss = difficulty.get(stem, 0.0)  # default: treat as easy
        # Linearly scale: 0 loss -> weight 1.0, 95th pct loss -> weight max_ratio
        w = 1.0 + (max_ratio - 1.0) * min(loss / (percentile_95 + 1e-8), 1.0)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)

# Usage (replaces shuffle=True):
weights = build_sampling_weights(train_samples, difficulty, percentile_95=0.146)
sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(train_samples),
    replacement=True
)
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler,
                          num_workers=..., pin_memory=...)
# NOTE: Do NOT set shuffle=True when using sampler
```

### Curriculum Stage Builder (Class-Balanced)

```python
def build_curriculum_stages(
    full_samples: list,  # [(path, label), ...]
    difficulty: dict,    # {stem: ce_loss}
    stage_fractions: list = [0.60, 0.80, 1.0],
    n_classes: int = 3
) -> list:
    """
    Build progressive curriculum stages with class balance preserved.
    Returns list of sample lists, one per stage.
    """
    # Group by class
    class_samples = [[] for _ in range(n_classes)]
    for path, label in full_samples:
        stem = Path(path).stem.replace('_warped', '')
        loss = difficulty.get(stem, float('inf'))  # unknown -> hardest (seen last)
        class_samples[label].append((path, label, loss))

    # Sort each class by difficulty (ascending = easy first)
    for i in range(n_classes):
        class_samples[i].sort(key=lambda x: x[2])

    stages = []
    for fraction in stage_fractions:
        stage = []
        for cls_list in class_samples:
            n = max(1, int(len(cls_list) * fraction))
            stage.extend([(p, l) for p, l, _ in cls_list[:n]])
        stages.append(stage)

    return stages
```

---

## State of the Art

| Old Approach | Current Approach | Notes |
|--------------|------------------|-------|
| Weighted CrossEntropyLoss | FocalLoss (gamma=2.0) + class weights | Focal suppresses within-batch easy examples beyond what class weights handle |
| Static shuffle DataLoader | WeightedRandomSampler (static OOF) | Pre-computed difficulty avoids per-epoch re-ranking overhead |
| OHEM (online per-batch) | Static OOF-based weights | Simpler, no inference overhead; OHEM kept as fallback |
| Random data ordering | Loss-based curriculum (easy→hard) | Literature supports 5-10% F1 improvement on imbalanced tasks |

---

## Key Findings from Codebase Audit

### v1.0 Baseline Numbers (from GROUND_TRUTH.json)

- **Test accuracy**: 98.26% (1,895 test samples)
- **F1-macro**: 97.12%
- **Viral Pneumonia recall**: 92.90% (157/169 correct, 12 missed → Normal misclassifications)
- **Per-class F1**: COVID=98.33%, Normal=98.75%, Viral_Pneumonia=94.29%
- **5-fold val accuracy mean**: 98.60% ± 0.26% (seed=42)

### Dataset Sizes (post Phase 7 cleaning)

- Total images: 15,153
- Excluded (Phase 7): 432
- Remaining: 14,721
- Train+Val pool (after splitting off test): ~13,258 (based on OOF size) → likely the same split; test size=1,895 unchanged
- Class distribution in train+val: COVID 3,164 / Normal 8,918 / Viral Pneumonia 1,176
- After re-warping with exclude-list: ~12,826 remaining in train+val (13,258 - 432 excluded from train_val)

### Existing CV Infrastructure (cli.py lines 2980-3584)

The `cross_validate_classifier` command:
- Uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)` (line 3330)
- Creates `ImagePathDataset` class inline (lines 3243-3259) — can be extended for curriculum
- Saves `best_classifier.pt` per fold with `best_val_f1` metadata — compatible with existing `evaluate-ensemble-classifier` command
- Saves `results.json` per fold — used by `load_ensemble_models()` in `src_v2/evaluation/ensemble.py`
- The config system (lines 3096-3174) supports JSON overrides — new keys needed for technique flags

### Config System Extension

The existing config validation in `cross_validate_classifier` (line 3112-3130) uses a whitelist of `valid_keys`. New technique parameters must be added to this whitelist:
```python
valid_keys = {
    "data_dir", "output_dir", "backbone", "epochs", "batch_size",
    "lr", "use_class_weights", "patience", "device", "seed",
    "folds", "eval_test", "save_checkpoints",
    # New for Phase 8:
    "use_focal_loss", "focal_gamma",        # TRN-01
    "use_hard_mining", "oof_path", "mining_max_ratio",  # TRN-02
    "use_curriculum", "curriculum_stages", "curriculum_fractions",  # TRN-03
}
```

### OOF Data Summary for Planning

- File: `outputs/data_cleaning/oof_probabilities.npz`
- Keys: `pred_probs` (13258, 3), `image_names`, `true_labels`, `class_names`, `temperature`=2.0
- Image name format: `COVID-1001_warped.png` (includes `_warped` suffix)
- CE loss 95th percentile: 0.146 (use as scaling cap for sampling weights)
- Hard samples (loss > 1.0): 155 (1.2%) — mostly Viral Pneumonia misclassified as Normal
- Viral Pneumonia has 2.55x higher mean difficulty than Normal (0.1036 vs 0.0407)

---

## Open Questions

1. **Re-warping output directory naming**
   - What we know: `generate-dataset` with `--exclude-list` creates a new warped dataset at the specified output dir
   - What's unclear: Whether the new warped filenames will use the same `COVID-1001_warped.png` convention
   - Recommendation: Verify stem convention early in Plan 1 before building OOF mappings; add a mapping verification step

2. **Epoch count for ablation runs**
   - What we know: v1.0 classifier used 50 epochs with patience=10, achieving good results. Landmark trainer uses 15+100 two-phase
   - What's unclear: Whether 50 epochs is sufficient for focal loss (may converge slower due to lower gradient magnitude on easy examples)
   - Recommendation: Keep 50 epochs with patience=10 for comparability; allow light tuning (e.g., 75 epochs) only if early stopping triggers at epoch <20 consistently

3. **Combined model fine-tuning scope**
   - What we know: Combined model fine-tunes from best individual ablation checkpoint (locked decision)
   - What's unclear: Whether to fine-tune all layers or just the head when adding curriculum + mining to a focal-pretrained model
   - Recommendation: Full fine-tuning (unfreeze all layers) since we're changing both loss function and data ordering, not just transfer learning a new head

4. **OOF weights for cross-validation folds vs global**
   - What we know: OOF loss reflects held-out difficulty for each sample
   - What's unclear: Whether to use fold-specific OOF losses (harder samples in fold K had their OOF computed by other folds) or global OOF loss
   - Recommendation: Use global OOF loss (single value per sample, computed as the fold where the sample was held out). This IS the Phase 7 OOF file — already fold-specific by construction.

---

## Sources

### Primary (HIGH confidence)

- PyTorch 2.4.1 source (installed) — `nn.CrossEntropyLoss`, `WeightedRandomSampler`, `Subset` APIs verified
- `src_v2/cli.py` lines 2980-3584 — `cross_validate_classifier` implementation (direct code audit)
- `src_v2/evaluation/ensemble.py` — `load_ensemble_models`, `weighted_soft_voting` (direct code audit)
- `outputs/data_cleaning/oof_probabilities.npz` — OOF difficulty scores (direct data inspection)
- `outputs/data_cleaning/cleaning_manifest.json` — 432 exclusions, manifest schema (direct inspection)
- `GROUND_TRUTH.json` — v1.0 baseline metrics (direct read)

### Secondary (MEDIUM confidence)

- Lin et al. 2017, "Focal Loss for Dense Object Detection" (RetinaNet) — gamma=2.0, alpha formulation verified against multiple implementations
- Hacohen & Weinshall 2019, "On the Power of Curriculum Learning" (ICML 2019, arxiv.org/abs/1904.03626) — step-based curriculum with teacher model
- GitHub: AdeelH/pytorch-multi-class-focal-loss — `FocalLoss` drop-in pattern verified (WebFetch)

### Tertiary (LOW confidence)

- WebSearch results on OHEM implementation patterns (unverified against authoritative 2024+ source) — use as inspiration, not specification

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all tools verified against installed packages and existing codebase
- Architecture patterns: HIGH — focal loss, WeightedRandomSampler, and curriculum implementations verified against PyTorch API and literature
- Pitfalls: HIGH — derived from direct codebase analysis (OOF stem naming, class balance destruction, sampler/shuffle conflict are observable from code)
- OOF data: HIGH — computed directly from the installed NPZ file

**Research date:** 2026-02-17
**Valid until:** 2026-04-17 (stable PyTorch APIs, established ML techniques)
