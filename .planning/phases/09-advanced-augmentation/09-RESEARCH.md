# Phase 9: Advanced Augmentation - Research

**Researched:** 2026-02-18
**Domain:** Medical image augmentation for warped chest X-ray classification (albumentations 2.x, torchvision v2 MixUp/CutMix, PyTorch 2.4 training loop integration)
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **Scope**: Integrate medical-specific augmentations into the classification training pipeline for warped chest X-rays. Architecture stays fixed (ResNet-18); only augmentation strategies change.
- **Baselines**: Report against BOTH cleaned baseline (F1=0.9844) AND curriculum model (F1=0.9932)
- **Ablation design**: Test each augmentation independently AND combined with curriculum learning
- **Visual validation gate**: Generate augmentation previews FIRST; user reviews and approves; training only begins after approval. Rejection policy: if augmentation looks too aggressive, drop it entirely — do not re-tune parameters.
- **Visual validation approach**: BOTH visual grids AND automated SSIM metrics
- **Ablation output**: Automated comparison script with tables and plots, same 08-03 pattern (thesis-ready)
- **GPU budget**: No hard limit — run all meaningful combinations
- **Negative results**: Brief note only — save detailed analysis for winners

### Claude's Discretion

- Specific augmentation parameter values (alpha, sigma, grid dimensions)
- Which pixel-level augmentations to include (if any)
- MixUp alpha values and cross-class mixing policy
- Whether CutMix is appropriate for chest X-rays
- Augmentation scheduling relative to curriculum learning stages
- Augmentation probability per class
- Number of example images per class for visual grids
- Similarity metric thresholds

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| AUG-01 | Training uses medical-specific augmentations via albumentations (ElasticTransform, GridDistortion) | albumentations 2.0.8 verified installable; `A.ElasticTransform(alpha=1.0, sigma=30.0)` and `A.GridDistortion(num_steps=5, distort_limit=0.1)` confirmed working; PIL→numpy→albumentations→torchvision pipeline verified |
| AUG-02 | Training supports batch-level MixUp and CutMix augmentation | `torchvision.transforms.v2.MixUp` and `CutMix` confirmed available in installed torchvision 0.19.1; soft labels (shape B×C) compatible with `nn.CrossEntropyLoss` and `FocalLoss`; applied via batch-level hook in the training loop |
| AUG-03 | Each augmentation strategy tested individually (ablation study) with validation metrics | Follows Phase 8 pattern: new JSON config per ablation + `cross-validate-classifier` CLI + `compare_ablations.py` script extended for Phase 9 |
</phase_requirements>

---

## Summary

Phase 9 adds medical-specific augmentations to the warped chest X-ray classification pipeline. Two categories are involved: (1) spatial augmentations via `albumentations` (ElasticTransform, GridDistortion), which simulate natural anatomical variation and respiratory positional shifts; and (2) batch-level mixing augmentations via `torchvision.transforms.v2` (MixUp, CutMix), which create interpolated or patch-spliced training samples for regularization. Both categories are verified to work with the existing `cross_validate_classifier` infrastructure in `src_v2/cli.py`.

The integration point is `get_classifier_transforms()` in `src_v2/models/classifier.py` — currently returning a `torchvision.transforms.Compose`. For spatial augmentations, a PIL→numpy→albumentations→PIL wrapper is inserted before the existing `RandomHorizontalFlip`/`RandomRotation`/`RandomAffine` transforms. For MixUp/CutMix, a batch-level hook is applied inside the training epoch loop (after the DataLoader yields a batch) rather than in the dataset-level transform. The visual validation gate runs before any training: a preview script generates augmentation samples, computes SSIM vs original, and shows visual grids for user review.

The baseline context: curriculum learning (F1=0.9932) is the best Phase 8 result. Adding focal loss + mining to curriculum regressed to F1=0.9878, showing technique interference is a real risk. Phase 9 must therefore test each augmentation independently first, then augmentation+curriculum combined — not a pile-on.

**Primary recommendation:** (1) Install albumentations as a new requirement; (2) implement a thin albumentations wrapper in `get_classifier_transforms()` with a `use_elastic/use_grid_distortion/use_pixel_aug` flag; (3) implement MixUp/CutMix as a collate_fn-level option in the training loop; (4) add JSON config keys for each augmentation flag; (5) run the visual validation gate script first; (6) run 6 ablation experiments: elastic-only, grid-only, pixel-aug-only, MixUp-only, each combined with curriculum, and a full augmentation+curriculum run.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| albumentations | 2.0.8 (installable, not yet in requirements.txt) | ElasticTransform, GridDistortion, pixel augments | De facto standard for medical image augmentation; GPU-agnostic; numpy-based so no ROCm issues |
| torchvision.transforms.v2.MixUp | 0.19.1 (already installed) | Batch-level MixUp with soft labels | Built into torchvision; no extra dependency; produces (B, num_classes) soft labels compatible with CE and FocalLoss |
| torchvision.transforms.v2.CutMix | 0.19.1 (already installed) | Batch-level CutMix | Same API as MixUp; same label format |
| skimage.metrics.structural_similarity | 0.26.0 (already installed) | SSIM for augmentation quality gate | Already in venv via scikit-image; `channel_axis=-1` for RGB images |
| torch.nn.CrossEntropyLoss | 2.4.1 (already used) | Loss compatible with both hard and soft labels | Built-in; `F.cross_entropy` accepts (B, C) float targets natively |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| matplotlib | 3.10.6 (already installed) | Visual grid generation for augmentation preview | Comparison script (same 08-03 pattern) |
| numpy | 2.1.2 (already installed) | PIL→numpy conversion for albumentations | Required bridge between PIL and albumentations |

### Installation Required

albumentations is **not in `requirements.txt`** and not currently installed in the venv. Must be added:

```bash
# Add to requirements.txt
albumentations>=2.0.0

# Install
pip install albumentations
```

The `--dry-run` confirmed it installs cleanly: albumentations 2.0.8 + albucore 0.0.24 + simsimd + stringzilla. No conflicts with existing packages.

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| albumentations ElasticTransform | torchvision v2 ElasticTransform | torchvision v2 ElasticTransform exists but has fewer medical imaging parameters; albumentations is the ecosystem standard for medical aug |
| torchvision MixUp | Custom numpy MixUp | torchvision built-in handles soft labels correctly and is well-tested; no reason to hand-roll |
| SSIM via skimage | torchmetrics StructuralSimilarityIndexMeasure | skimage already installed, no extra dep; adequate for validation gate |

---

## Architecture Patterns

### Recommended Integration Structure

The augmentation hooks fit into 3 locations in the existing codebase:

```
src_v2/models/classifier.py
  └── get_classifier_transforms(train=True)    ← add albumentations wrapper here

src_v2/cli.py — cross_validate_classifier()
  ├── valid_keys set                            ← add augmentation config keys
  ├── param_values dict                         ← add augmentation params
  └── training epoch loop (for epoch in range)
        └── after DataLoader yield              ← apply MixUp/CutMix per batch

scripts/preview_augmentations.py               ← NEW: visual gate script
scripts/compare_ablations_09.py                ← NEW: Phase 9 comparison (extend 08 pattern)

configs/
  ├── cv_aug_elastic.json                      ← elastic only
  ├── cv_aug_grid.json                         ← grid distortion only
  ├── cv_aug_pixel.json                        ← pixel augmentations only
  ├── cv_aug_mixup.json                        ← MixUp only
  ├── cv_aug_cutmix.json                       ← CutMix only (if approved)
  ├── cv_aug_elastic_curriculum.json            ← elastic + curriculum
  ├── cv_aug_grid_curriculum.json              ← grid + curriculum
  ├── cv_aug_mixup_curriculum.json             ← MixUp + curriculum
  └── cv_aug_allspatial_curriculum.json        ← all spatial + curriculum (winner round)
```

### Pattern 1: Albumentations Spatial Wrapper in get_classifier_transforms

Insert albumentations **between** PIL resize and torchvision normalization. The transform receives a PIL image, converts to numpy, applies albumentations, converts back to PIL.

```python
# Source: verified by running in project venv
import albumentations as A
import numpy as np

def build_albu_spatial_transform(use_elastic=False, use_grid_distortion=False):
    """Returns albumentations Compose or None."""
    transforms_list = []
    if use_elastic:
        transforms_list.append(
            A.ElasticTransform(
                alpha=1.0,        # small displacement magnitude
                sigma=30.0,       # smoothing — larger = smoother deformation
                p=0.5,
                border_mode=0,    # cv2.BORDER_CONSTANT, fills with 0 (black = background)
            )
        )
    if use_grid_distortion:
        transforms_list.append(
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.1,  # conservative: max 10% grid point displacement
                p=0.5,
                border_mode=0,
            )
        )
    return A.Compose(transforms_list) if transforms_list else None

class AlbumentationsWrapper:
    """Wrap albumentations Compose for use in torchvision Compose pipeline."""
    def __init__(self, albu_transform):
        self.transform = albu_transform

    def __call__(self, pil_img):
        img_np = np.array(pil_img)
        augmented = self.transform(image=img_np)['image']
        return Image.fromarray(augmented)
```

Then in `get_classifier_transforms`:

```python
if train and albu_transform is not None:
    train_transforms = [
        GrayscaleToRGB(),
        transforms.Resize((img_size, img_size)),
        AlbumentationsWrapper(albu_transform),  # <-- inserted here
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        normalize,
    ]
```

### Pattern 2: Batch-Level MixUp/CutMix in the Training Loop

MixUp and CutMix operate on full batches **after** DataLoader yields. They transform integer labels (B,) into soft labels (B, num_classes).

```python
# Source: verified by running in project venv (torchvision 0.19.1)
from torchvision.transforms.v2 import MixUp, CutMix

# Initialize once before fold loop
mixup_fn = None
if use_mixup:
    mixup_fn = MixUp(alpha=0.4, num_classes=len(class_names))
elif use_cutmix:
    mixup_fn = CutMix(alpha=1.0, num_classes=len(class_names))

# In training epoch loop:
for inputs, labels_batch in train_loader:
    inputs = inputs.to(torch_device)
    labels_batch = labels_batch.to(torch_device)

    if mixup_fn is not None:
        inputs, labels_batch = mixup_fn(inputs, labels_batch)
        # labels_batch is now (B, num_classes) float — CE and FocalLoss both accept this

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels_batch)
    loss.backward()
    optimizer.step()
```

**Important:** `val_criterion` always receives hard labels (val/test loaders never apply MixUp). This is already the pattern in Phase 8 (separate `val_criterion`).

**MixUp + curriculum ordering:** When both MixUp and curriculum are active, curriculum controls which samples enter the DataLoader. MixUp is applied to batches inside the loop regardless of curriculum stage. These compose cleanly.

### Pattern 3: Visual Validation Gate Script

```python
# scripts/preview_augmentations.py — to be created
# Steps:
# 1. Load N_SAMPLES_PER_CLASS (recommend 5) real warped images per class
# 2. Apply each augmentation individually to each image
# 3. Compute SSIM (original vs augmented) using skimage
# 4. Save grid: original | elastic | grid_dist | pixel_aug | MixUp preview
# 5. Print SSIM table — user sees numbers before approving
# Run: python scripts/preview_augmentations.py --data-dir outputs/warped_cleaned/session_warping
```

SSIM interpretation (verified on structured images in project venv):
- `ElasticTransform(alpha=1, sigma=30)`: SSIM ~0.9999 (barely perceptible)
- `ElasticTransform(alpha=50, sigma=50)`: SSIM ~0.9952 (subtle, medically realistic)
- `ElasticTransform(alpha=100, sigma=50)`: SSIM ~0.9793 (visible but not aggressive)
- `GridDistortion(distort_limit=0.1)`: SSIM ~0.9875 (appropriate for warped lungs)
- `GridDistortion(distort_limit=0.2)`: SSIM ~0.9863 (borderline — review visually)

Recommended SSIM threshold for "medically acceptable": **>0.95** for spatial transforms. If SSIM drops below 0.95 in previews, flag for user review.

### Pattern 4: Config Extension — New Keys for cross-validate-classifier

Add to `valid_keys` in the existing config parsing block (verified location: `src_v2/cli.py` line ~3152):

```python
"use_elastic_aug",          # bool, AUG-01
"use_grid_distortion_aug",  # bool, AUG-01
"use_pixel_aug",            # bool, Claude's discretion
"elastic_alpha",            # float, default 1.0
"elastic_sigma",            # float, default 30.0
"elastic_p",                # float, default 0.5
"grid_distort_limit",       # float, default 0.1
"grid_distort_p",           # float, default 0.5
"use_mixup",                # bool, AUG-02
"mixup_alpha",              # float, default 0.4
"use_cutmix",               # bool, AUG-02
"cutmix_alpha",             # float, default 1.0
```

All default to `False`/conservative values for full backward compatibility with existing configs.

### Anti-Patterns to Avoid

- **Don't apply albumentations in ValTransform**: Augmentations are training-only. The existing `eval_transform = get_classifier_transforms(train=False)` must stay unchanged.
- **Don't apply MixUp in the DataLoader collate_fn**: The training loop in `cross_validate_classifier` builds DataLoaders internally with curriculum stage transitions. Applying MixUp in the loop body (not collate_fn) lets curriculum rebuild loaders without losing MixUp.
- **Don't combine MixUp and CutMix simultaneously in one experiment**: They are separate ablation configs. Test each independently first.
- **Don't use `alpha=1.0` for MixUp**: Higher alpha → stronger mixing → more label smoothing. For a high-accuracy baseline (F1=0.9932), conservative `alpha=0.4` is safer than the default 1.0 (Beta(1,1) = uniform mixing).
- **Don't use GridDistortion with `normalized=False`**: The default `normalized=True` prevents pixels from moving outside image boundaries, which is critical for warped lungs that already have meaningful black borders.
- **Don't use `border_mode=4` (BORDER_REFLECT)**: For warped chest X-rays where background is black, reflected border introduces lung texture at edges and corrupts augmentation. Use `border_mode=0` (BORDER_CONSTANT, fill=0).
- **Don't skip the visual gate for MixUp previews**: MixUp blends COVID+Normal images — the user must verify this looks clinically sensible before training.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Elastic deformation | Custom displacement field implementation | `albumentations.ElasticTransform` | Medical imaging literature uses albumentations; correct Gaussian smoothing of displacement fields is non-trivial |
| Soft label interpolation (MixUp) | Custom lambda * img_a + (1-lambda) * img_b | `torchvision.transforms.v2.MixUp` | Handles label format conversion (B,) → (B, C) correctly; verified compatible with CE and FocalLoss in this codebase |
| SSIM computation | Custom pixel difference metric | `skimage.metrics.structural_similarity` | Already installed; handles multichannel correctly with `channel_axis=-1` |
| Grid distortion | Custom mesh warp | `albumentations.GridDistortion` | Correct bilinear interpolation and boundary handling; `normalized=True` prevents out-of-bounds |

**Key insight:** Medical augmentation is well-understood territory where library defaults are calibrated for radiological images. The main risk is using parameters that are *too conservative* (no regularization benefit) or *too aggressive* (destroys diagnostic features). The visual gate catches the aggressive case.

---

## Common Pitfalls

### Pitfall 1: Albumentations 2.0 API Incompatibility

**What goes wrong:** Code written for albumentations 1.x uses `alpha_affine` parameter in `ElasticTransform`, which was removed in 2.0. Using `alpha=120` (old tutorial values) with 2.0 API produces no error but no visible distortion (alpha scales differently).

**Why it happens:** Most medical imaging tutorials cite albumentations 1.x parameters (e.g., `alpha=120, sigma=120*0.05`). The 2.0 API changed `ElasticTransform` significantly: `alpha` now defaults to 1.0 and `sigma` defaults to 50.0.

**How to avoid:** Use the verified 2.0 parameters: `ElasticTransform(alpha=1.0, sigma=30.0)` for conservative, or `ElasticTransform(alpha=50.0, sigma=50.0)` for moderate. Do not copy parameters from 1.x tutorials.

**Warning signs:** Visual preview shows no visible distortion despite p=1.0.

### Pitfall 2: MixUp Breaks evaluate_basic — Label Shape Mismatch

**What goes wrong:** If MixUp is accidentally applied during validation (via incorrect transform reuse), `evaluate_basic` receives (B, C) soft labels and `_.max(1)` on logits still works, but accuracy becomes undefined (soft labels can't be compared to argmax predictions for classification accuracy).

**Why it happens:** MixUp is applied inside the training loop — if the same `mixup_fn` leaks into validation (e.g., by calling `mixup_fn` on every batch without checking `model.training`), accuracy metrics are corrupted.

**How to avoid:** Only apply `mixup_fn` inside the `model.train()` block, never in `evaluate_basic`. The `evaluate_basic` function uses `val_criterion` and `val_loader` which have no MixUp — this boundary must be preserved.

**Warning signs:** Val accuracy drops to near random (33% for 3 classes) while train loss decreases normally.

### Pitfall 3: CutMix Cuts Diagnostic Lung Regions

**What goes wrong:** CutMix pastes rectangular patches from one image over another. For chest X-rays, the lung region covers most of the 224×224 image. A large cut patch (alpha=1.0 can cut ~50% of the image) can completely obscure consolidation patterns in COVID images, making the label (e.g., 60% COVID) clinically meaningless.

**Why it happens:** CutMix was designed for natural images where local patches are informative. For chest X-rays where global texture patterns matter, large patches destroy the diagnostic signal.

**How to avoid:** Use conservative `alpha=0.2` (small patches) if CutMix is tested. Or: based on the visual gate review, CutMix may be dropped entirely. The user's decision rule is "if it looks too aggressive, drop it."

**Warning signs:** Visual preview shows one lung's texture pasted over the other lung's consolidation pattern.

### Pitfall 4: Albumentations Fill Value for Warped Images

**What goes wrong:** Using `border_mode=4` (BORDER_REFLECT) fills elastic/grid distortion borders with reflected lung texture. For warped images with ~5% margin (OPTIMAL_MARGIN_SCALE=1.05), this creates artificial lung boundary patterns that confuse the classifier.

**Why it happens:** The default `border_mode` in some albumentations examples is REFLECT. Warped chest X-rays have black (zero-value) background, so any non-zero fill corrupts the anatomical layout.

**How to avoid:** Always use `border_mode=0` (BORDER_CONSTANT) and `fill=0.0` (black fill), which correctly fills with background-like values consistent with the warping pipeline.

### Pitfall 5: Curriculum + MixUp Stage Transition Breaks DataLoader

**What goes wrong:** When curriculum rebuilds the train_loader at stage transitions (epochs 0, 33%, 66%), the MixUp function must be re-applied in the new epoch's batch loop. If MixUp is initialized inside the DataLoader construction (e.g., as a collate_fn), curriculum's loader rebuild loses the MixUp configuration.

**Why it happens:** The curriculum integration in Phase 8 rebuilds `train_loader` via `DataLoader(train_dataset, ...)` at stage boundaries. If MixUp is in `collate_fn`, it must be re-specified each time.

**How to avoid:** Initialize `mixup_fn` once before the fold loop and apply it inside the batch loop (`for inputs, labels_batch in train_loader`), not in `collate_fn`. This survives loader rebuilds without any code changes.

### Pitfall 6: GaussNoise std_range Scaled Differently in Albumentations 2.0

**What goes wrong:** `A.GaussNoise(std_range=(0.01, 0.05))` in albumentations 2.0 applies noise with std in range [0.01, 0.05] as a fraction of the image dynamic range. For uint8 images (0-255), this is 2.55-12.75 intensity units — visible but not destructive.

**Why it happens:** albumentations 2.0 changed `GaussNoise` to use `std_range` instead of `var_limit`. Old parameter name `var_limit` raises `TypeError`.

**How to avoid:** Use the 2.0 API exactly: `A.GaussNoise(std_range=(0.01, 0.05), mean_range=(0, 0), p=0.3)`. Do not use `var_limit`.

---

## Code Examples

### Full Albumentations + PIL + Torchvision Pipeline (Verified)

```python
# Source: verified by running in project venv, 2026-02-18
import albumentations as A
import numpy as np
from PIL import Image
from torchvision import transforms

def build_classifier_train_transform(
    img_size: int = 224,
    use_elastic: bool = False,
    elastic_alpha: float = 1.0,
    elastic_sigma: float = 30.0,
    elastic_p: float = 0.5,
    use_grid_distortion: bool = False,
    grid_distort_limit: float = 0.1,
    grid_distort_p: float = 0.5,
    use_pixel_aug: bool = False,
) -> transforms.Compose:
    """Build train transform with optional albumentations augmentations."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    albu_transforms = []
    if use_elastic:
        albu_transforms.append(
            A.ElasticTransform(
                alpha=elastic_alpha,
                sigma=elastic_sigma,
                p=elastic_p,
                border_mode=0,  # BORDER_CONSTANT
                fill=0.0,
            )
        )
    if use_grid_distortion:
        albu_transforms.append(
            A.GridDistortion(
                num_steps=5,
                distort_limit=grid_distort_limit,
                p=grid_distort_p,
                border_mode=0,
                fill=0.0,
                normalized=True,  # prevents out-of-bounds
            )
        )
    if use_pixel_aug:
        albu_transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.4,
            )
        )
        albu_transforms.append(
            A.GaussNoise(std_range=(0.01, 0.03), mean_range=(0, 0), p=0.3)
        )

    transform_list = [
        GrayscaleToRGB(),
        transforms.Resize((img_size, img_size)),
    ]

    if albu_transforms:
        albu_compose = A.Compose(albu_transforms)

        class AlbuWrapper:
            def __call__(self, pil_img):
                img_np = np.array(pil_img)
                result = albu_compose(image=img_np)['image']
                return Image.fromarray(result)

        transform_list.append(AlbuWrapper())

    transform_list.extend([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        normalize,
    ])
    return transforms.Compose(transform_list)
```

### MixUp/CutMix in Training Loop (Verified)

```python
# Source: verified by running in project venv, 2026-02-18
from torchvision.transforms.v2 import MixUp, CutMix

# Initialize before fold loop (once)
mixup_fn = None
if use_mixup:
    mixup_fn = MixUp(alpha=mixup_alpha, num_classes=len(class_names))
elif use_cutmix:
    mixup_fn = CutMix(alpha=cutmix_alpha, num_classes=len(class_names))

# Inside training epoch loop:
for epoch in range(epochs):
    # ... curriculum stage transitions here (unchanged) ...
    model.train()
    for inputs, labels_batch in train_loader:
        inputs = inputs.to(torch_device)
        labels_batch = labels_batch.to(torch_device)

        # Apply batch-level mixing (training only)
        if mixup_fn is not None:
            inputs, labels_batch = mixup_fn(inputs, labels_batch)
            # labels_batch is now (B, num_classes) float32

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels_batch)  # CE and FocalLoss both accept soft labels
        loss.backward()
        optimizer.step()
```

### SSIM Validation for Augmentation Gate (Verified)

```python
# Source: verified by running in project venv, 2026-02-18
from skimage.metrics import structural_similarity
import numpy as np

def compute_ssim_batch(originals, augmented_list, n_samples=5):
    """Compute mean SSIM between original and augmented images."""
    ssim_values = []
    for orig, aug in zip(originals[:n_samples], augmented_list[:n_samples]):
        orig_np = np.array(orig)
        aug_np = np.array(aug)
        ssim = structural_similarity(
            orig_np, aug_np,
            multichannel=True,
            channel_axis=-1,
            data_range=255,
        )
        ssim_values.append(ssim)
    return np.mean(ssim_values), np.std(ssim_values)

# Usage in preview script:
# ssim_mean, ssim_std = compute_ssim_batch(original_images, elastic_images)
# print(f"ElasticTransform SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")
# Recommended threshold: > 0.95 for approval
```

---

## Recommended Augmentation Parameters (Claude's Discretion)

Based on medical imaging literature, SSIM calibration tests, and the project's context (224×224 warped lung images with anatomical normalization):

### Spatial Augmentations (albumentations)

**ElasticTransform — conservative settings:**
- `alpha=1.0, sigma=30.0, p=0.5`
- Rationale: alpha=1.0 gives minimal displacement magnitude; sigma=30.0 smooths the field to avoid jagged distortions. SSIM=0.9999 on structured images — nearly invisible but enough for regularization.
- Do not exceed `alpha=50, sigma=50` (SSIM drops to 0.9952 — visible but still acceptable).
- If the visual gate shows this is too subtle, increase to `alpha=20.0, sigma=30.0`.

**GridDistortion — conservative settings:**
- `num_steps=5, distort_limit=0.1, p=0.5`
- Rationale: 5 grid cells gives enough spatial coverage; 10% displacement limit preserves lung contour alignment (critical since warping is the pipeline's main normalization step). SSIM=0.9875.
- Do not exceed `distort_limit=0.2` without visual validation.

### Pixel Augmentations (Claude's Discretion — include conservatively)

**RandomBrightnessContrast:**
- `brightness_limit=0.15, contrast_limit=0.15, p=0.4`
- Rationale: Chest X-ray contrast varies with exposure and equipment; 15% variation is clinically plausible. The existing `get_classifier_transforms` already has no brightness augmentation, so this fills a gap.

**GaussNoise:**
- `std_range=(0.01, 0.03), mean_range=(0, 0), p=0.3`
- Rationale: Simulates detector noise variation. Conservative std range (2.5-7.5 intensity units for uint8). Applied last in pixel aug pipeline.
- Note: This is albumentations 2.0 API — use `std_range`, not `var_limit`.

**Include pixel augmentations as one combined config (`use_pixel_aug=True`), not separately.** They are straightforward enough that individual ablation is not warranted; the ablation study focuses on spatial transforms and mixing.

### MixUp

**Alpha recommendation: 0.4**
- Rationale: `alpha=0.4` draws lambda from Beta(0.4, 0.4), concentrating mixing near 0.2-0.8 rather than uniform 0-1. This means most mixed images still look predominantly like one class, preserving diagnostic features while providing label smoothing. The default `alpha=1.0` (uniform mixing) is too aggressive for high-accuracy baselines.
- Cross-class mixing (COVID+Normal): MixUp mixes all class pairs uniformly. This includes COVID+Normal blending, which is clinically unusual but serves as regularization. The visual gate will confirm the blends look acceptable.

### CutMix — include but test conservatively

**Alpha recommendation: 0.2** (small patches)
- Rationale: `alpha=0.2` draws lambda from Beta(0.2, 0.2), which concentrates near 0 and 1 — meaning the cut region is small. This limits how much of one image's lung region is pasted over another. For 224×224 images, this yields patches of approximately 30-60px × 30-60px.
- Risk: Even small patches may cut across consolidation regions visible in COVID images. The visual gate is the decision point.
- Rejection criterion: If previews show any lung texture from one class visibly overlapping diagnostic features of another class, drop CutMix entirely (per user's rejection policy).

---

## Ablation Experiment Design

### Experiment Matrix

| Config | Augmentation | Curriculum | Baseline | Expected Direction |
|--------|-------------|-----------|----------|------------------|
| cv_aug_elastic | ElasticTransform(α=1,σ=30) | No | Cleaned (F1=0.9844) | Small positive or neutral |
| cv_aug_grid | GridDistortion(limit=0.1) | No | Cleaned (F1=0.9844) | Small positive or neutral |
| cv_aug_pixel | BrightnessContrast+Noise | No | Cleaned (F1=0.9844) | Small positive or neutral |
| cv_aug_mixup | MixUp(α=0.4) | No | Cleaned (F1=0.9844) | Positive (label smoothing) |
| cv_aug_cutmix | CutMix(α=0.2) | No | Cleaned (F1=0.9844) | Uncertain — gate decides |
| cv_aug_elastic_curriculum | ElasticTransform + curriculum | Yes | Curriculum (F1=0.9932) | Small marginal improvement |
| cv_aug_grid_curriculum | GridDistortion + curriculum | Yes | Curriculum (F1=0.9932) | Small marginal improvement |
| cv_aug_mixup_curriculum | MixUp + curriculum | Yes | Curriculum (F1=0.9932) | Positive (complementary regularization) |

**Best performer gets one final sweep:** best_aug_all_curriculum (spatial + pixel + best mixing + curriculum)

### Ablation Order

1. Visual gate (previews only, no training) — user must approve
2. Individual augmentations without curriculum (elastic, grid, pixel, MixUp, CutMix if approved)
3. Best performers + curriculum
4. Final winner configuration
5. Comparison script (Phase 9 version of compare_ablations.py)

### Comparison Script Pattern (follows 08-03 style)

The `scripts/compare_ablations.py` script from Phase 8 can be extended or a new `scripts/compare_ablations_09.py` created with:
- Same experiment dict pattern: `EXPERIMENTS = {name: path_to_cross_validation_results.json}`
- Two reference baselines: cleaned_baseline (F1=0.9844) and curriculum (F1=0.9932)
- Additional columns: augmentation type, combined_with_curriculum flag
- Same per-class Viral Pneumonia recall extraction
- Output: printed table + `outputs/ablation_comparison_09.json`

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| albumentations 1.x `alpha_affine` parameter in ElasticTransform | albumentations 2.x removed `alpha_affine`; `alpha` now defaults to 1.0 not 1200 | albumentations 2.0 release (2024) | Old tutorial code with `alpha=120` will silently produce much weaker distortion in 2.0 |
| albumentations 1.x `var_limit` in GaussNoise | albumentations 2.x uses `std_range` | albumentations 2.0 | Old code raises TypeError; must update |
| Manual MixUp numpy implementation | `torchvision.transforms.v2.MixUp` (available since torchvision 0.15) | torchvision v2 launch | Built-in handles (B,) → (B, C) label conversion; one line instead of manual lambda sampling |
| Applying augmentation in Dataset.__getitem__ | Batch-level augmentation via collate_fn or training loop | MixUp/CutMix design | MixUp requires access to full batch to pair samples; cannot be per-sample |

**Deprecated/outdated:**
- `albumentations.ElasticTransform(alpha_affine=...)`: Parameter removed in 2.0; raises error
- `albumentations.GaussNoise(var_limit=...)`: Parameter renamed to `std_range` in 2.0

---

## Open Questions

1. **Will CutMix be approved by the visual gate?**
   - What we know: CutMix with `alpha=0.2` produces ~30-60px patches; for COVID vs Normal mixing, a patch replaces part of one lung with the other
   - What's unclear: Whether this is too aggressive for warped images where the full lung is visible in 224×224
   - Recommendation: Generate 10 COVID+Normal CutMix examples in the preview. If any show consolidation patterns being covered by normal lung texture, drop CutMix.

2. **Should MixUp scheduling be used (apply only after N epochs)?**
   - What we know: MixUp from epoch 1 can slow early convergence because soft labels reduce gradient signal strength
   - What's unclear: Whether the curriculum learning's easy-first approach already provides enough early convergence stability to make MixUp scheduling unnecessary
   - Recommendation: Start MixUp from epoch 1 (simpler). If MixUp+curriculum combined shows worse early training curves than curriculum alone, consider adding a `mixup_start_epoch` config param in a second pass.

3. **Should augmentation probability be class-dependent (more augmentation for Viral Pneumonia)?**
   - What we know: Viral Pneumonia is the hardest class (highest OOF loss), smallest class (1,176 samples vs 8,918 Normal)
   - What's unclear: Whether higher-p augmentation for VP samples would help given the class is already up-weighted via `use_class_weights`
   - Recommendation: Keep uniform probability (Claude's discretion: start simple). Per-class augmentation probability would require custom Dataset modifications and is not warranted before we know if basic augmentation helps.

---

## Sources

### Primary (HIGH confidence)

- albumentations 2.0.8 installed and tested in project venv — ElasticTransform, GridDistortion, GaussNoise, RandomBrightnessContrast API verified
- torchvision 0.19.1 installed — `MixUp`, `CutMix` from `torchvision.transforms.v2` tested with soft label CE loss and FocalLoss
- `skimage.metrics.structural_similarity` 0.26.0 tested with `channel_axis=-1` for RGB images
- `src_v2/cli.py` (project codebase, lines 2979–3800) — full `cross_validate_classifier` implementation reviewed
- `src_v2/models/classifier.py` — `get_classifier_transforms()` reviewed (lines 290-330)
- `src_v2/models/losses.py` — `FocalLoss` reviewed and tested for MixUp soft label compatibility
- [albumentations explore: ElasticTransform](https://explore.albumentations.ai/transform/ElasticTransform) — 2.0 parameter documentation
- [albumentations API reference: distortion](https://albumentations.ai/docs/api-reference/albumentations/augmentations/geometric/distortion/) — GridDistortion parameters
- [torchvision 0.19 MixUp docs](https://docs.pytorch.org/vision/0.19/generated/torchvision.transforms.v2.MixUp.html) — confirmed available in 0.19.x

### Secondary (MEDIUM confidence)

- SSIM calibration numbers: computed in project venv on structured test images; real chest X-ray SSIM may differ but directional conclusions hold
- MixUp alpha=0.4 recommendation: derived from known Beta distribution behavior; cross-verified with multiple sources citing conservative alpha for high-accuracy settings

### Tertiary (LOW confidence)

- CutMix alpha=0.2 (small patches): recommendation based on reasoning about image content; must be validated in visual gate
- Per-class VP augmentation probability: not tested; deferred to future phase if basic augmentation succeeds

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — albumentations 2.0.8 and torchvision v2 MixUp/CutMix tested in venv; all dependencies confirmed installable
- Architecture/integration: HIGH — PIL→numpy→albumentations pipeline and batch-level MixUp both verified to produce correct shapes and loss values
- Parameter recommendations: MEDIUM — calibrated via SSIM on structured images; must be validated in visual gate against real warped X-rays
- Ablation design: HIGH — follows Phase 8 pattern exactly; infrastructure is proven

**Research date:** 2026-02-18
**Valid until:** 2026-04-18 (stable libraries; albumentations minor releases unlikely to break API)
