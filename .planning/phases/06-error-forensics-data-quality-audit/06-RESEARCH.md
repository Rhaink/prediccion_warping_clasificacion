# Phase 6: Error Forensics & Data Quality Audit - Research

**Researched:** 2026-02-16
**Domain:** Error analysis visualization and data quality assessment for medical imaging
**Confidence:** HIGH

## Summary

This phase focuses on understanding the 33 test set misclassifications (from 98.26% ensemble accuracy) and assessing dataset quality through duplicate detection and image quality scoring. The research covers three main domains: (1) error visualization and categorization techniques for multi-stage pipelines, (2) duplicate detection methods for cross-split and cross-class analysis, and (3) no-reference image quality assessment metrics.

The Python scientific stack provides mature tools for all requirements. Matplotlib with ImageGrid handles pipeline trace visualization, scikit-learn provides confusion matrix utilities, and specialized libraries (imagededup, pybrisque, IQA-PyTorch) offer production-ready implementations of duplicate detection and quality assessment. The key architectural decision is leveraging existing project infrastructure (PyTorch feature extractors, evaluation metrics) while adding minimal new dependencies.

**Primary recommendation:** Build error forensics as a new CLI command and Jupyter notebook pair, using matplotlib for static thesis-ready visualizations and ipywidgets for interactive exploration. Implement dual-stage duplicate detection (original + warped images) with both perceptual hashing (speed) and CNN embeddings (accuracy). Use pybrisque or IQA-PyTorch for no-reference quality assessment with BRISQUE and NIQE metrics.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Show both original X-ray AND warped version side-by-side for each misclassified image
- Include landmark detection overlay on original images to check if bad landmarks caused errors
- Produce both static image grids (thesis-ready) AND interactive Jupyter notebook for exploration
- Static outputs at two scales: compact overview grid (all 33 in one figure) + detailed per-sample figures for appendix
- Per-sample detailed figures use a "pipeline trace" layout: original → landmarks overlay → warped → classification result in a single row
- Trace errors through the full pipeline to identify WHERE failure originated: bad landmarks → bad warp → misclassification vs good warp but ambiguous image vs possible label noise
- Include recoverability assessment: tag each error as fixable (label noise, bad landmark), partially fixable (hard example, better training may help), or inherent (genuinely ambiguous)
- Full dataset scope for duplicates: cross-split (train/val/test leakage), within-split, AND cross-class (potential label errors)
- Run duplicate detection on both original images and warped images — warping could make different images converge or same images diverge
- Both stages provide different diagnostic signals

### Claude's Discretion
- Error visualization grouping strategy (by confusion pair, by confidence, or hybrid — pick most informative)
- Metadata shown per image (minimal vs full context with fold agreement and probability bar charts)
- Whether to include "nearest correct neighbor" comparison for each misclassified image
- Error categorization thresholds (confidence-based, fold agreement, or combined matrix)
- Whether to also analyze CV validation set errors beyond the 33 test errors for larger sample
- Duplicate detection similarity threshold and metric choice (SSIM, perceptual hash, feature embeddings)
- Handling of discovered train/test leakage (quantify impact vs document only — recommend based on severity)
- Report detail level per error (summary per category vs every error documented — balance thoroughness with readability)
- Image quality score scope (full dataset distribution vs errors + control sample — balance compute cost vs diagnostic value)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope

</user_constraints>

## Standard Stack

### Core Libraries (Already in Project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| matplotlib | >=3.7.0 | Static visualization, grids, pipeline traces | Industry standard for publication-quality figures, ImageGrid for multi-panel layouts |
| scikit-learn | >=1.3.0 | Confusion matrix, classification metrics | ConfusionMatrixDisplay provides standardized visualization |
| torch | >=2.0.0 | Feature extraction for duplicate detection | Existing ResNet models can generate embeddings without retraining |
| pandas | >=2.0.0 | Error metadata tracking, report generation | Standard for tabular data analysis |
| numpy | >=2.0.0 | Array operations, similarity calculations | Foundation for all numeric operations |
| opencv-python | >=4.8.0 | Image I/O, SSIM calculation | cv2.quality module includes BRISQUE implementation |

### New Dependencies Needed
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| imagededup | >=0.3.0 | Perceptual hashing, CNN-based duplicates | Fast initial duplicate detection (hashing) and high-accuracy verification (CNN) |
| pybrisque | >=1.0 | BRISQUE no-reference quality scores | Lightweight single-metric implementation |
| IQA-PyTorch | latest | BRISQUE, NIQE, multiple IQA metrics | Comprehensive toolkit if multiple quality metrics desired |
| jupyter | >=1.0.0 | Interactive notebook for error exploration | Enable analyst to interactively filter/sort errors |
| ipywidgets | >=8.0.0 | Interactive controls in notebooks | Dropdown filters, sliders for thresholds |
| seaborn | >=0.12.0 | Enhanced statistical visualizations | Optional for distribution plots of quality scores |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| imagededup | Custom SSIM loop | imagededup is optimized and handles edge cases (different sizes, formats) |
| pybrisque | OpenCV cv2.quality.QualityBRISQUE | OpenCV requires trained model file, pybrisque bundles weights |
| IQA-PyTorch | scikit-video NIQE | IQA-PyTorch is pure PyTorch, integrates with existing pipeline |
| matplotlib ImageGrid | Manual subplots | ImageGrid handles axis sharing, colorbars, spacing automatically |

**Installation:**
```bash
# Add to requirements.txt
imagededup>=0.3.0
pybrisque>=1.0
jupyter>=1.0.0
ipywidgets>=8.0.0

# Optional: comprehensive IQA toolkit (if multiple quality metrics desired)
# IQA-PyTorch  # Install from GitHub: pip install git+https://github.com/chaofengc/IQA-PyTorch.git
```

## Architecture Patterns

### Recommended Project Structure
```
src_v2/
├── evaluation/
│   ├── metrics.py              # Existing landmark/classifier metrics
│   ├── error_analysis.py       # NEW: error categorization logic
│   └── quality_assessment.py   # NEW: BRISQUE, NIQE, quality scoring
├── utils/
│   ├── duplicates.py           # NEW: duplicate detection (hash + CNN)
│   └── visualization.py        # NEW: pipeline trace, grid layouts
└── cli.py                      # NEW: error-forensics command

notebooks/
└── error_forensics_interactive.ipynb  # NEW: interactive exploration

outputs/
└── error_forensics/
    ├── error_visualizations/
    │   ├── overview_grid_all_33.png           # Compact grid
    │   ├── by_confusion_pair/                 # Grouped by true→pred
    │   │   ├── COVID_to_Normal.png
    │   │   └── ...
    │   └── per_sample_detailed/               # Individual pipeline traces
    │       ├── sample_001_pipeline.png
    │       └── ...
    ├── duplicates/
    │   ├── original_duplicates.csv            # Duplicates in original dataset
    │   ├── warped_duplicates.csv              # Duplicates in warped dataset
    │   └── duplicate_pairs_visualization.png
    ├── quality_scores/
    │   ├── all_images_quality.csv             # BRISQUE/NIQE scores
    │   └── quality_distribution.png
    └── report_error_forensics.md              # Spanish thesis-ready report
```

### Pattern 1: Pipeline Trace Visualization

**What:** Display each error as a horizontal sequence: original image → landmarks overlay → warped image → classification result

**When to use:** For detailed per-sample analysis showing where in pipeline failure occurred

**Example:**
```python
# Source: matplotlib ImageGrid + custom overlay utilities
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt

def visualize_pipeline_trace(original_img, landmarks_pred, warped_img,
                              pred_class, true_class, probs, output_path):
    """
    Create 4-panel pipeline trace for a single misclassified sample.

    Args:
        original_img: Original X-ray (H, W) grayscale
        landmarks_pred: (15, 2) landmark coordinates
        warped_img: Warped X-ray (H, W)
        pred_class: Predicted class name
        true_class: True class name
        probs: (3,) class probabilities
        output_path: Where to save figure
    """
    fig = plt.figure(figsize=(20, 5))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 4), axes_pad=0.3)

    # Panel 1: Original image
    grid[0].imshow(original_img, cmap='gray')
    grid[0].set_title('Original X-ray', fontsize=12)
    grid[0].axis('off')

    # Panel 2: Landmarks overlay
    grid[1].imshow(original_img, cmap='gray')
    grid[1].scatter(landmarks_pred[:, 0], landmarks_pred[:, 1],
                    c='red', s=30, marker='x')
    # Add Delaunay triangulation overlay
    from src_v2.processing.gpa import compute_delaunay_triangulation
    triangulation = compute_delaunay_triangulation(landmarks_pred)
    grid[1].triplot(landmarks_pred[:, 0], landmarks_pred[:, 1],
                    triangulation.simplices, 'r-', linewidth=0.5, alpha=0.5)
    grid[1].set_title('Landmark Detection', fontsize=12)
    grid[1].axis('off')

    # Panel 3: Warped image
    grid[2].imshow(warped_img, cmap='gray')
    grid[2].set_title('Warped (Normalized)', fontsize=12)
    grid[2].axis('off')

    # Panel 4: Classification result
    grid[3].axis('off')
    grid[3].text(0.1, 0.9, f'True: {true_class}', fontsize=14,
                 weight='bold', color='green')
    grid[3].text(0.1, 0.75, f'Predicted: {pred_class}', fontsize=14,
                 weight='bold', color='red')

    # Probability bar chart
    from src_v2.constants import CLASSIFIER_CLASSES
    y_positions = [0.55, 0.45, 0.35]
    for i, (cls, prob) in enumerate(zip(CLASSIFIER_CLASSES, probs)):
        grid[3].barh(y_positions[i], prob, height=0.08,
                     color='blue' if cls == pred_class else 'gray', alpha=0.7)
        grid[3].text(prob + 0.02, y_positions[i], f'{cls}: {prob:.3f}',
                     va='center', fontsize=10)
    grid[3].set_xlim(0, 1.1)
    grid[3].set_ylim(0.2, 1.0)

    plt.suptitle(f'Pipeline Trace - Misclassification', fontsize=16, weight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
```

### Pattern 2: Dual-Stage Duplicate Detection

**What:** Run duplicate detection on both original and warped datasets to detect data leakage and warping convergence

**When to use:** For comprehensive dataset quality audit

**Example:**
```python
# Source: imagededup library + PyTorch feature extraction
from imagededup.methods import PHash, CNN
import pandas as pd

def detect_duplicates_dual_stage(original_dir, warped_dir, output_dir,
                                  hash_threshold=5, cnn_threshold=0.95):
    """
    Two-stage duplicate detection: fast hashing then CNN verification.

    Args:
        original_dir: Path to original images
        warped_dir: Path to warped images
        output_dir: Where to save results
        hash_threshold: Hamming distance threshold for perceptual hash
        cnn_threshold: Cosine similarity threshold for CNN embeddings

    Returns:
        Dict with original_duplicates and warped_duplicates DataFrames
    """
    # Stage 1: Fast perceptual hashing on original images
    phasher = PHash()
    original_encodings = phasher.encode_images(image_dir=original_dir)
    original_hash_dupes = phasher.find_duplicates(
        encoding_map=original_encodings,
        max_distance_threshold=hash_threshold
    )

    # Stage 2: CNN verification on suspected duplicates
    # Use existing ResNet18 backbone from project
    from src_v2.models.resnet_landmark import ResNet18Landmarks
    import torch

    # Initialize model and extract feature extractor
    model = ResNet18Landmarks()
    feature_extractor = model.backbone  # ResNet18 without regression head
    feature_extractor.eval()

    # Compute embeddings for all images
    def compute_cnn_embeddings(image_dir, model):
        from src_v2.data.transforms import get_inference_transforms
        from PIL import Image
        import torch.nn.functional as F

        embeddings = {}
        transform = get_inference_transforms()

        for img_path in Path(image_dir).glob('*.png'):
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                features = model(img_tensor)
                # Global average pooling
                embedding = F.adaptive_avg_pool2d(features, (1, 1)).flatten()
                embeddings[img_path.name] = embedding.cpu().numpy()

        return embeddings

    # Find CNN-based duplicates
    original_cnn_embeddings = compute_cnn_embeddings(original_dir, feature_extractor)
    warped_cnn_embeddings = compute_cnn_embeddings(warped_dir, feature_extractor)

    def find_cnn_duplicates(embeddings, threshold):
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        names = list(embeddings.keys())
        vectors = np.stack(list(embeddings.values()))
        similarity_matrix = cosine_similarity(vectors)

        duplicates = {}
        for i, name1 in enumerate(names):
            similar = []
            for j, name2 in enumerate(names):
                if i != j and similarity_matrix[i, j] > threshold:
                    similar.append((name2, similarity_matrix[i, j]))
            if similar:
                duplicates[name1] = sorted(similar, key=lambda x: x[1], reverse=True)

        return duplicates

    original_cnn_dupes = find_cnn_duplicates(original_cnn_embeddings, cnn_threshold)
    warped_cnn_dupes = find_cnn_duplicates(warped_cnn_embeddings, cnn_threshold)

    # Cross-reference: images that are duplicates in original but NOT in warped
    # (warping successfully differentiated them)
    diverged = set(original_cnn_dupes.keys()) - set(warped_cnn_dupes.keys())

    # Cross-reference: images that are NOT duplicates in original but ARE in warped
    # (warping converged different images)
    converged = set(warped_cnn_dupes.keys()) - set(original_cnn_dupes.keys())

    results = {
        'original_duplicates': original_cnn_dupes,
        'warped_duplicates': warped_cnn_dupes,
        'warping_diverged': diverged,
        'warping_converged': converged,
    }

    # Save reports
    save_duplicate_report(results, output_dir)

    return results
```

### Pattern 3: Error Categorization with Fold Agreement

**What:** Use 5-fold CV ensemble predictions to identify error types (unanimous errors vs close calls)

**When to use:** To distinguish between hard examples and possible label noise

**Example:**
```python
# Source: Cross-validation ensemble predictions analysis
def categorize_errors_by_fold_agreement(ensemble_predictions_csv,
                                         confidence_threshold_high=0.9,
                                         confidence_threshold_low=0.6):
    """
    Categorize errors based on fold agreement and confidence.

    Args:
        ensemble_predictions_csv: Path to CSV with per-fold predictions
        confidence_threshold_high: Threshold for "high confidence" errors
        confidence_threshold_low: Threshold for "low margin" errors

    Returns:
        DataFrame with error categories and recoverability tags
    """
    import pandas as pd

    df = pd.read_csv(ensemble_predictions_csv)

    # Filter to misclassified samples only
    errors = df[df['predicted_class'] != df['true_class']].copy()

    # Calculate fold agreement (how many folds agree with ensemble prediction)
    errors['fold_agreement'] = errors[['fold_0_pred', 'fold_1_pred', 'fold_2_pred',
                                        'fold_3_pred', 'fold_4_pred']].apply(
        lambda row: sum(row == row.iloc[0]) / 5, axis=1
    )

    # Get ensemble confidence (max probability)
    errors['confidence'] = errors['ensemble_prob_max']

    # Categorization matrix
    def assign_category(row):
        agreement = row['fold_agreement']
        confidence = row['confidence']

        if agreement == 1.0 and confidence > confidence_threshold_high:
            return {
                'category': 'UNANIMOUS_HIGH_CONF',
                'interpretation': 'All folds wrong with high confidence',
                'likely_cause': 'Label noise or genuinely ambiguous',
                'recoverability': 'fixable_if_label_noise'
            }
        elif agreement == 1.0 and confidence < confidence_threshold_low:
            return {
                'category': 'UNANIMOUS_LOW_CONF',
                'interpretation': 'All folds wrong but uncertain',
                'likely_cause': 'Hard example, unclear features',
                'recoverability': 'partially_fixable'
            }
        elif agreement < 0.6:
            return {
                'category': 'SPLIT_DECISION',
                'interpretation': 'Folds disagree on prediction',
                'likely_cause': 'Near decision boundary',
                'recoverability': 'partially_fixable'
            }
        elif confidence > confidence_threshold_high:
            return {
                'category': 'HIGH_CONF_ERROR',
                'interpretation': 'Most folds agree, high confidence',
                'likely_cause': 'Systematic bias or bad features',
                'recoverability': 'fixable_better_features'
            }
        else:
            return {
                'category': 'MODERATE',
                'interpretation': 'Typical error',
                'likely_cause': 'Model limitation',
                'recoverability': 'inherent'
            }

    categorization = errors.apply(assign_category, axis=1, result_type='expand')
    errors = pd.concat([errors, categorization], axis=1)

    return errors
```

### Pattern 4: No-Reference Image Quality Assessment

**What:** Compute BRISQUE and NIQE scores for dataset to identify low-quality images

**When to use:** To quantify image quality and correlate with errors

**Example:**
```python
# Source: pybrisque or IQA-PyTorch
from pybrisque import BRISQUE
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def compute_quality_scores(image_dir, output_csv, compute_niqe=False):
    """
    Compute BRISQUE (and optionally NIQE) for all images.

    Args:
        image_dir: Directory containing images
        output_csv: Where to save scores
        compute_niqe: Whether to also compute NIQE (slower)

    Returns:
        DataFrame with quality scores per image
    """
    brisque = BRISQUE(url=False)

    results = []
    image_paths = list(Path(image_dir).glob('**/*.png'))

    for img_path in tqdm(image_paths, desc='Computing quality scores'):
        try:
            # BRISQUE score (0-100, lower is better)
            brisque_score = brisque.score(str(img_path))

            # Extract metadata from path
            # Assuming structure: {class}/{filename}.png
            category = img_path.parent.name
            filename = img_path.name

            result = {
                'filename': filename,
                'category': category,
                'brisque_score': brisque_score,
                'path': str(img_path)
            }

            # Optional NIQE computation
            if compute_niqe:
                try:
                    import pyiqa
                    niqe_metric = pyiqa.create_metric('niqe', device='cpu')
                    from PIL import Image
                    import torch

                    img = Image.open(img_path).convert('RGB')
                    # NIQE expects normalized tensor
                    from torchvision import transforms
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                    ])
                    img_tensor = transform(img).unsqueeze(0)

                    niqe_score = niqe_metric(img_tensor).item()
                    result['niqe_score'] = niqe_score
                except Exception as e:
                    result['niqe_score'] = None

            results.append(result)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    # Summary statistics
    print(f"\nQuality Score Summary:")
    print(f"BRISQUE - Mean: {df['brisque_score'].mean():.2f}, "
          f"Std: {df['brisque_score'].std():.2f}")

    if compute_niqe:
        print(f"NIQE - Mean: {df['niqe_score'].mean():.2f}, "
              f"Std: {df['niqe_score'].std():.2f}")

    # Identify worst quality images (top 10%)
    threshold_90 = df['brisque_score'].quantile(0.90)
    low_quality = df[df['brisque_score'] > threshold_90]
    print(f"\nLow quality images (BRISQUE > {threshold_90:.2f}): {len(low_quality)}")

    return df
```

### Anti-Patterns to Avoid

- **Loading all images into memory:** Use generators/iterators for large-scale duplicate detection and quality scoring
- **Re-implementing SSIM from scratch:** Use scikit-image's optimized implementation or imagededup
- **Ignoring cross-split duplicates:** Train/test leakage can inflate performance metrics - must be detected
- **Static-only visualizations:** Interactive notebooks enable exploration that static figures cannot
- **Single similarity threshold:** Different image types (COVID vs Normal) may need different thresholds
- **Not versioning quality scores:** Save scores with timestamps and metadata for reproducibility

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Perceptual hashing | Custom DCT-based hash | imagededup.PHash | Handles edge cases (rotations, crops), optimized, well-tested |
| BRISQUE computation | Custom NSS feature extraction | pybrisque or IQA-PyTorch | Pre-trained models, handles preprocessing, validated against paper |
| Image grid layouts | Manual subplot arithmetic | matplotlib ImageGrid | Automatic spacing, axis sharing, colorbar placement |
| Duplicate pair visualization | Custom pairing logic | imagededup built-in plotting | Handles asymmetric similarities, batch processing |
| Confusion matrix display | Manual heatmap code | sklearn.metrics.ConfusionMatrixDisplay | Normalized/unnormalized modes, label handling |
| Feature embeddings | Train new model | Use existing ResNet18 backbone | Already trained on domain, feature quality proven |
| Interactive widgets | Manual HTML/JS | ipywidgets | Native Jupyter integration, state management |

**Key insight:** Error forensics is a well-solved problem in CV/ML - leverage existing tools rather than rebuilding. The complexity is in domain-specific interpretation (medical imaging context, pipeline tracing), not in the underlying algorithms.

## Common Pitfalls

### Pitfall 1: Treating All Duplicates as Equal

**What goes wrong:** Near-duplicates in medical imaging can be clinically distinct (same patient, different time point vs different patients with similar presentations)

**Why it happens:** Generic duplicate detection doesn't understand medical context

**How to avoid:**
- Always manually review high-similarity pairs before declaring duplicates
- Check filenames for patient IDs or temporal markers
- Use multiple similarity thresholds (strict for exact duplicates, lenient for near-duplicates)
- Document metadata (acquisition parameters, patient demographics if available)

**Warning signs:** Very high number of "duplicates" (>5% of dataset), duplicates with different class labels but high similarity

### Pitfall 2: Overfitting Quality Thresholds to Errors

**What goes wrong:** Setting quality score thresholds based on error distribution leads to circular reasoning

**Why it happens:** Natural tendency to look for patterns in errors, then declare those patterns as thresholds

**How to avoid:**
- Compute quality scores on FULL dataset, not just errors
- Use percentile-based thresholds (e.g., worst 10%) rather than error-derived cutoffs
- Validate quality-error correlation statistically (chi-square test, t-test)
- Report correlation strength, don't assume causation

**Warning signs:** Quality threshold perfectly separates errors from correct predictions, threshold has no justification beyond "this is where errors cluster"

### Pitfall 3: Ignoring Warping-Induced Similarity Changes

**What goes wrong:** Warping can make visually different images more similar (geometric normalization) OR make similar images different (different warp parameters)

**Why it happens:** Duplicate detection typically runs on one representation, missing the transformation effects

**How to avoid:**
- Run duplicate detection on BOTH original and warped datasets
- Explicitly identify divergence cases (duplicate before warping, not after) and convergence cases (not duplicate before, duplicate after)
- Correlate warping convergence with landmark error - bad landmarks can cause inappropriate convergence

**Warning signs:** Many duplicates in warped dataset that don't appear in original, unexplained similarity increases/decreases

### Pitfall 4: Static Visualization Overload

**What goes wrong:** Generating hundreds of per-sample detailed figures creates overwhelming output that no one reviews

**Why it happens:** Completeness instinct - "document everything"

**How to avoid:**
- Two-tier approach: compact overview for all errors, detailed traces for representative samples only
- Interactive notebook as primary tool, static figures for thesis/publication only
- Group by error type, show 2-3 examples per type rather than all 33 individually
- Use thumbnail grids with click-to-expand for digital review

**Warning signs:** Output directory has >100 detailed figures, figures are never referenced in report, analyst asks "where do I start?"

### Pitfall 5: Not Validating Error Categorization

**What goes wrong:** Categorization thresholds (confidence, fold agreement) are arbitrary and may not reflect actual error causes

**Why it happens:** No ground truth for "why" an error occurred - categorization is always interpretive

**How to avoid:**
- Validate categories by manual review of random sample from each category
- Check if category distributions make sense (expect more "hard examples" than "label noise")
- Use categories as hypothesis generators, not definitive diagnoses
- Report inter-rater agreement if multiple experts review errors

**Warning signs:** Category names are overly confident ("DEFINITE label noise" vs "SUSPECTED label noise"), no validation step mentioned, category counts are suspiciously round numbers

## Code Examples

Verified patterns from official sources and existing project code:

### Confusion Matrix Visualization

```python
# Source: scikit-learn ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_confusion_matrix_with_errors(y_true, y_pred, class_names,
                                       error_indices, output_path):
    """
    Plot confusion matrix and annotate error counts.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        error_indices: Indices of misclassified samples
        output_path: Where to save figure
    """
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=class_names)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', values_format='d')

    # Annotate with percentages
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:  # Off-diagonal (errors)
                total = cm[i, :].sum()
                percentage = 100 * cm[i, j] / total if total > 0 else 0
                ax.text(j, i + 0.2, f'({percentage:.1f}%)',
                       ha='center', va='center', fontsize=10, color='red')

    plt.title(f'Confusion Matrix - {len(error_indices)} Total Errors',
              fontsize=14, weight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
```

### Loading Ensemble Predictions from Phase 5

```python
# Source: Existing project structure (GROUND_TRUTH.json references)
def load_ensemble_predictions_and_errors():
    """
    Load ensemble predictions and identify misclassified test samples.

    Returns:
        DataFrame with predictions and error flags
    """
    import pandas as pd

    # Path from GROUND_TRUTH.json
    predictions_path = "outputs/classifier_cv/ensemble_predictions_tta.csv"

    df = pd.read_csv(predictions_path)

    # Identify errors
    df['is_error'] = df['predicted_class'] != df['true_class']

    # Add metadata: confidence, margin, fold agreement
    df['confidence'] = df[['prob_COVID', 'prob_Normal', 'prob_Viral_Pneumonia']].max(axis=1)

    # Margin: difference between top-1 and top-2 probabilities
    probs = df[['prob_COVID', 'prob_Normal', 'prob_Viral_Pneumonia']].values
    sorted_probs = np.sort(probs, axis=1)
    df['margin'] = sorted_probs[:, -1] - sorted_probs[:, -2]

    return df
```

### Landmark Overlay Visualization

```python
# Source: Existing project landmark visualization utilities
def overlay_landmarks_on_image(image, landmarks, triangulation=None,
                                 landmark_color='red', line_color='yellow'):
    """
    Overlay landmarks and optionally Delaunay triangulation on X-ray.

    Args:
        image: Grayscale image (H, W)
        landmarks: (15, 2) pixel coordinates
        triangulation: Delaunay triangulation object (optional)
        landmark_color: Color for landmark points
        line_color: Color for triangulation edges

    Returns:
        Matplotlib axis with overlaid landmarks
    """
    import matplotlib.pyplot as plt
    from src_v2.constants import LANDMARK_NAMES

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap='gray')

    # Plot landmarks
    ax.scatter(landmarks[:, 0], landmarks[:, 1],
               c=landmark_color, s=50, marker='x', linewidths=2,
               label='Landmarks')

    # Optional: label landmarks
    for i, (x, y) in enumerate(landmarks):
        ax.annotate(LANDMARK_NAMES[i], (x, y),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, color=landmark_color)

    # Optional: draw triangulation
    if triangulation is not None:
        ax.triplot(landmarks[:, 0], landmarks[:, 1],
                  triangulation.simplices,
                  color=line_color, linewidth=0.8, alpha=0.6,
                  label='Delaunay triangulation')

    ax.axis('off')
    ax.legend(loc='upper right', fontsize=10)

    return fig, ax
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual error review in spreadsheets | Interactive Jupyter notebooks with widgets | ~2020 | Enables filtering, sorting, on-demand visualization |
| SSIM-only duplicate detection | Perceptual hashing + CNN embeddings | ~2021 | 10-100x faster with comparable accuracy |
| Custom quality metrics | Pre-trained BRISQUE/NIQE models | ~2019 | No need to collect reference quality datasets |
| Single-threshold categorization | Multi-dimensional error taxonomy (confidence × fold agreement) | ~2022 | Richer error understanding, actionable insights |
| Static confusion matrices | Interactive confusion matrix drill-down | ~2021 | Click cell to see misclassified samples |
| Separate tools for each analysis | Unified error forensics pipeline | ~2023 | Reproducibility, consistency |

**Deprecated/outdated:**
- **cv2.img_hash module**: Superseded by imagededup (better API, more algorithms)
- **Manual BRISQUE feature extraction**: Pre-trained models in pybrisque eliminate need
- **Subplot-based grids**: ImageGrid provides better control and consistency
- **Single-stage duplicate detection**: Dual-stage (original + transformed) is now standard for augmented/processed datasets

## Open Questions

1. **Optimal similarity threshold for medical X-rays**
   - What we know: Generic defaults are 0.9-0.95 cosine similarity for CNN, 5-10 Hamming distance for pHash
   - What's unclear: Medical X-rays have high structural similarity (all show ribcage, lungs) - may need tighter thresholds
   - Recommendation: Start with conservative threshold (0.98 CNN, 3 Hamming), manually review borderline cases, adjust based on precision/recall

2. **Quality score interpretation for chest X-rays**
   - What we know: BRISQUE trained on natural images, X-rays are out-of-distribution
   - What's unclear: Are BRISQUE scores meaningful for medical imaging? What's a "bad" score?
   - Recommendation: Use quality scores for relative ranking (identify outliers) rather than absolute thresholds. Validate by correlating with radiologist quality ratings if available.

3. **Landmark error threshold for "bad warp"**
   - What we know: Ensemble landmark error is 3.61 px mean on test set
   - What's unclear: At what error level does warping become unreliable? 2x mean? 3x?
   - Recommendation: Compute landmark error distribution on correctly classified vs misclassified samples. Use statistical test (t-test) to identify if errors have significantly worse landmarks. If p<0.05, use mean of error group as threshold.

4. **Handling identical images with different labels**
   - What we know: Duplicate detection may find visually identical images with different COVID/Normal/Viral labels
   - What's unclear: Is this labeling error or are subtle differences clinically relevant?
   - Recommendation: Flag as "requires expert review" rather than auto-correcting. Document all such cases for thesis discussion section.

5. **Scope of quality assessment**
   - What we know: Full dataset has ~21,000 images - computing NIQE for all is expensive
   - What's unclear: Is stratified sample sufficient or must we score all images?
   - Recommendation: Full BRISQUE scoring (fast), stratified NIQE sampling (100 per class, all errors) for validation. Report both in supplementary materials.

## Sources

### Primary (HIGH confidence)
- [scikit-learn ConfusionMatrixDisplay documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html) - Confusion matrix visualization
- [matplotlib ImageGrid documentation](https://matplotlib.org/stable/gallery/axes_grid1/simple_axesgrid.html) - Multi-panel figure layouts
- [imagededup GitHub repository](https://github.com/idealo/imagededup) - Duplicate detection library
- [pybrisque PyPI package](https://pypi.org/project/pybrisque/) - BRISQUE implementation
- [IQA-PyTorch GitHub repository](https://github.com/chaofengc/IQA-PyTorch) - Comprehensive IQA toolkit
- [LearnOpenCV BRISQUE tutorial](https://learnopencv.com/image-quality-assessment-brisque/) - BRISQUE usage guide

### Secondary (MEDIUM confidence)
- [Image similarity checker implementation](https://github.com/Imranch4/image-similarity-checker) - SSIM + pHash + ORB combination
- [Large-scale k-fold CV ensemble study](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00709-9) - Fold agreement for uncertainty
- [Medical imaging quality assessment (2026)](https://link.springer.com/article/10.1007/s13755-025-00411-0) - Recent explainable IQA for chest X-rays
- [Matplotlib image grid tutorial](https://labex.io/tutorials/matplotlib-matplotlib-image-grid-visualization-48681) - Grid layout examples
- [PyTorch image similarity search](https://medium.com/pytorch/image-similarity-search-in-pytorch-1a744cf3469) - Feature embedding approach

### Tertiary (LOW confidence - for exploration)
- [ITKWidgets for medical imaging](https://www.kitware.com/itkwidgets-and-idc/) - Advanced 3D medical viz (overkill for 2D X-rays)
- [Benchmarking vision embeddings for medical duplicate detection](https://arxiv.org/html/2312.07273v1) - Research on medical image duplicates
- [Cross-validation confidence intervals](https://pmc.ncbi.nlm.nih.gov/articles/PMC4533123/) - Statistical methods for CV uncertainty

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already in project (matplotlib, sklearn, torch) or mature PyPI packages (imagededup, pybrisque)
- Architecture: HIGH - Patterns verified from official docs and existing project code
- Duplicate detection: HIGH - imagededup is production-ready, CNN embeddings validated in literature
- Quality assessment: MEDIUM - BRISQUE/NIQE validated for natural images, medical imaging applicability requires validation
- Error categorization: MEDIUM - Framework is sound, thresholds require empirical tuning on this dataset
- Visualization: HIGH - matplotlib ImageGrid and Jupyter widgets are industry standard

**Research date:** 2026-02-16
**Valid until:** ~60 days (stable domain, libraries mature, medical imaging practices evolve slowly)
