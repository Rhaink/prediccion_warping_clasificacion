"""
Block A: Pipeline Overview Visualizations

Generates:
- A1: Complete flow of a single image through the entire pipeline
- A2: Comparison grid of original vs warped images
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.glass_box_visualizations.utils import (
    load_image, draw_landmarks_on_image, save_figure,
    load_representative_samples, create_comparison_grid, add_panel_labels,
    load_ground_truth
)
from src_v2.models.resnet_landmark import ResNet18Landmarks
from src_v2.models.classifier import ImageClassifier
from src_v2.processing.warp import piecewise_affine_warp
from src_v2.processing.gpa import load_canonical_shape
from src_v2.data.transforms import apply_clahe_transform


def generate_A1_complete_flow(
    output_dir: Path,
    predictions_path: Path,
    data_dir: Path,
    warped_dir: Path,
    canonical_shape_path: Path,
    classifier_path: Path
) -> None:
    """
    Generate Figure A1: Journey of a Single Image.

    Shows one image going through all pipeline steps:
    (a) Original image with hospital marks
    (b) Landmarks detected (15 points)
    (c) Warped image (normalized)
    (d) Classification result

    Args:
        output_dir: Directory to save outputs
        predictions_path: Path to predictions.npz
        data_dir: Path to original dataset
        warped_dir: Path to warped images
        canonical_shape_path: Path to canonical shape
        classifier_path: Path to trained classifier
    """
    print("Generating A1: Complete Flow...")

    # Load ground truth for metrics
    gt = load_ground_truth()
    landmark_error = gt['landmark_detection']['ensemble_best']['mean_error_px']
    landmark_std = gt['landmark_detection']['ensemble_best']['std_error_px']

    # Load a representative sample (COVID case with visible artifacts)
    samples = load_representative_samples(
        predictions_path,
        n_per_class=1,
        criteria='diverse',
        seed=42
    )

    # Find a COVID sample
    covid_sample = [s for s in samples if s['category'] == 'COVID'][0]

    # Load original image
    original_image = load_image(covid_sample['image_path'])
    landmarks = covid_sample['landmarks']

    # Create 4-panel figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Panel (a): Original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('(a) Imagen Original\n(299×299)', fontsize=11, weight='bold')
    axes[0].axis('off')
    axes[0].text(0.5, -0.08, 'Con marcas del hospital',
                ha='center', va='top', transform=axes[0].transAxes,
                fontsize=9, style='italic')

    # Panel (b): Landmarks detected
    img_with_landmarks = draw_landmarks_on_image(
        original_image,
        landmarks,
        radius=4,
        show_connections=True
    )
    axes[1].imshow(img_with_landmarks)
    axes[1].set_title(f'(b) Landmarks Detectados\n(15 puntos)', fontsize=11, weight='bold')
    axes[1].axis('off')
    axes[1].text(0.5, -0.08, f'Error: {landmark_error:.2f} ± {landmark_std:.2f} px',
                ha='center', va='top', transform=axes[1].transAxes,
                fontsize=9, style='italic')

    # Panel (c): Warped image
    # Load warped version
    warped_image_path = _find_warped_image(covid_sample['image_path'], warped_dir)
    if warped_image_path.exists():
        warped_image = load_image(warped_image_path)
    else:
        # Generate warped image on the fly
        print(f"Warped image not found, generating on the fly...")
        canonical_shape, triangulation = load_canonical_shape(canonical_shape_path)
        warped_image = piecewise_affine_warp(
            original_image,
            landmarks,
            canonical_shape,
            triangulation,
            output_size=(224, 224),
            margin_scale=1.05
        )

    axes[2].imshow(warped_image, cmap='gray')
    axes[2].set_title('(c) Imagen Normalizada\n(224×224)', fontsize=11, weight='bold')
    axes[2].axis('off')
    axes[2].text(0.5, -0.08, 'Pulmones alineados a forma estándar',
                ha='center', va='top', transform=axes[2].transAxes,
                fontsize=9, style='italic')

    # Panel (d): Classification
    # Load classifier and predict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = ImageClassifier(num_classes=3, model_name='resnet18')
    checkpoint = torch.load(classifier_path, map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()

    # Preprocess warped image
    img_tensor = torch.from_numpy(warped_image).float().unsqueeze(0).unsqueeze(0) / 255.0
    img_tensor = img_tensor.repeat(1, 3, 1, 1)  # Convert to 3 channels

    with torch.no_grad():
        outputs = classifier(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    class_names = ['COVID', 'Normal', 'Viral']
    colors = ['#2ECC71', '#3498DB', '#F39C12']

    # Show small warped image
    axes[3].imshow(warped_image, cmap='gray', extent=[0, 0.3, 0, 1])
    axes[3].set_xlim(0, 1)
    axes[3].set_ylim(0, 1)

    # Add probability bars
    bar_width = 0.6
    bar_start_x = 0.35
    for i, (cls, prob, color) in enumerate(zip(class_names, probs, colors)):
        y = 0.7 - i * 0.25
        # Background bar
        axes[3].barh(y, bar_width, height=0.08, left=bar_start_x,
                    color='lightgray', edgecolor='black', linewidth=1)
        # Filled bar
        axes[3].barh(y, prob * bar_width, height=0.08, left=bar_start_x,
                    color=color, edgecolor='black', linewidth=1)
        # Label
        axes[3].text(bar_start_x - 0.02, y, cls, ha='right', va='center',
                    fontsize=9, weight='bold')
        # Percentage
        axes[3].text(bar_start_x + bar_width + 0.02, y, f'{prob*100:.1f}%',
                    ha='left', va='center', fontsize=9)

    axes[3].set_title('(d) Clasificación', fontsize=11, weight='bold')
    axes[3].axis('off')

    predicted_class = class_names[np.argmax(probs)]
    confidence = probs.max() * 100
    axes[3].text(0.5, -0.08, f'Predicción: {predicted_class} ({confidence:.1f}% confianza)',
                ha='center', va='top', transform=axes[3].transAxes,
                fontsize=9, style='italic', weight='bold')

    # Add arrows between panels
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    for i in range(3):
        fig.patches.extend([plt.Arrow(
            (i + 1) / 4 - 0.02, 0.5,
            0.04, 0,
            transform=fig.transFigure,
            **arrow_props
        )])

    plt.suptitle('Pipeline Completo: Detección de COVID-19 desde Radiografías',
                fontsize=14, weight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = output_dir / 'block_a' / 'A1_complete_flow.png'
    save_figure(fig, output_path, dpi=300)
    plt.close(fig)


def generate_A2_comparison_grid(
    output_dir: Path,
    predictions_path: Path,
    data_dir: Path,
    warped_dir: Path,
    n_examples: int = 12
) -> None:
    """
    Generate Figure A2: Comparison Grid (Original vs Warped).

    Shows multiple examples of original images vs their warped counterparts
    in a 2-row grid.

    Args:
        output_dir: Directory to save outputs
        predictions_path: Path to predictions.npz
        data_dir: Path to original dataset
        warped_dir: Path to warped images
        n_examples: Number of example pairs to show
    """
    print("Generating A2: Comparison Grid...")

    # Load diverse samples from all classes
    samples = load_representative_samples(
        predictions_path,
        n_per_class=n_examples // 3,
        criteria='diverse',
        seed=123
    )[:n_examples]

    # Create figure with 2 rows
    n_cols = n_examples
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 2, 4))

    for col_idx, sample in enumerate(samples):
        # Row 1: Original image
        original_image = load_image(sample['image_path'])
        axes[0, col_idx].imshow(original_image, cmap='gray')
        axes[0, col_idx].axis('off')

        # Add category label
        category = sample['category']
        color_map = {'COVID': '#2ECC71', 'Normal': '#3498DB', 'Viral': '#F39C12'}
        axes[0, col_idx].set_title(category, fontsize=9, color=color_map.get(category, 'black'))

        # Row 2: Warped image
        warped_image_path = _find_warped_image(sample['image_path'], warped_dir)
        if warped_image_path.exists():
            warped_image = load_image(warped_image_path)
            axes[1, col_idx].imshow(warped_image, cmap='gray')
        else:
            axes[1, col_idx].text(0.5, 0.5, 'Warped\nimage\nmissing',
                                 ha='center', va='center',
                                 transform=axes[1, col_idx].transAxes)
        axes[1, col_idx].axis('off')

    # Add row labels
    axes[0, 0].text(-0.15, 0.5, 'ORIGINAL\n(con artefactos)',
                   rotation=90, ha='right', va='center',
                   transform=axes[0, 0].transAxes,
                   fontsize=11, weight='bold')
    axes[1, 0].text(-0.15, 0.5, 'WARPED\n(normalizado)',
                   rotation=90, ha='right', va='center',
                   transform=axes[1, 0].transAxes,
                   fontsize=11, weight='bold')

    plt.suptitle('Comparación: Imágenes Originales vs Normalizadas',
                fontsize=14, weight='bold', y=0.98)

    plt.tight_layout(rect=[0.05, 0, 1, 0.96])

    output_path = output_dir / 'block_a' / 'A2_comparison_grid.png'
    save_figure(fig, output_path, dpi=300)
    plt.close(fig)


def _find_warped_image(original_path: str, warped_dir: Path) -> Path:
    """Find corresponding warped image for an original image."""
    # Extract category and filename from original path
    original_path = Path(original_path)
    category = original_path.parent.name  # COVID, Normal, Viral Pneumonia
    filename = original_path.name

    # Try different possible structures
    possible_paths = [
        warped_dir / category / filename,
        warped_dir / 'session_warping' / category / filename,
        warped_dir / filename,
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # If not found, return first possibility (will be checked for existence later)
    return possible_paths[0]


def main():
    """Generate all Block A visualizations."""
    # Paths configuration
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / 'outputs' / 'glass_box'
    predictions_path = project_root / 'outputs' / 'landmark_predictions' / 'session_warping' / 'predictions.npz'
    data_dir = project_root / 'data' / 'dataset' / 'COVID-19_Radiography_Dataset'
    warped_dir = project_root / 'outputs' / 'warped_lung_best' / 'session_warping'
    canonical_shape_path = project_root / 'outputs' / 'shape_analysis' / 'canonical_shape.npz'
    classifier_path = project_root / 'outputs' / 'classifier_warped_lung_best' / 'best_classifier.pt'

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'block_a').mkdir(exist_ok=True)

    # Check required files
    if not predictions_path.exists():
        print(f"ERROR: Predictions not found at {predictions_path}")
        print("Please run: python scripts/predict_landmarks_dataset.py")
        return

    # Generate visualizations
    try:
        generate_A1_complete_flow(
            output_dir,
            predictions_path,
            data_dir,
            warped_dir,
            canonical_shape_path,
            classifier_path
        )
    except Exception as e:
        print(f"Error generating A1: {e}")
        import traceback
        traceback.print_exc()

    try:
        generate_A2_comparison_grid(
            output_dir,
            predictions_path,
            data_dir,
            warped_dir,
            n_examples=12
        )
    except Exception as e:
        print(f"Error generating A2: {e}")
        import traceback
        traceback.print_exc()

    print("\nBlock A visualizations complete!")
    print(f"Outputs saved to: {output_dir / 'block_a'}")


if __name__ == '__main__':
    main()
