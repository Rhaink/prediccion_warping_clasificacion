"""
Visualization utilities for error analysis and pipeline tracing.

This module provides functions to create thesis-ready visualizations of classification
errors, pipeline traces (original -> landmarks -> warped -> result), and overview grids.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from src_v2.constants import LANDMARK_NAMES, CATEGORIES
from src_v2.processing.gpa import compute_delaunay_triangulation


logger = logging.getLogger(__name__)


def overlay_landmarks(
    ax: plt.Axes,
    image: np.ndarray,
    landmarks: np.ndarray,
    triangulation: Optional[np.ndarray] = None
) -> plt.Axes:
    """
    Overlay landmarks on an image with optional Delaunay triangulation.

    Args:
        ax: Matplotlib axis to plot on
        image: Grayscale image (H, W)
        landmarks: Array of shape (15, 2) with (x, y) coordinates
        triangulation: Optional triangulation indices for edges

    Returns:
        Modified axis
    """
    # Show image
    ax.imshow(image, cmap='gray', vmin=0, vmax=255)

    # Plot landmark points
    ax.scatter(landmarks[:, 0], landmarks[:, 1], c='red', marker='x', s=30, linewidths=1.5)

    # Plot triangulation edges if provided
    if triangulation is not None:
        for tri in triangulation:
            triangle = landmarks[tri]
            # Close the triangle by appending first point
            triangle_closed = np.vstack([triangle, triangle[0]])
            ax.plot(triangle_closed[:, 0], triangle_closed[:, 1],
                   'y-', alpha=0.5, linewidth=0.8)

    # Add landmark labels
    for i, (x, y) in enumerate(landmarks):
        ax.text(x, y, LANDMARK_NAMES[i], fontsize=6, color='white',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

    ax.axis('off')
    return ax


def visualize_pipeline_trace(
    original_img: np.ndarray,
    landmarks: np.ndarray,
    warped_img: np.ndarray,
    true_class: str,
    pred_class: str,
    probs: np.ndarray,
    category: str,
    failure_origin: str,
    output_path: Path,
    triangulation: Optional[np.ndarray] = None
) -> None:
    """
    Create 4-panel pipeline trace visualization.

    Panels:
    1. Original X-ray
    2. Landmarks overlay on original
    3. Warped image
    4. Classification result with probability bars

    Args:
        original_img: Original grayscale image
        landmarks: Predicted landmarks (15, 2)
        warped_img: Warped image
        true_class: True class name
        pred_class: Predicted class name
        probs: Probability vector (3,)
        category: Error category
        failure_origin: Pipeline failure origin
        output_path: Path to save figure
        triangulation: Optional Delaunay triangulation
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5),
                             gridspec_kw={'width_ratios': [1, 1, 1, 0.8]})
    fig.subplots_adjust(wspace=0.3, top=0.85)

    # Panel 1: Original X-ray
    axes[0].imshow(original_img, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original X-ray', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Landmarks overlay
    overlay_landmarks(axes[1], original_img, landmarks, triangulation)
    axes[1].set_title('Landmarks Overlay', fontsize=12, fontweight='bold')

    # Panel 3: Warped image
    axes[2].imshow(warped_img, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('Warped Image', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # Panel 4: Classification result with probability bars
    axes[3].axis('off')
    axes[3].set_xlim(0, 1)
    axes[3].set_ylim(0, 1)

    # Text info
    axes[3].text(0.05, 0.95, f"True: {true_class}", fontsize=13, fontweight='bold',
                 color='green', va='top', transform=axes[3].transAxes)
    axes[3].text(0.05, 0.85, f"Pred: {pred_class}", fontsize=13, fontweight='bold',
                 color='red', va='top', transform=axes[3].transAxes)
    axes[3].text(0.05, 0.75, f"Category: {category}", fontsize=10,
                 va='top', transform=axes[3].transAxes)
    axes[3].text(0.05, 0.67, f"Failure: {failure_origin}", fontsize=10, style='italic',
                 va='top', transform=axes[3].transAxes)

    # Horizontal probability bars
    bar_colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    y_positions = [0.45, 0.30, 0.15]
    for i, (cls, prob) in enumerate(zip(CATEGORIES, probs)):
        y = y_positions[i]
        axes[3].barh(y, prob, height=0.10, color=bar_colors[i], alpha=0.8,
                     left=0.0, transform=axes[3].transAxes, clip_on=False)
        axes[3].text(0.0, y + 0.06, f"{cls}: {prob:.1%}", fontsize=9,
                     va='bottom', transform=axes[3].transAxes)

    # Main title
    title = f"True: {true_class} | Pred: {pred_class} | {category} | {failure_origin}"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.debug(f"Saved pipeline trace to {output_path}")


def create_overview_grid(
    error_samples: List[Dict],
    output_path: Path,
    ncols: int = 6
) -> None:
    """
    Create compact grid overview of all misclassified samples.

    Shows warped image thumbnails with annotations:
    - Border color: red (high conf), orange (low margin), gray (moderate)
    - Text: true -> predicted class, confidence score

    Args:
        error_samples: List of error sample dictionaries
        output_path: Path to save figure
        ncols: Number of columns in grid
    """
    n_errors = len(error_samples)
    nrows = (n_errors + ncols - 1) // ncols  # Ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3.5))
    fig.subplots_adjust(hspace=0.5, wspace=0.3, top=0.93)
    axes = axes.flatten() if nrows > 1 else (axes if ncols > 1 else [axes])

    # Sort by confusion pair for organized layout
    sorted_samples = sorted(
        error_samples,
        key=lambda x: (x['true_class'], x['predicted_class'], -x['confidence'])
    )

    for i, sample in enumerate(sorted_samples):
        if i >= len(axes):
            break

        ax = axes[i]

        # Load warped image
        warped_path = sample.get('warped_path')
        if warped_path and Path(warped_path).exists():
            warped_img = cv2.imread(str(warped_path), cv2.IMREAD_GRAYSCALE)
        else:
            warped_img = np.zeros((224, 224), dtype=np.uint8)

        # Determine border color based on confidence
        conf = sample['confidence']
        if conf > 0.9:
            border_color = 'red'
        elif sample.get('margin', 1.0) < 0.2:
            border_color = 'orange'
        else:
            border_color = 'gray'

        ax.imshow(warped_img, cmap='gray', vmin=0, vmax=255)

        # Add colored border
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
            spine.set_visible(True)
        ax.set_xticks([])
        ax.set_yticks([])

        # Annotation
        true_cls = sample['true_class']
        pred_cls = sample['predicted_class']
        title_text = f"{true_cls}\n-> {pred_cls}\n{conf:.1%}"
        ax.set_title(title_text, fontsize=8, fontweight='bold')

    # Hide unused subplots
    for i in range(len(sorted_samples), len(axes)):
        axes[i].axis('off')

    title = f"Resumen de {n_errors} errores de clasificacion - Ensemble+TTA (98.26%)"
    fig.suptitle(title, fontsize=16, fontweight='bold')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved overview grid to {output_path}")
