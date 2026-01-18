"""
Visualization functions for the GUI.

Provides rendering functions for landmarks, warped images, GradCAM heatmaps,
and comparison views.
"""
import io
from typing import Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Use non-interactive backend for thread safety
matplotlib.use('Agg')

from .config import (
    LANDMARK_COLORS,
    LANDMARK_GROUPS,
    LANDMARK_LABELS_ES,
    CLASS_COLORS,
    CLASS_NAMES_ES,
)
from .gradcam_utils import overlay_heatmap_on_image


def render_original(image: np.ndarray) -> Image.Image:
    """
    Render original image with minimal processing.

    Args:
        image: Input image (H, W) grayscale or (H, W, 3) RGB

    Returns:
        PIL Image
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    if len(image.shape) == 2:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)

    ax.axis('off')
    ax.set_title('Imagen Original', fontsize=14, color='white', pad=10)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Convert to PIL
    pil_image = _fig_to_pil(fig)
    plt.close(fig)

    return pil_image


def render_landmarks_overlay(
    image: np.ndarray,
    landmarks: np.ndarray,
    show_labels: bool = True,
    image_size: int = 224
) -> Image.Image:
    """
    Render landmarks overlaid on image with color-coded groups.

    Args:
        image: Input image (H, W) grayscale or (H, W, 3) RGB
        landmarks: Landmark coordinates (15, 2) in pixel space
        show_labels: Whether to show L1-L15 labels
        image_size: Expected image size for validation

    Returns:
        PIL Image with landmarks overlay
    """
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

    # Display image
    if len(image.shape) == 2:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)

    # Ensure landmarks are in pixel coordinates
    if landmarks.max() <= 1.0:
        landmarks_px = landmarks * image_size
    else:
        landmarks_px = landmarks

    # Draw each landmark with color by group
    for i in range(15):
        group = LANDMARK_GROUPS[i]
        color = LANDMARK_COLORS[group]

        # Draw landmark point (circle with white border)
        ax.scatter(
            landmarks_px[i, 0],
            landmarks_px[i, 1],
            c=color,
            s=150,
            marker='o',
            edgecolors='white',
            linewidths=2,
            alpha=0.9,
            zorder=10
        )

        # Add label if requested
        if show_labels:
            ax.annotate(
                f'L{i+1}',
                (landmarks_px[i, 0] + 3, landmarks_px[i, 1] - 3),
                fontsize=10,
                color='white',
                weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6),
                zorder=11
            )

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=LANDMARK_COLORS['axis'],
                   markersize=12, label=LANDMARK_LABELS_ES['axis']),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=LANDMARK_COLORS['central'],
                   markersize=12, label=LANDMARK_LABELS_ES['central']),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=LANDMARK_COLORS['lateral'],
                   markersize=12, label=LANDMARK_LABELS_ES['lateral']),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=LANDMARK_COLORS['border'],
                   markersize=12, label=LANDMARK_LABELS_ES['border']),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=LANDMARK_COLORS['costal'],
                   markersize=12, label=LANDMARK_LABELS_ES['costal']),
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=10,
        facecolor='black',
        edgecolor='white',
        labelcolor='white',
        framealpha=0.8
    )

    ax.axis('off')
    ax.set_title('Landmarks Detectados (15 puntos)', fontsize=14, color='white', pad=10)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Convert to PIL
    pil_image = _fig_to_pil(fig)
    plt.close(fig)

    return pil_image


def render_warped(warped_image: np.ndarray) -> Image.Image:
    """
    Render warped (normalized) image.

    Args:
        warped_image: Warped image (H, W) grayscale

    Returns:
        PIL Image
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    ax.imshow(warped_image, cmap='gray')
    ax.axis('off')
    ax.set_title('Imagen Normalizada (Warped)', fontsize=14, color='white', pad=10)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Convert to PIL
    pil_image = _fig_to_pil(fig)
    plt.close(fig)

    return pil_image


def render_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    predicted_class: Optional[str] = None
) -> Image.Image:
    """
    Render GradCAM heatmap overlay on image.

    Args:
        image: Original image (H, W) grayscale or (H, W, 3) RGB
        heatmap: Normalized heatmap (H, W) with values [0, 1]
        alpha: Heatmap transparency (0=invisible, 1=opaque)
        predicted_class: Optional predicted class name for title

    Returns:
        PIL Image with GradCAM overlay
    """
    # Create overlay
    overlay = overlay_heatmap_on_image(image, heatmap, alpha=alpha)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    ax.imshow(overlay)
    ax.axis('off')

    # Title with class name if provided
    if predicted_class:
        title = f'GradCAM: Regiones de Atención\nClase: {predicted_class}'
    else:
        title = 'GradCAM: Regiones de Atención'

    ax.set_title(title, fontsize=14, color='white', pad=10)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Add colorbar to show heatmap scale
    # Create a dummy mappable for colorbar
    from matplotlib import cm
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Activación', rotation=270, labelpad=20, color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Convert to PIL
    pil_image = _fig_to_pil(fig)
    plt.close(fig)

    return pil_image


def render_comparison_side_by_side(
    original: np.ndarray,
    warped: np.ndarray
) -> Image.Image:
    """
    Render side-by-side comparison of original and warped images.

    Args:
        original: Original image (H, W)
        warped: Warped image (H, W)

    Returns:
        PIL Image with comparison
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=150)

    # Original
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original', fontsize=16, color='white')
    ax1.axis('off')

    # Warped
    ax2.imshow(warped, cmap='gray')
    ax2.set_title('Normalizada (Warped)', fontsize=16, color='white')
    ax2.axis('off')

    fig.patch.set_facecolor('black')
    plt.tight_layout()

    # Convert to PIL
    pil_image = _fig_to_pil(fig)
    plt.close(fig)

    return pil_image


def create_probability_chart(
    probabilities: dict,
    title: str = "Probabilidades de Clasificación"
) -> Image.Image:
    """
    Create horizontal bar chart of class probabilities.

    Args:
        probabilities: Dict mapping class names to probabilities
        title: Chart title

    Returns:
        PIL Image with bar chart
    """
    # Convert class names to Spanish
    classes_es = []
    probs = []
    colors = []

    for cls, prob in probabilities.items():
        if cls in ['COVID', 'Normal', 'Viral_Pneumonia']:
            idx = ['COVID', 'Normal', 'Viral_Pneumonia'].index(cls)
            classes_es.append(CLASS_NAMES_ES[idx])
            probs.append(prob * 100)  # Convert to percentage
            colors.append(CLASS_COLORS[cls])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)

    # Horizontal bars
    y_pos = np.arange(len(classes_es))
    bars = ax.barh(y_pos, probs, color=colors, alpha=0.8, edgecolor='white', linewidth=2)

    # Add percentage labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        width = bar.get_width()
        ax.text(
            width + 1,
            bar.get_y() + bar.get_height() / 2,
            f'{prob:.2f}%',
            ha='left',
            va='center',
            fontsize=12,
            color='white',
            weight='bold'
        )

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes_es, fontsize=12, color='white')
    ax.set_xlabel('Probabilidad (%)', fontsize=12, color='white')
    ax.set_title(title, fontsize=14, color='white', pad=15)
    ax.set_xlim([0, 105])

    # Grid
    ax.grid(axis='x', alpha=0.3, color='white', linestyle='--')
    ax.set_axisbelow(True)

    # Background
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('black')

    # Tick colors
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Spine colors
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    plt.tight_layout()

    # Convert to PIL
    pil_image = _fig_to_pil(fig)
    plt.close(fig)

    return pil_image


def create_metrics_table(
    landmarks: np.ndarray,
    per_landmark_errors: Optional[dict] = None
) -> pd.DataFrame:
    """
    Create metrics table with landmark coordinates and errors.

    Args:
        landmarks: Predicted landmarks (15, 2) in pixel space
        per_landmark_errors: Optional dict with reference errors per landmark

    Returns:
        DataFrame with metrics
    """
    from .config import PER_LANDMARK_ERRORS

    if per_landmark_errors is None:
        per_landmark_errors = PER_LANDMARK_ERRORS

    data = []
    for i in range(15):
        landmark_name = f'L{i+1}'
        x, y = landmarks[i]
        error = per_landmark_errors.get(landmark_name, np.nan)
        group = LANDMARK_GROUPS[i]

        data.append({
            'Landmark': landmark_name,
            'Grupo': LANDMARK_LABELS_ES[group],
            'X (px)': f'{x:.1f}',
            'Y (px)': f'{y:.1f}',
            'Error Ref. (px)': f'{error:.2f}' if not np.isnan(error) else 'N/A'
        })

    df = pd.DataFrame(data)
    return df


def export_to_pdf(
    images: dict,
    metrics_df: pd.DataFrame,
    output_path: str,
    metadata: Optional[dict] = None
):
    """
    Export all visualizations to a multi-page PDF.

    Args:
        images: Dict with keys: 'original', 'landmarks', 'warped', 'gradcam'
        metrics_df: DataFrame with metrics
        output_path: Output PDF path
        metadata: Optional metadata (inference time, etc.)
    """
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(output_path) as pdf:
        # Page 1: Original + Landmarks
        if 'landmarks' in images:
            fig = plt.figure(figsize=(11, 8.5))
            plt.imshow(images['landmarks'])
            plt.axis('off')
            plt.title('Landmarks Detectados', fontsize=16, color='black', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        # Page 2: Warped
        if 'warped' in images:
            fig = plt.figure(figsize=(11, 8.5))
            plt.imshow(images['warped'], cmap='gray')
            plt.axis('off')
            plt.title('Imagen Normalizada', fontsize=16, color='black', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        # Page 3: GradCAM
        if 'gradcam' in images:
            fig = plt.figure(figsize=(11, 8.5))
            plt.imshow(images['gradcam'])
            plt.axis('off')
            plt.title('GradCAM: Regiones de Atención', fontsize=16, color='black', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        # Page 4: Metrics table
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(
            cellText=metrics_df.values,
            colLabels=metrics_df.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.15, 0.25, 0.15, 0.15, 0.2]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(len(metrics_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Metadata
        if metadata:
            metadata_text = '\n'.join([f'{k}: {v}' for k, v in metadata.items()])
            ax.text(
                0.5, 0.95, metadata_text,
                transform=fig.transFigure,
                ha='center', va='top',
                fontsize=10, family='monospace'
            )

        plt.title('Métricas Detalladas', fontsize=16, pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Metadata
        d = pdf.infodict()
        d['Title'] = 'Resultados de Detección COVID-19'
        d['Author'] = 'Sistema de Landmarks Anatómicos'
        d['Subject'] = 'Análisis de radiografía de tórax'
        d['Keywords'] = 'COVID-19, Landmarks, Deep Learning, GradCAM'
        d['CreationDate'] = None  # Will use current time


def _fig_to_pil(fig: plt.Figure) -> Image.Image:
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=150)
    buf.seek(0)
    pil_image = Image.open(buf).copy()  # Copy to avoid issues when closing buffer
    buf.close()
    return pil_image
