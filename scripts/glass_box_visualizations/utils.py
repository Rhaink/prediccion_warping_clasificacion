"""
Common utilities for glass-box visualizations.

This module provides helper functions used across all visualization scripts:
- Loading representative samples from the dataset
- Creating comparison grids
- Adding annotations and labels
- Color palette management
- Image preprocessing utilities
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches
from PIL import Image


# Color palette for consistent styling across all figures
COLORS = {
    'landmark_correct': '#2ECC71',      # Green
    'landmark_error': '#E74C3C',        # Red
    'attention_high': '#E74C3C',        # Red
    'attention_low': '#3498DB',         # Blue
    'covid': '#2ECC71',                 # Green
    'normal': '#3498DB',                # Blue
    'viral': '#F39C12',                 # Yellow/Orange
    'background': '#95A5A6',            # Gray
    'inactive': '#BDC3C7',              # Light gray
}

# Landmark colors for visualization
LANDMARK_COLORS = {
    'central': '#E74C3C',      # Red for central landmarks
    'left': '#3498DB',         # Blue for left lung
    'right': '#2ECC71',        # Green for right lung
}


def load_ground_truth() -> Dict:
    """Load validated metrics from GROUND_TRUTH.json."""
    ground_truth_path = Path(__file__).parent.parent.parent / 'GROUND_TRUTH.json'
    with open(ground_truth_path, 'r') as f:
        return json.load(f)


def load_representative_samples(
    predictions_path: Union[str, Path],
    n_per_class: int = 5,
    criteria: str = 'diverse',
    seed: int = 42
) -> List[Dict]:
    """
    Load representative samples from the dataset.

    Args:
        predictions_path: Path to predictions.npz file
        n_per_class: Number of examples per class
        criteria: Selection criteria ('diverse', 'typical', 'difficult')
        seed: Random seed for reproducibility

    Returns:
        List of dicts with keys: image_path, landmarks, category, metadata
    """
    np.random.seed(seed)

    # Load predictions
    data = np.load(predictions_path, allow_pickle=True)
    predictions = data['predictions']  # Shape: (N, 15, 2)
    image_paths = data['image_paths']

    # Extract categories from paths
    categories = []
    for path in image_paths:
        path_str = str(path)
        if 'COVID' in path_str:
            categories.append('COVID')
        elif 'Normal' in path_str:
            categories.append('Normal')
        elif 'Viral' in path_str:
            categories.append('Viral')
        else:
            categories.append('Unknown')

    categories = np.array(categories)

    # Select samples based on criteria
    samples = []
    for category in ['COVID', 'Normal', 'Viral']:
        mask = categories == category
        indices = np.where(mask)[0]

        if criteria == 'diverse':
            # Maximize variance of landmarks
            landmarks_subset = predictions[mask]
            centroid = landmarks_subset.mean(axis=1)  # (N, 2)
            distances_from_mean = np.linalg.norm(
                centroid - centroid.mean(axis=0), axis=1
            )
            selected = np.argsort(distances_from_mean)[-n_per_class:]
            selected_indices = indices[selected]

        elif criteria == 'typical':
            # Select samples close to median
            landmarks_subset = predictions[mask]
            centroid = landmarks_subset.mean(axis=1)
            distances_from_mean = np.linalg.norm(
                centroid - centroid.mean(axis=0), axis=1
            )
            median_idx = np.argsort(distances_from_mean)[len(distances_from_mean)//2]
            # Get n_per_class closest to median
            selected = np.argsort(distances_from_mean)[
                max(0, median_idx-n_per_class//2):median_idx+n_per_class//2+1
            ][:n_per_class]
            selected_indices = indices[selected]

        elif criteria == 'difficult':
            # Select samples with high prediction variance (if available)
            # For now, use random selection
            selected_indices = np.random.choice(indices, size=min(n_per_class, len(indices)), replace=False)

        else:
            # Random selection
            selected_indices = np.random.choice(indices, size=min(n_per_class, len(indices)), replace=False)

        # Create sample dicts
        for idx in selected_indices:
            samples.append({
                'image_path': str(image_paths[idx]),
                'landmarks': predictions[idx],
                'category': category,
                'index': int(idx)
            })

    return samples


def create_comparison_grid(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    n_cols: int = 4,
    figsize: Tuple[int, int] = (16, 12),
    suptitle: Optional[str] = None
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a grid of images for comparison.

    Args:
        images: List of images (H, W) or (H, W, C)
        titles: Optional titles for each image
        n_cols: Number of columns in grid
        figsize: Figure size
        suptitle: Super title for the entire figure

    Returns:
        Tuple of (figure, axes_array)
    """
    n_images = len(images)
    n_rows = int(np.ceil(n_images / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, img in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        if img.ndim == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)

        ax.axis('off')

        if titles and idx < len(titles):
            ax.set_title(titles[idx], fontsize=10)

    # Hide unused subplots
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, weight='bold', y=0.98)

    plt.tight_layout()
    return fig, axes


def add_panel_labels(
    axes: np.ndarray,
    labels: Optional[List[str]] = None,
    start_char: str = 'a',
    fontsize: int = 12,
    loc: str = 'upper left'
) -> None:
    """
    Add (a), (b), (c)... labels to subplots.

    Args:
        axes: Array of matplotlib axes
        labels: Custom labels (if None, use alphabetic)
        start_char: Starting character for alphabetic labels
        fontsize: Font size for labels
        loc: Location ('upper left', 'upper right', etc.)
    """
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, ax in enumerate(axes_flat):
        if labels:
            label = labels[idx] if idx < len(labels) else ''
        else:
            label = f'({chr(ord(start_char) + idx)})'

        if label:
            if loc == 'upper left':
                x, y = 0.05, 0.95
                ha, va = 'left', 'top'
            elif loc == 'upper right':
                x, y = 0.95, 0.95
                ha, va = 'right', 'top'
            elif loc == 'lower left':
                x, y = 0.05, 0.05
                ha, va = 'left', 'bottom'
            else:  # lower right
                x, y = 0.95, 0.05
                ha, va = 'right', 'bottom'

            ax.text(x, y, label, transform=ax.transAxes,
                   fontsize=fontsize, weight='bold',
                   ha=ha, va=va,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))


def load_image(image_path: Union[str, Path], target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load an image from disk.

    Args:
        image_path: Path to image
        target_size: Optional (H, W) to resize to

    Returns:
        Image as numpy array (H, W) or (H, W, C)
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    if target_size:
        img = cv2.resize(img, (target_size[1], target_size[0]))

    return img


def draw_landmarks_on_image(
    image: np.ndarray,
    landmarks: np.ndarray,
    colors: Optional[List[str]] = None,
    radius: int = 3,
    show_connections: bool = False
) -> np.ndarray:
    """
    Draw landmarks on an image.

    Args:
        image: Input image (H, W) grayscale or (H, W, 3) RGB
        landmarks: Array of shape (15, 2) with (x, y) coordinates
        colors: List of colors for each landmark (if None, use default)
        radius: Radius of landmark circles
        show_connections: Whether to draw lines connecting landmarks

    Returns:
        Image with landmarks drawn
    """
    # Convert to RGB if grayscale
    if image.ndim == 2:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = image.copy()

    # Default colors if not provided
    if colors is None:
        from src_v2.constants import CENTRAL_LANDMARKS
        colors = []
        for i in range(15):
            if i in CENTRAL_LANDMARKS:
                colors.append((255, 0, 0))  # Red for central
            elif i < 7:
                colors.append((0, 0, 255))  # Blue for left
            else:
                colors.append((0, 255, 0))  # Green for right

    # Draw connections if requested
    if show_connections:
        # Central axis: L1 -> L9 -> L10 -> L11 -> L2
        central_indices = [0, 8, 9, 10, 1]
        for i in range(len(central_indices) - 1):
            pt1 = tuple(landmarks[central_indices[i]].astype(int))
            pt2 = tuple(landmarks[central_indices[i+1]].astype(int))
            cv2.line(img_rgb, pt1, pt2, (128, 128, 128), 1)

    # Draw landmarks
    for i, (x, y) in enumerate(landmarks):
        color = colors[i] if isinstance(colors[i], tuple) else (255, 255, 255)
        cv2.circle(img_rgb, (int(x), int(y)), radius, color, -1)
        # Draw border for better visibility
        cv2.circle(img_rgb, (int(x), int(y)), radius, (0, 0, 0), 1)

    return img_rgb


def create_arrow(
    ax: plt.Axes,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label: str = '',
    color: str = 'black',
    arrowstyle: str = '->',
    lw: float = 2.0
) -> None:
    """
    Create an arrow between two points with optional label.

    Args:
        ax: Matplotlib axis
        x1, y1: Start coordinates
        x2, y2: End coordinates
        label: Optional label text
        color: Arrow color
        arrowstyle: Arrow style
        lw: Line width
    """
    arrow = patches.FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=arrowstyle,
        mutation_scale=20,
        linewidth=lw,
        color=color
    )
    ax.add_patch(arrow)

    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, label, ha='center', va='center',
               fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))


def normalize_image_for_display(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 255] uint8 for display.

    Args:
        image: Input image (any range)

    Returns:
        Normalized image as uint8
    """
    img = image.copy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)
    return img


def create_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Create an overlay of heatmap on image.

    Args:
        image: Base image (H, W) grayscale
        heatmap: Heatmap (H, W) with values in [0, 1]
        alpha: Transparency of overlay
        colormap: Matplotlib colormap name

    Returns:
        RGB image with heatmap overlay
    """
    # Normalize image
    if image.ndim == 2:
        img_rgb = cv2.cvtColor(normalize_image_for_display(image), cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = normalize_image_for_display(image)

    # Create heatmap
    heatmap_normalized = normalize_image_for_display(heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

    # Overlay
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay


def save_figure(
    fig: plt.Figure,
    output_path: Union[str, Path],
    dpi: int = 300,
    bbox_inches: str = 'tight'
) -> None:
    """
    Save figure to disk with publication quality settings.

    Args:
        fig: Matplotlib figure
        output_path: Output file path
        dpi: DPI for raster formats
        bbox_inches: Bbox setting for tight layout
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Saved figure: {output_path}")


def get_landmark_groups() -> Dict[str, List[int]]:
    """
    Get landmark groups for visualization.

    Returns:
        Dict with keys: 'central', 'left', 'right'
    """
    from src_v2.constants import CENTRAL_LANDMARKS, SYMMETRIC_PAIRS

    # Central landmarks
    central = list(CENTRAL_LANDMARKS)

    # Left and right from symmetric pairs
    left = [pair[0] for pair in SYMMETRIC_PAIRS]
    right = [pair[1] for pair in SYMMETRIC_PAIRS]

    return {
        'central': central,
        'left': left,
        'right': right
    }


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: int = 4
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        image: Input grayscale image
        clip_limit: Clipping limit for contrast
        tile_size: Grid size for adaptive equalization

    Returns:
        CLAHE-enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(image)


def load_checkpoint(checkpoint_path: Union[str, Path]) -> Dict:
    """
    Load a PyTorch checkpoint.

    Args:
        checkpoint_path: Path to checkpoint .pt file

    Returns:
        Checkpoint dictionary
    """
    return torch.load(checkpoint_path, map_location='cpu')


def denormalize_landmarks(
    landmarks: np.ndarray,
    image_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Denormalize landmarks from [0, 1] to image coordinates.

    Args:
        landmarks: Array of shape (15, 2) with normalized coordinates
        image_size: Target image size (H, W)

    Returns:
        Denormalized landmarks in pixel coordinates
    """
    landmarks_denorm = landmarks.copy()
    landmarks_denorm[:, 0] *= image_size[1]  # x
    landmarks_denorm[:, 1] *= image_size[0]  # y
    return landmarks_denorm
