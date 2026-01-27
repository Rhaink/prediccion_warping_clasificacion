"""
Utilities for creating architectural diagrams in strip-1-.png style.

This module provides functions to create clean, professional diagrams showing:
- Neural network architectures
- Feature maps with grid representations
- Flow diagrams with boxes and arrows
- Layered structures with annotations
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


class ArchitectureDiagramBuilder:
    """Builder class for creating neural network architecture diagrams."""

    def __init__(self, figsize: Tuple[int, int] = (16, 10)):
        """
        Initialize diagram builder.

        Args:
            figsize: Figure size (width, height)
        """
        self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.axis('off')
        self.ax.set_aspect('equal')

        self.current_x = 0.5
        self.layer_spacing = 1.2

    def add_layer_box(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        text: str,
        color: str = 'lightblue',
        feature_map: Optional[np.ndarray] = None,
        show_grid: bool = True,
        n_grid_cells: int = 8
    ) -> FancyBboxPatch:
        """
        Add a layer box representing a neural network layer.

        Args:
            x: X coordinate (left edge)
            y: Y coordinate (bottom edge)
            width: Box width
            height: Box height
            text: Text label for the layer
            color: Fill color
            feature_map: Optional feature map to display as grid
            show_grid: Whether to show grid pattern
            n_grid_cells: Number of grid cells to show

        Returns:
            The created box patch
        """
        # Create box with rounded corners
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.05",
            linewidth=2,
            edgecolor='black',
            facecolor=color,
            alpha=0.7,
            zorder=1
        )
        self.ax.add_patch(box)

        # Add grid pattern if requested
        if show_grid:
            self._add_grid_pattern(x, y, width, height, n_grid_cells, feature_map)

        # Add text label below box
        self.ax.text(
            x + width / 2, y - 0.15, text,
            ha='center', va='top',
            fontsize=9, weight='bold'
        )

        return box

    def _add_grid_pattern(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        n_cells: int,
        feature_map: Optional[np.ndarray] = None
    ) -> None:
        """Add a grid pattern to represent feature maps."""
        cell_width = width / n_cells
        cell_height = height / n_cells

        for i in range(n_cells):
            for j in range(n_cells):
                cell_x = x + i * cell_width
                cell_y = y + j * cell_height

                # Determine cell color
                if feature_map is not None:
                    # Downsample feature map to grid
                    row = int((j / n_cells) * feature_map.shape[0])
                    col = int((i / n_cells) * feature_map.shape[1])
                    value = feature_map[row, col]
                    # Normalize to grayscale
                    intensity = (value - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
                    cell_color = (intensity, intensity, intensity)
                else:
                    cell_color = 'white'

                # Draw cell
                cell = patches.Rectangle(
                    (cell_x, cell_y), cell_width, cell_height,
                    linewidth=0.5,
                    edgecolor='gray',
                    facecolor=cell_color,
                    alpha=0.8,
                    zorder=2
                )
                self.ax.add_patch(cell)

    def add_arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        label: str = '',
        color: str = 'black',
        arrowstyle: str = '->',
        lw: float = 2.0
    ) -> FancyArrowPatch:
        """
        Add an arrow between two points.

        Args:
            x1, y1: Start coordinates
            x2, y2: End coordinates
            label: Optional label text
            color: Arrow color
            arrowstyle: Arrow style
            lw: Line width

        Returns:
            The created arrow patch
        """
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=arrowstyle,
            mutation_scale=20,
            linewidth=lw,
            color=color,
            zorder=3
        )
        self.ax.add_patch(arrow)

        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            self.ax.text(
                mid_x, mid_y + 0.2, label,
                ha='center', va='center',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9)
            )

        return arrow

    def add_text_annotation(
        self,
        x: float,
        y: float,
        text: str,
        fontsize: int = 10,
        color: str = 'black',
        weight: str = 'normal',
        ha: str = 'center',
        va: str = 'center'
    ) -> None:
        """Add text annotation at specified position."""
        self.ax.text(
            x, y, text,
            ha=ha, va=va,
            fontsize=fontsize,
            color=color,
            weight=weight
        )

    def add_dimension_label(
        self,
        x: float,
        y: float,
        channels: int,
        height: int,
        width: int,
        fontsize: int = 8
    ) -> None:
        """
        Add dimension label (C×H×W format).

        Args:
            x, y: Position for label
            channels: Number of channels
            height: Height dimension
            width: Width dimension
            fontsize: Font size
        """
        text = f'{channels}×{height}×{width}'
        self.ax.text(
            x, y, text,
            ha='center', va='center',
            fontsize=fontsize,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8)
        )

    def save(self, output_path: str, dpi: int = 300) -> None:
        """Save the diagram to file."""
        from pathlib import Path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        self.fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved diagram: {output_path}")
        plt.close(self.fig)


def create_resnet_diagram(
    output_path: str,
    show_feature_maps: bool = True,
    feature_maps: Optional[Dict[str, np.ndarray]] = None
) -> None:
    """
    Create a diagram of ResNet-18 architecture for landmark detection.

    Args:
        output_path: Path to save the diagram
        show_feature_maps: Whether to show actual feature maps
        feature_maps: Optional dict of feature maps for each layer
    """
    builder = ArchitectureDiagramBuilder(figsize=(18, 8))

    # Define layer configurations
    layers = [
        {'name': 'Input\n3×224×224', 'width': 0.8, 'height': 2.0, 'color': 'lightgray'},
        {'name': 'Layer1\n64×56×56', 'width': 0.9, 'height': 1.8, 'color': 'lightblue'},
        {'name': 'Layer2\n128×28×28', 'width': 0.9, 'height': 1.5, 'color': 'lightgreen'},
        {'name': 'Layer3\n256×14×14', 'width': 0.9, 'height': 1.2, 'color': 'lightyellow'},
        {'name': 'Layer4\n512×7×7', 'width': 0.9, 'height': 0.9, 'color': 'lightcoral'},
        {'name': 'GAP\n512', 'width': 0.6, 'height': 0.6, 'color': 'plum'},
        {'name': 'FC\n512→256', 'width': 0.6, 'height': 0.6, 'color': 'wheat'},
        {'name': 'Output\n30', 'width': 0.6, 'height': 0.6, 'color': 'lightpink'},
    ]

    # Position layers
    x_positions = [0.5, 2.0, 3.5, 5.0, 6.5, 7.8, 8.8, 9.8]
    y_base = 4.0

    boxes = []
    for i, (layer, x) in enumerate(zip(layers, x_positions)):
        y = y_base - layer['height'] / 2

        # Get feature map if available
        fmap = None
        if show_feature_maps and feature_maps and i < 5:
            layer_key = f'layer{i}' if i > 0 else 'input'
            if layer_key in feature_maps:
                fmap = feature_maps[layer_key]

        box = builder.add_layer_box(
            x, y, layer['width'], layer['height'],
            layer['name'],
            color=layer['color'],
            feature_map=fmap,
            show_grid=(i < 5)  # Only show grid for conv layers
        )
        boxes.append((x, y, layer['width'], layer['height']))

    # Add arrows between layers
    for i in range(len(boxes) - 1):
        x1, y1, w1, h1 = boxes[i]
        x2, y2, w2, h2 = boxes[i + 1]

        # Arrow from right of box i to left of box i+1
        builder.add_arrow(
            x1 + w1, y1 + h1 / 2,
            x2, y2 + h2 / 2,
            label=''
        )

    # Add title
    builder.ax.text(
        5.0, 7.5, 'ResNet-18 Architecture for Landmark Detection',
        ha='center', va='center',
        fontsize=14, weight='bold'
    )

    # Add annotations
    builder.add_text_annotation(
        1.25, 6.5, 'Backbone (Feature Extraction)',
        fontsize=10, weight='bold'
    )
    builder.add_text_annotation(
        8.3, 6.5, 'Regression Head',
        fontsize=10, weight='bold'
    )

    builder.save(output_path)


def create_operation_diagram(
    operation_type: str,
    output_path: str,
    example_input: Optional[np.ndarray] = None
) -> None:
    """
    Create a diagram explaining a specific operation (Conv, Pooling, ReLU).

    Args:
        operation_type: Type of operation ('conv', 'pooling', 'relu')
        output_path: Path to save the diagram
        example_input: Optional input array for demonstration
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    if operation_type == 'conv':
        _create_conv_diagram(axes, example_input)
    elif operation_type == 'pooling':
        _create_pooling_diagram(axes, example_input)
    elif operation_type == 'relu':
        _create_relu_diagram(axes, example_input)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved operation diagram: {output_path}")
    plt.close(fig)


def _create_conv_diagram(axes: np.ndarray, example_input: Optional[np.ndarray]) -> None:
    """Create convolution operation diagram."""
    ax1, ax2, ax3 = axes

    # Panel 1: Input with kernel overlay
    ax1.set_title('(a) Input + Kernel (3×3)', fontsize=12, weight='bold')
    input_data = np.random.rand(8, 8) if example_input is None else example_input[:8, :8]
    ax1.imshow(input_data, cmap='gray', vmin=0, vmax=1)

    # Highlight 3×3 region
    rect = patches.Rectangle((0.5, 0.5), 3, 3, linewidth=3, edgecolor='red', facecolor='none')
    ax1.add_patch(rect)
    ax1.axis('off')
    ax1.text(0.5, -0.5, 'Kernel moves across input', ha='center', va='top', fontsize=9)

    # Panel 2: Kernel weights
    ax2.set_title('(b) Kernel Weights', fontsize=12, weight='bold')
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Sobel-like
    im = ax2.imshow(kernel, cmap='RdBu', vmin=-2, vmax=2)
    ax2.axis('off')
    # Add values as text
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f'{kernel[i, j]:.0f}', ha='center', va='center', fontsize=10, weight='bold')
    ax2.text(1, 3.5, 'Detects vertical edges', ha='center', va='top', fontsize=9)

    # Panel 3: Output feature map
    ax3.set_title('(c) Output Feature Map', fontsize=12, weight='bold')
    # Simulate convolution output
    from scipy.ndimage import convolve
    output = convolve(input_data, kernel, mode='constant')
    ax3.imshow(output, cmap='RdBu')
    ax3.axis('off')
    ax3.text(4, 8.5, 'Edges highlighted', ha='center', va='top', fontsize=9)


def _create_pooling_diagram(axes: np.ndarray, example_input: Optional[np.ndarray]) -> None:
    """Create max pooling operation diagram."""
    ax1, ax2, ax3 = axes

    # Panel 1: Input
    ax1.set_title('(a) Input (8×8)', fontsize=12, weight='bold')
    input_data = np.random.randint(0, 10, (8, 8)) if example_input is None else example_input[:8, :8]
    ax1.imshow(input_data, cmap='viridis', vmin=0, vmax=10)
    # Add grid
    for i in range(9):
        ax1.axhline(i - 0.5, color='white', linewidth=1)
        ax1.axvline(i - 0.5, color='white', linewidth=1)
    ax1.axis('off')
    ax1.text(4, -0.5, 'Original resolution', ha='center', va='top', fontsize=9)

    # Panel 2: Pooling regions
    ax2.set_title('(b) Max Pooling (2×2)', fontsize=12, weight='bold')
    ax2.imshow(input_data, cmap='viridis', vmin=0, vmax=10, alpha=0.3)
    # Highlight 2×2 blocks
    for i in range(0, 8, 2):
        for j in range(0, 8, 2):
            rect = patches.Rectangle((j - 0.5, i - 0.5), 2, 2, linewidth=2, edgecolor='red', facecolor='none')
            ax2.add_patch(rect)
    ax2.axis('off')
    ax2.text(4, -0.5, 'Take max from each 2×2 block', ha='center', va='top', fontsize=9)

    # Panel 3: Output
    ax3.set_title('(c) Output (4×4)', fontsize=12, weight='bold')
    # Apply max pooling
    output = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            output[i, j] = input_data[i*2:(i+1)*2, j*2:(j+1)*2].max()
    ax3.imshow(output, cmap='viridis', vmin=0, vmax=10)
    for i in range(5):
        ax3.axhline(i - 0.5, color='white', linewidth=1)
        ax3.axvline(i - 0.5, color='white', linewidth=1)
    ax3.axis('off')
    ax3.text(2, 4.5, 'Reduced resolution, kept important features', ha='center', va='top', fontsize=9)


def _create_relu_diagram(axes: np.ndarray, example_input: Optional[np.ndarray]) -> None:
    """Create ReLU activation diagram."""
    ax1, ax2, ax3 = axes

    # Panel 1: Input (with negative values)
    ax1.set_title('(a) Before ReLU', fontsize=12, weight='bold')
    input_data = np.random.randn(8, 8) if example_input is None else example_input[:8, :8]
    ax1.imshow(input_data, cmap='RdBu', vmin=-3, vmax=3)
    ax1.axis('off')
    ax1.text(4, -0.5, 'Contains negative values', ha='center', va='top', fontsize=9)

    # Panel 2: ReLU function plot
    ax2.set_title('(b) ReLU Function', fontsize=12, weight='bold')
    x = np.linspace(-3, 3, 100)
    y = np.maximum(0, x)
    ax2.plot(x, y, linewidth=3, color='blue')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)
    ax2.set_xlabel('Input', fontsize=10)
    ax2.set_ylabel('Output', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.text(0, -0.5, 'f(x) = max(0, x)', ha='center', va='top', fontsize=9)

    # Panel 3: Output (negatives zeroed)
    ax3.set_title('(c) After ReLU', fontsize=12, weight='bold')
    output = np.maximum(0, input_data)
    ax3.imshow(output, cmap='RdBu', vmin=-3, vmax=3)
    ax3.axis('off')
    ax3.text(4, -0.5, 'Negative values → 0 (non-linear)', ha='center', va='top', fontsize=9)


def create_flow_diagram(
    steps: List[Dict[str, str]],
    output_path: str,
    figsize: Tuple[int, int] = (14, 6),
    orientation: str = 'horizontal'
) -> None:
    """
    Create a simple flow diagram with boxes and arrows.

    Args:
        steps: List of dicts with keys 'label', 'description', 'color'
        output_path: Path to save the diagram
        figsize: Figure size
        orientation: 'horizontal' or 'vertical'
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    n_steps = len(steps)

    if orientation == 'horizontal':
        x_positions = np.linspace(1, 9, n_steps)
        y_position = 5
        box_width = 1.2
        box_height = 1.5
    else:  # vertical
        x_position = 5
        y_positions = np.linspace(8, 2, n_steps)
        box_width = 1.5
        box_height = 1.2

    for i, step in enumerate(steps):
        if orientation == 'horizontal':
            x, y = x_positions[i], y_position
        else:
            x, y = x_position, y_positions[i]

        # Draw box
        color = step.get('color', 'lightblue')
        box = FancyBboxPatch(
            (x - box_width / 2, y - box_height / 2),
            box_width, box_height,
            boxstyle="round,pad=0.1",
            linewidth=2,
            edgecolor='black',
            facecolor=color,
            alpha=0.7
        )
        ax.add_patch(box)

        # Add label
        ax.text(x, y, step['label'], ha='center', va='center',
               fontsize=10, weight='bold', wrap=True)

        # Add description below
        if 'description' in step:
            desc_y = y - box_height / 2 - 0.3
            ax.text(x, desc_y, step['description'], ha='center', va='top',
                   fontsize=8, style='italic')

        # Add arrow to next step
        if i < n_steps - 1:
            if orientation == 'horizontal':
                arrow_start = (x + box_width / 2, y)
                arrow_end = (x_positions[i + 1] - box_width / 2, y)
            else:
                arrow_start = (x, y - box_height / 2)
                arrow_end = (x, y_positions[i + 1] + box_height / 2)

            arrow = FancyArrowPatch(
                arrow_start, arrow_end,
                arrowstyle='->',
                mutation_scale=20,
                linewidth=2,
                color='black'
            )
            ax.add_patch(arrow)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved flow diagram: {output_path}")
    plt.close(fig)
