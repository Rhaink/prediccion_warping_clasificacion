"""Visualization module for COVID-19 classification explainability.

This module provides:
- GradCAM: Class Activation Maps for model interpretability
- Error Analysis: Tools for analyzing classification errors
- PFS Analysis: Pulmonary Focus Score analysis tools
"""

from src_v2.visualization.gradcam import (
    GradCAM,
    get_target_layer,
    calculate_pfs,
    overlay_heatmap,
)
from src_v2.visualization.error_analysis import (
    ErrorAnalyzer,
    analyze_classification_errors,
)
from src_v2.visualization.pfs_analysis import (
    PFSAnalyzer,
    PFSResult,
    PFSSummary,
    run_pfs_analysis,
    create_pfs_visualizations,
    load_lung_mask,
    find_mask_for_image,
    generate_approximate_mask,
)

__all__ = [
    "GradCAM",
    "get_target_layer",
    "calculate_pfs",
    "overlay_heatmap",
    "ErrorAnalyzer",
    "analyze_classification_errors",
    "PFSAnalyzer",
    "PFSResult",
    "PFSSummary",
    "run_pfs_analysis",
    "create_pfs_visualizations",
    "load_lung_mask",
    "find_mask_for_image",
    "generate_approximate_mask",
]
