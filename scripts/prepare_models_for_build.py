#!/usr/bin/env python3
"""
Prepare models for PyInstaller build by creating renamed copies.

This script copies models from their development locations to a staging
directory with standardized names for PyInstaller packaging.

Usage:
    python scripts/prepare_models_for_build.py
"""

import shutil
import sys
from pathlib import Path


def main():
    """Copy and rename models to staging directory."""
    project_root = Path(__file__).parent.parent
    staging_dir = project_root / 'build' / 'models_staging'

    print("Preparing models for build...")
    print(f"Staging directory: {staging_dir}")

    # Create staging directories
    (staging_dir / 'landmarks').mkdir(parents=True, exist_ok=True)
    (staging_dir / 'classifier').mkdir(parents=True, exist_ok=True)
    (staging_dir / 'shape_analysis').mkdir(parents=True, exist_ok=True)

    # Model mappings: (source_path, dest_name)
    model_mappings = [
        # Landmark models
        (
            'checkpoints/session10/ensemble/seed123/final_model.pt',
            'landmarks/resnet18_seed123_best.pt'
        ),
        (
            'checkpoints/session13/seed321/final_model.pt',
            'landmarks/resnet18_seed321_best.pt'
        ),
        (
            'checkpoints/repro_split111/session14/seed111/final_model.pt',
            'landmarks/resnet18_seed111_best.pt'
        ),
        (
            'checkpoints/repro_split666/session16/seed666/final_model.pt',
            'landmarks/resnet18_seed666_best.pt'
        ),
        # Classifier
        (
            'outputs/classifier_warped_lung_best/sweeps_2026-01-12/lr2e-4_seed321_on/best_classifier.pt',
            'classifier/best_classifier.pt'
        ),
        # Shape analysis
        (
            'outputs/shape_analysis/canonical_shape_gpa.json',
            'shape_analysis/canonical_shape_gpa.json'
        ),
        (
            'outputs/shape_analysis/canonical_delaunay_triangles.json',
            'shape_analysis/canonical_delaunay_triangles.json'
        ),
    ]

    # Copy files
    total_size = 0
    for source_rel, dest_rel in model_mappings:
        source = project_root / source_rel
        dest = staging_dir / dest_rel

        if not source.exists():
            print(f"✗ Missing: {source}")
            return 1

        print(f"Copying: {source.name} -> {dest.name}")
        shutil.copy2(source, dest)

        size_mb = dest.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  Size: {size_mb:.1f} MB")

    print(f"\n✓ All models prepared ({total_size:.1f} MB total)")
    print(f"✓ Staging directory: {staging_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
