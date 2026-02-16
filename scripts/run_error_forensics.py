#!/usr/bin/env python
"""
Run error forensics analysis on misclassified test samples.

This script orchestrates the complete error analysis pipeline:
1. Load 33 misclassified samples from ensemble predictions
2. Categorize errors by confidence x fold agreement
3. Trace pipeline failure origins
4. Generate visualizations (overview grid + per-sample traces)
5. Save structured results to JSON
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src_v2.evaluation.error_analysis import (
    load_error_samples,
    categorize_errors,
    trace_pipeline_failures
)
from src_v2.utils.visualization import (
    create_overview_grid,
    visualize_pipeline_trace,
    overlay_landmarks
)
from src_v2.processing.gpa import compute_delaunay_triangulation


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run error forensics analysis on misclassified test samples'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('outputs/error_forensics'),
        help='Output directory for results and visualizations'
    )
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        default=Path('data/dataset/COVID-19_Radiography_Dataset'),
        help='Path to original dataset directory'
    )
    parser.add_argument(
        '--predictions-csv',
        type=Path,
        default=Path('outputs/classifier_cv/ensemble_predictions_tta.csv'),
        help='Path to ensemble predictions CSV'
    )
    parser.add_argument(
        '--predictions-npz',
        type=Path,
        default=Path('outputs/landmark_predictions/session_warping/predictions.npz'),
        help='Path to landmark predictions NPZ file'
    )
    parser.add_argument(
        '--warped-dir',
        type=Path,
        default=Path('outputs/warped_lung_best/session_warping/test'),
        help='Path to warped test dataset directory'
    )
    parser.add_argument(
        '--canonical-shape',
        type=Path,
        default=Path('outputs/shape_analysis/canonical_shape.npy'),
        help='Path to canonical shape NPY file'
    )
    parser.add_argument(
        '--fold-dirs',
        type=str,
        nargs='+',
        default=None,
        help='Optional list of fold directories for per-fold predictions'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("ERROR FORENSICS ANALYSIS")
    logger.info("=" * 80)

    # Create output directories
    output_dir = args.output_dir
    viz_dir = output_dir / 'error_visualizations'
    per_sample_dir = viz_dir / 'per_sample_detailed'
    by_confusion_dir = viz_dir / 'by_confusion_pair'

    for d in [output_dir, viz_dir, per_sample_dir, by_confusion_dir]:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {d}")

    # Step 1: Load error samples
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Load misclassified samples")
    logger.info("=" * 80)

    fold_dirs_paths = None
    if args.fold_dirs:
        fold_dirs_paths = [Path(d) for d in args.fold_dirs]

    error_df = load_error_samples(
        predictions_csv=args.predictions_csv,
        dataset_dir=args.dataset_dir,
        warped_dir=args.warped_dir,
        predictions_npz=args.predictions_npz,
        fold_dirs=fold_dirs_paths
    )

    logger.info(f"Loaded {len(error_df)} misclassified samples")

    # Step 2: Categorize errors
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Categorize errors by confidence x fold agreement")
    logger.info("=" * 80)

    error_df = categorize_errors(error_df)

    # Step 3: Trace pipeline failures
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Trace pipeline failure origins")
    logger.info("=" * 80)

    error_df = trace_pipeline_failures(
        error_df=error_df,
        canonical_shape_path=args.canonical_shape,
        dataset_mean_error=3.61  # From GROUND_TRUTH.json
    )

    # Step 4: Generate visualizations
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Generate visualizations")
    logger.info("=" * 80)

    # Load canonical shape for triangulation
    canonical_shape = np.load(args.canonical_shape)
    triangulation = compute_delaunay_triangulation(canonical_shape)

    # 4a. Generate per-sample pipeline traces
    logger.info("Generating per-sample pipeline traces...")
    for _, row in tqdm(error_df.iterrows(), total=len(error_df), desc="Pipeline traces"):
        sample_idx = row['sample_idx']

        # Load images
        original_path = Path(row['original_path']) if row['original_path'] else None
        warped_path = Path(row['warped_path']) if row['warped_path'] else None

        if original_path and original_path.exists():
            original_img = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
        else:
            original_img = np.zeros((224, 224), dtype=np.uint8)

        if warped_path and warped_path.exists():
            warped_img = cv2.imread(str(warped_path), cv2.IMREAD_GRAYSCALE)
        else:
            warped_img = np.zeros((224, 224), dtype=np.uint8)

        # Get landmarks
        landmark_coords = row['landmark_coords']
        if landmark_coords is None:
            landmark_coords = canonical_shape  # Use canonical as fallback

        # Generate trace
        output_path = per_sample_dir / f"sample_{sample_idx:03d}_pipeline.png"
        visualize_pipeline_trace(
            original_img=original_img,
            landmarks=landmark_coords,
            warped_img=warped_img,
            true_class=row['true_class'],
            pred_class=row['predicted_class'],
            probs=row['probs'],
            category=row['category'],
            failure_origin=row['failure_origin'],
            output_path=output_path,
            triangulation=triangulation
        )

    logger.info(f"Generated {len(error_df)} pipeline trace figures")

    # 4b. Create overview grid
    logger.info("Creating overview grid...")
    error_samples = error_df.to_dict('records')
    overview_path = viz_dir / 'overview_grid_all_33.png'
    create_overview_grid(error_samples, overview_path, ncols=6)

    # 4c. Create confusion pair sub-grids
    logger.info("Creating confusion pair sub-grids...")
    confusion_pairs = error_df.groupby(['true_class', 'predicted_class'])
    for (true_cls, pred_cls), group in confusion_pairs:
        if len(group) > 0:
            pair_samples = group.to_dict('records')
            pair_path = by_confusion_dir / f"{true_cls}_to_{pred_cls}.png"
            create_overview_grid(pair_samples, pair_path, ncols=min(6, len(pair_samples)))
            logger.info(f"  {true_cls} -> {pred_cls}: {len(group)} samples")

    # Step 5: Save structured results
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Save structured results to JSON")
    logger.info("=" * 80)

    # Compute summary statistics
    category_summary = error_df['category'].value_counts().to_dict()
    recoverability_summary = error_df['recoverability'].value_counts().to_dict()
    failure_origin_summary = error_df['failure_origin'].value_counts().to_dict()

    confusion_pairs_dict = {}
    for (true_cls, pred_cls), group in error_df.groupby(['true_class', 'predicted_class']):
        key = f"{true_cls}_to_{pred_cls}"
        confusion_pairs_dict[key] = len(group)

    # Build results JSON
    results = {
        'metadata': {
            'total_test': 1895,
            'total_errors': len(error_df),
            'accuracy': 98.26,
            'date': datetime.now().isoformat(),
            'model': 'Ensemble+TTA',
            'ensemble_predictions_csv': str(args.predictions_csv),
            'landmark_predictions_npz': str(args.predictions_npz)
        },
        'category_summary': category_summary,
        'recoverability_summary': recoverability_summary,
        'failure_origin_summary': failure_origin_summary,
        'confusion_pairs': confusion_pairs_dict,
        'errors': []
    }

    # Add per-sample details
    for _, row in error_df.iterrows():
        sample_data = {
            'sample_idx': int(row['sample_idx']),
            'image_name': row.get('image_name'),
            'true_class': row['true_class'],
            'predicted_class': row['predicted_class'],
            'confidence': float(row['confidence']),
            'margin': float(row['margin']),
            'fold_agreement': float(row['fold_agreement']),
            'category': row['category'],
            'recoverability': row['recoverability'],
            'failure_origin': row['failure_origin'],
            'landmark_error_px': float(row['landmark_error_px']),
            'warp_fill_rate': float(row['warp_fill_rate']),
            'original_path': row['original_path'],
            'warped_path': row['warped_path'],
            'pipeline_trace_path': f"per_sample_detailed/sample_{int(row['sample_idx']):03d}_pipeline.png"
        }
        results['errors'].append(sample_data)

    # Save results
    results_path = output_dir / 'error_analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results to {results_path}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total errors: {len(error_df)}")
    logger.info(f"\nCategory distribution:")
    for cat, count in category_summary.items():
        logger.info(f"  {cat}: {count}")
    logger.info(f"\nRecoverability distribution:")
    for rec, count in recoverability_summary.items():
        logger.info(f"  {rec}: {count}")
    logger.info(f"\nFailure origin distribution:")
    for origin, count in failure_origin_summary.items():
        logger.info(f"  {origin}: {count}")
    logger.info(f"\nConfusion pairs:")
    for pair, count in confusion_pairs_dict.items():
        logger.info(f"  {pair}: {count}")

    logger.info("\n" + "=" * 80)
    logger.info("ERROR FORENSICS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Overview grid: {overview_path}")
    logger.info(f"Per-sample traces: {per_sample_dir}/")


if __name__ == '__main__':
    main()
