"""
Error analysis module for COVID-19 classification errors.

This module provides functions to categorize misclassified samples by confidence x fold
agreement, trace pipeline failure origins, and enrich error samples with metadata.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src_v2.constants import CATEGORIES
from src_v2.processing.warp import compute_fill_rate


logger = logging.getLogger(__name__)


def load_error_samples(
    predictions_csv: Path,
    dataset_dir: Path,
    warped_dir: Path,
    predictions_npz: Path,
    fold_dirs: Optional[List[Path]] = None
) -> pd.DataFrame:
    """
    Load misclassified samples with enriched metadata.

    Args:
        predictions_csv: Path to ensemble predictions CSV
        dataset_dir: Path to original dataset directory
        warped_dir: Path to warped dataset directory
        predictions_npz: Path to landmark predictions NPZ file
        fold_dirs: Optional list of fold directories for per-fold predictions

    Returns:
        DataFrame with columns: sample_idx, true_class, predicted_class, confidence,
        margin, fold_agreement, landmark_coords, original_path, warped_path, probs
    """
    logger.info(f"Loading ensemble predictions from {predictions_csv}")

    # Load ensemble predictions
    df = pd.read_csv(predictions_csv)
    logger.info(f"Total test samples: {len(df)}")

    # Filter to misclassified samples (soft_correct == 0)
    errors_df = df[df['soft_correct'] == 0].copy()
    logger.info(f"Misclassified samples: {len(errors_df)}")

    # Load test images CSV to map sample indices to filenames
    test_images_csv = warped_dir / 'images.csv'
    logger.info(f"Loading test images mapping from {test_images_csv}")
    test_images = pd.read_csv(test_images_csv)

    # Load landmark predictions
    logger.info(f"Loading landmark predictions from {predictions_npz}")
    landmarks_data = np.load(predictions_npz, allow_pickle=True)
    landmarks = landmarks_data['landmarks']  # Shape: (N, 15, 2)
    landmark_image_paths = landmarks_data['image_paths']
    landmark_image_names = landmarks_data['image_names']  # Basenames without extension
    landmark_categories = landmarks_data['categories']

    # Create lookup dict for landmarks: (image_name, category) -> landmark_coords
    landmark_lookup = {}
    for i, (img_name, cat, lm) in enumerate(zip(landmark_image_names,
                                                  landmark_categories, landmarks)):
        # img_name format: "COVID-123.png" -> key: "COVID-123"
        key = Path(img_name).stem if isinstance(img_name, str) else str(img_name)
        landmark_lookup[(key, cat)] = lm

    # Load ensemble test results for per-sample probabilities
    ensemble_results_path = Path(predictions_csv).parent / 'ensemble_test_results_tta.json'
    ensemble_probs = None
    if ensemble_results_path.exists():
        logger.info(f"Loading ensemble results from {ensemble_results_path}")
        with open(ensemble_results_path, 'r') as f:
            ensemble_results = json.load(f)
            # Try to extract per-sample probabilities if available
            # The JSON structure may not have per-sample data, only aggregates

    # Load per-fold predictions for fold agreement calculation
    # For now, we'll compute a placeholder fold agreement
    # True fold agreement would require per-sample, per-fold predictions

    # Enrich error samples
    enriched_rows = []
    for idx, row in errors_df.iterrows():
        sample_idx = row['sample_idx']
        true_label = row['true_label']
        pred_label = row['soft_prediction']

        # Map numeric labels to class names
        true_class = CATEGORIES[true_label]
        pred_class = CATEGORIES[pred_label]

        # Get image info from test_images CSV
        if sample_idx < len(test_images):
            image_row = test_images.iloc[sample_idx]
            image_name = image_row['image_name']
            warped_filename = image_row['warped_filename']

            # Construct paths
            # Original: {dataset_dir}/{class}/{image_name}.png
            original_path = dataset_dir / true_class.replace('_', ' ') / f"{image_name}.png"
            # Warped: {warped_dir}/{class}/{warped_filename}
            warped_path = warped_dir / true_class / warped_filename

            # Look up landmarks
            landmark_coords = landmark_lookup.get((image_name, true_class))
        else:
            image_name = None
            original_path = None
            warped_path = None
            landmark_coords = None

        # Compute confidence and margin (placeholder for now)
        # True confidence would come from ensemble probabilities
        confidence = 0.5  # Placeholder
        margin = 0.1      # Placeholder
        fold_agreement = 0.8  # Placeholder (5 folds, assume 4/5 agree = 0.8)
        probs = np.array([0.33, 0.33, 0.34])  # Placeholder

        enriched_row = {
            'sample_idx': sample_idx,
            'image_name': image_name,
            'true_label': true_label,
            'predicted_label': pred_label,
            'true_class': true_class,
            'predicted_class': pred_class,
            'confidence': confidence,
            'margin': margin,
            'fold_agreement': fold_agreement,
            'landmark_coords': landmark_coords,
            'original_path': str(original_path) if original_path else None,
            'warped_path': str(warped_path) if warped_path else None,
            'probs': probs
        }

        enriched_rows.append(enriched_row)

    result_df = pd.DataFrame(enriched_rows)

    logger.info(f"Enriched {len(result_df)} error samples")
    return result_df


def categorize_errors(error_df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize errors using confidence x fold agreement matrix.

    Categories:
    - UNANIMOUS_HIGH_CONF: all folds agree wrong + confidence > 0.9 -> likely label noise
    - UNANIMOUS_LOW_CONF: all folds agree wrong + confidence < 0.6 -> hard example
    - HIGH_CONF_ERROR: majority agree wrong + confidence > 0.9 -> systematic bias
    - SPLIT_DECISION: folds disagree (agreement < 0.6) -> near boundary
    - MODERATE: everything else -> model limitation

    Args:
        error_df: DataFrame with error samples (must have confidence and fold_agreement)

    Returns:
        DataFrame with added columns: category, interpretation, likely_cause, recoverability
    """
    result_df = error_df.copy()

    categories = []
    interpretations = []
    likely_causes = []
    recoverabilities = []

    for _, row in result_df.iterrows():
        conf = row['confidence']
        agreement = row['fold_agreement']

        if agreement >= 0.9 and conf > 0.9:
            category = 'UNANIMOUS_HIGH_CONF'
            interpretation = 'All folds confidently wrong'
            likely_cause = 'Possible label noise'
            recoverability = 'fixable'
        elif agreement >= 0.9 and conf < 0.6:
            category = 'UNANIMOUS_LOW_CONF'
            interpretation = 'All folds agree but uncertain'
            likely_cause = 'Inherently difficult example'
            recoverability = 'partially_fixable'
        elif agreement >= 0.6 and conf > 0.9:
            category = 'HIGH_CONF_ERROR'
            interpretation = 'Majority folds confidently wrong'
            likely_cause = 'Systematic feature bias'
            recoverability = 'fixable'
        elif agreement < 0.6:
            category = 'SPLIT_DECISION'
            interpretation = 'Folds disagree on prediction'
            likely_cause = 'Near decision boundary'
            recoverability = 'partially_fixable'
        else:
            category = 'MODERATE'
            interpretation = 'Moderate confidence error'
            likely_cause = 'Model limitation'
            recoverability = 'inherent'

        categories.append(category)
        interpretations.append(interpretation)
        likely_causes.append(likely_cause)
        recoverabilities.append(recoverability)

    result_df['category'] = categories
    result_df['interpretation'] = interpretations
    result_df['likely_cause'] = likely_causes
    result_df['recoverability'] = recoverabilities

    logger.info(f"Categorized {len(result_df)} errors")
    logger.info("Category distribution:")
    for cat, count in result_df['category'].value_counts().items():
        logger.info(f"  {cat}: {count}")

    return result_df


def trace_pipeline_failures(
    error_df: pd.DataFrame,
    canonical_shape_path: Path,
    dataset_mean_error: float = 3.61
) -> pd.DataFrame:
    """
    Trace pipeline failure origins for each error.

    Classifies failures into:
    - bad_landmarks: landmark error > 2x dataset mean (>7.22 px)
    - bad_warp: fill_rate < 0.85 or landmark error > 1.5x mean
    - ambiguous_image: good landmarks + good warp but low confidence
    - suspect_label_noise: good landmarks + good warp + high confidence wrong

    Args:
        error_df: DataFrame with error samples (must have landmark_coords, warped_path)
        canonical_shape_path: Path to canonical shape NPY file
        dataset_mean_error: Mean landmark error for the dataset (default 3.61 px)

    Returns:
        DataFrame with added columns: landmark_error_px, warp_fill_rate, failure_origin
    """
    result_df = error_df.copy()

    # Load canonical shape
    logger.info(f"Loading canonical shape from {canonical_shape_path}")
    canonical_shape = np.load(canonical_shape_path)

    landmark_errors = []
    fill_rates = []
    failure_origins = []

    for _, row in result_df.iterrows():
        # Compute landmark error (for now use placeholder)
        # In real implementation, would compute distance from predicted to GT
        landmark_error = dataset_mean_error  # Placeholder

        # Compute warp fill rate
        warped_path = row['warped_path']
        if warped_path and Path(warped_path).exists():
            import cv2
            warped_img = cv2.imread(str(warped_path), cv2.IMREAD_GRAYSCALE)
            fill_rate = compute_fill_rate(warped_img)
        else:
            fill_rate = 0.0

        # Classify failure origin
        confidence = row['confidence']

        if landmark_error > 2 * dataset_mean_error:
            failure_origin = 'bad_landmarks'
        elif fill_rate < 0.85 or landmark_error > 1.5 * dataset_mean_error:
            failure_origin = 'bad_warp'
        elif confidence < 0.7:
            failure_origin = 'ambiguous_image'
        else:
            failure_origin = 'suspect_label_noise'

        landmark_errors.append(landmark_error)
        fill_rates.append(fill_rate)
        failure_origins.append(failure_origin)

    result_df['landmark_error_px'] = landmark_errors
    result_df['warp_fill_rate'] = fill_rates
    result_df['failure_origin'] = failure_origins

    logger.info(f"Traced pipeline failures for {len(result_df)} errors")
    logger.info("Failure origin distribution:")
    for origin, count in result_df['failure_origin'].value_counts().items():
        logger.info(f"  {origin}: {count}")

    return result_df
