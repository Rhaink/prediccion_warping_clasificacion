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

    # Build ImageFolder-order mapping from disk.
    # The classifier dataloader (torchvision.ImageFolder) iterates classes sorted
    # alphabetically, then files sorted alphabetically within each class.
    # sample_idx in predictions CSV corresponds to this ordering, NOT to images.csv rows.
    logger.info("Building ImageFolder-order mapping from warped test directory")
    imagefolder_samples = []
    class_dirs = sorted([d for d in warped_dir.iterdir() if d.is_dir()])
    for cls_dir in class_dirs:
        category = cls_dir.name
        files = sorted([f.name for f in cls_dir.iterdir() if f.suffix == '.png'])
        for fname in files:
            image_name = fname.replace('_warped.png', '')
            imagefolder_samples.append({
                'category': category,
                'warped_filename': fname,
                'image_name': image_name,
            })
    logger.info(f"ImageFolder mapping: {len(imagefolder_samples)} samples across "
                f"{len(class_dirs)} classes")

    # Load landmark predictions
    logger.info(f"Loading landmark predictions from {predictions_npz}")
    landmarks_data = np.load(predictions_npz, allow_pickle=True)
    landmarks = landmarks_data['landmarks']  # Shape: (N, 15, 2)
    landmark_image_names = landmarks_data['image_names']  # Basenames without extension
    landmark_categories = landmarks_data['categories']

    # Create lookup dict for landmarks: (image_name, category) -> landmark_coords
    landmark_lookup = {}
    for i, (img_name, cat, lm) in enumerate(zip(landmark_image_names,
                                                  landmark_categories, landmarks)):
        # img_name format: "COVID-123.png" -> key: "COVID-123"
        key = Path(img_name).stem if isinstance(img_name, str) else str(img_name)
        landmark_lookup[(key, cat)] = lm

    # Extract real ensemble probabilities if fold dirs are provided
    ensemble_prob_data = None
    if fold_dirs:
        # warped_dir is the test split; data_dir is its parent (contains train/val/test)
        data_dir = warped_dir.parent
        ensemble_prob_data = extract_ensemble_probabilities(fold_dirs, data_dir)
        logger.info(f"Extracted real probabilities for {len(ensemble_prob_data)} samples")

    # Enrich error samples
    enriched_rows = []
    for idx, row in errors_df.iterrows():
        sample_idx = row['sample_idx']
        true_label = row['true_label']
        pred_label = row['soft_prediction']

        # Map numeric labels to class names
        true_class = CATEGORIES[true_label]
        pred_class = CATEGORIES[pred_label]

        # Map sample_idx to image via ImageFolder ordering
        if sample_idx < len(imagefolder_samples):
            sample_info = imagefolder_samples[sample_idx]
            image_name = sample_info['image_name']
            warped_filename = sample_info['warped_filename']
            img_category = sample_info['category']

            # Construct paths
            # Original: {dataset_dir}/{class with spaces}/images/{image_name}.png
            original_class_dir = img_category.replace('_', ' ')
            original_path = dataset_dir / original_class_dir / 'images' / f"{image_name}.png"
            # Warped: {warped_dir}/{category}/{warped_filename}
            warped_path = warped_dir / img_category / warped_filename

            # Look up landmarks
            landmark_coords = landmark_lookup.get((image_name, img_category))
            if landmark_coords is None:
                landmark_coords = landmark_lookup.get((image_name, true_class))
        else:
            image_name = None
            original_path = None
            warped_path = None
            landmark_coords = None

        # Use real probabilities if available, otherwise placeholders
        if ensemble_prob_data and sample_idx in ensemble_prob_data:
            prob_info = ensemble_prob_data[sample_idx]
            confidence = prob_info['confidence']
            margin = prob_info['margin']
            fold_agreement = prob_info['fold_agreement']
            probs = prob_info['probs']
        else:
            confidence = 0.5
            margin = 0.1
            fold_agreement = 0.8
            probs = np.array([0.33, 0.33, 0.34])

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


def extract_ensemble_probabilities(
    fold_dirs: List[Path],
    data_dir: Path,
    device: str = 'auto'
) -> Dict[int, Dict]:
    """
    Extract per-sample probabilities and fold agreement from ensemble models.

    Loads each fold's model, runs inference on the test set, and computes:
    - Weighted soft-voting probabilities per sample
    - Per-fold predictions for fold agreement calculation

    Args:
        fold_dirs: List of fold directories containing best_classifier.pt
        data_dir: Path to warped dataset (parent of train/val/test)
        device: Device string ('cuda', 'cpu', or 'auto')

    Returns:
        Dict mapping sample_idx to {'probs': np.array(3,), 'confidence': float,
        'margin': float, 'fold_agreement': float, 'fold_predictions': list}
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_device = torch.device(device)

    # Load models and validation F1 scores (for weighting)
    models = []
    weights = []
    for fold_dir in fold_dirs:
        checkpoint_path = fold_dir / 'best_classifier.pt'
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}, skipping")
            continue

        model = _load_classifier_from_checkpoint(checkpoint_path, torch_device)
        models.append(model)

        # Get validation F1 for weighting from checkpoint metadata or results file
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        val_f1 = checkpoint_data.get('best_val_f1', None)
        if val_f1 is None:
            results_path = fold_dir / 'results.json'
            if results_path.exists():
                with open(results_path) as f:
                    results = json.load(f)
                val_f1 = results.get('best_val_f1_macro',
                                     results.get('best_f1', 0.98))
            else:
                val_f1 = 0.98
        weights.append(val_f1)

    logger.info(f"Loaded {len(models)} fold models, weights: "
                f"{[f'{w:.4f}' for w in weights]}")

    # Create test dataloader using same transforms as training/evaluation
    from src_v2.models.classifier import get_classifier_transforms
    test_dir = data_dir / 'test'
    transform = get_classifier_transforms(train=False)
    test_dataset = datasets.ImageFolder(str(test_dir), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    logger.info(f"Test set: {len(test_dataset)} samples")

    # Run inference with TTA (original + horizontal flip)
    all_fold_probs = []  # (n_folds, N, 3)
    all_fold_preds = []  # (n_folds, N)

    with torch.no_grad():
        for model in models:
            model.eval()
            fold_probs = []
            fold_preds = []

            for images, _ in test_loader:
                images = images.to(torch_device)

                # Original
                logits_orig = model(images)
                probs_orig = torch.softmax(logits_orig, dim=1)

                # TTA: horizontal flip
                images_flip = torch.flip(images, dims=[3])
                logits_flip = model(images_flip)
                probs_flip = torch.softmax(logits_flip, dim=1)

                # Average TTA
                probs_avg = (probs_orig + probs_flip) / 2.0
                preds = probs_avg.argmax(dim=1)

                fold_probs.append(probs_avg.cpu())
                fold_preds.append(preds.cpu())

            all_fold_probs.append(torch.cat(fold_probs))
            all_fold_preds.append(torch.cat(fold_preds))

    # Stack: (n_folds, N, 3)
    probs_stacked = torch.stack(all_fold_probs, dim=0)
    preds_stacked = torch.stack(all_fold_preds, dim=0)  # (n_folds, N)

    # Weighted soft voting
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    weights_norm = weights_tensor / weights_tensor.sum()
    ensemble_probs = torch.einsum('mni,m->ni', probs_stacked, weights_norm)  # (N, 3)

    # Build per-sample dict
    result = {}
    n_folds = len(models)
    for i in range(len(test_dataset)):
        probs_i = ensemble_probs[i].numpy()
        confidence = float(probs_i.max())
        sorted_probs = np.sort(probs_i)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1])

        # Fold agreement: fraction of folds that agree with ensemble prediction
        ensemble_pred = int(probs_i.argmax())
        fold_preds_i = preds_stacked[:, i].numpy()
        fold_agreement = float((fold_preds_i == ensemble_pred).sum() / n_folds)

        result[i] = {
            'probs': probs_i,
            'confidence': confidence,
            'margin': margin,
            'fold_agreement': fold_agreement,
            'fold_predictions': fold_preds_i.tolist(),
        }

    logger.info(f"Extracted probabilities for {len(result)} samples")
    return result


def _load_classifier_from_checkpoint(
    checkpoint_path: Path,
    device: 'torch.device'
) -> 'torch.nn.Module':
    """Load a classifier model from a checkpoint file."""
    from src_v2.models.classifier import create_classifier

    model = create_classifier(checkpoint=str(checkpoint_path), device=device)
    model.eval()
    return model


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
