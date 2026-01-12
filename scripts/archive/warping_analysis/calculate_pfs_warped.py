#!/usr/bin/env python3
"""
Calculate PFS (Pulmonary Focus Score) with Warped Masks.

Este script corrige el problema metodologico del PFS anterior:
- Antes: PFS calculado con mascaras NO warped sobre imagenes warped = INVALIDO
- Ahora: PFS calculado con mascaras warped sobre imagenes warped = VALIDO

El script:
1. Carga imagen warped y mascara original
2. Aplica warp_mask() con los MISMOS landmarks usados para warpear la imagen
3. Calcula PFS con la mascara correctamente alineada

Uso:
    python scripts/calculate_pfs_warped.py --model outputs/classifier_warped_full_coverage/best_classifier.pt \
        --data outputs/full_coverage_warped_dataset \
        --masks data/dataset/COVID-19_Radiography_Dataset \
        --landmarks outputs/full_coverage_warped_dataset/landmarks.json \
        --output outputs/pfs_warped_valid

Requisitos:
    - Modelo clasificador entrenado
    - Dataset warped con imagenes
    - Archivo landmarks.json con source_landmarks para cada imagen
    - Mascaras originales del dataset COVID-19
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src_v2 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src_v2.processing.warp import warp_mask
from src_v2.models.classifier import ImageClassifier
from src_v2.visualization.gradcam import GradCAM, get_target_layer, calculate_pfs


def load_landmarks_mapping(landmarks_file: Path) -> Dict[str, np.ndarray]:
    """Load landmarks mapping from JSON file.

    Args:
        landmarks_file: Path to landmarks.json

    Returns:
        Dict mapping image filename to source landmarks array
    """
    with open(landmarks_file) as f:
        data = json.load(f)

    mapping = {}
    for entry in data:
        # Support both formats:
        # Format 1: {"filename": "...", "source_landmarks": [...]}
        # Format 2: {"image_name": "...", "landmarks": [...]}

        if 'filename' in entry:
            filename = Path(entry['filename']).stem
        elif 'image_name' in entry:
            filename = entry['image_name']
        else:
            continue

        if 'source_landmarks' in entry:
            landmarks = np.array(entry['source_landmarks'])
        elif 'landmarks' in entry:
            landmarks = np.array(entry['landmarks'])
        else:
            continue

        mapping[filename] = landmarks

    return mapping


def get_canonical_landmarks(output_size: int = 224) -> np.ndarray:
    """Get canonical (target) landmarks for warping.

    These are the standard landmarks that all images are warped TO.
    """
    # Standard canonical landmarks (15 points)
    canonical = np.array([
        [0.30, 0.15],  # Left shoulder
        [0.70, 0.15],  # Right shoulder
        [0.15, 0.40],  # Left upper thorax
        [0.85, 0.40],  # Right upper thorax
        [0.30, 0.50],  # Left mid thorax
        [0.70, 0.50],  # Right mid thorax
        [0.15, 0.65],  # Left lower thorax
        [0.85, 0.65],  # Right lower thorax
        [0.30, 0.75],  # Left costophrenic
        [0.70, 0.75],  # Right costophrenic
        [0.50, 0.20],  # Trachea
        [0.50, 0.35],  # Carina
        [0.50, 0.55],  # Heart center
        [0.35, 0.85],  # Left diaphragm
        [0.65, 0.85],  # Right diaphragm
    ]) * output_size

    return canonical


def load_original_mask(
    image_stem: str,
    class_name: str,
    masks_dir: Path
) -> Optional[np.ndarray]:
    """Load original (non-warped) lung mask.

    Args:
        image_stem: Image filename without extension (may include '_warped')
        class_name: COVID, Normal, or Viral_Pneumonia
        masks_dir: Base directory with mask subdirectories

    Returns:
        Binary mask array or None if not found
    """
    # Remove _warped suffix if present
    clean_stem = image_stem.replace('_warped', '')

    # Map class names to directory names
    class_mapping = {
        'COVID': 'COVID',
        'Normal': 'Normal',
        'Viral_Pneumonia': 'Viral Pneumonia',
    }

    dir_name = class_mapping.get(class_name, class_name)

    # Try different paths
    possible_paths = [
        masks_dir / dir_name / 'masks' / f'{clean_stem}.png',
        masks_dir / class_name / 'masks' / f'{clean_stem}.png',
        masks_dir / 'masks' / dir_name / f'{clean_stem}.png',
    ]

    for path in possible_paths:
        if path.exists():
            mask = np.array(Image.open(path).convert('L'))
            return mask

    return None


def calculate_pfs_with_warped_mask(
    model: nn.Module,
    image: torch.Tensor,
    mask_original: np.ndarray,
    source_landmarks: np.ndarray,
    target_landmarks: np.ndarray,
    device: torch.device,
    use_full_coverage: bool = True,
) -> Tuple[float, np.ndarray, int, float]:
    """Calculate PFS using properly warped mask.

    Args:
        model: Classifier model
        image: Input image tensor (1, C, H, W)
        mask_original: Original (non-warped) binary mask
        source_landmarks: Source landmarks from original image
        target_landmarks: Target/canonical landmarks
        device: Torch device
        use_full_coverage: Whether to use full coverage warping

    Returns:
        Tuple of (PFS value, warped mask, predicted class, confidence)
    """
    # Warp the mask using same transformation as image
    warped_mask = warp_mask(
        mask_original,
        source_landmarks,
        target_landmarks,
        output_size=224,
        use_full_coverage=use_full_coverage
    )

    # Normalize mask to [0, 1]
    warped_mask_norm = warped_mask.astype(np.float32) / 255.0

    # Generate GradCAM heatmap
    backbone_name = getattr(model, 'backbone_name', 'resnet18')
    target_layer = get_target_layer(model, backbone_name)

    with GradCAM(model, target_layer) as gradcam:
        heatmap, pred_class, confidence = gradcam(image.to(device))

    # Calculate PFS with aligned mask
    pfs = calculate_pfs(heatmap, warped_mask_norm)

    return pfs, warped_mask, pred_class, confidence


def main():
    parser = argparse.ArgumentParser(description='Calculate PFS with warped masks')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to classifier model (.pt file)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to warped dataset directory')
    parser.add_argument('--masks', type=str, required=True,
                        help='Path to original masks directory (COVID-19_Radiography_Dataset)')
    parser.add_argument('--landmarks', type=str, required=True,
                        help='Path to landmarks.json with source_landmarks')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to use (train/val/test)')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Max samples to process (None=all)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (1 required for GradCAM)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Setup paths
    model_path = Path(args.model)
    data_path = Path(args.data) / args.split
    masks_path = Path(args.masks)
    landmarks_path = Path(args.landmarks)
    output_path = Path(args.output)

    output_path.mkdir(parents=True, exist_ok=True)

    # Verify paths
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        sys.exit(1)
    if not masks_path.exists():
        logger.error(f"Masks directory not found: {masks_path}")
        sys.exit(1)
    if not landmarks_path.exists():
        logger.error(f"Landmarks file not found: {landmarks_path}")
        sys.exit(1)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = ImageClassifier(num_classes=3, backbone='resnet18')
    checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Load landmarks mapping
    logger.info(f"Loading landmarks from {landmarks_path}")
    landmarks_mapping = load_landmarks_mapping(landmarks_path)
    logger.info(f"Loaded landmarks for {len(landmarks_mapping)} images")

    # Get canonical landmarks
    target_landmarks = get_canonical_landmarks(224)

    # Setup dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(data_path, transform=transform)
    class_names = dataset.classes
    logger.info(f"Classes: {class_names}")

    # Process images
    results = []
    missing_landmarks = 0
    missing_masks = 0

    samples_to_process = args.num_samples or len(dataset)

    logger.info(f"Processing {samples_to_process} samples from {args.split} split...")

    for idx in tqdm(range(min(samples_to_process, len(dataset)))):
        img_path, label = dataset.samples[idx]
        img_stem = Path(img_path).stem
        true_class = class_names[label]

        # Get source landmarks
        clean_stem = img_stem.replace('_warped', '')
        if clean_stem not in landmarks_mapping:
            missing_landmarks += 1
            continue

        source_landmarks = landmarks_mapping[clean_stem]

        # Load original mask
        mask_original = load_original_mask(img_stem, true_class, masks_path)
        if mask_original is None:
            missing_masks += 1
            continue

        # Load image
        image, _ = dataset[idx]
        image = image.unsqueeze(0)

        try:
            # Calculate PFS with warped mask
            pfs, warped_mask, pred_class, confidence = calculate_pfs_with_warped_mask(
                model, image, mask_original, source_landmarks, target_landmarks,
                device, use_full_coverage=True
            )

            results.append({
                'image_path': str(img_path),
                'image_stem': img_stem,
                'true_class': true_class,
                'predicted_class': class_names[pred_class],
                'confidence': float(confidence),
                'pfs': float(pfs),
                'correct': pred_class == label,
            })

        except Exception as e:
            logger.warning(f"Error processing {img_stem}: {e}")
            continue

    # Calculate statistics
    if not results:
        logger.error("No results collected!")
        sys.exit(1)

    pfs_values = [r['pfs'] for r in results]
    pfs_by_class = {}

    for class_name in class_names:
        class_pfs = [r['pfs'] for r in results if r['true_class'] == class_name]
        if class_pfs:
            pfs_by_class[class_name] = {
                'mean': float(np.mean(class_pfs)),
                'std': float(np.std(class_pfs)),
                'count': len(class_pfs),
            }

    correct_pfs = [r['pfs'] for r in results if r['correct']]
    incorrect_pfs = [r['pfs'] for r in results if not r['correct']]

    summary = {
        'total_samples': len(results),
        'missing_landmarks': missing_landmarks,
        'missing_masks': missing_masks,
        'mean_pfs': float(np.mean(pfs_values)),
        'std_pfs': float(np.std(pfs_values)),
        'median_pfs': float(np.median(pfs_values)),
        'min_pfs': float(np.min(pfs_values)),
        'max_pfs': float(np.max(pfs_values)),
        'pfs_by_class': pfs_by_class,
        'pfs_correct': {
            'mean': float(np.mean(correct_pfs)) if correct_pfs else 0,
            'std': float(np.std(correct_pfs)) if correct_pfs else 0,
            'count': len(correct_pfs),
        },
        'pfs_incorrect': {
            'mean': float(np.mean(incorrect_pfs)) if incorrect_pfs else 0,
            'std': float(np.std(incorrect_pfs)) if incorrect_pfs else 0,
            'count': len(incorrect_pfs),
        },
        'methodology': 'PFS with warped masks (VALID)',
        'note': 'Masks transformed using warp_mask() with same landmarks as images',
    }

    # Save results
    summary_file = output_path / 'pfs_warped_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    details_file = output_path / 'pfs_warped_details.json'
    with open(details_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print results
    logger.info("=" * 60)
    logger.info("PFS RESULTS (WITH WARPED MASKS)")
    logger.info("=" * 60)
    logger.info(f"Total samples:     {summary['total_samples']}")
    logger.info(f"Missing landmarks: {missing_landmarks}")
    logger.info(f"Missing masks:     {missing_masks}")
    logger.info("")
    logger.info(f"Mean PFS:   {summary['mean_pfs']:.4f} (+/- {summary['std_pfs']:.4f})")
    logger.info(f"Median PFS: {summary['median_pfs']:.4f}")
    logger.info(f"Range:      [{summary['min_pfs']:.4f}, {summary['max_pfs']:.4f}]")
    logger.info("")
    logger.info("PFS by class:")
    for class_name, stats in pfs_by_class.items():
        logger.info(f"  {class_name}: {stats['mean']:.4f} (+/- {stats['std']:.4f}) n={stats['count']}")
    logger.info("")
    logger.info("PFS by prediction correctness:")
    logger.info(f"  Correct:   {summary['pfs_correct']['mean']:.4f} (+/- {summary['pfs_correct']['std']:.4f}) n={summary['pfs_correct']['count']}")
    logger.info(f"  Incorrect: {summary['pfs_incorrect']['mean']:.4f} (+/- {summary['pfs_incorrect']['std']:.4f}) n={summary['pfs_incorrect']['count']}")
    logger.info("")
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
