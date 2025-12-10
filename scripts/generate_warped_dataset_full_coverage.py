#!/usr/bin/env python3
"""
Generate warped dataset with full_coverage=True (~96% fill rate).

Este script genera un dataset warped con cobertura completa de la imagen,
resolviendo el sesgo metodologico identificado en Sesion 35:
- Dataset actual (use_full_coverage=False): ~47% fill rate
- Dataset nuevo (use_full_coverage=True): ~96% fill rate

Uso:
    python scripts/generate_warped_dataset_full_coverage.py

El script usa los mismos parametros que el dataset original para
permitir una comparacion justa.
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src_v2.models import create_model
from src_v2.processing.warp import (
    piecewise_affine_warp,
    scale_landmarks_from_centroid,
    clip_landmarks_to_image,
    compute_fill_rate,
)
from src_v2.cli import detect_architecture_from_checkpoint, get_device
from src_v2.constants import IMAGENET_MEAN, IMAGENET_STD

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default paths (same as original dataset generation)
DEFAULT_INPUT_DIR = "data/COVID-19_Radiography_Dataset"
DEFAULT_OUTPUT_DIR = "outputs/full_coverage_warped_dataset"
DEFAULT_CHECKPOINT = "checkpoints/session10/ensemble/seed123/final_model.pt"
DEFAULT_CANONICAL = "outputs/shape_analysis/canonical_shape_pixels.json"
DEFAULT_TRIANGLES = "outputs/shape_analysis/canonical_triangles.json"

# Same parameters as original dataset
MARGIN_SCALE = 1.05
SPLITS = (0.75, 0.125, 0.125)  # train, val, test
SEED = 42
CLASSES = ["COVID", "Normal", "Viral Pneumonia"]
IMAGE_SIZE = 224


def load_and_preprocess_image(
    path: Path,
    target_size: int = IMAGE_SIZE,
    use_clahe: bool = True,
    clahe_clip: float = 2.0,
    clahe_tile: int = 4,
) -> np.ndarray:
    """Load and preprocess image for landmark prediction."""
    img = Image.open(path).convert("RGB")
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    img = np.array(img)

    if use_clahe:
        # Apply CLAHE to luminance channel
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return img


def predict_landmarks(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Predict landmarks from image."""
    # Normalize for model
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    img_normalized = (image.astype(np.float32) / 255.0 - mean) / std
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)

    # Convert to pixel coordinates
    landmarks = output.cpu().numpy().reshape(-1, 2)
    landmarks = landmarks * IMAGE_SIZE

    return landmarks


def main():
    """Generate warped dataset with full coverage."""

    logger.info("=" * 60)
    logger.info("Generating Full Coverage Warped Dataset")
    logger.info("=" * 60)
    logger.info("use_full_coverage=True -> Expected fill rate: ~96%%")
    logger.info("=" * 60)

    # Check paths
    input_path = Path(DEFAULT_INPUT_DIR)
    output_path = Path(DEFAULT_OUTPUT_DIR)
    checkpoint_path = Path(DEFAULT_CHECKPOINT)
    canonical_path = Path(DEFAULT_CANONICAL)
    triangles_path = Path(DEFAULT_TRIANGLES)

    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        logger.info("Please ensure COVID-19_Radiography_Dataset is in data/")
        sys.exit(1)

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    if not canonical_path.exists():
        logger.error(f"Canonical shape not found: {canonical_path}")
        logger.info("Generate it first: python -m src_v2 compute-canonical")
        sys.exit(1)

    if not triangles_path.exists():
        logger.error(f"Triangles not found: {triangles_path}")
        sys.exit(1)

    # Load canonical shape and triangles
    logger.info(f"Loading canonical shape: {canonical_path}")
    with open(canonical_path, 'r') as f:
        canonical_data = json.load(f)
    canonical = np.array(canonical_data['canonical_shape_pixels'])

    logger.info(f"Loading triangles: {triangles_path}")
    with open(triangles_path, 'r') as f:
        tri_data = json.load(f)
    # Note: triangles will be recomputed with boundary points when use_full_coverage=True

    # Setup device and model
    device = get_device("auto")
    logger.info(f"Using device: {device}")

    logger.info(f"Loading model: {checkpoint_path}")
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint_data:
        state_dict = checkpoint_data["model_state_dict"]
    else:
        state_dict = checkpoint_data

    arch_params = detect_architecture_from_checkpoint(state_dict)
    logger.info(f"Architecture: coord_attention={arch_params['use_coord_attention']}, "
                f"deep_head={arch_params['deep_head']}, hidden_dim={arch_params['hidden_dim']}")

    model = create_model(
        pretrained=False,
        use_coord_attention=arch_params["use_coord_attention"],
        deep_head=arch_params["deep_head"],
        hidden_dim=arch_params["hidden_dim"],
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Collect images
    logger.info("Collecting images...")
    all_images = []

    for class_name in CLASSES:
        # Try typical structure: class_name/images/
        class_dir = input_path / class_name / "images"
        if not class_dir.exists():
            class_dir = input_path / class_name

        if not class_dir.exists():
            logger.warning(f"Class directory not found: {class_dir}")
            continue

        images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        for img_path in images:
            all_images.append((img_path, class_name))

        logger.info(f"  {class_name}: {len(images)} images")

    logger.info(f"Total images: {len(all_images)}")

    if len(all_images) == 0:
        logger.error("No images found!")
        sys.exit(1)

    # Split dataset
    np.random.seed(SEED)
    train_ratio, val_ratio, test_ratio = SPLITS

    train_imgs, temp_imgs = train_test_split(
        all_images, train_size=train_ratio, random_state=SEED,
        stratify=[x[1] for x in all_images]
    )

    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_imgs, test_imgs = train_test_split(
        temp_imgs, train_size=val_ratio_adjusted, random_state=SEED,
        stratify=[x[1] for x in temp_imgs]
    )

    logger.info(f"Split: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    for split_name in ['train', 'val', 'test']:
        for class_name in CLASSES:
            (output_path / split_name / class_name.replace(" ", "_")).mkdir(parents=True, exist_ok=True)

    # Process images
    splits_data = {
        'train': train_imgs,
        'val': val_imgs,
        'test': test_imgs,
    }

    fill_rates = []
    stats = {split: {'success': 0, 'failed': 0} for split in splits_data}

    for split_name, split_imgs in splits_data.items():
        logger.info(f"\nProcessing {split_name} split ({len(split_imgs)} images)...")

        for img_path, class_name in tqdm(split_imgs, desc=f"Warping {split_name}"):
            try:
                # Load and preprocess
                img = load_and_preprocess_image(img_path, use_clahe=True)

                # Predict landmarks
                landmarks = predict_landmarks(model, img, device)

                # Scale landmarks with margin
                scaled_landmarks = scale_landmarks_from_centroid(landmarks, MARGIN_SCALE)

                # Clip to image bounds
                landmarks_clipped = clip_landmarks_to_image(
                    scaled_landmarks, IMAGE_SIZE, margin=2
                )

                # Apply warping with FULL COVERAGE
                warped = piecewise_affine_warp(
                    image=img,
                    source_landmarks=landmarks_clipped.astype(np.float32),
                    target_landmarks=canonical.astype(np.float32),
                    triangles=None,  # Will be recomputed with boundary points
                    use_full_coverage=True,  # KEY CHANGE: Full coverage enabled
                )

                # Calculate fill rate
                fill_rate = compute_fill_rate(warped)
                fill_rates.append(fill_rate)

                # Save warped image
                output_class_dir = output_path / split_name / class_name.replace(" ", "_")
                output_file = output_class_dir / f"{img_path.stem}_warped.png"
                Image.fromarray(warped).save(output_file)

                stats[split_name]['success'] += 1

            except Exception as e:
                logger.warning(f"Failed to process {img_path.name}: {e}")
                stats[split_name]['failed'] += 1

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    total_success = sum(s['success'] for s in stats.values())
    total_failed = sum(s['failed'] for s in stats.values())

    logger.info(f"Total processed: {total_success} success, {total_failed} failed")

    if fill_rates:
        mean_fill = np.mean(fill_rates) * 100
        std_fill = np.std(fill_rates) * 100
        min_fill = np.min(fill_rates) * 100
        max_fill = np.max(fill_rates) * 100

        logger.info(f"\nFILL RATE STATISTICS:")
        logger.info(f"  Mean:  {mean_fill:.2f}%")
        logger.info(f"  Std:   {std_fill:.2f}%")
        logger.info(f"  Min:   {min_fill:.2f}%")
        logger.info(f"  Max:   {max_fill:.2f}%")

        logger.info(f"\nComparison with original dataset:")
        logger.info(f"  Original (use_full_coverage=False): ~47%")
        logger.info(f"  New (use_full_coverage=True):       ~{mean_fill:.0f}%")

    # Save summary
    summary = {
        'use_full_coverage': True,
        'margin_scale': MARGIN_SCALE,
        'seed': SEED,
        'splits': {'train': train_ratio, 'val': val_ratio, 'test': test_ratio},
        'classes': CLASSES,
        'stats': stats,
        'fill_rate': {
            'mean': float(np.mean(fill_rates)) if fill_rates else None,
            'std': float(np.std(fill_rates)) if fill_rates else None,
            'min': float(np.min(fill_rates)) if fill_rates else None,
            'max': float(np.max(fill_rates)) if fill_rates else None,
        },
        'comparison': {
            'original_fill_rate': 0.47,
            'new_fill_rate': float(np.mean(fill_rates)) if fill_rates else None,
        }
    }

    summary_path = output_path / 'dataset_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nSummary saved to: {summary_path}")
    logger.info(f"Dataset saved to: {output_path}")
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
