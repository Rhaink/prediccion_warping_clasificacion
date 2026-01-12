#!/usr/bin/env python3
"""
Generate Original Cropped 47% dataset for control experiment.

Este script crea un dataset de CONTROL para determinar si la robustez
observada en el modelo warped 47% viene de:
  A) Reduccion de informacion (47% fill = regularizacion implicita)
  B) Normalizacion geometrica (alineacion anatomica por warping)

LOGICA DEL EXPERIMENTO:
- Tomar imagenes ORIGINALES (sin warping)
- Aplicar crop/scale para lograr ~47% fill rate
- Si este dataset ES ROBUSTO -> robustez = reduccion de info
- Si este dataset NO ES ROBUSTO -> robustez = normalizacion geometrica

METODO:
- Imagen original: resize a 154x154 (47% de 224x224)
- Centrar en fondo negro de 224x224
- Fill rate resultante: 154*154 / 224*224 = 47.27%

Uso:
    python scripts/generate_original_cropped_47.py
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration - same as warped dataset for fair comparison
DEFAULT_INPUT_DIR = "data/dataset/COVID-19_Radiography_Dataset"
DEFAULT_OUTPUT_DIR = "outputs/original_cropped_47"
IMAGE_SIZE = 224
SPLITS = (0.75, 0.125, 0.125)  # train, val, test (same as warped)
SEED = 42  # same seed as warped dataset
CLASSES = ["COVID", "Normal", "Viral Pneumonia"]  # 3 classes (same as warped)

# Target fill rate: 47.07% (matching warped dataset)
# For 224x224 image: sqrt(224*224*0.4707) = 153.7 -> 154 pixels
CROP_SIZE = 154  # Results in 47.27% fill rate


def compute_fill_rate(image: np.ndarray) -> float:
    """Compute proportion of non-black pixels."""
    if len(image.shape) == 3:
        # RGB image: pixel is non-black if any channel > 0
        non_black = np.any(image > 0, axis=2)
    else:
        # Grayscale
        non_black = image > 0
    return np.mean(non_black)


def create_cropped_image(
    input_path: Path,
    output_size: int = IMAGE_SIZE,
    crop_size: int = CROP_SIZE,
) -> np.ndarray:
    """
    Create a cropped image with target fill rate.

    Process:
    1. Load original image
    2. Resize to crop_size x crop_size
    3. Place centered on black background of output_size x output_size

    Args:
        input_path: Path to original image
        output_size: Final image size (224)
        crop_size: Size of the cropped region (154 for ~47% fill)

    Returns:
        Cropped image with black background
    """
    # Load and resize image
    img = Image.open(input_path).convert("RGB")
    img_resized = img.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized)

    # Create black background
    output = np.zeros((output_size, output_size, 3), dtype=np.uint8)

    # Calculate center offset
    offset = (output_size - crop_size) // 2

    # Place resized image in center
    output[offset:offset+crop_size, offset:offset+crop_size] = img_array

    return output


def main():
    """Generate Original Cropped 47% dataset."""

    logger.info("=" * 60)
    logger.info("EXPERIMENTO DE CONTROL: Original Cropped 47%")
    logger.info("=" * 60)
    logger.info("")
    logger.info("PROPOSITO: Determinar si la robustez viene de:")
    logger.info("  A) Reduccion de informacion (47% fill)")
    logger.info("  B) Normalizacion geometrica (warping)")
    logger.info("")
    logger.info(f"Target fill rate: {CROP_SIZE}^2 / {IMAGE_SIZE}^2 = "
                f"{(CROP_SIZE**2 / IMAGE_SIZE**2)*100:.2f}%")
    logger.info("=" * 60)

    # Check paths
    input_path = Path(DEFAULT_INPUT_DIR)
    output_path = Path(DEFAULT_OUTPUT_DIR)

    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        sys.exit(1)

    # Collect images (same 3 classes as warped dataset)
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

    # Split dataset (same ratios and seed as warped)
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
            (output_path / split_name / class_name.replace(" ", "_")).mkdir(
                parents=True, exist_ok=True
            )

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

        for img_path, class_name in tqdm(split_imgs, desc=f"Cropping {split_name}"):
            try:
                # Create cropped image
                cropped = create_cropped_image(img_path)

                # Calculate fill rate
                fill_rate = compute_fill_rate(cropped)
                fill_rates.append(fill_rate)

                # Save cropped image
                output_class_dir = output_path / split_name / class_name.replace(" ", "_")
                output_file = output_class_dir / f"{img_path.stem}_cropped.png"
                Image.fromarray(cropped).save(output_file)

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

        logger.info(f"\nComparison:")
        logger.info(f"  Warped 47% dataset:        ~47.07%")
        logger.info(f"  Original Cropped (this):   ~{mean_fill:.2f}%")
        logger.info(f"  Target match:              {'YES' if abs(mean_fill - 47.07) < 1 else 'NO'}")

    # Save summary
    summary = {
        'experiment_type': 'control_original_cropped',
        'purpose': 'Determine if robustness comes from info reduction or geometric normalization',
        'crop_size': CROP_SIZE,
        'output_size': IMAGE_SIZE,
        'theoretical_fill_rate': CROP_SIZE**2 / IMAGE_SIZE**2,
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
            'warped_47_fill_rate': 0.4707,
            'warped_99_fill_rate': 0.9911,
            'original_100_fill_rate': 1.0,
            'this_dataset_fill_rate': float(np.mean(fill_rates)) if fill_rates else None,
        },
        'interpretation': {
            'if_robust': 'Robustez = REDUCCION DE INFORMACION (hipotesis original REFUTADA)',
            'if_not_robust': 'Robustez = NORMALIZACION GEOMETRICA (hipotesis original CONFIRMADA)',
        }
    }

    summary_path = output_path / 'dataset_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nSummary saved to: {summary_path}")
    logger.info(f"Dataset saved to: {output_path}")

    logger.info("\n" + "=" * 60)
    logger.info("NEXT STEPS:")
    logger.info("=" * 60)
    logger.info("1. Train classifier:")
    logger.info(f"   .venv/bin/python -m src_v2 train-classifier {output_path} \\")
    logger.info(f"       --output-dir outputs/classifier_original_cropped_47 --epochs 50")
    logger.info("")
    logger.info("2. Test robustness:")
    logger.info("   .venv/bin/python -m src_v2 test-robustness \\")
    logger.info("       outputs/classifier_original_cropped_47/best_classifier.pt \\")
    logger.info(f"       --data-dir {output_path} \\")
    logger.info("       --output outputs/robustness_original_cropped_47.json")
    logger.info("")
    logger.info("3. Compare JPEG Q50 degradation:")
    logger.info("   - Warped 47%:          0.53%  (ROBUSTO)")
    logger.info("   - Warped 99%:          7.34%  (NO ROBUSTO)")
    logger.info("   - Original Cropped 47%: ???   <- ESTE ES EL DATO CRITICO")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
