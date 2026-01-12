#!/usr/bin/env python3
"""
Filter Dataset to 3 Classes for Valid Cross-Evaluation.

Este script filtra el dataset original de COVID-19_Radiography_Dataset para
incluir solo 3 clases (COVID, Normal, Viral_Pneumonia), excluyendo Lung_Opacity.

PROBLEMA:
- Dataset original tiene 4 clases (COVID, Lung_Opacity, Normal, Viral Pneumonia)
- Datasets warped tienen 3 clases (COVID, Normal, Viral_Pneumonia)
- Cross-evaluation actual compara 4 clases vs 3 clases = INVALIDO

SOLUCION:
- Crear dataset original filtrado a 3 clases
- Usar MISMOS splits y seed que datasets warped
- Permitir cross-evaluation justo

Uso:
    python scripts/filter_dataset_3_classes.py

O con CLI (si se implementa):
    .venv/bin/python -m src_v2 filter-dataset-classes \
        --data-dir data/COVID-19_Radiography_Dataset \
        --output-dir outputs/original_3_classes \
        --exclude Lung_Opacity
"""

import sys
import json
import shutil
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

# Configuration - SAME as warped datasets for fair comparison
DEFAULT_INPUT_DIR = "data/dataset/COVID-19_Radiography_Dataset"
DEFAULT_OUTPUT_DIR = "outputs/original_3_classes"
IMAGE_SIZE = 224
SPLITS = (0.75, 0.125, 0.125)  # train, val, test (same as warped)
SEED = 42  # same seed as warped datasets

# 3 classes only (excluding Lung_Opacity)
CLASSES_TO_INCLUDE = ["COVID", "Normal", "Viral Pneumonia"]
CLASSES_TO_EXCLUDE = ["Lung_Opacity"]


def main():
    """Filter dataset to 3 classes."""

    logger.info("=" * 60)
    logger.info("FILTER DATASET: Original 3 Classes")
    logger.info("=" * 60)
    logger.info("")
    logger.info("PROPOSITO: Crear dataset original con solo 3 clases")
    logger.info("           para cross-evaluation valido vs datasets warped")
    logger.info("")
    logger.info(f"Clases a incluir: {CLASSES_TO_INCLUDE}")
    logger.info(f"Clases a excluir: {CLASSES_TO_EXCLUDE}")
    logger.info("=" * 60)

    # Check paths
    input_path = Path(DEFAULT_INPUT_DIR)
    output_path = Path(DEFAULT_OUTPUT_DIR)

    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        sys.exit(1)

    # Collect images from 3 classes only
    logger.info("\nCollecting images from 3 classes...")
    all_images = []

    for class_name in CLASSES_TO_INCLUDE:
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

    logger.info(f"\nTotal images (3 classes): {len(all_images)}")

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
        for class_name in CLASSES_TO_INCLUDE:
            (output_path / split_name / class_name.replace(" ", "_")).mkdir(
                parents=True, exist_ok=True
            )

    # Process images (resize and copy)
    splits_data = {
        'train': train_imgs,
        'val': val_imgs,
        'test': test_imgs,
    }

    stats = {split: {'success': 0, 'failed': 0} for split in splits_data}
    class_counts = {split: {c.replace(" ", "_"): 0 for c in CLASSES_TO_INCLUDE} for split in splits_data}

    for split_name, split_imgs in splits_data.items():
        logger.info(f"\nProcessing {split_name} split ({len(split_imgs)} images)...")

        for img_path, class_name in tqdm(split_imgs, desc=f"Processing {split_name}"):
            try:
                # Load and resize image to 224x224
                img = Image.open(img_path).convert("RGB")
                img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)

                # Save to output directory
                class_name_sanitized = class_name.replace(" ", "_")
                output_class_dir = output_path / split_name / class_name_sanitized
                output_file = output_class_dir / f"{img_path.stem}.png"
                img_resized.save(output_file)

                stats[split_name]['success'] += 1
                class_counts[split_name][class_name_sanitized] += 1

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

    # Class distribution per split
    logger.info("\nClass distribution:")
    for split_name in ['train', 'val', 'test']:
        logger.info(f"\n  {split_name}:")
        for class_name in CLASSES_TO_INCLUDE:
            cn = class_name.replace(" ", "_")
            logger.info(f"    {cn}: {class_counts[split_name][cn]}")

    # Verify against warped dataset
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION vs WARPED DATASETS")
    logger.info("=" * 60)

    warped_47_path = Path("outputs/warped_dataset")
    warped_99_path = Path("outputs/full_coverage_warped_dataset")

    if warped_99_path.exists():
        warped_summary = warped_99_path / "dataset_summary.json"
        if warped_summary.exists():
            with open(warped_summary) as f:
                warped_data = json.load(f)
            warped_total = sum(warped_data['stats'][s]['success'] for s in ['train', 'val', 'test'])
            logger.info(f"  Warped 99% dataset:    {warped_total} images")
            logger.info(f"  Original 3 classes:    {total_success} images")
            if total_success == warped_total:
                logger.info("  MATCH: Same number of images!")
            else:
                logger.warning(f"  MISMATCH: Difference of {abs(total_success - warped_total)} images")

    # Save summary
    summary = {
        'experiment_type': 'original_3_classes_filtered',
        'purpose': 'Valid cross-evaluation with same classes as warped datasets',
        'image_size': IMAGE_SIZE,
        'seed': SEED,
        'splits': {'train': train_ratio, 'val': val_ratio, 'test': test_ratio},
        'classes_included': CLASSES_TO_INCLUDE,
        'classes_excluded': CLASSES_TO_EXCLUDE,
        'stats': stats,
        'class_distribution': class_counts,
        'total_images': total_success,
        'comparison': {
            'original_4_classes': '~42,330 images (includes Lung_Opacity)',
            'original_3_classes': f'{total_success} images (this dataset)',
            'warped_47_fill': '15,153 images',
            'warped_99_fill': '15,153 images',
        },
        'use_case': 'Train classifier on this dataset for valid cross-evaluation'
    }

    summary_path = output_path / 'dataset_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nSummary saved to: {summary_path}")
    logger.info(f"Dataset saved to: {output_path}")

    logger.info("\n" + "=" * 60)
    logger.info("NEXT STEPS:")
    logger.info("=" * 60)
    logger.info("1. Train classifier on this 3-class original dataset:")
    logger.info(f"   .venv/bin/python -m src_v2 train-classifier {output_path} \\")
    logger.info("       --output-dir outputs/classifier_original_3classes --epochs 50")
    logger.info("")
    logger.info("2. Valid cross-evaluation (3 classes vs 3 classes):")
    logger.info("   .venv/bin/python -m src_v2 cross-evaluate \\")
    logger.info("       outputs/classifier_original_3classes/best_classifier.pt \\")
    logger.info("       outputs/classifier_warped_full_coverage/best_classifier.pt \\")
    logger.info(f"       --data-a {output_path} \\")
    logger.info("       --data-b outputs/full_coverage_warped_dataset \\")
    logger.info("       --output-dir outputs/cross_evaluation_valid_3classes")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
