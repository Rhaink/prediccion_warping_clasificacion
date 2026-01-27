"""
Verify Alignment Between Warped Dataset and Comparison Dataset

This script verifies that the comparison dataset is perfectly aligned with the
warped classification dataset, ensuring:
1. Same splits (train/val/test)
2. Comparison images are a subset of warped images
3. Same category distribution within each split
4. Proper file naming conventions

Usage:
    python scripts/verify_comparison_alignment.py \\
        --warped outputs/warped_lung_best/session_warping \\
        --comparison outputs/landmark_comparisons/aligned_with_classifier
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd


def verify_alignment(warped_dir: Path, comparison_dir: Path) -> bool:
    """
    Verify that comparison is a properly aligned subset of warped dataset.

    Returns:
        True if all checks pass, False otherwise
    """
    print("=" * 60)
    print("Verifying Alignment Between Datasets")
    print("=" * 60)
    print(f"Warped dataset:     {warped_dir}")
    print(f"Comparison dataset: {comparison_dir}")
    print()

    all_checks_pass = True
    splits = ['train', 'val', 'test']

    for split in splits:
        print(f"\n=== Checking {split.upper()} split ===")

        # Read warped dataset images.csv
        warped_csv = warped_dir / split / 'images.csv'
        if not warped_csv.exists():
            print(f"  ✗ Warped CSV not found: {warped_csv}")
            all_checks_pass = False
            continue

        warped_df = pd.read_csv(warped_csv)
        warped_images = set(warped_df['image_name'])
        print(f"  Warped dataset: {len(warped_images)} images")

        # Collect comparison images
        comparison_images = set()
        comparison_by_category = defaultdict(int)

        for category in ['COVID', 'Normal', 'Viral_Pneumonia']:
            category_dir = comparison_dir / split / category
            if not category_dir.exists():
                print(f"  ✗ Category dir not found: {category_dir}")
                all_checks_pass = False
                continue

            category_files = list(category_dir.glob('*_comparison.png'))
            for img_path in category_files:
                image_name = img_path.stem.replace('_comparison', '')
                comparison_images.add(image_name)
                comparison_by_category[category] += 1

        print(f"  Comparison dataset: {len(comparison_images)} images")

        # Verify subset
        if not comparison_images.issubset(warped_images):
            extra_images = comparison_images - warped_images
            print(f"  ✗ Comparison has {len(extra_images)} images NOT in warped dataset")
            print(f"    First 5: {list(extra_images)[:5]}")
            all_checks_pass = False
        else:
            print(f"  ✓ All comparison images are in warped dataset")

        # Verify category distribution
        print(f"\n  Category distribution:")
        for category, count in sorted(comparison_by_category.items()):
            warped_cat_count = len(warped_df[warped_df['category'] == category])
            percentage = (count / warped_cat_count * 100) if warped_cat_count > 0 else 0
            print(f"    {category:20s}: {count:4d} / {warped_cat_count:4d} ({percentage:5.2f}%)")

        # Verify file naming
        print(f"\n  File naming consistency:")
        naming_issues = 0
        for image_name in list(comparison_images)[:10]:  # Check first 10
            # Find category
            category = None
            for cat in ['COVID', 'Normal', 'Viral_Pneumonia']:
                if (comparison_dir / split / cat / f"{image_name}_comparison.png").exists():
                    category = cat
                    break

            if category:
                warped_file = warped_dir / split / category / f"{image_name}_warped.png"
                if not warped_file.exists():
                    print(f"    ✗ No matching warped file for {image_name}")
                    naming_issues += 1

        if naming_issues == 0:
            print(f"    ✓ File naming conventions match (checked 10 samples)")
        else:
            print(f"    ✗ Found {naming_issues} naming issues")
            all_checks_pass = False

    # Load and verify metadata
    print(f"\n=== Checking Metadata ===")
    metadata_path = comparison_dir / 'metadata.json'
    if not metadata_path.exists():
        print(f"  ✗ Metadata file not found: {metadata_path}")
        all_checks_pass = False
    else:
        with open(metadata_path) as f:
            metadata = json.load(f)

        print(f"  Schema version: {metadata.get('schema_version', 'N/A')}")
        print(f"  Aligned with classifier: {metadata.get('alignment', {}).get('aligned_with_classifier', False)}")
        print(f"  Total in warped: {metadata.get('coverage', {}).get('total_in_warped', 0)}")
        print(f"  With ground truth: {metadata.get('coverage', {}).get('with_ground_truth', 0)}")
        print(f"  Processed: {metadata.get('coverage', {}).get('processed', 0)}")
        print(f"  Coverage: {metadata.get('coverage', {}).get('coverage_percentage', 0):.2f}%")

        if metadata.get('schema_version') != '2.0.0':
            print(f"  ⚠ Warning: Expected schema version 2.0.0")
        if not metadata.get('alignment', {}).get('aligned_with_classifier', False):
            print(f"  ✗ Metadata indicates NOT aligned with classifier")
            all_checks_pass = False
        else:
            print(f"  ✓ Metadata indicates proper alignment")

    # Final summary
    print("\n" + "=" * 60)
    if all_checks_pass:
        print("✓ ALL CHECKS PASSED - Datasets are properly aligned")
        print("=" * 60)
        return True
    else:
        print("✗ SOME CHECKS FAILED - Review errors above")
        print("=" * 60)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify alignment between warped and comparison datasets"
    )
    parser.add_argument(
        '--warped',
        type=str,
        default='outputs/warped_lung_best/session_warping',
        help='Path to warped dataset directory'
    )
    parser.add_argument(
        '--comparison',
        type=str,
        default='outputs/landmark_comparisons/aligned_with_classifier',
        help='Path to comparison dataset directory'
    )

    args = parser.parse_args()

    warped_dir = Path(args.warped)
    comparison_dir = Path(args.comparison)

    if not warped_dir.exists():
        print(f"Error: Warped dataset not found: {warped_dir}")
        sys.exit(1)

    if not comparison_dir.exists():
        print(f"Error: Comparison dataset not found: {comparison_dir}")
        sys.exit(1)

    success = verify_alignment(warped_dir, comparison_dir)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
