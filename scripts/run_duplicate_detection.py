#!/usr/bin/env python3
"""Run duplicate detection on original and warped datasets.

Performs dual-stage duplicate detection:
1. Original dataset: Detect inherent duplicates
2. Warped dataset: Detect post-normalization duplicates
3. Cross-reference: Analyze convergence/divergence between datasets

Outputs:
- original_duplicates.csv: All duplicate pairs in original dataset
- warped_duplicates.csv: All duplicate pairs in warped dataset
- cross_split_leakage.csv: Duplicates crossing train/val/test boundaries (CRITICAL)
- duplicate_analysis_summary.json: Structured results for downstream analysis
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm import tqdm

from src_v2.utils.duplicates import (
    classify_duplicate_types,
    compare_original_vs_warped,
    detect_duplicates,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_dataset_structure(original_dir: Path, warped_dir: Path) -> Dict[str, Dict[str, str]]:
    """Build mapping from filename to split and class.

    Args:
        original_dir: Original dataset directory (class-based structure)
        warped_dir: Warped dataset directory (split-based structure)

    Returns:
        Dict mapping filename -> {"split": str, "class": str}
    """
    structure = {}

    # Original dataset: COVID/, Normal/, Viral_Pneumonia/
    for class_dir in original_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        if class_name not in ["COVID", "Normal", "Viral_Pneumonia"]:
            continue

        for img_path in class_dir.glob("*.png"):
            filename = img_path.name
            structure[filename] = {
                "split": "unknown",  # Original dataset doesn't have split info
                "class": class_name,
            }

    # Warped dataset: train/, val/, test/ each with class subdirs
    if warped_dir.exists():
        for split_dir in warped_dir.iterdir():
            if not split_dir.is_dir():
                continue

            split_name = split_dir.name
            if split_name not in ["train", "val", "test"]:
                continue

            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue

                class_name = class_dir.name
                if class_name not in ["COVID", "Normal", "Viral_Pneumonia"]:
                    continue

                for img_path in class_dir.glob("*.png"):
                    filename = img_path.name
                    structure[filename] = {
                        "split": split_name,
                        "class": class_name,
                    }

    logger.info(f"Built dataset structure with {len(structure)} images")
    return structure


def save_pairs_to_csv(pairs, output_path: Path, dataset_structure: Dict):
    """Save duplicate pairs to CSV with metadata."""
    if len(pairs) == 0:
        # Create empty CSV with headers
        df = pd.DataFrame(
            columns=[
                "img1",
                "img2",
                "hash_distance",
                "cnn_similarity",
                "class1",
                "class2",
                "split1",
                "split2",
                "duplicate_type",
            ]
        )
        df.to_csv(output_path, index=False)
        logger.info(f"No duplicates found - saved empty CSV to {output_path}")
        return

    # Extract metadata for each pair
    rows = []
    for img1, img2, hash_dist, cnn_sim in pairs:
        fname1 = Path(img1).name
        fname2 = Path(img2).name

        meta1 = dataset_structure.get(fname1, {"split": "unknown", "class": "unknown"})
        meta2 = dataset_structure.get(fname2, {"split": "unknown", "class": "unknown"})

        # Determine duplicate type
        split1, class1 = meta1["split"], meta1["class"]
        split2, class2 = meta2["split"], meta2["class"]

        if split1 != split2 and class1 != class2:
            dup_type = "cross_split_cross_class"
        elif split1 != split2:
            dup_type = "cross_split"
        elif class1 != class2:
            dup_type = "cross_class"
        else:
            dup_type = "within_split_within_class"

        rows.append({
            "img1": img1,
            "img2": img2,
            "hash_distance": hash_dist,
            "cnn_similarity": cnn_sim if cnn_sim is not None else "",
            "class1": class1,
            "class2": class2,
            "split1": split1,
            "split2": split2,
            "duplicate_type": dup_type,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} duplicate pairs to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run duplicate detection on original and warped datasets"
    )
    parser.add_argument(
        "--original-dir",
        type=Path,
        default=Path("data/dataset/COVID-19_Radiography_Dataset"),
        help="Original dataset directory",
    )
    parser.add_argument(
        "--warped-dir",
        type=Path,
        default=Path("outputs/warped_lung_best/session_warping"),
        help="Warped dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/error_forensics/duplicates"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--hash-threshold",
        type=int,
        default=3,
        help="PHash Hamming distance threshold (default: 3)",
    )
    parser.add_argument(
        "--cnn-threshold",
        type=float,
        default=0.98,
        help="CNN similarity threshold (default: 0.98)",
    )
    parser.add_argument(
        "--skip-cnn",
        action="store_true",
        help="Skip CNN verification (faster, may have false positives)",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.original_dir.exists():
        logger.error(f"Original dataset not found: {args.original_dir}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset structure
    logger.info("Building dataset structure...")
    dataset_structure = build_dataset_structure(args.original_dir, args.warped_dir)

    # Detect duplicates in original dataset
    logger.info("=" * 80)
    logger.info("STAGE 1: Detecting duplicates in ORIGINAL dataset")
    logger.info("=" * 80)

    original_results = detect_duplicates(
        str(args.original_dir),
        hash_threshold=args.hash_threshold,
        min_similarity_cnn=args.cnn_threshold,
        use_cnn_verification=not args.skip_cnn,
    )

    original_pairs = original_results["pairs"]
    logger.info(
        f"Original dataset: {original_results['total_images']} images, "
        f"{original_results['total_pairs']} duplicate pairs"
    )

    # Save original duplicates
    original_csv = args.output_dir / "original_duplicates.csv"
    save_pairs_to_csv(original_pairs, original_csv, dataset_structure)

    # Classify original duplicates
    original_classified = classify_duplicate_types(original_pairs, dataset_structure)

    # Detect duplicates in warped dataset (if it exists)
    warped_results = None
    warped_pairs = []
    warped_classified = None

    if args.warped_dir.exists():
        logger.info("=" * 80)
        logger.info("STAGE 2: Detecting duplicates in WARPED dataset")
        logger.info("=" * 80)

        warped_results = detect_duplicates(
            str(args.warped_dir),
            hash_threshold=args.hash_threshold,
            min_similarity_cnn=args.cnn_threshold,
            use_cnn_verification=not args.skip_cnn,
        )

        warped_pairs = warped_results["pairs"]
        logger.info(
            f"Warped dataset: {warped_results['total_images']} images, "
            f"{warped_results['total_pairs']} duplicate pairs"
        )

        # Save warped duplicates
        warped_csv = args.output_dir / "warped_duplicates.csv"
        save_pairs_to_csv(warped_pairs, warped_csv, dataset_structure)

        # Classify warped duplicates
        warped_classified = classify_duplicate_types(warped_pairs, dataset_structure)

    else:
        logger.warning(f"Warped dataset not found at {args.warped_dir}")

    # Cross-split leakage analysis
    logger.info("=" * 80)
    logger.info("STAGE 3: Cross-split leakage analysis")
    logger.info("=" * 80)

    # Combine all cross-split pairs from both datasets
    cross_split_pairs = []

    if original_classified:
        cross_split_pairs.extend(original_classified["cross_split"])
        cross_split_pairs.extend(original_classified["cross_split_cross_class"])

    if warped_classified:
        cross_split_pairs.extend(warped_classified["cross_split"])
        cross_split_pairs.extend(warped_classified["cross_split_cross_class"])

    # Remove duplicates (same pair might be in both datasets)
    seen = set()
    unique_cross_split = []
    for pair in cross_split_pairs:
        key = tuple(sorted([pair["img1"], pair["img2"]]))
        if key not in seen:
            seen.add(key)
            unique_cross_split.append(pair)

    # Save cross-split leakage
    cross_split_csv = args.output_dir / "cross_split_leakage.csv"
    if unique_cross_split:
        df = pd.DataFrame(unique_cross_split)
        df.to_csv(cross_split_csv, index=False)
        logger.warning(f"CRITICAL: Found {len(unique_cross_split)} cross-split duplicates!")
    else:
        # Empty CSV
        df = pd.DataFrame(
            columns=[
                "img1",
                "img2",
                "hash_distance",
                "cnn_similarity",
                "split1",
                "split2",
                "class1",
                "class2",
            ]
        )
        df.to_csv(cross_split_csv, index=False)
        logger.info("No cross-split duplicates found")

    # Convergence/divergence analysis
    comparison = None
    if warped_results:
        logger.info("=" * 80)
        logger.info("STAGE 4: Original vs Warped comparison")
        logger.info("=" * 80)

        comparison = compare_original_vs_warped(original_pairs, warped_pairs)
        logger.info(f"Diverged: {comparison['counts']['diverged']} pairs")
        logger.info(f"Converged: {comparison['counts']['converged']} pairs")
        logger.info(f"Persistent: {comparison['counts']['persistent']} pairs")

    # Generate summary JSON
    summary = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "hash_threshold": args.hash_threshold,
            "cnn_threshold": args.cnn_threshold,
            "cnn_verification_enabled": not args.skip_cnn,
        },
        "original": {
            "total_images": original_results["total_images"],
            "total_pairs": original_results["total_pairs"],
            "cross_split": original_classified["counts"]["cross_split"],
            "cross_class": original_classified["counts"]["cross_class"],
            "cross_split_cross_class": original_classified["counts"]["cross_split_cross_class"],
            "within_split": original_classified["counts"]["within_split_within_class"],
        },
    }

    if warped_results:
        summary["warped"] = {
            "total_images": warped_results["total_images"],
            "total_pairs": warped_results["total_pairs"],
            "cross_split": warped_classified["counts"]["cross_split"],
            "cross_class": warped_classified["counts"]["cross_class"],
            "cross_split_cross_class": warped_classified["counts"]["cross_split_cross_class"],
            "within_split": warped_classified["counts"]["within_split_within_class"],
        }

    if comparison:
        summary["comparison"] = comparison["counts"]
        summary["comparison"]["analysis"] = comparison["analysis"]

    # Critical findings
    critical_findings = []
    total_cross_split = summary["original"]["cross_split"] + summary["original"][
        "cross_split_cross_class"
    ]
    if warped_results:
        total_cross_split += summary["warped"]["cross_split"] + summary["warped"][
            "cross_split_cross_class"
        ]

    if total_cross_split > 0:
        critical_findings.append(f"Cross-split leakage found: {len(unique_cross_split)} pairs")

    total_cross_class = summary["original"]["cross_class"]
    if warped_results:
        total_cross_class += summary["warped"]["cross_class"]

    if total_cross_class > 0:
        critical_findings.append(f"Potential label errors: {total_cross_class} cross-class pairs")

    if comparison and comparison["counts"]["converged"] > 0:
        critical_findings.append(
            f"Warping convergence: {comparison['counts']['converged']} pairs became similar"
        )

    summary["critical_findings"] = critical_findings

    # Save summary
    summary_path = args.output_dir / "duplicate_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved summary to {summary_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("DUPLICATE DETECTION SUMMARY")
    print("=" * 80)
    print(f"\nOriginal Dataset:")
    print(f"  Total images: {summary['original']['total_images']}")
    print(f"  Duplicate pairs: {summary['original']['total_pairs']}")
    print(f"  - Cross-split: {summary['original']['cross_split']} (CRITICAL)")
    print(
        f"  - Cross-split + Cross-class: {summary['original']['cross_split_cross_class']} (CRITICAL)"
    )
    print(f"  - Cross-class: {summary['original']['cross_class']} (WARNING)")
    print(f"  - Within-split: {summary['original']['within_split']} (INFO)")

    if warped_results:
        print(f"\nWarped Dataset:")
        print(f"  Total images: {summary['warped']['total_images']}")
        print(f"  Duplicate pairs: {summary['warped']['total_pairs']}")
        print(f"  - Cross-split: {summary['warped']['cross_split']} (CRITICAL)")
        print(
            f"  - Cross-split + Cross-class: {summary['warped']['cross_split_cross_class']} (CRITICAL)"
        )
        print(f"  - Cross-class: {summary['warped']['cross_class']} (WARNING)")
        print(f"  - Within-split: {summary['warped']['within_split']} (INFO)")

    if comparison:
        print(f"\nOriginal vs Warped Comparison:")
        print(f"  Diverged: {comparison['counts']['diverged']} pairs (warping differentiated)")
        print(f"  Converged: {comparison['counts']['converged']} pairs (warping made similar)")
        print(f"  Persistent: {comparison['counts']['persistent']} pairs (unchanged)")

    if critical_findings:
        print("\nCRITICAL FINDINGS:")
        for finding in critical_findings:
            print(f"  - {finding}")
    else:
        print("\nNo critical issues found!")

    print("=" * 80)


if __name__ == "__main__":
    main()
