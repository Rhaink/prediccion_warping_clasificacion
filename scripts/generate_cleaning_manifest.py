#!/usr/bin/env python3
"""
Generate cleaning manifest by merging all data quality signals.

Merges:
- Landmark outlier detection (outputs/data_cleaning/landmark_outliers.csv)
- Cross-split duplicate resolution (outputs/data_cleaning/cross_split_exclusions.csv)
- Cleanlab label noise detection (outputs/data_cleaning/cleanlab_issues.csv)

Produces a single JSON manifest with per-image flags and decisions.
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Generate cleaning manifest from quality signals")
    parser.add_argument(
        "--landmark-outliers",
        default="outputs/data_cleaning/landmark_outliers.csv",
        help="Path to landmark outlier detection CSV",
    )
    parser.add_argument(
        "--cross-split-exclusions",
        default="outputs/data_cleaning/cross_split_exclusions.csv",
        help="Path to cross-split duplicate exclusion CSV",
    )
    parser.add_argument(
        "--cleanlab-issues",
        default="outputs/data_cleaning/cleanlab_issues.csv",
        help="Path to cleanlab label noise CSV",
    )
    parser.add_argument(
        "--dataset-summary",
        default="outputs/warped_lung_best/session_warping/dataset_summary.json",
        help="Path to dataset summary JSON (for test set identification)",
    )
    parser.add_argument(
        "--test-images-csv",
        default="outputs/warped_lung_best/session_warping/test/images.csv",
        help="Path to test split images.csv (for test set identification)",
    )
    parser.add_argument(
        "--output",
        default="outputs/data_cleaning/cleaning_manifest.json",
        help="Output path for cleaning manifest JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load all quality signal CSVs
    print("Loading quality signals...")
    landmark_df = pd.read_csv(args.landmark_outliers)
    print(f"  Landmark outliers: {len(landmark_df)} rows, {landmark_df.flagged.sum()} flagged")

    cross_split_df = pd.read_csv(args.cross_split_exclusions)
    print(f"  Cross-split exclusions: {len(cross_split_df)} rows")

    cleanlab_df = pd.read_csv(args.cleanlab_issues)
    print(f"  Cleanlab issues: {len(cleanlab_df)} rows, {cleanlab_df.is_label_issue.sum()} flagged")

    # Build test set for protection
    test_images_df = pd.read_csv(args.test_images_csv)
    test_image_names = set(test_images_df["image_name"].values)
    print(f"  Test set images (protected): {len(test_image_names)}")

    # Build cross-split exclusion lookup (keyed on base image_name)
    cross_split_lookup = {}
    for _, row in cross_split_df.iterrows():
        cross_split_lookup[row["image_name"]] = {
            "reason": row["reason"],
            "duplicate_partner": row["duplicate_partner"],
            "duplicate_partner_category": row["duplicate_partner_category"],
        }

    # Build cleanlab lookup (keyed on base image_name, strip _warped.png)
    cleanlab_lookup = {}
    for _, row in cleanlab_df.iterrows():
        # Cleanlab uses warped filenames like "COVID-1001_warped.png"
        base_name = row["image_name"].replace("_warped.png", "")
        cleanlab_lookup[base_name] = {
            "self_confidence": float(row["self_confidence"]),
            "max_prob": float(row["max_prob"]),
            "is_label_issue": bool(row["is_label_issue"]),
            "predicted_label": int(row["predicted_label"]),
            "predicted_category": row["predicted_category"],
            "cleanlab_rank": float(row["cleanlab_rank"]) if pd.notna(row["cleanlab_rank"]) else None,
            "decision": row["decision"],
        }

    # Build per-image entries from landmark_outliers (has all 15,153 images)
    entries = []
    excluded_breakdown = defaultdict(int)
    review_pending_breakdown = defaultdict(int)
    by_class = defaultdict(lambda: {"excluded": 0, "review_pending": 0, "kept": 0})
    total_excluded = 0
    total_review_pending = 0
    total_kept = 0
    test_set_excluded = 0

    for _, row in landmark_df.iterrows():
        image_name = row["image_name"]
        category = row["category"]
        is_test = image_name in test_image_names

        # Gather flags
        procrustes_z = float(row["procrustes_z"])
        max_landmark_z = float(row["max_landmark_z"])
        is_landmark_outlier = bool(row["flagged"])

        # Cross-split duplicate info
        cs_info = cross_split_lookup.get(image_name, None)
        is_cross_split_duplicate = cs_info is not None
        duplicate_reason = cs_info["reason"] if cs_info else None
        duplicate_partner = cs_info["duplicate_partner"] if cs_info else None
        duplicate_partner_category = cs_info["duplicate_partner_category"] if cs_info else None

        # Cleanlab info (only for train+val, not test)
        cl_info = cleanlab_lookup.get(image_name, None)
        cleanlab_self_confidence = cl_info["self_confidence"] if cl_info else None
        is_cleanlab_issue = cl_info["is_label_issue"] if cl_info else None
        cleanlab_decision = cl_info["decision"] if cl_info else None
        predicted_category = cl_info["predicted_category"] if cl_info else None

        # Determine decision (priority order)
        reasons = []
        decision = "kept"

        if is_test:
            # Test set images are NEVER excluded
            decision = "kept"
        else:
            # 1. Cross-class duplicate → always excluded
            if is_cross_split_duplicate and duplicate_reason == "cross_class_duplicate":
                reasons.append("cross_class_duplicate")
                decision = "excluded"
                excluded_breakdown["cross_class_duplicate"] += 1

            # 2. Same-class duplicate → excluded
            if is_cross_split_duplicate and duplicate_reason == "same_class_duplicate":
                reasons.append("same_class_duplicate")
                if decision != "excluded":
                    decision = "excluded"
                    excluded_breakdown["cross_split_same_class"] += 1

            # 3. Landmark outlier → excluded
            if is_landmark_outlier:
                reasons.append("landmark_outlier")
                if decision != "excluded":
                    decision = "excluded"
                    excluded_breakdown["landmark_outlier"] += 1

            # 4. Cleanlab auto-excluded
            if cleanlab_decision == "auto_excluded":
                reasons.append("cleanlab_auto_excluded")
                if decision != "excluded":
                    decision = "excluded"
                    excluded_breakdown["cleanlab_auto_excluded"] += 1

            # 5. Cleanlab manual review
            if cleanlab_decision == "manual_review":
                reasons.append("cleanlab_manual_review")
                if decision == "kept":
                    decision = "review_pending"
                    review_pending_breakdown["cleanlab_manual_review"] += 1

        # Update counters
        if decision == "excluded":
            total_excluded += 1
            if is_test:
                test_set_excluded += 1
        elif decision == "review_pending":
            total_review_pending += 1
        else:
            total_kept += 1

        by_class[category][decision] += 1

        entry = {
            "image_name": image_name,
            "category": category,
            "split": "test" if is_test else "train_val",
            "decision": decision,
            "reasons": reasons,
            "flags": {
                "procrustes_z": procrustes_z,
                "max_landmark_z": max_landmark_z,
                "is_landmark_outlier": is_landmark_outlier,
                "cleanlab_self_confidence": cleanlab_self_confidence,
                "is_cleanlab_issue": is_cleanlab_issue,
                "cleanlab_decision": cleanlab_decision,
                "predicted_category": predicted_category,
                "is_cross_split_duplicate": is_cross_split_duplicate,
                "duplicate_partner": duplicate_partner,
                "duplicate_partner_category": duplicate_partner_category,
            },
            "review_notes": None,
        }
        entries.append(entry)

    # Validate
    total_images = len(entries)
    assert test_set_excluded == 0, f"TEST SET CONTAMINATION: {test_set_excluded} test images excluded!"
    assert total_excluded + total_review_pending + total_kept == total_images, (
        f"Count mismatch: {total_excluded} + {total_review_pending} + {total_kept} != {total_images}"
    )

    exclusion_rate = total_excluded / total_images
    if exclusion_rate > 0.15:
        print(f"\nWARNING: Exclusion rate {exclusion_rate:.1%} exceeds 15% threshold!")

    # Build summary
    summary = {
        "total_images_original": total_images,
        "total_flagged": total_excluded + total_review_pending,
        "total_excluded": total_excluded,
        "total_review_pending": total_review_pending,
        "total_kept": total_kept,
        "excluded_breakdown": dict(excluded_breakdown),
        "review_pending_breakdown": dict(review_pending_breakdown),
        "by_class": {k: dict(v) for k, v in by_class.items()},
        "test_set_excluded": test_set_excluded,
    }

    # Build manifest
    manifest = {
        "schema_version": "v1",
        "generated_at": datetime.now().isoformat(),
        "thresholds": {
            "landmark_procrustes_sigma": 3.0,
            "landmark_per_landmark_sigma": 4.0,
            "cleanlab_auto_exclude_self_confidence": 0.05,
            "cleanlab_manual_review_self_confidence": 0.40,
            "phash_threshold": 3,
        },
        "summary": summary,
        "entries": entries,
    }

    # Write manifest
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("CLEANING MANIFEST SUMMARY")
    print("=" * 60)
    print(f"Total images:      {total_images:,}")
    print(f"Excluded:          {total_excluded:,} ({total_excluded / total_images * 100:.1f}%)")
    print(f"Review pending:    {total_review_pending:,}")
    print(f"Kept:              {total_kept:,}")
    print(f"Test set excluded: {test_set_excluded} (MUST be 0)")
    print()
    print("Exclusion breakdown:")
    for reason, count in sorted(excluded_breakdown.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count:,}")
    if review_pending_breakdown:
        print("\nReview pending breakdown:")
        for reason, count in sorted(review_pending_breakdown.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count:,}")
    print()
    print("By class:")
    print(f"  {'Class':<20} {'Excluded':>10} {'Review':>10} {'Kept':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for cls in sorted(by_class.keys()):
        d = by_class[cls]
        print(f"  {cls:<20} {d['excluded']:>10,} {d.get('review_pending', 0):>10,} {d['kept']:>10,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
