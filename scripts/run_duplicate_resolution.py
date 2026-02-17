"""
Cross-split duplicate resolution for the data cleaning pipeline.

Reverse-maps warped filenames from cross_split_leakage.csv back to original
image names using the authoritative images.csv mapping. Applies the locked
resolution strategy:

  - Same-class cross-split pairs: keep alphabetically first, exclude the other
  - Cross-class pairs (label ambiguity): exclude BOTH images

Output:
  outputs/data_cleaning/cross_split_exclusions.csv
  outputs/data_cleaning/duplicate_resolution_summary.json

Usage:
    python scripts/run_duplicate_resolution.py
"""

import argparse
import json
import os
import sys

import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_warped_to_original_map(splits_dir: str) -> dict:
    """
    Build a warped_path → (image_name, category, split) mapping from images.csv.

    The warped_path key matches the format used in cross_split_leakage.csv:
        "{split}/{category}/{warped_filename}"

    Args:
        splits_dir: Root directory containing {train,val,test}/images.csv

    Returns:
        mapping: dict[str, dict] with keys warped_path and values
                 {'image_name': str, 'category': str, 'split': str}
    """
    mapping = {}
    for split in ("train", "val", "test"):
        csv_path = os.path.join(splits_dir, split, "images.csv")
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found — skipping split '{split}'.")
            continue

        df = pd.read_csv(csv_path)
        required_cols = {"image_name", "category", "warped_filename"}
        if not required_cols.issubset(df.columns):
            print(f"  ERROR: {csv_path} missing columns {required_cols - set(df.columns)}")
            continue

        for _, row in df.iterrows():
            # Construct the same path format used in cross_split_leakage.csv
            warped_path = f"{split}/{row['category']}/{row['warped_filename']}"
            mapping[warped_path] = {
                "image_name": row["image_name"],
                "category": row["category"],
                "split": split,
            }

    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve cross-split duplicates and produce exclusion list."
    )
    parser.add_argument(
        "--leakage-csv",
        default="outputs/error_forensics/duplicates/cross_split_leakage.csv",
        help="Path to cross_split_leakage.csv from Phase 6.",
    )
    parser.add_argument(
        "--splits-dir",
        default="outputs/warped_lung_best/session_warping",
        help="Root directory containing train/val/test splits with images.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/data_cleaning",
        help="Directory to write outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load cross-split leakage CSV
    # ------------------------------------------------------------------
    print(f"Loading leakage CSV from: {args.leakage_csv}")
    leakage_df = pd.read_csv(args.leakage_csv)
    print(f"  Loaded {len(leakage_df):,} cross-split duplicate pairs.")
    print(f"  Columns: {leakage_df.columns.tolist()}")

    # ------------------------------------------------------------------
    # 2. Build authoritative warped-to-original mapping
    # ------------------------------------------------------------------
    print(f"\nBuilding warped-to-original mapping from: {args.splits_dir}")
    warp_map = build_warped_to_original_map(args.splits_dir)
    print(f"  Mapped {len(warp_map):,} warped paths to original image names.")

    # ------------------------------------------------------------------
    # 3. Validate: check for unmapped warped filenames (Pitfall 3 check)
    # ------------------------------------------------------------------
    all_warped_paths = set(leakage_df["img1"].tolist()) | set(leakage_df["img2"].tolist())
    unmapped = all_warped_paths - set(warp_map.keys())

    if unmapped:
        print(f"\n  WARNING: {len(unmapped)} warped paths in leakage CSV not found in images.csv:")
        for p in sorted(unmapped)[:10]:
            print(f"    - {p}")
        if len(unmapped) > 10:
            print(f"    ... and {len(unmapped) - 10} more.")
    else:
        print("  Validation passed: all warped paths mapped successfully.")

    # ------------------------------------------------------------------
    # 4. Resolve duplicates and build exclusion list
    # ------------------------------------------------------------------
    print("\nApplying resolution strategy...")

    exclusions: dict[str, dict] = {}  # image_name → exclusion record

    n_same_class = 0
    n_cross_class = 0
    n_unmapped_pairs = 0

    for _, row in leakage_df.iterrows():
        img1_path = row["img1"]
        img2_path = row["img2"]
        class1 = row["class1"]
        class2 = row["class2"]

        # Skip pairs where we can't reverse-map (already warned above)
        if img1_path not in warp_map or img2_path not in warp_map:
            n_unmapped_pairs += 1
            continue

        orig1 = warp_map[img1_path]["image_name"]
        orig2 = warp_map[img2_path]["image_name"]
        cat1 = warp_map[img1_path]["category"]
        cat2 = warp_map[img2_path]["category"]

        if class1 != class2:
            # Cross-class: label ambiguity → exclude both
            n_cross_class += 1
            for orig, cat, partner, partner_cat in [
                (orig1, cat1, orig2, cat2),
                (orig2, cat2, orig1, cat1),
            ]:
                if orig not in exclusions:
                    exclusions[orig] = {
                        "image_name": orig,
                        "category": cat,
                        "reason": "cross_class_duplicate",
                        "duplicate_partner": partner,
                        "duplicate_partner_category": partner_cat,
                    }
                # If already excluded for another reason, keep existing entry
        else:
            # Same-class: keep alphabetically first, exclude the other
            n_same_class += 1
            keep, excl = sorted([orig1, orig2])
            # The excluded image is the one that comes second alphabetically
            if excl == orig1:
                excl_cat, excl_partner, excl_partner_cat = cat1, orig2, cat2
            else:
                excl_cat, excl_partner, excl_partner_cat = cat2, orig1, cat1

            if excl not in exclusions:
                exclusions[excl] = {
                    "image_name": excl,
                    "category": excl_cat,
                    "reason": "same_class_duplicate",
                    "duplicate_partner": excl_partner,
                    "duplicate_partner_category": excl_partner_cat,
                }

    if n_unmapped_pairs > 0:
        print(f"  Skipped {n_unmapped_pairs:,} pairs with unmapped warped paths.")

    # ------------------------------------------------------------------
    # 5. Save cross_split_exclusions.csv
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    excl_df = pd.DataFrame(list(exclusions.values()))
    excl_df = excl_df.sort_values("image_name").reset_index(drop=True)

    out_excl = os.path.join(args.output_dir, "cross_split_exclusions.csv")
    excl_df.to_csv(out_excl, index=False)
    print(f"\nSaved: {out_excl}")

    # ------------------------------------------------------------------
    # 6. Save duplicate_resolution_summary.json
    # ------------------------------------------------------------------
    reason_counts = excl_df["reason"].value_counts().to_dict() if len(excl_df) > 0 else {}
    category_counts = excl_df["category"].value_counts().to_dict() if len(excl_df) > 0 else {}

    summary = {
        "total_pairs_processed": len(leakage_df),
        "same_class_pairs": n_same_class,
        "cross_class_pairs": n_cross_class,
        "unmapped_pairs_skipped": n_unmapped_pairs,
        "unique_images_excluded": len(exclusions),
        "exclusion_by_reason": reason_counts,
        "exclusion_by_category": category_counts,
    }

    out_summary = os.path.join(args.output_dir, "duplicate_resolution_summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_summary}")

    # ------------------------------------------------------------------
    # 7. Print summary
    # ------------------------------------------------------------------
    print("\n=== Cross-Split Duplicate Resolution Summary ===")
    print(f"  Total pairs processed:         {len(leakage_df):>8,}")
    print(f"    - Same-class pairs:          {n_same_class:>8,}")
    print(f"    - Cross-class pairs:         {n_cross_class:>8,}")
    print(f"    - Unmapped pairs skipped:    {n_unmapped_pairs:>8,}")
    print(f"  Unique images excluded:        {len(exclusions):>8,}")
    print(f"  Exclusions by reason:")
    for reason, count in reason_counts.items():
        print(f"    - {reason}: {count:,}")
    print(f"  Exclusions by category:")
    for cat, count in category_counts.items():
        print(f"    - {cat}: {count:,}")


if __name__ == "__main__":
    main()
