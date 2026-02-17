"""
Landmark outlier detection for the data cleaning pipeline.

Detects images whose predicted landmarks deviate from the canonical shape using
robust statistics (median + MAD) to avoid contamination by outliers.

Flagging criterion (combined):
  - Overall Procrustes distance > 3 sigma (robust), OR
  - Any single landmark deviation > 4 sigma (robust)

Usage:
    python scripts/run_landmark_outlier_detection.py
    python scripts/run_landmark_outlier_detection.py --procrustes-sigma 3.0 --per-landmark-sigma 4.0
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_v2.processing.gpa import (
    center_shape,
    scale_shape,
    align_shape,
    procrustes_distance,
)


def compute_per_landmark_deviation(
    predicted: np.ndarray, canonical: np.ndarray
) -> np.ndarray:
    """
    Compute per-landmark Euclidean deviation after full Procrustes alignment.

    Both shapes are centered, scaled, and the predicted shape is aligned
    (rotated) to the canonical. The result is the Euclidean distance for
    each of the 15 landmarks.

    Args:
        predicted: (15, 2) predicted landmark coordinates in pixel space
        canonical: (15, 2) canonical shape (unit-normalised)

    Returns:
        deviations: (15,) Euclidean distance per landmark (in normalised units)
    """
    # Centre and scale predicted
    p_c, _ = center_shape(predicted)
    p_s, _ = scale_shape(p_c)

    # Centre and scale canonical
    c_c, _ = center_shape(canonical)
    c_s, _ = scale_shape(c_c)

    # Align predicted to canonical (rotation only, translation+scale already removed)
    p_aligned = align_shape(p_s, c_s)

    # Per-landmark Euclidean distance
    deviations = np.linalg.norm(p_aligned - c_s, axis=1)  # shape (15,)
    return deviations


def robust_sigma(values: np.ndarray) -> tuple:
    """
    Compute robust median and sigma-equivalent from MAD.

    Pitfall 4 mitigation: use median + k*MAD instead of mean + k*std so that
    outliers themselves do not inflate the threshold.

    Returns:
        median: float
        sigma_equiv: float  — 1.4826 * MAD  (consistent with Gaussian sigma)
    """
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    sigma_equiv = 1.4826 * mad
    return float(median), float(sigma_equiv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect landmark outliers using robust Procrustes-based statistics."
    )
    parser.add_argument(
        "--predictions",
        default="outputs/landmark_predictions/session_warping/predictions.npz",
        help="Path to NPZ file with landmark predictions.",
    )
    parser.add_argument(
        "--canonical-shape",
        default="outputs/shape_analysis/canonical_shape_gpa.json",
        help="Path to JSON with canonical shape.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/data_cleaning",
        help="Directory to write landmark_outliers.csv.",
    )
    parser.add_argument(
        "--procrustes-sigma",
        type=float,
        default=3.0,
        help="Sigma threshold for Procrustes distance outlier flag.",
    )
    parser.add_argument(
        "--per-landmark-sigma",
        type=float,
        default=4.0,
        help="Sigma threshold for per-landmark deviation outlier flag.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load predictions
    # ------------------------------------------------------------------
    print(f"Loading predictions from: {args.predictions}")
    data = np.load(args.predictions, allow_pickle=True)

    landmarks = data["landmarks"]      # (N, 15, 2)
    image_names = data["image_names"]  # (N,)
    categories = data["categories"]    # (N,)

    n_images = landmarks.shape[0]
    print(f"  Loaded {n_images} images with {landmarks.shape[1]} landmarks each.")

    # ------------------------------------------------------------------
    # 2. Load canonical shape
    # ------------------------------------------------------------------
    print(f"Loading canonical shape from: {args.canonical_shape}")
    with open(args.canonical_shape) as f:
        shape_data = json.load(f)

    # Use the unit-normalised version for Procrustes comparisons
    canonical = np.array(shape_data["canonical_shape_normalized"])  # (15, 2)
    print(f"  Canonical shape: {canonical.shape}")

    # ------------------------------------------------------------------
    # 3. Compute Procrustes distance and per-landmark deviation per image
    # ------------------------------------------------------------------
    print("Computing Procrustes distances and per-landmark deviations...")
    procrustes_distances = np.zeros(n_images)
    max_landmark_deviations = np.zeros(n_images)

    for i in range(n_images):
        pred = landmarks[i]  # (15, 2)
        procrustes_distances[i] = procrustes_distance(pred, canonical)
        per_lm_dev = compute_per_landmark_deviation(pred, canonical)
        max_landmark_deviations[i] = per_lm_dev.max()

    # ------------------------------------------------------------------
    # 4. Robust statistics (median + MAD)
    # ------------------------------------------------------------------
    proc_median, proc_sigma = robust_sigma(procrustes_distances)
    lm_median, lm_sigma = robust_sigma(max_landmark_deviations)

    print(f"\nRobust statistics (MAD-based):")
    print(f"  Procrustes distance — median: {proc_median:.6f}, sigma: {proc_sigma:.6f}")
    print(f"  Max landmark deviation — median: {lm_median:.6f}, sigma: {lm_sigma:.6f}")

    # Z-scores (signed deviations in robust-sigma units)
    if proc_sigma > 1e-12:
        procrustes_z = (procrustes_distances - proc_median) / proc_sigma
    else:
        procrustes_z = np.zeros(n_images)
        print("  WARNING: Procrustes sigma near zero — all z-scores set to 0.")

    if lm_sigma > 1e-12:
        max_landmark_z = (max_landmark_deviations - lm_median) / lm_sigma
    else:
        max_landmark_z = np.zeros(n_images)
        print("  WARNING: Per-landmark sigma near zero — all z-scores set to 0.")

    # ------------------------------------------------------------------
    # 5. Combined flagging
    # ------------------------------------------------------------------
    flag_procrustes = procrustes_z > args.procrustes_sigma
    flag_per_landmark = max_landmark_z > args.per_landmark_sigma
    flagged = flag_procrustes | flag_per_landmark

    # Human-readable flag reason
    flag_reason = []
    for fp, fl in zip(flag_procrustes, flag_per_landmark):
        if fp and fl:
            flag_reason.append("both")
        elif fp:
            flag_reason.append("procrustes_only")
        elif fl:
            flag_reason.append("per_landmark_only")
        else:
            flag_reason.append("")

    # ------------------------------------------------------------------
    # 6. Save CSV
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "landmark_outliers.csv")

    df = pd.DataFrame(
        {
            "image_name": image_names,
            "category": categories,
            "procrustes_distance": procrustes_distances,
            "max_landmark_deviation": max_landmark_deviations,
            "procrustes_z": procrustes_z,
            "max_landmark_z": max_landmark_z,
            "flagged": flagged,
            "flag_reason": flag_reason,
        }
    )
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    n_flagged = int(flagged.sum())
    n_proc_only = int((flag_procrustes & ~flag_per_landmark).sum())
    n_lm_only = int((~flag_procrustes & flag_per_landmark).sum())
    n_both = int((flag_procrustes & flag_per_landmark).sum())
    pct = n_flagged / n_images * 100

    print("\n=== Landmark Outlier Detection Summary ===")
    print(f"  Total images:             {n_images:>8,}")
    print(f"  Total flagged:            {n_flagged:>8,}  ({pct:.2f}%)")
    print(f"    - Procrustes only:      {n_proc_only:>8,}")
    print(f"    - Per-landmark only:    {n_lm_only:>8,}")
    print(f"    - Both criteria:        {n_both:>8,}")
    print(f"  Thresholds used:")
    print(f"    - Procrustes sigma:     {args.procrustes_sigma}")
    print(f"    - Per-landmark sigma:   {args.per_landmark_sigma}")
    print(f"  Robust sigma estimates (MAD-based):")
    print(f"    - Procrustes sigma_equiv:    {proc_sigma:.6f}")
    print(f"    - Per-landmark sigma_equiv:  {lm_sigma:.6f}")

    # Warn if flagged % is outside expected range (Pitfall 4 sanity check)
    if pct < 0.1:
        print(f"\n  WARNING: Only {pct:.2f}% flagged — threshold may be too permissive.")
    if pct > 10.0:
        print(f"\n  WARNING: {pct:.2f}% flagged — threshold may be too strict.")


if __name__ == "__main__":
    main()
