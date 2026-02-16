"""Image quality assessment for medical imaging datasets.

This module provides BRISQUE-based quality scoring and analysis:
- BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
- Per-class quality distribution analysis
- Error vs correct prediction quality comparison
- Outlier detection

Lower BRISQUE scores indicate better quality (0 = perfect, 100 = worst).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from imquality import brisque
from scipy import stats
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_quality_scores(
    image_dir: str,
    output_csv: Optional[str] = None,
    recursive: bool = True,
) -> pd.DataFrame:
    """Compute BRISQUE quality scores for all images in a directory.

    Args:
        image_dir: Root directory containing images
        output_csv: Optional path to save results CSV
        recursive: Whether to search subdirectories (default: True)

    Returns:
        DataFrame with columns: filename, class, split, brisque_score, path

    Notes:
        - BRISQUE expects RGB images (grayscale converted automatically)
        - Lower scores = better quality (0 = perfect, 100 = worst)
        - Typical range for natural images: 20-80
        - Medical X-rays may have different ranges due to specialized characteristics
        - Processing time: ~0.5-1 sec per image on CPU
    """
    image_path = Path(image_dir)
    if not image_path.exists():
        raise ValueError(f"Image directory does not exist: {image_dir}")

    # Collect all image files
    image_files = []
    if recursive:
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
            image_files.extend(image_path.rglob(ext))
    else:
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
            image_files.extend(image_path.glob(ext))

    total_images = len(image_files)
    logger.info(f"Found {total_images} images in {image_dir}")

    if total_images == 0:
        logger.warning("No images found!")
        return pd.DataFrame(columns=["filename", "class", "split", "brisque_score", "path"])

    results = []

    for img_path in tqdm(image_files, desc="Computing BRISQUE scores"):
        try:
            # Load image
            img = Image.open(img_path)

            # Convert grayscale to RGB (BRISQUE expects 3 channels)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Compute BRISQUE score (using imquality library)
            score = brisque.score(img)

            # Extract metadata from path structure
            # Expected structure: .../class/image.png or .../split/class/image.png
            parts = img_path.parts
            filename = img_path.name

            # Try to infer class and split
            class_name = None
            split_name = None

            if len(parts) >= 2:
                # Check if parent is a class name
                parent = parts[-2]
                if parent in ["COVID", "Normal", "Viral_Pneumonia"]:
                    class_name = parent

                    # Check if grandparent is a split name
                    if len(parts) >= 3:
                        grandparent = parts[-3]
                        if grandparent in ["train", "val", "test"]:
                            split_name = grandparent

            results.append({
                "filename": filename,
                "class": class_name if class_name else "unknown",
                "split": split_name if split_name else "unknown",
                "brisque_score": float(score),
                "path": str(img_path),
            })

        except Exception as e:
            logger.warning(f"Failed to process {img_path}: {e}")
            continue

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save if requested
    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved quality scores to {output_csv}")

    logger.info(f"Successfully processed {len(df)} images")
    return df


def analyze_quality_distribution(
    scores_df: pd.DataFrame,
    error_sample_indices: Optional[List[str]] = None,
) -> Dict:
    """Analyze quality score distribution with per-class statistics.

    Args:
        scores_df: DataFrame from compute_quality_scores()
        error_sample_indices: Optional list of filenames that were misclassified

    Returns:
        Dict with keys:
            - overall: Overall statistics (mean, std, median, p10, p90)
            - per_class: Per-class statistics
            - outliers: High-BRISQUE outliers (worst 10%)
            - error_comparison: Quality comparison if error_sample_indices provided

    Notes:
        - Outliers defined as BRISQUE > P90 (worst 10% quality)
        - Error comparison uses independent t-test (assumes normal distribution)
        - Returns p-value for significance testing (p < 0.05 = significant)
    """
    if scores_df.empty:
        logger.warning("Empty scores DataFrame")
        return {
            "overall": {},
            "per_class": {},
            "outliers": {"count": 0, "threshold": None, "images": []},
            "error_comparison": None,
        }

    # Overall statistics
    overall_scores = scores_df["brisque_score"]
    overall_stats = {
        "mean": float(overall_scores.mean()),
        "std": float(overall_scores.std()),
        "median": float(overall_scores.median()),
        "p10": float(overall_scores.quantile(0.10)),
        "p90": float(overall_scores.quantile(0.90)),
        "min": float(overall_scores.min()),
        "max": float(overall_scores.max()),
    }

    # Per-class statistics
    per_class_stats = {}
    for class_name in scores_df["class"].unique():
        if class_name == "unknown":
            continue

        class_scores = scores_df[scores_df["class"] == class_name]["brisque_score"]
        per_class_stats[class_name] = {
            "mean": float(class_scores.mean()),
            "std": float(class_scores.std()),
            "median": float(class_scores.median()),
            "p10": float(class_scores.quantile(0.10)),
            "p90": float(class_scores.quantile(0.90)),
            "count": len(class_scores),
        }

    # Outlier detection (worst 10%)
    outlier_threshold = overall_stats["p90"]
    outliers_df = scores_df[scores_df["brisque_score"] > outlier_threshold]
    outliers_sorted = outliers_df.sort_values("brisque_score", ascending=False)

    outliers = {
        "count": len(outliers_df),
        "threshold": outlier_threshold,
        "images": outliers_sorted[["filename", "class", "split", "brisque_score"]].to_dict(
            "records"
        ),
    }

    # Error vs correct comparison (if error indices provided)
    error_comparison = None
    if error_sample_indices is not None and len(error_sample_indices) > 0:
        # Mark errors in DataFrame
        scores_df["is_error"] = scores_df["filename"].isin(error_sample_indices)

        error_scores = scores_df[scores_df["is_error"]]["brisque_score"]
        correct_scores = scores_df[~scores_df["is_error"]]["brisque_score"]

        if len(error_scores) > 0 and len(correct_scores) > 0:
            # Independent t-test
            t_stat, p_value = stats.ttest_ind(error_scores, correct_scores)

            error_comparison = {
                "errors_mean": float(error_scores.mean()),
                "errors_std": float(error_scores.std()),
                "errors_count": len(error_scores),
                "correct_mean": float(correct_scores.mean()),
                "correct_std": float(correct_scores.std()),
                "correct_count": len(correct_scores),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
                "interpretation": (
                    "Errors have significantly different quality than correct predictions"
                    if p_value < 0.05
                    else "No significant quality difference between errors and correct predictions"
                ),
            }

            # Additional analysis: what percentage of errors are outliers?
            error_outliers = scores_df[scores_df["is_error"] & (
                scores_df["brisque_score"] > outlier_threshold
            )]
            error_comparison["error_outlier_percentage"] = round(
                100 * len(error_outliers) / len(error_scores), 2
            )

    return {
        "overall": overall_stats,
        "per_class": per_class_stats,
        "outliers": outliers,
        "error_comparison": error_comparison,
    }
