#!/usr/bin/env python3
"""Run image quality assessment (BRISQUE) on dataset.

Computes BRISQUE scores for all images and analyzes distribution:
- Overall quality statistics
- Per-class quality distribution
- Outlier detection (worst 10%)
- Optional: Compare error vs correct prediction quality

Outputs:
- all_images_quality.csv: BRISQUE scores for every image
- quality_distribution.png: Visualization of quality distribution
- quality_analysis_summary.json: Structured results for downstream analysis
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src_v2.evaluation.quality_assessment import (
    analyze_quality_distribution,
    compute_quality_scores,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_quality_distribution(scores_df: pd.DataFrame, output_path: Path, error_df=None):
    """Generate quality distribution visualizations."""
    # Set style
    sns.set_style("whitegrid")

    # Determine subplot layout
    if error_df is not None and not error_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes = list(axes) + [None]  # Pad to 3 for consistent indexing

    # Plot 1: Overall BRISQUE histogram
    ax1 = axes[0]
    scores_df["brisque_score"].hist(bins=50, ax=ax1, edgecolor="black")
    ax1.set_xlabel("BRISQUE Score")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Overall BRISQUE Distribution")
    ax1.axvline(
        scores_df["brisque_score"].median(),
        color="red",
        linestyle="--",
        label=f"Median: {scores_df['brisque_score'].median():.2f}",
    )
    ax1.legend()

    # Plot 2: Per-class box plots
    ax2 = axes[1]
    # Filter out unknown class
    plot_df = scores_df[scores_df["class"] != "unknown"]
    if not plot_df.empty:
        sns.boxplot(data=plot_df, x="class", y="brisque_score", ax=ax2)
        ax2.set_xlabel("Class")
        ax2.set_ylabel("BRISQUE Score")
        ax2.set_title("BRISQUE Distribution by Class")
        ax2.tick_params(axis='x', rotation=45)
    else:
        ax2.text(
            0.5,
            0.5,
            "No class information available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("BRISQUE Distribution by Class")

    # Plot 3: Error vs Correct comparison (if available)
    if error_df is not None and not error_df.empty and axes[2] is not None:
        ax3 = axes[2]

        # Create comparison data
        error_scores = error_df[error_df["is_error"]]["brisque_score"]
        correct_scores = error_df[~error_df["is_error"]]["brisque_score"]

        data = pd.DataFrame({
            "BRISQUE Score": pd.concat([error_scores, correct_scores]),
            "Type": (
                ["Error"] * len(error_scores) + ["Correct"] * len(correct_scores)
            ),
        })

        sns.boxplot(data=data, x="Type", y="BRISQUE Score", ax=ax3)
        ax3.set_title("Error vs Correct Prediction Quality")

        # Add mean values as text
        error_mean = error_scores.mean()
        correct_mean = correct_scores.mean()
        ax3.text(
            0.5,
            0.95,
            f"Error mean: {error_mean:.2f}, Correct mean: {correct_mean:.2f}",
            ha="center",
            transform=ax3.transAxes,
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved quality distribution plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run image quality assessment on dataset"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/dataset/COVID-19_Radiography_Dataset"),
        help="Dataset directory to analyze",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/error_forensics/quality_scores"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--error-results",
        type=Path,
        default=Path("outputs/error_forensics/error_analysis_results.json"),
        help="Optional: Error analysis results for error vs correct comparison",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.dataset_dir.exists():
        logger.error(f"Dataset not found: {args.dataset_dir}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Compute quality scores
    logger.info("=" * 80)
    logger.info("STAGE 1: Computing BRISQUE quality scores")
    logger.info("=" * 80)

    output_csv = args.output_dir / "all_images_quality.csv"
    scores_df = compute_quality_scores(
        str(args.dataset_dir),
        output_csv=str(output_csv),
        recursive=True,
    )

    if scores_df.empty:
        logger.error("No images processed!")
        return

    # Load error indices if available
    error_sample_indices = None
    if args.error_results.exists():
        try:
            with open(args.error_results) as f:
                error_data = json.load(f)

            # Extract filenames of misclassified images
            if "error_samples" in error_data:
                error_sample_indices = [
                    Path(sample["image_path"]).name
                    for sample in error_data["error_samples"]
                ]
                logger.info(f"Loaded {len(error_sample_indices)} error samples for comparison")
        except Exception as e:
            logger.warning(f"Failed to load error results: {e}")

    # Analyze quality distribution
    logger.info("=" * 80)
    logger.info("STAGE 2: Analyzing quality distribution")
    logger.info("=" * 80)

    analysis = analyze_quality_distribution(scores_df, error_sample_indices)

    # Generate visualizations
    logger.info("=" * 80)
    logger.info("STAGE 3: Generating visualizations")
    logger.info("=" * 80)

    plot_path = args.output_dir / "quality_distribution.png"

    # Prepare error comparison data if available
    error_df = None
    if error_sample_indices:
        scores_df["is_error"] = scores_df["filename"].isin(error_sample_indices)
        error_df = scores_df

    plot_quality_distribution(scores_df, plot_path, error_df)

    # Save summary JSON
    summary = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "total_images": len(scores_df),
            "dataset_dir": str(args.dataset_dir),
        },
        "overall": analysis["overall"],
        "per_class": analysis["per_class"],
        "outliers": {
            "count": analysis["outliers"]["count"],
            "threshold": analysis["outliers"]["threshold"],
            "top_10_worst": analysis["outliers"]["images"][:10],  # Save only top 10
        },
    }

    if analysis["error_comparison"]:
        summary["error_comparison"] = analysis["error_comparison"]

    summary_path = args.output_dir / "quality_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved summary to {summary_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("IMAGE QUALITY ASSESSMENT SUMMARY")
    print("=" * 80)
    print(f"\nTotal images analyzed: {len(scores_df)}")
    print(f"\nOverall Statistics:")
    print(f"  Mean BRISQUE: {analysis['overall']['mean']:.2f}")
    print(f"  Std Dev: {analysis['overall']['std']:.2f}")
    print(f"  Median: {analysis['overall']['median']:.2f}")
    print(f"  Range: [{analysis['overall']['min']:.2f}, {analysis['overall']['max']:.2f}]")
    print(f"  P10: {analysis['overall']['p10']:.2f} (best 10%)")
    print(f"  P90: {analysis['overall']['p90']:.2f} (worst 10%)")

    print(f"\nPer-Class Statistics:")
    for class_name, stats in analysis["per_class"].items():
        print(f"  {class_name}:")
        print(f"    Count: {stats['count']}")
        print(f"    Mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
        print(f"    Median: {stats['median']:.2f}")

    print(f"\nOutliers (worst 10%):")
    print(f"  Count: {analysis['outliers']['count']}")
    print(f"  Threshold: BRISQUE > {analysis['outliers']['threshold']:.2f}")

    if analysis["error_comparison"]:
        ec = analysis["error_comparison"]
        print(f"\nError vs Correct Comparison:")
        print(f"  Errors: {ec['errors_mean']:.2f} ± {ec['errors_std']:.2f} (n={ec['errors_count']})")
        print(f"  Correct: {ec['correct_mean']:.2f} ± {ec['correct_std']:.2f} (n={ec['correct_count']})")
        print(f"  t-statistic: {ec['t_statistic']:.2f}, p-value: {ec['p_value']:.4f}")
        print(f"  Significant: {'YES' if ec['significant'] else 'NO'} (p < 0.05)")
        print(f"  {ec['interpretation']}")
        print(f"  Error outlier rate: {ec['error_outlier_percentage']:.1f}%")

    print("=" * 80)


if __name__ == "__main__":
    main()
