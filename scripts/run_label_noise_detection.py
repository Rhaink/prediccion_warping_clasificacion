"""
Cleanlab Label Noise Detection on Out-of-Fold Probabilities.

Uses cleanlab confident learning to identify potential label issues in the
train+val dataset. Categorizes issues into auto_excluded / manual_review /
flagged_kept / kept based on self-confidence thresholds.

Usage:
    python scripts/run_label_noise_detection.py
    python scripts/run_label_noise_detection.py --auto-exclude-threshold 0.05
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cleanlab label noise detection on OOF probabilities"
    )
    parser.add_argument(
        "--oof-path",
        type=Path,
        default=Path("outputs/data_cleaning/oof_probabilities.npz"),
        help="Path to OOF probabilities npz file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/data_cleaning"),
        help="Output directory for cleanlab issues CSV",
    )
    parser.add_argument(
        "--auto-exclude-threshold",
        type=float,
        default=0.05,
        help="Self-confidence below this threshold triggers auto_excluded decision",
    )
    parser.add_argument(
        "--manual-review-threshold",
        type=float,
        default=0.40,
        help="Self-confidence below this threshold (and >= auto-exclude) triggers manual_review",
    )
    return parser.parse_args()


def print_text_histogram(values, n_bins=10, label=""):
    """Print a simple text-based histogram for a 1D array."""
    if len(values) == 0:
        print(f"  {label}: (empty)")
        return
    min_v, max_v = values.min(), values.max()
    bin_edges = np.linspace(min_v, max_v + 1e-9, n_bins + 1)
    counts, _ = np.histogram(values, bins=bin_edges)
    max_count = counts.max() if counts.max() > 0 else 1
    bar_width = 30

    print(f"  {label} distribution (n={len(values)}):")
    for i in range(n_bins):
        bar_len = int(counts[i] / max_count * bar_width)
        print(
            f"    [{bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}] "
            f"{'#' * bar_len:30s} {counts[i]}"
        )


def main():
    args = parse_args()

    # Step 1: Load OOF probabilities
    print("=== Step 1: Loading OOF probabilities ===")
    if not args.oof_path.exists():
        raise FileNotFoundError(
            f"OOF probabilities not found: {args.oof_path}. "
            "Run scripts/run_oof_extraction.py first."
        )

    oof = np.load(args.oof_path, allow_pickle=True)
    pred_probs = oof["pred_probs"].astype(np.float64)   # cleanlab needs float64
    true_labels = oof["true_labels"].astype(np.int32)
    image_names = oof["image_names"]
    class_names = list(oof["class_names"])
    temperature = float(oof["temperature"])

    n_samples = len(true_labels)
    n_classes = pred_probs.shape[1]

    print(f"  OOF samples: {n_samples}")
    print(f"  Classes: {class_names}")
    print(f"  Temperature used: {temperature}")
    print(f"  pred_probs shape: {pred_probs.shape}")

    # Ensure probabilities sum to 1 (numerical precision)
    pred_probs = pred_probs / pred_probs.sum(axis=1, keepdims=True)

    # Step 2: Run cleanlab confident learning
    print("\n=== Step 2: Running cleanlab confident learning ===")
    from cleanlab.filter import find_label_issues

    issue_indices = find_label_issues(
        labels=true_labels,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
        filter_by="confident_learning",
    )

    n_issues = len(issue_indices)
    print(f"  Label issues detected: {n_issues} / {n_samples} ({n_issues/n_samples:.2%})")

    if n_issues == 0:
        print("  WARNING: Cleanlab detected 0 label issues.")
        print("  This may indicate overconfident predictions bypassing detection.")
        print("  Consider increasing temperature scaling or using a lower threshold.")

    # Step 3: Compute per-sample self-confidence
    print("\n=== Step 3: Computing self-confidence scores ===")
    # self_confidence = model's probability for the given (potentially noisy) label
    self_confidence = pred_probs[np.arange(n_samples), true_labels]
    max_prob = pred_probs.max(axis=1)
    predicted_label = pred_probs.argmax(axis=1)

    # Step 4: Build is_label_issue boolean array
    is_label_issue = np.zeros(n_samples, dtype=bool)
    is_label_issue[issue_indices] = True

    # Cleanlab rank: rank among label issues (lower = more likely issue), NaN for non-issues
    cleanlab_rank = np.full(n_samples, np.nan)
    for rank, idx in enumerate(issue_indices):
        cleanlab_rank[idx] = rank

    # Step 5: Apply decision thresholds
    print("\n=== Step 4: Applying decision thresholds ===")
    print(f"  auto_excluded threshold: self_confidence < {args.auto_exclude_threshold}")
    print(f"  manual_review threshold: {args.auto_exclude_threshold} <= self_confidence < {args.manual_review_threshold}")
    print(f"  flagged_kept: self_confidence >= {args.manual_review_threshold} but still flagged by cleanlab")
    print(f"  kept: not flagged by cleanlab")

    decisions = np.full(n_samples, "kept", dtype=object)
    for idx in issue_indices:
        sc = self_confidence[idx]
        if sc < args.auto_exclude_threshold:
            decisions[idx] = "auto_excluded"
        elif sc < args.manual_review_threshold:
            decisions[idx] = "manual_review"
        else:
            decisions[idx] = "flagged_kept"

    # Step 6: Build output DataFrame
    print("\n=== Step 5: Building output CSV ===")
    idx_to_name = {i: name for i, name in enumerate(class_names)}

    df = pd.DataFrame({
        "image_name": image_names,
        "category": [idx_to_name[lbl] for lbl in true_labels],
        "true_label": true_labels,
        "predicted_label": predicted_label,
        "predicted_category": [idx_to_name[lbl] for lbl in predicted_label],
        "self_confidence": self_confidence,
        "max_prob": max_prob,
        "is_label_issue": is_label_issue,
        "cleanlab_rank": cleanlab_rank,
        "decision": decisions,
    })

    # Step 7: Save CSV
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "cleanlab_issues.csv"
    df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv} ({len(df)} rows)")

    # Step 8: Print summary
    print("\n=== Summary ===")
    print(f"Total samples: {n_samples}")
    print(f"Label issues detected by cleanlab: {n_issues} ({n_issues/n_samples:.2%})")
    print()

    print("Decision breakdown:")
    decision_counts = df["decision"].value_counts()
    for dec, cnt in decision_counts.items():
        pct = cnt / n_samples * 100
        print(f"  {dec:20s}: {cnt:5d} ({pct:.2f}%)")
    print()

    print("Label issues by class (true label):")
    issues_df = df[df["is_label_issue"]]
    if len(issues_df) > 0:
        class_counts = issues_df["category"].value_counts()
        for cls, cnt in class_counts.items():
            print(f"  {cls:20s}: {cnt}")
    else:
        print("  (no issues detected)")
    print()

    print("Label issues by true -> predicted transition:")
    if len(issues_df) > 0:
        transition_counts = (
            issues_df.groupby(["category", "predicted_category"]).size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        for _, row in transition_counts.iterrows():
            print(f"  {row['category']:20s} -> {row['predicted_category']:20s}: {row['count']}")
    else:
        print("  (no issues detected)")
    print()

    print("Top 10 most likely label errors:")
    if len(issues_df) > 0:
        top10 = issues_df.nsmallest(min(10, len(issues_df)), "self_confidence")
        for _, r in top10.iterrows():
            print(
                f"  {r['image_name']:40s}  "
                f"{r['category']:15s} -> {r['predicted_category']:15s}  "
                f"(self_conf={r['self_confidence']:.4f}, max_prob={r['max_prob']:.4f})"
            )
    else:
        print("  (no issues detected)")
    print()

    print("Self-confidence distribution for label issues:")
    if len(issues_df) > 0:
        print_text_histogram(issues_df["self_confidence"].values, n_bins=10, label="self_confidence")
    else:
        print("  (no issues detected)")

    print("\n=== Label Noise Detection Complete ===")
    print(f"Output CSV: {output_csv}")


if __name__ == "__main__":
    main()
