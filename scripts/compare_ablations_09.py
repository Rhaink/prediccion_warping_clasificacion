#!/usr/bin/env python3
"""Compare ablation results across Phase 9 augmentation experiments.

Produces:
- A printed comparison table to stdout with dual baseline deltas
- outputs/ablation_comparison_09.json with structured metrics

Phase 9 experiments:
  Individual augmentations: elastic, grid, pixel, mixup, cutmix
  Curriculum-combined:      elastic+curriculum, grid+curriculum, mixup+curriculum
"""

import json
import sys
from pathlib import Path

import numpy as np


EXPERIMENTS = {
    # Phase 8 baselines (for dual-baseline delta columns)
    "cleaned_baseline": "outputs/classifier_cv_cleaned_baseline/cross_validation_results.json",
    "curriculum": "outputs/classifier_cv_curriculum/cross_validation_results.json",
    # Phase 9 individual augmentations
    "elastic": "outputs/classifier_cv_aug_elastic/cross_validation_results.json",
    "grid": "outputs/classifier_cv_aug_grid/cross_validation_results.json",
    "pixel": "outputs/classifier_cv_aug_pixel/cross_validation_results.json",
    "mixup": "outputs/classifier_cv_aug_mixup/cross_validation_results.json",
    "cutmix": "outputs/classifier_cv_aug_cutmix/cross_validation_results.json",
    # Phase 9 curriculum-combined
    "elastic+curriculum": "outputs/classifier_cv_aug_elastic_curriculum/cross_validation_results.json",
    "grid+curriculum": "outputs/classifier_cv_aug_grid_curriculum/cross_validation_results.json",
    "mixup+curriculum": "outputs/classifier_cv_aug_mixup_curriculum/cross_validation_results.json",
}

# Baseline names for delta computation
CLEANED_BASELINE_KEY = "cleaned_baseline"
CURRICULUM_KEY = "curriculum"

CLASSES = ["COVID", "Normal", "Viral_Pneumonia"]


def extract_per_class_metrics(data: dict) -> dict:
    """Extract per-class precision, recall, F1 averaged across folds."""
    class_metrics = {cls: {"precision": [], "recall": [], "f1-score": []} for cls in CLASSES}

    for fold in data["fold_results"]:
        pcm = fold.get("val_metrics", {}).get("per_class_metrics", {})
        for cls in CLASSES:
            cls_data = pcm.get(cls, {})
            for metric in ["precision", "recall", "f1-score"]:
                if metric in cls_data:
                    class_metrics[cls][metric].append(cls_data[metric])

    result = {}
    for cls in CLASSES:
        result[cls] = {}
        for metric in ["precision", "recall", "f1-score"]:
            vals = class_metrics[cls][metric]
            if vals:
                arr = np.array(vals)
                result[cls][metric] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "per_fold": [float(v) for v in arr],
                }
    return result


def main():
    results = {}
    missing = []

    for name, path in EXPERIMENTS.items():
        if not Path(path).exists():
            missing.append(name)
            continue

        with open(path) as f:
            data = json.load(f)

        summary = data["summary"]
        per_class = extract_per_class_metrics(data)

        results[name] = {
            "val_accuracy_mean": summary["val_accuracy_mean"],
            "val_accuracy_std": summary["val_accuracy_std"],
            "val_f1_macro_mean": summary["val_f1_macro_mean"],
            "val_f1_macro_std": summary["val_f1_macro_std"],
            "val_f1_weighted_mean": summary.get("val_f1_weighted_mean", None),
            "val_f1_weighted_std": summary.get("val_f1_weighted_std", None),
            "per_class": per_class,
            "n_folds": len(data["fold_results"]),
        }

    if missing:
        print(f"WARNING: Missing experiments: {missing}", file=sys.stderr)

    # Find best by val_f1_macro_mean
    if not results:
        print("ERROR: No experiment results found.", file=sys.stderr)
        sys.exit(1)

    best_name = max(results, key=lambda n: results[n]["val_f1_macro_mean"])
    baseline_f1 = results.get(CLEANED_BASELINE_KEY, {}).get("val_f1_macro_mean", 0.0)
    curriculum_f1 = results.get(CURRICULUM_KEY, {}).get("val_f1_macro_mean", 0.0)

    # -------------------------------------------------------------------------
    # Main comparison table
    # -------------------------------------------------------------------------
    print("=" * 92)
    print("Phase 9 Augmentation Ablation Comparison")
    print("=" * 92)
    print()
    hdr = (
        f"{'Experiment':<22} {'Val F1-Macro':>16} {'VP Recall':>16}"
        f" {'vs Cleaned':>12} {'vs Curriculum':>14}"
    )
    print(hdr)
    print("-" * 92)

    # Print a horizontal separator between sections
    section_markers = {
        "elastic": "--- Phase 9: Individual Augmentations ---",
        "elastic+curriculum": "--- Phase 9: Curriculum-Combined ---",
    }

    for name, r in results.items():
        # Print section header if needed
        if name in section_markers:
            print(f"\n  {section_markers[name]}")

        vp = r["per_class"].get("Viral_Pneumonia", {}).get("recall", {})
        vp_str = (
            f"{vp.get('mean', 0) * 100:5.2f} +/- {vp.get('std', 0) * 100:4.2f}"
            if vp
            else "N/A"
        )

        delta_cleaned = r["val_f1_macro_mean"] - baseline_f1
        delta_curriculum = r["val_f1_macro_mean"] - curriculum_f1

        if name == CLEANED_BASELINE_KEY:
            delta_cleaned_str = "--"
            delta_curriculum_str = f"{delta_curriculum * 100:+6.2f}pp"
        elif name == CURRICULUM_KEY:
            delta_cleaned_str = f"{delta_cleaned * 100:+6.2f}pp"
            delta_curriculum_str = "--"
        else:
            delta_cleaned_str = f"{delta_cleaned * 100:+6.2f}pp"
            delta_curriculum_str = f"{delta_curriculum * 100:+6.2f}pp"

        marker = " *" if name == best_name else ""

        print(
            f"{name:<22} "
            f"{r['val_f1_macro_mean'] * 100:5.2f} +/- {r['val_f1_macro_std'] * 100:4.2f}  "
            f"{vp_str:>16s}  "
            f"{delta_cleaned_str:>10s}  "
            f"{delta_curriculum_str:>12s}{marker}"
        )

    print("-" * 92)
    print(
        f"\nBest configuration: {best_name} "
        f"(val_f1_macro_mean={results[best_name]['val_f1_macro_mean']:.4f})"
    )

    # Check if any augmentation+curriculum beats curriculum alone
    aug_curriculum_experiments = [
        n for n in results
        if n not in (CLEANED_BASELINE_KEY, CURRICULUM_KEY)
        and results[n]["val_f1_macro_mean"] > curriculum_f1
    ]
    augmentation_improved_over_curriculum = len(aug_curriculum_experiments) > 0

    if augmentation_improved_over_curriculum:
        best_aug = max(aug_curriculum_experiments, key=lambda n: results[n]["val_f1_macro_mean"])
        improvement_pp = (results[best_aug]["val_f1_macro_mean"] - curriculum_f1) * 100
        print(
            f"Augmentation improved over curriculum: YES "
            f"({best_aug} +{improvement_pp:.2f}pp over curriculum)"
        )
    else:
        print("Augmentation improved over curriculum: NO")

    # -------------------------------------------------------------------------
    # Per-class recall detail table
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("Per-Class Recall Detail")
    print(f"{'=' * 78}")
    print(f"{'Experiment':<22} {'COVID':>16} {'Normal':>16} {'VP':>16}")
    print("-" * 72)

    for name, r in results.items():
        if name in section_markers:
            print(f"\n  {section_markers[name]}")
        cols = []
        for cls in CLASSES:
            rec = r["per_class"].get(cls, {}).get("recall", {})
            if rec:
                cols.append(f"{rec['mean'] * 100:5.2f} +/- {rec['std'] * 100:4.2f}")
            else:
                cols.append("N/A")
        print(f"{name:<22} {cols[0]:>16s} {cols[1]:>16s} {cols[2]:>16s}")

    # -------------------------------------------------------------------------
    # Brief interpretation
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("Interpretation")
    print(f"{'=' * 78}")

    individual_keys = ["elastic", "grid", "pixel", "mixup", "cutmix"]
    curriculum_combo_keys = ["elastic+curriculum", "grid+curriculum", "mixup+curriculum"]

    aug_helped = [n for n in individual_keys if n in results and
                  results[n]["val_f1_macro_mean"] > baseline_f1]
    aug_hurt = [n for n in individual_keys if n in results and
                results[n]["val_f1_macro_mean"] < baseline_f1]
    aug_neutral = [n for n in individual_keys if n in results and
                   n not in aug_helped and n not in aug_hurt]

    curriculum_improved = [n for n in curriculum_combo_keys if n in results and
                            results[n]["val_f1_macro_mean"] > curriculum_f1]
    curriculum_regressed = [n for n in curriculum_combo_keys if n in results and
                             results[n]["val_f1_macro_mean"] < curriculum_f1]

    print(f"Individual augmentations vs cleaned baseline ({baseline_f1 * 100:.2f}%):")
    print(f"  Helped:   {aug_helped if aug_helped else 'none'}")
    print(f"  Hurt:     {aug_hurt if aug_hurt else 'none'}")
    print(f"  Neutral:  {aug_neutral if aug_neutral else 'none'}")
    print()
    print(f"Curriculum-combined augmentations vs curriculum ({curriculum_f1 * 100:.2f}%):")
    print(f"  Improved: {curriculum_improved if curriculum_improved else 'none'}")
    print(f"  Regressed:{curriculum_regressed if curriculum_regressed else 'none'}")
    print()

    # Recommendation for Phase 10
    print("Recommendation for Phase 10:")
    best_f1 = results[best_name]["val_f1_macro_mean"]
    print(f"  Use '{best_name}' configuration (val_f1_macro={best_f1:.4f})")
    print(f"  Output dir: outputs/classifier_cv_aug_{best_name.replace('+', '_').replace('-', '_')}/")

    # -------------------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------------------
    output = {
        "phase": 9,
        "experiments": results,
        "best_experiment": best_name,
        "best_f1_macro": results[best_name]["val_f1_macro_mean"],
        "baselines": {
            "cleaned": {
                "f1": baseline_f1,
                "key": CLEANED_BASELINE_KEY,
            },
            "curriculum": {
                "f1": curriculum_f1,
                "key": CURRICULUM_KEY,
            },
        },
        "augmentation_improved_over_curriculum": augmentation_improved_over_curriculum,
        "aug_experiments_beating_curriculum": aug_curriculum_experiments,
        "missing_experiments": missing,
    }

    output_path = Path("outputs/ablation_comparison_09.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
