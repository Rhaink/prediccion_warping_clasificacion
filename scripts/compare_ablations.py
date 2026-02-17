#!/usr/bin/env python3
"""Compare ablation results across Phase 8 training experiments.

Produces:
- A printed comparison table to stdout
- outputs/ablation_comparison.json with structured metrics
"""

import json
import sys
from pathlib import Path

import numpy as np


EXPERIMENTS = {
    "cleaned_baseline": "outputs/classifier_cv_cleaned_baseline/cross_validation_results.json",
    "focal": "outputs/classifier_cv_focal/cross_validation_results.json",
    "mining": "outputs/classifier_cv_mining/cross_validation_results.json",
    "curriculum": "outputs/classifier_cv_curriculum/cross_validation_results.json",
    "combined": "outputs/classifier_cv_combined/cross_validation_results.json",
}

# v1.0 reference from GROUND_TRUTH.json
V1_REF = {
    "val_accuracy_mean": 0.9860,
    "val_accuracy_std": 0.0026,
    "vp_recall_test": 0.929,  # test set, not validation
}

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
    best_name = max(results, key=lambda n: results[n]["val_f1_macro_mean"])
    baseline_f1 = results.get("cleaned_baseline", {}).get("val_f1_macro_mean", 0)

    # Print comparison table
    print("=" * 80)
    print("Phase 8 Ablation Comparison")
    print("=" * 80)
    print()
    print(f"{'Experiment':<20} {'Val Acc':>15} {'Val F1-Macro':>15} {'VP Recall':>15} {'Delta vs BL':>12}")
    print("-" * 80)

    # v1.0 reference
    print(
        f"{'v1.0 (ref)':20s} "
        f"{V1_REF['val_accuracy_mean']*100:5.2f} +/- {V1_REF['val_accuracy_std']*100:4.2f}  "
        f"{'N/A':>15s} "
        f"{V1_REF['vp_recall_test']*100:5.2f} (test)  "
        f"{'--':>10s}"
    )

    for name, r in results.items():
        vp = r["per_class"].get("Viral_Pneumonia", {}).get("recall", {})
        vp_str = f"{vp.get('mean', 0)*100:5.2f} +/- {vp.get('std', 0)*100:4.2f}" if vp else "N/A"
        delta = r["val_f1_macro_mean"] - baseline_f1
        delta_str = f"{delta*100:+5.2f}pp" if name != "cleaned_baseline" else "--"
        marker = " *" if name == best_name else ""

        print(
            f"{name:20s} "
            f"{r['val_accuracy_mean']*100:5.2f} +/- {r['val_accuracy_std']*100:4.2f}  "
            f"{r['val_f1_macro_mean']*100:5.2f} +/- {r['val_f1_macro_std']*100:4.2f}  "
            f"{vp_str:>15s}  "
            f"{delta_str:>10s}{marker}"
        )

    print("-" * 80)
    print(f"\nBest configuration: {best_name} (val_f1_macro_mean={results[best_name]['val_f1_macro_mean']:.4f})")

    # Check VP recall target
    vp_target = 0.929
    vp_target_met = False
    for name, r in results.items():
        vp_recall = r["per_class"].get("Viral_Pneumonia", {}).get("recall", {}).get("mean", 0)
        if vp_recall > vp_target:
            vp_target_met = True

    print(f"VP recall target (>{vp_target*100:.1f}%): {'MET' if vp_target_met else 'NOT MET'}")

    # Per-class detail table
    print(f"\n{'=' * 80}")
    print("Per-Class Recall Detail")
    print(f"{'=' * 80}")
    print(f"{'Experiment':<20} {'COVID':>15} {'Normal':>15} {'VP':>15}")
    print("-" * 68)
    for name, r in results.items():
        cols = []
        for cls in CLASSES:
            rec = r["per_class"].get(cls, {}).get("recall", {})
            if rec:
                cols.append(f"{rec['mean']*100:5.2f} +/- {rec['std']*100:4.2f}")
            else:
                cols.append("N/A")
        print(f"{name:20s} {cols[0]:>15s} {cols[1]:>15s} {cols[2]:>15s}")

    # Save JSON
    output = {
        "experiments": results,
        "best_experiment": best_name,
        "best_f1_macro": results[best_name]["val_f1_macro_mean"],
        "vp_recall_target": vp_target,
        "vp_recall_target_met": vp_target_met,
        "v1_reference": V1_REF,
    }

    output_path = Path("outputs/ablation_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
