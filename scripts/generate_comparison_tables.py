#!/usr/bin/env python3
"""
Script to generate LaTeX comparison tables for thesis Chapter 5.
Generates two tables:
1. Overall comparison: Baseline vs Ensemble+TTA
2. Per-class TTA impact breakdown
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def load_fold_results(cv_dir: Path) -> Dict:
    """
    Load test results from all 5 folds.

    Args:
        cv_dir: Directory with CV results

    Returns:
        Dict with aggregated metrics from baseline models
    """
    accuracies = []
    f1_macros = []
    f1_weighteds = []
    per_class_f1 = {"COVID": [], "Normal": [], "Viral_Pneumonia": []}

    for fold in range(1, 6):
        fold_path = cv_dir / f"fold_{fold:02d}" / "test_results.json"

        if not fold_path.exists():
            raise FileNotFoundError(f"Missing file: {fold_path}")

        with open(fold_path) as f:
            results = json.load(f)

        accuracies.append(results["metrics"]["accuracy"])
        f1_macros.append(results["metrics"]["f1_macro"])
        f1_weighteds.append(results["metrics"]["f1_weighted"])

        # Per-class F1
        for class_name in ["COVID", "Normal", "Viral_Pneumonia"]:
            per_class_f1[class_name].append(
                results["per_class"][class_name]["f1-score"]
            )

    return {
        "accuracy_mean": np.mean(accuracies),
        "accuracy_std": np.std(accuracies),
        "f1_macro_mean": np.mean(f1_macros),
        "f1_macro_std": np.std(f1_macros),
        "f1_weighted_mean": np.mean(f1_weighteds),
        "f1_weighted_std": np.std(f1_weighteds),
        "per_class_f1_mean": {
            class_name: np.mean(f1_list)
            for class_name, f1_list in per_class_f1.items()
        },
        "test_set_size": 1895,
        "n_folds": 5,
    }


def load_ensemble_results(cv_dir: Path, tta: bool = True) -> Dict:
    """
    Load ensemble test results (with or without TTA).

    Args:
        cv_dir: Directory with CV results
        tta: If True, load TTA results; otherwise no-TTA

    Returns:
        Dict with ensemble metrics
    """
    filename = "ensemble_test_results_tta.json" if tta else "ensemble_test_results_no_tta.json"
    ensemble_path = cv_dir / filename

    if not ensemble_path.exists():
        raise FileNotFoundError(f"Missing file: {ensemble_path}")

    with open(ensemble_path) as f:
        results = json.load(f)

    ensemble_metrics = results["ensemble_soft_voting"]["metrics"]
    per_class = results["ensemble_soft_voting"]["per_class"]

    output = {
        "accuracy": ensemble_metrics["accuracy"],
        "f1_macro": ensemble_metrics["f1_macro"],
        "f1_weighted": ensemble_metrics["f1_weighted"],
        "per_class_f1": {
            class_name: per_class[class_name]["f1-score"]
            for class_name in ["COVID", "Normal", "Viral_Pneumonia"]
        },
    }

    # Add case-level analysis if TTA
    if tta and "case_level_analysis" in results:
        output["case_level"] = results["case_level_analysis"]["summary"]
        output["net_improvement"] = results["case_level_analysis"]["net_improvement"]

    # Add TTA delta metrics if available
    if tta and "tta_delta_metrics" in results:
        output["tta_deltas"] = results["tta_delta_metrics"]

    return output


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format float as percentage with specified decimals."""
    return f"{value * 100:.{decimals}f}\\%"


def format_delta(delta: float, decimals: int = 2) -> str:
    """Format delta with sign and pp units."""
    sign = "+" if delta >= 0 else "$-$"
    abs_delta = abs(delta * 100)
    return f"{sign}{abs_delta:.{decimals}f} pp"


def classify_impact(delta: float) -> str:
    """Classify TTA impact based on delta magnitude."""
    if delta > 0.001:  # > 0.1pp
        return "Mejora"
    elif delta < -0.001:  # < -0.1pp
        return "Leve degradación"
    else:
        return "Neutro"


def generate_table1_overall_comparison(
    baseline: Dict,
    ensemble_tta: Dict,
) -> str:
    """
    Generate Table 1: Overall comparison (Baseline vs Ensemble+TTA).

    Args:
        baseline: Baseline metrics from 5 folds
        ensemble_tta: Ensemble+TTA metrics

    Returns:
        LaTeX table string
    """
    # Calculate deltas
    acc_delta = ensemble_tta["accuracy"] - baseline["accuracy_mean"]
    f1_macro_delta = ensemble_tta["f1_macro"] - baseline["f1_macro_mean"]
    f1_weighted_delta = ensemble_tta["f1_weighted"] - baseline["f1_weighted_mean"]

    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("    \\centering")
    latex.append("    \\caption{Comparación de rendimiento: Modelo individual (baseline) vs Ensemble con TTA.}")
    latex.append("    \\label{tab:comparison_ensemble_tta}")
    latex.append("    \\small")
    latex.append("    \\begin{tabular}{@{}lccc@{}}")
    latex.append("        \\toprule")
    latex.append("        \\textbf{Métrica} & \\textbf{Baseline Individual} & \\textbf{Ensemble + TTA} & \\textbf{Mejora} \\\\")
    latex.append("        \\midrule")

    # Accuracy row
    latex.append(
        f"        Exactitud (Accuracy) & "
        f"{format_percentage(baseline['accuracy_mean'])} $\\pm$ {format_percentage(baseline['accuracy_std'])} & "
        f"\\textbf{{{format_percentage(ensemble_tta['accuracy'])}}} & "
        f"{format_delta(acc_delta)} \\\\"
    )

    # F1-Macro row
    latex.append(
        f"        F1-Score Macro & "
        f"{format_percentage(baseline['f1_macro_mean'])} $\\pm$ {format_percentage(baseline['f1_macro_std'])} & "
        f"\\textbf{{{format_percentage(ensemble_tta['f1_macro'])}}} & "
        f"{format_delta(f1_macro_delta)} \\\\"
    )

    # F1-Weighted row
    latex.append(
        f"        F1-Score Ponderado & "
        f"{format_percentage(baseline['f1_weighted_mean'])} $\\pm$ {format_percentage(baseline['f1_weighted_std'])} & "
        f"\\textbf{{{format_percentage(ensemble_tta['f1_weighted'])}}} & "
        f"{format_delta(f1_weighted_delta)} \\\\"
    )

    latex.append("        \\midrule")

    # Sample counts
    test_size_formatted = f"{baseline['test_set_size']:,}".replace(",", "{,}")
    latex.append(
        f"        Muestras evaluadas & "
        f"{test_size_formatted} $\\times$ {baseline['n_folds']} & "
        f"{test_size_formatted} & "
        f"--- \\\\"
    )

    latex.append("        \\bottomrule")
    latex.append("    \\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def generate_table2_per_class_impact(
    baseline: Dict,
    ensemble_tta: Dict,
) -> str:
    """
    Generate Table 2: Per-class TTA impact.

    Args:
        baseline: Baseline metrics from 5 folds
        ensemble_tta: Ensemble+TTA metrics

    Returns:
        LaTeX table string
    """
    class_order = ["COVID", "Normal", "Viral_Pneumonia"]
    class_display = {
        "COVID": "COVID-19",
        "Normal": "Normal",
        "Viral_Pneumonia": "Neumonía Viral",
    }

    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("    \\centering")
    latex.append("    \\caption{Impacto del ensemble con TTA por categoría diagnóstica.}")
    latex.append("    \\label{tab:tta_per_class_impact}")
    latex.append("    \\small")
    latex.append("    \\begin{tabular}{@{}lcccc@{}}")
    latex.append("        \\toprule")
    latex.append("        \\textbf{Categoría} & \\textbf{F1 Baseline} & \\textbf{F1 Ensemble+TTA} & \\textbf{$\\Delta$ F1} & \\textbf{Impacto} \\\\")
    latex.append("        \\midrule")

    # Per-class rows
    for class_name in class_order:
        display_name = class_display[class_name]
        baseline_f1 = baseline["per_class_f1_mean"][class_name]
        ensemble_f1 = ensemble_tta["per_class_f1"][class_name]
        delta = ensemble_f1 - baseline_f1
        impact = classify_impact(delta)

        latex.append(
            f"        {display_name} & "
            f"{format_percentage(baseline_f1)} & "
            f"\\textbf{{{format_percentage(ensemble_f1)}}} & "
            f"{format_delta(delta)} & "
            f"{impact} \\\\"
        )

    latex.append("        \\midrule")

    # Case-level impact summary
    if "case_level" in ensemble_tta:
        helped = ensemble_tta["case_level"]["helped"]
        hurt = ensemble_tta["case_level"]["hurt"]
        neutral = ensemble_tta["case_level"]["neutral"]
        net = ensemble_tta["net_improvement"]

        latex.append(
            f"        \\multicolumn{{5}}{{l}}{{\\small Impacto a nivel de caso: "
            f"{helped} mejorados, {hurt} perjudicados, {neutral:,}".replace(",", "{,}") +
            f" neutrales (neto: {net:+d})}} \\\\"
        )

    latex.append("        \\bottomrule")
    latex.append("    \\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def generate_latex_file(
    baseline: Dict,
    ensemble_tta: Dict,
    output_path: Path,
):
    """
    Generate complete LaTeX file with both tables.

    Args:
        baseline: Baseline metrics
        ensemble_tta: Ensemble+TTA metrics
        output_path: Output .tex file path
    """
    latex = []

    # Header comments
    latex.append("% Auto-generated by scripts/generate_comparison_tables.py")
    latex.append("% Do not edit manually — regenerate with: python scripts/generate_comparison_tables.py")
    latex.append("")

    # Table 1: Overall comparison
    latex.append("% Table 1: Overall Comparison (Baseline vs Ensemble+TTA)")
    latex.append(generate_table1_overall_comparison(baseline, ensemble_tta))
    latex.append("")

    # Table 2: Per-class impact
    latex.append("% Table 2: Per-class TTA Impact")
    latex.append(generate_table2_per_class_impact(baseline, ensemble_tta))

    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX comparison tables for thesis."
    )
    parser.add_argument(
        "--cv-dir",
        type=Path,
        default=Path("outputs/classifier_cv"),
        help="Directory with CV results (default: outputs/classifier_cv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/classifier_cv/comparison_tables.tex"),
        help="Output .tex file path (default: outputs/classifier_cv/comparison_tables.tex)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve paths
    base_dir = Path(__file__).parent.parent
    cv_dir = base_dir / args.cv_dir if not args.cv_dir.is_absolute() else args.cv_dir
    output_path = base_dir / args.output if not args.output.is_absolute() else args.output

    print("=" * 70)
    print("GENERATING LATEX COMPARISON TABLES")
    print("=" * 70)
    print(f"\nCV directory: {cv_dir}")
    print(f"Output file: {output_path}")

    # Load data
    print("\n[1/3] Loading baseline results (5 folds)...")
    baseline = load_fold_results(cv_dir)
    print(f"  ✓ Baseline accuracy: {format_percentage(baseline['accuracy_mean'])} ± {format_percentage(baseline['accuracy_std'])}")
    print(f"  ✓ Baseline F1-macro: {format_percentage(baseline['f1_macro_mean'])} ± {format_percentage(baseline['f1_macro_std'])}")

    print("\n[2/3] Loading ensemble+TTA results...")
    ensemble_tta = load_ensemble_results(cv_dir, tta=True)
    print(f"  ✓ Ensemble+TTA accuracy: {format_percentage(ensemble_tta['accuracy'])}")
    print(f"  ✓ Ensemble+TTA F1-macro: {format_percentage(ensemble_tta['f1_macro'])}")

    print("\n[3/3] Generating LaTeX tables...")
    generate_latex_file(baseline, ensemble_tta, output_path)
    print(f"  ✓ LaTeX file written: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nBaseline (5-fold mean):")
    print(f"  Accuracy: {format_percentage(baseline['accuracy_mean'])} ± {format_percentage(baseline['accuracy_std'])}")
    print(f"  F1-Macro: {format_percentage(baseline['f1_macro_mean'])} ± {format_percentage(baseline['f1_macro_std'])}")
    print(f"  F1-Weighted: {format_percentage(baseline['f1_weighted_mean'])} ± {format_percentage(baseline['f1_weighted_std'])}")

    print(f"\nEnsemble + TTA:")
    print(f"  Accuracy: {format_percentage(ensemble_tta['accuracy'])}")
    print(f"  F1-Macro: {format_percentage(ensemble_tta['f1_macro'])}")
    print(f"  F1-Weighted: {format_percentage(ensemble_tta['f1_weighted'])}")

    print(f"\nImprovements:")
    print(f"  Accuracy: {format_delta(ensemble_tta['accuracy'] - baseline['accuracy_mean'])}")
    print(f"  F1-Macro: {format_delta(ensemble_tta['f1_macro'] - baseline['f1_macro_mean'])}")
    print(f"  F1-Weighted: {format_delta(ensemble_tta['f1_weighted'] - baseline['f1_weighted_mean'])}")

    print(f"\nPer-class F1 deltas:")
    for class_name in ["COVID", "Normal", "Viral_Pneumonia"]:
        delta = ensemble_tta["per_class_f1"][class_name] - baseline["per_class_f1_mean"][class_name]
        impact = classify_impact(delta)
        print(f"  {class_name}: {format_delta(delta)} ({impact})")

    if "case_level" in ensemble_tta:
        print(f"\nCase-level impact:")
        print(f"  Helped: {ensemble_tta['case_level']['helped']}")
        print(f"  Hurt: {ensemble_tta['case_level']['hurt']}")
        print(f"  Neutral: {ensemble_tta['case_level']['neutral']}")
        print(f"  Net: {ensemble_tta['net_improvement']:+d}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
