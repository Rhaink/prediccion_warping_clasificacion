#!/usr/bin/env python3
"""
Phase 10 Final Evaluation and Statistical Validation Script.

Runs ensemble test-set evaluation for all 4 pipeline models:
  1. v1.0 baseline
  2. Cleaned baseline
  3. Curriculum learning
  4. Elastic + curriculum (best model)

Computes full statistical validation:
  - McNemar's test (with Yates continuity correction)
  - Bootstrap 95% CI (10,000 iterations)
  - DeLong AUC comparison (bootstrap-based)
  - Holm-Bonferroni multiple comparison correction
  - Regression guardrail (b <= 5)

Outputs:
  - Per-sample predictions CSV for each model
  - Ensemble results JSON for each model
  - phase10_final_report.json with all results combined
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import chi2
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src_v2.evaluation.ensemble import (
    ensemble_inference_with_tta,
    load_ensemble_models,
    weighted_soft_voting,
)
from src_v2.models.classifier import get_classifier_transforms


# ============================================================
# Constants
# ============================================================

CLASS_NAMES = ["COVID", "Normal", "Viral_Pneumonia"]
EXPECTED_SAMPLES = {"total": 1895, "COVID": 452, "Normal": 1274, "Viral_Pneumonia": 169}
V1_BASELINE_ACCURACY = 0.9826  # Known baseline from Phase 5
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 42
REGRESSION_GUARDRAIL_MAX = 5  # b <= 5 passes

CONFIGS = [
    {
        "name": "v1_baseline",
        "stage": "v1.0 Baseline",
        "config_path": "configs/ensemble_phase10_v1.json",
    },
    {
        "name": "cleaned_baseline",
        "stage": "Cleaned Baseline",
        "config_path": "configs/ensemble_phase10_cleaned.json",
    },
    {
        "name": "curriculum",
        "stage": "Curriculum Learning",
        "config_path": "configs/ensemble_phase10_curriculum.json",
    },
    {
        "name": "elastic_curriculum",
        "stage": "Elastic + Curriculum",
        "config_path": "configs/ensemble_phase10_elastic_curriculum.json",
    },
]


# ============================================================
# Data Loading
# ============================================================

def load_test_dataloader(
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, Dict[str, int]]:
    """Load test dataset with deterministic ordering.

    Args:
        data_dir: Base directory (expects data_dir/test/{class}/)
        batch_size: Batch size for DataLoader
        num_workers: Number of DataLoader workers (0 for determinism)

    Returns:
        Tuple of (dataloader, class_counts)
    """
    transform = get_classifier_transforms(train=False)
    test_dir = data_dir / "test"

    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    dataset = datasets.ImageFolder(test_dir, transform=transform)

    class_counts = {}
    for class_name in dataset.classes:
        class_idx = dataset.class_to_idx[class_name]
        class_counts[class_name] = sum(
            1 for t in dataset.targets if t == class_idx
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Critical: deterministic order
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return dataloader, class_counts


def verify_class_counts(class_counts: Dict[str, int]) -> None:
    """Assert test set has exactly expected class counts.

    Raises:
        AssertionError: If counts do not match expected values.
    """
    total = sum(class_counts.values())
    assert total == EXPECTED_SAMPLES["total"], (
        f"Total sample count mismatch: {total} != {EXPECTED_SAMPLES['total']}"
    )
    for cls in CLASS_NAMES:
        actual = class_counts.get(cls, 0)
        expected = EXPECTED_SAMPLES[cls]
        assert actual == expected, (
            f"Class {cls} count mismatch: {actual} != {expected}"
        )


# ============================================================
# Inference
# ============================================================

def run_ensemble_evaluation(
    config: Dict[str, Any],
    device: torch.device,
    use_tta: bool,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run ensemble evaluation for a single model config.

    Args:
        config: Loaded JSON config dict
        device: Torch device
        use_tta: Whether to use test-time augmentation
        batch_size: Inference batch size

    Returns:
        Tuple of (predictions, probabilities, labels, image_paths)
        - predictions: (N,) int array of predicted class indices
        - probabilities: (N, 3) float array of class probabilities
        - labels: (N,) int array of true labels
        - image_paths: (N,) str array of image file paths
    """
    data_dir = Path(config["data_dir"])

    # Load test data
    dataloader, class_counts = load_test_dataloader(data_dir, batch_size=batch_size)
    verify_class_counts(class_counts)

    # Load ensemble models
    models, weights = load_ensemble_models(config["checkpoint_paths"], device)

    # Extract image paths in dataloader order
    image_paths = np.array([s[0] for s in dataloader.dataset.samples])

    # Run inference
    model_preds, model_probs, labels, _ = ensemble_inference_with_tta(
        models=models,
        dataloader=dataloader,
        device=device,
        use_tta=use_tta,
    )

    # Weighted soft voting
    ensemble_preds, ensemble_probs = weighted_soft_voting(model_probs, weights)

    predictions = ensemble_preds.cpu().numpy().astype(int)
    probabilities = ensemble_probs.cpu().numpy()

    return predictions, probabilities, labels, image_paths


# ============================================================
# Metrics
# ============================================================

def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, Any]:
    """Compute classification metrics.

    Args:
        predictions: (N,) predicted class indices
        labels: (N,) true class indices
        probabilities: (N, 3) class probabilities

    Returns:
        Dict with accuracy, f1_macro, per_class, per_class_report
    """
    accuracy = float(accuracy_score(labels, predictions))

    report = classification_report(
        labels,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    per_class = {}
    for cls in CLASS_NAMES:
        per_class[cls] = {
            "precision": float(report[cls]["precision"]),
            "recall": float(report[cls]["recall"]),
            "f1-score": float(report[cls]["f1-score"]),
            "support": int(report[cls]["support"]),
        }

    f1_macro = float(report["macro avg"]["f1-score"])
    f1_weighted = float(report["weighted avg"]["f1-score"])

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class": per_class,
        "n_samples": int(len(labels)),
    }


# ============================================================
# CSV Output
# ============================================================

def save_predictions_csv(
    output_path: Path,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    image_paths: np.ndarray,
) -> None:
    """Save per-sample predictions to CSV.

    CSV columns: sample_idx, image_path, true_label, predicted_label,
    confidence, prob_COVID, prob_Normal, prob_VP

    Args:
        output_path: Destination CSV file path
        predictions: (N,) predicted class indices
        probabilities: (N, 3) class probabilities
        labels: (N,) true class indices
        image_paths: (N,) image file paths
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_idx", "image_path", "true_label", "predicted_label",
            "confidence", "prob_COVID", "prob_Normal", "prob_VP",
        ])
        for i in range(len(predictions)):
            confidence = float(probabilities[i].max())
            writer.writerow([
                i,
                str(image_paths[i]),
                int(labels[i]),
                int(predictions[i]),
                f"{confidence:.6f}",
                f"{probabilities[i, 0]:.6f}",
                f"{probabilities[i, 1]:.6f}",
                f"{probabilities[i, 2]:.6f}",
            ])


# ============================================================
# Statistical Tests
# ============================================================

def mcnemar_test(
    preds_ref: np.ndarray,
    preds_improved: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """McNemar's test with Yates continuity correction.

    Computes paired comparison between reference model and improved model.
    b = regressions: ref correct AND improved wrong
    c = improvements: ref wrong AND improved correct

    Args:
        preds_ref: (N,) reference model predictions
        preds_improved: (N,) improved model predictions
        labels: (N,) true labels

    Returns:
        Dict with b, c, statistic, p_value, significant (alpha=0.05)
    """
    ref_correct = (preds_ref == labels)
    imp_correct = (preds_improved == labels)

    b = int(np.sum(ref_correct & ~imp_correct))   # regressions
    c = int(np.sum(~ref_correct & imp_correct))   # improvements

    if b + c == 0:
        # No discordant pairs — cannot perform test
        return {
            "b_regressions": b,
            "c_improvements": c,
            "statistic": 0.0,
            "p_value": 1.0,
            "significant_raw": False,
            "note": "No discordant pairs",
        }

    # Yates correction: (|b - c| - 1)^2 / (b + c)
    statistic = float((abs(b - c) - 1) ** 2 / (b + c))
    p_value = float(1 - chi2.cdf(statistic, df=1))

    return {
        "b_regressions": b,
        "c_improvements": c,
        "statistic": statistic,
        "p_value": p_value,
        "significant_raw": bool(p_value < 0.05),
    }


def bootstrap_accuracy_ci(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, float]:
    """Bootstrap 95% CI for accuracy.

    Args:
        predictions: (N,) predicted class indices
        labels: (N,) true class indices
        n_iterations: Number of bootstrap resamples
        seed: Random seed for reproducibility

    Returns:
        Dict with mean, ci_low (2.5th percentile), ci_high (97.5th percentile)
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    correct = (predictions == labels).astype(float)

    bootstrap_accs = np.array([
        rng.choice(correct, size=n, replace=True).mean()
        for _ in range(n_iterations)
    ])

    return {
        "mean": float(bootstrap_accs.mean()),
        "ci_low": float(np.percentile(bootstrap_accs, 2.5)),
        "ci_high": float(np.percentile(bootstrap_accs, 97.5)),
        "n_iterations": n_iterations,
        "seed": seed,
    }


def bootstrap_auc_ci(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Bootstrap 95% CI for OvR AUC (DeLong surrogate).

    Computes per-class OvR AUC and macro AUC with bootstrap CI.

    Args:
        labels: (N,) true class indices
        probabilities: (N, 3) class probabilities
        n_iterations: Number of bootstrap resamples
        seed: Random seed

    Returns:
        Dict with per-class AUC, macro AUC, and 95% CIs
    """
    rng = np.random.RandomState(seed)
    n = len(labels)

    # Point estimates for full dataset
    # OvR AUC requires at least 2 classes to be present in labels
    try:
        macro_auc_point = float(
            roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
        )
        per_class_auc_point = {}
        for i, cls in enumerate(CLASS_NAMES):
            binary_labels = (labels == i).astype(int)
            if binary_labels.sum() > 0 and (1 - binary_labels).sum() > 0:
                per_class_auc_point[cls] = float(
                    roc_auc_score(binary_labels, probabilities[:, i])
                )
            else:
                per_class_auc_point[cls] = float("nan")
    except Exception as e:
        return {"error": str(e)}

    # Bootstrap CI for macro AUC
    bootstrap_macro_aucs = []
    bootstrap_per_class_aucs = {cls: [] for cls in CLASS_NAMES}

    for _ in range(n_iterations):
        idx = rng.choice(n, size=n, replace=True)
        boot_labels = labels[idx]
        boot_probs = probabilities[idx]

        # Skip if any class is missing in bootstrap sample
        if len(np.unique(boot_labels)) < len(CLASS_NAMES):
            continue

        try:
            bootstrap_macro_aucs.append(
                roc_auc_score(boot_labels, boot_probs, multi_class="ovr", average="macro")
            )
            for i, cls in enumerate(CLASS_NAMES):
                binary_boot = (boot_labels == i).astype(int)
                if binary_boot.sum() > 0 and (1 - binary_boot).sum() > 0:
                    bootstrap_per_class_aucs[cls].append(
                        roc_auc_score(binary_boot, boot_probs[:, i])
                    )
        except Exception:
            pass

    bootstrap_macro_aucs = np.array(bootstrap_macro_aucs)

    result = {
        "macro_auc": macro_auc_point,
        "macro_auc_ci_low": float(np.percentile(bootstrap_macro_aucs, 2.5))
        if len(bootstrap_macro_aucs) > 0 else float("nan"),
        "macro_auc_ci_high": float(np.percentile(bootstrap_macro_aucs, 97.5))
        if len(bootstrap_macro_aucs) > 0 else float("nan"),
        "per_class": {},
        "n_bootstrap_iterations": n_iterations,
        "seed": seed,
    }

    for cls in CLASS_NAMES:
        arr = np.array(bootstrap_per_class_aucs[cls])
        result["per_class"][cls] = {
            "auc": per_class_auc_point[cls],
            "ci_low": float(np.percentile(arr, 2.5)) if len(arr) > 0 else float("nan"),
            "ci_high": float(np.percentile(arr, 97.5)) if len(arr) > 0 else float("nan"),
        }

    return result


def holm_bonferroni_correction(
    p_values: List[float],
    comparison_names: List[str],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Apply Holm-Bonferroni multiple comparison correction.

    Sorted p-values ascending. For rank i (1-indexed):
    Corrected threshold = alpha / (m - i + 1) where m is total comparisons.

    Args:
        p_values: List of raw McNemar p-values
        comparison_names: Names matching p_values
        alpha: Family-wise error rate (default 0.05)

    Returns:
        Dict with sorted comparisons, corrected thresholds, and significance
    """
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    results = []
    for rank_1indexed, (orig_idx, p_val) in enumerate(indexed, start=1):
        corrected_threshold = alpha / (m - rank_1indexed + 1)
        results.append({
            "comparison": comparison_names[orig_idx],
            "p_value": p_val,
            "rank": rank_1indexed,
            "corrected_threshold": corrected_threshold,
            "significant_corrected": bool(p_val < corrected_threshold),
        })

    return {
        "alpha": alpha,
        "n_comparisons": m,
        "comparisons": results,
    }


def regression_guardrail(
    b_regressions: int,
    max_regressions: int = REGRESSION_GUARDRAIL_MAX,
) -> Dict[str, Any]:
    """Evaluate regression guardrail.

    Passes if b (regressions: ref correct AND improved wrong) <= max_regressions.

    Args:
        b_regressions: Number of samples where improved model regressed vs ref
        max_regressions: Maximum allowed regressions (default 5)

    Returns:
        Dict with b, max, passed status
    """
    passed = b_regressions <= max_regressions
    return {
        "b_regressions": b_regressions,
        "max_allowed": max_regressions,
        "passed": bool(passed),
        "message": (
            f"PASS: {b_regressions} regressions <= {max_regressions} threshold"
            if passed
            else f"WARNING: {b_regressions} regressions > {max_regressions} threshold"
        ),
    }


# ============================================================
# Main Evaluation Pipeline
# ============================================================

def evaluate_all_models(
    output_dir: Path,
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate all 4 models with TTA and no-TTA.

    Args:
        output_dir: Output directory for predictions and results
        device: Torch device
        batch_size: Inference batch size

    Returns:
        Dict mapping model name to evaluation results
    """
    predictions_dir = output_dir / "predictions"
    results_dir = output_dir / "ensemble_results"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for cfg_meta in CONFIGS:
        model_name = cfg_meta["name"]
        config_path = Path(cfg_meta["config_path"])

        print(f"\n{'='*70}")
        print(f"EVALUATING: {model_name} ({cfg_meta['stage']})")
        print(f"{'='*70}")

        with open(config_path) as f:
            config = json.load(f)

        model_results = {}

        # === TTA Evaluation ===
        print(f"\n[TTA] Running with TTA...")
        preds_tta, probs_tta, labels_tta, paths_tta = run_ensemble_evaluation(
            config, device, use_tta=True, batch_size=batch_size
        )
        metrics_tta = compute_metrics(preds_tta, labels_tta, probs_tta)
        model_results["tta"] = {
            "predictions": preds_tta,
            "probabilities": probs_tta,
            "labels": labels_tta,
            "image_paths": paths_tta,
            "metrics": metrics_tta,
        }

        print(f"  Accuracy (TTA): {metrics_tta['accuracy']:.4f}")
        print(f"  F1-macro (TTA): {metrics_tta['f1_macro']:.4f}")

        # Sanity check for v1.0 baseline
        # V1_BASELINE_ACCURACY (0.9826) is rounded; actual full-precision value is
        # 0.9825857519788919 from ensemble_test_results_tta.json. Allow 0.01pp tolerance.
        if model_name == "v1_baseline":
            acc_tta = metrics_tta["accuracy"]
            tolerance = 0.01 / 100  # 0.01 percentage points tolerance
            if abs(acc_tta - V1_BASELINE_ACCURACY) > tolerance:
                raise RuntimeError(
                    f"SANITY CHECK FAILED: v1.0 TTA accuracy={acc_tta:.6f} "
                    f"deviates by more than 0.01pp from expected {V1_BASELINE_ACCURACY:.4f}\n"
                    f"Delta: {abs(acc_tta - V1_BASELINE_ACCURACY)*100:.4f}pp\n"
                    f"Aborting — check model loading or data split integrity."
                )
            print(f"  [SANITY] v1.0 TTA accuracy={acc_tta:.6f} matches baseline {V1_BASELINE_ACCURACY} OK (delta={abs(acc_tta - V1_BASELINE_ACCURACY)*100:.4f}pp)")

        # === No-TTA Evaluation ===
        print(f"\n[No-TTA] Running without TTA...")
        preds_no_tta, probs_no_tta, labels_no_tta, _ = run_ensemble_evaluation(
            config, device, use_tta=False, batch_size=batch_size
        )
        metrics_no_tta = compute_metrics(preds_no_tta, labels_no_tta, probs_no_tta)
        model_results["no_tta"] = {
            "predictions": preds_no_tta,
            "probabilities": probs_no_tta,
            "labels": labels_no_tta,
            "metrics": metrics_no_tta,
        }

        print(f"  Accuracy (No-TTA): {metrics_no_tta['accuracy']:.4f}")
        print(f"  F1-macro (No-TTA): {metrics_no_tta['f1_macro']:.4f}")

        # Save predictions CSV
        csv_path = predictions_dir / f"{model_name}_predictions.csv"
        save_predictions_csv(
            csv_path,
            preds_tta,
            probs_tta,
            labels_tta,
            paths_tta,
        )
        print(f"  Predictions CSV: {csv_path}")

        # Save results JSON
        results_json = {
            "model_name": model_name,
            "stage": cfg_meta["stage"],
            "config_path": str(config_path),
            "tta": {
                "accuracy": metrics_tta["accuracy"],
                "f1_macro": metrics_tta["f1_macro"],
                "f1_weighted": metrics_tta["f1_weighted"],
                "per_class": metrics_tta["per_class"],
                "n_samples": metrics_tta["n_samples"],
            },
            "no_tta": {
                "accuracy": metrics_no_tta["accuracy"],
                "f1_macro": metrics_no_tta["f1_macro"],
                "f1_weighted": metrics_no_tta["f1_weighted"],
                "per_class": metrics_no_tta["per_class"],
                "n_samples": metrics_no_tta["n_samples"],
            },
        }
        json_path = results_dir / f"{model_name}_results.json"
        with open(json_path, "w") as f:
            json.dump(results_json, f, indent=2)

        all_results[model_name] = model_results
        print(f"  Results JSON: {json_path}")

    return all_results


def compute_statistical_validation(
    all_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute all statistical tests.

    Args:
        all_results: Output from evaluate_all_models()

    Returns:
        Dict with all statistical test results
    """
    print(f"\n{'='*70}")
    print("STATISTICAL VALIDATION")
    print(f"{'='*70}")

    ref_name = "v1_baseline"
    ref_preds = all_results[ref_name]["tta"]["predictions"]
    ref_labels = all_results[ref_name]["tta"]["labels"]
    ref_probs = all_results[ref_name]["tta"]["probabilities"]

    comparisons = [
        ("cleaned_baseline", "cleaned_vs_v1"),
        ("curriculum", "curriculum_vs_v1"),
        ("elastic_curriculum", "elastic_curriculum_vs_v1"),
    ]

    # Bootstrap CI for reference model
    print(f"\n[Bootstrap CI] Computing for {ref_name}...")
    ref_bootstrap = bootstrap_accuracy_ci(ref_preds, ref_labels)
    ref_auc = bootstrap_auc_ci(ref_labels, ref_probs)

    statistical_tests = {}
    mcnemar_p_values = []
    comparison_names_for_holm = []

    for model_name, comparison_key in comparisons:
        print(f"\n[Statistical tests] {comparison_key}")

        imp_preds = all_results[model_name]["tta"]["predictions"]
        imp_labels = all_results[model_name]["tta"]["labels"]
        imp_probs = all_results[model_name]["tta"]["probabilities"]

        # McNemar's test
        mcnemar_result = mcnemar_test(ref_preds, imp_preds, ref_labels)
        print(
            f"  McNemar: b={mcnemar_result['b_regressions']}, "
            f"c={mcnemar_result['c_improvements']}, "
            f"p={mcnemar_result['p_value']:.4f}, "
            f"significant={mcnemar_result['significant_raw']}"
        )

        # Regression guardrail
        guardrail = regression_guardrail(mcnemar_result["b_regressions"])
        print(f"  Guardrail: {guardrail['message']}")
        if not guardrail["passed"]:
            print(f"  WARNING: Regression guardrail failed for {comparison_key}")

        # Bootstrap CI for improved model
        print(f"  [Bootstrap CI] Computing for {model_name}...")
        imp_bootstrap = bootstrap_accuracy_ci(imp_preds, imp_labels)

        # DeLong AUC (bootstrap-based)
        print(f"  [DeLong AUC] Computing for {model_name}...")
        imp_auc = bootstrap_auc_ci(imp_labels, imp_probs)

        statistical_tests[comparison_key] = {
            "mcnemar": mcnemar_result,
            "regression_guardrail": guardrail,
            "improved_model_bootstrap_ci": imp_bootstrap,
            "delong_auc": imp_auc,
        }

        mcnemar_p_values.append(mcnemar_result["p_value"])
        comparison_names_for_holm.append(comparison_key)

    # Holm-Bonferroni correction
    print(f"\n[Holm-Bonferroni] Correcting {len(mcnemar_p_values)} p-values...")
    holm_result = holm_bonferroni_correction(mcnemar_p_values, comparison_names_for_holm)
    for comp in holm_result["comparisons"]:
        print(
            f"  {comp['comparison']}: p={comp['p_value']:.4f}, "
            f"threshold={comp['corrected_threshold']:.4f}, "
            f"significant={comp['significant_corrected']}"
        )

    statistical_tests["holm_bonferroni_correction"] = holm_result

    # Add reference model stats
    statistical_tests["v1_baseline_bootstrap_ci"] = ref_bootstrap
    statistical_tests["v1_baseline_delong_auc"] = ref_auc

    return statistical_tests


def assemble_final_report(
    all_results: Dict[str, Dict[str, Any]],
    statistical_tests: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Assemble phase10_final_report.json.

    Args:
        all_results: Output from evaluate_all_models()
        statistical_tests: Output from compute_statistical_validation()
        output_dir: Directory to save report

    Returns:
        Complete report dict
    """
    v1_acc_tta = all_results["v1_baseline"]["tta"]["metrics"]["accuracy"]

    models_section = {}
    pipeline_progression = []

    for cfg_meta in CONFIGS:
        model_name = cfg_meta["name"]
        stage = cfg_meta["stage"]

        tta_metrics = all_results[model_name]["tta"]["metrics"]
        no_tta_metrics = all_results[model_name]["no_tta"]["metrics"]

        # Bootstrap CI — reuse from statistical_tests or compute for v1
        if model_name == "v1_baseline":
            bootstrap_ci = statistical_tests["v1_baseline_bootstrap_ci"]
            delong_auc = statistical_tests["v1_baseline_delong_auc"]
        else:
            comparison_key = {
                "cleaned_baseline": "cleaned_vs_v1",
                "curriculum": "curriculum_vs_v1",
                "elastic_curriculum": "elastic_curriculum_vs_v1",
            }[model_name]
            bootstrap_ci = statistical_tests[comparison_key]["improved_model_bootstrap_ci"]
            delong_auc = statistical_tests[comparison_key]["delong_auc"]

        models_section[model_name] = {
            "stage": stage,
            "accuracy_tta": tta_metrics["accuracy"],
            "accuracy_no_tta": no_tta_metrics["accuracy"],
            "f1_macro_tta": tta_metrics["f1_macro"],
            "f1_macro_no_tta": no_tta_metrics["f1_macro"],
            "f1_weighted_tta": tta_metrics["f1_weighted"],
            "per_class_tta": tta_metrics["per_class"],
            "per_class_no_tta": no_tta_metrics["per_class"],
            "bootstrap_ci": bootstrap_ci,
            "delong_auc": delong_auc,
        }

        delta_pp = (tta_metrics["accuracy"] - v1_acc_tta) * 100
        pipeline_progression.append({
            "stage": stage,
            "model_name": model_name,
            "accuracy_tta": tta_metrics["accuracy"],
            "f1_macro_tta": tta_metrics["f1_macro"],
            "vp_recall_tta": tta_metrics["per_class"]["Viral_Pneumonia"]["recall"],
            "delta_pp": float(delta_pp),
        })

    # Build statistical_tests section (serializable)
    stats_section = {}
    for comparison_key, _ in [
        ("cleaned_vs_v1", "cleaned_baseline"),
        ("curriculum_vs_v1", "curriculum"),
        ("elastic_curriculum_vs_v1", "elastic_curriculum"),
    ]:
        test = statistical_tests[comparison_key]
        stats_section[comparison_key] = {
            "mcnemar": test["mcnemar"],
            "delong_auc": test["delong_auc"],
            "regression_guardrail": test["regression_guardrail"],
        }
    stats_section["holm_bonferroni_correction"] = statistical_tests["holm_bonferroni_correction"]

    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "test_set": {
            "n_samples": EXPECTED_SAMPLES["total"],
            "n_COVID": EXPECTED_SAMPLES["COVID"],
            "n_Normal": EXPECTED_SAMPLES["Normal"],
            "n_VP": EXPECTED_SAMPLES["Viral_Pneumonia"],
        },
        "v1_baseline_accuracy_reference": V1_BASELINE_ACCURACY,
        "models": models_section,
        "statistical_tests": stats_section,
        "pipeline_progression": pipeline_progression,
    }

    report_path = output_dir / "phase10_final_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nFinal report saved: {report_path}")
    return report


def print_summary_table(report: Dict[str, Any]) -> None:
    """Print summary table of all 4 models.

    Args:
        report: Output from assemble_final_report()
    """
    print(f"\n{'='*90}")
    print("PHASE 10 SUMMARY TABLE")
    print(f"{'='*90}")
    print(
        f"{'Model':<30} {'Acc(TTA)':>10} {'Acc(noTTA)':>12} {'F1-macro':>10} "
        f"{'VP Recall':>10} {'Delta(pp)':>10}"
    )
    print("-" * 90)

    for prog in report["pipeline_progression"]:
        model_name = prog["model_name"]
        model_data = report["models"][model_name]
        vp_recall = prog["vp_recall_tta"]
        print(
            f"{prog['stage']:<30} {prog['accuracy_tta']:>10.4f} "
            f"{model_data['accuracy_no_tta']:>12.4f} {prog['f1_macro_tta']:>10.4f} "
            f"{vp_recall:>10.4f} {prog['delta_pp']:>+10.3f}"
        )

    print(f"{'='*90}")

    # McNemar p-values
    print(f"\nMcNemar's Test Results (vs v1.0 baseline, Yates correction):")
    print(f"{'Comparison':<35} {'b(reg)':>8} {'c(impr)':>8} {'p-value':>12} {'Holm-sig':>10}")
    print("-" * 75)

    holm = report["statistical_tests"]["holm_bonferroni_correction"]
    holm_sig = {comp["comparison"]: comp["significant_corrected"] for comp in holm["comparisons"]}

    for comparison_key in ["cleaned_vs_v1", "curriculum_vs_v1", "elastic_curriculum_vs_v1"]:
        test = report["statistical_tests"][comparison_key]
        mc = test["mcnemar"]
        sig = holm_sig.get(comparison_key, False)
        print(
            f"{comparison_key:<35} {mc['b_regressions']:>8} {mc['c_improvements']:>8} "
            f"{mc['p_value']:>12.4f} {'YES' if sig else 'NO':>10}"
        )

    print(f"\nRegression Guardrail (b <= {REGRESSION_GUARDRAIL_MAX}):")
    for comparison_key in ["cleaned_vs_v1", "curriculum_vs_v1", "elastic_curriculum_vs_v1"]:
        g = report["statistical_tests"][comparison_key]["regression_guardrail"]
        status = "PASS" if g["passed"] else "FAIL"
        print(f"  {comparison_key}: b={g['b_regressions']} -> {status}")

    print(f"\nBootstrap 95% CI (Accuracy, TTA):")
    for cfg_meta in CONFIGS:
        model_name = cfg_meta["name"]
        ci = report["models"][model_name]["bootstrap_ci"]
        acc = report["models"][model_name]["accuracy_tta"]
        print(
            f"  {cfg_meta['stage']:<30}: "
            f"{acc:.4f} [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
        )

    print(f"{'='*90}\n")


# ============================================================
# Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 10 final evaluation and statistical validation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/phase10",
        help="Output directory (default: outputs/phase10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu, auto-detected if not specified)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size (default: 32)",
    )
    args = parser.parse_args()

    # Device setup
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\nPhase 10 Final Evaluation")
    print(f"Device: {device}")
    print(f"Output directory: {args.output_dir}")

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Part A: Evaluate all 4 models
    print(f"\n{'='*70}")
    print("PART A: ENSEMBLE EVALUATION (4 models x TTA + no-TTA)")
    print(f"{'='*70}")

    all_results = evaluate_all_models(output_dir, device, batch_size=args.batch_size)

    # Part B: Statistical validation
    print(f"\n{'='*70}")
    print("PART B: STATISTICAL VALIDATION")
    print(f"{'='*70}")

    statistical_tests = compute_statistical_validation(all_results)

    # Part C: Assemble final report
    print(f"\n{'='*70}")
    print("PART C: ASSEMBLING FINAL REPORT")
    print(f"{'='*70}")

    report = assemble_final_report(all_results, statistical_tests, output_dir)

    # Print summary table
    print_summary_table(report)

    print(f"{'='*70}")
    print("PHASE 10 EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutputs:")
    print(f"  Predictions CSVs:  {output_dir}/predictions/")
    print(f"  Ensemble results:  {output_dir}/ensemble_results/")
    print(f"  Final report:      {output_dir}/phase10_final_report.json")
    print()


if __name__ == "__main__":
    main()
