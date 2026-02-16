#!/usr/bin/env python3
"""
Script para generar matrices de confusión de comparación baseline vs ensemble+TTA.

Genera:
- confusion_matrix_baseline.png: Matriz agregada de 5 folds (baseline individual)
- confusion_matrix_ensemble_tta.png: Matriz de ensemble con TTA
- comparison_metrics.json: Métricas estructuradas con deltas

Uso:
    python scripts/generate_confusion_matrices_comparison.py --cv-dir outputs/classifier_cv --output-dir outputs/classifier_cv
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Configurar matplotlib para mejor renderizado
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1


def load_baseline_data(cv_dir: Path) -> dict:
    """
    Carga y agrega resultados de los 5 folds evaluados en test set (baseline individual).

    Args:
        cv_dir: Directorio con resultados de CV

    Returns:
        Dict con matriz agregada, métricas promedio y por clase
    """
    confusion_matrices = []
    accuracies = []
    f1_macros = []
    f1_weighteds = []

    # Cargar resultados de test de cada fold
    for fold in range(1, 6):
        fold_path = cv_dir / f"fold_{fold:02d}" / "test_results.json"

        if not fold_path.exists():
            raise FileNotFoundError(f"No se encontró: {fold_path}")

        with open(fold_path) as f:
            results = json.load(f)

        cm = np.array(results["confusion_matrix"])
        confusion_matrices.append(cm)
        accuracies.append(results["metrics"]["accuracy"])
        f1_macros.append(results["metrics"]["f1_macro"])
        f1_weighteds.append(results["metrics"]["f1_weighted"])

    # Agregar matrices de confusión del test set (5 evaluaciones)
    aggregated_cm = np.sum(confusion_matrices, axis=0)

    # Calcular métricas promedio y desviación estándar
    accuracy_mean = np.mean(accuracies)
    accuracy_std = np.std(accuracies)
    f1_macro_mean = np.mean(f1_macros)
    f1_macro_std = np.std(f1_macros)
    f1_weighted_mean = np.mean(f1_weighteds)
    f1_weighted_std = np.std(f1_weighteds)

    # Calcular precision, recall, F1 por clase desde la matriz agregada
    class_names = ["COVID", "Normal", "Viral_Pneumonia"]
    per_class_metrics = {}

    for i, class_name in enumerate(class_names):
        tp = aggregated_cm[i, i]
        fp = np.sum(aggregated_cm[:, i]) - tp
        fn = np.sum(aggregated_cm[i, :]) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        per_class_metrics[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1-score": float(f1),
            "support": int(np.sum(aggregated_cm[i, :]))
        }

    # Total de evaluaciones
    test_set_size = int(np.sum(confusion_matrices[0]))
    total_evaluations = len(confusion_matrices) * test_set_size

    # Encontrar el mejor fold
    best_fold_idx = np.argmax(accuracies)
    best_fold = int(best_fold_idx + 1)
    best_fold_accuracy = accuracies[best_fold_idx]

    # Calcular per_fold_accuracies
    per_fold_accuracies = [float(acc) for acc in accuracies]

    return {
        "confusion_matrix": aggregated_cm,
        "accuracy_mean": float(accuracy_mean),
        "accuracy_std": float(accuracy_std),
        "f1_macro_mean": float(f1_macro_mean),
        "f1_macro_std": float(f1_macro_std),
        "f1_weighted_mean": float(f1_weighted_mean),
        "f1_weighted_std": float(f1_weighted_std),
        "per_class_metrics": per_class_metrics,
        "test_set_size": test_set_size,
        "n_folds": len(confusion_matrices),
        "total_evaluations": total_evaluations,
        "per_fold_accuracies": per_fold_accuracies,
        "best_fold": best_fold,
        "best_fold_accuracy": float(best_fold_accuracy)
    }


def load_ensemble_tta_data(cv_dir: Path) -> dict:
    """
    Carga resultados de ensemble con TTA desde ensemble_test_results_tta.json.

    Args:
        cv_dir: Directorio con resultados de CV

    Returns:
        Dict con matriz, métricas y análisis a nivel de caso
    """
    tta_path = cv_dir / "ensemble_test_results_tta.json"

    if not tta_path.exists():
        raise FileNotFoundError(f"No se encontró: {tta_path}")

    with open(tta_path) as f:
        results = json.load(f)

    ensemble_soft = results["ensemble_soft_voting"]

    return {
        "confusion_matrix": np.array(ensemble_soft["confusion_matrix"]),
        "metrics": ensemble_soft["metrics"],
        "per_class": ensemble_soft["per_class"],
        "tta_delta_metrics": results["tta_delta_metrics"],
        "case_level_analysis": results["case_level_analysis"]
    }


def load_ensemble_no_tta_data(cv_dir: Path) -> dict:
    """
    Carga resultados de ensemble sin TTA desde ensemble_test_results_no_tta.json.

    Args:
        cv_dir: Directorio con resultados de CV

    Returns:
        Dict con métricas del ensemble sin TTA
    """
    no_tta_path = cv_dir / "ensemble_test_results_no_tta.json"

    if not no_tta_path.exists():
        raise FileNotFoundError(f"No se encontró: {no_tta_path}")

    with open(no_tta_path) as f:
        results = json.load(f)

    ensemble_soft = results["ensemble_soft_voting"]

    return {
        "metrics": ensemble_soft["metrics"]
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    title: str,
    subtitle: str,
    output_path: Path
):
    """
    Genera un heatmap de matriz de confusión con anotaciones duales.

    Args:
        cm: Matriz de confusión (array 3x3)
        class_names: Nombres de las clases para mostrar
        title: Título principal
        subtitle: Subtítulo con métricas
        output_path: Ruta de salida
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    # Calcular porcentajes (normalización por filas)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Crear heatmap
    heatmap = sns.heatmap(
        cm_percent,
        annot=False,
        cmap='Blues',
        cbar_kws={'label': 'Porcentaje (%)'},
        ax=ax,
        vmin=0,
        vmax=100
    )

    # Configurar colorbar
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Porcentaje (%)', fontsize=15, fontweight='bold')
    cbar.ax.tick_params(labelsize=13)

    # Anotar con valores absolutos y porcentajes
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = cm[i, j]
            percent = cm_percent[i, j]

            # Color del texto (blanco para celdas oscuras)
            if percent > 50:
                text_color = 'white'
            else:
                text_color = 'black'

            # Negrita para la diagonal
            if i == j:
                weight = 'bold'
            else:
                weight = 'normal'

            text = f'{value}\n({percent:.1f}%)'

            ax.text(
                j + 0.5, i + 0.5, text,
                ha='center', va='center',
                color=text_color,
                fontsize=15,
                weight=weight
            )

    # Configurar ejes
    ax.set_xlabel('Predicción', fontsize=17, fontweight='bold')
    ax.set_ylabel('Categoría Real', fontsize=17, fontweight='bold')
    ax.set_title(
        f'{title}\n{subtitle}',
        fontsize=18,
        fontweight='bold',
        pad=20
    )

    # Etiquetas de las clases
    ax.set_xticklabels(class_names, rotation=0, ha='center')
    ax.set_yticklabels(class_names, rotation=0)
    ax.tick_params(axis='both', labelsize=15)

    # Ajustar layout
    plt.tight_layout()

    # Guardar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Matriz de confusión guardada: {output_path}")
    plt.close()


def generate_comparison_metrics_json(
    baseline_data: dict,
    ensemble_no_tta_data: dict,
    ensemble_tta_data: dict,
    output_path: Path
):
    """
    Genera comparison_metrics.json con métricas estructuradas y deltas.

    Args:
        baseline_data: Datos del baseline (5 folds agregados)
        ensemble_no_tta_data: Datos del ensemble sin TTA
        ensemble_tta_data: Datos del ensemble con TTA
        output_path: Ruta de salida del JSON
    """
    # Calcular mejoras
    ensemble_over_baseline_acc_delta = ensemble_no_tta_data["metrics"]["accuracy"] - baseline_data["accuracy_mean"]
    ensemble_over_baseline_acc_delta_pp = ensemble_over_baseline_acc_delta * 100

    tta_over_ensemble_acc_delta = ensemble_tta_data["metrics"]["accuracy"] - ensemble_no_tta_data["metrics"]["accuracy"]
    tta_over_ensemble_acc_delta_pp = tta_over_ensemble_acc_delta * 100

    total_over_baseline_acc_delta = ensemble_tta_data["metrics"]["accuracy"] - baseline_data["accuracy_mean"]
    total_over_baseline_acc_delta_pp = total_over_baseline_acc_delta * 100

    comparison_data = {
        "description": "Comparison: Baseline Individual vs Ensemble+TTA",
        "timestamp": datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "baseline": {
            "source": "5-fold individual model evaluation on test set",
            "accuracy_mean": baseline_data["accuracy_mean"],
            "accuracy_std": baseline_data["accuracy_std"],
            "f1_macro_mean": baseline_data["f1_macro_mean"],
            "f1_macro_std": baseline_data["f1_macro_std"],
            "f1_weighted_mean": baseline_data["f1_weighted_mean"],
            "f1_weighted_std": baseline_data["f1_weighted_std"],
            "best_fold": baseline_data["best_fold"],
            "best_fold_accuracy": baseline_data["best_fold_accuracy"],
            "per_class": baseline_data["per_class_metrics"],
            "test_set_size": baseline_data["test_set_size"],
            "n_folds": baseline_data["n_folds"]
        },
        "ensemble_no_tta": {
            "source": "ensemble_test_results_no_tta.json",
            "accuracy": ensemble_no_tta_data["metrics"]["accuracy"],
            "f1_macro": ensemble_no_tta_data["metrics"]["f1_macro"],
            "f1_weighted": ensemble_no_tta_data["metrics"]["f1_weighted"]
        },
        "ensemble_tta": {
            "source": "ensemble_test_results_tta.json",
            "accuracy": ensemble_tta_data["metrics"]["accuracy"],
            "f1_macro": ensemble_tta_data["metrics"]["f1_macro"],
            "f1_weighted": ensemble_tta_data["metrics"]["f1_weighted"],
            "per_class": ensemble_tta_data["per_class"],
            "case_level_impact": ensemble_tta_data["case_level_analysis"]["summary"]
        },
        "improvements": {
            "ensemble_over_baseline": {
                "accuracy_delta": ensemble_over_baseline_acc_delta,
                "accuracy_delta_pp": ensemble_over_baseline_acc_delta_pp,
                "f1_macro_delta": ensemble_no_tta_data["metrics"]["f1_macro"] - baseline_data["f1_macro_mean"]
            },
            "tta_over_ensemble": {
                "accuracy_delta": tta_over_ensemble_acc_delta,
                "accuracy_delta_pp": tta_over_ensemble_acc_delta_pp,
                "f1_macro_delta": ensemble_tta_data["metrics"]["f1_macro"] - ensemble_no_tta_data["metrics"]["f1_macro"]
            },
            "total_over_baseline": {
                "accuracy_delta": total_over_baseline_acc_delta,
                "accuracy_delta_pp": total_over_baseline_acc_delta_pp,
                "f1_macro_delta": ensemble_tta_data["metrics"]["f1_macro"] - baseline_data["f1_macro_mean"]
            }
        },
        "tta_per_class_f1_delta": ensemble_tta_data["tta_delta_metrics"]["per_class_f1_delta"]
    }

    # Guardar JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)

    print(f"✓ Comparison metrics guardado: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera matrices de confusión de comparación y JSON de métricas."
    )
    parser.add_argument(
        "--cv-dir",
        type=Path,
        default=Path("outputs/classifier_cv"),
        help="Directorio con resultados de CV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classifier_cv"),
        help="Directorio de salida.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Rutas
    base_dir = Path(__file__).parent.parent
    cv_dir = base_dir / args.cv_dir if not args.cv_dir.is_absolute() else args.cv_dir
    output_dir = base_dir / args.output_dir if not args.output_dir.is_absolute() else args.output_dir

    print("=" * 70)
    print("GENERACIÓN DE MATRICES DE CONFUSIÓN - COMPARACIÓN")
    print("=" * 70)

    # Cargar datos
    print(f"\nCargando datos de: {cv_dir}")
    baseline_data = load_baseline_data(cv_dir)
    ensemble_no_tta_data = load_ensemble_no_tta_data(cv_dir)
    ensemble_tta_data = load_ensemble_tta_data(cv_dir)

    print(f"\n✓ Baseline (5 folds agregados): {baseline_data['accuracy_mean']*100:.2f}% ± {baseline_data['accuracy_std']*100:.2f}%")
    print(f"✓ Ensemble (sin TTA): {ensemble_no_tta_data['metrics']['accuracy']*100:.2f}%")
    print(f"✓ Ensemble (con TTA): {ensemble_tta_data['metrics']['accuracy']*100:.2f}%")

    # Nombres de las clases para display
    class_display_names = ["COVID-19", "Normal", "Neumonía Viral"]

    # Generar matriz de confusión baseline
    print("\nGenerando matriz de confusión baseline...")
    baseline_title = "Matriz de Confusión — Modelo Individual (Baseline)"
    baseline_subtitle = f"Promedio Individual (k=5) — Accuracy: {baseline_data['accuracy_mean']*100:.2f}% ± {baseline_data['accuracy_std']*100:.2f}%"
    baseline_output_path = output_dir / "confusion_matrix_baseline.png"

    plot_confusion_matrix(
        cm=baseline_data["confusion_matrix"],
        class_names=class_display_names,
        title=baseline_title,
        subtitle=baseline_subtitle,
        output_path=baseline_output_path
    )

    # Generar matriz de confusión ensemble+TTA
    print("Generando matriz de confusión ensemble+TTA...")
    ensemble_title = "Matriz de Confusión — Ensemble + TTA"
    ensemble_subtitle = f"Ensemble + TTA — Accuracy: {ensemble_tta_data['metrics']['accuracy']*100:.2f}%"
    ensemble_output_path = output_dir / "confusion_matrix_ensemble_tta.png"

    plot_confusion_matrix(
        cm=ensemble_tta_data["confusion_matrix"],
        class_names=class_display_names,
        title=ensemble_title,
        subtitle=ensemble_subtitle,
        output_path=ensemble_output_path
    )

    # Generar comparison_metrics.json
    print("\nGenerando comparison_metrics.json...")
    comparison_json_path = output_dir / "comparison_metrics.json"
    generate_comparison_metrics_json(
        baseline_data=baseline_data,
        ensemble_no_tta_data=ensemble_no_tta_data,
        ensemble_tta_data=ensemble_tta_data,
        output_path=comparison_json_path
    )

    # Mostrar resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE MEJORAS")
    print("=" * 70)

    ensemble_delta = ensemble_no_tta_data["metrics"]["accuracy"] - baseline_data["accuracy_mean"]
    tta_delta = ensemble_tta_data["metrics"]["accuracy"] - ensemble_no_tta_data["metrics"]["accuracy"]
    total_delta = ensemble_tta_data["metrics"]["accuracy"] - baseline_data["accuracy_mean"]

    print(f"\nEnsemble sobre Baseline: +{ensemble_delta*100:.2f} pp")
    print(f"TTA sobre Ensemble: +{tta_delta*100:.2f} pp")
    print(f"Total sobre Baseline: +{total_delta*100:.2f} pp")

    print("\nCasos impactados por TTA:")
    print(f"  Ayudados: {ensemble_tta_data['case_level_analysis']['summary']['helped']}")
    print(f"  Perjudicados: {ensemble_tta_data['case_level_analysis']['summary']['hurt']}")
    print(f"  Neutrales: {ensemble_tta_data['case_level_analysis']['summary']['neutral']}")

    print("\n" + "=" * 70)
    print("PROCESO COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
