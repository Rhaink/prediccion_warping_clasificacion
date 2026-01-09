#!/usr/bin/env python3
"""
Script para generar resultados de Fase 7: Comparacion 2 clases vs 3 clases.

FASE 7 DEL PIPELINE FISHER-WARPING (CORREGIDO)
===============================================

PROBLEMA ANTERIOR:
El experimento de 3 clases reutilizaba features de Fase 4 (generadas del CSV
de 2 clases con 12,402 imagenes) y solo re-etiquetaba por nombre de image_id.
Esto era metodologicamente incorrecto.

SOLUCION:
Este script ahora:
1. Carga imagenes DIRECTAMENTE del CSV de 3 clases (01_full_balanced_3class_*.csv)
2. Ejecuta pipeline COMPLETO: PCA -> Z-score -> Fisher multiclase -> KNN
3. Agrega experimento 2C-Comparable usando las MISMAS imagenes del CSV de 3 clases
   pero reagrupando: COVID + Viral_Pneumonia -> Enfermo

DATASETS:
- 3 clases: 01_full_balanced_3class_*.csv (6,725 imgs, test=680)
- 2C-Comparable: Mismas 6,725 imagenes, reagrupadas a 2 clases

Esto permite comparacion directa 2C vs 3C con las MISMAS imagenes.

Los resultados de 2 clases originales (12,402 imgs) se cargan de Fase 6.

Uso:
    python src/generate_phase7.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Agregar directorio src al path
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from data_loader import load_dataset, Dataset
from pca import PCA
from features import StandardScaler
from fisher import FisherRatioMulticlass, plot_fisher_ratios, save_fisher_results
from classifier import (
    KNNClassifier,
    find_best_k,
    evaluate_classifier,
    plot_confusion_matrix,
    plot_k_optimization,
    save_classification_results,
    save_predictions,
    ClassificationResult,
    KOptimizationResult
)


# ============================================================================
# CONFIGURACION
# ============================================================================

# Directorios
BASE_DIR = Path(__file__).parent.parent
PROJECT_BASE = BASE_DIR.parent.parent  # Raiz del proyecto para cargar imagenes
CSV_3CLASS_WARPED = BASE_DIR / "results" / "metrics" / "01_full_balanced_3class_warped.csv"
CSV_3CLASS_ORIGINAL = BASE_DIR / "results" / "metrics" / "01_full_balanced_3class_original.csv"
PHASE6_METRICS_DIR = BASE_DIR / "results" / "metrics" / "phase6_classification"
OUTPUT_METRICS_DIR = BASE_DIR / "results" / "metrics" / "phase7_comparison"
OUTPUT_FIGURES_DIR = BASE_DIR / "results" / "figures" / "phase7_comparison"

# Configuracion de PCA
N_COMPONENTS = 50  # Igual que Fase 4

# Nombres de clases
CLASS_NAMES_3C = ["COVID", "Normal", "Viral_Pneumonia"]
CLASS_NAMES_2C = ["Enfermo", "Normal"]

# Valores de K a probar
K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21, 31, 41, 51]


@dataclass
class ExperimentResult:
    """Resultado de un experimento completo."""
    name: str
    n_classes: int
    class_names: List[str]
    dataset_size: int
    test_size: int
    classification_result: ClassificationResult
    k_optimization: KOptimizationResult
    fisher_info: Dict


def run_full_pipeline(
    csv_path: Path,
    scenario: str,
    experiment_name: str,
    use_mask: bool = True,
    verbose: bool = True
) -> ExperimentResult:
    """
    Ejecuta el pipeline completo: Cargar datos -> PCA -> Z-score -> Fisher -> KNN.

    Args:
        csv_path: Ruta al CSV con los splits
        scenario: "3class" o "2class"
        experiment_name: Nombre para identificar el experimento
        use_mask: Si True, usa mascara para imagenes warped
        verbose: Si True, imprime progreso

    Returns:
        ExperimentResult con todos los resultados
    """
    if verbose:
        print()
        print("=" * 70)
        print(f"EXPERIMENTO: {experiment_name.upper()}")
        print(f"Escenario: {scenario}")
        print("=" * 70)

    # === PASO 1: Cargar datos ===
    if verbose:
        print("\n[1/5] Cargando datos...")

    dataset = load_dataset(
        csv_path=csv_path,
        base_path=PROJECT_BASE,
        scenario=scenario,
        use_mask=use_mask,
        verbose=verbose
    )

    class_names = dataset.class_names
    n_classes = len(class_names)

    if verbose:
        print(f"\n  Total: {len(dataset.train.ids) + len(dataset.val.ids) + len(dataset.test.ids)} imagenes")
        print(f"  Train: {len(dataset.train.ids)}, Val: {len(dataset.val.ids)}, Test: {len(dataset.test.ids)}")

    # === PASO 2: PCA ===
    if verbose:
        print("\n[2/5] Aplicando PCA...")

    pca = PCA(n_components=N_COMPONENTS)
    X_train_pca = pca.fit_transform(dataset.train.X, verbose=verbose)
    X_val_pca = pca.transform(dataset.val.X)
    X_test_pca = pca.transform(dataset.test.X)

    if verbose:
        print(f"  Shape despues de PCA: {X_train_pca.shape}")

    # === PASO 3: Z-score ===
    if verbose:
        print("\n[3/5] Aplicando estandarizacion Z-score...")

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_pca, verbose=verbose)
    X_val_std = scaler.transform(X_val_pca)
    X_test_std = scaler.transform(X_test_pca)

    # === PASO 4: Fisher ===
    if verbose:
        print("\n[4/5] Calculando Fisher ratios...")

    if n_classes == 2:
        from fisher import FisherRatio
        fisher = FisherRatio()
    else:
        fisher = FisherRatioMulticlass()

    fisher.fit(X_train_std, dataset.train.y, class_names=class_names, verbose=verbose)

    # Amplificar caracteristicas
    X_train_amp = fisher.amplify(X_train_std)
    X_val_amp = fisher.amplify(X_val_std)
    X_test_amp = fisher.amplify(X_test_std)

    fisher_info = {
        "fisher_ratios": fisher.fisher_ratios_.tolist(),
        "n_classes": n_classes,
        "class_names": class_names
    }

    # Crear directorio de salida
    dataset_output_dir = OUTPUT_FIGURES_DIR / experiment_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    # Guardar figura de Fisher ratios
    fig_fisher = plot_fisher_ratios(
        fisher.get_result(),
        output_path=dataset_output_dir / "fisher_ratios.png",
        top_k=10
    )
    plt.close(fig_fisher)

    # === PASO 5: KNN ===
    if verbose:
        print("\n[5/5] Entrenando y evaluando KNN...")

    # Ajustar K_VALUES al tamano del dataset
    max_k = min(X_train_amp.shape[0], max(K_VALUES))
    k_values = [k for k in K_VALUES if k <= max_k]

    if verbose:
        print(f"  Valores de K a probar: {k_values}")

    # Optimizar K
    opt_result = find_best_k(
        X_train_amp, dataset.train.y,
        X_val_amp, dataset.val.y,
        k_values=k_values,
        verbose=verbose
    )

    # Guardar grafico de optimizacion
    fig_k = plot_k_optimization(
        opt_result,
        output_path=dataset_output_dir / "k_optimization.png"
    )
    plt.close(fig_k)

    # Entrenar con K optimo
    if verbose:
        print(f"\n  Entrenando KNN con K={opt_result.best_k}...")

    knn = KNNClassifier(k=opt_result.best_k)
    knn.fit(X_train_amp, dataset.train.y, verbose=False)

    # Evaluar en test
    if verbose:
        print(f"\n  Evaluando en conjunto de TEST ({len(dataset.test.ids)} muestras)...")

    test_result = evaluate_classifier(
        knn, X_test_amp, dataset.test.y, class_names, verbose=verbose
    )

    # Guardar matriz de confusion
    fig_cm = plot_confusion_matrix(
        test_result,
        output_path=dataset_output_dir / "confusion_matrix.png",
        normalize=False,
        title=f"Matriz de Confusion - {experiment_name}\nK={opt_result.best_k}"
    )
    plt.close(fig_cm)

    fig_cm_norm = plot_confusion_matrix(
        test_result,
        output_path=dataset_output_dir / "confusion_matrix_normalized.png",
        normalize=True,
        title=f"Matriz de Confusion (%) - {experiment_name}\nK={opt_result.best_k}"
    )
    plt.close(fig_cm_norm)

    # Guardar resultados
    OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    save_classification_results(
        test_result,
        experiment_name,
        OUTPUT_METRICS_DIR / f"{experiment_name}_results.csv"
    )

    save_predictions(
        test_result,
        dataset.test.ids,
        OUTPUT_METRICS_DIR / f"{experiment_name}_predictions.csv"
    )

    dataset_size = len(dataset.train.ids) + len(dataset.val.ids) + len(dataset.test.ids)

    return ExperimentResult(
        name=experiment_name,
        n_classes=n_classes,
        class_names=class_names,
        dataset_size=dataset_size,
        test_size=len(dataset.test.ids),
        classification_result=test_result,
        k_optimization=opt_result,
        fisher_info=fisher_info
    )


def load_2class_results() -> Dict[str, Dict]:
    """
    Carga resultados de 2 clases de Fase 6.
    """
    results = {}

    summary_path = PHASE6_METRICS_DIR / "summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            data = json.load(f)
            results = data.get("datasets", {})

    return results


def generate_comparison_table(
    results_3c: Dict[str, ExperimentResult],
    results_2c_comparable: Dict[str, ExperimentResult],
    results_2c_original: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Genera tabla comparativa completa.
    """
    rows = []

    # Resultados de 2 clases originales (dataset de 12,402 imgs)
    for dataset_name, r2c in results_2c_original.items():
        if "full" in dataset_name:  # Solo full datasets
            rows.append({
                "Dataset": "2C Original (12,402 imgs)",
                "Variante": dataset_name,
                "Clases": "Enfermo/Normal",
                "Test_Size": 1245,
                "K_optimo": r2c.get("best_k", "?"),
                "Test_Accuracy": r2c.get("test_metrics", {}).get("accuracy", 0),
                "Macro_F1": r2c.get("test_metrics", {}).get("macro_f1", 0),
            })

    # Resultados de 2 clases comparable (mismas imgs que 3C)
    for name, result in results_2c_comparable.items():
        rows.append({
            "Dataset": "2C Comparable (6,725 imgs)",
            "Variante": name,
            "Clases": "Enfermo/Normal",
            "Test_Size": result.test_size,
            "K_optimo": result.k_optimization.best_k,
            "Test_Accuracy": result.classification_result.accuracy,
            "Macro_F1": result.classification_result.get_macro_f1(),
        })

    # Resultados de 3 clases
    for name, result in results_3c.items():
        rows.append({
            "Dataset": "3C (6,725 imgs)",
            "Variante": name,
            "Clases": "COVID/Normal/Viral",
            "Test_Size": result.test_size,
            "K_optimo": result.k_optimization.best_k,
            "Test_Accuracy": result.classification_result.accuracy,
            "Macro_F1": result.classification_result.get_macro_f1(),
        })

    return pd.DataFrame(rows)


def generate_comparison_figure(
    results_3c: Dict[str, ExperimentResult],
    results_2c_comparable: Dict[str, ExperimentResult],
    results_2c_original: Dict[str, Dict]
) -> plt.Figure:
    """
    Genera figura comparativa.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Comparacion: 2 Clases vs 3 Clases | Warped vs Original",
                 fontsize=14, fontweight='bold')

    # Recolectar datos para graficar
    conditions = []
    accuracies = []
    colors = []

    # 2C Original (12,402 imgs)
    if "full_original" in results_2c_original:
        conditions.append("2C-12K\nOriginal")
        accuracies.append(results_2c_original["full_original"].get("test_metrics", {}).get("accuracy", 0))
        colors.append('#3498DB')

    if "full_warped" in results_2c_original:
        conditions.append("2C-12K\nWarped")
        accuracies.append(results_2c_original["full_warped"].get("test_metrics", {}).get("accuracy", 0))
        colors.append('#2ECC71')

    # 2C Comparable (6,725 imgs)
    for name, result in sorted(results_2c_comparable.items()):
        if "original" in name:
            conditions.append("2C-6K\nOriginal")
            colors.append('#9B59B6')
        else:
            conditions.append("2C-6K\nWarped")
            colors.append('#E67E22')
        accuracies.append(result.classification_result.accuracy)

    # 3C (6,725 imgs)
    for name, result in sorted(results_3c.items()):
        if "original" in name:
            conditions.append("3C-6K\nOriginal")
            colors.append('#E74C3C')
        else:
            conditions.append("3C-6K\nWarped")
            colors.append('#F39C12')
        accuracies.append(result.classification_result.accuracy)

    # Panel izquierdo: Accuracy
    ax_acc = axes[0]
    bars = ax_acc.bar(range(len(conditions)), accuracies, color=colors, alpha=0.8, edgecolor='black')

    for bar, val in zip(bars, accuracies):
        ax_acc.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax_acc.set_xticks(range(len(conditions)))
    ax_acc.set_xticklabels(conditions, fontsize=9)
    ax_acc.set_ylabel('Accuracy', fontsize=11)
    ax_acc.set_title('Accuracy por Experimento', fontsize=12)
    ax_acc.set_ylim(0, 1.0)
    ax_acc.grid(True, alpha=0.3, axis='y')

    # Panel derecho: Mejora por warping
    ax_improve = axes[1]

    improvements = []
    labels = []
    imp_colors = []

    # 2C Original (12K)
    if "full_original" in results_2c_original and "full_warped" in results_2c_original:
        acc_o = results_2c_original["full_original"].get("test_metrics", {}).get("accuracy", 0)
        acc_w = results_2c_original["full_warped"].get("test_metrics", {}).get("accuracy", 0)
        improvements.append((acc_w - acc_o) * 100)
        labels.append("2C (12K imgs)")
        imp_colors.append('#2ECC71')

    # 2C Comparable (6K)
    orig_2c = next((r for n, r in results_2c_comparable.items() if "original" in n), None)
    warp_2c = next((r for n, r in results_2c_comparable.items() if "warped" in n), None)
    if orig_2c and warp_2c:
        acc_o = orig_2c.classification_result.accuracy
        acc_w = warp_2c.classification_result.accuracy
        improvements.append((acc_w - acc_o) * 100)
        labels.append("2C (6K imgs)")
        imp_colors.append('#E67E22')

    # 3C (6K)
    orig_3c = next((r for n, r in results_3c.items() if "original" in n), None)
    warp_3c = next((r for n, r in results_3c.items() if "warped" in n), None)
    if orig_3c and warp_3c:
        acc_o = orig_3c.classification_result.accuracy
        acc_w = warp_3c.classification_result.accuracy
        improvements.append((acc_w - acc_o) * 100)
        labels.append("3C (6K imgs)")
        imp_colors.append('#F39C12')

    bars = ax_improve.bar(labels, improvements, color=imp_colors, alpha=0.8, edgecolor='black')

    for bar, val in zip(bars, improvements):
        sign = "+" if val >= 0 else ""
        ax_improve.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{sign}{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax_improve.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_improve.set_ylabel('Mejora por Warping (puntos %)', fontsize=11)
    ax_improve.set_title('Mejora: Warped vs Original', fontsize=12)
    ax_improve.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def generate_comparison_figure_simple(
    results_3c: Dict[str, ExperimentResult],
    results_2c_original: Dict[str, Dict]
) -> plt.Figure:
    """
    Genera figura comparativa simplificada (sin 2C Comparable).
    Solo incluye: 2C (12K imgs) y 3C (6K imgs).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Comparacion: 2 Clases vs 3 Clases | Warped vs Original",
                 fontsize=14, fontweight='bold')

    # Recolectar datos para graficar
    conditions = []
    accuracies = []
    colors = []

    # 2C Original (12,402 imgs)
    if "full_original" in results_2c_original:
        conditions.append("2C-12K\nOriginal")
        accuracies.append(results_2c_original["full_original"].get("test_metrics", {}).get("accuracy", 0))
        colors.append('#3498DB')

    if "full_warped" in results_2c_original:
        conditions.append("2C-12K\nWarped")
        accuracies.append(results_2c_original["full_warped"].get("test_metrics", {}).get("accuracy", 0))
        colors.append('#2ECC71')

    # 3C (6,725 imgs)
    for name, result in sorted(results_3c.items()):
        if "original" in name:
            conditions.append("3C-6K\nOriginal")
            colors.append('#E74C3C')
        else:
            conditions.append("3C-6K\nWarped")
            colors.append('#F39C12')
        accuracies.append(result.classification_result.accuracy)

    # Panel izquierdo: Accuracy
    ax_acc = axes[0]
    bars = ax_acc.bar(range(len(conditions)), accuracies, color=colors, alpha=0.8, edgecolor='black')

    for bar, val in zip(bars, accuracies):
        ax_acc.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax_acc.set_xticks(range(len(conditions)))
    ax_acc.set_xticklabels(conditions, fontsize=9)
    ax_acc.set_ylabel('Accuracy', fontsize=11)
    ax_acc.set_title('Accuracy por Experimento', fontsize=12)
    ax_acc.set_ylim(0, 1.0)
    ax_acc.grid(True, alpha=0.3, axis='y')

    # Panel derecho: Mejora por warping
    ax_improve = axes[1]

    improvements = []
    labels = []
    imp_colors = []

    # 2C Original (12K)
    if "full_original" in results_2c_original and "full_warped" in results_2c_original:
        acc_o = results_2c_original["full_original"].get("test_metrics", {}).get("accuracy", 0)
        acc_w = results_2c_original["full_warped"].get("test_metrics", {}).get("accuracy", 0)
        improvements.append((acc_w - acc_o) * 100)
        labels.append("2C (12K imgs)")
        imp_colors.append('#2ECC71')

    # 3C (6K)
    orig_3c = next((r for n, r in results_3c.items() if "original" in n), None)
    warp_3c = next((r for n, r in results_3c.items() if "warped" in n), None)
    if orig_3c and warp_3c:
        acc_o = orig_3c.classification_result.accuracy
        acc_w = warp_3c.classification_result.accuracy
        improvements.append((acc_w - acc_o) * 100)
        labels.append("3C (6K imgs)")
        imp_colors.append('#F39C12')

    bars = ax_improve.bar(labels, improvements, color=imp_colors, alpha=0.8, edgecolor='black')

    for bar, val in zip(bars, improvements):
        sign = "+" if val >= 0 else ""
        ax_improve.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{sign}{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax_improve.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_improve.set_ylabel('Mejora por Warping (puntos %)', fontsize=11)
    ax_improve.set_title('Mejora: Warped vs Original', fontsize=12)
    ax_improve.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def generate_summary(
    results_3c: Dict[str, ExperimentResult],
    results_2c_comparable: Dict[str, ExperimentResult],
    results_2c_original: Dict[str, Dict]
) -> None:
    """
    Genera resumen final.
    """
    print()
    print("=" * 70)
    print("GENERANDO RESUMEN FINAL")
    print("=" * 70)

    # Tabla comparativa
    df_comparison = generate_comparison_table(results_3c, results_2c_comparable, results_2c_original)
    df_comparison.to_csv(OUTPUT_METRICS_DIR / "comparacion_2c_vs_3c.csv", index=False)
    print(f"Tabla guardada: {OUTPUT_METRICS_DIR / 'comparacion_2c_vs_3c.csv'}")

    # Figura comparativa completa
    fig = generate_comparison_figure(results_3c, results_2c_comparable, results_2c_original)
    fig.savefig(OUTPUT_FIGURES_DIR / "comparacion_final.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Figura guardada: {OUTPUT_FIGURES_DIR / 'comparacion_final.png'}")

    # Figura comparativa simplificada (sin 2C Comparable)
    fig_simple = generate_comparison_figure_simple(results_3c, results_2c_original)
    fig_simple.savefig(OUTPUT_FIGURES_DIR / "comparacion_simple.png",
                       dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_simple)
    print(f"Figura simplificada guardada: {OUTPUT_FIGURES_DIR / 'comparacion_simple.png'}")

    # JSON resumen
    summary_json = {
        "phase": 7,
        "description": "Comparacion 2 clases vs 3 clases (CORREGIDO)",
        "methodology": {
            "3class": "Pipeline completo desde CSV 01_full_balanced_3class_*.csv (6,725 imgs)",
            "2class_comparable": "Mismas imagenes que 3class, reagrupadas (COVID+Viral -> Enfermo)",
            "2class_original": "Resultados de Fase 6, CSV 02_full_balanced_2class_*.csv (12,402 imgs)"
        },
        "experiments_3class": {},
        "experiments_2class_comparable": {},
        "comparison": []
    }

    for name, result in results_3c.items():
        summary_json["experiments_3class"][name] = {
            "dataset_size": result.dataset_size,
            "test_size": result.test_size,
            "best_k": result.k_optimization.best_k,
            "val_accuracy": result.k_optimization.best_val_accuracy,
            "test_accuracy": result.classification_result.accuracy,
            "macro_f1": result.classification_result.get_macro_f1(),
            "confusion_matrix": result.classification_result.confusion_matrix.tolist()
        }

    for name, result in results_2c_comparable.items():
        summary_json["experiments_2class_comparable"][name] = {
            "dataset_size": result.dataset_size,
            "test_size": result.test_size,
            "best_k": result.k_optimization.best_k,
            "val_accuracy": result.k_optimization.best_val_accuracy,
            "test_accuracy": result.classification_result.accuracy,
            "macro_f1": result.classification_result.get_macro_f1(),
            "confusion_matrix": result.classification_result.confusion_matrix.tolist()
        }

    # Comparacion warped vs original
    comparison_data = {}

    # 2C Original (12K)
    if "full_original" in results_2c_original and "full_warped" in results_2c_original:
        acc_o = results_2c_original["full_original"].get("test_metrics", {}).get("accuracy", 0)
        acc_w = results_2c_original["full_warped"].get("test_metrics", {}).get("accuracy", 0)
        comparison_data["2class_12k"] = {
            "original_acc": acc_o,
            "warped_acc": acc_w,
            "improvement": acc_w - acc_o,
            "dataset_size": 12402,
            "test_size": 1245
        }

    # 2C Comparable (6K)
    orig_2c = next((r for n, r in results_2c_comparable.items() if "original" in n), None)
    warp_2c = next((r for n, r in results_2c_comparable.items() if "warped" in n), None)
    if orig_2c and warp_2c:
        comparison_data["2class_6k_comparable"] = {
            "original_acc": orig_2c.classification_result.accuracy,
            "warped_acc": warp_2c.classification_result.accuracy,
            "improvement": warp_2c.classification_result.accuracy - orig_2c.classification_result.accuracy,
            "dataset_size": 6725,
            "test_size": 680
        }

    # 3C (6K)
    orig_3c = next((r for n, r in results_3c.items() if "original" in n), None)
    warp_3c = next((r for n, r in results_3c.items() if "warped" in n), None)
    if orig_3c and warp_3c:
        comparison_data["3class_6k"] = {
            "original_acc": orig_3c.classification_result.accuracy,
            "warped_acc": warp_3c.classification_result.accuracy,
            "improvement": warp_3c.classification_result.accuracy - orig_3c.classification_result.accuracy,
            "dataset_size": 6725,
            "test_size": 680
        }

    summary_json["comparison"] = comparison_data

    with open(OUTPUT_METRICS_DIR / "summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"JSON guardado: {OUTPUT_METRICS_DIR / 'summary.json'}")

    # Imprimir tabla resumen
    print()
    print("-" * 80)
    print("TABLA COMPARATIVA FINAL")
    print("-" * 80)
    print()
    print(f"{'Experimento':<25} {'Dataset':<15} {'Test':>6} {'K':>4} {'Accuracy':>10} {'Macro F1':>10}")
    print("-" * 80)

    # 2C Original (12K)
    for dataset_name in ["full_original", "full_warped"]:
        if dataset_name in results_2c_original:
            r2c = results_2c_original[dataset_name]
            k = r2c.get("best_k", "?")
            acc = r2c.get("test_metrics", {}).get("accuracy", 0)
            f1 = r2c.get("test_metrics", {}).get("macro_f1", 0)
            label = "2C (12K imgs)"
            variant = "warped" if "warped" in dataset_name else "original"
            print(f"{label:<25} {variant:<15} {1245:>6} {k:>4} {acc:>9.2%} {f1:>10.4f}")

    print()

    # 2C Comparable (6K)
    for name, result in sorted(results_2c_comparable.items()):
        variant = "warped" if "warped" in name else "original"
        label = "2C Comparable (6K imgs)"
        print(f"{label:<25} {variant:<15} {result.test_size:>6} {result.k_optimization.best_k:>4} "
              f"{result.classification_result.accuracy:>9.2%} {result.classification_result.get_macro_f1():>10.4f}")

    print()

    # 3C (6K)
    for name, result in sorted(results_3c.items()):
        variant = "warped" if "warped" in name else "original"
        label = "3C (6K imgs)"
        print(f"{label:<25} {variant:<15} {result.test_size:>6} {result.k_optimization.best_k:>4} "
              f"{result.classification_result.accuracy:>9.2%} {result.classification_result.get_macro_f1():>10.4f}")

    print("-" * 80)

    # Analisis de mejora por warping
    print()
    print("MEJORA POR WARPING:")
    print()

    for exp_name, data in comparison_data.items():
        acc_o = data["original_acc"]
        acc_w = data["warped_acc"]
        diff = data["improvement"]
        sign = "+" if diff > 0 else ""
        print(f"  {exp_name}: {acc_o:.2%} -> {acc_w:.2%} ({sign}{diff:.2%})")

    print()


def main():
    """Funcion principal."""
    print("=" * 70)
    print("FASE 7: COMPARACION 2 CLASES vs 3 CLASES (CORREGIDO)")
    print("=" * 70)
    print()
    print("Pipeline: Cargar imagenes -> PCA -> Z-score -> Fisher -> KNN")
    print()
    print("METODOLOGIA CORREGIDA:")
    print("  - 3 clases: Carga directa desde 01_full_balanced_3class_*.csv (6,725 imgs)")
    print("  - 2C Comparable: Mismas imagenes, reagrupadas a 2 clases")
    print("  - 2C Original: Resultados de Fase 6 (12,402 imgs)")
    print()

    # Crear directorios
    OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar resultados de 2 clases originales (Fase 6)
    results_2c_original = load_2class_results()
    print(f"Resultados 2C originales cargados: {list(results_2c_original.keys())}")

    # === EXPERIMENTOS 3 CLASES ===
    results_3c: Dict[str, ExperimentResult] = {}

    # 3C Warped
    if CSV_3CLASS_WARPED.exists():
        result = run_full_pipeline(
            csv_path=CSV_3CLASS_WARPED,
            scenario="3class",
            experiment_name="3class_warped",
            use_mask=True,
            verbose=True
        )
        results_3c["3class_warped"] = result

    # 3C Original
    if CSV_3CLASS_ORIGINAL.exists():
        result = run_full_pipeline(
            csv_path=CSV_3CLASS_ORIGINAL,
            scenario="3class",
            experiment_name="3class_original",
            use_mask=False,
            verbose=True
        )
        results_3c["3class_original"] = result

    # === EXPERIMENTOS 2C COMPARABLE (mismas imagenes que 3C) ===
    results_2c_comparable: Dict[str, ExperimentResult] = {}

    # 2C Comparable Warped
    if CSV_3CLASS_WARPED.exists():
        result = run_full_pipeline(
            csv_path=CSV_3CLASS_WARPED,
            scenario="2class",  # Reagrupa COVID+Viral -> Enfermo
            experiment_name="2class_comparable_warped",
            use_mask=True,
            verbose=True
        )
        results_2c_comparable["2class_comparable_warped"] = result

    # 2C Comparable Original
    if CSV_3CLASS_ORIGINAL.exists():
        result = run_full_pipeline(
            csv_path=CSV_3CLASS_ORIGINAL,
            scenario="2class",  # Reagrupa COVID+Viral -> Enfermo
            experiment_name="2class_comparable_original",
            use_mask=False,
            verbose=True
        )
        results_2c_comparable["2class_comparable_original"] = result

    # Generar resumen final
    generate_summary(results_3c, results_2c_comparable, results_2c_original)

    print()
    print("=" * 70)
    print("FASE 7 COMPLETADA")
    print("=" * 70)
    print()
    print("VERIFICACION:")
    for name, result in results_3c.items():
        print(f"  {name}: Test size = {result.test_size} (esperado: 680)")
    for name, result in results_2c_comparable.items():
        print(f"  {name}: Test size = {result.test_size} (esperado: 680)")
    print()


if __name__ == "__main__":
    main()
