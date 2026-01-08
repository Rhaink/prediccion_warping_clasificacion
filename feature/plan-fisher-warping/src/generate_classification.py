#!/usr/bin/env python3
"""
Script para generar resultados de clasificacion KNN para los 4 datasets.

FASE 6 DEL PIPELINE FISHER-WARPING
===================================

Este script:
1. Carga caracteristicas amplificadas de Fase 5
2. Busca K optimo usando validacion
3. Entrena KNN con K optimo
4. Evalua en test
5. Genera matrices de confusion y comparaciones

Datasets procesados:
- full_warped: Dataset completo con imagenes warped
- full_original: Dataset completo con imagenes originales
- manual_warped: Dataset manual con imagenes warped
- manual_original: Dataset manual con imagenes originales

Uso:
    python src/generate_classification.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
import gc
from typing import Dict, List, Tuple

# Agregar directorio src al path
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from classifier import (
    KNNClassifier,
    find_best_k,
    evaluate_classifier,
    plot_confusion_matrix,
    plot_k_optimization,
    plot_metrics_comparison,
    save_classification_results,
    save_predictions,
    load_amplified_features,
    ClassificationResult,
    KOptimizationResult
)


# ============================================================================
# CONFIGURACION
# ============================================================================

# Directorios
BASE_DIR = Path(__file__).parent.parent
FISHER_METRICS_DIR = BASE_DIR / "results" / "metrics" / "phase5_fisher"
OUTPUT_METRICS_DIR = BASE_DIR / "results" / "metrics" / "phase6_classification"
OUTPUT_FIGURES_DIR = BASE_DIR / "results" / "figures" / "phase6_classification"

# Datasets a procesar
DATASETS = [
    "full_warped",
    "full_original",
    "manual_warped",
    "manual_original"
]

# Nombres de clases (binario: Enfermo vs Normal)
CLASS_NAMES = ["Enfermo", "Normal"]

# Valores de K a probar (impares para evitar empates)
K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21, 31, 41, 51]


def load_dataset(
    dataset_name: str
) -> Tuple[np.ndarray, np.ndarray, List[str],
           np.ndarray, np.ndarray, List[str],
           np.ndarray, np.ndarray, List[str]]:
    """
    Carga train, val, test para un dataset.

    Args:
        dataset_name: Nombre del dataset (ej. "full_warped")

    Returns:
        Tuple de (X_train, y_train, ids_train,
                  X_val, y_val, ids_val,
                  X_test, y_test, ids_test)
    """
    # Rutas de los archivos
    train_path = FISHER_METRICS_DIR / f"{dataset_name}_train_amplified.csv"
    val_path = FISHER_METRICS_DIR / f"{dataset_name}_val_amplified.csv"
    test_path = FISHER_METRICS_DIR / f"{dataset_name}_test_amplified.csv"

    # Verificar que existen
    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"No se encontro: {path}")

    # Cargar datos
    X_train, y_train, ids_train = load_amplified_features(train_path)
    X_val, y_val, ids_val = load_amplified_features(val_path)
    X_test, y_test, ids_test = load_amplified_features(test_path)

    return (X_train, y_train, ids_train,
            X_val, y_val, ids_val,
            X_test, y_test, ids_test)


def process_dataset(
    dataset_name: str,
    verbose: bool = True
) -> Tuple[ClassificationResult, KOptimizationResult]:
    """
    Procesa un dataset completo: optimiza K, entrena, evalua.

    Args:
        dataset_name: Nombre del dataset
        verbose: Si True, imprime progreso

    Returns:
        Tuple de (ClassificationResult, KOptimizationResult)
    """
    if verbose:
        print()
        print("=" * 70)
        print(f"PROCESANDO: {dataset_name.upper()}")
        print("=" * 70)

    # Cargar datos
    (X_train, y_train, ids_train,
     X_val, y_val, ids_val,
     X_test, y_test, ids_test) = load_dataset(dataset_name)

    if verbose:
        print(f"\nDatos cargados:")
        print(f"  Train: {X_train.shape[0]} muestras x {X_train.shape[1]} caracteristicas")
        print(f"  Val:   {X_val.shape[0]} muestras")
        print(f"  Test:  {X_test.shape[0]} muestras")

        # Distribucion de clases
        for split_name, labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            unique, counts = np.unique(labels, return_counts=True)
            dist = {CLASS_NAMES[u]: c for u, c in zip(unique, counts)}
            print(f"  {split_name} distribucion: {dist}")

    # Liberar memoria de ids que no usaremos mas adelante
    del ids_train, ids_val
    gc.collect()

    # Ajustar K_VALUES al tamano del dataset
    max_k = min(X_train.shape[0], max(K_VALUES))
    k_values = [k for k in K_VALUES if k <= max_k]

    if verbose:
        print(f"\nBuscando K optimo...")
        print(f"  Valores a probar: {k_values}")

    # Optimizar K
    opt_result = find_best_k(
        X_train, y_train,
        X_val, y_val,
        k_values=k_values,
        verbose=verbose
    )

    # Crear directorio de salida
    dataset_output_dir = OUTPUT_FIGURES_DIR / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    # Guardar grafico de optimizacion de K
    fig_k = plot_k_optimization(
        opt_result,
        output_path=dataset_output_dir / "k_optimization.png"
    )
    plt.close(fig_k)
    del fig_k
    gc.collect()

    # Liberar datos de validacion (ya no los necesitamos)
    del X_val, y_val
    gc.collect()

    # Entrenar con K optimo
    if verbose:
        print(f"\nEntrenando KNN con K={opt_result.best_k}...")

    knn = KNNClassifier(k=opt_result.best_k)
    knn.fit(X_train, y_train, verbose=False)

    # Evaluar en test
    if verbose:
        print(f"\nEvaluando en conjunto de TEST...")

    test_result = evaluate_classifier(
        knn, X_test, y_test, CLASS_NAMES, verbose=verbose
    )

    # Guardar matriz de confusion
    fig_cm = plot_confusion_matrix(
        test_result,
        output_path=dataset_output_dir / "confusion_matrix.png",
        normalize=False,
        title=f"Matriz de Confusion - {dataset_name}\nK={opt_result.best_k}"
    )
    plt.close(fig_cm)
    del fig_cm

    # Guardar matriz de confusion normalizada
    fig_cm_norm = plot_confusion_matrix(
        test_result,
        output_path=dataset_output_dir / "confusion_matrix_normalized.png",
        normalize=True,
        title=f"Matriz de Confusion (%) - {dataset_name}\nK={opt_result.best_k}"
    )
    plt.close(fig_cm_norm)
    del fig_cm_norm

    # Guardar resultados en CSV
    save_classification_results(
        test_result,
        dataset_name,
        OUTPUT_METRICS_DIR / f"{dataset_name}_results.csv"
    )

    # Guardar predicciones individuales
    save_predictions(
        test_result,
        ids_test,
        OUTPUT_METRICS_DIR / f"{dataset_name}_predictions.csv"
    )

    # Liberar datos grandes antes de retornar
    del X_train, y_train, X_test, y_test, ids_test, knn
    gc.collect()

    return test_result, opt_result


def generate_comparison_figures(
    results: Dict[str, ClassificationResult]
) -> None:
    """
    Genera figuras comparativas entre datasets.

    Args:
        results: Dict {dataset_name: ClassificationResult}
    """
    print()
    print("=" * 70)
    print("GENERANDO FIGURAS COMPARATIVAS")
    print("=" * 70)

    comparisons_dir = OUTPUT_FIGURES_DIR / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    # 1. Comparacion de metricas entre los 4 datasets
    fig_metrics = plot_metrics_comparison(
        results,
        output_path=comparisons_dir / "metrics_4datasets.png"
    )
    plt.close(fig_metrics)

    # 2. Comparacion Warped vs Original (Full)
    if "full_warped" in results and "full_original" in results:
        fig_full = plot_metrics_comparison(
            {
                "Full Warped": results["full_warped"],
                "Full Original": results["full_original"]
            },
            output_path=comparisons_dir / "warped_vs_original_full.png"
        )
        plt.close(fig_full)

    # 3. Comparacion Warped vs Original (Manual)
    if "manual_warped" in results and "manual_original" in results:
        fig_manual = plot_metrics_comparison(
            {
                "Manual Warped": results["manual_warped"],
                "Manual Original": results["manual_original"]
            },
            output_path=comparisons_dir / "warped_vs_original_manual.png"
        )
        plt.close(fig_manual)

    # 4. Figura resumen: matrices de confusion 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Matrices de Confusion - Comparacion de Datasets",
                 fontsize=14, fontweight='bold')

    dataset_order = ["full_warped", "full_original", "manual_warped", "manual_original"]
    titles = ["Full Warped", "Full Original", "Manual Warped", "Manual Original"]

    for idx, (dataset, title) in enumerate(zip(dataset_order, titles)):
        if dataset not in results:
            continue

        row, col = idx // 2, idx % 2
        ax = axes[row, col]

        res = results[dataset]
        cm = res.confusion_matrix

        # Visualizar
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

        # Agregar texto
        thresh = cm.max() / 2.
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                ax.text(j, i, f'{int(cm[i, j])}',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=12)

        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel('Predicha')
        ax.set_ylabel('Real')
        ax.set_title(f'{title}\nAcc={res.accuracy:.2%}, K={res.k}')

    plt.tight_layout()
    fig.savefig(comparisons_dir / "confusion_matrices_4datasets.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Matrices comparativas guardadas en: {comparisons_dir / 'confusion_matrices_4datasets.png'}")


def generate_summary(
    results: Dict[str, ClassificationResult],
    opt_results: Dict[str, KOptimizationResult]
) -> None:
    """
    Genera resumen final en JSON y CSV.

    Args:
        results: Dict {dataset_name: ClassificationResult}
        opt_results: Dict {dataset_name: KOptimizationResult}
    """
    print()
    print("=" * 70)
    print("GENERANDO RESUMEN FINAL")
    print("=" * 70)

    # Crear CSV comparativo
    summary_data = []

    for dataset_name in DATASETS:
        if dataset_name not in results:
            continue

        res = results[dataset_name]
        opt = opt_results[dataset_name]

        row = {
            "dataset": dataset_name,
            "is_warped": "warped" in dataset_name,
            "is_manual": "manual" in dataset_name,
            "best_k": opt.best_k,
            "val_accuracy": opt.best_val_accuracy,
            "test_accuracy": res.accuracy,
            "test_macro_f1": res.get_macro_f1(),
            "test_weighted_f1": res.get_weighted_f1(),
        }

        for cls in sorted(res.precision.keys()):
            class_name = CLASS_NAMES[cls]
            row[f"precision_{class_name}"] = res.precision[cls]
            row[f"recall_{class_name}"] = res.recall[cls]
            row[f"f1_{class_name}"] = res.f1[cls]

        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    df.to_csv(OUTPUT_METRICS_DIR / "comparison_summary.csv", index=False)
    print(f"Resumen guardado en: {OUTPUT_METRICS_DIR / 'comparison_summary.csv'}")

    # Crear JSON con todos los detalles
    summary_json = {
        "phase": 6,
        "description": "Clasificacion KNN con caracteristicas amplificadas por Fisher",
        "datasets": {}
    }

    for dataset_name in DATASETS:
        if dataset_name not in results:
            continue

        res = results[dataset_name]
        opt = opt_results[dataset_name]

        summary_json["datasets"][dataset_name] = {
            "best_k": opt.best_k,
            "k_values_tried": opt.k_values,
            "val_accuracies": opt.val_accuracies,
            "test_metrics": {
                "n_samples": res.n_samples,
                "accuracy": res.accuracy,
                "macro_f1": res.get_macro_f1(),
                "weighted_f1": res.get_weighted_f1(),
                "precision": {CLASS_NAMES[k]: v for k, v in res.precision.items()},
                "recall": {CLASS_NAMES[k]: v for k, v in res.recall.items()},
                "f1": {CLASS_NAMES[k]: v for k, v in res.f1.items()},
                "confusion_matrix": res.confusion_matrix.tolist()
            }
        }

    with open(OUTPUT_METRICS_DIR / "summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"Resumen JSON guardado en: {OUTPUT_METRICS_DIR / 'summary.json'}")

    # Imprimir tabla resumen
    print()
    print("-" * 70)
    print("RESUMEN DE RESULTADOS")
    print("-" * 70)
    print()
    print(f"{'Dataset':<20} {'K':>4} {'Val Acc':>10} {'Test Acc':>10} {'Macro F1':>10}")
    print("-" * 70)

    for dataset_name in DATASETS:
        if dataset_name not in results:
            continue

        res = results[dataset_name]
        opt = opt_results[dataset_name]

        print(f"{dataset_name:<20} {opt.best_k:>4} "
              f"{opt.best_val_accuracy:>10.4f} {res.accuracy:>10.4f} "
              f"{res.get_macro_f1():>10.4f}")

    print("-" * 70)

    # Comparacion warped vs original
    print()
    print("COMPARACION WARPED vs ORIGINAL:")
    print()

    for prefix in ["full", "manual"]:
        warped_name = f"{prefix}_warped"
        original_name = f"{prefix}_original"

        if warped_name in results and original_name in results:
            warped_acc = results[warped_name].accuracy
            original_acc = results[original_name].accuracy
            diff = warped_acc - original_acc
            sign = "+" if diff > 0 else ""

            print(f"  {prefix.upper()}:")
            print(f"    Warped:   {warped_acc:.4f} ({warped_acc*100:.2f}%)")
            print(f"    Original: {original_acc:.4f} ({original_acc*100:.2f}%)")
            print(f"    Diferencia: {sign}{diff:.4f} ({sign}{diff*100:.2f}%)")
            print()


def main():
    """Funcion principal."""
    print("=" * 70)
    print("FASE 6: CLASIFICACION KNN")
    print("=" * 70)
    print()
    print(f"Directorio de entrada: {FISHER_METRICS_DIR}")
    print(f"Directorio de salida (metricas): {OUTPUT_METRICS_DIR}")
    print(f"Directorio de salida (figuras): {OUTPUT_FIGURES_DIR}")
    print()
    print("OPTIMIZACION DE MEMORIA: Procesando datasets uno por uno")
    print("=" * 70)

    # Crear directorios de salida
    OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Procesar cada dataset UNO POR UNO con liberacion de memoria
    all_results: Dict[str, ClassificationResult] = {}
    all_opt_results: Dict[str, KOptimizationResult] = {}

    for idx, dataset_name in enumerate(DATASETS, 1):
        print()
        print(f"\n>>> PROCESANDO DATASET {idx}/{len(DATASETS)}: {dataset_name}")
        print(f">>> Memoria sera liberada al finalizar este dataset")
        print()

        try:
            # Procesar dataset
            result, opt_result = process_dataset(dataset_name)

            # Guardar resultados minimos necesarios
            # (evitamos guardar objetos grandes innecesarios)
            all_results[dataset_name] = result
            all_opt_results[dataset_name] = opt_result

            # CRITICO: Liberar memoria explicitamente
            print()
            print(f"[MEMORIA] Liberando memoria tras procesar {dataset_name}...")

            # Forzar recoleccion de basura
            gc.collect()

            print(f"[MEMORIA] Memoria liberada. Continuando...")

        except FileNotFoundError as e:
            print(f"AVISO: No se pudo procesar {dataset_name}: {e}")
            continue
        except Exception as e:
            print(f"ERROR al procesar {dataset_name}: {e}")
            print("Continuando con siguiente dataset...")
            continue

    # Generar figuras comparativas
    print()
    print("=" * 70)
    print("GENERANDO FIGURAS COMPARATIVAS")
    print("=" * 70)

    if len(all_results) > 1:
        generate_comparison_figures(all_results)

    # Liberar memoria antes del resumen
    gc.collect()

    # Generar resumen
    generate_summary(all_results, all_opt_results)

    print()
    print("=" * 70)
    print("FASE 6 COMPLETADA")
    print("=" * 70)


if __name__ == "__main__":
    main()
