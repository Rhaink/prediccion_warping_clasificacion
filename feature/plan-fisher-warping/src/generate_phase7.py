#!/usr/bin/env python3
"""
Script para generar resultados de Fase 7: Comparacion 2 clases vs 3 clases.

FASE 7 DEL PIPELINE FISHER-WARPING
===================================

Este script:
1. Reutiliza caracteristicas PCA estandarizadas de Fase 4
2. Extrae clase original del image_id (COVID, Normal, Viral_Pneumonia)
3. Calcula Fisher multiclase (pairwise) para 3 clases
4. Amplifica caracteristicas con Fisher
5. Entrena KNN con 3 clases
6. Compara con resultados de 2 clases de Fase 6
7. Genera tabla comparativa final

NOTA SOBRE FISHER MULTICLASE:
Para 3 clases usamos Fisher pairwise (promedio de pares):
    J_final = (J_COVID_Normal + J_COVID_Viral + J_Normal_Viral) / 3

Esto es una extension natural del criterio binario del asesor.

Uso:
    python src/generate_phase7.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
from typing import Dict, List, Tuple

# Agregar directorio src al path
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

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
PHASE4_METRICS_DIR = BASE_DIR / "results" / "metrics" / "phase4_features"
PHASE6_METRICS_DIR = BASE_DIR / "results" / "metrics" / "phase6_classification"
OUTPUT_METRICS_DIR = BASE_DIR / "results" / "metrics" / "phase7_comparison"
OUTPUT_FIGURES_DIR = BASE_DIR / "results" / "figures" / "phase7_comparison"

# Datasets a procesar
DATASETS = [
    "full_warped",
    "full_original",
    "manual_warped",
    "manual_original"
]

# Nombres de clases
CLASS_NAMES_3C = ["COVID", "Normal", "Viral_Pneumonia"]
CLASS_NAMES_2C = ["Enfermo", "Normal"]

# Mapeo de clase original a indice (3 clases)
LABEL_MAP_3C = {"COVID": 0, "Normal": 1, "Viral_Pneumonia": 2}

# Valores de K a probar
K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21, 31, 41, 51]


def extract_original_class(image_id: str) -> str:
    """
    Extrae la clase original del image_id.

    Los image_ids tienen formato: "ClassName-Number"
    Ejemplos:
        "COVID-1234" -> "COVID"
        "Normal-567" -> "Normal"
        "Viral_Pneumonia-890" -> "Viral_Pneumonia"
        "Viral Pneumonia-890" -> "Viral_Pneumonia" (con espacio)
    """
    if image_id.startswith("Viral_Pneumonia") or image_id.startswith("Viral Pneumonia"):
        return "Viral_Pneumonia"
    elif image_id.startswith("COVID"):
        return "COVID"
    elif image_id.startswith("Normal"):
        return "Normal"
    else:
        raise ValueError(f"No se pudo extraer clase de: {image_id}")


def load_features_3class(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Carga caracteristicas de Fase 4 y reasigna etiquetas a 3 clases.

    Args:
        csv_path: Ruta al archivo CSV de features (Fase 4)

    Returns:
        Tuple de (features, labels_3class, image_ids)
    """
    df = pd.read_csv(csv_path)

    # Extraer columnas de features (PC1, PC2, ... - SIN _amp)
    feature_cols = [col for col in df.columns if col.startswith('PC') and not col.endswith('_amp')]
    features = df[feature_cols].values.astype(np.float32)

    # Extraer clase original del image_id
    original_classes = [extract_original_class(img_id) for img_id in df['image_id']]

    # Crear etiquetas numericas para 3 clases
    labels = np.array([LABEL_MAP_3C[cls] for cls in original_classes], dtype=np.int32)

    image_ids = df['image_id'].tolist()

    return features, labels, image_ids


def load_dataset_3class(dataset_name: str) -> Dict:
    """
    Carga train, val, test para un dataset con 3 clases.

    Args:
        dataset_name: Nombre del dataset (ej. "full_warped")

    Returns:
        Dict con X_train, y_train, ids_train, X_val, y_val, ids_val, X_test, y_test, ids_test
    """
    data = {}

    for split in ["train", "val", "test"]:
        csv_path = PHASE4_METRICS_DIR / f"{dataset_name}_{split}_features.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"No se encontro: {csv_path}")

        X, y, ids = load_features_3class(csv_path)
        data[f"X_{split}"] = X
        data[f"y_{split}"] = y
        data[f"ids_{split}"] = ids

    return data


def process_dataset_3class(
    dataset_name: str,
    verbose: bool = True
) -> Tuple[ClassificationResult, KOptimizationResult, Dict]:
    """
    Procesa un dataset con 3 clases: Fisher multiclase + KNN.

    Args:
        dataset_name: Nombre del dataset
        verbose: Si True, imprime progreso

    Returns:
        Tuple de (ClassificationResult, KOptimizationResult, fisher_info)
    """
    if verbose:
        print()
        print("=" * 70)
        print(f"PROCESANDO 3 CLASES: {dataset_name.upper()}")
        print("=" * 70)

    # Cargar datos con 3 clases
    data = load_dataset_3class(dataset_name)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    ids_test = data["ids_test"]

    if verbose:
        print(f"\nDatos cargados:")
        print(f"  Train: {X_train.shape[0]} muestras x {X_train.shape[1]} caracteristicas")
        print(f"  Val:   {X_val.shape[0]} muestras")
        print(f"  Test:  {X_test.shape[0]} muestras")

        for split_name, labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            unique, counts = np.unique(labels, return_counts=True)
            dist = {CLASS_NAMES_3C[u]: c for u, c in zip(unique, counts)}
            print(f"  {split_name} distribucion: {dist}")

    # === FISHER MULTICLASE ===
    if verbose:
        print()

    fisher = FisherRatioMulticlass()
    fisher.fit(X_train, y_train, class_names=CLASS_NAMES_3C, verbose=verbose)

    # Amplificar caracteristicas
    X_train_amp = fisher.amplify(X_train)
    X_val_amp = fisher.amplify(X_val)
    X_test_amp = fisher.amplify(X_test)

    # Guardar Fisher ratios
    fisher_info = {
        "fisher_ratios": fisher.fisher_ratios_.tolist(),
        "n_classes": fisher.n_classes_,
        "class_names": CLASS_NAMES_3C
    }

    # Crear directorio de salida
    dataset_output_dir = OUTPUT_FIGURES_DIR / f"{dataset_name}_3class"
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    # Guardar figura de Fisher ratios
    fig_fisher = plot_fisher_ratios(
        fisher.get_result(),
        output_path=dataset_output_dir / "fisher_ratios.png",
        top_k=10
    )
    plt.close(fig_fisher)

    # === KNN ===
    # Ajustar K_VALUES al tamano del dataset
    max_k = min(X_train.shape[0], max(K_VALUES))
    k_values = [k for k in K_VALUES if k <= max_k]

    if verbose:
        print(f"\nBuscando K optimo...")
        print(f"  Valores a probar: {k_values}")

    # Optimizar K usando caracteristicas amplificadas
    opt_result = find_best_k(
        X_train_amp, y_train,
        X_val_amp, y_val,
        k_values=k_values,
        verbose=verbose
    )

    # Guardar grafico de optimizacion de K
    fig_k = plot_k_optimization(
        opt_result,
        output_path=dataset_output_dir / "k_optimization.png"
    )
    plt.close(fig_k)

    # Entrenar con K optimo
    if verbose:
        print(f"\nEntrenando KNN con K={opt_result.best_k}...")

    knn = KNNClassifier(k=opt_result.best_k)
    knn.fit(X_train_amp, y_train, verbose=False)

    # Evaluar en test
    if verbose:
        print(f"\nEvaluando en conjunto de TEST...")

    test_result = evaluate_classifier(
        knn, X_test_amp, y_test, CLASS_NAMES_3C, verbose=verbose
    )

    # Guardar matriz de confusion
    fig_cm = plot_confusion_matrix(
        test_result,
        output_path=dataset_output_dir / "confusion_matrix.png",
        normalize=False,
        title=f"Matriz de Confusion (3 clases) - {dataset_name}\nK={opt_result.best_k}"
    )
    plt.close(fig_cm)

    fig_cm_norm = plot_confusion_matrix(
        test_result,
        output_path=dataset_output_dir / "confusion_matrix_normalized.png",
        normalize=True,
        title=f"Matriz de Confusion (%) - {dataset_name}\nK={opt_result.best_k}"
    )
    plt.close(fig_cm_norm)

    # Guardar resultados
    save_classification_results(
        test_result,
        f"{dataset_name}_3class",
        OUTPUT_METRICS_DIR / f"{dataset_name}_3class_results.csv"
    )

    save_predictions(
        test_result,
        ids_test,
        OUTPUT_METRICS_DIR / f"{dataset_name}_3class_predictions.csv"
    )

    return test_result, opt_result, fisher_info


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
    results_3c: Dict[str, ClassificationResult],
    results_2c: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Genera tabla comparativa 2C vs 3C.
    """
    rows = []

    for dataset in DATASETS:
        is_warped = "warped" in dataset
        is_manual = "manual" in dataset

        # Resultados 2 clases
        if dataset in results_2c:
            r2c = results_2c[dataset]
            rows.append({
                "Dataset": dataset,
                "Escenario": "2 clases",
                "Clases": "Enfermo/Normal",
                "K_optimo": r2c.get("best_k", "N/A"),
                "Test_Accuracy": r2c.get("test_metrics", {}).get("accuracy", 0),
                "Macro_F1": r2c.get("test_metrics", {}).get("macro_f1", 0),
                "is_warped": is_warped,
                "is_manual": is_manual
            })

        # Resultados 3 clases
        if dataset in results_3c:
            r3c = results_3c[dataset]
            rows.append({
                "Dataset": dataset,
                "Escenario": "3 clases",
                "Clases": "COVID/Normal/Viral",
                "K_optimo": r3c.k,
                "Test_Accuracy": r3c.accuracy,
                "Macro_F1": r3c.get_macro_f1(),
                "is_warped": is_warped,
                "is_manual": is_manual
            })

    return pd.DataFrame(rows)


def generate_final_comparison_figure(
    results_3c: Dict[str, ClassificationResult],
    results_2c: Dict[str, Dict]
) -> plt.Figure:
    """
    Genera figura comparativa final.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comparacion: 2 Clases vs 3 Clases | Warped vs Original",
                 fontsize=14, fontweight='bold')

    scenarios = [
        ("full", "Dataset Completo (Full)"),
        ("manual", "Dataset Manual")
    ]

    for row_idx, (prefix, title) in enumerate(scenarios):
        warped_name = f"{prefix}_warped"
        original_name = f"{prefix}_original"

        # Panel izquierdo: Accuracy
        ax_acc = axes[row_idx, 0]

        conditions = ["2C\nOriginal", "2C\nWarped", "3C\nOriginal", "3C\nWarped"]
        accuracies = []

        # 2C Original
        acc = results_2c.get(original_name, {}).get("test_metrics", {}).get("accuracy", 0)
        accuracies.append(acc)

        # 2C Warped
        acc = results_2c.get(warped_name, {}).get("test_metrics", {}).get("accuracy", 0)
        accuracies.append(acc)

        # 3C Original
        acc = results_3c[original_name].accuracy if original_name in results_3c else 0
        accuracies.append(acc)

        # 3C Warped
        acc = results_3c[warped_name].accuracy if warped_name in results_3c else 0
        accuracies.append(acc)

        colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12']
        bars = ax_acc.bar(conditions, accuracies, color=colors, alpha=0.8, edgecolor='black')

        for bar, val in zip(bars, accuracies):
            if val > 0:
                ax_acc.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax_acc.set_ylabel('Accuracy', fontsize=11)
        ax_acc.set_title(f'{title} - Accuracy', fontsize=12)
        ax_acc.set_ylim(0, 1.0)
        ax_acc.grid(True, alpha=0.3, axis='y')

        # Panel derecho: Macro F1
        ax_f1 = axes[row_idx, 1]

        f1_scores = []

        # 2C Original
        f1 = results_2c.get(original_name, {}).get("test_metrics", {}).get("macro_f1", 0)
        f1_scores.append(f1)

        # 2C Warped
        f1 = results_2c.get(warped_name, {}).get("test_metrics", {}).get("macro_f1", 0)
        f1_scores.append(f1)

        # 3C Original
        f1 = results_3c[original_name].get_macro_f1() if original_name in results_3c else 0
        f1_scores.append(f1)

        # 3C Warped
        f1 = results_3c[warped_name].get_macro_f1() if warped_name in results_3c else 0
        f1_scores.append(f1)

        bars = ax_f1.bar(conditions, f1_scores, color=colors, alpha=0.8, edgecolor='black')

        for bar, val in zip(bars, f1_scores):
            if val > 0:
                ax_f1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax_f1.set_ylabel('Macro F1', fontsize=11)
        ax_f1.set_title(f'{title} - Macro F1', fontsize=12)
        ax_f1.set_ylim(0, 1.0)
        ax_f1.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def generate_summary(
    results_3c: Dict[str, ClassificationResult],
    opt_results_3c: Dict[str, KOptimizationResult],
    fisher_infos: Dict[str, Dict],
    results_2c: Dict[str, Dict]
) -> None:
    """
    Genera resumen final.
    """
    print()
    print("=" * 70)
    print("GENERANDO RESUMEN FINAL")
    print("=" * 70)

    # Tabla comparativa
    df_comparison = generate_comparison_table(results_3c, results_2c)
    df_comparison.to_csv(OUTPUT_METRICS_DIR / "comparacion_2c_vs_3c.csv", index=False)
    print(f"Tabla guardada: {OUTPUT_METRICS_DIR / 'comparacion_2c_vs_3c.csv'}")

    # Figura comparativa
    fig = generate_final_comparison_figure(results_3c, results_2c)
    fig.savefig(OUTPUT_FIGURES_DIR / "comparacion_final.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Figura guardada: {OUTPUT_FIGURES_DIR / 'comparacion_final.png'}")

    # JSON resumen
    summary_json = {
        "phase": 7,
        "description": "Comparacion 2 clases vs 3 clases con Fisher multiclase (pairwise)",
        "datasets_3class": {},
        "comparison": []
    }

    for dataset in DATASETS:
        if dataset in results_3c:
            r3c = results_3c[dataset]
            opt = opt_results_3c[dataset]

            summary_json["datasets_3class"][dataset] = {
                "best_k": opt.best_k,
                "val_accuracy": opt.best_val_accuracy,
                "test_accuracy": r3c.accuracy,
                "macro_f1": r3c.get_macro_f1(),
                "fisher_info": fisher_infos.get(dataset, {}),
                "confusion_matrix": r3c.confusion_matrix.tolist()
            }

    # Comparacion warped vs original
    for prefix in ["full", "manual"]:
        warped = f"{prefix}_warped"
        original = f"{prefix}_original"

        comparison = {"dataset_type": prefix}

        # 2 clases
        if warped in results_2c and original in results_2c:
            acc_w = results_2c[warped].get("test_metrics", {}).get("accuracy", 0)
            acc_o = results_2c[original].get("test_metrics", {}).get("accuracy", 0)
            comparison["2class_warped_acc"] = acc_w
            comparison["2class_original_acc"] = acc_o
            comparison["2class_improvement"] = acc_w - acc_o

        # 3 clases
        if warped in results_3c and original in results_3c:
            acc_w = results_3c[warped].accuracy
            acc_o = results_3c[original].accuracy
            comparison["3class_warped_acc"] = acc_w
            comparison["3class_original_acc"] = acc_o
            comparison["3class_improvement"] = acc_w - acc_o

        summary_json["comparison"].append(comparison)

    with open(OUTPUT_METRICS_DIR / "summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"JSON guardado: {OUTPUT_METRICS_DIR / 'summary.json'}")

    # Imprimir tabla resumen
    print()
    print("-" * 80)
    print("TABLA COMPARATIVA FINAL: 2 CLASES vs 3 CLASES")
    print("-" * 80)
    print()
    print(f"{'Dataset':<18} {'Escenario':<12} {'K':>4} {'Test Acc':>12} {'Macro F1':>10}")
    print("-" * 80)

    for dataset in DATASETS:
        # 2 clases
        if dataset in results_2c:
            r2c = results_2c[dataset]
            k = r2c.get("best_k", "?")
            acc = r2c.get("test_metrics", {}).get("accuracy", 0)
            f1 = r2c.get("test_metrics", {}).get("macro_f1", 0)
            print(f"{dataset:<18} {'2 clases':<12} {k:>4} {acc:>11.2%} {f1:>10.4f}")

        # 3 clases
        if dataset in results_3c:
            r3c = results_3c[dataset]
            opt = opt_results_3c[dataset]
            print(f"{dataset:<18} {'3 clases':<12} {opt.best_k:>4} {r3c.accuracy:>11.2%} {r3c.get_macro_f1():>10.4f}")

        print()

    print("-" * 80)

    # Analisis de mejora por warping
    print()
    print("MEJORA POR WARPING:")
    print()

    for prefix in ["full", "manual"]:
        warped = f"{prefix}_warped"
        original = f"{prefix}_original"

        print(f"  {prefix.upper()}:")

        # 2 clases
        if warped in results_2c and original in results_2c:
            acc_w = results_2c[warped].get("test_metrics", {}).get("accuracy", 0)
            acc_o = results_2c[original].get("test_metrics", {}).get("accuracy", 0)
            diff = acc_w - acc_o
            sign = "+" if diff > 0 else ""
            print(f"    2 clases: {acc_o:.2%} -> {acc_w:.2%} ({sign}{diff:.2%})")

        # 3 clases
        if warped in results_3c and original in results_3c:
            acc_w = results_3c[warped].accuracy
            acc_o = results_3c[original].accuracy
            diff = acc_w - acc_o
            sign = "+" if diff > 0 else ""
            print(f"    3 clases: {acc_o:.2%} -> {acc_w:.2%} ({sign}{diff:.2%})")

        print()


def main():
    """Funcion principal."""
    print("=" * 70)
    print("FASE 7: COMPARACION 2 CLASES vs 3 CLASES")
    print("=" * 70)
    print()
    print("Pipeline: PCA -> Z-score -> Fisher Multiclase -> KNN")
    print()
    print(f"Directorio entrada (Fase 4): {PHASE4_METRICS_DIR}")
    print(f"Directorio entrada (Fase 6): {PHASE6_METRICS_DIR}")
    print(f"Directorio salida (metricas): {OUTPUT_METRICS_DIR}")
    print(f"Directorio salida (figuras): {OUTPUT_FIGURES_DIR}")

    # Crear directorios
    OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar resultados de 2 clases
    results_2c = load_2class_results()
    print(f"\nResultados 2 clases cargados: {list(results_2c.keys())}")

    # Procesar cada dataset con 3 clases
    results_3c: Dict[str, ClassificationResult] = {}
    opt_results_3c: Dict[str, KOptimizationResult] = {}
    fisher_infos: Dict[str, Dict] = {}

    for dataset_name in DATASETS:
        try:
            result, opt_result, fisher_info = process_dataset_3class(dataset_name)
            results_3c[dataset_name] = result
            opt_results_3c[dataset_name] = opt_result
            fisher_infos[dataset_name] = fisher_info
        except FileNotFoundError as e:
            print(f"AVISO: No se pudo procesar {dataset_name}: {e}")
            continue

    # Generar figuras de matrices de confusion (4 datasets)
    if len(results_3c) == 4:
        print()
        print("=" * 70)
        print("GENERANDO MATRICES DE CONFUSION COMPARATIVAS")
        print("=" * 70)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("Matrices de Confusion (3 Clases) - Todos los Datasets",
                     fontsize=14, fontweight='bold')

        dataset_order = ["full_warped", "full_original", "manual_warped", "manual_original"]
        titles = ["Full Warped", "Full Original", "Manual Warped", "Manual Original"]

        for idx, (dataset, title) in enumerate(zip(dataset_order, titles)):
            if dataset not in results_3c:
                continue

            row, col = idx // 2, idx % 2
            ax = axes[row, col]

            res = results_3c[dataset]
            cm = res.confusion_matrix

            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

            thresh = cm.max() / 2.
            for i in range(len(CLASS_NAMES_3C)):
                for j in range(len(CLASS_NAMES_3C)):
                    ax.text(j, i, f'{int(cm[i, j])}',
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black",
                            fontsize=11)

            ax.set_xticks(range(len(CLASS_NAMES_3C)))
            ax.set_yticks(range(len(CLASS_NAMES_3C)))
            ax.set_xticklabels(['COVID', 'Normal', 'Viral'], fontsize=9)
            ax.set_yticklabels(['COVID', 'Normal', 'Viral'], fontsize=9)
            ax.set_xlabel('Predicha')
            ax.set_ylabel('Real')
            ax.set_title(f'{title}\nAcc={res.accuracy:.2%}, K={res.k}')

        plt.tight_layout()
        fig.savefig(OUTPUT_FIGURES_DIR / "confusion_matrices_3class.png",
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Guardado: {OUTPUT_FIGURES_DIR / 'confusion_matrices_3class.png'}")

    # Generar resumen final
    generate_summary(results_3c, opt_results_3c, fisher_infos, results_2c)

    print()
    print("=" * 70)
    print("FASE 7 COMPLETADA")
    print("=" * 70)


if __name__ == "__main__":
    main()
