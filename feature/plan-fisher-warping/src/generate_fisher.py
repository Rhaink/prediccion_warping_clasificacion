#!/usr/bin/env python3
"""
Script para generar Fisher Ratios y amplificar caracteristicas.

FASE 5 DEL PIPELINE FISHER-WARPING
==================================

Este script:
1. Carga las caracteristicas estandarizadas de la Fase 4
2. Calcula Fisher ratios para cada caracteristica (usando SOLO training)
3. Amplifica caracteristicas de train/val/test
4. Genera visualizaciones comparativas
5. Guarda resultados en CSV

IMPORTANTE - FLUJO DE DATOS
---------------------------
El Fisher ratio se calcula SOLO con datos de training:
1. Separar training por clase (Enfermo vs Normal)
2. Calcular medias y stds por clase
3. Aplicar formula Fisher

Luego, los mismos Fisher ratios se aplican a val y test
para amplificar sus caracteristicas (igual que con Z-score).

DATASETS PROCESADOS
-------------------
1. full_warped: Dataset completo con imagenes warped (5040 train)
2. full_original: Dataset completo con imagenes originales (5040 train)
3. manual_warped: Dataset manual con imagenes warped (717 train)
4. manual_original: Dataset manual con imagenes originales (717 train)

ENTREGABLES
-----------
- results/metrics/phase5_fisher/{dataset}_fisher_ratios.csv
- results/metrics/phase5_fisher/{dataset}_{split}_amplified.csv
- results/figures/phase5_fisher/{dataset}/fisher_ratios.png
- results/figures/phase5_fisher/{dataset}/class_separation.png
- results/figures/phase5_fisher/comparisons/fisher_comparison.png
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Importar modulo Fisher
import sys
sys.path.insert(0, str(Path(__file__).parent))
from fisher import (
    FisherRatio,
    FisherResult,
    plot_fisher_ratios,
    plot_class_separation,
    plot_amplification_effect,
    save_fisher_results,
    save_amplified_features,
    verify_fisher_calculation
)


# Configuracion
DATASETS = ['full_warped', 'full_original', 'manual_warped', 'manual_original']
CLASS_NAMES = ['Enfermo', 'Normal']

# Rutas
BASE_DIR = Path(__file__).parent.parent
FEATURES_DIR = BASE_DIR / "results" / "metrics" / "phase4_features"
OUTPUT_METRICS_DIR = BASE_DIR / "results" / "metrics" / "phase5_fisher"
OUTPUT_FIGURES_DIR = BASE_DIR / "results" / "figures" / "phase5_fisher"


def load_features(dataset: str, split: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Carga caracteristicas de la Fase 4.

    Args:
        dataset: Nombre del dataset (full_warped, etc.)
        split: Split a cargar (train, val, test)

    Returns:
        Tuple de (X, y, ids)
        - X: Matriz de caracteristicas (N, K)
        - y: Vector de etiquetas (N,)
        - ids: Lista de IDs de imagen
    """
    csv_path = FEATURES_DIR / f"{dataset}_{split}_features.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontro: {csv_path}")

    df = pd.read_csv(csv_path)

    # Extraer caracteristicas (columnas PC1, PC2, ..., PCK)
    pc_cols = [col for col in df.columns if col.startswith('PC')]
    X = df[pc_cols].values.astype(np.float32)

    # Etiquetas y IDs
    y = df['label'].values
    ids = df['image_id'].tolist()

    return X, y, ids


def process_dataset(dataset: str, verbose: bool = True) -> Dict:
    """
    Procesa un dataset completo: calcula Fisher y amplifica.

    Args:
        dataset: Nombre del dataset
        verbose: Si True, imprime progreso

    Returns:
        Dict con resultados del procesamiento
    """
    if verbose:
        print()
        print("="*70)
        print(f"PROCESANDO: {dataset.upper()}")
        print("="*70)

    # Crear directorios de salida
    dataset_figures_dir = OUTPUT_FIGURES_DIR / dataset
    dataset_figures_dir.mkdir(parents=True, exist_ok=True)

    # === 1. CARGAR DATOS ===
    if verbose:
        print("\n[1/4] Cargando caracteristicas de Fase 4...")

    X_train, y_train, ids_train = load_features(dataset, 'train')
    X_val, y_val, ids_val = load_features(dataset, 'val')
    X_test, y_test, ids_test = load_features(dataset, 'test')

    if verbose:
        print(f"      Train: {X_train.shape[0]} muestras x {X_train.shape[1]} caracteristicas")
        print(f"      Val:   {X_val.shape[0]} muestras")
        print(f"      Test:  {X_test.shape[0]} muestras")

    # === 2. CALCULAR FISHER RATIOS (solo con training) ===
    if verbose:
        print("\n[2/4] Calculando Fisher Ratios...")

    fisher = FisherRatio()
    fisher.fit(X_train, y_train, class_names=CLASS_NAMES, verbose=verbose)
    fisher_result = fisher.get_result()

    # Verificar calculo
    verification = verify_fisher_calculation(X_train, y_train, fisher_result, verbose=verbose)

    # Guardar Fisher ratios
    save_fisher_results(
        fisher_result,
        OUTPUT_METRICS_DIR / f"{dataset}_fisher_ratios.csv"
    )

    # === 3. AMPLIFICAR CARACTERISTICAS ===
    if verbose:
        print("\n[3/4] Amplificando caracteristicas...")

    X_train_amp = fisher.amplify(X_train)
    X_val_amp = fisher.amplify(X_val)
    X_test_amp = fisher.amplify(X_test)

    if verbose:
        print(f"      Train amplificado: std min={np.std(X_train_amp, axis=0).min():.4f}, "
              f"max={np.std(X_train_amp, axis=0).max():.4f}")

    # Guardar caracteristicas amplificadas
    for split, X_amp, y, ids in [
        ('train', X_train_amp, y_train, ids_train),
        ('val', X_val_amp, y_val, ids_val),
        ('test', X_test_amp, y_test, ids_test)
    ]:
        save_amplified_features(
            X_amp, y, ids, CLASS_NAMES,
            OUTPUT_METRICS_DIR / f"{dataset}_{split}_amplified.csv"
        )

    # === 4. GENERAR VISUALIZACIONES ===
    if verbose:
        print("\n[4/4] Generando visualizaciones...")

    # Fisher ratios
    plot_fisher_ratios(
        fisher_result,
        output_path=dataset_figures_dir / "fisher_ratios.png",
        top_k=10
    )

    # Separacion de clases
    plot_class_separation(
        X_train, y_train, fisher_result,
        top_k=6,
        output_path=dataset_figures_dir / "class_separation.png"
    )

    # Efecto de amplificacion
    plot_amplification_effect(
        X_train, X_train_amp, fisher_result,
        top_k=6,
        output_path=dataset_figures_dir / "amplification_effect.png"
    )

    plt.close('all')

    # Resumen
    results = {
        'dataset': dataset,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'n_features': fisher_result.n_features,
        'fisher_min': float(fisher_result.fisher_ratios.min()),
        'fisher_max': float(fisher_result.fisher_ratios.max()),
        'fisher_mean': float(fisher_result.fisher_ratios.mean()),
        'fisher_std': float(fisher_result.fisher_ratios.std()),
        'top5_features': [int(i+1) for i in fisher_result.get_top_k(5)[0]],
        'top5_values': [float(v) for v in fisher_result.get_top_k(5)[1]],
        'verification_passed': all(verification.values())
    }

    return results


def create_comparison_figure(all_results: List[Dict]) -> plt.Figure:
    """
    Crea figura comparativa de Fisher ratios entre los 4 datasets.

    Args:
        all_results: Lista de resultados de cada dataset

    Returns:
        Figura de matplotlib
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    fig.suptitle('Comparacion de Fisher Ratios entre Datasets',
                 fontsize=16, fontweight='bold', y=1.02)

    for idx, result in enumerate(all_results):
        ax = axes[idx]
        dataset = result['dataset']

        # Cargar Fisher ratios del CSV guardado
        df = pd.read_csv(OUTPUT_METRICS_DIR / f"{dataset}_fisher_ratios.csv")
        ratios = df['fisher_ratio'].values

        # Grafico de barras
        x = np.arange(1, len(ratios) + 1)
        ax.bar(x, ratios, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.3)

        # Estadisticas
        ax.axhline(y=ratios.mean(), color='red', linestyle='--',
                   label=f'Media = {ratios.mean():.3f}')

        # Titulo con info
        title = dataset.replace('_', ' ').title()
        ax.set_title(f'{title}\n(max J = {ratios.max():.2f})', fontsize=12)
        ax.set_xlabel('Componente Principal')
        ax.set_ylabel('Fisher Ratio')
        ax.legend(loc='upper right', fontsize=9)

        # Destacar top 3
        top3_idx = np.argsort(ratios)[-3:][::-1]
        for i in top3_idx:
            ax.annotate(f'PC{i+1}', xy=(i+1, ratios[i]),
                        xytext=(i+1, ratios[i] + 0.05),
                        ha='center', fontsize=8)

    plt.tight_layout()
    return fig


def create_warped_vs_original_figure(all_results: List[Dict]) -> plt.Figure:
    """
    Crea figura comparando warped vs original.

    Muestra que el warping afecta (o no) los Fisher ratios.

    Args:
        all_results: Lista de resultados

    Returns:
        Figura de matplotlib
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fig.suptitle('Impacto del Warping en Fisher Ratios',
                 fontsize=14, fontweight='bold')

    # Full dataset
    ax = axes[0]
    df_warped = pd.read_csv(OUTPUT_METRICS_DIR / "full_warped_fisher_ratios.csv")
    df_original = pd.read_csv(OUTPUT_METRICS_DIR / "full_original_fisher_ratios.csv")

    x = np.arange(1, len(df_warped) + 1)
    width = 0.35

    ax.bar(x - width/2, df_warped['fisher_ratio'], width,
           label='Warped', color='darkorange', alpha=0.7)
    ax.bar(x + width/2, df_original['fisher_ratio'], width,
           label='Original', color='steelblue', alpha=0.7)

    ax.set_xlabel('Componente Principal')
    ax.set_ylabel('Fisher Ratio')
    ax.set_title('Full Dataset: Warped vs Original')
    ax.legend()
    ax.set_xticks(x[::5])

    # Manual dataset
    ax = axes[1]
    df_warped = pd.read_csv(OUTPUT_METRICS_DIR / "manual_warped_fisher_ratios.csv")
    df_original = pd.read_csv(OUTPUT_METRICS_DIR / "manual_original_fisher_ratios.csv")

    ax.bar(x - width/2, df_warped['fisher_ratio'], width,
           label='Warped', color='darkorange', alpha=0.7)
    ax.bar(x + width/2, df_original['fisher_ratio'], width,
           label='Original', color='steelblue', alpha=0.7)

    ax.set_xlabel('Componente Principal')
    ax.set_ylabel('Fisher Ratio')
    ax.set_title('Manual Dataset: Warped vs Original')
    ax.legend()
    ax.set_xticks(x[::5])

    plt.tight_layout()
    return fig


def create_summary_table(all_results: List[Dict]) -> plt.Figure:
    """
    Crea tabla resumen de Fisher ratios.

    Args:
        all_results: Lista de resultados

    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    # Preparar datos para tabla
    headers = ['Dataset', 'Train', 'Features', 'J min', 'J max', 'J mean', 'Top 3 PCs']
    rows = []

    for r in all_results:
        row = [
            r['dataset'].replace('_', ' ').title(),
            str(r['n_train']),
            str(r['n_features']),
            f"{r['fisher_min']:.4f}",
            f"{r['fisher_max']:.4f}",
            f"{r['fisher_mean']:.4f}",
            ', '.join([f"PC{i}" for i in r['top5_features'][:3]])
        ]
        rows.append(row)

    # Crear tabla
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0'] * len(headers)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Estilo
    for i in range(len(headers)):
        table[(0, i)].set_text_props(weight='bold')

    ax.set_title('Resumen de Fisher Ratios por Dataset',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig


def main():
    """Funcion principal."""
    print("="*70)
    print("FASE 5: CALCULO DE FISHER RATIOS")
    print("="*70)
    print()
    print("Este script calcula el Criterio de Fisher para cada caracteristica")
    print("y amplifica las caracteristicas multiplicando por el Fisher ratio.")
    print()
    print(f"Datasets a procesar: {DATASETS}")
    print(f"Directorio de entrada: {FEATURES_DIR}")
    print(f"Directorio de salida (metricas): {OUTPUT_METRICS_DIR}")
    print(f"Directorio de salida (figuras): {OUTPUT_FIGURES_DIR}")

    # Crear directorios
    OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Procesar cada dataset
    all_results = []

    for dataset in DATASETS:
        try:
            result = process_dataset(dataset, verbose=True)
            all_results.append(result)
        except Exception as e:
            print(f"\nERROR procesando {dataset}: {e}")
            import traceback
            traceback.print_exc()

    # Generar figuras comparativas
    print()
    print("="*70)
    print("GENERANDO FIGURAS COMPARATIVAS")
    print("="*70)

    comparisons_dir = OUTPUT_FIGURES_DIR / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    # Comparacion entre datasets
    fig = create_comparison_figure(all_results)
    fig.savefig(comparisons_dir / "fisher_4datasets.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Guardado: {comparisons_dir / 'fisher_4datasets.png'}")
    plt.close(fig)

    # Warped vs Original
    fig = create_warped_vs_original_figure(all_results)
    fig.savefig(comparisons_dir / "warped_vs_original.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Guardado: {comparisons_dir / 'warped_vs_original.png'}")
    plt.close(fig)

    # Tabla resumen
    fig = create_summary_table(all_results)
    fig.savefig(comparisons_dir / "summary_table.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Guardado: {comparisons_dir / 'summary_table.png'}")
    plt.close(fig)

    # Guardar resumen JSON
    summary = {
        'phase': 5,
        'description': 'Fisher Ratio calculation and feature amplification',
        'datasets_processed': len(all_results),
        'results': all_results
    }

    with open(OUTPUT_METRICS_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Resumen guardado: {OUTPUT_METRICS_DIR / 'summary.json'}")

    # Resumen final
    print()
    print("="*70)
    print("RESUMEN FASE 5")
    print("="*70)
    print()

    for r in all_results:
        print(f"{r['dataset']}:")
        print(f"  Fisher max: {r['fisher_max']:.4f}")
        print(f"  Fisher mean: {r['fisher_mean']:.4f}")
        print(f"  Top 3 PCs: {r['top5_features'][:3]}")
        print()

    print("="*70)
    print("FASE 5 COMPLETADA")
    print("="*70)
    print()
    print("Entregables generados:")
    print(f"  - Metricas: {OUTPUT_METRICS_DIR}")
    print(f"  - Figuras: {OUTPUT_FIGURES_DIR}")


if __name__ == "__main__":
    main()
