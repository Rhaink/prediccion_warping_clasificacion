"""
Generador de Figuras PCA para los 4 Datasets.

Este script procesa los 4 escenarios del experimento:
1. Full Warped   - Dataset completo (6,725 imgs) con alineación
2. Full Original - Dataset completo (6,725 imgs) sin alineación
3. Manual Warped   - Dataset manual (957 imgs) con alineación
4. Manual Original - Dataset manual (957 imgs) sin alineación

Estructura de salida:
    results/figures/phase3_pca/
    ├── full_warped/
    │   ├── mean_face.png
    │   ├── eigenfaces.png
    │   └── variance.png
    ├── full_original/
    │   └── ...
    ├── manual_warped/
    │   └── ...
    ├── manual_original/
    │   └── ...
    └── comparisons/
        ├── eigenfaces_4datasets.png
        ├── variance_4datasets.png
        └── summary_table.png

Visualización científica:
- Warped: Normalización con fondo negro (varianza fondo ≈ 0)
- Original: Método estándar sklearn (imshow automático)

Referencias:
- Turk & Pentland (1991): Eigenfaces for Recognition
- scikit-learn eigenfaces example
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_dataset, Dataset
from pca import PCA, PCAResult


@dataclass
class DatasetConfig:
    """Configuración de un dataset para análisis."""
    name: str           # Nombre corto para figuras
    csv_filename: str   # Nombre del CSV en metrics/
    is_warped: bool     # Si las imágenes están warped
    output_dir: str     # Subdirectorio de salida


# Configuración de los 4 datasets
DATASETS = [
    DatasetConfig("Full Warped", "01_full_balanced_3class_warped.csv", True, "full_warped"),
    DatasetConfig("Full Original", "01_full_balanced_3class_original.csv", False, "full_original"),
    DatasetConfig("Manual Warped", "03_manual_warped.csv", True, "manual_warped"),
    DatasetConfig("Manual Original", "03_manual_original.csv", False, "manual_original"),
]


def set_paper_style():
    """Configura matplotlib para figuras de publicación."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def normalize_eigenface(eigenface_2d: np.ndarray,
                       is_warped: bool,
                       mean_2d: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Normaliza eigenface para visualización.

    Para warped: fondo negro (científicamente justificado, varianza=0)
    Para original: normalización min-max estándar

    Args:
        eigenface_2d: Eigenface como imagen 2D
        is_warped: Si la imagen es warped
        mean_2d: Imagen media (para calcular máscara en warped)

    Returns:
        Imagen normalizada para visualización
    """
    if is_warped and mean_2d is not None:
        # Para warped: fondo negro
        content_mask = mean_2d > 0.01
        content_vals = eigenface_2d[content_mask]
        vmin, vmax = content_vals.min(), content_vals.max()
        normalized = (eigenface_2d - vmin) / (vmax - vmin + 1e-8)
        return np.where(content_mask, normalized, 0)
    else:
        # Para original: normalización min-max estándar
        vmin, vmax = eigenface_2d.min(), eigenface_2d.max()
        return (eigenface_2d - vmin) / (vmax - vmin + 1e-8)


def generate_mean_face(pca_result: PCAResult,
                      image_shape: Tuple[int, int],
                      config: DatasetConfig,
                      output_path: Path) -> None:
    """Genera figura de radiografía promedio."""
    set_paper_style()

    fig, ax = plt.subplots(figsize=(6, 6))

    mean_img = pca_result.mean.reshape(image_shape)
    ax.imshow(mean_img, cmap='gray')
    ax.set_title(f'Radiografía Promedio\n({config.name})', fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_eigenfaces(pca_result: PCAResult,
                       image_shape: Tuple[int, int],
                       config: DatasetConfig,
                       output_path: Path,
                       n_show: int = 10) -> None:
    """Genera figura con top eigenfaces."""
    set_paper_style()

    n_cols = 5
    n_rows = (n_show + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows))
    axes = axes.flatten()

    mean_2d = pca_result.mean.reshape(image_shape)

    for i in range(n_show):
        eigenface_2d = pca_result.components[i].reshape(image_shape)
        eigenface_display = normalize_eigenface(eigenface_2d, config.is_warped, mean_2d)

        axes[i].imshow(eigenface_display, cmap='gray')
        var = pca_result.explained_variance_ratio[i] * 100
        axes[i].set_title(f'PC{i+1} ({var:.1f}%)', fontsize=10)
        axes[i].axis('off')

    for i in range(n_show, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Eigenfaces - {config.name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_variance_plot(pca_result: PCAResult,
                          config: DatasetConfig,
                          output_path: Path) -> None:
    """Genera figura de varianza explicada."""
    set_paper_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    n_comp = len(pca_result.explained_variance_ratio)
    x = np.arange(1, n_comp + 1)
    cumulative = np.cumsum(pca_result.explained_variance_ratio) * 100

    # Panel izquierdo: varianza individual (top 20)
    n_bars = min(20, n_comp)
    ax1.bar(x[:n_bars], pca_result.explained_variance_ratio[:n_bars] * 100,
            color='steelblue', alpha=0.8)
    ax1.set_xlabel('Componente Principal')
    ax1.set_ylabel('Varianza Explicada (%)')
    ax1.set_title('(a) Varianza por Componente', fontweight='bold')

    # Panel derecho: varianza acumulada
    ax2.plot(x, cumulative, 'o-', color='steelblue', linewidth=2, markersize=3)
    ax2.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='95%')
    ax2.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='90%')

    # Encontrar K para 90% y 95%
    k_90 = np.argmax(cumulative >= 90) + 1 if np.any(cumulative >= 90) else n_comp
    k_95 = np.argmax(cumulative >= 95) + 1 if np.any(cumulative >= 95) else n_comp
    ax2.axvline(x=k_90, color='orange', linestyle=':', alpha=0.5)
    ax2.axvline(x=k_95, color='red', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Número de Componentes')
    ax2.set_ylabel('Varianza Acumulada (%)')
    ax2.set_title('(b) Varianza Acumulada', fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.set_ylim([0, 102])

    # Info
    ax2.text(0.95, 0.15, f'K={k_90} para 90%\nK={k_95} para 95%',
            transform=ax2.transAxes, fontsize=10, ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Varianza Explicada - {config.name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_comparison_eigenfaces(results: List[Tuple[DatasetConfig, PCAResult, Tuple[int, int]]],
                                  output_path: Path) -> None:
    """Genera comparación de eigenfaces entre los 4 datasets."""
    set_paper_style()

    n_datasets = len(results)
    n_show = 5

    fig, axes = plt.subplots(n_datasets, n_show, figsize=(14, 3 * n_datasets))

    for row, (config, pca_result, image_shape) in enumerate(results):
        mean_2d = pca_result.mean.reshape(image_shape)

        for col in range(n_show):
            ax = axes[row, col]

            eigenface_2d = pca_result.components[col].reshape(image_shape)
            eigenface_display = normalize_eigenface(eigenface_2d, config.is_warped, mean_2d)

            ax.imshow(eigenface_display, cmap='gray')
            ax.axis('off')

            if row == 0:
                var = pca_result.explained_variance_ratio[col] * 100
                ax.set_title(f'PC{col+1}', fontsize=11, fontweight='bold')

        # Etiqueta de fila con varianza de PC1
        var_pc1 = pca_result.explained_variance_ratio[0] * 100
        label = f'{config.name}\n(PC1={var_pc1:.1f}%)'
        axes[row, 0].text(-0.18, 0.5, label,
                         transform=axes[row, 0].transAxes,
                         fontsize=10, fontweight='bold',
                         va='center', ha='right')

    plt.suptitle('Comparación de Eigenfaces: 4 Datasets',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_comparison_variance(results: List[Tuple[DatasetConfig, PCAResult, Tuple[int, int]]],
                                output_path: Path) -> None:
    """Genera comparación de varianza entre datasets."""
    set_paper_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Colores: azul para warped, naranja para original
    colors = {
        'Full Warped': '#1f77b4',
        'Full Original': '#ff7f0e',
        'Manual Warped': '#2ca02c',
        'Manual Original': '#d62728',
    }

    linestyles = {
        'Full Warped': '-',
        'Full Original': '-',
        'Manual Warped': '--',
        'Manual Original': '--',
    }

    for config, pca_result, _ in results:
        cumulative = np.cumsum(pca_result.explained_variance_ratio) * 100
        n_comp = len(cumulative)
        x = np.arange(1, n_comp + 1)

        ax.plot(x, cumulative,
               linestyle=linestyles[config.name],
               color=colors[config.name],
               linewidth=2,
               label=config.name,
               marker='o', markersize=3,
               markevery=max(1, n_comp//10))

    ax.axhline(y=95, color='gray', linestyle=':', alpha=0.7)
    ax.text(2, 96, '95%', fontsize=9, color='gray')

    ax.set_xlabel('Número de Componentes')
    ax.set_ylabel('Varianza Acumulada (%)')
    ax.set_title('Comparación de Varianza Explicada: 4 Datasets', fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim([0, 102])

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_summary_table(results: List[Tuple[DatasetConfig, PCAResult, Tuple[int, int]]],
                          output_path: Path) -> None:
    """Genera tabla resumen como figura."""
    set_paper_style()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    # Preparar datos de la tabla
    columns = ['Dataset', 'N imgs', 'Dims', 'PC1 (%)', 'Top 5 (%)', 'Top 10 (%)', 'K@90%', 'K@95%']
    rows = []

    for config, pca_result, image_shape in results:
        n_imgs = pca_result.n_features // (image_shape[0] * image_shape[1]) if pca_result.n_features else 0
        dims = image_shape[0] * image_shape[1]
        cumulative = np.cumsum(pca_result.explained_variance_ratio)

        pc1 = pca_result.explained_variance_ratio[0] * 100
        top5 = cumulative[4] * 100 if len(cumulative) > 4 else cumulative[-1] * 100
        top10 = cumulative[9] * 100 if len(cumulative) > 9 else cumulative[-1] * 100

        k_90 = np.argmax(cumulative >= 0.90) + 1 if np.any(cumulative >= 0.90) else len(cumulative)
        k_95 = np.argmax(cumulative >= 0.95) + 1 if np.any(cumulative >= 0.95) else len(cumulative)

        # El número de imágenes se calcula desde el dataset
        rows.append([
            config.name,
            '-',  # Se llenará después
            f'{dims:,}',
            f'{pc1:.1f}',
            f'{top5:.1f}',
            f'{top10:.1f}',
            str(k_90),
            str(k_95)
        ])

    # Crear tabla
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colColours=['#f0f0f0'] * len(columns)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Estilizar encabezados
    for i in range(len(columns)):
        table[(0, i)].set_text_props(fontweight='bold')

    plt.title('Resumen de Análisis PCA: 4 Datasets', fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    """Función principal."""

    # Rutas
    base_path = Path(__file__).parent.parent.parent.parent
    results_dir = Path(__file__).parent.parent / "results"
    metrics_dir = results_dir / "metrics"
    figures_base = results_dir / "figures" / "phase3_pca"

    print("=" * 70)
    print("GENERACIÓN DE FIGURAS PCA - 4 DATASETS")
    print("=" * 70)

    # Procesar cada dataset
    all_results = []

    for config in DATASETS:
        print(f"\n{'='*70}")
        print(f"Procesando: {config.name}")
        print(f"{'='*70}")

        csv_path = metrics_dir / config.csv_filename

        if not csv_path.exists():
            print(f"  ADVERTENCIA: CSV no encontrado: {csv_path}")
            continue

        # Crear directorio de salida
        output_dir = figures_base / config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Cargar dataset
        print(f"  Cargando dataset...")
        try:
            dataset = load_dataset(csv_path, base_path, scenario="2class",
                                  use_mask=False, verbose=False)
        except Exception as e:
            print(f"  ERROR al cargar: {e}")
            continue

        print(f"    Train: {dataset.train.X.shape[0]} imágenes")
        print(f"    Dimensiones: {dataset.image_shape}")

        # Aplicar PCA
        print(f"  Aplicando PCA...")
        n_components = min(50, dataset.train.X.shape[0] - 1)
        pca = PCA(n_components=n_components)
        pca.fit(dataset.train.X, verbose=False)
        pca_result = pca.get_result()

        # Estadísticas
        cumulative = np.cumsum(pca_result.explained_variance_ratio)
        k_95 = np.argmax(cumulative >= 0.95) + 1 if np.any(cumulative >= 0.95) else n_components

        print(f"    PC1: {pca_result.explained_variance_ratio[0]*100:.1f}%")
        print(f"    Top 10: {cumulative[9]*100:.1f}%" if len(cumulative) > 9 else "")
        print(f"    K para 95%: {k_95}")

        # Generar figuras individuales
        print(f"  Generando figuras...")
        generate_mean_face(pca_result, dataset.image_shape, config,
                          output_dir / "mean_face.png")
        generate_eigenfaces(pca_result, dataset.image_shape, config,
                           output_dir / "eigenfaces.png", n_show=10)
        generate_variance_plot(pca_result, config,
                              output_dir / "variance.png")

        print(f"    Guardado en: {output_dir}")

        # Guardar para comparaciones
        all_results.append((config, pca_result, dataset.image_shape))

    # Generar figuras comparativas
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("GENERANDO FIGURAS COMPARATIVAS")
        print(f"{'='*70}")

        comparisons_dir = figures_base / "comparisons"
        comparisons_dir.mkdir(parents=True, exist_ok=True)

        print("  Comparación de eigenfaces...")
        generate_comparison_eigenfaces(all_results,
                                       comparisons_dir / "eigenfaces_4datasets.png")

        print("  Comparación de varianza...")
        generate_comparison_variance(all_results,
                                    comparisons_dir / "variance_4datasets.png")

        print("  Tabla resumen...")
        generate_summary_table(all_results,
                              comparisons_dir / "summary_table.png")

    # Imprimir resumen final
    print(f"\n{'='*70}")
    print("RESUMEN FINAL")
    print(f"{'='*70}")
    print(f"\n| {'Dataset':<18} | {'PC1 (%)':<8} | {'Top 10 (%)':<10} | {'K@95%':<6} |")
    print(f"|{'-'*20}|{'-'*10}|{'-'*12}|{'-'*8}|")

    for config, pca_result, _ in all_results:
        cumulative = np.cumsum(pca_result.explained_variance_ratio)
        pc1 = pca_result.explained_variance_ratio[0] * 100
        top10 = cumulative[9] * 100 if len(cumulative) > 9 else cumulative[-1] * 100
        k_95 = np.argmax(cumulative >= 0.95) + 1 if np.any(cumulative >= 0.95) else len(cumulative)
        print(f"| {config.name:<18} | {pc1:<8.1f} | {top10:<10.1f} | {k_95:<6} |")

    print(f"\nFiguras guardadas en: {figures_base}")


if __name__ == "__main__":
    main()
