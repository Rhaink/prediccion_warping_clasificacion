"""
Generador de Figuras PCA para Paper.

Este script genera las figuras finales de PCA con visualización correcta:
- Fondo negro (opción elegida)
- Normalización solo del contenido
- Sin artefactos de máscara

Figuras generadas:
1. fig_01_mean_face.png - Radiografía promedio
2. fig_02_eigenfaces.png - Top 10 eigenfaces
3. fig_03_variance.png - Varianza explicada
4. fig_04_comparison.png - Comparación Warped vs Original
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_dataset
from pca import PCA


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


def normalize_with_black_background(eigenface_2d, content_mask):
    """
    Normaliza eigenface poniendo fondo negro.

    Args:
        eigenface_2d: Eigenface como imagen 2D
        content_mask: Máscara booleana de contenido (True donde hay datos)

    Returns:
        Imagen normalizada con fondo negro
    """
    # Obtener solo valores de contenido
    content_vals = eigenface_2d[content_mask]
    content_min, content_max = content_vals.min(), content_vals.max()

    # Normalizar
    normalized = (eigenface_2d - content_min) / (content_max - content_min + 1e-8)

    # Poner fondo negro
    result = np.where(content_mask, normalized, 0)

    return result


def generate_mean_face(pca_result, image_shape, content_mask, output_path):
    """Genera figura de radiografía promedio."""
    set_paper_style()

    fig, ax = plt.subplots(figsize=(6, 6))

    mean_img = pca_result.mean.reshape(image_shape)
    ax.imshow(mean_img, cmap='gray')
    ax.set_title('Radiografía Promedio (Mean Face)', fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Guardado: {output_path.name}")


def generate_eigenfaces(pca_result, image_shape, content_mask, output_path, n_show=10):
    """Genera figura con top eigenfaces."""
    set_paper_style()

    n_cols = 5
    n_rows = (n_show + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows))
    axes = axes.flatten()

    for i in range(n_show):
        eigenface_2d = pca_result.components[i].reshape(image_shape)
        eigenface_display = normalize_with_black_background(eigenface_2d, content_mask)

        axes[i].imshow(eigenface_display, cmap='gray')
        var = pca_result.explained_variance_ratio[i] * 100
        axes[i].set_title(f'PC{i+1} ({var:.1f}%)', fontsize=10)
        axes[i].axis('off')

    # Ocultar axes vacíos
    for i in range(n_show, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Eigenfaces (Componentes Principales)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Guardado: {output_path.name}")


def generate_variance_plot(pca_result, output_path):
    """Genera figura de varianza explicada."""
    set_paper_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    n_comp = len(pca_result.explained_variance_ratio)
    x = np.arange(1, n_comp + 1)
    cumulative = np.cumsum(pca_result.explained_variance_ratio) * 100

    # Panel izquierdo: varianza individual
    ax1.bar(x[:20], pca_result.explained_variance_ratio[:20] * 100,
            color='steelblue', alpha=0.8)
    ax1.set_xlabel('Componente Principal')
    ax1.set_ylabel('Varianza Explicada (%)')
    ax1.set_title('(a) Varianza por Componente (Top 20)', fontweight='bold')

    # Panel derecho: varianza acumulada
    ax2.plot(x, cumulative, 'o-', color='steelblue', linewidth=2, markersize=3)
    ax2.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='95%')
    ax2.axhline(y=99, color='orange', linestyle='--', alpha=0.7, label='99%')

    # Encontrar K para 95%
    k_95 = np.argmax(cumulative >= 95) + 1 if np.any(cumulative >= 95) else n_comp
    ax2.axvline(x=k_95, color='red', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Número de Componentes')
    ax2.set_ylabel('Varianza Acumulada (%)')
    ax2.set_title('(b) Varianza Acumulada', fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.set_ylim([0, 102])

    # Info
    ax2.text(0.95, 0.15, f'K={k_95} para 95%', transform=ax2.transAxes,
             fontsize=10, ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Guardado: {output_path.name}")


def generate_comparison(results_list, output_path):
    """
    Genera figura comparativa de eigenfaces entre escenarios.

    Args:
        results_list: Lista de tuplas (nombre, pca_result, image_shape, content_mask)
    """
    set_paper_style()

    n_scenarios = len(results_list)
    n_show = 5

    fig, axes = plt.subplots(n_scenarios, n_show, figsize=(12, 3 * n_scenarios))

    if n_scenarios == 1:
        axes = axes.reshape(1, -1)

    for row, (name, pca_result, image_shape, content_mask) in enumerate(results_list):
        for col in range(n_show):
            ax = axes[row, col]

            eigenface_2d = pca_result.components[col].reshape(image_shape)
            eigenface_display = normalize_with_black_background(eigenface_2d, content_mask)

            ax.imshow(eigenface_display, cmap='gray')
            ax.axis('off')

            if row == 0:
                var = pca_result.explained_variance_ratio[col] * 100
                ax.set_title(f'PC{col+1} ({var:.1f}%)', fontsize=10)

        # Etiqueta de fila
        axes[row, 0].text(-0.15, 0.5, f'({chr(97+row)}) {name}',
                         transform=axes[row, 0].transAxes,
                         fontsize=11, fontweight='bold',
                         va='center', ha='right')

    plt.suptitle('Comparación de Eigenfaces: Warped vs Original', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Guardado: {output_path.name}")


def generate_variance_comparison(results_list, output_path):
    """Genera comparación de varianza entre escenarios."""
    set_paper_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, (name, pca_result, _, _) in enumerate(results_list):
        cumulative = np.cumsum(pca_result.explained_variance_ratio) * 100
        n_comp = len(cumulative)
        x = np.arange(1, n_comp + 1)

        ax.plot(x, cumulative, '-', color=colors[i], linewidth=2,
                label=name, marker='o', markersize=3, markevery=max(1, n_comp//10))

    ax.axhline(y=95, color='gray', linestyle='--', alpha=0.7)
    ax.text(2, 96, '95%', fontsize=9, color='gray')

    ax.set_xlabel('Número de Componentes')
    ax.set_ylabel('Varianza Acumulada (%)')
    ax.set_title('Comparación de Varianza Explicada', fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim([0, 102])

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Guardado: {output_path.name}")


def main():
    """Función principal."""
    # Rutas
    base_path = Path(__file__).parent.parent.parent.parent
    results_dir = Path(__file__).parent.parent / "results"
    metrics_dir = results_dir / "metrics"
    figures_dir = results_dir / "figures"

    figures_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("GENERACIÓN DE FIGURAS PCA")
    print("="*60)

    # Escenarios a analizar (SIN máscara - los datos se cargan completos)
    scenarios = [
        ("Warped", metrics_dir / "01_full_balanced_3class_warped.csv"),
        ("Original", metrics_dir / "01_full_balanced_3class_original.csv"),
    ]

    results_list = []

    for name, csv_path in scenarios:
        print(f"\nCargando: {name}...")

        # Cargar SIN máscara
        dataset = load_dataset(csv_path, base_path, scenario="2class",
                              use_mask=False, verbose=False)

        # PCA
        print(f"  Aplicando PCA...")
        pca = PCA(n_components=50)
        pca.fit(dataset.train.X, verbose=False)
        pca_result = pca.get_result()

        # Crear máscara de contenido para visualización
        mean_2d = pca_result.mean.reshape(dataset.image_shape)
        content_mask = mean_2d > 0.01

        results_list.append((name, pca_result, dataset.image_shape, content_mask))

        # Estadísticas
        cumulative = np.cumsum(pca_result.explained_variance_ratio)
        k_95 = np.argmax(cumulative >= 0.95) + 1 if np.any(cumulative >= 0.95) else 50
        print(f"  PC1: {pca_result.explained_variance_ratio[0]*100:.1f}%")
        print(f"  Top 10: {cumulative[9]*100:.1f}%")
        print(f"  K para 95%: {k_95}")

    # Generar figuras
    print("\n" + "="*60)
    print("GENERANDO FIGURAS")
    print("="*60)

    # Usar datos de Warped para figuras individuales
    warped_name, warped_pca, warped_shape, warped_mask = results_list[0]

    print("\nFiguras individuales (dataset Warped):")
    generate_mean_face(warped_pca, warped_shape, warped_mask,
                      figures_dir / "fig_01_mean_face.png")
    generate_eigenfaces(warped_pca, warped_shape, warped_mask,
                       figures_dir / "fig_02_eigenfaces.png", n_show=10)
    generate_variance_plot(warped_pca, figures_dir / "fig_03_variance.png")

    print("\nFiguras comparativas:")
    generate_comparison(results_list, figures_dir / "fig_04_comparison_eigenfaces.png")
    generate_variance_comparison(results_list, figures_dir / "fig_05_comparison_variance.png")

    # Tabla resumen
    print("\n" + "="*60)
    print("TABLA RESUMEN")
    print("="*60)
    print("\n| Escenario | Dims | PC1 (%) | Top 5 (%) | Top 10 (%) |")
    print("|-----------|------|---------|-----------|------------|")

    for name, pca_result, shape, _ in results_list:
        dims = shape[0] * shape[1]
        cumulative = np.cumsum(pca_result.explained_variance_ratio)
        pc1 = pca_result.explained_variance_ratio[0] * 100
        top5 = cumulative[4] * 100
        top10 = cumulative[9] * 100
        print(f"| {name:9} | {dims:,} | {pc1:5.1f} | {top5:9.1f} | {top10:10.1f} |")

    print(f"\nFiguras guardadas en: {figures_dir}")
    print("\nLista de figuras generadas:")
    for f in sorted(figures_dir.glob("fig_*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
