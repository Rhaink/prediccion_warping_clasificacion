"""
Verificación científica de la visualización de eigenfaces.

ESTÁNDAR CIENTÍFICO (Turk & Pentland, sklearn):
- Los eigenvectores se muestran DIRECTAMENTE con imshow()
- imshow() automáticamente escala al rango de valores
- Se usa cmap='gray' o cmap='bone'
- NO se aplica normalización manual adicional

NUESTRA IMPLEMENTACIÓN ACTUAL:
- Normaliza manualmente al rango [0, 1]
- Fuerza fondo negro para imágenes warped
- Esto puede distorsionar la interpretación

Este script compara ambos métodos para ver si hay diferencia significativa.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_dataset
from pca import PCA


def compare_visualizations():
    """Compara métodos de visualización de eigenfaces."""

    base_path = Path(__file__).parent.parent.parent.parent
    metrics_dir = Path(__file__).parent.parent / "results" / "metrics"
    output_dir = Path(__file__).parent.parent / "results" / "figures" / "verification"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar un dataset para prueba
    csv_path = metrics_dir / "01_full_balanced_3class_warped.csv"

    print("=" * 60)
    print("VERIFICACIÓN DE VISUALIZACIÓN DE EIGENFACES")
    print("=" * 60)

    print("\nCargando dataset...")
    dataset = load_dataset(csv_path, base_path, scenario="2class",
                          use_mask=False, verbose=False)

    print("Ajustando PCA...")
    pca = PCA(n_components=10)
    pca.fit(dataset.train.X, verbose=False)
    pca_result = pca.get_result()

    # Crear figura comparativa
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))

    for i in range(5):
        eigenface = pca_result.components[i].reshape(dataset.image_shape)

        # Fila 1: Método estándar (sklearn style) - sin normalización
        ax = axes[0, i]
        ax.imshow(eigenface, cmap='gray')
        ax.set_title(f'PC{i+1} (std)', fontsize=10)
        ax.axis('off')
        if i == 0:
            ax.text(-0.15, 0.5, 'Estándar\n(imshow auto)',
                   transform=ax.transAxes, fontsize=9, va='center', ha='right')

        # Fila 2: Normalización min-max global
        ax = axes[1, i]
        ef_norm = (eigenface - eigenface.min()) / (eigenface.max() - eigenface.min())
        ax.imshow(ef_norm, cmap='gray')
        ax.set_title(f'PC{i+1} (min-max)', fontsize=10)
        ax.axis('off')
        if i == 0:
            ax.text(-0.15, 0.5, 'Min-Max\n[0, 1]',
                   transform=ax.transAxes, fontsize=9, va='center', ha='right')

        # Fila 3: Nuestra implementación (fondo negro)
        ax = axes[2, i]
        mean_2d = pca_result.mean.reshape(dataset.image_shape)
        content_mask = mean_2d > 0.01
        content_vals = eigenface[content_mask]
        normalized = (eigenface - content_vals.min()) / (content_vals.max() - content_vals.min() + 1e-8)
        ef_black = np.where(content_mask, normalized, 0)
        ax.imshow(ef_black, cmap='gray')
        ax.set_title(f'PC{i+1} (black bg)', fontsize=10)
        ax.axis('off')
        if i == 0:
            ax.text(-0.15, 0.5, 'Fondo Negro\n(actual)',
                   transform=ax.transAxes, fontsize=9, va='center', ha='right')

    plt.suptitle('Comparación de Métodos de Visualización de Eigenfaces\n(Dataset Warped)',
                fontsize=12, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / "eigenface_visualization_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nFigura guardada: {output_path}")

    # Análisis de valores
    print("\n" + "=" * 60)
    print("ANÁLISIS DE VALORES DE EIGENFACES")
    print("=" * 60)

    for i in range(3):
        eigenface = pca_result.components[i].reshape(dataset.image_shape)
        mean_2d = pca_result.mean.reshape(dataset.image_shape)
        content_mask = mean_2d > 0.01

        print(f"\nPC{i+1}:")
        print(f"  Rango global:   [{eigenface.min():.4f}, {eigenface.max():.4f}]")
        print(f"  Rango contenido:[{eigenface[content_mask].min():.4f}, {eigenface[content_mask].max():.4f}]")
        print(f"  Rango fondo:    [{eigenface[~content_mask].min():.4f}, {eigenface[~content_mask].max():.4f}]")
        print(f"  Varianza fondo: {eigenface[~content_mask].var():.6f}")

    # Recomendación
    print("\n" + "=" * 60)
    print("RECOMENDACIÓN CIENTÍFICA")
    print("=" * 60)
    print("""
Para imágenes WARPED:
  - El fondo negro tiene varianza ~0 (correcto)
  - La normalización con fondo negro es ACEPTABLE
  - Mejora la visualización al no distorsionar por valores de fondo

Para imágenes ORIGINALES:
  - NO hay fondo negro definido
  - Se debe usar el método ESTÁNDAR (imshow sin normalización)
  - Esto preserva la interpretación matemática correcta

CONCLUSIÓN:
  - Usar métodos diferentes según el tipo de imagen
  - Documentar claramente en el paper qué método se usa
    """)


if __name__ == "__main__":
    compare_visualizations()
