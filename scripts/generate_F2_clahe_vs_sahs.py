#!/usr/bin/env python3
"""
Script para generar figura comparativa: CLAHE vs SAHS sobre imagenes originales.

Genera la figura F2.3_clahe_vs_sahs.png para el marco teorico de la tesis.

Esta figura muestra la diferencia entre CLAHE (preprocesamiento local) y
SAHS (preprocesamiento global asimetrico) aplicados a radiografias originales.
"""

import matplotlib
matplotlib.use('Agg')  # Backend sin display

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 4) -> np.ndarray:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Parameters:
        image: Imagen en escala de grises
        clip_limit: Limite de contraste para evitar amplificacion de ruido
        tile_size: Tamano de la cuadricula de tiles (tile_size x tile_size)

    Returns:
        Imagen con CLAHE aplicado
    """
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(image)


def apply_sahs(image: np.ndarray) -> np.ndarray:
    """
    Aplica el algoritmo SAHS (Statistical Asymmetrical Histogram Stretching).

    SAHS calcula limites de estiramiento asimetricos basados en la media:
    - Factor 2.5 para el limite superior
    - Factor 2.0 para el limite inferior

    Parameters:
        image: Imagen en escala de grises

    Returns:
        Imagen con SAHS aplicado
    """
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    pixels = gray_image.astype(np.float64).ravel()

    # Calcular la media
    gray_mean = np.mean(pixels)

    # Separar pixeles por encima y debajo de la media
    above_mean = pixels[pixels > gray_mean]
    below_or_equal_mean = pixels[pixels <= gray_mean]

    # Calcular limites usando desviacion estandar asimetrica
    max_value = gray_mean
    min_value = gray_mean

    if above_mean.size > 0:
        # Factor 2.5 para el limite superior
        std_above = np.sqrt(np.mean((above_mean - gray_mean) ** 2))
        max_value = gray_mean + 2.5 * std_above

    if below_or_equal_mean.size > 0:
        # Factor 2.0 para el limite inferior
        std_below = np.sqrt(np.mean((below_or_equal_mean - gray_mean) ** 2))
        min_value = gray_mean - 2.0 * std_below

    # Aplicar transformacion lineal
    if max_value != min_value:
        enhanced = (255 / (max_value - min_value)) * (gray_image.astype(np.float64) - min_value)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    else:
        enhanced = gray_image

    return enhanced


def create_histogram(image: np.ndarray, ax, title: str, color: str = 'steelblue'):
    """Crea un histograma de la imagen."""
    ax.hist(image.ravel(), bins=50, range=(0, 256), color=color, alpha=0.7,
            edgecolor='black', linewidth=0.5)
    ax.set_xlim(0, 260)
    ax.set_xlabel('Intensidad', fontsize=9)
    ax.set_ylabel('Frecuencia', fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis='both', labelsize=8)

    # Linea vertical para la media
    mean_val = image.mean()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5,
               label=f'Media: {mean_val:.0f}')
    ax.legend(fontsize=8)


def main():
    # Configuracion de rutas
    base_dir = Path("/home/donrobot/Projects/prediccion_warping_clasificacion")
    dataset_dir = base_dir / "data/dataset/COVID-19_Radiography_Dataset"
    output_path = base_dir / "docs/Tesis/Figures/F2.3_clahe_vs_sahs.png"

    # Seleccionar imagenes representativas de cada categoria
    sample_images = {
        'COVID-19': dataset_dir / "COVID/images/COVID-1522.png",
        'Normal': dataset_dir / "Normal/images/Normal-3945.png",
        'Neumonia Viral': dataset_dir / "Viral Pneumonia/images/Viral Pneumonia-600.png",
    }

    # Verificar y buscar alternativas si es necesario
    for name, path in list(sample_images.items()):
        if not path.exists():
            print(f"Advertencia: No se encontro {path}")
            category_dir = path.parent
            if category_dir.exists():
                alternatives = list(category_dir.glob("*.png"))
                if alternatives:
                    sample_images[name] = alternatives[0]
                    print(f"  Usando alternativa: {alternatives[0].name}")

    # Crear figura con subplots: 3 filas (clases) x 6 columnas (original, hist, CLAHE, hist, SAHS, hist)
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 6, figure=fig, hspace=0.35, wspace=0.3,
                  width_ratios=[1.2, 1, 1.2, 1, 1.2, 1])

    for row, (category, img_path) in enumerate(sample_images.items()):
        # Cargar imagen original
        img_original = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if img_original is None:
            print(f"Error: No se pudo cargar {img_path}")
            continue

        # Aplicar preprocesamiento
        img_clahe = apply_clahe(img_original, clip_limit=2.0, tile_size=4)
        img_sahs = apply_sahs(img_original)

        # Columna 0: Imagen original
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.imshow(img_original, cmap='gray', vmin=0, vmax=255)
        ax1.set_title(f'{category}\n(a) Original', fontsize=11)
        ax1.axis('off')

        # Columna 1: Histograma original
        ax2 = fig.add_subplot(gs[row, 1])
        create_histogram(img_original, ax2, 'Histograma', 'gray')

        # Columna 2: CLAHE
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.imshow(img_clahe, cmap='gray', vmin=0, vmax=255)
        ax3.set_title(f'(b) CLAHE', fontsize=11)
        ax3.axis('off')

        # Columna 3: Histograma CLAHE
        ax4 = fig.add_subplot(gs[row, 3])
        create_histogram(img_clahe, ax4, 'Histograma CLAHE', 'steelblue')

        # Columna 4: SAHS
        ax5 = fig.add_subplot(gs[row, 4])
        ax5.imshow(img_sahs, cmap='gray', vmin=0, vmax=255)
        ax5.set_title(f'(c) SAHS', fontsize=11)
        ax5.axis('off')

        # Columna 5: Histograma SAHS
        ax6 = fig.add_subplot(gs[row, 5])
        create_histogram(img_sahs, ax6, 'Histograma SAHS', 'darkorange')

        # Imprimir estadisticas
        print(f"\n{category}:")
        print(f"  Original - Media: {img_original.mean():.1f}, Std: {img_original.std():.1f}")
        print(f"  CLAHE    - Media: {img_clahe.mean():.1f}, Std: {img_clahe.std():.1f}")
        print(f"  SAHS     - Media: {img_sahs.mean():.1f}, Std: {img_sahs.std():.1f}")

    # Titulo general
    fig.suptitle('Comparacion de tecnicas de mejora de contraste: CLAHE vs SAHS',
                 fontsize=14, fontweight='bold', y=0.98)

    # Guardar figura
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nFigura guardada en: {output_path}")

    plt.close('all')
    print("\nFigura generada exitosamente!")


if __name__ == "__main__":
    main()
