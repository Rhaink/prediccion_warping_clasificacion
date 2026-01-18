#!/usr/bin/env python3
"""
Script para generar figura comparativa: imagen normalizada vs imagen normalizada + SAHS.

Genera la figura F4.13_warped_sahs.png para la tesis.
"""

import matplotlib
matplotlib.use('Agg')  # Backend sin display

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def enhance_contrast_sahs_masked(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Aplica el algoritmo SAHS solo a la región pulmonar (píxeles > threshold).

    El fondo negro (píxeles <= threshold) se mantiene intacto.

    Statistical Asymmetrical Histogram Stretching (SAHS):
    - Calcula límites de estiramiento asimétricos basados en la media
    - Factor 2.5 para el límite superior
    - Factor 2.0 para el límite inferior
    """
    if image is None:
        raise ValueError("La imagen de entrada es None")

    # Convertir a escala de grises si es necesario
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    # Crear máscara de la región pulmonar (excluyendo fondo negro)
    mask = gray_image > threshold
    lung_pixels = gray_image[mask].astype(np.float64)

    if lung_pixels.size == 0:
        return gray_image

    # Calcular la media solo de los píxeles pulmonares
    gray_mean = np.mean(lung_pixels)

    # Separar píxeles por encima y debajo de la media
    above_mean = lung_pixels[lung_pixels > gray_mean]
    below_or_equal_mean = lung_pixels[lung_pixels <= gray_mean]

    # Calcular límites usando desviación estándar asimétrica
    max_value = gray_mean
    min_value = gray_mean

    if above_mean.size > 0:
        # Factor 2.5 para el límite superior
        std_above = np.sqrt(np.mean((above_mean - gray_mean) ** 2))
        max_value = gray_mean + 2.5 * std_above

    if below_or_equal_mean.size > 0:
        # Factor 2.0 para el límite inferior
        std_below = np.sqrt(np.mean((below_or_equal_mean - gray_mean) ** 2))
        min_value = gray_mean - 2.0 * std_below

    # Crear imagen de salida (iniciar con ceros para mantener fondo negro)
    enhanced_image = np.zeros_like(gray_image)

    # Aplicar transformación solo a la región pulmonar
    if max_value != min_value:
        transformed = (255 / (max_value - min_value)) * (gray_image.astype(np.float64) - min_value)
        transformed = np.clip(transformed, 0, 255).astype(np.uint8)
        # Aplicar solo donde hay región pulmonar
        enhanced_image[mask] = transformed[mask]
    else:
        enhanced_image[mask] = gray_image[mask]

    return enhanced_image


def create_histogram_lung_only(image: np.ndarray, ax, title: str, color: str = 'steelblue', threshold: int = 10):
    """Crea un histograma solo de la región pulmonar (excluyendo fondo)."""
    # Extraer solo píxeles de la región pulmonar
    lung_pixels = image[image > threshold]

    if lung_pixels.size > 0:
        ax.hist(lung_pixels.ravel(), bins=50, range=(threshold, 256), color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_xlim(0, 260)
        ax.set_xlabel('Intensidad', fontsize=9)
        ax.set_ylabel('Frecuencia', fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis='both', labelsize=8)

        # Añadir línea vertical para la media
        mean_val = lung_pixels.mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Media: {mean_val:.0f}')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=10)


def main():
    # Configuración de rutas
    base_dir = Path("/home/donrobot/Projects/prediccion_warping_clasificacion")
    warped_dir = base_dir / "outputs/shape_analysis/warped/lung_only_images"
    output_path = base_dir / "docs/Tesis/Figures/F4.13_warped_sahs.png"

    # Umbral para considerar píxeles como fondo
    BG_THRESHOLD = 10

    # Seleccionar imágenes representativas de cada categoría
    sample_images = {
        'COVID-19': warped_dir / "COVID/COVID-1522_warped.png",
        'Normal': warped_dir / "Normal/Normal-3945_warped.png",
        'Neumonía Viral': warped_dir / "Viral_Pneumonia/Viral Pneumonia-600_warped.png",
    }

    # Verificar que existan las imágenes
    for name, path in sample_images.items():
        if not path.exists():
            print(f"Advertencia: No se encontró {path}")
            category_dir = path.parent
            if category_dir.exists():
                alternatives = list(category_dir.glob("*.png"))
                if alternatives:
                    sample_images[name] = alternatives[0]
                    print(f"  Usando alternativa: {alternatives[0].name}")

    # Crear figura con subplots
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)

    for row, (category, img_path) in enumerate(sample_images.items()):
        # Cargar imagen normalizada
        img_warped = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if img_warped is None:
            print(f"Error: No se pudo cargar {img_path}")
            continue

        # Aplicar SAHS solo a región pulmonar
        img_sahs = enhance_contrast_sahs_masked(img_warped, threshold=BG_THRESHOLD)

        # Subplot: imagen normalizada
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
        ax1.set_title(f'{category}\n(a) Normalizada', fontsize=11)
        ax1.axis('off')

        # Subplot: histograma de imagen normalizada (solo región pulmonar)
        ax2 = fig.add_subplot(gs[row, 1])
        create_histogram_lung_only(img_warped, ax2, 'Histograma (región pulmonar)', 'steelblue', BG_THRESHOLD)

        # Subplot: imagen con SAHS aplicado
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.imshow(img_sahs, cmap='gray', vmin=0, vmax=255)
        ax3.set_title(f'{category}\n(b) SAHS aplicado', fontsize=11)
        ax3.axis('off')

        # Subplot: histograma SAHS (solo región pulmonar)
        ax4 = fig.add_subplot(gs[row, 3])
        create_histogram_lung_only(img_sahs, ax4, 'Histograma SAHS', 'darkorange', BG_THRESHOLD)

        # Calcular estadísticas solo de región pulmonar
        lung_orig = img_warped[img_warped > BG_THRESHOLD]
        lung_sahs = img_sahs[img_sahs > BG_THRESHOLD]

        print(f"\n{category} (solo región pulmonar):")
        print(f"  Original - Media: {lung_orig.mean():.1f}, Std: {lung_orig.std():.1f}, "
              f"Min: {lung_orig.min()}, Max: {lung_orig.max()}")
        print(f"  SAHS     - Media: {lung_sahs.mean():.1f}, Std: {lung_sahs.std():.1f}, "
              f"Min: {lung_sahs.min()}, Max: {lung_sahs.max()}")

    # Título general
    fig.suptitle('Efecto del preprocesamiento SAHS sobre imágenes normalizadas',
                 fontsize=14, fontweight='bold', y=0.98)

    # Guardar figura
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nFigura guardada en: {output_path}")

    plt.close('all')
    print("\nFigura generada exitosamente!")


if __name__ == "__main__":
    main()
