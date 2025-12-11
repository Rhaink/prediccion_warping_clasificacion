#!/usr/bin/env python3
"""
Analisis Visual de Marcas Hospitalarias: Original vs Warped

Este script compara visualmente imagenes originales vs warped para verificar
la hipotesis de que el warping "elimina marcas hospitalarias".

Funcionalidad:
1. Carga imagenes originales del dataset COVID-19_Radiography_Dataset
2. Carga las correspondientes imagenes warped
3. Crea visualizaciones lado-a-lado con anotaciones en regiones criticas
4. Calcula y muestra el fill rate de cada imagen warped
5. Genera un mosaico de ejemplos representativos (3x2 grid)

Objetivo: Proveer evidencia visual para verificar o refutar la hipotesis
de que el warping elimina marcas hospitalarias (texto en esquinas, logos, fechas).

Autor: Proyecto Tesis Maestria
Fecha: Diciembre 2024
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Optional
import random

from src_v2.processing.warp import compute_fill_rate


# Configuracion
ORIGINAL_DATASET_DIR = PROJECT_ROOT / "data" / "dataset" / "COVID-19_Radiography_Dataset"
WARPED_DATASET_DIR = PROJECT_ROOT / "outputs" / "full_warped_dataset"
FULL_COVERAGE_WARPED_DIR = PROJECT_ROOT / "outputs" / "full_coverage_warped_dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visual_analysis"

# Clases disponibles
CLASSES = ['COVID', 'Normal', 'Viral Pneumonia']
CLASS_MAPPING = {
    'COVID': 'COVID',
    'Normal': 'Normal',
    'Viral Pneumonia': 'Viral_Pneumonia'
}

# Regiones tipicas de marcas hospitalarias (como proporcion de la imagen)
# Formato: (x_start, y_start, width, height) en proporcion [0, 1]
HOSPITAL_MARK_REGIONS = {
    'top_left': (0.0, 0.0, 0.25, 0.15),
    'top_right': (0.75, 0.0, 0.25, 0.15),
    'bottom_left': (0.0, 0.85, 0.25, 0.15),
    'bottom_right': (0.75, 0.85, 0.25, 0.15),
}

# Colores para anotaciones
REGION_COLORS = {
    'top_left': 'red',
    'top_right': 'orange',
    'bottom_left': 'cyan',
    'bottom_right': 'yellow',
}


def load_image(image_path: Path) -> Optional[np.ndarray]:
    """
    Carga una imagen en escala de grises.

    Args:
        image_path: Path a la imagen

    Returns:
        Imagen como numpy array o None si falla
    """
    if not image_path.exists():
        return None

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    return img


def find_matching_warped_image(original_path: Path, class_name: str,
                                use_full_coverage: bool = False) -> Optional[Path]:
    """
    Encuentra la imagen warped correspondiente a una imagen original.

    Args:
        original_path: Path a la imagen original
        class_name: Nombre de la clase (COVID, Normal, Viral Pneumonia)
        use_full_coverage: Si usar el dataset full_coverage

    Returns:
        Path a la imagen warped o None si no existe
    """
    base_name = original_path.stem
    warped_name = f"{base_name}_warped.png"
    warped_class = CLASS_MAPPING[class_name]

    warped_dir = FULL_COVERAGE_WARPED_DIR if use_full_coverage else WARPED_DATASET_DIR

    for split in ['train', 'val', 'test']:
        warped_path = warped_dir / split / warped_class / warped_name
        if warped_path.exists():
            return warped_path

    return None


def annotate_hospital_regions(ax, img_shape: Tuple[int, int], alpha: float = 0.3):
    """
    Anota las regiones tipicas de marcas hospitalarias en un axis.

    Args:
        ax: Matplotlib axis
        img_shape: (height, width) de la imagen
        alpha: Transparencia de las anotaciones
    """
    height, width = img_shape

    for region_name, (x_prop, y_prop, w_prop, h_prop) in HOSPITAL_MARK_REGIONS.items():
        x = x_prop * width
        y = y_prop * height
        w = w_prop * width
        h = h_prop * height

        color = REGION_COLORS[region_name]

        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor=color,
            facecolor=color,
            alpha=alpha,
            linestyle='--'
        )
        ax.add_patch(rect)


def create_comparison_figure(
    original_img: np.ndarray,
    warped_img: np.ndarray,
    title: str,
    fill_rate: float
) -> plt.Figure:
    """
    Crea una figura comparando original vs warped lado a lado.

    Args:
        original_img: Imagen original
        warped_img: Imagen warped
        title: Titulo de la figura
        fill_rate: Fill rate de la imagen warped

    Returns:
        Figura de matplotlib
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Original
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    annotate_hospital_regions(axes[0], original_img.shape, alpha=0.25)

    for i, (region_name, color) in enumerate(REGION_COLORS.items()):
        axes[0].plot([], [], color=color, linewidth=2, linestyle='--',
                    label=region_name.replace('_', ' ').title())
    axes[0].legend(loc='upper left', fontsize=8, framealpha=0.8)

    # Warped
    axes[1].imshow(warped_img, cmap='gray')
    axes[1].set_title(f'Warped Image (Fill Rate: {fill_rate:.2%})',
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')
    annotate_hospital_regions(axes[1], warped_img.shape, alpha=0.25)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    return fig


def create_mosaic_figure(
    image_pairs: List[Tuple[np.ndarray, np.ndarray, str, float]],
    n_rows: int = 3,
    n_cols: int = 2
) -> plt.Figure:
    """
    Crea un mosaico de comparaciones original/warped.

    Args:
        image_pairs: Lista de tuplas (original, warped, title, fill_rate)
        n_rows: Numero de filas en el mosaico
        n_cols: Numero de columnas (siempre 2 por par original/warped)

    Returns:
        Figura de matplotlib
    """
    fig = plt.figure(figsize=(7 * n_cols, 5 * n_rows))
    gs = GridSpec(n_rows, n_cols * 2, figure=fig, hspace=0.3, wspace=0.15)

    for i, (original, warped, title, fill_rate) in enumerate(image_pairs[:n_rows]):
        row = i

        # Original
        ax_orig = fig.add_subplot(gs[row, 0:2])
        ax_orig.imshow(original, cmap='gray')
        ax_orig.set_title(f'{title}\nOriginal', fontsize=11, fontweight='bold')
        ax_orig.axis('off')
        annotate_hospital_regions(ax_orig, original.shape, alpha=0.2)

        # Warped
        ax_warp = fig.add_subplot(gs[row, 2:4])
        ax_warp.imshow(warped, cmap='gray')
        ax_warp.set_title(f'Warped (Fill: {fill_rate:.1%})',
                         fontsize=11, fontweight='bold')
        ax_warp.axis('off')
        annotate_hospital_regions(ax_warp, warped.shape, alpha=0.2)

    fig.suptitle('Hospital Marks Analysis: Original vs Warped Comparison',
                fontsize=16, fontweight='bold', y=0.995)

    return fig


def analyze_regional_intensities(
    original_img: np.ndarray,
    warped_img: np.ndarray
) -> dict:
    """
    Analiza las intensidades promedio en las regiones de marcas hospitalarias.

    Args:
        original_img: Imagen original
        warped_img: Imagen warped

    Returns:
        Diccionario con estadisticas por region
    """
    height, width = original_img.shape
    stats = {}

    for region_name, (x_prop, y_prop, w_prop, h_prop) in HOSPITAL_MARK_REGIONS.items():
        x = int(x_prop * width)
        y = int(y_prop * height)
        w = int(w_prop * width)
        h = int(h_prop * height)

        orig_region = original_img[y:y+h, x:x+w]
        warp_region = warped_img[y:y+h, x:x+w]

        stats[region_name] = {
            'original_mean': orig_region.mean(),
            'original_std': orig_region.std(),
            'warped_mean': warp_region.mean(),
            'warped_std': warp_region.std(),
            'warped_black_pixels': (warp_region == 0).sum(),
            'warped_black_ratio': (warp_region == 0).sum() / warp_region.size
        }

    return stats


def collect_sample_images(
    n_samples_per_class: int = 2,
    seed: int = 42,
    use_full_coverage: bool = False
) -> List[Tuple[np.ndarray, np.ndarray, str, float]]:
    """
    Recolecta imagenes de muestra de cada clase para analisis.

    Args:
        n_samples_per_class: Numero de muestras por clase
        seed: Semilla para reproducibilidad
        use_full_coverage: Si usar el dataset full_coverage

    Returns:
        Lista de tuplas (original, warped, title, fill_rate)
    """
    random.seed(seed)
    image_pairs = []

    for class_name in CLASSES:
        print(f"\nRecolectando muestras de {class_name}...")

        if class_name == 'Viral Pneumonia':
            orig_dir = ORIGINAL_DATASET_DIR / 'Viral Pneumonia' / 'images'
        else:
            orig_dir = ORIGINAL_DATASET_DIR / class_name / 'images'

        if not orig_dir.exists():
            print(f"  WARNING: Directorio no encontrado: {orig_dir}")
            continue

        image_files = list(orig_dir.glob('*.png'))

        if len(image_files) == 0:
            print(f"  WARNING: No se encontraron imagenes en {orig_dir}")
            continue

        selected_files = random.sample(image_files, min(n_samples_per_class, len(image_files)))

        for img_path in selected_files:
            original = load_image(img_path)
            if original is None:
                print(f"  WARNING: No se pudo cargar {img_path}")
                continue

            warped_path = find_matching_warped_image(img_path, class_name, use_full_coverage)
            if warped_path is None:
                print(f"  WARNING: No se encontro warped para {img_path.name}")
                continue

            warped = load_image(warped_path)
            if warped is None:
                print(f"  WARNING: No se pudo cargar {warped_path}")
                continue

            fill_rate = compute_fill_rate(warped)

            title = f"{class_name}: {img_path.name}"
            image_pairs.append((original, warped, title, fill_rate))
            print(f"  Agregado: {img_path.name} (fill rate: {fill_rate:.2%})")

    return image_pairs


def main():
    """Funcion principal."""
    import argparse

    parser = argparse.ArgumentParser(description='Analisis visual de marcas hospitalarias')
    parser.add_argument('--full-coverage', action='store_true',
                       help='Usar dataset full_coverage en lugar del original')
    parser.add_argument('--samples', type=int, default=2,
                       help='Numero de muestras por clase')
    args = parser.parse_args()

    print("="*80)
    print("ANALISIS VISUAL DE MARCAS HOSPITALARIAS: ORIGINAL VS WARPED")
    print("="*80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDirectorio de salida: {OUTPUT_DIR}")
    print(f"Usando dataset: {'full_coverage' if args.full_coverage else 'standard'}")

    print("\n" + "="*80)
    print("PASO 1: Recolectar imagenes de muestra")
    print("="*80)

    image_pairs = collect_sample_images(
        n_samples_per_class=args.samples,
        seed=42,
        use_full_coverage=args.full_coverage
    )

    if len(image_pairs) == 0:
        print("\nERROR: No se encontraron pares de imagenes para analizar.")
        return

    print(f"\nTotal de pares recolectados: {len(image_pairs)}")

    print("\n" + "="*80)
    print("PASO 2: Generar comparaciones individuales")
    print("="*80)

    for i, (original, warped, title, fill_rate) in enumerate(image_pairs):
        print(f"\nProcesando {i+1}/{len(image_pairs)}: {title}")

        fig = create_comparison_figure(original, warped, title, fill_rate)

        safe_title = title.replace(':', '_').replace(' ', '_').replace('/', '_')
        output_path = OUTPUT_DIR / f"comparison_{i+1:02d}_{safe_title}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  Guardado: {output_path.name}")

        stats = analyze_regional_intensities(original, warped)
        print(f"  Estadisticas regionales:")
        for region_name, region_stats in stats.items():
            print(f"    {region_name}:")
            print(f"      Original: mean={region_stats['original_mean']:.1f}, std={region_stats['original_std']:.1f}")
            print(f"      Warped:   mean={region_stats['warped_mean']:.1f}, std={region_stats['warped_std']:.1f}, black={region_stats['warped_black_ratio']:.1%}")

    print("\n" + "="*80)
    print("PASO 3: Generar mosaico de ejemplos")
    print("="*80)

    mosaic_fig = create_mosaic_figure(image_pairs, n_rows=3, n_cols=2)
    suffix = "_full_coverage" if args.full_coverage else ""
    mosaic_path = OUTPUT_DIR / f"hospital_marks_mosaic{suffix}.png"
    mosaic_fig.savefig(mosaic_path, dpi=200, bbox_inches='tight')
    plt.close(mosaic_fig)

    print(f"\nMosaico guardado: {mosaic_path}")

    print("\n" + "="*80)
    print("RESUMEN DE FILL RATES")
    print("="*80)

    fill_rates = [fr for _, _, _, fr in image_pairs]
    print(f"Fill Rate Promedio: {np.mean(fill_rates):.2%}")
    print(f"Fill Rate Minimo:   {np.min(fill_rates):.2%}")
    print(f"Fill Rate Maximo:   {np.max(fill_rates):.2%}")
    print(f"Desviacion Std:     {np.std(fill_rates):.2%}")

    print("\n" + "="*80)
    print("ANALISIS COMPLETADO")
    print("="*80)
    print(f"\nArchivos generados en: {OUTPUT_DIR}")
    print(f"  - {len(image_pairs)} comparaciones individuales")
    print(f"  - 1 mosaico de ejemplos")
    print("\nInterpretacion:")
    print("  - Las regiones marcadas (esquinas) muestran donde tipicamente aparecen marcas hospitalarias")
    print("  - Compare visualmente si estas regiones contienen texto/logos en originales")
    print("  - Verifique si el warping preserva o elimina informacion en estas regiones")
    print("  - El fill rate indica cuanto de la imagen warped es contenido (vs negro)")
    print("  - Black pixels en regiones criticas pueden indicar perdida de informacion")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
