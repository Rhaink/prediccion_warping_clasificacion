#!/usr/bin/env python3
"""
Script para generar visualizaciones de triangulacion de Delaunay para TODAS las imagenes del test set.

Genera dos visualizaciones por imagen:
1. Ground Truth (GT) con triangulos verdes
2. Prediccion con triangulos rojos

Las imagenes se muestran correctamente en escala de grises.

Uso:
    python scripts/generate_all_visualizations.py
    python scripts/generate_all_visualizations.py --output-dir outputs/all_viz
    python scripts/generate_all_visualizations.py --side-by-side  # GT y Pred lado a lado
"""

import sys
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
from tqdm import tqdm

from src_v2.data.utils import get_image_path


# Nombres de landmarks
LANDMARK_NAMES = [
    'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8',
    'L9', 'L10', 'L11', 'L12', 'L13', 'L14', 'L15'
]


def load_predictions_and_triangles(predictions_dir: Path):
    """Carga predicciones y triangulacion de Delaunay."""
    # Cargar predicciones
    npz_path = predictions_dir / 'test_predictions.npz'
    if not npz_path.exists():
        raise FileNotFoundError(f"No se encontro {npz_path}. Ejecuta primero extract_predictions.py")

    npz_data = np.load(npz_path, allow_pickle=True)
    data = {
        'predictions': npz_data['predictions'],  # (96, 15, 2) en pixeles 224
        'ground_truth': npz_data['ground_truth'],  # (96, 15, 2) en pixeles 224
        'errors': npz_data['errors'],
        'image_names': npz_data['image_names'].tolist(),
        'categories': npz_data['categories'].tolist(),
    }

    # Cargar triangulacion
    tri_path = predictions_dir / 'delaunay_triangles.json'
    with open(tri_path, 'r') as f:
        tri_data = json.load(f)
    triangles = np.array(tri_data['triangles'])

    return data, triangles


def visualize_single_image(
    image_path: Path,
    landmarks: np.ndarray,
    triangles: np.ndarray,
    output_path: Path,
    title: str = None,
    landmark_color: str = 'red',
    triangle_color: str = 'yellow',
    show_indices: bool = True,
    figsize: tuple = (8, 8)
):
    """
    Visualiza landmarks y triangulacion sobre una imagen en escala de grises.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Cargar imagen en escala de grises
    img = Image.open(image_path)
    img_array = np.array(img)

    # Mostrar imagen en escala de grises
    ax.imshow(img_array, cmap='gray', vmin=0, vmax=255)

    # Dibujar triangulos
    for tri in triangles:
        pts = landmarks[tri]
        polygon = Polygon(pts, fill=False, edgecolor=triangle_color, linewidth=1.5, alpha=0.9)
        ax.add_patch(polygon)

    # Dibujar landmarks
    ax.scatter(landmarks[:, 0], landmarks[:, 1], c=landmark_color, s=80, zorder=5,
               edgecolors='white', linewidths=1.5)

    # Etiquetas de landmarks
    if show_indices:
        for i, (x, y) in enumerate(landmarks):
            ax.annotate(f'L{i+1}', (x, y), xytext=(4, 4), textcoords='offset points',
                       fontsize=7, color='white', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.15', facecolor=landmark_color, alpha=0.8))

    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white',
                pad_inches=0.1)
    plt.close()


def visualize_comparison(
    image_path: Path,
    gt_landmarks: np.ndarray,
    pred_landmarks: np.ndarray,
    triangles: np.ndarray,
    output_path: Path,
    title: str = None,
    figsize: tuple = (8, 8)
):
    """
    Visualiza comparacion GT vs Prediccion en una sola imagen.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Cargar imagen
    img = Image.open(image_path)
    img_array = np.array(img)
    ax.imshow(img_array, cmap='gray', vmin=0, vmax=255)

    # Triangulos GT (verde, solido)
    for tri in triangles:
        pts = gt_landmarks[tri]
        polygon = Polygon(pts, fill=False, edgecolor='lime', linewidth=2, alpha=0.9)
        ax.add_patch(polygon)

    # Triangulos Pred (rojo, punteado)
    for tri in triangles:
        pts = pred_landmarks[tri]
        polygon = Polygon(pts, fill=False, edgecolor='red', linewidth=2,
                         alpha=0.9, linestyle='--')
        ax.add_patch(polygon)

    # Landmarks GT
    ax.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], c='lime', s=60, zorder=5,
               edgecolors='white', linewidths=1, marker='o', label='Ground Truth')

    # Landmarks Pred
    ax.scatter(pred_landmarks[:, 0], pred_landmarks[:, 1], c='red', s=60, zorder=5,
               edgecolors='white', linewidths=1, marker='^', label='Prediccion')

    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white',
                pad_inches=0.1)
    plt.close()


def visualize_side_by_side(
    image_path: Path,
    gt_landmarks: np.ndarray,
    pred_landmarks: np.ndarray,
    triangles: np.ndarray,
    output_path: Path,
    title: str = None,
    error: float = None
):
    """
    Visualiza GT y Prediccion lado a lado.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Cargar imagen
    img = Image.open(image_path)
    img_array = np.array(img)

    # GT (izquierda)
    axes[0].imshow(img_array, cmap='gray', vmin=0, vmax=255)
    for tri in triangles:
        pts = gt_landmarks[tri]
        polygon = Polygon(pts, fill=False, edgecolor='lime', linewidth=1.5, alpha=0.9)
        axes[0].add_patch(polygon)
    axes[0].scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], c='lime', s=60,
                    edgecolors='white', linewidths=1)
    axes[0].set_title('Ground Truth', fontsize=12, fontweight='bold', color='green')
    axes[0].axis('off')

    # Prediccion (derecha)
    axes[1].imshow(img_array, cmap='gray', vmin=0, vmax=255)
    for tri in triangles:
        pts = pred_landmarks[tri]
        polygon = Polygon(pts, fill=False, edgecolor='red', linewidth=1.5, alpha=0.9)
        axes[1].add_patch(polygon)
    axes[1].scatter(pred_landmarks[:, 0], pred_landmarks[:, 1], c='red', s=60,
                    edgecolors='white', linewidths=1)
    error_text = f' (Error: {error:.2f} px)' if error is not None else ''
    axes[1].set_title(f'Prediccion{error_text}', fontsize=12, fontweight='bold', color='red')
    axes[1].axis('off')

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white',
                pad_inches=0.1)
    plt.close()


def generate_all_visualizations(
    data: dict,
    triangles: np.ndarray,
    data_root: Path,
    output_dir: Path,
    mode: str = 'comparison'  # 'comparison', 'side_by_side', 'separate'
):
    """
    Genera visualizaciones para todas las imagenes del test set.
    """
    n_images = len(data['image_names'])

    # Crear subdirectorios por categoria
    for cat in set(data['categories']):
        cat_dir = output_dir / cat.lower().replace(' ', '_')
        cat_dir.mkdir(parents=True, exist_ok=True)

    # Escalar de 224 a 299 (tamaño original de las imagenes)
    scale = 299 / 224

    print(f"\nGenerando {n_images} visualizaciones (modo: {mode})...")

    for i in tqdm(range(n_images), desc="Procesando"):
        image_name = data['image_names'][i]
        category = data['categories'][i]

        # Obtener landmarks escalados
        gt_landmarks = data['ground_truth'][i] * scale
        pred_landmarks = data['predictions'][i] * scale
        error = data['errors'][i].mean()

        # Obtener path de imagen
        image_path = get_image_path(image_name, category, data_root)

        # Directorio de salida por categoria
        cat_dir = output_dir / category.lower().replace(' ', '_')

        # Generar visualizacion segun el modo
        if mode == 'comparison':
            output_path = cat_dir / f'{image_name}_comparison.png'
            visualize_comparison(
                image_path, gt_landmarks, pred_landmarks, triangles, output_path,
                title=f'{image_name} ({category}) - Error: {error:.2f} px'
            )

        elif mode == 'side_by_side':
            output_path = cat_dir / f'{image_name}_sidebyside.png'
            visualize_side_by_side(
                image_path, gt_landmarks, pred_landmarks, triangles, output_path,
                title=f'{image_name} ({category})', error=error
            )

        elif mode == 'separate':
            # GT
            gt_path = cat_dir / f'{image_name}_gt.png'
            visualize_single_image(
                image_path, gt_landmarks, triangles, gt_path,
                title=f'{image_name} - Ground Truth',
                landmark_color='lime', triangle_color='lime'
            )

            # Prediccion
            pred_path = cat_dir / f'{image_name}_pred.png'
            visualize_single_image(
                image_path, pred_landmarks, triangles, pred_path,
                title=f'{image_name} - Prediccion (Error: {error:.2f} px)',
                landmark_color='red', triangle_color='red'
            )


def generate_summary_grid(
    data: dict,
    triangles: np.ndarray,
    data_root: Path,
    output_path: Path,
    n_samples: int = 12
):
    """
    Genera una grilla resumen con ejemplos de cada categoria.
    """
    categories = ['Normal', 'COVID', 'Viral_Pneumonia']
    samples_per_cat = n_samples // len(categories)

    fig, axes = plt.subplots(len(categories), samples_per_cat, figsize=(samples_per_cat * 3, len(categories) * 3))

    scale = 299 / 224

    for cat_idx, category in enumerate(categories):
        # Encontrar indices de esta categoria
        cat_indices = [i for i, c in enumerate(data['categories']) if c == category]

        # Ordenar por error para mostrar variedad
        cat_errors = [(i, data['errors'][i].mean()) for i in cat_indices]
        cat_errors.sort(key=lambda x: x[1])

        # Seleccionar muestras distribuidas
        step = max(1, len(cat_errors) // samples_per_cat)
        selected = [cat_errors[j * step][0] for j in range(min(samples_per_cat, len(cat_errors)))]

        for col_idx, i in enumerate(selected):
            ax = axes[cat_idx, col_idx]

            image_name = data['image_names'][i]
            gt_landmarks = data['ground_truth'][i] * scale
            pred_landmarks = data['predictions'][i] * scale
            error = data['errors'][i].mean()

            image_path = get_image_path(image_name, category, data_root)
            img = Image.open(image_path)
            img_array = np.array(img)

            ax.imshow(img_array, cmap='gray', vmin=0, vmax=255)

            # Triangulos GT (verde)
            for tri in triangles:
                pts = gt_landmarks[tri]
                polygon = Polygon(pts, fill=False, edgecolor='lime', linewidth=1, alpha=0.8)
                ax.add_patch(polygon)

            # Triangulos Pred (rojo, punteado)
            for tri in triangles:
                pts = pred_landmarks[tri]
                polygon = Polygon(pts, fill=False, edgecolor='red', linewidth=1,
                                 alpha=0.8, linestyle='--')
                ax.add_patch(polygon)

            ax.set_title(f'{error:.1f} px', fontsize=9)
            ax.axis('off')

        # Etiqueta de categoria
        axes[cat_idx, 0].set_ylabel(category, fontsize=11, fontweight='bold', rotation=0,
                                     ha='right', va='center', labelpad=50)

    plt.suptitle('Resumen de Triangulacion: GT (verde) vs Prediccion (rojo)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Grilla resumen guardada: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Generar visualizaciones de Delaunay para todas las imagenes')
    parser.add_argument('--predictions-dir', type=str, default='outputs/predictions',
                        help='Directorio con predicciones')
    parser.add_argument('--output-dir', type=str, default='outputs/predictions/all_visualizations',
                        help='Directorio de salida')
    parser.add_argument('--mode', type=str, choices=['comparison', 'side_by_side', 'separate'],
                        default='side_by_side', help='Modo de visualizacion')
    parser.add_argument('--summary-only', action='store_true',
                        help='Solo generar grilla resumen')
    return parser.parse_args()


def main():
    args = parse_args()

    predictions_dir = PROJECT_ROOT / args.predictions_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = PROJECT_ROOT / 'data'

    # Cargar datos
    print("=== Cargando predicciones y triangulacion ===")
    data, triangles = load_predictions_and_triangles(predictions_dir)
    print(f"  Imagenes: {len(data['image_names'])}")
    print(f"  Triangulos: {len(triangles)}")

    # Generar grilla resumen
    print("\n=== Generando grilla resumen ===")
    generate_summary_grid(data, triangles, data_root, output_dir / 'summary_grid.png', n_samples=12)

    if args.summary_only:
        print("\nModo --summary-only: Solo se genero la grilla resumen")
        return

    # Generar todas las visualizaciones
    generate_all_visualizations(data, triangles, data_root, output_dir, mode=args.mode)

    # Contar archivos generados
    total_files = sum(1 for _ in output_dir.rglob('*.png'))

    print("\n" + "=" * 60)
    print("VISUALIZACIONES COMPLETADAS")
    print("=" * 60)
    print(f"\nArchivos generados: {total_files}")
    print(f"Directorio: {output_dir}")

    # Mostrar estructura
    print("\nEstructura de salida:")
    for cat in ['normal', 'covid', 'viral_pneumonia']:
        cat_dir = output_dir / cat
        if cat_dir.exists():
            n_files = len(list(cat_dir.glob('*.png')))
            print(f"  {cat}/: {n_files} archivos")


if __name__ == '__main__':
    main()
