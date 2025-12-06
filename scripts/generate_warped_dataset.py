#!/usr/bin/env python3
"""
Sesion 21: Generacion de Dataset Warpeado Completo (SOLO AREA PULMONAR)

Este script genera el dataset warpeado completo (957 imagenes) para
entrenamiento de clasificador CNN.

IMPORTANTE:
- Solo se warpea el AREA PULMONAR (18 triangulos de Delaunay)
- NO se incluyen puntos de borde (full_coverage=False)
- El fondo queda negro, solo los pulmones tienen contenido
- Para train/val/test: usamos landmarks GROUND TRUTH

Estructura de salida:
outputs/warped_dataset/
├── train/
│   ├── Normal/
│   ├── COVID/
│   └── Viral_Pneumonia/
├── val/
│   ├── Normal/
│   ├── COVID/
│   └── Viral_Pneumonia/
└── test/
    ├── Normal/
    ├── COVID/
    └── Viral_Pneumonia/

Autor: Proyecto Tesis Maestria
Fecha: 28-Nov-2024
Sesion: 21
"""

import numpy as np
import cv2
import json
from pathlib import Path
from typing import Dict, Tuple, List
from collections import defaultdict
import sys

# Configuracion de paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "warped_dataset"
SHAPE_ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "shape_analysis"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
DATA_DIR = PROJECT_ROOT / "data" / "dataset"

# Importar funciones de warping de sesion 20
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from piecewise_affine_warp import (
    load_canonical_shape,
    load_delaunay_triangles,
    piecewise_affine_warp
)


def load_all_landmarks() -> Dict:
    """
    Cargar landmarks GT de todo el dataset.

    Returns:
        Dict con landmarks y metadatos para train/val/test
    """
    npz_path = PREDICTIONS_DIR / "all_landmarks.npz"
    data = np.load(npz_path, allow_pickle=True)

    return {
        'train': {
            'landmarks': data['train_landmarks'],
            'image_names': data['train_image_names'],
            'categories': data['train_categories']
        },
        'val': {
            'landmarks': data['val_landmarks'],
            'image_names': data['val_image_names'],
            'categories': data['val_categories']
        },
        'test': {
            'landmarks': data['test_landmarks'],
            'image_names': data['test_image_names'],
            'categories': data['test_categories']
        }
    }


def get_image_path(image_name: str, category: str) -> Path:
    """Construir path a imagen original."""
    return DATA_DIR / category / f"{image_name}.png"


def process_split(split_name: str,
                  landmarks: np.ndarray,
                  image_names: np.ndarray,
                  categories: np.ndarray,
                  canonical_shape: np.ndarray,
                  triangles: np.ndarray,
                  output_dir: Path) -> Dict:
    """
    Procesar un split completo (train/val/test).

    Args:
        split_name: Nombre del split ('train', 'val', 'test')
        landmarks: Array (N, 15, 2) con landmarks GT
        image_names: Array de nombres de imagen
        categories: Array de categorias
        canonical_shape: Forma canonica (15, 2)
        triangles: Triangulos de Delaunay (18, 3) - solo area pulmonar
        output_dir: Directorio base de salida

    Returns:
        Dict con estadisticas del procesamiento
    """
    split_dir = output_dir / split_name

    # Crear directorios por categoria
    for cat in ['Normal', 'COVID', 'Viral_Pneumonia']:
        (split_dir / cat).mkdir(parents=True, exist_ok=True)

    n_images = len(landmarks)
    stats = {
        'total': n_images,
        'processed': 0,
        'failed': 0,
        'fill_rates': [],
        'by_category': defaultdict(lambda: {'count': 0, 'fill_rates': []})
    }

    print(f"\n  Procesando {split_name}: {n_images} imagenes...")

    for i in range(n_images):
        lm = landmarks[i]
        name = str(image_names[i])
        cat = str(categories[i])

        # Cargar imagen original
        img_path = get_image_path(name, cat)

        if not img_path.exists():
            print(f"    [!] Imagen no encontrada: {img_path}")
            stats['failed'] += 1
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"    [!] Error leyendo: {img_path}")
            stats['failed'] += 1
            continue

        # Resize imagen a 224x224
        if image.shape[0] != 224 or image.shape[1] != 224:
            image = cv2.resize(image, (224, 224))

        # NOTA: Los landmarks en all_landmarks.npz YA estan en escala 224
        # (fueron normalizados durante la extraccion)
        # NO aplicar escalado adicional

        # Aplicar warping SOLO AREA PULMONAR (18 triangulos Delaunay)
        try:
            warped = piecewise_affine_warp(
                image=image,
                source_landmarks=lm,  # Ya en escala 224
                target_landmarks=canonical_shape,
                triangles=triangles,
                use_full_coverage=False  # Solo area pulmonar
            )
        except Exception as e:
            print(f"    [!] Error warping {name}: {e}")
            stats['failed'] += 1
            continue

        # Calcular fill rate
        black_pixels = np.sum(warped == 0)
        fill_rate = 1 - (black_pixels / warped.size)

        # Guardar imagen warpeada
        output_path = split_dir / cat / f"{name}_warped.png"
        cv2.imwrite(str(output_path), warped)

        # Actualizar estadisticas
        stats['processed'] += 1
        stats['fill_rates'].append(fill_rate)
        stats['by_category'][cat]['count'] += 1
        stats['by_category'][cat]['fill_rates'].append(fill_rate)

        # Progreso cada 100 imagenes
        if (i + 1) % 100 == 0:
            print(f"    Procesadas {i+1}/{n_images} ({(i+1)*100/n_images:.1f}%)")

    print(f"  Completado: {stats['processed']}/{n_images} procesadas, {stats['failed']} fallidas")

    return stats


def generate_summary_statistics(all_stats: Dict, output_dir: Path) -> None:
    """
    Generar resumen estadistico del dataset warpeado.

    Args:
        all_stats: Estadisticas de todos los splits
        output_dir: Directorio de salida
    """
    summary = {
        'total_images': 0,
        'total_processed': 0,
        'total_failed': 0,
        'splits': {},
        'overall_fill_rate': {
            'mean': 0,
            'std': 0,
            'min': 0,
            'max': 0
        },
        'by_category': {}
    }

    all_fill_rates = []
    category_fill_rates = defaultdict(list)

    for split_name, stats in all_stats.items():
        summary['total_images'] += stats['total']
        summary['total_processed'] += stats['processed']
        summary['total_failed'] += stats['failed']

        fill_rates = np.array(stats['fill_rates'])
        summary['splits'][split_name] = {
            'total': stats['total'],
            'processed': stats['processed'],
            'failed': stats['failed'],
            'fill_rate_mean': float(fill_rates.mean()) if len(fill_rates) > 0 else 0,
            'fill_rate_std': float(fill_rates.std()) if len(fill_rates) > 0 else 0,
            'by_category': {}
        }

        for cat, cat_stats in stats['by_category'].items():
            cat_rates = np.array(cat_stats['fill_rates'])
            summary['splits'][split_name]['by_category'][cat] = {
                'count': cat_stats['count'],
                'fill_rate_mean': float(cat_rates.mean()) if len(cat_rates) > 0 else 0
            }
            category_fill_rates[cat].extend(cat_stats['fill_rates'])

        all_fill_rates.extend(stats['fill_rates'])

    # Estadisticas globales
    all_rates = np.array(all_fill_rates)
    summary['overall_fill_rate'] = {
        'mean': float(all_rates.mean()),
        'std': float(all_rates.std()),
        'min': float(all_rates.min()),
        'max': float(all_rates.max())
    }

    # Por categoria global
    for cat, rates in category_fill_rates.items():
        cat_rates = np.array(rates)
        summary['by_category'][cat] = {
            'total_count': len(rates),
            'fill_rate_mean': float(cat_rates.mean()),
            'fill_rate_std': float(cat_rates.std())
        }

    # Guardar JSON
    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResumen guardado: {summary_path}")

    # Imprimir resumen
    print("\n" + "=" * 60)
    print("RESUMEN DEL DATASET WARPEADO")
    print("=" * 60)
    print(f"\nTotal de imagenes: {summary['total_processed']}/{summary['total_images']}")
    print(f"Fill rate global: {summary['overall_fill_rate']['mean']*100:.1f}% ± {summary['overall_fill_rate']['std']*100:.1f}%")

    print("\nPor split:")
    for split_name, split_stats in summary['splits'].items():
        print(f"  {split_name}: {split_stats['processed']} imagenes, "
              f"fill rate: {split_stats['fill_rate_mean']*100:.1f}%")
        for cat, cat_stats in split_stats['by_category'].items():
            print(f"    - {cat}: {cat_stats['count']} imagenes")

    print("\nPor categoria (global):")
    for cat, cat_stats in summary['by_category'].items():
        print(f"  {cat}: {cat_stats['total_count']} imagenes, "
              f"fill rate: {cat_stats['fill_rate_mean']*100:.1f}%")


def create_split_info_files(data: Dict, output_dir: Path) -> None:
    """
    Crear archivos con informacion de cada split para facilitar carga.

    Args:
        data: Diccionario con datos de todos los splits
        output_dir: Directorio de salida
    """
    for split_name, split_data in data.items():
        split_dir = output_dir / split_name

        # Crear CSV con nombres y categorias
        csv_path = split_dir / "images.csv"
        with open(csv_path, 'w') as f:
            f.write("image_name,category,warped_filename\n")
            for name, cat in zip(split_data['image_names'], split_data['categories']):
                f.write(f"{name},{cat},{name}_warped.png\n")

        print(f"  Creado: {csv_path}")


def main():
    """
    Generar dataset warpeado completo (SOLO AREA PULMONAR).
    """
    print("=" * 60)
    print("SESION 21: Generacion de Dataset Warpeado (SOLO AREA PULMONAR)")
    print("=" * 60)

    # Crear directorio de salida
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cargar datos
    print("\n1. Cargando datos...")
    canonical_shape, image_size = load_canonical_shape()
    triangles = load_delaunay_triangles()  # 18 triangulos de area pulmonar
    data = load_all_landmarks()

    print(f"   Forma canonica: {canonical_shape.shape}")
    print(f"   Triangulos Delaunay: {triangles.shape} (solo area pulmonar)")
    print(f"   Tamaño de imagen: {image_size}x{image_size}")
    print(f"   Train: {len(data['train']['landmarks'])} imagenes")
    print(f"   Val: {len(data['val']['landmarks'])} imagenes")
    print(f"   Test: {len(data['test']['landmarks'])} imagenes")

    # 2. Procesar cada split
    print("\n2. Procesando splits (solo area pulmonar)...")
    all_stats = {}

    for split_name in ['train', 'val', 'test']:
        split_data = data[split_name]
        stats = process_split(
            split_name=split_name,
            landmarks=split_data['landmarks'],
            image_names=split_data['image_names'],
            categories=split_data['categories'],
            canonical_shape=canonical_shape,
            triangles=triangles,
            output_dir=OUTPUT_DIR
        )
        all_stats[split_name] = stats

    # 3. Crear archivos de informacion de splits
    print("\n3. Creando archivos de informacion...")
    create_split_info_files(data, OUTPUT_DIR)

    # 4. Generar resumen estadistico
    print("\n4. Generando resumen estadistico...")
    generate_summary_statistics(all_stats, OUTPUT_DIR)

    # 5. Guardar configuracion
    config = {
        'source': 'Ground Truth landmarks',
        'canonical_shape_file': str(SHAPE_ANALYSIS_DIR / "canonical_shape_gpa.json"),
        'triangles_file': str(SHAPE_ANALYSIS_DIR / "canonical_delaunay_triangles.json"),
        'warping_method': 'piecewise_affine',
        'full_coverage': False,  # Solo area pulmonar
        'num_triangles': 18,
        'coverage_type': 'lung_only',
        'expected_fill_rate': '~47%',
        'image_size': 224,
        'session': 21,
        'date': '2024-11-28',
        'splits': {
            'train': len(data['train']['landmarks']),
            'val': len(data['val']['landmarks']),
            'test': len(data['test']['landmarks'])
        }
    }

    config_path = OUTPUT_DIR / "warping_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfiguracion guardada: {config_path}")

    # Resumen final
    print("\n" + "=" * 60)
    print("COMPLETADO")
    print("=" * 60)
    print(f"\nDataset warpeado generado en: {OUTPUT_DIR}")
    print("\nEstructura:")
    print("  warped_dataset/")
    print("  ├── train/  (717 imagenes)")
    print("  │   ├── Normal/")
    print("  │   ├── COVID/")
    print("  │   └── Viral_Pneumonia/")
    print("  ├── val/    (144 imagenes)")
    print("  │   ├── Normal/")
    print("  │   ├── COVID/")
    print("  │   └── Viral_Pneumonia/")
    print("  └── test/   (96 imagenes)")
    print("      ├── Normal/")
    print("      ├── COVID/")
    print("      └── Viral_Pneumonia/")

    return all_stats


if __name__ == "__main__":
    main()
