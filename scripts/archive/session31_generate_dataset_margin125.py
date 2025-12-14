#!/usr/bin/env python3
"""
Sesión 31: Generación de Dataset Warped con Margin 1.25 (Óptimo)

Este script genera el dataset warped de 15K imágenes con margin_scale=1.25,
el valor óptimo encontrado en la sesión 28.

Autor: Proyecto Tesis Maestría
Fecha: 29-Nov-2024
Sesión: 31
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2
import json
import torch
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import time

from scripts.predict import EnsemblePredictor
from scripts.piecewise_affine_warp import (
    load_canonical_shape,
    load_delaunay_triangles,
    piecewise_affine_warp
)


# Configuración - MARGIN 1.25 (ÓPTIMO)
MARGIN_SCALE = 1.25  # Óptimo encontrado en sesión 28
IMAGE_SIZE = 224
BATCH_LOG_INTERVAL = 100

# Paths
FULL_DATASET_DIR = PROJECT_ROOT / "data" / "dataset" / "COVID-19_Radiography_Dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "session31_multi_arch" / "datasets" / "full_warped_margin125"

# Clases a procesar (excluimos Lung_Opacity para mantener 3 clases)
CLASSES = ['COVID', 'Normal', 'Viral Pneumonia']
CLASS_MAPPING = {
    'COVID': 'COVID',
    'Normal': 'Normal',
    'Viral Pneumonia': 'Viral_Pneumonia'
}


def scale_landmarks_from_centroid(landmarks: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Escalar landmarks desde su centroide."""
    centroid = landmarks.mean(axis=0)
    scaled = centroid + (landmarks - centroid) * scale
    return scaled


def clip_landmarks_to_image(landmarks: np.ndarray, image_size: int = 224, margin: int = 2) -> np.ndarray:
    """Asegurar que landmarks estén dentro de la imagen."""
    clipped = np.clip(landmarks, margin, image_size - margin - 1)
    return clipped


def create_splits(image_paths: list, train_ratio=0.75, val_ratio=0.15, seed=42):
    """
    Crear splits train/val/test estratificados por clase.
    IMPORTANTE: Usa seed=42 para reproducibilidad idéntica al margin 1.05.
    """
    np.random.seed(seed)

    by_class = defaultdict(list)
    for path, class_name in image_paths:
        by_class[class_name].append(path)

    splits = {'train': [], 'val': [], 'test': []}

    for class_name, paths in by_class.items():
        np.random.shuffle(paths)
        n = len(paths)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits['train'].extend([(p, class_name) for p in paths[:n_train]])
        splits['val'].extend([(p, class_name) for p in paths[n_train:n_train+n_val]])
        splits['test'].extend([(p, class_name) for p in paths[n_train+n_val:]])

    for split in splits.values():
        np.random.shuffle(split)

    return splits


def process_and_save_image(predictor, image_path, output_path,
                           canonical_shape, triangles, margin_scale):
    """
    Procesar una imagen: predecir landmarks, aplicar warping, guardar.
    """
    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None

        if image.shape[0] != IMAGE_SIZE or image.shape[1] != IMAGE_SIZE:
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        landmarks_norm = predictor.predict(image_path, return_normalized=True)
        landmarks = landmarks_norm * IMAGE_SIZE

        scaled_landmarks = scale_landmarks_from_centroid(landmarks, margin_scale)
        scaled_landmarks = clip_landmarks_to_image(scaled_landmarks)

        warped = piecewise_affine_warp(
            image=image,
            source_landmarks=scaled_landmarks,
            target_landmarks=canonical_shape,
            triangles=triangles,
            use_full_coverage=False
        )

        black_pixels = np.sum(warped == 0)
        fill_rate = 1 - (black_pixels / warped.size)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), warped)

        return {
            'fill_rate': fill_rate,
            'landmarks': landmarks.tolist()
        }

    except Exception as e:
        print(f"Error procesando {image_path}: {e}")
        return None


def main():
    print("="*70)
    print("SESIÓN 31: GENERACIÓN DE DATASET WARPED (MARGIN 1.25 ÓPTIMO)")
    print("="*70)
    print(f"\nMargin scale: {MARGIN_SCALE}")
    print(f"Output: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Recolectar todas las imágenes
    print("\n1. Recolectando imágenes del dataset completo...")
    all_images = []

    for class_name in CLASSES:
        class_dir = FULL_DATASET_DIR / class_name / "images"
        if class_dir.exists():
            images = list(class_dir.glob("*.png"))
            mapped_class = CLASS_MAPPING[class_name]
            all_images.extend([(img, mapped_class) for img in images])
            print(f"   {class_name}: {len(images)} imágenes")

    print(f"\n   Total: {len(all_images)} imágenes")

    # 2. Crear splits (seed=42 para reproducibilidad)
    print("\n2. Creando splits train/val/test (seed=42)...")
    splits = create_splits(all_images, train_ratio=0.75, val_ratio=0.15, seed=42)

    for split_name, split_images in splits.items():
        by_class = defaultdict(int)
        for _, class_name in split_images:
            by_class[class_name] += 1
        print(f"   {split_name}: {len(split_images)} imágenes")
        for class_name, count in sorted(by_class.items()):
            print(f"      - {class_name}: {count}")

    # 3. Cargar modelo y configuración
    print("\n3. Cargando modelo de landmarks y configuración...")
    predictor = EnsemblePredictor(use_clahe=True)
    canonical_shape, _ = load_canonical_shape()
    triangles = load_delaunay_triangles()

    print(f"   Forma canónica: {canonical_shape.shape}")
    print(f"   Triángulos: {triangles.shape}")

    # 4. Procesar imágenes por split
    print("\n4. Procesando imágenes...")

    all_stats = {}
    all_landmarks = {}
    start_time = time.time()

    for split_name, split_images in splits.items():
        print(f"\n   === {split_name.upper()} ===")

        stats = {
            'processed': 0,
            'failed': 0,
            'fill_rates': [],
            'by_class': defaultdict(lambda: {'count': 0, 'fill_rates': []})
        }
        landmarks_data = []

        pbar = tqdm(split_images, desc=f"   {split_name}", ncols=80)

        for image_path, class_name in pbar:
            output_filename = f"{image_path.stem}_warped.png"
            output_path = OUTPUT_DIR / split_name / class_name / output_filename

            result = process_and_save_image(
                predictor, image_path, output_path,
                canonical_shape, triangles, MARGIN_SCALE
            )

            if result:
                stats['processed'] += 1
                stats['fill_rates'].append(result['fill_rate'])
                stats['by_class'][class_name]['count'] += 1
                stats['by_class'][class_name]['fill_rates'].append(result['fill_rate'])

                landmarks_data.append({
                    'image_name': image_path.stem,
                    'class': class_name,
                    'landmarks': result['landmarks']
                })
            else:
                stats['failed'] += 1

            if stats['processed'] % BATCH_LOG_INTERVAL == 0:
                avg_fill = np.mean(stats['fill_rates'][-BATCH_LOG_INTERVAL:]) if stats['fill_rates'] else 0
                pbar.set_postfix({'fill_rate': f'{avg_fill:.1%}'})

        all_stats[split_name] = stats
        all_landmarks[split_name] = landmarks_data

        fill_rates = np.array(stats['fill_rates'])
        print(f"   Procesadas: {stats['processed']}/{len(split_images)}")
        print(f"   Fill rate: {fill_rates.mean():.1%} ± {fill_rates.std():.1%}")

    elapsed = time.time() - start_time
    print(f"\n   Tiempo total: {elapsed/60:.1f} minutos")

    # 5. Guardar metadatos
    print("\n5. Guardando metadatos...")

    summary = {
        'margin_scale': MARGIN_SCALE,
        'source_dataset': str(FULL_DATASET_DIR),
        'classes': list(CLASS_MAPPING.values()),
        'splits': {},
        'processing_time_minutes': elapsed / 60,
        'seed': 42
    }

    for split_name, stats in all_stats.items():
        fill_rates = np.array(stats['fill_rates'])
        summary['splits'][split_name] = {
            'total': len(splits[split_name]),
            'processed': stats['processed'],
            'failed': stats['failed'],
            'fill_rate_mean': float(fill_rates.mean()) if len(fill_rates) > 0 else 0,
            'fill_rate_std': float(fill_rates.std()) if len(fill_rates) > 0 else 0,
            'by_class': {
                class_name: {
                    'count': class_stats['count'],
                    'fill_rate_mean': float(np.mean(class_stats['fill_rates'])) if class_stats['fill_rates'] else 0
                }
                for class_name, class_stats in stats['by_class'].items()
            }
        }

    with open(OUTPUT_DIR / "dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    for split_name, landmarks_data in all_landmarks.items():
        landmarks_path = OUTPUT_DIR / split_name / "landmarks.json"
        with open(landmarks_path, 'w') as f:
            json.dump(landmarks_data, f)

    for split_name in splits.keys():
        split_dir = OUTPUT_DIR / split_name
        csv_path = split_dir / "images.csv"

        with open(csv_path, 'w') as f:
            f.write("image_name,category,warped_filename\n")
            for image_path, class_name in splits[split_name]:
                warped_name = f"{image_path.stem}_warped.png"
                f.write(f"{image_path.stem},{class_name},{warped_name}\n")

    # 6. Resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)

    total_processed = sum(s['processed'] for s in all_stats.values())
    total_failed = sum(s['failed'] for s in all_stats.values())

    print(f"\nTotal procesadas: {total_processed}")
    print(f"Total fallidas: {total_failed}")
    print(f"Margin scale usado: {MARGIN_SCALE}")
    print(f"Tiempo: {elapsed/60:.1f} minutos ({elapsed/total_processed:.2f}s por imagen)")

    print(f"\nDataset generado en: {OUTPUT_DIR}")
    print("\nEstructura:")
    print("  full_warped_margin125/")
    for split_name, stats in all_stats.items():
        print(f"  ├── {split_name}/ ({stats['processed']} imágenes)")
        for class_name in sorted(stats['by_class'].keys()):
            count = stats['by_class'][class_name]['count']
            print(f"  │   ├── {class_name}/ ({count})")

    print("\n¡Dataset margin 1.25 listo para entrenamiento!")

    return summary


if __name__ == "__main__":
    main()
