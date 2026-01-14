#!/usr/bin/env python3
"""
Sesion 37: Warpear Dataset3 (FedCOVIDx) para evaluacion justa.

Este script aplica warping geometrico a las imagenes de Dataset3 usando
el ensemble de landmarks para permitir una comparacion justa:
- Modelos ORIGINAL -> evaluados en Dataset3 original
- Modelos WARPED -> evaluados en Dataset3 warpeado

Pipeline:
1. Cargar imagen de Dataset3 (ya preprocesada a 299x299 grayscale)
2. Predecir landmarks con ensemble
3. Aplicar piecewise affine warping a forma canonica
4. Guardar imagen warpeada

Autor: Proyecto Tesis Maestria
Fecha: 03-Dic-2024
Sesion: 37
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
import argparse

from scripts.predict import EnsemblePredictor
from scripts.piecewise_affine_warp import (
    load_canonical_shape,
    load_delaunay_triangles,
    piecewise_affine_warp
)


# Configuracion
MARGIN_SCALE = 1.05  # Optimo encontrado en experimentos anteriores
IMAGE_SIZE = 224
BATCH_LOG_INTERVAL = 100


def scale_landmarks_from_centroid(landmarks: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Escalar landmarks desde su centroide."""
    centroid = landmarks.mean(axis=0)
    scaled = centroid + (landmarks - centroid) * scale
    return scaled


def clip_landmarks_to_image(landmarks: np.ndarray, image_size: int = 224, margin: int = 2) -> np.ndarray:
    """Asegurar que landmarks esten dentro de la imagen."""
    clipped = np.clip(landmarks, margin, image_size - margin - 1)
    return clipped


def process_and_save_image(predictor, image_path, output_path,
                           canonical_shape, triangles, margin_scale):
    """
    Procesar una imagen: predecir landmarks, aplicar warping, guardar.

    Returns:
        Dict con estadisticas o None si falla
    """
    try:
        # Cargar imagen
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None

        # Resize a 224x224
        if image.shape[0] != IMAGE_SIZE or image.shape[1] != IMAGE_SIZE:
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        # Predecir landmarks (en escala 224)
        landmarks_norm = predictor.predict(image_path, return_normalized=True)
        landmarks = landmarks_norm * IMAGE_SIZE

        # Aplicar margin_scale
        scaled_landmarks = scale_landmarks_from_centroid(landmarks, margin_scale)
        scaled_landmarks = clip_landmarks_to_image(scaled_landmarks)

        # Aplicar warping
        warped = piecewise_affine_warp(
            image=image,
            source_landmarks=scaled_landmarks,
            target_landmarks=canonical_shape,
            triangles=triangles,
            use_full_coverage=False  # Solo area pulmonar
        )

        # Calcular fill rate
        black_pixels = np.sum(warped == 0)
        fill_rate = 1 - (black_pixels / warped.size)

        # Guardar
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
    parser = argparse.ArgumentParser(description='Warpear Dataset3 para evaluacion externa')
    parser.add_argument('--input-dir', type=str,
                       default='outputs/external_validation/dataset3',
                       help='Directorio con Dataset3 procesado')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/external_validation/dataset3_warped',
                       help='Directorio de salida para imagenes warpeadas')
    parser.add_argument('--split', type=str, default='test',
                       help='Split a procesar (default: test)')
    parser.add_argument('--margin-scale', type=float, default=MARGIN_SCALE,
                       help=f'Margin scale para warping (default: {MARGIN_SCALE})')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limitar numero de imagenes (para pruebas)')
    args = parser.parse_args()

    print("=" * 70)
    print("SESION 37: WARPING DE DATASET3 (FedCOVIDx)")
    print("=" * 70)

    input_dir = PROJECT_ROOT / args.input_dir / args.split
    output_dir = PROJECT_ROOT / args.output_dir / args.split

    print(f"\nEntrada: {input_dir}")
    print(f"Salida: {output_dir}")
    print(f"Margin scale: {args.margin_scale}")

    # Verificar que existe Dataset3
    if not input_dir.exists():
        print(f"\nError: No existe {input_dir}")
        print("Ejecuta primero: python scripts/archive/classification/prepare_dataset3.py")
        return

    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Recolectar imagenes
    print("\n1. Recolectando imagenes...")
    all_images = []
    for label in ['positive', 'negative']:
        label_dir = input_dir / label
        if label_dir.exists():
            images = list(label_dir.glob('*.png'))
            all_images.extend([(img, label) for img in images])
            print(f"   {label}: {len(images)} imagenes")

    if args.limit:
        print(f"\n   Limitando a {args.limit} imagenes (modo prueba)")
        all_images = all_images[:args.limit]

    print(f"\n   Total a procesar: {len(all_images)} imagenes")

    # 2. Cargar modelo y configuracion
    print("\n2. Cargando modelo de landmarks y configuracion...")
    predictor = EnsemblePredictor(use_clahe=True)
    canonical_shape, _ = load_canonical_shape()
    triangles = load_delaunay_triangles()

    print(f"   Forma canonica: {canonical_shape.shape}")
    print(f"   Triangulos: {triangles.shape}")

    # 3. Procesar imagenes
    print("\n3. Procesando imagenes...")

    stats = {
        'processed': 0,
        'failed': 0,
        'fill_rates': [],
        'by_class': defaultdict(lambda: {'count': 0, 'fill_rates': []})
    }
    landmarks_data = []
    start_time = time.time()

    pbar = tqdm(all_images, desc="   Warping", ncols=80)

    for image_path, label in pbar:
        # Definir path de salida
        output_filename = image_path.name  # Mantener nombre original
        output_path = output_dir / label / output_filename

        # Procesar
        result = process_and_save_image(
            predictor, image_path, output_path,
            canonical_shape, triangles, args.margin_scale
        )

        if result:
            stats['processed'] += 1
            stats['fill_rates'].append(result['fill_rate'])
            stats['by_class'][label]['count'] += 1
            stats['by_class'][label]['fill_rates'].append(result['fill_rate'])

            landmarks_data.append({
                'image_name': image_path.stem,
                'label': label,
                'landmarks': result['landmarks'],
                'fill_rate': result['fill_rate']
            })
        else:
            stats['failed'] += 1

        # Actualizar barra de progreso
        if stats['processed'] % BATCH_LOG_INTERVAL == 0 and stats['fill_rates']:
            avg_fill = np.mean(stats['fill_rates'][-BATCH_LOG_INTERVAL:])
            pbar.set_postfix({'fill': f'{avg_fill:.1%}'})

    elapsed = time.time() - start_time

    # 4. Guardar metadatos
    print("\n4. Guardando metadatos...")

    fill_rates = np.array(stats['fill_rates'])

    summary = {
        'margin_scale': args.margin_scale,
        'source_dataset': str(input_dir),
        'output_dataset': str(output_dir),
        'split': args.split,
        'total_images': len(all_images),
        'processed': stats['processed'],
        'failed': stats['failed'],
        'fill_rate_mean': float(fill_rates.mean()) if len(fill_rates) > 0 else 0,
        'fill_rate_std': float(fill_rates.std()) if len(fill_rates) > 0 else 0,
        'fill_rate_min': float(fill_rates.min()) if len(fill_rates) > 0 else 0,
        'fill_rate_max': float(fill_rates.max()) if len(fill_rates) > 0 else 0,
        'processing_time_minutes': elapsed / 60,
        'by_class': {
            label: {
                'count': class_stats['count'],
                'fill_rate_mean': float(np.mean(class_stats['fill_rates'])) if class_stats['fill_rates'] else 0,
                'fill_rate_std': float(np.std(class_stats['fill_rates'])) if class_stats['fill_rates'] else 0
            }
            for label, class_stats in stats['by_class'].items()
        }
    }

    with open(output_dir.parent / f'{args.split}_warping_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Guardar landmarks predichos
    landmarks_path = output_dir.parent / f'{args.split}_landmarks.json'
    with open(landmarks_path, 'w') as f:
        json.dump(landmarks_data, f)

    # Copiar metadata del dataset original
    orig_metadata = input_dir / 'metadata.csv'
    if orig_metadata.exists():
        import shutil
        shutil.copy(orig_metadata, output_dir / 'metadata.csv')

    # 5. Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)

    print(f"\nImagenes procesadas: {stats['processed']}/{len(all_images)}")
    print(f"Imagenes fallidas: {stats['failed']}")
    print(f"Tiempo: {elapsed/60:.1f} minutos ({elapsed/max(stats['processed'],1):.2f}s por imagen)")

    print(f"\nFill rate (area cubierta):")
    print(f"   Media: {fill_rates.mean()*100:.1f}%")
    print(f"   Std: {fill_rates.std()*100:.1f}%")
    print(f"   Min: {fill_rates.min()*100:.1f}%")
    print(f"   Max: {fill_rates.max()*100:.1f}%")

    print(f"\nPor clase:")
    for label, class_stats in stats['by_class'].items():
        class_fills = np.array(class_stats['fill_rates'])
        if len(class_fills) > 0:
            print(f"   {label}: {class_stats['count']} imagenes, fill={class_fills.mean()*100:.1f}%")

    print(f"\nDataset warpeado guardado en: {output_dir}")

    print("\n" + "=" * 70)
    print("SIGUIENTE PASO:")
    print("  Ejecutar: python scripts/archive/classification/evaluate_external_warped.py")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    main()
