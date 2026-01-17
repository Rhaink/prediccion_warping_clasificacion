#!/usr/bin/env python3
"""
Script para generar dataset de imágenes warped con SAHS aplicado.

SAHS se aplica SOLO a la región pulmonar (píxeles > threshold),
manteniendo el fondo negro intacto.

Uso:
    python scripts/generate_warped_sahs_dataset.py \
        --input-dir outputs/warped_lung_best/session_warping \
        --output-dir outputs/warped_lung_sahs \
        --threshold 10
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import cv2


def enhance_contrast_sahs_masked(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Aplica el algoritmo SAHS solo a la región pulmonar (píxeles > threshold).

    El fondo negro (píxeles <= threshold) se mantiene intacto.

    Statistical Asymmetrical Histogram Stretching (SAHS):
    - Calcula límites de estiramiento asimétricos basados en la media
    - Factor 2.5 para el límite superior
    - Factor 2.0 para el límite inferior

    Args:
        image: Imagen en escala de grises
        threshold: Umbral para considerar píxeles como fondo (default: 10)

    Returns:
        Imagen con contraste mejorado
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


def process_single_image(args):
    """Procesa una sola imagen (para paralelización)."""
    input_path, output_path, threshold = args
    try:
        # Cargar imagen
        img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return input_path, False, "No se pudo cargar"

        # Aplicar SAHS
        img_sahs = enhance_contrast_sahs_masked(img, threshold=threshold)

        # Crear directorio de salida si no existe
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Guardar imagen
        cv2.imwrite(str(output_path), img_sahs)
        return input_path, True, None
    except Exception as e:
        return input_path, False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Genera dataset de imágenes warped con SAHS aplicado a la región pulmonar"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/warped_lung_best/session_warping"),
        help="Directorio con el dataset warped original"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/warped_lung_sahs"),
        help="Directorio de salida para el dataset con SAHS"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Umbral para considerar píxeles como fondo (default: 10)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Número de workers para procesamiento paralelo"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Generación de Dataset Warped + SAHS")
    print("=" * 60)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Workers: {args.workers}")
    print()

    # Verificar que exista el directorio de entrada
    if not args.input_dir.exists():
        print(f"Error: No existe el directorio {args.input_dir}")
        return

    # Crear directorio de salida
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Recopilar todas las imágenes a procesar
    splits = ["train", "val", "test"]
    categories = ["COVID", "Normal", "Viral_Pneumonia"]

    tasks = []
    stats = {split: {cat: 0 for cat in categories} for split in splits}

    for split in splits:
        for category in categories:
            input_cat_dir = args.input_dir / split / category
            output_cat_dir = args.output_dir / split / category

            if not input_cat_dir.exists():
                print(f"Advertencia: No existe {input_cat_dir}")
                continue

            for img_path in input_cat_dir.glob("*.png"):
                output_path = output_cat_dir / img_path.name
                tasks.append((img_path, output_path, args.threshold))
                stats[split][category] += 1

    total_images = len(tasks)
    print(f"Total de imágenes a procesar: {total_images}")
    print()

    # Mostrar distribución
    print("Distribución del dataset:")
    for split in splits:
        total_split = sum(stats[split].values())
        print(f"  {split}: {total_split}")
        for cat in categories:
            print(f"    - {cat}: {stats[split][cat]}")
    print()

    # Procesar imágenes en paralelo
    print("Procesando imágenes...")
    successful = 0
    failed = 0
    errors = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_image, task): task for task in tasks}

        with tqdm(total=total_images, desc="Aplicando SAHS") as pbar:
            for future in as_completed(futures):
                input_path, success, error = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
                    errors.append((input_path, error))
                pbar.update(1)

    print()
    print("=" * 60)
    print("Resultados")
    print("=" * 60)
    print(f"Imágenes procesadas exitosamente: {successful}")
    print(f"Imágenes con errores: {failed}")

    if errors:
        print("\nErrores:")
        for path, error in errors[:10]:  # Mostrar solo los primeros 10
            print(f"  {path}: {error}")
        if len(errors) > 10:
            print(f"  ... y {len(errors) - 10} más")

    # Guardar metadatos
    metadata = {
        "created_at": datetime.now().isoformat(),
        "source_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "threshold": args.threshold,
        "method": "SAHS (Statistical Asymmetrical Histogram Stretching)",
        "description": "SAHS aplicado solo a la región pulmonar (píxeles > threshold)",
        "sahs_params": {
            "upper_factor": 2.5,
            "lower_factor": 2.0
        },
        "stats": {
            "total_images": total_images,
            "successful": successful,
            "failed": failed,
            "distribution": stats
        }
    }

    metadata_path = args.output_dir / "dataset_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadatos guardados en: {metadata_path}")

    # Copiar dataset_summary.json si existe
    summary_src = args.input_dir / "dataset_summary.json"
    if summary_src.exists():
        shutil.copy(summary_src, args.output_dir / "dataset_summary_original.json")

    print(f"\nDataset generado en: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
