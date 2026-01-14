#!/usr/bin/env python3
"""
Sesion 36: Preparacion de Dataset3 (FedCOVIDx) para validacion externa.

Este script:
1. Parsea los archivos de etiquetas (train.txt, val.txt, test.txt)
2. Redimensiona las imagenes a 299x299
3. Crea estructura de directorios compatible con nuestros DataLoaders
4. Genera estadisticas del dataset

Dataset3 (FedCOVIDx):
- Total: 84,818 imagenes
- Test: 8,482 (4,241 positive + 4,241 negative) - balanceado 50/50
- Clases: 2 (positive=COVID, negative=no-COVID)
- Fuentes: bimcv (95%), ricord, rsna
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

# Agregar directorio raiz al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_label_file(txt_path):
    """
    Parsea archivo de etiquetas de Dataset3.

    Formato: patient_id filename label source
    Ejemplo: 419639-003251 MIDRC-RICORD-1C-419639-003251-46647-0.png positive ricord

    Returns:
        List of dicts with keys: patient_id, filename, label, source
    """
    samples = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                samples.append({
                    'patient_id': parts[0],
                    'filename': parts[1],
                    'label': parts[2],  # positive or negative
                    'source': parts[3]  # bimcv, ricord, rsna
                })
            elif len(parts) == 3:
                # Algunos archivos pueden no tener source
                samples.append({
                    'patient_id': parts[0],
                    'filename': parts[1],
                    'label': parts[2],
                    'source': 'unknown'
                })
    return samples


def resize_and_save_image(src_path, dst_path, target_size=(299, 299)):
    """
    Redimensiona imagen a target_size y guarda en dst_path.
    Convierte a escala de grises si es necesario.
    """
    try:
        img = Image.open(src_path)

        # Convertir a escala de grises si no lo es
        if img.mode != 'L':
            img = img.convert('L')

        # Redimensionar con LANCZOS para mejor calidad
        img_resized = img.resize(target_size, Image.LANCZOS)

        # Guardar como PNG
        img_resized.save(dst_path, 'PNG')
        return True
    except Exception as e:
        print(f"Error procesando {src_path}: {e}")
        return False


def prepare_split(samples, src_dir, dst_dir, split_name, target_size=(299, 299)):
    """
    Procesa un split (train/val/test) de Dataset3.

    Args:
        samples: Lista de dicts con info de las imagenes
        src_dir: Directorio fuente con imagenes originales
        dst_dir: Directorio destino para imagenes procesadas
        split_name: Nombre del split (train, val, test)
        target_size: Tamano objetivo de las imagenes

    Returns:
        dict con estadisticas del procesamiento
    """
    split_dst = dst_dir / split_name

    # Crear subdirectorios por clase
    (split_dst / 'positive').mkdir(parents=True, exist_ok=True)
    (split_dst / 'negative').mkdir(parents=True, exist_ok=True)

    stats = {
        'total': len(samples),
        'processed': 0,
        'failed': 0,
        'by_label': Counter(),
        'by_source': Counter(),
        'missing_files': []
    }

    # Crear CSV con metadata
    metadata = []

    for sample in tqdm(samples, desc=f"Procesando {split_name}"):
        filename = sample['filename']
        label = sample['label']
        source = sample['source']

        # Ruta fuente
        src_path = src_dir / filename

        if not src_path.exists():
            stats['failed'] += 1
            stats['missing_files'].append(filename)
            continue

        # Ruta destino
        # Usar nombre base para evitar colisiones
        dst_filename = filename.replace('/', '_').replace('\\', '_')
        if not dst_filename.endswith('.png'):
            dst_filename = dst_filename.rsplit('.', 1)[0] + '.png'

        dst_path = split_dst / label / dst_filename

        # Procesar imagen
        if resize_and_save_image(src_path, dst_path, target_size):
            stats['processed'] += 1
            stats['by_label'][label] += 1
            stats['by_source'][source] += 1

            metadata.append({
                'original_filename': filename,
                'processed_filename': dst_filename,
                'label': label,
                'source': source,
                'patient_id': sample['patient_id']
            })
        else:
            stats['failed'] += 1

    # Guardar metadata como CSV
    df = pd.DataFrame(metadata)
    df.to_csv(split_dst / 'metadata.csv', index=False)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Preparar Dataset3 (FedCOVIDx) para validacion externa'
    )
    parser.add_argument(
        '--source-dir',
        type=str,
        default='data/dataset/dataset3',
        help='Directorio fuente de Dataset3'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/external_validation/dataset3',
        help='Directorio de salida para imagenes procesadas'
    )
    parser.add_argument(
        '--target-size',
        type=int,
        default=299,
        help='Tamano objetivo de las imagenes (cuadradas)'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['test'],
        choices=['train', 'val', 'test'],
        help='Splits a procesar (default: solo test)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Solo mostrar estadisticas sin procesar imagenes'
    )

    args = parser.parse_args()

    # Rutas
    source_dir = PROJECT_ROOT / args.source_dir
    output_dir = PROJECT_ROOT / args.output_dir
    target_size = (args.target_size, args.target_size)

    print("=" * 60)
    print("PREPARACION DE DATASET3 (FedCOVIDx)")
    print("=" * 60)
    print(f"\nFuente: {source_dir}")
    print(f"Destino: {output_dir}")
    print(f"Tamano objetivo: {target_size}")
    print(f"Splits a procesar: {args.splits}")

    # Verificar que existe el directorio fuente
    if not source_dir.exists():
        print(f"\nError: No se encuentra el directorio fuente {source_dir}")
        sys.exit(1)

    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)

    # Estadisticas globales
    global_stats = {
        'source_dir': str(source_dir),
        'output_dir': str(output_dir),
        'target_size': list(target_size),
        'timestamp': datetime.now().isoformat(),
        'splits': {}
    }

    # Procesar cada split
    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"Procesando split: {split.upper()}")
        print(f"{'='*60}")

        # Parsear archivo de etiquetas
        label_file = source_dir / f'{split}.txt'
        if not label_file.exists():
            print(f"Warning: No se encuentra {label_file}, saltando...")
            continue

        samples = parse_label_file(label_file)
        print(f"Total de muestras en {split}.txt: {len(samples)}")

        # Estadisticas de etiquetas
        label_counts = Counter(s['label'] for s in samples)
        source_counts = Counter(s['source'] for s in samples)

        print(f"\nDistribucion por etiqueta:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count} ({100*count/len(samples):.1f}%)")

        print(f"\nDistribucion por fuente:")
        for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
            print(f"  {src}: {count} ({100*count/len(samples):.1f}%)")

        if args.dry_run:
            global_stats['splits'][split] = {
                'total': len(samples),
                'by_label': dict(label_counts),
                'by_source': dict(source_counts)
            }
            continue

        # Directorio fuente de imagenes para este split
        src_images_dir = source_dir / split

        if not src_images_dir.exists():
            print(f"Warning: No se encuentra directorio de imagenes {src_images_dir}")
            continue

        # Procesar imagenes
        stats = prepare_split(
            samples=samples,
            src_dir=src_images_dir,
            dst_dir=output_dir,
            split_name=split,
            target_size=target_size
        )

        print(f"\nResultados del procesamiento:")
        print(f"  Total: {stats['total']}")
        print(f"  Procesadas: {stats['processed']}")
        print(f"  Fallidas: {stats['failed']}")

        if stats['missing_files']:
            print(f"\n  Archivos no encontrados ({len(stats['missing_files'])}):")
            for f in stats['missing_files'][:5]:
                print(f"    - {f}")
            if len(stats['missing_files']) > 5:
                print(f"    ... y {len(stats['missing_files'])-5} mas")

        global_stats['splits'][split] = {
            'total': stats['total'],
            'processed': stats['processed'],
            'failed': stats['failed'],
            'by_label': dict(stats['by_label']),
            'by_source': dict(stats['by_source']),
            'missing_count': len(stats['missing_files'])
        }

    # Guardar estadisticas globales
    stats_path = output_dir / 'preparation_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(global_stats, f, indent=2)
    print(f"\nEstadisticas guardadas en: {stats_path}")

    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)

    for split, stats in global_stats['splits'].items():
        print(f"\n{split.upper()}:")
        if 'processed' in stats:
            print(f"  Procesadas: {stats['processed']}/{stats['total']}")
        else:
            print(f"  Total: {stats['total']}")
        print(f"  Por etiqueta: {stats['by_label']}")

    if not args.dry_run:
        print(f"\nImagenes procesadas guardadas en: {output_dir}")
        print("\nEstructura creada:")
        print(f"  {output_dir}/")
        for split in args.splits:
            print(f"    {split}/")
            print(f"      positive/")
            print(f"      negative/")
            print(f"      metadata.csv")


if __name__ == '__main__':
    main()
