#!/usr/bin/env python3
"""
Script de verificación exhaustiva de integridad de dataset splits.

Verifica que:
1. No haya overlap entre train/val/test (crucial para defensa de tesis)
2. Imágenes originales sean binariamente idénticas al dataset original (sin preprocesamiento)
3. Imágenes warped sean binariamente idénticas al dataset warped original (sin preprocesamiento adicional)

Usage:
    python scripts/verify_dataset_splits.py
"""

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from tqdm import tqdm


def compute_md5(file_path: Path) -> str:
    """Calcula el checksum MD5 de un archivo."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def map_category_to_original_dir(category: str) -> str:
    """Mapea categoría del CSV al directorio original."""
    if category == "Viral_Pneumonia":
        return "Viral Pneumonia"
    return category


def verify_no_overlap(
    extracted_dir: Path
) -> Tuple[bool, Dict[str, Set[str]], List[str]]:
    """
    Verifica que no haya overlap entre train/val/test.

    Returns:
        Tuple[bool, Dict, List]: (sin_overlap, sets_por_split, errores)
    """
    print("\n" + "="*60)
    print("VERIFICACIÓN 1: No Overlap entre Splits")
    print("="*60)

    splits = ["train", "val", "test"]
    image_sets = {}
    errors = []

    # Recolectar nombres de archivos de cada split
    for split in splits:
        csv_path = extracted_dir / "metadata" / f"{split}_images.csv"
        if not csv_path.exists():
            errors.append(f"CSV no encontrado: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        # Usar image_name sin extensión para comparar
        image_names = set(df["image_name"].str.replace('.png', '', regex=False))
        image_sets[split] = image_names
        print(f"  {split}: {len(image_names)} imágenes únicas")

    if errors:
        return False, image_sets, errors

    # Verificar overlaps entre cada par de splits
    no_overlap = True

    # Train vs Val
    overlap_train_val = image_sets["train"] & image_sets["val"]
    if overlap_train_val:
        no_overlap = False
        errors.append(f"Overlap train-val: {len(overlap_train_val)} imágenes")
        errors.append(f"  Ejemplos: {list(overlap_train_val)[:5]}")
    else:
        print("  ✓ Train vs Val: Sin overlap")

    # Train vs Test
    overlap_train_test = image_sets["train"] & image_sets["test"]
    if overlap_train_test:
        no_overlap = False
        errors.append(f"Overlap train-test: {len(overlap_train_test)} imágenes")
        errors.append(f"  Ejemplos: {list(overlap_train_test)[:5]}")
    else:
        print("  ✓ Train vs Test: Sin overlap")

    # Val vs Test
    overlap_val_test = image_sets["val"] & image_sets["test"]
    if overlap_val_test:
        no_overlap = False
        errors.append(f"Overlap val-test: {len(overlap_val_test)} imágenes")
        errors.append(f"  Ejemplos: {list(overlap_val_test)[:5]}")
    else:
        print("  ✓ Val vs Test: Sin overlap")

    if no_overlap:
        print("\n  ✅ VERIFICACIÓN EXITOSA: No hay overlap entre splits")
    else:
        print("\n  ❌ ERROR: Se encontró overlap entre splits")

    return no_overlap, image_sets, errors


def verify_original_images(
    extracted_dir: Path,
    original_dataset_dir: Path
) -> Tuple[bool, List[str], Dict[str, int]]:
    """
    Verifica que las imágenes originales sean binariamente idénticas al dataset original.

    Returns:
        Tuple[bool, List, Dict]: (todas_identicas, errores, stats)
    """
    print("\n" + "="*60)
    print("VERIFICACIÓN 2: Imágenes Originales son Idénticas")
    print("="*60)

    errors = []
    stats = {
        "verificadas": 0,
        "identicas": 0,
        "diferentes": 0,
        "faltantes_extracted": 0,
        "faltantes_original": 0
    }

    splits = ["train", "val", "test"]

    for split in splits:
        print(f"\nVerificando split: {split}")
        csv_path = extracted_dir / "metadata" / f"{split}_images.csv"

        if not csv_path.exists():
            errors.append(f"CSV no encontrado: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split}"):
            image_name = row["image_name"]
            category = row["category"]

            # Asegurar extensión .png
            if not image_name.endswith('.png'):
                image_name_with_ext = f"{image_name}.png"
            else:
                image_name_with_ext = image_name

            # Rutas
            category_orig_dir = map_category_to_original_dir(category)
            extracted_path = extracted_dir / "original" / split / category / image_name_with_ext
            original_path = original_dataset_dir / category_orig_dir / "images" / image_name_with_ext

            stats["verificadas"] += 1

            # Verificar que ambos archivos existan
            if not extracted_path.exists():
                stats["faltantes_extracted"] += 1
                errors.append(f"[{split}] Faltante en extracted: {extracted_path}")
                continue

            if not original_path.exists():
                stats["faltantes_original"] += 1
                errors.append(f"[{split}] Faltante en original: {original_path}")
                continue

            # Comparar MD5
            md5_extracted = compute_md5(extracted_path)
            md5_original = compute_md5(original_path)

            if md5_extracted == md5_original:
                stats["identicas"] += 1
            else:
                stats["diferentes"] += 1
                errors.append(
                    f"[{split}] DIFERENTE: {image_name_with_ext}\n"
                    f"  Extracted MD5: {md5_extracted}\n"
                    f"  Original MD5:  {md5_original}"
                )

    # Resumen
    print("\n" + "-"*60)
    print("RESUMEN:")
    print(f"  Verificadas: {stats['verificadas']}")
    print(f"  Idénticas: {stats['identicas']}")
    print(f"  Diferentes: {stats['diferentes']}")
    print(f"  Faltantes en extracted: {stats['faltantes_extracted']}")
    print(f"  Faltantes en original: {stats['faltantes_original']}")

    todas_identicas = (
        stats["diferentes"] == 0 and
        stats["faltantes_extracted"] == 0 and
        stats["faltantes_original"] == 0
    )

    if todas_identicas:
        print("\n  ✅ VERIFICACIÓN EXITOSA: Todas las imágenes originales son idénticas")
    else:
        print("\n  ❌ ERROR: Se encontraron diferencias en imágenes originales")

    return todas_identicas, errors, stats


def verify_warped_images(
    extracted_dir: Path,
    warped_dataset_dir: Path
) -> Tuple[bool, List[str], Dict[str, int]]:
    """
    Verifica que las imágenes warped sean binariamente idénticas al dataset warped original.

    Returns:
        Tuple[bool, List, Dict]: (todas_identicas, errores, stats)
    """
    print("\n" + "="*60)
    print("VERIFICACIÓN 3: Imágenes Warped son Idénticas")
    print("="*60)

    errors = []
    stats = {
        "verificadas": 0,
        "identicas": 0,
        "diferentes": 0,
        "faltantes_extracted": 0,
        "faltantes_warped": 0
    }

    splits = ["train", "val", "test"]

    for split in splits:
        print(f"\nVerificando split: {split}")
        csv_path = extracted_dir / "metadata" / f"{split}_images.csv"

        if not csv_path.exists():
            errors.append(f"CSV no encontrado: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split}"):
            warped_filename = row["warped_filename"]
            category = row["category"]

            # Rutas
            extracted_path = extracted_dir / "warped" / split / category / warped_filename
            warped_path = warped_dataset_dir / split / category / warped_filename

            stats["verificadas"] += 1

            # Verificar que ambos archivos existan
            if not extracted_path.exists():
                stats["faltantes_extracted"] += 1
                errors.append(f"[{split}] Faltante en extracted: {extracted_path}")
                continue

            if not warped_path.exists():
                stats["faltantes_warped"] += 1
                errors.append(f"[{split}] Faltante en warped: {warped_path}")
                continue

            # Comparar MD5
            md5_extracted = compute_md5(extracted_path)
            md5_warped = compute_md5(warped_path)

            if md5_extracted == md5_warped:
                stats["identicas"] += 1
            else:
                stats["diferentes"] += 1
                errors.append(
                    f"[{split}] DIFERENTE: {warped_filename}\n"
                    f"  Extracted MD5: {md5_extracted}\n"
                    f"  Warped MD5:    {md5_warped}"
                )

    # Resumen
    print("\n" + "-"*60)
    print("RESUMEN:")
    print(f"  Verificadas: {stats['verificadas']}")
    print(f"  Idénticas: {stats['identicas']}")
    print(f"  Diferentes: {stats['diferentes']}")
    print(f"  Faltantes en extracted: {stats['faltantes_extracted']}")
    print(f"  Faltantes en warped: {stats['faltantes_warped']}")

    todas_identicas = (
        stats["diferentes"] == 0 and
        stats["faltantes_extracted"] == 0 and
        stats["faltantes_warped"] == 0
    )

    if todas_identicas:
        print("\n  ✅ VERIFICACIÓN EXITOSA: Todas las imágenes warped son idénticas")
    else:
        print("\n  ❌ ERROR: Se encontraron diferencias en imágenes warped")

    return todas_identicas, errors, stats


def verify_csv_consistency(
    extracted_dir: Path,
    warped_dataset_dir: Path
) -> Tuple[bool, List[str]]:
    """
    Verifica que los CSVs del dataset extraído sean idénticos a los originales.

    Returns:
        Tuple[bool, List]: (todos_identicos, errores)
    """
    print("\n" + "="*60)
    print("VERIFICACIÓN 4: CSVs son Idénticos")
    print("="*60)

    errors = []
    all_identical = True

    splits = ["train", "val", "test"]

    for split in splits:
        extracted_csv = extracted_dir / "metadata" / f"{split}_images.csv"
        original_csv = warped_dataset_dir / split / "images.csv"

        if not extracted_csv.exists():
            errors.append(f"CSV extraído no existe: {extracted_csv}")
            all_identical = False
            continue

        if not original_csv.exists():
            errors.append(f"CSV original no existe: {original_csv}")
            all_identical = False
            continue

        # Leer CSVs
        df_extracted = pd.read_csv(extracted_csv)
        df_original = pd.read_csv(original_csv)

        # Verificar que tengan las mismas columnas
        if list(df_extracted.columns) != list(df_original.columns):
            errors.append(f"[{split}] Columnas diferentes")
            all_identical = False
            continue

        # Verificar que tengan el mismo número de filas
        if len(df_extracted) != len(df_original):
            errors.append(
                f"[{split}] Diferente número de filas: "
                f"extracted={len(df_extracted)}, original={len(df_original)}"
            )
            all_identical = False
            continue

        # Verificar contenido idéntico (ordenado)
        df_extracted_sorted = df_extracted.sort_values(by=list(df_extracted.columns)).reset_index(drop=True)
        df_original_sorted = df_original.sort_values(by=list(df_original.columns)).reset_index(drop=True)

        if not df_extracted_sorted.equals(df_original_sorted):
            errors.append(f"[{split}] Contenido del CSV difiere")
            all_identical = False
        else:
            print(f"  ✓ {split}: CSV idéntico ({len(df_extracted)} filas)")

    if all_identical:
        print("\n  ✅ VERIFICACIÓN EXITOSA: Todos los CSVs son idénticos")
    else:
        print("\n  ❌ ERROR: Se encontraron diferencias en CSVs")

    return all_identical, errors


def generate_verification_report(
    output_file: Path,
    results: Dict
):
    """Genera un reporte de verificación."""
    with open(output_file, "w") as f:
        f.write("="*70 + "\n")
        f.write("REPORTE DE VERIFICACIÓN DE INTEGRIDAD DE DATASET SPLITS\n")
        f.write("="*70 + "\n\n")

        f.write(f"Fecha: {results['timestamp']}\n")
        f.write(f"Directorio verificado: {results['extracted_dir']}\n\n")

        # Resumen general
        f.write("RESUMEN GENERAL\n")
        f.write("-"*70 + "\n")
        all_passed = all([
            results['no_overlap'],
            results['originals_identical'],
            results['warped_identical'],
            results['csvs_identical']
        ])

        if all_passed:
            f.write("✅ TODAS LAS VERIFICACIONES EXITOSAS\n\n")
            f.write("El dataset extraído es VÁLIDO para usar en la defensa de tesis:\n")
            f.write("  - No hay overlap entre train/val/test\n")
            f.write("  - Imágenes originales sin preprocesamiento (idénticas al dataset)\n")
            f.write("  - Imágenes warped sin preprocesamiento adicional (idénticas)\n")
            f.write("  - CSVs coinciden exactamente con el dataset original\n")
        else:
            f.write("❌ FALLÓ AL MENOS UNA VERIFICACIÓN\n\n")
            f.write("NO USAR ESTE DATASET HASTA RESOLVER LOS ERRORES\n")

        f.write("\n" + "="*70 + "\n\n")

        # Detalles de cada verificación
        f.write("DETALLE DE VERIFICACIONES\n")
        f.write("-"*70 + "\n\n")

        # 1. No overlap
        f.write("1. No Overlap entre Splits\n")
        f.write(f"   Estado: {'✅ PASS' if results['no_overlap'] else '❌ FAIL'}\n")
        if results['overlap_errors']:
            f.write("   Errores:\n")
            for error in results['overlap_errors']:
                f.write(f"     - {error}\n")
        f.write("\n")

        # 2. Imágenes originales
        f.write("2. Imágenes Originales Idénticas\n")
        f.write(f"   Estado: {'✅ PASS' if results['originals_identical'] else '❌ FAIL'}\n")
        f.write(f"   Verificadas: {results['originals_stats']['verificadas']}\n")
        f.write(f"   Idénticas: {results['originals_stats']['identicas']}\n")
        f.write(f"   Diferentes: {results['originals_stats']['diferentes']}\n")
        if results['originals_errors']:
            f.write("   Primeros errores:\n")
            for error in results['originals_errors'][:10]:
                f.write(f"     - {error}\n")
        f.write("\n")

        # 3. Imágenes warped
        f.write("3. Imágenes Warped Idénticas\n")
        f.write(f"   Estado: {'✅ PASS' if results['warped_identical'] else '❌ FAIL'}\n")
        f.write(f"   Verificadas: {results['warped_stats']['verificadas']}\n")
        f.write(f"   Idénticas: {results['warped_stats']['identicas']}\n")
        f.write(f"   Diferentes: {results['warped_stats']['diferentes']}\n")
        if results['warped_errors']:
            f.write("   Primeros errores:\n")
            for error in results['warped_errors'][:10]:
                f.write(f"     - {error}\n")
        f.write("\n")

        # 4. CSVs
        f.write("4. CSVs Idénticos\n")
        f.write(f"   Estado: {'✅ PASS' if results['csvs_identical'] else '❌ FAIL'}\n")
        if results['csv_errors']:
            f.write("   Errores:\n")
            for error in results['csv_errors']:
                f.write(f"     - {error}\n")
        f.write("\n")

        f.write("="*70 + "\n")
        f.write("FIN DEL REPORTE\n")
        f.write("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Verifica integridad del dataset extraído"
    )
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=Path("outputs/dataset_splits_for_gui"),
        help="Directorio del dataset extraído"
    )
    parser.add_argument(
        "--original-dir",
        type=Path,
        default=Path("data/dataset/COVID-19_Radiography_Dataset"),
        help="Directorio del dataset original"
    )
    parser.add_argument(
        "--warped-dir",
        type=Path,
        default=Path("outputs/warped_lung_best/session_warping"),
        help="Directorio del dataset warped original"
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("outputs/dataset_splits_for_gui/VERIFICATION_REPORT.txt"),
        help="Archivo de salida para el reporte"
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("VERIFICACIÓN DE INTEGRIDAD DE DATASET SPLITS")
    print("="*70)
    print(f"\nDirectorio extraído: {args.extracted_dir}")
    print(f"Dataset original: {args.original_dir}")
    print(f"Dataset warped: {args.warped_dir}")

    # Verificar que los directorios existan
    if not args.extracted_dir.exists():
        print(f"\n❌ ERROR: Directorio extraído no existe: {args.extracted_dir}")
        return 1

    if not args.original_dir.exists():
        print(f"\n❌ ERROR: Dataset original no existe: {args.original_dir}")
        return 1

    if not args.warped_dir.exists():
        print(f"\n❌ ERROR: Dataset warped no existe: {args.warped_dir}")
        return 1

    # Resultados
    from datetime import datetime
    results = {
        "timestamp": datetime.now().isoformat(),
        "extracted_dir": str(args.extracted_dir)
    }

    # Verificación 1: No overlap
    no_overlap, image_sets, overlap_errors = verify_no_overlap(args.extracted_dir)
    results["no_overlap"] = no_overlap
    results["overlap_errors"] = overlap_errors

    # Verificación 2: Imágenes originales idénticas
    originals_identical, originals_errors, originals_stats = verify_original_images(
        args.extracted_dir,
        args.original_dir
    )
    results["originals_identical"] = originals_identical
    results["originals_errors"] = originals_errors
    results["originals_stats"] = originals_stats

    # Verificación 3: Imágenes warped idénticas
    warped_identical, warped_errors, warped_stats = verify_warped_images(
        args.extracted_dir,
        args.warped_dir
    )
    results["warped_identical"] = warped_identical
    results["warped_errors"] = warped_errors
    results["warped_stats"] = warped_stats

    # Verificación 4: CSVs idénticos
    csvs_identical, csv_errors = verify_csv_consistency(
        args.extracted_dir,
        args.warped_dir
    )
    results["csvs_identical"] = csvs_identical
    results["csv_errors"] = csv_errors

    # Generar reporte
    generate_verification_report(args.output_report, results)
    print(f"\n\nReporte guardado en: {args.output_report}")

    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)

    all_passed = all([
        no_overlap,
        originals_identical,
        warped_identical,
        csvs_identical
    ])

    if all_passed:
        print("\n✅ TODAS LAS VERIFICACIONES EXITOSAS")
        print("\nEl dataset es VÁLIDO para usar en la defensa de tesis:")
        print("  ✓ No hay overlap entre train/val/test")
        print("  ✓ Imágenes originales SIN preprocesamiento (binariamente idénticas)")
        print("  ✓ Imágenes warped SIN preprocesamiento adicional (binariamente idénticas)")
        print("  ✓ CSVs coinciden exactamente")
        print("\n🎓 SEGURO PARA DEFENSA DE TESIS")
        return 0
    else:
        print("\n❌ FALLÓ AL MENOS UNA VERIFICACIÓN")
        print("\n⚠️  NO USAR ESTE DATASET HASTA RESOLVER LOS ERRORES")
        print("\nVerificaciones fallidas:")
        if not no_overlap:
            print("  ✗ Hay overlap entre splits")
        if not originals_identical:
            print("  ✗ Imágenes originales difieren del dataset")
        if not warped_identical:
            print("  ✗ Imágenes warped difieren del dataset")
        if not csvs_identical:
            print("  ✗ CSVs difieren del original")

        print(f"\nVer detalles en: {args.output_report}")
        return 1


if __name__ == "__main__":
    exit(main())
