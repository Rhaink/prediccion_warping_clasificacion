"""
Script para crear CSVs de splits para el dataset manual.

Este script genera dos CSVs:
1. manual_warped.csv - Dataset manual con imágenes warped
2. manual_original.csv - Dataset manual con imágenes originales

Los splits (train/val/test) se extraen de la estructura de directorios
del dataset warped existente en outputs/warped_dataset/
"""

import pandas as pd
from pathlib import Path


def create_manual_csvs():
    """Crea los CSVs para datasets manuales (warped y original)."""

    # Rutas
    base_path = Path(__file__).parent.parent.parent.parent
    warped_dataset_path = base_path / "outputs" / "warped_dataset"
    metrics_dir = Path(__file__).parent.parent / "results" / "metrics"

    print("=" * 60)
    print("CREACIÓN DE CSVs PARA DATASET MANUAL")
    print("=" * 60)

    # Recolectar información de cada split
    records_warped = []
    records_original = []

    for split in ["train", "val", "test"]:
        split_path = warped_dataset_path / split

        # Leer el CSV de imágenes si existe
        images_csv = split_path / "images.csv"
        if images_csv.exists():
            df_split = pd.read_csv(images_csv)
            print(f"\n{split}: {len(df_split)} imágenes desde images.csv")
        else:
            # Si no hay CSV, escanear directorios
            print(f"\n{split}: escaneando directorios...")
            df_split = None

        # Escanear las clases
        for class_name in ["COVID", "Normal", "Viral_Pneumonia"]:
            class_path = split_path / class_name
            if not class_path.exists():
                continue

            for img_file in class_path.glob("*_warped.png"):
                # Extraer image_id del nombre (quitar _warped.png)
                image_id = img_file.stem.replace("_warped", "")

                # Registro para warped
                warped_path = f"outputs/warped_dataset/{split}/{class_name}/{img_file.name}"
                records_warped.append({
                    "image_id": image_id,
                    "class": class_name,
                    "split": split,
                    "path": warped_path
                })

                # Registro para original
                original_path = f"data/dataset/{class_name}/{image_id}.png"
                records_original.append({
                    "image_id": image_id,
                    "class": class_name,
                    "split": split,
                    "path": original_path
                })

    # Crear DataFrames
    df_warped = pd.DataFrame(records_warped)
    df_original = pd.DataFrame(records_original)

    # Ordenar por image_id para consistencia
    df_warped = df_warped.sort_values(["split", "class", "image_id"]).reset_index(drop=True)
    df_original = df_original.sort_values(["split", "class", "image_id"]).reset_index(drop=True)

    # Guardar CSVs
    warped_csv = metrics_dir / "03_manual_warped.csv"
    original_csv = metrics_dir / "03_manual_original.csv"

    df_warped.to_csv(warped_csv, index=False)
    df_original.to_csv(original_csv, index=False)

    print(f"\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"\nDataset Manual Warped: {warped_csv.name}")
    print(f"  Total: {len(df_warped)} imágenes")
    for split in ["train", "val", "test"]:
        count = len(df_warped[df_warped["split"] == split])
        print(f"  {split}: {count}")

    print(f"\nDataset Manual Original: {original_csv.name}")
    print(f"  Total: {len(df_original)} imágenes")
    for split in ["train", "val", "test"]:
        count = len(df_original[df_original["split"] == split])
        print(f"  {split}: {count}")

    # Verificar distribución de clases
    print(f"\nDistribución de clases (Manual Warped):")
    for cls in ["COVID", "Normal", "Viral_Pneumonia"]:
        for split in ["train", "val", "test"]:
            count = len(df_warped[(df_warped["class"] == cls) & (df_warped["split"] == split)])
            print(f"  {cls} - {split}: {count}")

    # Verificar que las imágenes originales existen
    print(f"\nVerificando existencia de imágenes originales...")
    missing = 0
    for _, row in df_original.iterrows():
        img_path = base_path / row["path"]
        if not img_path.exists():
            missing += 1
            if missing <= 5:
                print(f"  FALTA: {row['path']}")

    if missing > 0:
        print(f"  Total faltantes: {missing}")
    else:
        print(f"  Todas las imágenes originales existen!")

    print(f"\nCSVs guardados en: {metrics_dir}")

    return df_warped, df_original


if __name__ == "__main__":
    create_manual_csvs()
