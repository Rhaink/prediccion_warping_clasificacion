"""
Script para generar características (ponderantes estandarizados) para todos los datasets.

FASE 4: EXTRACCIÓN DE CARACTERÍSTICAS Y ESTANDARIZACIÓN
========================================================

Este script:
1. Carga cada uno de los 4 datasets
2. Aplica PCA para extraer ponderantes (K componentes)
3. Estandariza con Z-score (media/std SOLO del training)
4. Verifica que training tenga media≈0 y std≈1
5. Guarda CSVs con ponderantes por split
6. Genera figuras de verificación

DATASETS PROCESADOS:
- Full Warped (5,040 train / 1,005 val / 680 test)
- Full Original (5,040 train / 1,005 val / 680 test)
- Manual Warped (717 train / 144 val / 96 test)
- Manual Original (717 train / 144 val / 96 test)

SALIDAS:
- results/metrics/phase4_features/{dataset}_{split}_features.csv
- results/figures/phase4_features/{dataset}/
    - distribution.png
    - scaler_params.png
    - verification_stats.txt
"""

import sys
from pathlib import Path
import numpy as np
import json

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_dataset
from pca import PCA
from features import (
    StandardScaler,
    verify_standardization,
    plot_standardization_distribution,
    plot_feature_statistics,
    save_features_to_csv
)


def process_dataset(
    name: str,
    csv_path: Path,
    base_path: Path,
    output_dir: Path,
    n_components: int = 50,
    scenario: str = "2class",
    verbose: bool = True
) -> dict:
    """
    Procesa un dataset completo: PCA + estandarización + guardado.

    Args:
        name: Nombre del dataset (para archivos de salida)
        csv_path: Ruta al CSV de splits
        base_path: Ruta base del proyecto
        output_dir: Directorio base de salida
        n_components: Número de componentes PCA
        scenario: "2class" o "3class"
        verbose: Si True, imprime progreso

    Returns:
        Dict con estadísticas del procesamiento
    """
    print("\n" + "="*70)
    print(f"PROCESANDO: {name}")
    print("="*70)

    # Crear directorios de salida
    figures_dir = output_dir / "figures" / "phase4_features" / name
    metrics_dir = output_dir / "metrics" / "phase4_features"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar dataset
    print("\n[1/5] Cargando dataset...")
    dataset = load_dataset(csv_path, base_path, scenario=scenario, verbose=verbose)

    # 2. Aplicar PCA (SOLO con training)
    print("\n[2/5] Aplicando PCA...")
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(dataset.train.X, verbose=verbose)
    X_val_pca = pca.transform(dataset.val.X)
    X_test_pca = pca.transform(dataset.test.X)

    print(f"\nPonderantes extraídos:")
    print(f"  Train: {X_train_pca.shape}")
    print(f"  Val:   {X_val_pca.shape}")
    print(f"  Test:  {X_test_pca.shape}")

    # 3. Estandarizar con Z-score (SOLO con training)
    print("\n[3/5] Estandarizando (Z-score)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca, verbose=verbose)
    X_val_scaled = scaler.transform(X_val_pca)
    X_test_scaled = scaler.transform(X_test_pca)

    # 4. Verificar estandarización
    print("\n[4/5] Verificando estandarización...")
    stats = verify_standardization(
        X_train_scaled, X_val_scaled, X_test_scaled,
        verbose=verbose
    )

    # Guardar estadísticas de verificación
    stats_path = figures_dir / "verification_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # 5. Guardar resultados
    print("\n[5/5] Guardando resultados...")

    # CSVs de características
    for split_name, features, labels, ids in [
        ("train", X_train_scaled, dataset.train.y, dataset.train.ids),
        ("val", X_val_scaled, dataset.val.y, dataset.val.ids),
        ("test", X_test_scaled, dataset.test.y, dataset.test.ids)
    ]:
        csv_out = metrics_dir / f"{name}_{split_name}_features.csv"
        save_features_to_csv(
            features=features,
            labels=labels,
            ids=ids,
            class_names=dataset.class_names,
            output_path=csv_out
        )

    # Figuras
    print("\nGenerando figuras...")

    # Distribución de características
    fig1 = plot_standardization_distribution(
        X_train_scaled,
        dataset.train.y,
        dataset.class_names,
        n_features_to_show=6,
        output_path=figures_dir / "distribution.png"
    )

    # Parámetros del scaler
    fig2 = plot_feature_statistics(
        scaler.get_params(),
        output_path=figures_dir / "scaler_params.png"
    )

    # Cerrar figuras para liberar memoria
    import matplotlib.pyplot as plt
    plt.close(fig1)
    plt.close(fig2)

    # Resultado
    result = {
        "name": name,
        "n_train": len(dataset.train.y),
        "n_val": len(dataset.val.y),
        "n_test": len(dataset.test.y),
        "n_components": n_components,
        "train_mean_of_means": stats["train"]["mean_of_means"],
        "train_mean_of_stds": stats["train"]["mean_of_stds"],
        "verified": abs(stats["train"]["mean_of_means"]) < 1e-6 and
                   abs(stats["train"]["mean_of_stds"] - 1.0) < 0.01
    }

    print(f"\n✓ {name} procesado correctamente.")

    return result


def main():
    """Punto de entrada principal."""
    print("="*70)
    print("FASE 4: EXTRACCIÓN DE CARACTERÍSTICAS Y ESTANDARIZACIÓN")
    print("="*70)

    # Rutas
    base_path = Path(__file__).parent.parent.parent.parent
    results_dir = Path(__file__).parent.parent / "results"
    metrics_dir = results_dir / "metrics"

    # Configuración
    n_components = 50  # Mismo que en Fase 3

    # Datasets a procesar
    datasets = [
        {
            "name": "full_warped",
            "csv": metrics_dir / "01_full_balanced_3class_warped.csv"
        },
        {
            "name": "full_original",
            "csv": metrics_dir / "01_full_balanced_3class_original.csv"
        },
        {
            "name": "manual_warped",
            "csv": metrics_dir / "03_manual_warped.csv"
        },
        {
            "name": "manual_original",
            "csv": metrics_dir / "03_manual_original.csv"
        }
    ]

    # Procesar cada dataset
    all_results = []

    for ds in datasets:
        if not ds["csv"].exists():
            print(f"\n⚠ AVISO: {ds['csv']} no existe, saltando...")
            continue

        result = process_dataset(
            name=ds["name"],
            csv_path=ds["csv"],
            base_path=base_path,
            output_dir=results_dir,
            n_components=n_components,
            scenario="2class",
            verbose=True
        )
        all_results.append(result)

    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN DE FASE 4")
    print("="*70)

    print("\n{:<20} {:>8} {:>8} {:>8} {:>12} {:>10}".format(
        "Dataset", "Train", "Val", "Test", "Media≈0?", "Std≈1?"
    ))
    print("-"*70)

    for r in all_results:
        mean_ok = "✓" if abs(r["train_mean_of_means"]) < 1e-6 else "✗"
        std_ok = "✓" if abs(r["train_mean_of_stds"] - 1.0) < 0.01 else "✗"
        print("{:<20} {:>8} {:>8} {:>8} {:>12} {:>10}".format(
            r["name"],
            r["n_train"],
            r["n_val"],
            r["n_test"],
            mean_ok,
            std_ok
        ))

    # Guardar resumen
    summary_path = results_dir / "metrics" / "phase4_features" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResumen guardado en: {summary_path}")

    print("\n" + "="*70)
    print("FASE 4 COMPLETADA")
    print("="*70)
    print("\nEntregables generados:")
    print("  - results/metrics/phase4_features/{dataset}_{split}_features.csv")
    print("  - results/figures/phase4_features/{dataset}/distribution.png")
    print("  - results/figures/phase4_features/{dataset}/scaler_params.png")
    print("  - results/figures/phase4_features/{dataset}/verification_stats.json")


if __name__ == "__main__":
    main()
