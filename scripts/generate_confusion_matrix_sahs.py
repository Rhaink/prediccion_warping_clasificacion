#!/usr/bin/env python3
"""
Script para generar matriz de confusión visual del experimento SAHS.
Genera la matriz de confusión del clasificador warped_sahs_masked
con los valores correctos verificados.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    title: str,
    output_path: Path,
    accuracy: float,
    f1_macro: float,
):
    """
    Genera un heatmap de la matriz de confusión.

    Args:
        cm: Matriz de confusión (array 3x3)
        class_names: Nombres de las clases
        title: Título del gráfico
        output_path: Ruta de salida
        accuracy: Accuracy del modelo
        f1_macro: F1-Score macro del modelo
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    # Calcular porcentajes
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Crear heatmap
    heatmap = sns.heatmap(
        cm_percent,
        annot=False,
        fmt='.1f',
        cmap='Blues',
        cbar_kws={'label': 'Porcentaje (%)'},
        ax=ax,
        vmin=0,
        vmax=100
    )
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Porcentaje (%)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)

    # Anotar con valores absolutos y porcentajes
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = cm[i, j]
            percent = cm_percent[i, j]

            # Color del texto (blanco para celdas oscuras)
            if percent > 50:
                text_color = 'white'
            else:
                text_color = 'black'

            # Negrita para la diagonal
            if i == j:
                weight = 'bold'
                text = f'{value}\n({percent:.1f}%)'
            else:
                weight = 'normal'
                text = f'{value}\n({percent:.1f}%)'

            ax.text(
                j + 0.5, i + 0.5, text,
                ha='center', va='center',
                color=text_color,
                fontsize=14,
                weight=weight
            )

    # Configurar ejes
    ax.set_xlabel('Predicción', fontsize=16, fontweight='bold')
    ax.set_ylabel('Categoría Real', fontsize=16, fontweight='bold')
    ax.set_title(
        f'{title}\nAccuracy: {accuracy:.2f}% | F1-Macro: {f1_macro:.2f}%',
        fontsize=18,
        fontweight='bold',
        pad=20
    )

    # Etiquetas de las clases
    ax.set_xticklabels(class_names, rotation=0, ha='center')
    ax.set_yticklabels(class_names, rotation=0)
    ax.tick_params(axis='both', labelsize=14)

    # Ajustar layout
    plt.tight_layout()

    # Guardar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Matriz de confusión guardada: {output_path}")
    plt.close()


def main():
    # Rutas
    base_dir = Path(__file__).parent.parent
    results_path = base_dir / "outputs/classifier_warped_sahs_masked/results.json"
    output_dir = base_dir / "docs/Tesis/Figures"

    # Cargar resultados
    with open(results_path) as f:
        results = json.load(f)

    # Extraer datos
    cm = np.array(results["confusion_matrix"])
    class_names = results["class_names"]
    accuracy = results["test_metrics"]["accuracy"] * 100
    f1_macro = results["test_metrics"]["f1_macro"] * 100

    # Mapeo de nombres más legibles
    class_display_names = {
        "COVID": "COVID-19",
        "Normal": "Normal",
        "Viral_Pneumonia": "Neumonía Viral"
    }
    display_names = [class_display_names.get(name, name) for name in class_names]

    print("=" * 70)
    print("GENERACIÓN DE MATRIZ DE CONFUSIÓN - EXPERIMENTO SAHS")
    print("=" * 70)
    print(f"\nArchivo de resultados: {results_path}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"F1-Macro: {f1_macro:.2f}%")
    print(f"Test samples: {results['test_samples']}")
    print("\nMatriz de confusión:")
    print(cm)

    # Generar figura
    output_path = output_dir / "F5.7_matriz_confusion_sahs.png"
    plot_confusion_matrix(
        cm=cm,
        class_names=display_names,
        title="Matriz de Confusión - Clasificador Normalizado + SAHS",
        output_path=output_path,
        accuracy=accuracy,
        f1_macro=f1_macro
    )

    # Generar también comparación de las 3 configuraciones
    print("\n" + "=" * 70)
    print("GENERACIÓN DE COMPARACIÓN DE CONFIGURACIONES")
    print("=" * 70)

    # Cargar resultados de las 3 configuraciones
    configs = {
        "Original + SAHS": base_dir / "outputs/classifier_original_sahs/results.json",
        "Normalizado + SAHS": base_dir / "outputs/classifier_warped_sahs_masked/results.json",
        "Cropped + SAHS": base_dir / "outputs/classifier_cropped_12_sahs/results.json",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (config_name, config_path) in enumerate(configs.items()):
        with open(config_path) as f:
            config_results = json.load(f)

        cm_config = np.array(config_results["confusion_matrix"])
        acc_config = config_results["test_metrics"]["accuracy"] * 100
        f1_config = config_results["test_metrics"]["f1_macro"] * 100

        ax = axes[idx]

        # Calcular porcentajes
        cm_percent = cm_config.astype('float') / cm_config.sum(axis=1)[:, np.newaxis] * 100

        # Crear heatmap
        sns.heatmap(
            cm_percent,
            annot=False,
            fmt='.1f',
            cmap='Blues',
            cbar=False,
            ax=ax,
            vmin=0,
            vmax=100
        )

        # Anotar valores
        for i in range(3):
            for j in range(3):
                value = cm_config[i, j]
                percent = cm_percent[i, j]

                if percent > 50:
                    text_color = 'white'
                else:
                    text_color = 'black'

                if i == j:
                    weight = 'bold'
                else:
                    weight = 'normal'

                ax.text(
                    j + 0.5, i + 0.5, f'{value}',
                    ha='center', va='center',
                    color=text_color,
                    fontsize=10,
                    weight=weight
                )

        # Configurar título y etiquetas
        ax.set_title(
            f'{config_name}\nAcc: {acc_config:.2f}% | F1: {f1_config:.2f}%',
            fontsize=12,
            fontweight='bold'
        )

        if idx == 0:
            ax.set_ylabel('Real', fontsize=11, fontweight='bold')
            ax.set_yticklabels(display_names, rotation=0)
        else:
            ax.set_yticklabels([])

        ax.set_xlabel('Predicción', fontsize=11, fontweight='bold')
        ax.set_xticklabels(display_names, rotation=45, ha='right')

    plt.suptitle(
        'Comparación de Configuraciones SAHS',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )
    plt.tight_layout()

    # Guardar comparación
    comparison_path = output_dir / "F5.8_comparacion_sahs.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparación guardada: {comparison_path}")
    plt.close()

    print("\n" + "=" * 70)
    print("PROCESO COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
