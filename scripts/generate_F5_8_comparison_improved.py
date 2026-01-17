#!/usr/bin/env python3
"""
Script mejorado para generar F5.8: Comparación de matrices de confusión SAHS.

Mejoras de legibilidad:
- Fuentes más grandes
- Mejor contraste
- Layout optimizado para publicación científica
"""

import json
import matplotlib
matplotlib.use('Agg')  # Backend sin display para mejor renderizado
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Configurar matplotlib para mejor renderizado de texto
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1
plt.rcParams['text.antialiased'] = True


def plot_comparison_improved(
    configs: dict,
    output_path: Path,
):
    """
    Genera comparación mejorada de matrices de confusión.

    Args:
        configs: Dict con {nombre: path_results.json}
        output_path: Ruta de salida
    """
    # Crear figura más grande con mejor espaciado
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    # Mapeo de nombres de clases (sin saltos de línea para mejor legibilidad)
    class_display_names = {
        "COVID": "COVID-19",
        "Normal": "Normal",
        "Viral_Pneumonia": "Neumonia Viral"
    }

    for idx, (config_name, config_path) in enumerate(configs.items()):
        # Cargar resultados
        with open(config_path) as f:
            config_results = json.load(f)

        cm_config = np.array(config_results["confusion_matrix"])
        class_names = config_results["class_names"]
        acc_config = config_results["test_metrics"]["accuracy"] * 100
        f1_config = config_results["test_metrics"]["f1_macro"] * 100

        # Nombres en español
        display_names = [class_display_names.get(name, name) for name in class_names]

        ax = axes[idx]

        # Calcular porcentajes para colorear
        cm_percent = cm_config.astype('float') / cm_config.sum(axis=1)[:, np.newaxis] * 100

        # Crear heatmap con mejor contraste
        sns.heatmap(
            cm_percent,
            annot=False,
            fmt='.1f',
            cmap='Blues',
            cbar=False,
            ax=ax,
            vmin=0,
            vmax=100,
            linewidths=2,
            linecolor='white'
        )

        # Anotar SOLO valores absolutos en grande
        for i in range(3):
            for j in range(3):
                value = cm_config[i, j]
                percent = cm_percent[i, j]

                # Color de texto contrastado
                if percent > 60:
                    text_color = 'white'
                else:
                    text_color = 'black'

                # Negrita y tamaño mayor para diagonal
                if i == j:
                    weight = 'bold'
                    fontsize = 18
                else:
                    weight = 'normal'
                    fontsize = 16

                # Mostrar SOLO el valor absoluto (sin porcentaje)
                ax.text(
                    j + 0.5, i + 0.5, f'{value}',
                    ha='center', va='center',
                    color=text_color,
                    fontsize=fontsize,
                    weight=weight
                )

        # Título con métricas (más grande y claro)
        title_lines = [
            f'{config_name}',
            f'Acc: {acc_config:.2f}% | F1-Macro: {f1_config:.2f}%'
        ]
        ax.set_title(
            '\n'.join(title_lines),
            fontsize=15,
            fontweight='bold',
            pad=20
        )

        # Etiquetas de ejes (más grandes)
        if idx == 0:
            ax.set_ylabel('Clase Real', fontsize=15, fontweight='bold', labelpad=10)
            ax.set_yticklabels(display_names, rotation=0, fontsize=13, fontweight='normal')
        else:
            ax.set_ylabel('', fontsize=15, fontweight='bold', labelpad=10)
            ax.set_yticklabels(display_names, rotation=0, fontsize=13, fontweight='normal')

        ax.set_xlabel('Clase Predicha', fontsize=15, fontweight='bold', labelpad=10)
        ax.set_xticklabels(display_names, rotation=0, ha='center', fontsize=13, fontweight='normal')

        # Añadir letra de subfigura
        letters = ['(a)', '(b)', '(c)']
        ax.text(
            -0.15, 1.15, letters[idx],
            transform=ax.transAxes,
            fontsize=16,
            fontweight='bold',
            va='top'
        )

    # Título general
    plt.suptitle(
        'Comparación de Configuraciones de Preprocesamiento con SAHS',
        fontsize=17,
        fontweight='bold',
        y=0.99
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Guardar con alta resolución y formato PNG optimizado
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
        format='png',
        metadata={'Software': 'Matplotlib'}
    )
    print(f"✓ Figura mejorada guardada: {output_path}")
    plt.close('all')


def main():
    """Función principal."""
    # Rutas
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "docs/Tesis/Figures"

    # Configuraciones a comparar
    configs = {
        "Original + SAHS": base_dir / "outputs/classifier_original_sahs/results.json",
        "Warped + SAHS": base_dir / "outputs/classifier_warped_sahs_masked/results.json",
        "Cropped + SAHS": base_dir / "outputs/classifier_cropped_12_sahs/results.json",
    }

    # Verificar que existan los archivos
    for config_name, config_path in configs.items():
        if not config_path.exists():
            print(f"⚠ Advertencia: No se encontró {config_path}")
            return

    print("=" * 70)
    print("GENERACIÓN DE COMPARACIÓN MEJORADA - F5.8")
    print("=" * 70)

    # Cargar y mostrar resumen
    for config_name, config_path in configs.items():
        with open(config_path) as f:
            results = json.load(f)

        acc = results["test_metrics"]["accuracy"] * 100
        f1 = results["test_metrics"]["f1_macro"] * 100

        print(f"\n{config_name}:")
        print(f"  Accuracy: {acc:.2f}%")
        print(f"  F1-Macro: {f1:.2f}%")
        print(f"  Matriz:")
        print(f"  {results['confusion_matrix']}")

    # Generar figura mejorada
    output_path = output_dir / "F5.8_comparacion_sahs.png"
    plot_comparison_improved(
        configs=configs,
        output_path=output_path
    )

    print("\n" + "=" * 70)
    print("PROCESO COMPLETADO")
    print("=" * 70)
    print("\nMejoras aplicadas:")
    print("  ✓ Fuentes más grandes (16-18pt para valores, 15pt para ejes)")
    print("  ✓ Solo valores absolutos (sin porcentajes)")
    print("  ✓ Mejor contraste de colores")
    print("  ✓ Subfiguras etiquetadas (a, b, c)")
    print("  ✓ Layout optimizado para publicación (22x8 pulgadas)")
    print("  ✓ Renderizado mejorado (anti-aliasing activado)")
    print("  ✓ Etiquetas de ejes más nítidas (fontsize 15pt)")


if __name__ == "__main__":
    main()
