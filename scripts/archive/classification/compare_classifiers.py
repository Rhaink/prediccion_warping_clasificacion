#!/usr/bin/env python3
"""
Sesion 22: Comparacion de clasificadores warpeado vs original.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent

def main():
    # Cargar resultados
    warped_results = json.load(open(PROJECT_ROOT / 'outputs/classifier/results.json'))
    original_results = json.load(open(PROJECT_ROOT / 'outputs/classifier_original/results_original.json'))

    print("=" * 70)
    print("COMPARACION: CLASIFICADOR WARPEADO vs ORIGINAL")
    print("=" * 70)

    # Tabla de metricas
    print("\n{:<25} {:>15} {:>15}".format("Metrica", "Warpeado", "Original"))
    print("-" * 55)

    metrics = ['accuracy', 'f1_macro', 'f1_weighted']
    for metric in metrics:
        w_val = warped_results['test_metrics'][metric] * 100
        o_val = original_results['test_metrics'][metric] * 100
        diff = o_val - w_val
        print("{:<25} {:>14.2f}% {:>14.2f}%  ({:+.2f}%)".format(
            metric, w_val, o_val, diff
        ))

    print("\n" + "=" * 70)
    print("METRICAS POR CLASE")
    print("=" * 70)

    classes = ['COVID', 'Normal', 'Viral_Pneumonia']
    for cls in classes:
        print(f"\n{cls}:")
        print("{:<15} {:>12} {:>12}".format("", "Warpeado", "Original"))
        print("-" * 40)

        for metric in ['precision', 'recall', 'f1-score']:
            w_val = warped_results['per_class_metrics'][cls][metric] * 100
            o_val = original_results['per_class_metrics'][cls][metric] * 100
            diff = o_val - w_val
            print("{:<15} {:>11.2f}% {:>11.2f}%  ({:+.2f}%)".format(
                metric, w_val, o_val, diff
            ))

    # Matrices de confusion
    print("\n" + "=" * 70)
    print("MATRICES DE CONFUSION")
    print("=" * 70)

    print("\nWarpeado:")
    cm_w = np.array(warped_results['confusion_matrix'])
    print(cm_w)

    print("\nOriginal:")
    cm_o = np.array(original_results['confusion_matrix'])
    print(cm_o)

    # Generar grafico de comparacion
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Grafico de barras para accuracy y F1
    metrics_names = ['Accuracy', 'F1 Macro', 'F1 Weighted']
    warped_vals = [warped_results['test_metrics'][m] * 100 for m in metrics]
    original_vals = [original_results['test_metrics'][m] * 100 for m in metrics]

    x = np.arange(len(metrics_names))
    width = 0.35

    axes[0].bar(x - width/2, warped_vals, width, label='Warpeado', color='steelblue')
    axes[0].bar(x + width/2, original_vals, width, label='Original', color='coral')
    axes[0].set_ylabel('Porcentaje (%)')
    axes[0].set_title('Metricas Generales')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_names)
    axes[0].legend()
    axes[0].set_ylim([80, 100])
    axes[0].grid(axis='y', alpha=0.3)

    # F1 por clase
    f1_warped = [warped_results['per_class_metrics'][c]['f1-score'] * 100 for c in classes]
    f1_original = [original_results['per_class_metrics'][c]['f1-score'] * 100 for c in classes]

    x = np.arange(len(classes))
    axes[1].bar(x - width/2, f1_warped, width, label='Warpeado', color='steelblue')
    axes[1].bar(x + width/2, f1_original, width, label='Original', color='coral')
    axes[1].set_ylabel('F1-Score (%)')
    axes[1].set_title('F1-Score por Clase')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes, rotation=15)
    axes[1].legend()
    axes[1].set_ylim([80, 100])
    axes[1].grid(axis='y', alpha=0.3)

    # Diferencias
    diffs = [o - w for w, o in zip(warped_vals, original_vals)]
    colors = ['green' if d > 0 else 'red' for d in diffs]
    axes[2].bar(metrics_names, diffs, color=colors, alpha=0.7)
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2].set_ylabel('Diferencia (Original - Warpeado)')
    axes[2].set_title('Diferencia en Metricas')
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = PROJECT_ROOT / 'outputs/classifier/comparison_warped_vs_original.png'
    plt.savefig(save_path, dpi=150)
    print(f"\nGrafico guardado en: {save_path}")
    plt.close()

    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"""
El clasificador en IMAGENES ORIGINALES supera al clasificador en
imagenes WARPEADAS en todas las metricas:

  - Accuracy:    +{(original_vals[0] - warped_vals[0]):.2f}% (Original mejor)
  - F1 Macro:    +{(original_vals[1] - warped_vals[1]):.2f}% (Original mejor)
  - F1 Weighted: +{(original_vals[2] - warped_vals[2]):.2f}% (Original mejor)

POSIBLES EXPLICACIONES:
1. El warping elimina informacion contextual (areas fuera de los pulmones)
2. El fondo negro (~53% de la imagen) puede dificultar el aprendizaje
3. Las imagenes originales tienen mas textura y detalles diagnosticos
4. El preprocesamiento de warping puede introducir artefactos

NOTA: El proposito del warping era normalizar la geometria, no necesariamente
mejorar la clasificacion. El warping puede ser util para:
- Comparar regiones anatomicas especificas entre pacientes
- Reducir variabilidad geometrica para ciertos analisis
- Facilitar la interpretabilidad de modelos
""")


if __name__ == '__main__':
    main()
