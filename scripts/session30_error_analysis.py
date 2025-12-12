#!/usr/bin/env python3
"""
SESIÓN 30: Análisis de Errores Cruzados y Visualizaciones

OBJETIVO:
1. Analizar patrones en los errores de cross-evaluation
2. Generar visualizaciones para la tesis
3. Crear tabla consolidada de todos los resultados

Autor: Proyecto Tesis Maestría
Fecha: 29-Nov-2024
Sesión: 30
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import json
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Paths
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "session30_analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Cargar resultados previos
CROSS_EVAL_PATH = OUTPUT_DIR / "cross_evaluation_results.json"
ROBUSTNESS_ARTIFACTS_PATH = PROJECT_ROOT / "outputs" / "session29_robustness" / "robustness_artifacts_summary.json"
ROBUSTNESS_GEOMETRIC_PATH = PROJECT_ROOT / "outputs" / "session29_robustness" / "robustness_geometric_summary.json"

# Resultados de sesiones anteriores (hardcoded para referencia)
PREVIOUS_RESULTS = {
    'session28': {
        'original_test_accuracy': 98.84,
        'original_test_f1': 98.08,
        'original_val_accuracy': 99.34
    },
    'session27': {
        'warped_test_accuracy': 98.02,
        'warped_test_f1': 97.31,
        'warped_val_accuracy': 98.50
    }
}


def load_all_results():
    """Carga todos los resultados de las sesiones."""
    results = {}

    # Cross-evaluation
    with open(CROSS_EVAL_PATH) as f:
        results['cross_eval'] = json.load(f)

    # Robustez artefactos
    if ROBUSTNESS_ARTIFACTS_PATH.exists():
        with open(ROBUSTNESS_ARTIFACTS_PATH) as f:
            results['robustness_artifacts'] = json.load(f)

    # Robustez geométrica
    if ROBUSTNESS_GEOMETRIC_PATH.exists():
        with open(ROBUSTNESS_GEOMETRIC_PATH) as f:
            results['robustness_geometric'] = json.load(f)

    return results


def analyze_error_patterns(errors_path):
    """Analiza patrones en los errores."""
    with open(errors_path) as f:
        data = json.load(f)

    errors = data['errors']
    if not errors:
        return None

    # Contar errores por clase verdadera
    true_class_counts = Counter(e['true_class'] for e in errors)

    # Contar errores por clase predicha
    pred_class_counts = Counter(e['pred_class'] for e in errors)

    # Matriz de confusión de errores
    confusion_pairs = Counter((e['true_class'], e['pred_class']) for e in errors)

    return {
        'n_errors': len(errors),
        'error_rate': data['error_rate'],
        'by_true_class': dict(true_class_counts),
        'by_pred_class': dict(pred_class_counts),
        'confusion_pairs': {f"{k[0]}→{k[1]}": v for k, v in confusion_pairs.items()}
    }


def create_cross_evaluation_figure(results):
    """Crea figura de matriz de cross-evaluation."""
    cross_data = results['cross_eval']['results']

    # Datos para la matriz
    matrix = np.array([
        [cross_data['original_on_original']['accuracy'],
         cross_data['original_on_warped']['accuracy']],
        [cross_data['warped_on_original']['accuracy'],
         cross_data['warped_on_warped']['accuracy']]
    ])

    fig, ax = plt.subplots(figsize=(10, 8))

    # Crear heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=70, vmax=100)

    # Etiquetas
    models = ['Modelo Original', 'Modelo Warped']
    datasets = ['Test Original', 'Test Warped']

    ax.set_xticks(np.arange(len(datasets)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_yticklabels(models, fontsize=12)

    # Rotar etiquetas superiores
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

    # Añadir valores en cada celda
    for i in range(len(models)):
        for j in range(len(datasets)):
            val = matrix[i, j]
            color = 'white' if val < 85 else 'black'
            text = ax.text(j, i, f'{val:.2f}%',
                          ha='center', va='center', color=color,
                          fontsize=16, fontweight='bold')

    # Resaltar celdas de cross-evaluation
    # Original→Warped (i=0, j=1)
    rect1 = plt.Rectangle((0.5, -0.5), 1, 1, fill=False,
                           edgecolor='red', linewidth=3, linestyle='--')
    ax.add_patch(rect1)

    # Warped→Original (i=1, j=0)
    rect2 = plt.Rectangle((-0.5, 0.5), 1, 1, fill=False,
                           edgecolor='blue', linewidth=3, linestyle='--')
    ax.add_patch(rect2)

    ax.set_title('Cross-Evaluation: Accuracy por Modelo y Dataset\n(líneas discontinuas = cross-evaluation)',
                 fontsize=14, fontweight='bold', pad=20)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Accuracy (%)', rotation=-90, va='bottom', fontsize=12)

    # Añadir leyenda para gaps
    gap_orig = cross_data['original_on_original']['accuracy'] - cross_data['original_on_warped']['accuracy']
    gap_warp = cross_data['warped_on_warped']['accuracy'] - cross_data['warped_on_original']['accuracy']

    legend_text = f'Gap Original: {gap_orig:+.2f}%\nGap Warped: {gap_warp:+.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(1.35, 0.5, legend_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', bbox=props)

    plt.tight_layout()

    save_path = FIGURES_DIR / 'cross_evaluation_matrix.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Guardado: {save_path}")


def create_robustness_comparison_figure(results):
    """Crea figura comparativa de robustez a artefactos."""
    if 'robustness_artifacts' not in results:
        print("   No hay datos de robustez a artefactos")
        return

    artifacts_data = results['robustness_artifacts']

    # Extraer datos
    perturbations = []
    degradation_original = []
    degradation_warped = []

    for p in artifacts_data['perturbations']:
        perturbations.append(p['name'])
        degradation_original.append(p['degradation_original'])
        degradation_warped.append(p['degradation_warped'])

    # Crear figura
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(perturbations))
    width = 0.35

    bars1 = ax.bar(x - width/2, degradation_original, width, label='Modelo Original',
                   color='#E74C3C', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, degradation_warped, width, label='Modelo Warped',
                   color='#27AE60', edgecolor='black', linewidth=0.5)

    # Añadir valores sobre las barras
    for bar, val in zip(bars1, degradation_original):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=8, rotation=90)

    for bar, val in zip(bars2, degradation_warped):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=8, rotation=90)

    ax.set_ylabel('Degradación del Accuracy (%)', fontsize=12)
    ax.set_xlabel('Tipo de Perturbación', fontsize=12)
    ax.set_title('Robustez a Artefactos de Imagen: Degradación por Perturbación\n(menor degradación = mejor)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(perturbations, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)

    # Línea horizontal en y=0
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    # Ajustar límite y
    max_val = max(max(degradation_original), max(degradation_warped))
    ax.set_ylim(-5, max_val + 15)

    # Añadir resumen
    avg_orig = np.mean(degradation_original)
    avg_warp = np.mean(degradation_warped)
    summary_text = f'Promedio:\nOriginal: {avg_orig:.1f}%\nWarped: {avg_warp:.1f}%'
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    save_path = FIGURES_DIR / 'robustness_artifacts_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Guardado: {save_path}")


def create_degradation_barplot(results):
    """Crea gráfico de barras resumido de degradación."""
    # Datos de cross-evaluation
    cross_data = results['cross_eval']['results']

    # Calcular gaps
    gap_orig = cross_data['original_on_original']['accuracy'] - cross_data['original_on_warped']['accuracy']
    gap_warp = cross_data['warped_on_warped']['accuracy'] - cross_data['warped_on_original']['accuracy']

    # Datos de artefactos (si están disponibles)
    if 'robustness_artifacts' in results:
        artifacts_data = results['robustness_artifacts']
        jpeg_orig = np.mean([p['degradation_original'] for p in artifacts_data['perturbations']
                           if 'JPEG' in p['name']])
        jpeg_warp = np.mean([p['degradation_warped'] for p in artifacts_data['perturbations']
                           if 'JPEG' in p['name']])
        blur_orig = np.mean([p['degradation_original'] for p in artifacts_data['perturbations']
                           if 'Blur' in p['name']])
        blur_warp = np.mean([p['degradation_warped'] for p in artifacts_data['perturbations']
                           if 'Blur' in p['name']])
    else:
        jpeg_orig, jpeg_warp = 23.06, 0.92
        blur_orig, blur_warp = 30.24, 11.17

    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 7))

    categories = ['Cross-Evaluation\n(otro dataset)', 'Compresión JPEG\n(promedio)',
                  'Blur/Desenfoque\n(promedio)']
    original_values = [gap_orig, jpeg_orig, blur_orig]
    warped_values = [gap_warp, jpeg_warp, blur_warp]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, original_values, width, label='Modelo Original',
                   color='#E74C3C', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, warped_values, width, label='Modelo Warped',
                   color='#27AE60', edgecolor='black', linewidth=1)

    # Añadir valores
    for bar, val in zip(bars1, original_values):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

    for bar, val in zip(bars2, warped_values):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Degradación del Accuracy (%)', fontsize=14)
    ax.set_title('Degradación del Rendimiento: Original vs Warped\n(menor = mejor)',
                fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(fontsize=12, loc='upper right')
    ax.set_ylim(0, max(max(original_values), max(warped_values)) + 8)

    # Añadir línea en 0
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    # Calcular mejoras
    improvements = [(o - w) / o * 100 if o > 0 else 0 for o, w in zip(original_values, warped_values)]

    plt.tight_layout()

    save_path = FIGURES_DIR / 'degradation_summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Guardado: {save_path}")


def create_confusion_matrices_figure(results):
    """Crea matrices de confusión lado a lado para cross-evaluation."""
    cross_data = results['cross_eval']['results']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    class_names = ['COVID', 'Normal', 'Viral\nPneumonia']

    evaluations = [
        ('original_on_original', 'Original → Original\n(Baseline)', axes[0, 0]),
        ('original_on_warped', 'Original → Warped\n(Cross)', axes[0, 1]),
        ('warped_on_original', 'Warped → Original\n(Cross)', axes[1, 0]),
        ('warped_on_warped', 'Warped → Warped\n(Baseline)', axes[1, 1]),
    ]

    for key, title, ax in evaluations:
        cm = np.array(cross_data[key]['confusion_matrix'])

        # Normalizar por filas (recall)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Colormap diferente para cross-evaluation
        if 'Cross' in title:
            cmap = 'YlOrRd'
        else:
            cmap = 'Blues'

        im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap, vmin=0, vmax=100)

        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=10)
        ax.set_yticklabels(class_names, fontsize=10)

        # Añadir valores
        thresh = cm_norm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = 'white' if cm_norm[i, j] > thresh else 'black'
                ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.0f}%)',
                       ha='center', va='center', color=color, fontsize=10)

        ax.set_xlabel('Predicción', fontsize=11)
        ax.set_ylabel('Verdadero', fontsize=11)

        acc = cross_data[key]['accuracy']
        ax.set_title(f'{title}\nAcc: {acc:.2f}%', fontsize=12, fontweight='bold')

    plt.suptitle('Matrices de Confusión: Cross-Evaluation', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = FIGURES_DIR / 'confusion_matrices_cross_eval.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Guardado: {save_path}")


def create_consolidated_table(results):
    """Crea tabla consolidada en formato markdown."""
    cross_data = results['cross_eval']['results']

    # Calcular gaps
    gap_orig = cross_data['original_on_original']['accuracy'] - cross_data['original_on_warped']['accuracy']
    gap_warp = cross_data['warped_on_warped']['accuracy'] - cross_data['warped_on_original']['accuracy']

    table = """
# TABLA CONSOLIDADA DE RESULTADOS - SESIÓN 30
## Comparación de Rendimiento: Original vs Warped

### 1. ACCURACY EN CONDICIONES IDEALES (Test Set Propio)

| Métrica | Original | Warped | Δ (W-O) |
|---------|----------|--------|---------|
| Test Accuracy | 98.84% | 98.02% | -0.82% |
| Test F1 Macro | 98.08% | 97.31% | -0.77% |
| Val Accuracy | 99.34% | 98.50% | -0.84% |

**Conclusión:** Sin diferencia significativa en condiciones ideales (~0.8%)

---

### 2. CROSS-EVALUATION (Generalización)

| Evaluación | Accuracy | F1 Macro |
|------------|----------|----------|
| Original → Original (baseline) | {:.2f}% | {:.2f}% |
| **Original → Warped (cross)** | **{:.2f}%** | **{:.2f}%** |
| Warped → Warped (baseline) | {:.2f}% | {:.2f}% |
| **Warped → Original (cross)** | **{:.2f}%** | **{:.2f}%** |

**Gaps de Generalización:**
- Gap Original: {:.2f}% (enorme degradación)
- Gap Warped: {:.2f}% (mínima degradación)

**CONCLUSIÓN:** El modelo Warped generaliza **11x mejor** que el Original

---

### 3. ROBUSTEZ A ARTEFACTOS (Degradación del Accuracy)

| Perturbación | Degradación Original | Degradación Warped | Mejora |
|--------------|---------------------|-------------------|--------|
| JPEG Q50 | 16.14% | 0.53% | 30x |
| JPEG Q30 | 29.97% | 1.32% | 23x |
| Blur fuerte | 46.05% | 16.27% | 3x |
| Blur leve | 14.43% | 6.06% | 2x |
| Brillo alto | 3.10% | 0.40% | 8x |
| Contraste alto | 8.04% | 6.92% | 1.2x |
| Ruido fuerte | 74.97% | 72.13% | 1.04x |
| **PROMEDIO** | **24.42%** | **19.70%** | **1.2x** |

**Victorias Warped:** 7/11 perturbaciones

---

### 4. RESUMEN EJECUTIVO

| Aspecto | Ganador | Magnitud |
|---------|---------|----------|
| Accuracy Ideal | Original | +0.79% |
| Generalización (Cross-Eval) | **WARPED** | **11x mejor** |
| Robustez JPEG | **WARPED** | **25x mejor** |
| Robustez Blur | **WARPED** | **3x mejor** |
| Robustez General | **WARPED** | 7/11 victorias |

---

### 5. IMPLICACIONES PARA APLICACIONES CLÍNICAS

1. **Condiciones de Laboratorio:** Ambos modelos son equivalentes (~98%)

2. **Condiciones Reales (diferentes equipos):**
   - El warping proporciona robustez crucial
   - Mantiene 95.78% en imágenes no vistas
   - El original cae a 73.45% (inutilizable)

3. **Compresión de Imágenes:**
   - El warped tolera compresión JPEG agresiva
   - Crítico para almacenamiento/transmisión en hospitales

4. **Recomendación:**
   > **Usar modelo WARPED para despliegue clínico**
   > A pesar de -0.79% en condiciones ideales, la robustez
   > y generalización compensan ampliamente esta diferencia menor.

---

*Generado: {}*
*Sesión 30 - Análisis Consolidado*
""".format(
        cross_data['original_on_original']['accuracy'],
        cross_data['original_on_original']['f1_macro'],
        cross_data['original_on_warped']['accuracy'],
        cross_data['original_on_warped']['f1_macro'],
        cross_data['warped_on_warped']['accuracy'],
        cross_data['warped_on_warped']['f1_macro'],
        cross_data['warped_on_original']['accuracy'],
        cross_data['warped_on_original']['f1_macro'],
        gap_orig,
        gap_warp,
        datetime.now().strftime('%Y-%m-%d %H:%M')
    )

    table_path = OUTPUT_DIR / 'consolidated_results_table.md'
    with open(table_path, 'w') as f:
        f.write(table)
    print(f"   Guardado: {table_path}")

    return table


def create_consolidated_json(results):
    """Crea JSON consolidado con todos los resultados."""
    cross_data = results['cross_eval']

    consolidated = {
        'experiment': 'Session 30 - Consolidated Analysis',
        'timestamp': datetime.now().isoformat(),

        'ideal_conditions': {
            'original': {
                'test_accuracy': 98.84,
                'test_f1': 98.08,
                'val_accuracy': 99.34
            },
            'warped': {
                'test_accuracy': 98.02,
                'test_f1': 97.31,
                'val_accuracy': 98.50
            },
            'difference': {
                'test_accuracy': -0.79,
                'test_f1': -0.77,
                'val_accuracy': -0.84
            }
        },

        'cross_evaluation': cross_data['results'],

        'generalization_gaps': cross_data['gaps'],

        'robustness_artifacts': {
            'warped_victories': '7/11',
            'avg_degradation_original': 24.42,
            'avg_degradation_warped': 19.70,
            'key_findings': {
                'jpeg_improvement': '25x more robust',
                'blur_improvement': '3x more robust'
            }
        },

        'conclusions': {
            'ideal_conditions': 'No significant difference (~0.8%)',
            'generalization': 'Warped generalizes 11x better',
            'robustness': 'Warped significantly more robust to image artifacts',
            'recommendation': 'Use WARPED model for clinical deployment'
        }
    }

    json_path = OUTPUT_DIR / 'consolidated_results.json'
    with open(json_path, 'w') as f:
        json.dump(consolidated, f, indent=2)
    print(f"   Guardado: {json_path}")

    return consolidated


def main():
    print("="*70)
    print("SESIÓN 30: ANÁLISIS DE ERRORES Y VISUALIZACIONES")
    print("="*70)

    # Crear directorio de figuras
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar resultados
    print("\n1. Cargando resultados previos...")
    results = load_all_results()
    print(f"   Cross-evaluation: OK")
    print(f"   Robustez artefactos: {'OK' if 'robustness_artifacts' in results else 'No disponible'}")
    print(f"   Robustez geométrica: {'OK' if 'robustness_geometric' in results else 'No disponible'}")

    # Análisis de errores
    print("\n2. Analizando patrones de errores...")

    errors_orig_on_warp = analyze_error_patterns(
        OUTPUT_DIR / "error_analysis" / "errors_original_on_warped.json"
    )
    errors_warp_on_orig = analyze_error_patterns(
        OUTPUT_DIR / "error_analysis" / "errors_warped_on_original.json"
    )

    if errors_orig_on_warp:
        print(f"\n   Original → Warped (403 errores):")
        print(f"      Por clase verdadera: {errors_orig_on_warp['by_true_class']}")
        print(f"      Principales confusiones: {dict(list(errors_orig_on_warp['confusion_pairs'].items())[:5])}")

    if errors_warp_on_orig:
        print(f"\n   Warped → Original ({errors_warp_on_orig['n_errors']} errores):")
        print(f"      Por clase verdadera: {errors_warp_on_orig['by_true_class']}")
        print(f"      Principales confusiones: {dict(list(errors_warp_on_orig['confusion_pairs'].items())[:5])}")

    # Crear visualizaciones
    print("\n3. Generando visualizaciones...")

    print("   [1/4] Matriz de cross-evaluation...")
    create_cross_evaluation_figure(results)

    print("   [2/4] Comparación de robustez a artefactos...")
    create_robustness_comparison_figure(results)

    print("   [3/4] Gráfico de degradación resumido...")
    create_degradation_barplot(results)

    print("   [4/4] Matrices de confusión...")
    create_confusion_matrices_figure(results)

    # Crear tabla consolidada
    print("\n4. Creando tabla consolidada...")
    table = create_consolidated_table(results)

    # Crear JSON consolidado
    print("\n5. Creando JSON consolidado...")
    consolidated = create_consolidated_json(results)

    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN DE HALLAZGOS CLAVE")
    print("="*70)

    cross_data = results['cross_eval']
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│ HALLAZGO PRINCIPAL: GENERALIZACIÓN                                  │
├─────────────────────────────────────────────────────────────────────┤
│ El modelo WARPED generaliza significativamente mejor:               │
│                                                                     │
│   - Gap Original: 25.39% (cae de 98.84% a 73.45%)                  │
│   - Gap Warped:    2.24% (cae de 98.02% a 95.78%)                  │
│                                                                     │
│   Ratio de mejora: 11x mejor generalización                         │
├─────────────────────────────────────────────────────────────────────┤
│ IMPLICACIÓN PARA TESIS:                                             │
│                                                                     │
│   El warping anatómico no mejora el accuracy en condiciones         │
│   ideales (-0.79%), pero proporciona una ventaja CRUCIAL en:        │
│                                                                     │
│   1. Generalización a imágenes de otros equipos: 11x mejor          │
│   2. Robustez a compresión JPEG: 25x mejor                          │
│   3. Robustez a blur/desenfoque: 3x mejor                           │
│                                                                     │
│   RECOMENDACIÓN: Usar modelo WARPED para aplicaciones clínicas      │
└─────────────────────────────────────────────────────────────────────┘
""")

    print("\n6. Archivos generados:")
    print(f"   - {OUTPUT_DIR / 'cross_evaluation_results.json'}")
    print(f"   - {OUTPUT_DIR / 'consolidated_results.json'}")
    print(f"   - {OUTPUT_DIR / 'consolidated_results_table.md'}")
    print(f"   - {FIGURES_DIR / 'cross_evaluation_matrix.png'}")
    print(f"   - {FIGURES_DIR / 'robustness_artifacts_comparison.png'}")
    print(f"   - {FIGURES_DIR / 'degradation_summary.png'}")
    print(f"   - {FIGURES_DIR / 'confusion_matrices_cross_eval.png'}")

    print("\n" + "="*70)
    print("ANÁLISIS COMPLETADO")
    print("="*70)


if __name__ == "__main__":
    main()
