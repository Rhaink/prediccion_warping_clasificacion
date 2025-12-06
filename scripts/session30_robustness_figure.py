#!/usr/bin/env python3
"""
SESIÓN 30: Generación de figura de robustez a artefactos

Genera visualizaciones de alta calidad para la tesis.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import json
import matplotlib.pyplot as plt

# Paths
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "session30_analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"
ROBUSTNESS_PATH = PROJECT_ROOT / "outputs" / "session29_robustness" / "artifact_robustness_results.json"


def create_robustness_figure():
    """Crea figura de robustez a artefactos."""
    # Cargar datos
    with open(ROBUSTNESS_PATH) as f:
        data = json.load(f)

    # Preparar datos para el gráfico
    perturbations = list(data['perturbations'].keys())
    names_display = {
        'ruido_gaussiano_leve': 'Ruido\nGauss. Leve',
        'ruido_gaussiano_fuerte': 'Ruido\nGauss. Fuerte',
        'blur_leve': 'Blur\nLeve',
        'blur_fuerte': 'Blur\nFuerte',
        'contraste_bajo': 'Contraste\nBajo',
        'contraste_alto': 'Contraste\nAlto',
        'brillo_bajo': 'Brillo\nBajo',
        'brillo_alto': 'Brillo\nAlto',
        'jpeg_q50': 'JPEG\nQ50',
        'jpeg_q30': 'JPEG\nQ30',
        'ruido_blur_combo': 'Ruido+Blur\nCombo'
    }

    labels = [names_display.get(p, p) for p in perturbations]
    degradation_original = [data['perturbations'][p]['degradation_original'] for p in perturbations]
    degradation_warped = [data['perturbations'][p]['degradation_warped'] for p in perturbations]
    winners = [data['perturbations'][p]['winner'] for p in perturbations]

    # Crear figura
    fig, ax = plt.subplots(figsize=(16, 9))

    x = np.arange(len(perturbations))
    width = 0.35

    bars1 = ax.bar(x - width/2, degradation_original, width, label='Modelo Original',
                   color='#E74C3C', edgecolor='black', linewidth=0.8, alpha=0.9)
    bars2 = ax.bar(x + width/2, degradation_warped, width, label='Modelo Warped',
                   color='#27AE60', edgecolor='black', linewidth=0.8, alpha=0.9)

    # Añadir valores sobre las barras
    for bar, val in zip(bars1, degradation_original):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9, rotation=90)

    for bar, val in zip(bars2, degradation_warped):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9, rotation=90)

    # Marcar ganadores con estrellas
    for i, winner in enumerate(winners):
        if winner == 'WARPED':
            ax.annotate('★', xy=(i + width/2, degradation_warped[i] + 5),
                       ha='center', fontsize=14, color='#27AE60')
        else:
            ax.annotate('★', xy=(i - width/2, degradation_original[i] + 5),
                       ha='center', fontsize=14, color='#E74C3C')

    ax.set_ylabel('Degradación del Accuracy (%)', fontsize=13)
    ax.set_xlabel('Tipo de Perturbación', fontsize=13)
    ax.set_title('Robustez a Artefactos de Imagen: Degradación por Perturbación\n(menor degradación = mejor, ★ = ganador)',
                fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=12, loc='upper left')

    # Línea horizontal en y=0
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    # Ajustar límite y
    max_val = max(max(degradation_original), max(degradation_warped))
    ax.set_ylim(-3, max_val + 12)

    # Añadir resumen
    avg_orig = np.mean(degradation_original)
    avg_warp = np.mean(degradation_warped)
    summary_text = f'Promedio:\n  Original: {avg_orig:.1f}%\n  Warped: {avg_warp:.1f}%\n\nVictorias Warped: 7/11'
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, edgecolor='gray')
    ax.text(0.02, 0.97, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    save_path = FIGURES_DIR / 'robustness_artifacts_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {save_path}")

    return save_path


def create_key_comparisons_figure():
    """Crea figura de comparaciones clave (JPEG y Blur)."""
    with open(ROBUSTNESS_PATH) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Datos
    comparisons = [
        {
            'title': 'Compresión JPEG',
            'labels': ['Q50', 'Q30'],
            'orig': [data['perturbations']['jpeg_q50']['degradation_original'],
                     data['perturbations']['jpeg_q30']['degradation_original']],
            'warp': [data['perturbations']['jpeg_q50']['degradation_warped'],
                     data['perturbations']['jpeg_q30']['degradation_warped']]
        },
        {
            'title': 'Blur/Desenfoque',
            'labels': ['Leve', 'Fuerte'],
            'orig': [data['perturbations']['blur_leve']['degradation_original'],
                     data['perturbations']['blur_fuerte']['degradation_original']],
            'warp': [data['perturbations']['blur_leve']['degradation_warped'],
                     data['perturbations']['blur_fuerte']['degradation_warped']]
        },
        {
            'title': 'Ruido Gaussiano',
            'labels': ['Leve', 'Fuerte'],
            'orig': [data['perturbations']['ruido_gaussiano_leve']['degradation_original'],
                     data['perturbations']['ruido_gaussiano_fuerte']['degradation_original']],
            'warp': [data['perturbations']['ruido_gaussiano_leve']['degradation_warped'],
                     data['perturbations']['ruido_gaussiano_fuerte']['degradation_warped']]
        }
    ]

    for ax, comp in zip(axes, comparisons):
        x = np.arange(len(comp['labels']))
        width = 0.35

        bars1 = ax.bar(x - width/2, comp['orig'], width, label='Original',
                       color='#E74C3C', edgecolor='black')
        bars2 = ax.bar(x + width/2, comp['warp'], width, label='Warped',
                       color='#27AE60', edgecolor='black')

        # Valores
        for bar, val in zip(bars1, comp['orig']):
            ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        for bar, val in zip(bars2, comp['warp']):
            ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Degradación (%)', fontsize=11)
        ax.set_title(comp['title'], fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comp['labels'], fontsize=11)
        ax.legend(fontsize=10)
        ax.set_ylim(0, max(max(comp['orig']), max(comp['warp'])) + 15)

    plt.suptitle('Comparación de Robustez: Original vs Warped', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = FIGURES_DIR / 'robustness_key_comparisons.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {save_path}")


def create_summary_figure():
    """Crea figura resumen con todos los hallazgos principales."""
    # Cargar cross-evaluation
    cross_path = OUTPUT_DIR / 'cross_evaluation_results.json'
    with open(cross_path) as f:
        cross_data = json.load(f)

    with open(ROBUSTNESS_PATH) as f:
        rob_data = json.load(f)

    fig = plt.figure(figsize=(16, 10))

    # Subplot 1: Cross-evaluation matrix
    ax1 = fig.add_subplot(2, 2, 1)

    matrix = np.array([
        [cross_data['results']['original_on_original']['accuracy'],
         cross_data['results']['original_on_warped']['accuracy']],
        [cross_data['results']['warped_on_original']['accuracy'],
         cross_data['results']['warped_on_warped']['accuracy']]
    ])

    im = ax1.imshow(matrix, cmap='RdYlGn', vmin=70, vmax=100)

    models = ['Original', 'Warped']
    datasets = ['Test Original', 'Test Warped']

    ax1.set_xticks(np.arange(len(datasets)))
    ax1.set_yticks(np.arange(len(models)))
    ax1.set_xticklabels(datasets, fontsize=10)
    ax1.set_yticklabels(models, fontsize=10)

    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            color = 'white' if val < 85 else 'black'
            ax1.text(j, i, f'{val:.1f}%', ha='center', va='center',
                    color=color, fontsize=14, fontweight='bold')

    ax1.set_title('Cross-Evaluation', fontsize=12, fontweight='bold')

    # Subplot 2: Gaps de generalización
    ax2 = fig.add_subplot(2, 2, 2)

    gap_orig = cross_data['gaps']['original_gap']
    gap_warp = cross_data['gaps']['warped_gap']

    bars = ax2.bar(['Original', 'Warped'], [gap_orig, gap_warp],
                   color=['#E74C3C', '#27AE60'], edgecolor='black', linewidth=2)
    ax2.set_ylabel('Gap de Generalización (%)', fontsize=11)
    ax2.set_title('Gap de Generalización\n(menor = mejor)', fontsize=12, fontweight='bold')

    for bar, val in zip(bars, [gap_orig, gap_warp]):
        ax2.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', fontsize=14, fontweight='bold')

    # Añadir ratio
    ax2.text(0.5, 0.85, f'Ratio: {gap_orig/gap_warp:.0f}x', transform=ax2.transAxes,
            fontsize=13, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Subplot 3: Robustez JPEG
    ax3 = fig.add_subplot(2, 2, 3)

    jpeg_orig = [rob_data['perturbations']['jpeg_q50']['degradation_original'],
                 rob_data['perturbations']['jpeg_q30']['degradation_original']]
    jpeg_warp = [rob_data['perturbations']['jpeg_q50']['degradation_warped'],
                 rob_data['perturbations']['jpeg_q30']['degradation_warped']]

    x = np.arange(2)
    width = 0.35
    ax3.bar(x - width/2, jpeg_orig, width, label='Original', color='#E74C3C', edgecolor='black')
    ax3.bar(x + width/2, jpeg_warp, width, label='Warped', color='#27AE60', edgecolor='black')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['JPEG Q50', 'JPEG Q30'], fontsize=10)
    ax3.set_ylabel('Degradación (%)', fontsize=11)
    ax3.set_title('Robustez a Compresión JPEG', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)

    # Mejora
    improvement = np.mean(jpeg_orig) / np.mean(jpeg_warp)
    ax3.text(0.5, 0.85, f'Warped: {improvement:.0f}x más robusto', transform=ax3.transAxes,
            fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Subplot 4: Resumen textual
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    summary_text = """
RESUMEN DE HALLAZGOS

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONDICIONES IDEALES:
  • Original: 98.81%  |  Warped: 98.02%
  • Diferencia: -0.79% (no significativa)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GENERALIZACIÓN (Cross-Evaluation):
  • Gap Original: 25.4%  →  Cae a 73.5%
  • Gap Warped:    2.2%  →  Mantiene 95.8%
  • Mejora: 11x mejor generalización

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROBUSTEZ A ARTEFACTOS:
  • Warped gana en 7/11 perturbaciones
  • JPEG: 25x más robusto
  • Blur: 3x más robusto

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RECOMENDACIÓN:
  Usar modelo WARPED para aplicación clínica
"""

    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.suptitle('Sesión 30: Análisis Consolidado - Original vs Warped',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    save_path = FIGURES_DIR / 'session30_summary_figure.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {save_path}")


def main():
    print("="*60)
    print("SESIÓN 30: Generando figuras de robustez")
    print("="*60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n1. Figura de robustez a artefactos...")
    create_robustness_figure()

    print("\n2. Figura de comparaciones clave...")
    create_key_comparisons_figure()

    print("\n3. Figura resumen...")
    create_summary_figure()

    print("\n" + "="*60)
    print("FIGURAS GENERADAS")
    print("="*60)


if __name__ == "__main__":
    main()
