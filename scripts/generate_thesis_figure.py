#!/usr/bin/env python3
"""
Genera figura comparativa para la defensa de tesis.

Contenido:
- Gráfico de accuracy vs fill_rate
- Gráfico de robustez (JPEG Q50 degradation) vs fill_rate
- Tabla resumen con recomendación

Uso:
    python scripts/generate_thesis_figure.py
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configurar estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.titlesize'] = 14

# Datos de GROUND_TRUTH.json v2.1.0
data = {
    'original_100': {'fill_rate': 100, 'accuracy': 98.84, 'jpeg_q50': 16.14, 'label': 'Original 100%'},
    'warped_47': {'fill_rate': 47, 'accuracy': 98.02, 'jpeg_q50': 0.53, 'label': 'Warped 47%'},
    'warped_96': {'fill_rate': 96, 'accuracy': 99.10, 'jpeg_q50': 3.06, 'label': 'Warped 96%\n(RECOMMENDED)'},
    'warped_99': {'fill_rate': 99, 'accuracy': 98.73, 'jpeg_q50': 7.34, 'label': 'Warped 99%'},
}

# Colores
colors = {
    'original_100': '#d62728',  # Rojo
    'warped_47': '#2ca02c',      # Verde
    'warped_96': '#1f77b4',      # Azul (destacado)
    'warped_99': '#ff7f0e',      # Naranja
}

def create_figure():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Preparar datos
    fill_rates = [d['fill_rate'] for d in data.values()]
    accuracies = [d['accuracy'] for d in data.values()]
    jpeg_degradations = [d['jpeg_q50'] for d in data.values()]
    labels = list(data.keys())

    # === Panel 1: Accuracy vs Fill Rate ===
    ax1 = axes[0]
    for key, d in data.items():
        marker = '*' if key == 'warped_96' else 'o'
        size = 200 if key == 'warped_96' else 100
        ax1.scatter(d['fill_rate'], d['accuracy'],
                   c=colors[key], s=size, marker=marker,
                   label=d['label'], zorder=5, edgecolors='black', linewidth=1)

    ax1.set_xlabel('Fill Rate (%)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('A) Classification Accuracy vs Fill Rate')
    ax1.set_xlim(40, 105)
    ax1.set_ylim(97.5, 99.5)
    ax1.axhline(y=99.10, color=colors['warped_96'], linestyle='--', alpha=0.5, label='_nolegend_')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Anotación para warped_96
    ax1.annotate('Best Accuracy', xy=(96, 99.10), xytext=(75, 99.3),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, color='gray')

    # === Panel 2: Robustness vs Fill Rate ===
    ax2 = axes[1]
    for key, d in data.items():
        marker = '*' if key == 'warped_96' else 'o'
        size = 200 if key == 'warped_96' else 100
        ax2.scatter(d['fill_rate'], d['jpeg_q50'],
                   c=colors[key], s=size, marker=marker,
                   label=d['label'], zorder=5, edgecolors='black', linewidth=1)

    ax2.set_xlabel('Fill Rate (%)')
    ax2.set_ylabel('JPEG Q50 Degradation (%)')
    ax2.set_title('B) Robustness to JPEG Compression (Q50)')
    ax2.set_xlim(40, 105)
    ax2.set_ylim(-1, 18)
    ax2.axhline(y=3.06, color=colors['warped_96'], linestyle='--', alpha=0.5, label='_nolegend_')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Anotación: menor es mejor
    ax2.annotate('Lower is better\n↓', xy=(45, 0.53), fontsize=9, color='green', ha='center')

    # Anotación para warped_96
    ax2.annotate('Good balance', xy=(96, 3.06), xytext=(75, 8),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, color='gray')

    # Título general
    fig.suptitle('Fill Rate Trade-off Analysis: Accuracy vs Robustness',
                fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


def create_summary_figure():
    """Crea una figura con tabla resumen."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')

    # Datos para la tabla
    columns = ['Dataset', 'Fill Rate', 'Accuracy', 'JPEG Q50 Deg.', 'Recommendation']
    cell_data = [
        ['Warped 47%', '47%', '98.02%', '0.53%', 'Maximum robustness'],
        ['Warped 96%', '96%', '99.10%', '3.06%', '★ RECOMMENDED'],
        ['Warped 99%', '99%', '98.73%', '7.34%', 'Legacy'],
        ['Original 100%', '100%', '98.84%', '16.14%', 'Baseline'],
    ]

    # Colores de las filas
    row_colors = ['#e8f5e9', '#bbdefb', '#fff3e0', '#ffebee']

    table = ax.table(cellText=cell_data, colLabels=columns,
                    loc='center', cellLoc='center',
                    rowColours=row_colors,
                    colColours=['#f5f5f5'] * 5)

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Destacar la fila recomendada (fila 2, índice 1 en datos)
    for col in range(5):
        table[(2, col)].set_text_props(fontweight='bold')

    ax.set_title('Summary: Fill Rate Trade-off (Session 53)',
                fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig


def main():
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    # Figura principal
    fig1 = create_figure()
    output_path1 = output_dir / 'thesis_figure_tradeoff.png'
    fig1.savefig(output_path1, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Figura guardada: {output_path1}")

    # Figura con tabla resumen
    fig2 = create_summary_figure()
    output_path2 = output_dir / 'thesis_figure_summary_table.png'
    fig2.savefig(output_path2, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Tabla guardada: {output_path2}")

    # También crear versión combinada
    fig3, axes = plt.subplots(2, 2, figsize=(14, 10),
                              gridspec_kw={'height_ratios': [2, 1]})

    # Reutilizar gráficos
    # Panel A: Accuracy
    ax_acc = axes[0, 0]
    for key, d in data.items():
        marker = '*' if key == 'warped_96' else 'o'
        size = 200 if key == 'warped_96' else 100
        ax_acc.scatter(d['fill_rate'], d['accuracy'],
                      c=colors[key], s=size, marker=marker,
                      label=d['label'], zorder=5, edgecolors='black', linewidth=1)
    ax_acc.set_xlabel('Fill Rate (%)')
    ax_acc.set_ylabel('Accuracy (%)')
    ax_acc.set_title('A) Classification Accuracy vs Fill Rate')
    ax_acc.set_xlim(40, 105)
    ax_acc.set_ylim(97.5, 99.5)
    ax_acc.legend(loc='lower right', fontsize=9)
    ax_acc.grid(True, alpha=0.3)

    # Panel B: Robustness
    ax_rob = axes[0, 1]
    for key, d in data.items():
        marker = '*' if key == 'warped_96' else 'o'
        size = 200 if key == 'warped_96' else 100
        ax_rob.scatter(d['fill_rate'], d['jpeg_q50'],
                      c=colors[key], s=size, marker=marker,
                      label=d['label'], zorder=5, edgecolors='black', linewidth=1)
    ax_rob.set_xlabel('Fill Rate (%)')
    ax_rob.set_ylabel('JPEG Q50 Degradation (%)')
    ax_rob.set_title('B) Robustness to JPEG Compression')
    ax_rob.set_xlim(40, 105)
    ax_rob.set_ylim(-1, 18)
    ax_rob.legend(loc='upper right', fontsize=9)
    ax_rob.grid(True, alpha=0.3)

    # Panel C: Composite Score
    ax_score = axes[1, 0]
    scores = {k: d['accuracy'] - d['jpeg_q50'] for k, d in data.items()}
    bars = ax_score.bar(range(len(scores)), scores.values(),
                       color=[colors[k] for k in scores.keys()],
                       edgecolor='black', linewidth=1)
    ax_score.set_xticks(range(len(scores)))
    ax_score.set_xticklabels([data[k]['label'].replace('\n', ' ') for k in scores.keys()],
                            fontsize=9, rotation=15, ha='right')
    ax_score.set_ylabel('Composite Score\n(Accuracy - Degradation)')
    ax_score.set_title('C) Composite Score (Higher is Better)')
    ax_score.axhline(y=scores['warped_96'], color=colors['warped_96'], linestyle='--', alpha=0.5)
    ax_score.grid(True, alpha=0.3, axis='y')

    # Destacar el mejor
    best_idx = list(scores.keys()).index('warped_96')
    bars[best_idx].set_linewidth(3)

    # Panel D: Tabla resumen
    ax_table = axes[1, 1]
    ax_table.axis('off')

    columns = ['Dataset', 'Accuracy', 'JPEG Deg.', 'Score']
    cell_data = [
        ['Warped 47%', '98.02%', '0.53%', f"{scores['warped_47']:.2f}"],
        ['Warped 96%', '99.10%', '3.06%', f"{scores['warped_96']:.2f} ★"],
        ['Warped 99%', '98.73%', '7.34%', f"{scores['warped_99']:.2f}"],
        ['Original', '98.84%', '16.14%', f"{scores['original_100']:.2f}"],
    ]

    table = ax_table.table(cellText=cell_data, colLabels=columns,
                          loc='center', cellLoc='center',
                          colColours=['#f0f0f0'] * 4)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.6)

    # Destacar fila recomendada
    for col in range(4):
        table[(2, col)].set_text_props(fontweight='bold')
        table[(2, col)].set_facecolor('#bbdefb')

    ax_table.set_title('D) Summary Table', fontsize=12)

    fig3.suptitle('Fill Rate Trade-off Analysis for COVID-19 Classification\n'
                 '(Session 53 - Pre-Defense)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path3 = output_dir / 'thesis_figure_combined.png'
    fig3.savefig(output_path3, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Figura combinada guardada: {output_path3}")

    plt.close('all')
    print("\nTodas las figuras generadas exitosamente.")


if __name__ == '__main__':
    main()
