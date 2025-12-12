#!/usr/bin/env python3
"""
Generador de diagramas de arquitectura para tesis.
Sesión 15: Documentación final y visualizaciones.

Genera 4 diagramas principales:
1. Arquitectura completa del modelo
2. Módulo CoordinateAttention
3. Pipeline de ensemble + TTA
4. Pipeline de entrenamiento en 2 fases
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import os

# Configuración global para calidad de publicación
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Colores consistentes para toda la documentación
COLORS = {
    'input': '#E3F2FD',      # Azul muy claro
    'backbone': '#BBDEFB',    # Azul claro
    'attention': '#90CAF9',   # Azul medio
    'head': '#64B5F6',        # Azul
    'output': '#42A5F5',      # Azul oscuro
    'loss': '#FF8A65',        # Naranja
    'preprocess': '#C8E6C9',  # Verde claro
    'ensemble': '#FFF59D',    # Amarillo claro
    'tta': '#CE93D8',         # Púrpura claro
    'arrow': '#455A64',       # Gris azulado
    'text': '#212121',        # Negro
    'border': '#37474F'       # Gris oscuro
}


def draw_block(ax, x, y, width, height, text, color, text_color='black',
               fontsize=10, fontweight='normal', multi_line=False):
    """Dibuja un bloque con texto centrado."""
    rect = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor=color, edgecolor=COLORS['border'],
                          linewidth=1.5)
    ax.add_patch(rect)

    if multi_line and '\n' in text:
        lines = text.split('\n')
        line_height = height / (len(lines) + 1)
        for i, line in enumerate(lines):
            y_pos = y + height/2 - (i + 1) * line_height
            ax.text(x, y_pos, line, ha='center', va='center',
                   fontsize=fontsize, fontweight=fontweight, color=text_color)
    else:
        ax.text(x, y, text, ha='center', va='center',
               fontsize=fontsize, fontweight=fontweight, color=text_color)

    return rect


def draw_arrow(ax, start, end, color=None, style='->', connectionstyle="arc3,rad=0"):
    """Dibuja una flecha entre dos puntos."""
    if color is None:
        color = COLORS['arrow']
    arrow = FancyArrowPatch(start, end, arrowstyle=style,
                            connectionstyle=connectionstyle,
                            color=color, linewidth=2,
                            mutation_scale=15)
    ax.add_patch(arrow)
    return arrow


def generate_model_architecture_diagram(output_dir):
    """
    Genera el diagrama de arquitectura completa del modelo.
    Input → CLAHE → ResNet-18 → CoordAttention → DeepHead → Output
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 9)
    ax.axis('off')
    ax.set_aspect('equal')

    # Título
    ax.text(7, 8.5, 'Arquitectura del Modelo de Predicción de Landmarks',
            ha='center', va='center', fontsize=16, fontweight='bold')

    # Bloques principales - fila superior (pipeline)
    y_main = 6

    # 1. Input
    draw_block(ax, 1, y_main, 2.2, 1.2, 'Imagen\n(224×224×3)',
               COLORS['input'], fontsize=10, multi_line=True)

    # 2. CLAHE
    draw_block(ax, 3.5, y_main, 2, 1.2, 'CLAHE\nPreprocessing',
               COLORS['preprocess'], fontsize=10, multi_line=True)

    # 3. ResNet-18
    draw_block(ax, 6.5, y_main, 2.5, 1.2, 'ResNet-18\n(pretrained)',
               COLORS['backbone'], fontsize=10, multi_line=True)

    # 4. CoordAttention
    draw_block(ax, 9.5, y_main, 2.3, 1.2, 'Coordinate\nAttention',
               COLORS['attention'], fontsize=10, multi_line=True)

    # 5. Deep Head
    draw_block(ax, 12.5, y_main, 2, 1.2, 'Deep Head\n(768→256→30)',
               COLORS['head'], fontsize=10, multi_line=True)

    # Flechas principales
    draw_arrow(ax, (2.1, y_main), (2.5, y_main))
    draw_arrow(ax, (4.5, y_main), (5.25, y_main))
    draw_arrow(ax, (7.75, y_main), (8.35, y_main))
    draw_arrow(ax, (10.65, y_main), (11.5, y_main))

    # Output final
    draw_arrow(ax, (13.5, y_main), (14.2, y_main))
    ax.text(14.5, y_main, '30\ncoords', ha='center', va='center', fontsize=10)

    # Detalles de cada componente - fila inferior
    y_detail = 3

    # CLAHE details
    draw_block(ax, 3.5, y_detail, 2.8, 2,
               'CLAHE\n─────────\nclip_limit: 2.0\ntile_size: 4\nLAB color space',
               COLORS['preprocess'], fontsize=9, multi_line=True)
    draw_arrow(ax, (3.5, y_main - 0.6), (3.5, y_detail + 1), style='-')

    # ResNet-18 details
    draw_block(ax, 6.5, y_detail, 2.8, 2,
               'ResNet-18\n─────────\n11.3M params\nImageNet weights\nOutput: 512-dim',
               COLORS['backbone'], fontsize=9, multi_line=True)
    draw_arrow(ax, (6.5, y_main - 0.6), (6.5, y_detail + 1), style='-')

    # CoordAttention details
    draw_block(ax, 9.5, y_detail, 2.8, 2,
               'CoordAttention\n─────────\nCVPR 2021\nSpatial encoding\nH/W pooling',
               COLORS['attention'], fontsize=9, multi_line=True)
    draw_arrow(ax, (9.5, y_main - 0.6), (9.5, y_detail + 1), style='-')

    # Deep Head details
    draw_block(ax, 12.5, y_detail, 2.8, 2,
               'Deep Head\n─────────\nGroupNorm\nDropout: 0.3\nSigmoid output',
               COLORS['head'], fontsize=9, multi_line=True)
    draw_arrow(ax, (12.5, y_main - 0.6), (12.5, y_detail + 1), style='-')

    # Leyenda de métricas
    ax.text(7, 0.8, 'Mejor Error: 3.71 px (ensemble 4 modelos) | Mejora: 59% vs baseline',
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'model_architecture.png')
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Diagrama de arquitectura guardado: {output_path}")
    return output_path


def generate_coordinate_attention_diagram(output_dir):
    """
    Genera el diagrama detallado del módulo CoordinateAttention.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 11)
    ax.axis('off')
    ax.set_aspect('equal')

    # Título
    ax.text(6, 10.5, 'Módulo Coordinate Attention (Hou et al., CVPR 2021)',
            ha='center', va='center', fontsize=14, fontweight='bold')

    # Input feature map
    draw_block(ax, 6, 9, 2.5, 1, 'Feature Map\n(C×H×W)',
               COLORS['input'], fontsize=10, multi_line=True)

    # Split into two paths
    draw_arrow(ax, (5, 8.5), (3, 7.5))
    draw_arrow(ax, (7, 8.5), (9, 7.5))

    # Path 1: Horizontal pooling
    draw_block(ax, 3, 7, 2.5, 1, 'AdaptiveAvgPool\n(H×1)',
               COLORS['attention'], fontsize=9, multi_line=True)

    # Path 2: Vertical pooling
    draw_block(ax, 9, 7, 2.5, 1, 'AdaptiveAvgPool\n(1×W)',
               COLORS['attention'], fontsize=9, multi_line=True)

    # Concatenate
    draw_arrow(ax, (3, 6.5), (4.5, 5.5))
    draw_arrow(ax, (9, 6.5), (7.5, 5.5))

    draw_block(ax, 6, 5, 3, 1, 'Concatenate & Permute\n(C, H+W, 1)',
               COLORS['backbone'], fontsize=9, multi_line=True)

    # Conv 1x1 (reduce)
    draw_arrow(ax, (6, 4.5), (6, 4))
    draw_block(ax, 6, 3.5, 2.5, 1, 'Conv 1×1\nBN + ReLU',
               COLORS['head'], fontsize=10, multi_line=True)

    # Split
    draw_arrow(ax, (5, 3), (3, 2))
    draw_arrow(ax, (7, 3), (9, 2))

    # Separate convolutions
    draw_block(ax, 3, 1.5, 2.2, 1, 'Conv 1×1\n(Height att.)',
               COLORS['attention'], fontsize=9, multi_line=True)
    draw_block(ax, 9, 1.5, 2.2, 1, 'Conv 1×1\n(Width att.)',
               COLORS['attention'], fontsize=9, multi_line=True)

    # Sigmoid
    draw_arrow(ax, (3, 1), (3, 0))
    draw_arrow(ax, (9, 1), (9, 0))

    ax.text(3, -0.3, 'Sigmoid', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor=COLORS['output'], alpha=0.7))
    ax.text(9, -0.3, 'Sigmoid', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor=COLORS['output'], alpha=0.7))

    # Final multiplication
    draw_arrow(ax, (3, -0.6), (5, -1.2))
    draw_arrow(ax, (9, -0.6), (7, -1.2))

    ax.text(6, -1.5, '⊗ Element-wise Multiply ⊗', ha='center', va='center',
            fontsize=10, fontweight='bold')

    # Output
    draw_block(ax, 6, -2.5, 2.5, 0.8, 'Attended Features',
               COLORS['output'], fontsize=10)

    # Annotations
    ax.text(11.5, 7, 'Captura\ndependencias\nespaciales',
            ha='center', va='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.text(0.5, 7, 'Codifica\nposición\nvertical',
            ha='center', va='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'coordinate_attention.png')
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Diagrama de CoordAttention guardado: {output_path}")
    return output_path


def generate_ensemble_tta_diagram(output_dir):
    """
    Genera el diagrama del proceso de ensemble + TTA.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 11)
    ax.axis('off')
    ax.set_aspect('equal')

    # Título
    ax.text(7, 10.5, 'Pipeline de Inferencia: Ensemble + TTA',
            ha='center', va='center', fontsize=14, fontweight='bold')

    # Input image
    draw_block(ax, 1.5, 8, 2.5, 1.2, 'Imagen\nde Entrada',
               COLORS['input'], fontsize=10, multi_line=True)

    # TTA split
    draw_arrow(ax, (2.75, 8), (4, 8.8))
    draw_arrow(ax, (2.75, 8), (4, 7.2))

    draw_block(ax, 5, 8.8, 2, 0.8, 'Original', COLORS['tta'], fontsize=10)
    draw_block(ax, 5, 7.2, 2, 0.8, 'Flip H', COLORS['tta'], fontsize=10)

    # Models (4 models)
    model_y_positions = [9, 7.5, 6, 4.5]
    model_colors = ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6']

    # Valores validados: Sesion 13 - seeds 321/789 tienen ~4.0 px
    for i, (y, color) in enumerate(zip(model_y_positions, model_colors)):
        seed = [123, 456, 321, 789][i]
        error = [4.05, 4.04, 4.0, 4.0][i]  # S50: valores corregidos
        draw_block(ax, 8, y, 2.2, 1, f'Modelo {i+1}\nseed={seed}\n({error} px)',
                   color, fontsize=8, multi_line=True)

    # Arrows to models
    for y in model_y_positions:
        draw_arrow(ax, (6, 8.5), (6.9, y))
        draw_arrow(ax, (6, 7.5), (6.9, y))

    # TTA label
    ax.text(5, 6, 'TTA\n(×2)', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor=COLORS['tta'], alpha=0.7))

    # Predictions per model
    for i, y in enumerate(model_y_positions):
        draw_arrow(ax, (9.1, y), (10, y))
        ax.text(10.5, y, f'pred_{i+1}', ha='center', va='center', fontsize=9)

    # Average
    draw_arrow(ax, (11, 9), (11.5, 6.75))
    draw_arrow(ax, (11, 7.5), (11.5, 6.75))
    draw_arrow(ax, (11, 6), (11.5, 6.75))
    draw_arrow(ax, (11, 4.5), (11.5, 6.75))

    draw_block(ax, 12.5, 6.75, 2, 1.2, 'Promedio\n(Mean)',
               COLORS['ensemble'], fontsize=10, multi_line=True)

    # Final output
    draw_arrow(ax, (13.5, 6.75), (14.2, 6.75))
    ax.text(14.5, 6.75, '30\ncoords', ha='center', va='center', fontsize=10)

    # Statistics box
    stats_text = """Ensemble de 4 Modelos + TTA
─────────────────────────
• 4 modelos × 2 TTA = 8 predicciones
• Promedio simple (pesos iguales)
• Error final: 3.71 px
• Mejora vs mejor individual: +8%"""

    ax.text(7, 2, stats_text, ha='center', va='center', fontsize=10,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    # Why ensemble works
    why_text = """¿Por qué funciona el Ensemble?
• Errores no correlacionados (diferentes seeds)
• TTA reduce varianza
• Promedio cancela errores aleatorios"""

    ax.text(7, -0.2, why_text, ha='center', va='center', fontsize=9,
            style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ensemble_tta_pipeline.png')
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Diagrama de Ensemble+TTA guardado: {output_path}")
    return output_path


def generate_training_pipeline_diagram(output_dir):
    """
    Genera el diagrama del pipeline de entrenamiento en 2 fases.
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 10)
    ax.axis('off')
    ax.set_aspect('equal')

    # Título
    ax.text(7, 9.5, 'Pipeline de Entrenamiento en Dos Fases',
            ha='center', va='center', fontsize=14, fontweight='bold')

    # Phase 1
    y1 = 7
    draw_block(ax, 2, y1, 3, 1.5, 'FASE 1\nBackbone Congelado',
               COLORS['preprocess'], fontsize=11, fontweight='bold', multi_line=True)

    phase1_details = """• 15 épocas
• Solo entrena Head
• LR: 1e-3
• Batch: 16
• ~19 → 16 px"""
    draw_block(ax, 2, y1 - 2.2, 3, 2.2, phase1_details,
               COLORS['preprocess'], fontsize=9, multi_line=True)

    # Arrow between phases
    draw_arrow(ax, (3.5, y1), (5.5, y1), style='-|>')
    ax.text(4.5, y1 + 0.4, 'checkpoint', ha='center', va='center', fontsize=8, style='italic')

    # Phase 2
    y2 = 7
    draw_block(ax, 7, y2, 3.2, 1.5, 'FASE 2\nFine-tuning Completo',
               COLORS['attention'], fontsize=11, fontweight='bold', multi_line=True)

    phase2_details = """• 100 épocas
• Todas las capas
• LR backbone: 2e-5
• LR head: 2e-4
• CosineAnnealingLR
• Early stop: 15"""
    draw_block(ax, 7, y2 - 2.5, 3.2, 2.7, phase2_details,
               COLORS['attention'], fontsize=9, multi_line=True)

    # Arrow to evaluation
    draw_arrow(ax, (8.6, y2), (10.5, y2), style='-|>')

    # Evaluation
    draw_block(ax, 12, y2, 2.5, 1.5, 'Evaluación\ncon TTA',
               COLORS['output'], fontsize=11, fontweight='bold', multi_line=True)

    eval_details = """• Test set: 96 imgs
• TTA: flip H
• Error: ~4 px"""
    draw_block(ax, 12, y2 - 1.8, 2.5, 1.5, eval_details,
               COLORS['output'], fontsize=9, multi_line=True)

    # Loss function
    draw_block(ax, 7, 2, 3, 1.2, 'Wing Loss\n(normalized)',
               COLORS['loss'], fontsize=11, multi_line=True)

    # Arrows to loss
    draw_arrow(ax, (2, y1 - 3.3), (5.5, 2))
    draw_arrow(ax, (7, y2 - 3.8), (7, 2.6))

    # Data augmentation
    aug_text = """Data Augmentation
─────────────────
• CLAHE (clip=2.0, tile=4)
• Flip H (swap pairs)
• Rotation ±10°
• Color jitter"""

    ax.text(12, 2, aug_text, ha='center', va='center', fontsize=9,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor=COLORS['ensemble'], edgecolor='gray', alpha=0.9))

    # Timeline at bottom
    ax.annotate('', xy=(13, -0.3), xytext=(1, -0.3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.text(7, -0.6, 'Tiempo de entrenamiento (~2 horas/modelo en GPU)',
            ha='center', va='center', fontsize=9, style='italic')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_pipeline.png')
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Diagrama de Training Pipeline guardado: {output_path}")
    return output_path


def generate_data_flow_diagram(output_dir):
    """
    Genera un diagrama de flujo de datos simplificado (alternativo).
    """
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 13)
    ax.axis('off')
    ax.set_aspect('equal')

    # Título
    ax.text(5, 12.5, 'Flujo de Datos del Sistema',
            ha='center', va='center', fontsize=14, fontweight='bold')

    # Bloques verticales
    blocks = [
        (5, 11, 'Imagen Raw\n(299×299)', COLORS['input']),
        (5, 9.5, 'CLAHE\n(LAB space)', COLORS['preprocess']),
        (5, 8, 'Resize\n(224×224)', COLORS['preprocess']),
        (5, 6.5, 'Normalización\n(ImageNet)', COLORS['preprocess']),
        (5, 5, 'ResNet-18\nFeatures (512)', COLORS['backbone']),
        (5, 3.5, 'CoordAttention\n(spatial)', COLORS['attention']),
        (5, 2, 'Deep Head\n(768→256→30)', COLORS['head']),
        (5, 0.5, '15 Landmarks\n(x,y)', COLORS['output']),
    ]

    for x, y, text, color in blocks:
        draw_block(ax, x, y, 3, 1, text, color, fontsize=10, multi_line=True)

    # Flechas entre bloques
    for i in range(len(blocks) - 1):
        y1 = blocks[i][1] - 0.5
        y2 = blocks[i+1][1] + 0.5
        draw_arrow(ax, (5, y1), (5, y2))

    # Dimensiones al lado
    dims = [
        (8, 11, '(299, 299, 3)'),
        (8, 9.5, '(299, 299, 3)'),
        (8, 8, '(224, 224, 3)'),
        (8, 6.5, '(224, 224, 3)'),
        (8, 5, '(512,)'),
        (8, 3.5, '(512,)'),
        (8, 2, '(30,)'),
        (8, 0.5, '(15, 2)'),
    ]

    for x, y, text in dims:
        ax.text(x, y, text, ha='left', va='center', fontsize=9,
                family='monospace', color='gray')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'data_flow.png')
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Diagrama de flujo de datos guardado: {output_path}")
    return output_path


def main():
    """Genera todos los diagramas de arquitectura."""
    output_dir = 'outputs/diagrams'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Generando diagramas de arquitectura para tesis")
    print("=" * 60)

    # Generar todos los diagramas
    diagrams = []

    diagrams.append(generate_model_architecture_diagram(output_dir))
    diagrams.append(generate_coordinate_attention_diagram(output_dir))
    diagrams.append(generate_ensemble_tta_diagram(output_dir))
    diagrams.append(generate_training_pipeline_diagram(output_dir))
    diagrams.append(generate_data_flow_diagram(output_dir))

    print("\n" + "=" * 60)
    print("Diagramas generados exitosamente:")
    print("=" * 60)
    for d in diagrams:
        print(f"  • {d}")

    print(f"\nTotal: {len(diagrams)} diagramas en {output_dir}/")
    print("Resolución: 300 DPI (calidad para publicación)")


if __name__ == '__main__':
    main()
