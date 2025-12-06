#!/usr/bin/env python3
"""
Generador de diagramas de arquitectura detallados.
Sesion 17: Visualizaciones Detalladas del Pipeline.

Este script genera:
1. Arquitectura ResNet-18 detallada con dimensiones
2. Coordinate Attention detallado (flujo interno)
3. Deep Head detallado (capas y activaciones)
4. Wing Loss visualizada (grafico de la funcion)

Las visualizaciones son para usar en la tesis y presentacion de defensa.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import matplotlib.lines as mlines
from pathlib import Path

# Configuracion global de matplotlib
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
})

# Colores consistentes
COLORS = {
    'conv': '#3498db',        # Azul - convoluciones
    'bn': '#9b59b6',          # Morado - batch norm
    'relu': '#e74c3c',        # Rojo - activaciones
    'pool': '#2ecc71',        # Verde - pooling
    'fc': '#f39c12',          # Naranja - fully connected
    'skip': '#1abc9c',        # Turquesa - skip connections
    'attention': '#e91e63',   # Rosa - attention
    'input': '#95a5a6',       # Gris - input/output
    'dropout': '#607d8b',     # Gris azulado - dropout
    'sigmoid': '#ff9800',     # Naranja - sigmoid
}


def draw_layer_box(ax, x, y, width, height, text, color, text_color='white', fontsize=8):
    """Dibuja una caja que representa una capa."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2),
        width, height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=color,
        edgecolor='black',
        linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color,
            wrap=True)
    return box


def draw_arrow(ax, x1, y1, x2, y2, color='black', style='->', lw=1.5):
    """Dibuja una flecha entre dos puntos."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle=style, lw=lw, color=color))


# ============================================================================
# 1. ARQUITECTURA RESNET-18 DETALLADA
# ============================================================================

def generate_resnet18_detailed(output_dir):
    """
    Genera diagrama detallado de ResNet-18 con bloques residuales.
    """
    print("\n" + "="*60)
    print("Generando diagrama de ResNet-18 detallado...")
    print("="*60)

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Titulo
    ax.text(8, 11.5, 'Arquitectura ResNet-18 para Landmark Detection',
           ha='center', va='center', fontsize=14, fontweight='bold')

    # Posiciones verticales
    y_levels = {
        'input': 10.5,
        'conv1': 9.5,
        'pool': 8.5,
        'layer1': 7.0,
        'layer2': 5.5,
        'layer3': 4.0,
        'layer4': 2.5,
        'gap': 1.5,
        'output': 0.5
    }

    # ========== INPUT ==========
    draw_layer_box(ax, 2, y_levels['input'], 3, 0.6, 'Input\n224x224x3', COLORS['input'])

    # ========== CONV1 + BN + ReLU ==========
    draw_layer_box(ax, 2, y_levels['conv1'], 1.5, 0.5, 'Conv2d\n7x7, 64', COLORS['conv'])
    draw_layer_box(ax, 4, y_levels['conv1'], 1.2, 0.5, 'BN', COLORS['bn'])
    draw_layer_box(ax, 5.5, y_levels['conv1'], 1.2, 0.5, 'ReLU', COLORS['relu'])
    ax.text(7, y_levels['conv1'], '112x112x64', fontsize=8, va='center')

    # Flechas
    draw_arrow(ax, 2, y_levels['input']-0.3, 2, y_levels['conv1']+0.3)
    draw_arrow(ax, 2.75, y_levels['conv1'], 3.4, y_levels['conv1'])
    draw_arrow(ax, 4.6, y_levels['conv1'], 4.9, y_levels['conv1'])

    # ========== MAXPOOL ==========
    draw_layer_box(ax, 2, y_levels['pool'], 2, 0.5, 'MaxPool2d\n3x3, stride=2', COLORS['pool'])
    ax.text(4.5, y_levels['pool'], '56x56x64', fontsize=8, va='center')
    draw_arrow(ax, 5.5, y_levels['conv1']-0.3, 2, y_levels['pool']+0.3)

    # ========== HELPER: DRAW RESIDUAL BLOCK ==========
    def draw_residual_block(x_center, y_center, in_ch, out_ch, stride, name, downsample=False):
        """Dibuja un bloque residual con skip connection."""
        block_width = 3.5
        block_height = 1.2

        # Fondo del bloque
        bg = FancyBboxPatch(
            (x_center - block_width/2 - 0.1, y_center - block_height/2 - 0.1),
            block_width + 0.2, block_height + 0.2,
            boxstyle="round,pad=0.02",
            facecolor='#f0f0f0',
            edgecolor='gray',
            linewidth=1,
            linestyle='--'
        )
        ax.add_patch(bg)

        # Conv1
        draw_layer_box(ax, x_center - 1.2, y_center, 0.8, 0.4, f'3x3\n{out_ch}', COLORS['conv'], fontsize=7)
        # BN + ReLU
        draw_layer_box(ax, x_center - 0.3, y_center, 0.5, 0.4, 'BN+R', COLORS['relu'], fontsize=6)
        # Conv2
        draw_layer_box(ax, x_center + 0.5, y_center, 0.8, 0.4, f'3x3\n{out_ch}', COLORS['conv'], fontsize=7)
        # BN
        draw_layer_box(ax, x_center + 1.3, y_center, 0.4, 0.4, 'BN', COLORS['bn'], fontsize=6)

        # Skip connection (flecha curva)
        if downsample:
            # Skip con 1x1 conv
            ax.annotate('', xy=(x_center + 1.5, y_center + 0.3),
                       xytext=(x_center - 1.7, y_center + 0.3),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color=COLORS['skip'],
                                      connectionstyle='arc3,rad=0.3'))
            ax.text(x_center, y_center + 0.7, '1x1', fontsize=6, color=COLORS['skip'],
                   ha='center')
        else:
            ax.annotate('', xy=(x_center + 1.5, y_center + 0.3),
                       xytext=(x_center - 1.7, y_center + 0.3),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color=COLORS['skip'],
                                      connectionstyle='arc3,rad=0.3'))

        # Nombre del bloque
        ax.text(x_center - 1.7, y_center, name, fontsize=7, ha='right', va='center')

        return x_center + block_width/2 + 0.3

    # ========== LAYER 1 (2 bloques, 64 canales) ==========
    ax.text(0.5, y_levels['layer1'], 'Layer1', fontsize=10, fontweight='bold', va='center')
    draw_residual_block(4, y_levels['layer1'], 64, 64, 1, 'Block1')
    draw_residual_block(8.5, y_levels['layer1'], 64, 64, 1, 'Block2')
    ax.text(12, y_levels['layer1'], '56x56x64', fontsize=8, va='center')

    # ========== LAYER 2 (2 bloques, 128 canales) ==========
    ax.text(0.5, y_levels['layer2'], 'Layer2', fontsize=10, fontweight='bold', va='center')
    draw_residual_block(4, y_levels['layer2'], 64, 128, 2, 'Block1', downsample=True)
    draw_residual_block(8.5, y_levels['layer2'], 128, 128, 1, 'Block2')
    ax.text(12, y_levels['layer2'], '28x28x128', fontsize=8, va='center')

    # ========== LAYER 3 (2 bloques, 256 canales) ==========
    ax.text(0.5, y_levels['layer3'], 'Layer3', fontsize=10, fontweight='bold', va='center')
    draw_residual_block(4, y_levels['layer3'], 128, 256, 2, 'Block1', downsample=True)
    draw_residual_block(8.5, y_levels['layer3'], 256, 256, 1, 'Block2')
    ax.text(12, y_levels['layer3'], '14x14x256', fontsize=8, va='center')

    # ========== LAYER 4 (2 bloques, 512 canales) ==========
    ax.text(0.5, y_levels['layer4'], 'Layer4', fontsize=10, fontweight='bold', va='center')
    draw_residual_block(4, y_levels['layer4'], 256, 512, 2, 'Block1', downsample=True)
    draw_residual_block(8.5, y_levels['layer4'], 512, 512, 1, 'Block2')
    ax.text(12, y_levels['layer4'], '7x7x512', fontsize=8, va='center')

    # ========== GLOBAL AVERAGE POOLING ==========
    draw_layer_box(ax, 4, y_levels['gap'], 3, 0.5, 'Global Average Pooling', COLORS['pool'])
    ax.text(7, y_levels['gap'], '1x1x512', fontsize=8, va='center')

    # ========== OUTPUT ==========
    draw_layer_box(ax, 4, y_levels['output'], 3, 0.5, 'Feature Vector', COLORS['fc'])
    ax.text(7, y_levels['output'], '512-dim', fontsize=8, va='center')

    # Flechas verticales entre layers
    draw_arrow(ax, 2, y_levels['pool']-0.3, 2, y_levels['layer1']+0.7)
    for i, (y1, y2) in enumerate([
        (y_levels['layer1'], y_levels['layer2']),
        (y_levels['layer2'], y_levels['layer3']),
        (y_levels['layer3'], y_levels['layer4']),
        (y_levels['layer4'], y_levels['gap']),
        (y_levels['gap'], y_levels['output'])
    ]):
        draw_arrow(ax, 6, y1 - 0.7, 6, y2 + 0.3)

    # Leyenda
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['conv'], edgecolor='black', label='Convolution'),
        mpatches.Patch(facecolor=COLORS['bn'], edgecolor='black', label='BatchNorm'),
        mpatches.Patch(facecolor=COLORS['relu'], edgecolor='black', label='ReLU'),
        mpatches.Patch(facecolor=COLORS['pool'], edgecolor='black', label='Pooling'),
        mlines.Line2D([0], [0], color=COLORS['skip'], lw=2, label='Skip Connection'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8, ncol=2)

    # Guardar
    output_path = os.path.join(output_dir, 'resnet18_detailed.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"  Guardado: {output_path}")
    return output_path


# ============================================================================
# 2. COORDINATE ATTENTION DETALLADO
# ============================================================================

def generate_coord_attention_detailed(output_dir):
    """
    Genera diagrama detallado del modulo Coordinate Attention.
    """
    print("\n" + "="*60)
    print("Generando diagrama de Coordinate Attention...")
    print("="*60)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Titulo
    ax.text(7, 9.5, 'Coordinate Attention Module',
           ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(7, 9.0, '(Hou et al., CVPR 2021)',
           ha='center', va='center', fontsize=10, style='italic', color='gray')

    # Input
    draw_layer_box(ax, 2, 7.5, 2.5, 0.8, 'Input\nH x W x C', COLORS['input'])

    # Pool Horizontal y Vertical (en paralelo)
    draw_layer_box(ax, 5, 8.2, 2.5, 0.7, 'Pool H\n(1, W) \u2192 (H, 1, C)', COLORS['pool'])
    draw_layer_box(ax, 5, 6.8, 2.5, 0.7, 'Pool W\n(H, 1) \u2192 (1, W, C)', COLORS['pool'])

    draw_arrow(ax, 3.25, 7.8, 3.75, 8.2)
    draw_arrow(ax, 3.25, 7.2, 3.75, 6.8)

    # Concatenate
    draw_layer_box(ax, 8, 7.5, 2, 0.8, 'Concat\n(H+W, 1, C)', COLORS['attention'])
    draw_arrow(ax, 6.25, 8.0, 7, 7.7)
    draw_arrow(ax, 6.25, 7.0, 7, 7.3)

    # Conv 1x1 (reduccion)
    draw_layer_box(ax, 8, 6.0, 2.5, 0.7, 'Conv 1x1\nC \u2192 C/r', COLORS['conv'])
    ax.text(9.5, 6.0, 'r=32', fontsize=8, va='center', color='gray')
    draw_arrow(ax, 8, 7.1, 8, 6.4)

    # BatchNorm + ReLU
    draw_layer_box(ax, 8, 5.0, 1.5, 0.6, 'BN+ReLU', COLORS['relu'])
    draw_arrow(ax, 8, 5.65, 8, 5.3)

    # Split
    draw_layer_box(ax, 8, 4.0, 1.5, 0.6, 'Split', COLORS['attention'])
    draw_arrow(ax, 8, 4.7, 8, 4.3)

    # Dos Conv 1x1 (expansion)
    draw_layer_box(ax, 5.5, 3.0, 2, 0.6, 'Conv 1x1\nC/r \u2192 C', COLORS['conv'])
    draw_layer_box(ax, 10.5, 3.0, 2, 0.6, 'Conv 1x1\nC/r \u2192 C', COLORS['conv'])
    draw_arrow(ax, 7.4, 3.8, 5.5, 3.3)
    draw_arrow(ax, 8.6, 3.8, 10.5, 3.3)

    # Sigmoid
    draw_layer_box(ax, 5.5, 2.0, 1.5, 0.6, 'Sigmoid', COLORS['sigmoid'])
    draw_layer_box(ax, 10.5, 2.0, 1.5, 0.6, 'Sigmoid', COLORS['sigmoid'])
    draw_arrow(ax, 5.5, 2.7, 5.5, 2.3)
    draw_arrow(ax, 10.5, 2.7, 10.5, 2.3)

    # Etiquetas de atencion
    ax.text(5.5, 1.3, 'Attention H\n(H, 1, C)', ha='center', fontsize=8, style='italic')
    ax.text(10.5, 1.3, 'Attention W\n(1, W, C)', ha='center', fontsize=8, style='italic')

    # Multiplicacion con input
    draw_layer_box(ax, 8, 0.5, 2, 0.7, 'Multiply\nInput \u00d7 AttH \u00d7 AttW', COLORS['attention'])

    # Flechas de multiplicacion
    ax.annotate('', xy=(7.2, 0.8), xytext=(5.5, 1.5),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='gray',
                             connectionstyle='arc3,rad=0.2'))
    ax.annotate('', xy=(8.8, 0.8), xytext=(10.5, 1.5),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='gray',
                             connectionstyle='arc3,rad=-0.2'))

    # Conexion del input a la multiplicacion
    ax.annotate('', xy=(2, 6.9), xytext=(2, 7.1),
               arrowprops=dict(arrowstyle='-', lw=1.5, color=COLORS['skip']))
    ax.plot([2, 2, 1, 1, 7], [6.9, 0.5, 0.5, 0.5, 0.5], color=COLORS['skip'],
           linestyle='--', linewidth=1.5)
    ax.annotate('', xy=(7, 0.5), xytext=(6.5, 0.5),
               arrowprops=dict(arrowstyle='->', lw=1.5, color=COLORS['skip']))

    # Output
    ax.text(8, -0.2, 'Output: H x W x C\n(con informacion posicional)',
           ha='center', fontsize=9, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Leyenda explicativa
    explanation = """
    Coordinate Attention captura dependencias de largo alcance
    en direcciones horizontal y vertical por separado.

    Ventajas:
    - Preserva informacion posicional precisa
    - Computacionalmente eficiente
    - Mejora localizacion de landmarks
    """
    ax.text(13, 6, explanation.strip(), fontsize=8, va='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Guardar
    output_path = os.path.join(output_dir, 'coord_attention_detailed.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"  Guardado: {output_path}")
    return output_path


# ============================================================================
# 3. DEEP HEAD DETALLADO
# ============================================================================

def generate_deep_head_detailed(output_dir):
    """
    Genera diagrama detallado del Deep Head con capas y activaciones.
    """
    print("\n" + "="*60)
    print("Generando diagrama de Deep Head...")
    print("="*60)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Titulo
    ax.text(6, 7.5, 'Deep Head para Regression de Landmarks',
           ha='center', va='center', fontsize=14, fontweight='bold')

    # Posiciones verticales
    y_pos = [6.5, 5.5, 4.5, 3.5, 2.5, 1.5, 0.5]
    x_center = 6

    # Input
    draw_layer_box(ax, x_center, y_pos[0], 3, 0.6, 'Input: 512-dim', COLORS['input'])
    ax.text(9, y_pos[0], 'Feature vector\ndel backbone', fontsize=8, va='center', color='gray')

    # Linear 512 -> 768
    draw_layer_box(ax, x_center, y_pos[1], 3, 0.6, 'Linear(512 \u2192 768)', COLORS['fc'])
    draw_arrow(ax, x_center, y_pos[0]-0.3, x_center, y_pos[1]+0.3)

    # GroupNorm + ReLU + Dropout
    draw_layer_box(ax, 4, y_pos[2], 2, 0.5, 'GroupNorm(32)', COLORS['bn'])
    draw_layer_box(ax, 6, y_pos[2], 1.5, 0.5, 'ReLU', COLORS['relu'])
    draw_layer_box(ax, 8, y_pos[2], 2, 0.5, 'Dropout(0.3)', COLORS['dropout'])
    draw_arrow(ax, x_center, y_pos[1]-0.3, x_center, y_pos[2]+0.3)

    # Nota sobre GroupNorm
    ax.text(10.5, y_pos[2], 'GroupNorm es mas\nestable que BatchNorm\ncon batch pequeño',
           fontsize=7, va='center', color='gray', style='italic')

    # Linear 768 -> 256
    draw_layer_box(ax, x_center, y_pos[3], 3, 0.6, 'Linear(768 \u2192 256)', COLORS['fc'])
    draw_arrow(ax, x_center, y_pos[2]-0.3, x_center, y_pos[3]+0.3)

    # GroupNorm + ReLU + Dropout
    draw_layer_box(ax, 4, y_pos[4], 2, 0.5, 'GroupNorm(16)', COLORS['bn'])
    draw_layer_box(ax, 6, y_pos[4], 1.5, 0.5, 'ReLU', COLORS['relu'])
    draw_layer_box(ax, 8, y_pos[4], 2, 0.5, 'Dropout(0.3)', COLORS['dropout'])
    draw_arrow(ax, x_center, y_pos[3]-0.3, x_center, y_pos[4]+0.3)

    # Linear 256 -> 30
    draw_layer_box(ax, x_center, y_pos[5], 3, 0.6, 'Linear(256 \u2192 30)', COLORS['fc'])
    draw_arrow(ax, x_center, y_pos[4]-0.3, x_center, y_pos[5]+0.3)

    # Sigmoid
    draw_layer_box(ax, x_center, y_pos[6], 2.5, 0.6, 'Sigmoid', COLORS['sigmoid'])
    draw_arrow(ax, x_center, y_pos[5]-0.3, x_center, y_pos[6]+0.3)

    # Output
    ax.text(x_center, -0.1, 'Output: 30 valores en [0, 1]\n(15 landmarks x 2 coordenadas)',
           ha='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Conteo de parametros
    params_text = """
    Parametros:
    - Linear(512\u2192768): 394,752
    - Linear(768\u2192256): 196,864
    - Linear(256\u219230): 7,710
    - Total Head: ~599,326
    """
    ax.text(0.5, 3, params_text.strip(), fontsize=8, va='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Guardar
    output_path = os.path.join(output_dir, 'deep_head_detailed.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"  Guardado: {output_path}")
    return output_path


# ============================================================================
# 4. WING LOSS VISUALIZADA
# ============================================================================

def generate_wing_loss_detailed(output_dir):
    """
    Genera visualizacion de la funcion Wing Loss con comparacion MSE/L1.
    """
    print("\n" + "="*60)
    print("Generando visualizacion de Wing Loss...")
    print("="*60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Parametros de Wing Loss
    omega = 10.0
    epsilon = 2.0
    C = omega - omega * np.log(1 + omega/epsilon)

    # Rango de errores
    x = np.linspace(0, 20, 500)

    # ========== GRAFICO 1: Wing Loss vs MSE vs L1 ==========
    ax = axes[0]

    # Wing Loss
    wing_loss = np.where(
        x < omega,
        omega * np.log(1 + x / epsilon),
        x - C
    )

    # MSE (escalado)
    mse_loss = 0.1 * x ** 2

    # L1 Loss
    l1_loss = x

    # Smooth L1 (Huber)
    delta = 1.0
    smooth_l1 = np.where(x < delta, 0.5 * x**2, delta * (x - 0.5 * delta))

    ax.plot(x, wing_loss, 'b-', linewidth=2.5, label='Wing Loss')
    ax.plot(x, mse_loss, 'r--', linewidth=2, label='MSE (0.1x$^2$)')
    ax.plot(x, l1_loss, 'g-.', linewidth=2, label='L1')
    ax.plot(x, smooth_l1, 'm:', linewidth=2, label='Smooth L1')

    # Marcar region de transicion
    ax.axvline(x=omega, color='gray', linestyle='--', alpha=0.5)
    ax.text(omega + 0.3, 15, f'\u03c9={omega}', fontsize=9, color='gray')

    ax.set_xlabel('|Error| (pixels)', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_title('Comparacion de Loss Functions', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)

    # ========== GRAFICO 2: Gradientes ==========
    ax = axes[1]

    # Gradientes
    dx = x[1] - x[0]
    grad_wing = np.gradient(wing_loss, dx)
    grad_mse = np.gradient(mse_loss, dx)
    grad_l1 = np.ones_like(x)

    ax.plot(x, grad_wing, 'b-', linewidth=2.5, label='Wing Loss')
    ax.plot(x, grad_mse, 'r--', linewidth=2, label='MSE')
    ax.plot(x, grad_l1, 'g-.', linewidth=2, label='L1')

    ax.axvline(x=omega, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('|Error| (pixels)', fontsize=10)
    ax.set_ylabel('Gradiente (dL/dx)', fontsize=10)
    ax.set_title('Gradientes de Loss Functions', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 2.5)

    # Anotaciones
    ax.annotate('Mayor sensibilidad\na errores pequeños',
               xy=(2, 1.5), xytext=(5, 2.0),
               fontsize=8, arrowprops=dict(arrowstyle='->', color='blue'))
    ax.annotate('Region lineal\n(igual que L1)',
               xy=(15, 1.0), xytext=(12, 0.5),
               fontsize=8, arrowprops=dict(arrowstyle='->', color='blue'))

    # ========== GRAFICO 3: Formula y explicacion ==========
    ax = axes[2]
    ax.axis('off')

    formula = r"""
    $\mathbf{Wing\ Loss}$

    $L(x) = \begin{cases}
    \omega \cdot \ln(1 + \frac{|x|}{\epsilon}) & |x| < \omega \\
    |x| - C & \text{otherwise}
    \end{cases}$

    donde $C = \omega - \omega \cdot \ln(1 + \frac{\omega}{\epsilon})$

    $\mathbf{Parametros:}$
    $\omega = 10$ (umbral de transicion)
    $\epsilon = 2$ (curvatura region log)

    $\mathbf{Ventajas:}$
    1. Mayor sensibilidad a errores pequeños
       (importante para landmarks)
    2. Transicion suave log → lineal
    3. No explota con outliers (como MSE)
    4. Mejor que L1 para precision fina
    """

    ax.text(0.1, 0.95, formula, fontsize=10, va='top',
           family='serif', transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.suptitle('Wing Loss: Funcion de Perdida para Deteccion de Landmarks',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Guardar
    output_path = os.path.join(output_dir, 'wing_loss_detailed.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"  Guardado: {output_path}")
    return output_path


# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

def main():
    """Genera todos los diagramas detallados."""

    output_dir = 'outputs/pipeline_viz/diagrams'
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print(" GENERADOR DE DIAGRAMAS DETALLADOS")
    print(" Sesion 17 - Tesis de Maestria")
    print("="*60)

    figures = []

    # 1. ResNet-18 detallado
    try:
        result = generate_resnet18_detailed(output_dir)
        if result:
            figures.append(result)
    except Exception as e:
        print(f"Error en ResNet-18: {e}")
        import traceback
        traceback.print_exc()

    # 2. Coordinate Attention detallado
    try:
        result = generate_coord_attention_detailed(output_dir)
        if result:
            figures.append(result)
    except Exception as e:
        print(f"Error en CoordAttention: {e}")
        import traceback
        traceback.print_exc()

    # 3. Deep Head detallado
    try:
        result = generate_deep_head_detailed(output_dir)
        if result:
            figures.append(result)
    except Exception as e:
        print(f"Error en Deep Head: {e}")
        import traceback
        traceback.print_exc()

    # 4. Wing Loss detallado
    try:
        result = generate_wing_loss_detailed(output_dir)
        if result:
            figures.append(result)
    except Exception as e:
        print(f"Error en Wing Loss: {e}")
        import traceback
        traceback.print_exc()

    # Resumen
    print("\n" + "="*60)
    print(" RESUMEN DE DIAGRAMAS GENERADOS")
    print("="*60)

    for fig_path in figures:
        print(f"  \u2713 {fig_path}")

    print(f"\nTotal: {len(figures)} diagramas")
    print(f"Directorio: {output_dir}/")

    return figures


if __name__ == '__main__':
    main()
