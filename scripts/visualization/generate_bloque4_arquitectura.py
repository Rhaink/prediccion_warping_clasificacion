#!/usr/bin/env python3
"""
Script: Visualizaciones Bloque 4 - ARQUITECTURA (Slides 17-22)
Estilo: v2_profesional

ESTRUCTURA:
1. Assets individuales (objetos reutilizables):
   - assets/diagramas/ -> Diagramas de arquitectura
   - assets/graficas/  -> Gráficas de loss y gradientes

2. Composiciones de slides (usando assets):
   - v2_profesional/slide17_*.png ... slide22_*.png

Slides:
- 17: La arquitectura combina ResNet-18 con módulos especializados
- 18: ResNet-18 extrae características jerárquicas
- 19: Coordinate Attention captura dependencias espaciales
- 20: La cabeza de regresión produce 30 coordenadas
- 21: Wing Loss amplifica gradientes para errores pequeños
- 22: Entrenamiento en 2 fases protege features pre-entrenadas
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN ESTILO v2_PROFESIONAL
# ============================================================================

COLORS = {
    'text_primary': '#1a1a2e',
    'text_secondary': '#4a4a4a',
    'background': '#ffffff',
    'background_alt': '#f8f9fa',
    'accent_primary': '#003366',
    'accent_secondary': '#0066cc',
    'accent_light': '#e8f4fc',
    'data_1': '#003366',
    'data_2': '#2d6a4f',
    'data_3': '#cc6600',
    'data_4': '#7b2cbf',
    'success': '#2d6a4f',
    'warning': '#b07d2b',
    'danger': '#c44536',
    # Arquitectura
    'input': '#e8f4fc',
    'backbone': '#cce5ff',
    'attention': '#d4edda',
    'head': '#fff3cd',
    'output': '#f8d7da',
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'serif'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'text.color': COLORS['text_primary'],
    'figure.facecolor': COLORS['background'],
    'axes.facecolor': COLORS['background'],
})

# Rutas
BASE_DIR = Path('/home/donrobot/Projects/prediccion_coordenadas')
OUTPUT_DIR = BASE_DIR / 'presentacion' / '04_arquitectura'
ASSETS_DIAGRAMAS = OUTPUT_DIR / 'assets' / 'diagramas'
ASSETS_GRAFICAS = OUTPUT_DIR / 'assets' / 'graficas'
SLIDES_DIR = OUTPUT_DIR / 'v2_profesional'

DPI = 100


def create_directories():
    ASSETS_DIAGRAMAS.mkdir(parents=True, exist_ok=True)
    ASSETS_GRAFICAS.mkdir(parents=True, exist_ok=True)
    SLIDES_DIR.mkdir(parents=True, exist_ok=True)


def draw_box(ax, x, y, w, h, label, sublabel=None, color='#cce5ff',
             border_color='#003366', fontsize=10):
    """Dibuja caja para diagramas."""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                         facecolor=color, edgecolor=border_color, linewidth=1.5)
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w/2, y + h*0.65, label, ha='center', va='center',
               fontsize=fontsize, fontweight='bold', color=COLORS['text_primary'])
        ax.text(x + w/2, y + h*0.35, sublabel, ha='center', va='center',
               fontsize=fontsize-2, color=COLORS['text_secondary'])
    else:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
               fontsize=fontsize, fontweight='bold', color=COLORS['text_primary'])


def draw_arrow(ax, x1, y1, x2, y2, color='#003366'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                            mutation_scale=15, color=color, lw=2)
    ax.add_patch(arrow)


# ============================================================================
# PARTE 1: ASSETS INDIVIDUALES
# ============================================================================

def asset_pipeline_arquitectura():
    """Asset: Pipeline completo de arquitectura (sin título)."""
    print("  -> Generando asset: pipeline_arquitectura.png")

    fig, ax = plt.subplots(figsize=(12, 4), facecolor=COLORS['background'])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')

    box_h = 1.4
    box_y = 0.8

    # Componentes
    draw_box(ax, 0.1, box_y, 1.1, box_h, 'Imagen', '224×224×3',
             color=COLORS['input'], border_color=COLORS['accent_primary'])
    draw_box(ax, 1.6, box_y, 1.1, box_h, 'CLAHE', 'Preproceso',
             color=COLORS['input'], border_color=COLORS['accent_primary'])
    draw_box(ax, 3.1, box_y, 1.4, box_h, 'ResNet-18', '11.3M params',
             color=COLORS['backbone'], border_color=COLORS['accent_primary'])
    draw_box(ax, 4.9, box_y, 1.4, box_h, 'Coordinate', 'Attention',
             color=COLORS['attention'], border_color=COLORS['data_2'])
    draw_box(ax, 6.7, box_y, 1.2, box_h, 'Deep Head', '768→256→30',
             color=COLORS['head'], border_color=COLORS['data_3'])
    draw_box(ax, 8.3, box_y, 0.9, box_h, '30', 'coords',
             color=COLORS['output'], border_color=COLORS['danger'])

    # Flechas
    arrow_y = box_y + box_h/2
    for x1, x2 in [(1.2, 1.6), (2.7, 3.1), (4.5, 4.9), (6.3, 6.7), (7.9, 8.3)]:
        draw_arrow(ax, x1, arrow_y, x2, arrow_y)

    # Detalles
    details = [(0.65, 'JPEG'), (2.15, 'clip=2.0'), (3.8, 'ImageNet'),
               (5.6, 'H/W pool'), (7.3, 'Dropout'), (8.75, '[0,1]')]
    for x, txt in details:
        ax.text(x, box_y - 0.15, txt, ha='center', fontsize=8,
               color=COLORS['text_secondary'])

    plt.tight_layout()
    plt.savefig(ASSETS_DIAGRAMAS / 'pipeline_arquitectura.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def asset_resnet18_capas():
    """Asset: Capas de ResNet-18 con dimensiones."""
    print("  -> Generando asset: resnet18_capas.png")

    fig, ax = plt.subplots(figsize=(11, 4), facecolor=COLORS['background'])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.axis('off')

    layers = [
        ('Conv1', '64ch', '112×112', 0.3, 2.2),
        ('Layer1', '64ch', '56×56', 1.8, 1.8),
        ('Layer2', '128ch', '28×28', 3.3, 1.4),
        ('Layer3', '256ch', '14×14', 4.8, 1.0),
        ('Layer4', '512ch', '7×7', 6.3, 0.7),
        ('GAP', '512ch', '1×1', 7.8, 0.5),
    ]

    prev_x, prev_w = None, 0
    for name, ch, size, x, h_scale in layers:
        h = h_scale * 1.3
        y = 1.5 - h/2
        w = 1.0
        depth = layers.index((name, ch, size, x, h_scale))
        color = plt.cm.Blues(0.35 + depth * 0.1)

        # Texto blanco para capas oscuras (depth >= 4)
        text_color = 'white' if depth >= 4 else COLORS['text_primary']
        sub_color = '#cccccc' if depth >= 4 else COLORS['text_secondary']

        # Dibujar caja manualmente para controlar color de texto
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                             facecolor=color, edgecolor=COLORS['accent_primary'], linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.65, name, ha='center', va='center',
               fontsize=9, fontweight='bold', color=text_color)
        ax.text(x + w/2, y + h*0.35, f'{ch}\n{size}', ha='center', va='center',
               fontsize=7, color=sub_color, linespacing=1.0)

        if prev_x is not None:
            draw_arrow(ax, prev_x + prev_w, 1.5, x, 1.5)
        prev_x, prev_w = x, w

    # Leyenda
    ax.text(5.0, 3.2, 'ResNet-18: Reducción progresiva de resolución, aumento de canales',
           ha='center', fontsize=10, color=COLORS['accent_primary'], fontweight='bold')

    plt.tight_layout()
    plt.savefig(ASSETS_DIAGRAMAS / 'resnet18_capas.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def asset_coordinate_attention():
    """Asset: Diagrama del módulo Coordinate Attention."""
    print("  -> Generando asset: coordinate_attention.png")

    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor=COLORS['background'])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.axis('off')

    # Input
    draw_box(ax, 0.3, 2.5, 1.2, 1.4, 'Feature\nMap', 'C×H×W',
             color=COLORS['backbone'], border_color=COLORS['accent_primary'])

    # Pooling branches
    draw_box(ax, 2.3, 4.0, 1.6, 0.9, 'AvgPool (H×1)', '',
             color=COLORS['attention'], border_color=COLORS['data_2'], fontsize=9)
    draw_box(ax, 2.3, 1.5, 1.6, 0.9, 'AvgPool (1×W)', '',
             color=COLORS['attention'], border_color=COLORS['data_2'], fontsize=9)

    # Concat + Conv
    draw_box(ax, 4.5, 2.5, 1.4, 1.4, 'Concat +\nConv 1×1', 'BN+ReLU',
             color=COLORS['input'], border_color=COLORS['accent_primary'], fontsize=9)

    # Split convs
    draw_box(ax, 6.5, 4.0, 1.2, 0.9, 'Conv 1×1', 'Height',
             color=COLORS['head'], border_color=COLORS['data_3'], fontsize=9)
    draw_box(ax, 6.5, 1.5, 1.2, 0.9, 'Conv 1×1', 'Width',
             color=COLORS['head'], border_color=COLORS['data_3'], fontsize=9)

    # Sigmoid
    ax.text(8.0, 4.45, 'σ', fontsize=16, ha='center', fontweight='bold', color=COLORS['data_3'])
    ax.text(8.0, 1.95, 'σ', fontsize=16, ha='center', fontweight='bold', color=COLORS['data_3'])

    # Multiply
    ax.text(8.7, 3.2, '⊗', fontsize=24, ha='center', fontweight='bold', color=COLORS['accent_primary'])

    # Output
    draw_box(ax, 9.0, 2.5, 0.8, 1.4, 'Output', '',
             color=COLORS['output'], border_color=COLORS['danger'], fontsize=9)

    # Flechas
    draw_arrow(ax, 1.5, 3.2, 2.3, 4.4, COLORS['data_2'])
    draw_arrow(ax, 1.5, 3.2, 2.3, 2.0, COLORS['data_2'])
    draw_arrow(ax, 3.9, 4.4, 4.5, 3.5, COLORS['accent_primary'])
    draw_arrow(ax, 3.9, 2.0, 4.5, 2.9, COLORS['accent_primary'])
    draw_arrow(ax, 5.9, 3.5, 6.5, 4.4, COLORS['data_3'])
    draw_arrow(ax, 5.9, 2.9, 6.5, 2.0, COLORS['data_3'])
    draw_arrow(ax, 7.7, 4.4, 8.5, 3.5, COLORS['accent_primary'])
    draw_arrow(ax, 7.7, 2.0, 8.5, 2.9, COLORS['accent_primary'])
    draw_arrow(ax, 8.85, 3.2, 9.0, 3.2, COLORS['accent_primary'])

    # Etiquetas (movidas hacia adentro del diagrama)
    ax.text(3.1, 5.2, 'Codifica posición Y', fontsize=8, ha='center',
           color=COLORS['data_2'], style='italic')
    ax.text(3.1, 1.0, 'Codifica posición X', fontsize=8, ha='center',
           color=COLORS['data_2'], style='italic')

    plt.tight_layout()
    plt.savefig(ASSETS_DIAGRAMAS / 'coordinate_attention.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def asset_deep_head():
    """Asset: Arquitectura del Deep Head."""
    print("  -> Generando asset: deep_head.png")

    fig, ax = plt.subplots(figsize=(11, 3.5), facecolor=COLORS['background'])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')

    layers = [
        ('Input', '512', 0.3, COLORS['backbone']),
        ('Linear', '768', 1.6, COLORS['head']),
        ('GN+GELU', '', 2.8, COLORS['input']),
        ('Dropout', '0.3', 3.9, '#fff3cd'),
        ('Linear', '256', 5.0, COLORS['head']),
        ('GN+GELU', '', 6.1, COLORS['input']),
        ('Linear', '30', 7.2, COLORS['head']),
        ('Sigmoid', '', 8.3, COLORS['attention']),
        ('Output', '[0,1]', 9.2, COLORS['output']),
    ]

    prev_x = None
    for name, dim, x, color in layers:
        w = 0.85
        h = 1.2
        y = 0.9
        draw_box(ax, x, y, w, h, name, dim if dim else None, color=color,
                border_color=COLORS['accent_primary'], fontsize=9)
        if prev_x is not None:
            draw_arrow(ax, prev_x + 0.85, 1.5, x, 1.5)
        prev_x = x

    # Leyenda
    ax.text(5.0, 2.6, 'Deep Head: 512 → 768 → 256 → 30 (coordenadas normalizadas)',
           ha='center', fontsize=10, fontweight='bold', color=COLORS['accent_primary'])

    plt.tight_layout()
    plt.savefig(ASSETS_DIAGRAMAS / 'deep_head.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def asset_wing_loss_comparacion():
    """Asset: Gráfica comparativa de Wing Loss vs otras losses."""
    print("  -> Generando asset: wing_loss_comparacion.png")

    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=COLORS['background'])

    x = np.linspace(0, 20, 500)
    omega, epsilon = 10, 2
    C = omega - omega * np.log(1 + omega/epsilon)

    wing = np.where(np.abs(x) < omega, omega * np.log(1 + np.abs(x)/epsilon), np.abs(x) - C)
    l1 = np.abs(x)
    mse = 0.1 * x**2
    smooth_l1 = np.where(np.abs(x) < 1, 0.5 * x**2, np.abs(x) - 0.5)

    ax.plot(x, wing, color=COLORS['accent_primary'], linewidth=2.5, label='Wing Loss')
    ax.plot(x, mse, '--', color=COLORS['danger'], linewidth=1.5, label='MSE (0.1x²)')
    ax.plot(x, l1, '-.', color=COLORS['data_2'], linewidth=1.5, label='L1')
    ax.plot(x, smooth_l1, ':', color=COLORS['data_4'], linewidth=1.5, label='Smooth L1')

    ax.axvline(x=omega, color=COLORS['text_secondary'], linestyle='--', alpha=0.5)
    ax.text(omega + 0.5, 17, f'ω={omega}', fontsize=9, color=COLORS['text_secondary'])

    ax.set_xlabel('|Error| (píxeles)', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_title('Comparación de Funciones de Pérdida', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ASSETS_GRAFICAS / 'wing_loss_comparacion.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def asset_wing_loss_gradientes():
    """Asset: Gráfica de gradientes de Wing Loss."""
    print("  -> Generando asset: wing_loss_gradientes.png")

    fig, ax = plt.subplots(figsize=(5, 4), facecolor=COLORS['background'])

    x = np.linspace(0.1, 20, 500)
    omega, epsilon = 10, 2

    wing_grad = np.where(x < omega, omega / (epsilon + x), 1.0)
    l1_grad = np.ones_like(x)
    mse_grad = 0.2 * x

    ax.plot(x, wing_grad, color=COLORS['accent_primary'], linewidth=2.5, label='Wing')
    ax.plot(x, l1_grad, '-.', color=COLORS['data_2'], linewidth=1.5, label='L1')
    ax.plot(x, mse_grad, '--', color=COLORS['danger'], linewidth=1.5, label='MSE')

    ax.axvline(x=omega, color=COLORS['text_secondary'], linestyle='--', alpha=0.5)
    ax.axvspan(0, omega, alpha=0.1, color=COLORS['accent_primary'])
    ax.text(5, 2.0, 'Mayor\ngradiente', ha='center', fontsize=9,
           color=COLORS['accent_primary'], fontweight='bold')

    ax.set_xlabel('|Error| (píxeles)', fontsize=10)
    ax.set_ylabel('|Gradiente|', fontsize=10)
    ax.set_title('Gradientes de Loss', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 2.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ASSETS_GRAFICAS / 'wing_loss_gradientes.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def asset_training_2fases():
    """Asset: Diagrama de entrenamiento en 2 fases."""
    print("  -> Generando asset: training_2fases.png")

    fig, ax = plt.subplots(figsize=(11, 5), facecolor=COLORS['background'])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Fase 1
    fase1 = FancyBboxPatch((0.3, 1.2), 2.8, 3.0, boxstyle="round,pad=0.02",
                            facecolor='#e3f2fd', edgecolor=COLORS['accent_primary'], linewidth=2)
    ax.add_patch(fase1)
    ax.text(1.7, 3.9, 'FASE 1', ha='center', fontsize=12, fontweight='bold',
           color=COLORS['accent_primary'])
    ax.text(1.7, 3.5, 'Backbone Congelado', ha='center', fontsize=10,
           color=COLORS['accent_primary'])

    fase1_items = ['15 épocas', 'Solo entrena Head', 'LR: 1e-3', 'Batch: 16']
    for i, item in enumerate(fase1_items):
        ax.text(0.6, 3.0 - i*0.45, f'• {item}', fontsize=9, color=COLORS['text_secondary'])

    # Flecha
    draw_arrow(ax, 3.1, 2.7, 3.8, 2.7, COLORS['accent_primary'])
    ax.text(3.45, 3.0, 'checkpoint', ha='center', fontsize=8, style='italic',
           color=COLORS['text_secondary'])

    # Fase 2
    fase2 = FancyBboxPatch((3.8, 1.2), 2.8, 3.0, boxstyle="round,pad=0.02",
                            facecolor='#e8f5e9', edgecolor=COLORS['data_2'], linewidth=2)
    ax.add_patch(fase2)
    ax.text(5.2, 3.9, 'FASE 2', ha='center', fontsize=12, fontweight='bold',
           color=COLORS['data_2'])
    ax.text(5.2, 3.5, 'Fine-tuning Completo', ha='center', fontsize=10,
           color=COLORS['data_2'])

    fase2_items = ['100 épocas', 'Todas las capas', 'LR backbone: 2e-5', 'LR head: 2e-4']
    for i, item in enumerate(fase2_items):
        ax.text(4.1, 3.0 - i*0.45, f'• {item}', fontsize=9, color=COLORS['text_secondary'])

    # Flecha
    draw_arrow(ax, 6.6, 2.7, 7.3, 2.7, COLORS['data_2'])

    # Evaluación
    eval_box = FancyBboxPatch((7.3, 1.5), 2.2, 2.4, boxstyle="round,pad=0.02",
                               facecolor='#fff3e0', edgecolor=COLORS['data_3'], linewidth=2)
    ax.add_patch(eval_box)
    ax.text(8.4, 3.5, 'EVALUACIÓN', ha='center', fontsize=11, fontweight='bold',
           color=COLORS['data_3'])

    eval_items = ['Test: 96 imgs', 'TTA: flip H', 'Error: ~4 px']
    for i, item in enumerate(eval_items):
        ax.text(7.6, 3.0 - i*0.45, f'• {item}', fontsize=9, color=COLORS['text_secondary'])

    # Info inferior
    info_box = FancyBboxPatch((0.3, 0.2), 5.5, 0.8, boxstyle="round,pad=0.02",
                               facecolor=COLORS['background_alt'],
                               edgecolor=COLORS['text_secondary'], linewidth=1)
    ax.add_patch(info_box)
    ax.text(3.05, 0.6, 'Learning Rate Diferencial: Backbone 2e-5 | Head 2e-4 (10×)',
           ha='center', fontsize=9, color=COLORS['text_primary'])

    plt.tight_layout()
    plt.savefig(ASSETS_DIAGRAMAS / 'training_2fases.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def asset_formula_wing_loss():
    """Asset: Fórmula y parámetros de Wing Loss."""
    print("  -> Generando asset: formula_wing_loss.png")

    fig, ax = plt.subplots(figsize=(4, 5), facecolor=COLORS['background'])
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.95, 'Wing Loss', ha='center', fontsize=14, fontweight='bold',
           color=COLORS['accent_primary'])

    # Fórmula
    formula_box = FancyBboxPatch((0.05, 0.62), 0.9, 0.28, boxstyle="round,pad=0.02",
                                  facecolor=COLORS['accent_light'],
                                  edgecolor=COLORS['accent_primary'])
    ax.add_patch(formula_box)

    ax.text(0.5, 0.82, 'Si |x| < ω:', ha='center', fontsize=10, color=COLORS['text_primary'])
    ax.text(0.5, 0.73, 'ω · ln(1 + |x|/ε)', ha='center', fontsize=12,
           family='monospace', color=COLORS['accent_primary'], fontweight='bold')
    ax.text(0.5, 0.65, 'Si no: |x| - C', ha='center', fontsize=10,
           family='monospace', color=COLORS['text_secondary'])

    # Parámetros
    ax.text(0.5, 0.52, 'Parámetros:', ha='center', fontsize=11, fontweight='bold',
           color=COLORS['text_primary'])
    params = ['ω = 10 px (umbral)', 'ε = 2 px (curvatura)', 'C = ω - ω·ln(1+ω/ε)']
    for i, p in enumerate(params):
        ax.text(0.5, 0.44 - i*0.09, p, ha='center', fontsize=9, color=COLORS['text_secondary'])

    # Ventaja
    ax.text(0.5, 0.10, '~5× mayor gradiente\npara errores <10px', ha='center',
           fontsize=9, fontweight='bold', color=COLORS['success'],
           bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['attention'],
                    edgecolor=COLORS['success'], alpha=0.5))

    plt.tight_layout()
    plt.savefig(ASSETS_GRAFICAS / 'formula_wing_loss.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


# ============================================================================
# PARTE 2: COMPOSICIONES DE SLIDES
# ============================================================================

def slide17_arquitectura():
    """Slide 17: Composición - Arquitectura completa."""
    print("  -> Generando slide17_arquitectura.png")

    fig = plt.figure(figsize=(14, 7.5), facecolor=COLORS['background'])
    fig.suptitle('La arquitectura combina ResNet-18 pre-entrenada con\nmódulos especializados para regresión de landmarks',
                fontsize=14, fontweight='bold', y=0.96)

    # Cargar asset del pipeline
    ax_main = fig.add_axes([0.02, 0.25, 0.96, 0.55])
    try:
        img = plt.imread(ASSETS_DIAGRAMAS / 'pipeline_arquitectura.png')
        ax_main.imshow(img)
    except:
        ax_main.text(0.5, 0.5, '[Pipeline de Arquitectura]', ha='center', va='center',
                    fontsize=12, color=COLORS['text_secondary'])
    ax_main.axis('off')

    # Panel inferior con características
    features = [
        ('Transfer Learning', 'ImageNet pre-training', COLORS['accent_primary']),
        ('Atención Espacial', 'Coordinate Attention', COLORS['data_2']),
        ('Regularización', 'GroupNorm + Dropout', COLORS['data_3']),
    ]

    for i, (title, desc, color) in enumerate(features):
        ax_feat = fig.add_axes([0.05 + i*0.32, 0.03, 0.28, 0.15])
        ax_feat.axis('off')
        box = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                              facecolor='white', edgecolor=color, linewidth=2)
        ax_feat.add_patch(box)
        ax_feat.text(0.5, 0.65, title, ha='center', fontsize=10, fontweight='bold', color=color)
        ax_feat.text(0.5, 0.30, desc, ha='center', fontsize=9, color=COLORS['text_secondary'])

    # Resultado
    ax_result = fig.add_axes([0.30, 0.82, 0.40, 0.08])
    ax_result.axis('off')
    result_box = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                                 facecolor=COLORS['attention'], edgecolor=COLORS['success'],
                                 alpha=0.7, linewidth=2)
    ax_result.add_patch(result_box)
    ax_result.text(0.5, 0.5, 'Mejor Error: 3.71 px (ensemble) | 59% mejora vs baseline',
                  ha='center', va='center', fontsize=11, fontweight='bold', color=COLORS['success'])

    plt.savefig(SLIDES_DIR / 'slide17_arquitectura.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def slide18_resnet():
    """Slide 18: Composición - ResNet-18 features."""
    print("  -> Generando slide18_resnet_features.png")

    fig = plt.figure(figsize=(14, 7.5), facecolor=COLORS['background'])
    fig.suptitle('ResNet-18 extrae características jerárquicas:\nde bordes simples a patrones anatómicos complejos',
                fontsize=14, fontweight='bold', y=0.96)

    # Cargar asset
    ax_main = fig.add_axes([0.02, 0.15, 0.75, 0.70])
    try:
        img = plt.imread(ASSETS_DIAGRAMAS / 'resnet18_capas.png')
        ax_main.imshow(img)
    except:
        ax_main.text(0.5, 0.5, '[ResNet-18 Capas]', ha='center', va='center')
    ax_main.axis('off')

    # Panel derecho: pirámide de features
    ax_pyramid = fig.add_axes([0.78, 0.20, 0.20, 0.60])
    ax_pyramid.axis('off')
    ax_pyramid.set_xlim(0, 1)
    ax_pyramid.set_ylim(0, 1)

    ax_pyramid.text(0.5, 0.95, 'Jerarquía', ha='center', fontsize=11, fontweight='bold',
                   color=COLORS['accent_primary'])

    levels = [('Semántica', 0.85), ('Estructuras', 0.65), ('Partes', 0.45),
              ('Texturas', 0.25), ('Bordes', 0.05)]
    for label, y in levels:
        width = 0.3 + (1-y) * 0.5
        rect = FancyBboxPatch((0.5 - width/2, y), width, 0.15,
                              boxstyle="round,pad=0.01",
                              facecolor=plt.cm.Blues(0.3 + y*0.5),
                              edgecolor='white', alpha=0.85)
        ax_pyramid.add_patch(rect)
        ax_pyramid.text(0.5, y + 0.075, label, ha='center', va='center',
                       fontsize=9, color='white', fontweight='bold')

    plt.savefig(SLIDES_DIR / 'slide18_resnet_features.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def slide19_coord_attention():
    """Slide 19: Composición - Coordinate Attention."""
    print("  -> Generando slide19_coordinate_attention.png")

    fig = plt.figure(figsize=(14, 7.5), facecolor=COLORS['background'])
    fig.suptitle('Coordinate Attention captura dependencias espaciales\ncodificando posición en ambas direcciones (H y W)',
                fontsize=14, fontweight='bold', y=0.96)

    # Asset principal
    ax_main = fig.add_axes([0.02, 0.12, 0.65, 0.75])
    try:
        img = plt.imread(ASSETS_DIAGRAMAS / 'coordinate_attention.png')
        ax_main.imshow(img)
    except:
        ax_main.text(0.5, 0.5, '[Coordinate Attention]', ha='center', va='center')
    ax_main.axis('off')

    # Panel derecho
    ax_info = fig.add_axes([0.68, 0.15, 0.30, 0.70])
    ax_info.axis('off')

    ax_info.text(0.5, 0.95, 'Coordinate Attention', ha='center', fontsize=12,
                fontweight='bold', color=COLORS['data_2'])
    ax_info.text(0.5, 0.88, '(Hou et al., CVPR 2021)', ha='center', fontsize=9,
                color=COLORS['text_secondary'], style='italic')

    benefits = [('Codifica posición', 'Preserva ubicación espacial'),
                ('Eficiente', 'Menor costo que self-attention'),
                ('Direccional', 'Captura H y W por separado'),
                ('Para landmarks', 'Ideal para posiciones anatómicas')]

    y_pos = 0.75
    for title, desc in benefits:
        circle = Circle((0.08, y_pos), 0.025, color=COLORS['data_2'], transform=ax_info.transAxes)
        ax_info.add_patch(circle)
        ax_info.text(0.15, y_pos, title, fontsize=10, fontweight='bold', va='center')
        ax_info.text(0.15, y_pos - 0.07, desc, fontsize=8, color=COLORS['text_secondary'])
        y_pos -= 0.18

    # Nota inferior
    fig.text(0.5, 0.02, 'CA preserva información posicional crítica para localización de landmarks',
            ha='center', fontsize=9, color=COLORS['text_secondary'], style='italic')

    plt.savefig(SLIDES_DIR / 'slide19_coordinate_attention.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def slide20_deep_head():
    """Slide 20: Composición - Deep Head."""
    print("  -> Generando slide20_deep_head.png")

    fig = plt.figure(figsize=(14, 7.5), facecolor=COLORS['background'])
    fig.suptitle('La cabeza de regresión profunda transforma 512 features\nen 30 coordenadas normalizadas [0, 1]',
                fontsize=14, fontweight='bold', y=0.96)

    # Asset principal
    ax_main = fig.add_axes([0.02, 0.30, 0.65, 0.55])
    try:
        img = plt.imread(ASSETS_DIAGRAMAS / 'deep_head.png')
        ax_main.imshow(img)
    except:
        ax_main.text(0.5, 0.5, '[Deep Head]', ha='center', va='center')
    ax_main.axis('off')

    # Panel derecho: componentes
    ax_info = fig.add_axes([0.68, 0.15, 0.30, 0.70])
    ax_info.axis('off')

    ax_info.text(0.5, 0.95, 'Componentes', ha='center', fontsize=12,
                fontweight='bold', color=COLORS['accent_primary'])

    components = [
        ('GroupNorm', 'Normalización por grupos\n(32 grupos)', COLORS['accent_primary']),
        ('GELU', 'Activación suave\nGaussian Error Linear', COLORS['data_2']),
        ('Dropout 0.3', 'Regularización fuerte\npara dataset pequeño', COLORS['data_3']),
        ('Sigmoid', 'Salida en [0, 1]\n15 pares (x, y)', COLORS['danger']),
    ]

    y_pos = 0.82
    for title, desc, color in components:
        box = FancyBboxPatch((0.02, y_pos - 0.15), 0.96, 0.17,
                             boxstyle="round,pad=0.01",
                             facecolor='white', edgecolor=color, linewidth=1.5,
                             transform=ax_info.transAxes)
        ax_info.add_patch(box)
        ax_info.text(0.08, y_pos - 0.03, title, fontsize=9, fontweight='bold', color=color)
        ax_info.text(0.08, y_pos - 0.11, desc, fontsize=7, color=COLORS['text_secondary'])
        y_pos -= 0.21

    # Info superior
    fig.text(0.35, 0.88, 'Output: 30 valores = 15 landmarks × 2 coordenadas (x, y)',
            ha='center', fontsize=10, fontweight='bold', color=COLORS['accent_primary'])

    plt.savefig(SLIDES_DIR / 'slide20_deep_head.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def slide21_wing_loss():
    """Slide 21: Composición - Wing Loss."""
    print("  -> Generando slide21_wing_loss.png")

    fig = plt.figure(figsize=(14, 7.5), facecolor=COLORS['background'])
    fig.suptitle('Wing Loss amplifica gradientes para errores pequeños,\ncrucial para precisión sub-pixel en landmarks',
                fontsize=14, fontweight='bold', y=0.96)

    # Asset: gráfica comparativa
    ax_loss = fig.add_axes([0.04, 0.15, 0.38, 0.70])
    try:
        img = plt.imread(ASSETS_GRAFICAS / 'wing_loss_comparacion.png')
        ax_loss.imshow(img)
    except:
        ax_loss.text(0.5, 0.5, '[Wing Loss Comparación]', ha='center', va='center')
    ax_loss.axis('off')

    # Asset: gradientes
    ax_grad = fig.add_axes([0.44, 0.15, 0.30, 0.70])
    try:
        img = plt.imread(ASSETS_GRAFICAS / 'wing_loss_gradientes.png')
        ax_grad.imshow(img)
    except:
        ax_grad.text(0.5, 0.5, '[Gradientes]', ha='center', va='center')
    ax_grad.axis('off')

    # Asset: fórmula
    ax_formula = fig.add_axes([0.76, 0.15, 0.22, 0.70])
    try:
        img = plt.imread(ASSETS_GRAFICAS / 'formula_wing_loss.png')
        ax_formula.imshow(img)
    except:
        ax_formula.text(0.5, 0.5, '[Fórmula]', ha='center', va='center')
    ax_formula.axis('off')

    # Nota
    fig.text(0.5, 0.03, 'Wing Loss (Feng et al., CVPR 2018): Diseñada para regresión de landmarks',
            ha='center', fontsize=9, color=COLORS['text_secondary'], style='italic')

    plt.savefig(SLIDES_DIR / 'slide21_wing_loss.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def slide22_two_phase():
    """Slide 22: Composición - Entrenamiento 2 fases."""
    print("  -> Generando slide22_two_phase.png")

    fig = plt.figure(figsize=(14, 7.5), facecolor=COLORS['background'])
    fig.suptitle('El entrenamiento en 2 fases protege las features pre-entrenadas\nmientras adapta el modelo a landmarks anatómicos',
                fontsize=14, fontweight='bold', y=0.96)

    # Asset principal
    ax_main = fig.add_axes([0.02, 0.12, 0.96, 0.75])
    try:
        img = plt.imread(ASSETS_DIAGRAMAS / 'training_2fases.png')
        ax_main.imshow(img)
    except:
        ax_main.text(0.5, 0.5, '[Training 2 Fases]', ha='center', va='center')
    ax_main.axis('off')

    # Nota
    fig.text(0.5, 0.03, 'Protege features ImageNet evitando "catastrophic forgetting" | CosineAnnealingLR scheduler',
            ha='center', fontsize=9, color=COLORS['text_secondary'], style='italic')

    plt.savefig(SLIDES_DIR / 'slide22_two_phase.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("BLOQUE 4 - ARQUITECTURA")
    print("=" * 70)

    create_directories()

    print("\n[1/2] Generando ASSETS individuales...")
    asset_pipeline_arquitectura()
    asset_resnet18_capas()
    asset_coordinate_attention()
    asset_deep_head()
    asset_wing_loss_comparacion()
    asset_wing_loss_gradientes()
    asset_training_2fases()
    asset_formula_wing_loss()

    print("\n[2/2] Generando COMPOSICIONES de slides...")
    slide17_arquitectura()
    slide18_resnet()
    slide19_coord_attention()
    slide20_deep_head()
    slide21_wing_loss()
    slide22_two_phase()

    print("\n" + "=" * 70)
    print("COMPLETADO")
    print("=" * 70)
    print(f"\nAssets generados en: {ASSETS_DIAGRAMAS.parent}")
    print(f"Slides generados en: {SLIDES_DIR}")

    print("\nAssets:")
    for f in sorted(ASSETS_DIAGRAMAS.glob('*.png')):
        print(f"  - diagramas/{f.name}")
    for f in sorted(ASSETS_GRAFICAS.glob('*.png')):
        print(f"  - graficas/{f.name}")

    print("\nSlides:")
    for f in sorted(SLIDES_DIR.glob('*.png')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
