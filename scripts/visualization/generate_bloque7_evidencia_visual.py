#!/usr/bin/env python3
"""
Script: Visualizaciones Bloque 7 - EVIDENCIA VISUAL (Slides 33-35)
Estilo: v2_profesional

ESTRUCTURA:
1. Assets individuales:
   - assets/diagramas/ -> Diagramas conceptuales
   - assets/graficas/  -> Visualizaciones de atención

2. Composiciones de slides:
   - v2_profesional/slide33_*.png ... slide35_*.png

Slides:
- 33: GradCAM revela que el modelo atiende a regiones anatómicamente relevantes
- 34: El modelo aprende a enfocar cada landmark en su región correspondiente
- 35: El error está cerca del límite teórico impuesto por ruido de etiquetado
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle, Wedge
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
    # GradCAM colors
    'gradcam_hot': '#ff4444',
    'gradcam_warm': '#ff8844',
    'gradcam_cool': '#4488ff',
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = PROJECT_ROOT
OUTPUT_DIR = BASE_DIR / 'presentacion' / '07_evidencia_visual'
ASSETS_DIAGRAMAS = OUTPUT_DIR / 'assets' / 'diagramas'
ASSETS_GRAFICAS = OUTPUT_DIR / 'assets' / 'graficas'
SLIDES_DIR = OUTPUT_DIR / 'v2_profesional'
THESIS_FIGURES = BASE_DIR / 'outputs' / 'thesis_figures'
PIPELINE_VIZ = BASE_DIR / 'outputs' / 'pipeline_viz'

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


def draw_arrow(ax, x1, y1, x2, y2, color='#003366', lw=2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                            mutation_scale=15, color=color, lw=lw)
    ax.add_patch(arrow)


# ============================================================================
# PARTE 1: ASSETS INDIVIDUALES
# ============================================================================

def asset_gradcam_explanation():
    """Asset: Diagrama explicativo de GradCAM."""
    print("  -> Generando asset: gradcam_explanation.png")

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=COLORS['background'])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Paso 1: Imagen de entrada
    draw_box(ax, 0.3, 1.5, 1.8, 2.0, 'Imagen', 'Radiografía',
             color=COLORS['accent_light'], border_color=COLORS['accent_primary'])

    # Flecha
    draw_arrow(ax, 2.1, 2.5, 2.8, 2.5, COLORS['accent_primary'])

    # Paso 2: CNN
    draw_box(ax, 2.8, 1.3, 2.0, 2.4, 'CNN', 'ResNet-18',
             color=COLORS['accent_light'], border_color=COLORS['accent_primary'])

    # Feature maps
    ax.text(3.8, 3.9, 'Feature Maps', fontsize=9, ha='center', color=COLORS['text_secondary'])

    # Flecha
    draw_arrow(ax, 4.8, 2.5, 5.5, 2.5, COLORS['accent_primary'])

    # Paso 3: Gradientes
    draw_box(ax, 5.5, 1.3, 2.0, 2.4, 'Gradientes', '∂y/∂A',
             color=COLORS['data_3'] + '40', border_color=COLORS['data_3'])

    ax.text(6.5, 3.9, 'Backprop desde', fontsize=8, ha='center', color=COLORS['text_secondary'])
    ax.text(6.5, 3.6, 'landmark target', fontsize=8, ha='center', color=COLORS['text_secondary'])

    # Flecha
    draw_arrow(ax, 7.5, 2.5, 8.2, 2.5, COLORS['data_3'])

    # Paso 4: Weighted combination
    draw_box(ax, 8.2, 1.3, 2.0, 2.4, 'Ponderación', 'Σ αk · Ak',
             color=COLORS['data_2'] + '40', border_color=COLORS['data_2'])

    # Flecha
    draw_arrow(ax, 10.2, 2.5, 10.8, 2.5, COLORS['data_2'])

    # Paso 5: Heatmap final
    # Simular heatmap
    ax_heat = fig.add_axes([0.88, 0.30, 0.10, 0.40])
    np.random.seed(42)
    heatmap = np.random.random((10, 10))
    heatmap[4:7, 4:7] = 1.0  # Hot spot
    ax_heat.imshow(heatmap, cmap='jet', aspect='auto')
    ax_heat.axis('off')
    ax.text(11.2, 0.9, 'GradCAM', fontsize=10, ha='center', fontweight='bold',
           color=COLORS['accent_primary'])

    # Título
    ax.text(6, 4.6, 'GradCAM: Visualización de Regiones de Atención del Modelo',
           ha='center', fontsize=12, fontweight='bold', color=COLORS['accent_primary'])

    # Explicación inferior
    ax.text(6, 0.3, 'El modelo "mira" las regiones rojas al predecir cada landmark',
           ha='center', fontsize=10, style='italic', color=COLORS['text_secondary'])

    plt.tight_layout()
    plt.savefig(ASSETS_DIAGRAMAS / 'gradcam_explanation.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def asset_attention_regions():
    """Asset: Diagrama de regiones de atención por tipo de landmark."""
    print("  -> Generando asset: attention_regions.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), facecolor=COLORS['background'])

    landmarks_info = [
        ('L1 (Superior)', 'Vértice pulmonar', COLORS['data_1'], (0.5, 0.8)),
        ('L10 (Central)', 'Carena traqueal', COLORS['data_2'], (0.5, 0.5)),
        ('L14 (Costofrenico)', 'Ángulo costofrenico', COLORS['data_3'], (0.7, 0.25)),
    ]

    for ax, (landmark, region, color, pos) in zip(axes, landmarks_info):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Silueta simplificada de tórax
        torax_x = [0.2, 0.5, 0.8, 0.85, 0.8, 0.5, 0.2, 0.15]
        torax_y = [0.85, 0.95, 0.85, 0.5, 0.15, 0.05, 0.15, 0.5]
        ax.fill(torax_x, torax_y, alpha=0.2, color='gray', edgecolor='gray', linewidth=2)

        # Región de atención (círculo difuso)
        circle = plt.Circle(pos, 0.15, color=color, alpha=0.4)
        ax.add_patch(circle)
        circle2 = plt.Circle(pos, 0.08, color=color, alpha=0.7)
        ax.add_patch(circle2)

        # Punto del landmark
        ax.scatter([pos[0]], [pos[1]], s=100, c=color, edgecolor='white',
                  linewidth=2, zorder=10, marker='o')

        # Etiquetas
        ax.set_title(landmark, fontsize=11, fontweight='bold', color=color)
        ax.text(0.5, -0.05, region, ha='center', fontsize=9, color=COLORS['text_secondary'])

    fig.suptitle('Regiones de Atención por Tipo de Landmark', fontsize=13,
                fontweight='bold', y=1.02, color=COLORS['accent_primary'])

    plt.tight_layout()
    plt.savefig(ASSETS_GRAFICAS / 'attention_regions.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def asset_error_limit_diagram():
    """Asset: Diagrama del límite teórico de error."""
    print("  -> Generando asset: error_limit_diagram.png")

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['background'])

    # Datos
    components = ['Ruido de\netiquetado\n(GT)', 'Error\nirreducible\n(modelo)', 'Error\nobservado\n(total)']
    values = [1.8, 1.9, 3.71]
    colors = [COLORS['warning'], COLORS['accent_secondary'], COLORS['success']]

    # Barras apiladas conceptuales
    x = [0, 1, 2]
    bars = ax.bar(x, values, color=colors, edgecolor='white', linewidth=2, width=0.6)

    # Etiquetas
    for bar, val, comp in zip(bars, values, components):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{val:.2f} px', ha='center', fontsize=12, fontweight='bold',
               color=COLORS['text_primary'])

    # Línea de comparación
    ax.axhline(y=8.0, color=COLORS['danger'], linestyle='--', linewidth=2, alpha=0.7)
    ax.text(2.4, 8.1, 'Objetivo: 8 px', fontsize=10, color=COLORS['danger'])

    # Anotación de límite teórico
    ax.axhline(y=1.8, color=COLORS['warning'], linestyle=':', linewidth=2, alpha=0.7)
    ax.text(2.4, 1.6, 'Límite teórico (~1.5-2.0 px)', fontsize=9,
           color=COLORS['warning'], style='italic')

    # Flecha indicando cercanía al límite
    ax.annotate('', xy=(2, 1.8), xytext=(2, 3.71),
               arrowprops=dict(arrowstyle='<->', color=COLORS['data_2'], lw=2))
    ax.text(2.15, 2.75, '1.9 px\nsobre\nlímite', fontsize=9,
           color=COLORS['data_2'], va='center')

    ax.set_xticks(x)
    ax.set_xticklabels(components, fontsize=10)
    ax.set_ylabel('Error (píxeles)', fontsize=11)
    ax.set_title('El Error del Modelo Está Cerca del Límite Teórico',
                fontsize=13, fontweight='bold', color=COLORS['accent_primary'])
    ax.set_ylim(0, 9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    # Nota: Texto explicativo removido para evitar superposición con etiquetas del eje X
    # La explicación está en el título del slide 35

    plt.tight_layout()
    plt.savefig(ASSETS_DIAGRAMAS / 'error_limit_diagram.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def asset_model_vs_human():
    """Asset: Comparación conceptual modelo vs humano."""
    print("  -> Generando asset: model_vs_human.png")

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=COLORS['background'])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Panel izquierdo: Humano
    draw_box(ax, 0.5, 1.5, 3.5, 2.5, '', '',
             color=COLORS['accent_light'], border_color=COLORS['accent_primary'])
    ax.text(2.25, 3.5, 'Etiquetado Manual', ha='center', fontsize=11,
           fontweight='bold', color=COLORS['accent_primary'])
    ax.text(2.25, 2.8, 'Variabilidad: 5-15 px', ha='center', fontsize=10,
           color=COLORS['danger'])
    ax.text(2.25, 2.3, 'Tiempo: 3-5 min/imagen', ha='center', fontsize=10,
           color=COLORS['text_secondary'])
    ax.text(2.25, 1.8, 'Fatiga y subjetividad', ha='center', fontsize=10,
           color=COLORS['text_secondary'])

    # Flecha de comparación
    ax.annotate('', xy=(5.5, 2.75), xytext=(4.5, 2.75),
               arrowprops=dict(arrowstyle='<->', color=COLORS['data_2'], lw=3))

    # Panel derecho: Modelo
    draw_box(ax, 6.0, 1.5, 3.5, 2.5, '', '',
             color=COLORS['success'] + '30', border_color=COLORS['success'])
    ax.text(7.75, 3.5, 'Modelo Deep Learning', ha='center', fontsize=11,
           fontweight='bold', color=COLORS['success'])
    ax.text(7.75, 2.8, 'Error: 3.71 px', ha='center', fontsize=10,
           color=COLORS['success'], fontweight='bold')
    ax.text(7.75, 2.3, 'Tiempo: <0.1 seg/imagen', ha='center', fontsize=10,
           color=COLORS['text_secondary'])
    ax.text(7.75, 1.8, 'Consistente y escalable', ha='center', fontsize=10,
           color=COLORS['text_secondary'])

    # Título
    ax.text(5, 4.6, 'Comparación: Etiquetado Manual vs Modelo Automatizado',
           ha='center', fontsize=12, fontweight='bold', color=COLORS['accent_primary'])

    # Conclusión
    ax.text(5, 0.6, 'El modelo logra precisión comparable al etiquetado manual\n'
                   'con velocidad y consistencia muy superiores',
           ha='center', fontsize=10, style='italic', color=COLORS['text_secondary'])

    plt.tight_layout()
    plt.savefig(ASSETS_DIAGRAMAS / 'model_vs_human.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


# ============================================================================
# PARTE 2: COMPOSICIONES DE SLIDES
# ============================================================================

def slide33_gradcam():
    """Slide 33: GradCAM revela regiones anatómicamente relevantes."""
    print("  -> Generando slide33_gradcam.png")

    fig = plt.figure(figsize=(14, 7.5), facecolor=COLORS['background'])
    fig.suptitle('GradCAM revela que el modelo atiende a regiones\nanatómicamente relevantes para cada landmark',
                fontsize=14, fontweight='bold', y=0.96)

    # Diagrama explicativo
    ax_explain = fig.add_axes([0.02, 0.50, 0.96, 0.38])
    try:
        img = plt.imread(ASSETS_DIAGRAMAS / 'gradcam_explanation.png')
        ax_explain.imshow(img)
    except:
        ax_explain.text(0.5, 0.5, '[Explicación GradCAM]', ha='center', va='center')
    ax_explain.axis('off')

    # Visualizaciones de GradCAM existentes
    ax_gradcam = fig.add_axes([0.02, 0.05, 0.96, 0.42])
    try:
        # Intentar cargar imagen existente
        img = plt.imread(PIPELINE_VIZ / 'attention_maps' / 'gradcam_all_landmarks.png')
        ax_gradcam.imshow(img)
    except:
        try:
            img = plt.imread(BASE_DIR / 'presentacion' / '07_evidencia_visual' / 'gradcam_all_landmarks.png')
            ax_gradcam.imshow(img)
        except:
            ax_gradcam.text(0.5, 0.5, '[Ver: outputs/pipeline_viz/attention_maps/gradcam_all_landmarks.png]',
                           ha='center', va='center', fontsize=10, color=COLORS['text_secondary'])
    ax_gradcam.axis('off')

    plt.savefig(SLIDES_DIR / 'slide33_gradcam.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def slide34_attention_focus():
    """Slide 34: El modelo aprende a enfocar cada landmark."""
    print("  -> Generando slide34_attention_focus.png")

    fig = plt.figure(figsize=(14, 7.5), facecolor=COLORS['background'])
    fig.suptitle('El modelo aprende a enfocar cada landmark\nen su región anatómica correspondiente',
                fontsize=14, fontweight='bold', y=0.96)

    # Asset: regiones de atención
    ax_regions = fig.add_axes([0.02, 0.30, 0.50, 0.58])
    try:
        img = plt.imread(ASSETS_GRAFICAS / 'attention_regions.png')
        ax_regions.imshow(img)
    except:
        ax_regions.text(0.5, 0.5, '[Regiones de Atención]', ha='center', va='center')
    ax_regions.axis('off')

    # Comparación de atención existente
    ax_compare = fig.add_axes([0.52, 0.30, 0.46, 0.58])
    try:
        img = plt.imread(PIPELINE_VIZ / 'attention_maps' / 'attention_comparison.png')
        ax_compare.imshow(img)
    except:
        try:
            img = plt.imread(BASE_DIR / 'presentacion' / '07_evidencia_visual' / 'attention_comparison.png')
            ax_compare.imshow(img)
        except:
            ax_compare.text(0.5, 0.5, '[Comparación de Atención]', ha='center', va='center')
    ax_compare.axis('off')

    # Panel inferior: insights
    insights = [
        ('Landmarks superiores', 'Atención en vértices pulmonares', COLORS['data_1']),
        ('Landmarks centrales', 'Atención en mediastino y carena', COLORS['data_2']),
        ('Landmarks costofrenicos', 'Atención en ángulos diafragmáticos', COLORS['data_3']),
    ]

    for i, (title, desc, color) in enumerate(insights):
        ax = fig.add_axes([0.05 + i*0.32, 0.02, 0.28, 0.22])
        ax.axis('off')
        box = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                              facecolor='white', edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(0.5, 0.70, title, ha='center', fontsize=10, fontweight='bold', color=color)
        ax.text(0.5, 0.35, desc, ha='center', fontsize=9, color=COLORS['text_secondary'])

    plt.savefig(SLIDES_DIR / 'slide34_attention_focus.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def slide35_theoretical_limit():
    """Slide 35: El error está cerca del límite teórico."""
    print("  -> Generando slide35_theoretical_limit.png")

    fig = plt.figure(figsize=(14, 7.5), facecolor=COLORS['background'])
    fig.suptitle('El error de 3.71 px está cerca del límite teórico\nimpuesto por el ruido de etiquetado (~1.5-2.0 px)',
                fontsize=14, fontweight='bold', y=0.96)

    # Diagrama de límite teórico
    ax_limit = fig.add_axes([0.02, 0.18, 0.50, 0.70])
    try:
        img = plt.imread(ASSETS_DIAGRAMAS / 'error_limit_diagram.png')
        ax_limit.imshow(img)
    except:
        ax_limit.text(0.5, 0.5, '[Diagrama Límite Teórico]', ha='center', va='center')
    ax_limit.axis('off')

    # Comparación modelo vs humano
    ax_compare = fig.add_axes([0.52, 0.18, 0.46, 0.70])
    try:
        img = plt.imread(ASSETS_DIAGRAMAS / 'model_vs_human.png')
        ax_compare.imshow(img)
    except:
        ax_compare.text(0.5, 0.5, '[Modelo vs Humano]', ha='center', va='center')
    ax_compare.axis('off')

    # Panel inferior: conclusión clave
    ax_conclusion = fig.add_axes([0.15, 0.02, 0.70, 0.12])
    ax_conclusion.axis('off')
    box = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                          facecolor=COLORS['success'] + '20',
                          edgecolor=COLORS['success'], linewidth=2)
    ax_conclusion.add_patch(box)
    ax_conclusion.text(0.5, 0.5, 'El modelo ha alcanzado un rendimiento cercano al óptimo teórico.\n'
                                'Mejoras adicionales requerirían reducir la variabilidad en el ground truth.',
                      ha='center', va='center', fontsize=11, color=COLORS['accent_primary'])

    plt.savefig(SLIDES_DIR / 'slide35_theoretical_limit.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("BLOQUE 7 - EVIDENCIA VISUAL (Slides 33-35)")
    print("=" * 70)

    create_directories()

    print("\n[1/2] Generando ASSETS individuales...")
    asset_gradcam_explanation()
    asset_attention_regions()
    asset_error_limit_diagram()
    asset_model_vs_human()

    print("\n[2/2] Generando COMPOSICIONES de slides...")
    slide33_gradcam()
    slide34_attention_focus()
    slide35_theoretical_limit()

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
