#!/usr/bin/env python3
"""
Script: Visualizaciones Bloque 8 - CONCLUSIONES (Slides 36-38)
Estilo: v2_profesional

ESTRUCTURA:
1. Assets individuales:
   - assets/diagramas/ -> Diagramas conceptuales
   - assets/graficas/  -> Infografías

2. Composiciones de slides:
   - v2_profesional/slide36_*.png ... slide38_*.png

Slides:
- 36: El sistema automatiza la detección con precisión comparable al etiquetado manual
- 37: Las contribuciones principales: CLAHE, ensemble selectivo, análisis granular
- 38: El trabajo abre camino para diagnóstico asistido y control de calidad
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle, Wedge, Polygon
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
    # Contribuciones
    'contrib_1': '#003366',
    'contrib_2': '#2d6a4f',
    'contrib_3': '#cc6600',
    # Aplicaciones
    'app_diagnosis': '#7b2cbf',
    'app_tracking': '#0066cc',
    'app_quality': '#2d6a4f',
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
OUTPUT_DIR = BASE_DIR / 'presentacion' / '08_conclusiones'
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


def draw_arrow(ax, x1, y1, x2, y2, color='#003366', lw=2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                            mutation_scale=15, color=color, lw=lw)
    ax.add_patch(arrow)


# ============================================================================
# PARTE 1: ASSETS INDIVIDUALES
# ============================================================================

def asset_automation_comparison():
    """Asset: Comparación automatización vs manual."""
    print("  -> Generando asset: automation_comparison.png")

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS['background'])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Panel Manual (izquierda)
    draw_box(ax, 0.5, 0.5, 5.0, 4.5, '', '',
             color=COLORS['danger'] + '20', border_color=COLORS['danger'])
    ax.text(3.0, 4.5, 'Etiquetado Manual', ha='center', fontsize=12,
           fontweight='bold', color=COLORS['danger'])

    manual_items = [
        ('Tiempo', '3-5 min/imagen'),
        ('Variabilidad', '5-15 px'),
        ('Escalabilidad', 'Limitada'),
        ('Fatiga', 'Afecta precisión'),
        ('Costo', 'Alto'),
    ]

    for i, (label, value) in enumerate(manual_items):
        y = 3.7 - i * 0.7
        ax.text(1.0, y, f'• {label}:', fontsize=10, color=COLORS['text_primary'])
        ax.text(3.5, y, value, fontsize=10, color=COLORS['danger'])

    # Flecha central
    ax.annotate('', xy=(6.8, 2.75), xytext=(5.7, 2.75),
               arrowprops=dict(arrowstyle='-|>', color=COLORS['success'], lw=3))
    ax.text(6.25, 3.3, 'Automatización', ha='center', fontsize=9, fontweight='bold',
           color=COLORS['success'])

    # Panel Automatizado (derecha)
    draw_box(ax, 6.5, 0.5, 5.0, 4.5, '', '',
             color=COLORS['success'] + '20', border_color=COLORS['success'])
    ax.text(9.0, 4.5, 'Sistema Propuesto', ha='center', fontsize=12,
           fontweight='bold', color=COLORS['success'])

    auto_items = [
        ('Tiempo', '<0.1 seg/imagen'),
        ('Error', '3.71 px'),
        ('Escalabilidad', 'Ilimitada'),
        ('Consistencia', '100%'),
        ('Costo', 'Bajo (una vez entrenado)'),
    ]

    for i, (label, value) in enumerate(auto_items):
        y = 3.7 - i * 0.7
        ax.text(7.0, y, f'• {label}:', fontsize=10, color=COLORS['text_primary'])
        ax.text(9.5, y, value, fontsize=10, color=COLORS['success'], fontweight='bold')

    # Título
    ax.text(6.0, 5.6, 'De Manual a Automatizado: Mejora en Todas las Métricas',
           ha='center', fontsize=13, fontweight='bold', color=COLORS['accent_primary'])

    plt.tight_layout()
    plt.savefig(ASSETS_DIAGRAMAS / 'automation_comparison.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def asset_contributions():
    """Asset: Infografía de contribuciones principales."""
    print("  -> Generando asset: contributions.png")

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS['background'])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Título
    ax.text(6, 5.6, 'Tres Contribuciones Principales', ha='center',
           fontsize=14, fontweight='bold', color=COLORS['accent_primary'])

    # Contribución 1: CLAHE para patología
    contrib1_color = COLORS['contrib_1']
    draw_box(ax, 0.3, 1.0, 3.5, 3.8, '', '', color='white', border_color=contrib1_color)

    # Icono conceptual (lupa/contraste)
    circle1 = plt.Circle((2.05, 4.0), 0.4, fill=True, color=contrib1_color, alpha=0.3)
    ax.add_patch(circle1)
    ax.text(2.05, 4.0, '1', ha='center', va='center', fontsize=16,
           fontweight='bold', color=contrib1_color)

    ax.text(2.05, 3.2, 'CLAHE para\nPatología COVID', ha='center', fontsize=11,
           fontweight='bold', color=contrib1_color)
    ax.text(2.05, 2.3, 'Mejora la visibilidad\nde landmarks en\nconsolidaciones\npulmonares',
           ha='center', fontsize=9, color=COLORS['text_secondary'])
    ax.text(2.05, 1.3, '-65% error en COVID', ha='center', fontsize=10,
           fontweight='bold', color=COLORS['success'])

    # Contribución 2: Ensemble selectivo
    contrib2_color = COLORS['contrib_2']
    draw_box(ax, 4.25, 1.0, 3.5, 3.8, '', '', color='white', border_color=contrib2_color)

    circle2 = plt.Circle((6.0, 4.0), 0.4, fill=True, color=contrib2_color, alpha=0.3)
    ax.add_patch(circle2)
    ax.text(6.0, 4.0, '2', ha='center', va='center', fontsize=16,
           fontweight='bold', color=contrib2_color)

    ax.text(6.0, 3.2, 'Ensemble\nSelectivo', ha='center', fontsize=11,
           fontweight='bold', color=contrib2_color)
    ax.text(6.0, 2.3, 'Excluir modelos\ndébiles es más\nefectivo que\nponderarlos',
           ha='center', fontsize=9, color=COLORS['text_secondary'])
    ax.text(6.0, 1.3, '-4% error adicional', ha='center', fontsize=10,
           fontweight='bold', color=COLORS['success'])

    # Contribución 3: Análisis granular
    contrib3_color = COLORS['contrib_3']
    draw_box(ax, 8.2, 1.0, 3.5, 3.8, '', '', color='white', border_color=contrib3_color)

    circle3 = plt.Circle((9.95, 4.0), 0.4, fill=True, color=contrib3_color, alpha=0.3)
    ax.add_patch(circle3)
    ax.text(9.95, 4.0, '3', ha='center', va='center', fontsize=16,
           fontweight='bold', color=contrib3_color)

    ax.text(9.95, 3.2, 'Análisis\nGranular', ha='center', fontsize=11,
           fontweight='bold', color=contrib3_color)
    ax.text(9.95, 2.3, 'Métricas por\nlandmark y categoría\npara diagnóstico\ndiferenciado',
           ha='center', fontsize=9, color=COLORS['text_secondary'])
    ax.text(9.95, 1.3, 'Identifica casos difíciles', ha='center', fontsize=10,
           fontweight='bold', color=COLORS['success'])

    # Línea de conexión inferior
    ax.plot([2.05, 6.0, 9.95], [0.7, 0.7, 0.7], 'o-', color=COLORS['accent_primary'],
           linewidth=2, markersize=8)
    ax.text(6, 0.35, 'Combinación → 59% mejora total', ha='center', fontsize=11,
           fontweight='bold', color=COLORS['accent_primary'])

    plt.tight_layout()
    plt.savefig(ASSETS_DIAGRAMAS / 'contributions.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def asset_future_applications():
    """Asset: Diagrama de aplicaciones futuras."""
    print("  -> Generando asset: future_applications.png")

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS['background'])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Centro: Sistema desarrollado
    draw_box(ax, 4.5, 2.0, 3.0, 2.0, 'Sistema de\nLandmarks', '3.71 px error',
             color=COLORS['accent_light'], border_color=COLORS['accent_primary'])

    # Aplicación 1: Diagnóstico asistido
    app1_color = COLORS['app_diagnosis']
    draw_box(ax, 0.3, 3.5, 3.5, 2.0, 'Diagnóstico\nAsistido', '',
             color=app1_color + '30', border_color=app1_color)
    ax.text(2.05, 3.3, 'Detección automática\nde patologías', ha='center', fontsize=9,
           color=COLORS['text_secondary'])
    draw_arrow(ax, 3.8, 4.5, 4.5, 3.5, app1_color)

    # Aplicación 2: Seguimiento longitudinal
    app2_color = COLORS['app_tracking']
    draw_box(ax, 8.2, 3.5, 3.5, 2.0, 'Seguimiento\nLongitudinal', '',
             color=app2_color + '30', border_color=app2_color)
    ax.text(9.95, 3.3, 'Evolución temporal\nde patología', ha='center', fontsize=9,
           color=COLORS['text_secondary'])
    draw_arrow(ax, 7.5, 3.5, 8.2, 4.5, app2_color)

    # Aplicación 3: Control de calidad
    app3_color = COLORS['app_quality']
    draw_box(ax, 0.3, 0.3, 3.5, 2.0, 'Control de\nCalidad', '',
             color=app3_color + '30', border_color=app3_color)
    ax.text(2.05, 0.1, 'Verificación de\nposicionamiento', ha='center', fontsize=9,
           color=COLORS['text_secondary'])
    draw_arrow(ax, 3.8, 1.3, 4.5, 2.5, app3_color)

    # Aplicación 4: Warping geométrico
    app4_color = COLORS['data_3']
    draw_box(ax, 8.2, 0.3, 3.5, 2.0, 'Normalización\nGeométrica', '',
             color=app4_color + '30', border_color=app4_color)
    ax.text(9.95, 0.1, 'Preprocesamiento\npara clasificación', ha='center', fontsize=9,
           color=COLORS['text_secondary'])
    draw_arrow(ax, 7.5, 2.5, 8.2, 1.3, app4_color)

    # Título
    ax.text(6, 5.6, 'Aplicaciones Clínicas del Sistema', ha='center',
           fontsize=14, fontweight='bold', color=COLORS['accent_primary'])

    plt.tight_layout()
    plt.savefig(ASSETS_DIAGRAMAS / 'future_applications.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def asset_summary_metrics():
    """Asset: Resumen visual de métricas clave."""
    print("  -> Generando asset: summary_metrics.png")

    fig = plt.figure(figsize=(12, 4), facecolor=COLORS['background'])

    metrics = [
        ('Error Final', '3.71 px', 'vs objetivo 8 px', COLORS['success']),
        ('Mejora Total', '59%', 'reducción del error', COLORS['accent_primary']),
        ('Velocidad', '<0.1 s', 'por imagen', COLORS['data_2']),
        ('Landmarks', '15', 'por radiografía', COLORS['data_3']),
    ]

    for i, (title, value, desc, color) in enumerate(metrics):
        ax = fig.add_axes([0.02 + i*0.245, 0.1, 0.23, 0.8])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Fondo
        box = FancyBboxPatch((0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.02",
                              facecolor='white', edgecolor=color, linewidth=3)
        ax.add_patch(box)

        # Contenido
        ax.text(0.5, 0.75, title, ha='center', fontsize=10, color=COLORS['text_secondary'])
        ax.text(0.5, 0.45, value, ha='center', fontsize=22, fontweight='bold', color=color)
        ax.text(0.5, 0.18, desc, ha='center', fontsize=9, color=COLORS['text_secondary'])

    plt.savefig(ASSETS_GRAFICAS / 'summary_metrics.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def asset_conclusion_diagram():
    """Asset: Diagrama de conclusión con flujo."""
    print("  -> Generando asset: conclusion_diagram.png")

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=COLORS['background'])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Problema inicial
    draw_box(ax, 0.3, 1.5, 2.5, 2.0, 'Problema', 'Etiquetado\nlento y variable',
             color=COLORS['danger'] + '30', border_color=COLORS['danger'])

    # Flecha
    draw_arrow(ax, 2.8, 2.5, 3.5, 2.5, COLORS['text_secondary'])

    # Solución
    draw_box(ax, 3.5, 1.5, 2.5, 2.0, 'Solución', 'Deep Learning\n+ Ensemble + TTA',
             color=COLORS['accent_light'], border_color=COLORS['accent_primary'])

    # Flecha
    draw_arrow(ax, 6.0, 2.5, 6.7, 2.5, COLORS['text_secondary'])

    # Resultado
    draw_box(ax, 6.7, 1.5, 2.5, 2.0, 'Resultado', '3.71 px error\n59% mejora',
             color=COLORS['success'] + '30', border_color=COLORS['success'])

    # Flecha
    draw_arrow(ax, 9.2, 2.5, 9.9, 2.5, COLORS['text_secondary'])

    # Impacto
    draw_box(ax, 9.9, 1.5, 1.8, 2.0, 'Impacto', 'Clínico',
             color=COLORS['data_4'] + '30', border_color=COLORS['data_4'])

    # Título
    ax.text(6, 4.5, 'De Problema a Impacto Clínico', ha='center',
           fontsize=14, fontweight='bold', color=COLORS['accent_primary'])

    # Línea de tiempo conceptual
    ax.plot([0.3, 11.7], [0.8, 0.8], '-', color=COLORS['text_secondary'], alpha=0.3, linewidth=2)
    stages = ['Análisis', 'Diseño', 'Desarrollo', 'Evaluación', 'Aplicación']
    for i, stage in enumerate(stages):
        x = 0.5 + i * 2.5
        ax.plot([x], [0.8], 'o', color=COLORS['accent_primary'], markersize=8)
        ax.text(x, 0.4, stage, ha='center', fontsize=8, color=COLORS['text_secondary'])

    plt.tight_layout()
    plt.savefig(ASSETS_DIAGRAMAS / 'conclusion_diagram.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


# ============================================================================
# PARTE 2: COMPOSICIONES DE SLIDES
# ============================================================================

def slide36_automation():
    """Slide 36: El sistema automatiza la detección con precisión comparable."""
    print("  -> Generando slide36_automation.png")

    fig = plt.figure(figsize=(14, 7.5), facecolor=COLORS['background'])
    fig.suptitle('El sistema automatiza la detección de landmarks\ncon precisión comparable al etiquetado manual',
                fontsize=14, fontweight='bold', y=0.96)

    # Asset principal: comparación
    ax_compare = fig.add_axes([0.02, 0.25, 0.96, 0.62])
    try:
        img = plt.imread(ASSETS_DIAGRAMAS / 'automation_comparison.png')
        ax_compare.imshow(img)
    except:
        ax_compare.text(0.5, 0.5, '[Comparación Automatización]', ha='center', va='center')
    ax_compare.axis('off')

    # Panel inferior: métricas resumen
    ax_metrics = fig.add_axes([0.05, 0.02, 0.90, 0.18])
    try:
        img = plt.imread(ASSETS_GRAFICAS / 'summary_metrics.png')
        ax_metrics.imshow(img)
    except:
        ax_metrics.text(0.5, 0.5, '[Métricas Resumen]', ha='center', va='center')
    ax_metrics.axis('off')

    plt.savefig(SLIDES_DIR / 'slide36_automation.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def slide37_contributions():
    """Slide 37: Las contribuciones principales del trabajo."""
    print("  -> Generando slide37_contributions.png")

    fig = plt.figure(figsize=(14, 7.5), facecolor=COLORS['background'])
    fig.suptitle('Tres contribuciones principales:\nCLAHE para patología, ensemble selectivo, y análisis granular',
                fontsize=14, fontweight='bold', y=0.96)

    # Asset principal: contribuciones
    ax_contrib = fig.add_axes([0.02, 0.08, 0.96, 0.80])
    try:
        img = plt.imread(ASSETS_DIAGRAMAS / 'contributions.png')
        ax_contrib.imshow(img)
    except:
        ax_contrib.text(0.5, 0.5, '[Contribuciones]', ha='center', va='center')
    ax_contrib.axis('off')

    plt.savefig(SLIDES_DIR / 'slide37_contributions.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def slide38_future():
    """Slide 38: El trabajo abre camino para aplicaciones clínicas."""
    print("  -> Generando slide38_future.png")

    fig = plt.figure(figsize=(14, 7.5), facecolor=COLORS['background'])
    fig.suptitle('El trabajo abre camino para diagnóstico asistido,\nseguimiento longitudinal y control de calidad',
                fontsize=14, fontweight='bold', y=0.96)

    # Asset principal: aplicaciones futuras
    ax_apps = fig.add_axes([0.02, 0.25, 0.96, 0.62])
    try:
        img = plt.imread(ASSETS_DIAGRAMAS / 'future_applications.png')
        ax_apps.imshow(img)
    except:
        ax_apps.text(0.5, 0.5, '[Aplicaciones Futuras]', ha='center', va='center')
    ax_apps.axis('off')

    # Panel inferior: conclusión
    ax_conclusion = fig.add_axes([0.10, 0.02, 0.80, 0.18])
    try:
        img = plt.imread(ASSETS_DIAGRAMAS / 'conclusion_diagram.png')
        ax_conclusion.imshow(img)
    except:
        ax_conclusion.text(0.5, 0.5, '[Diagrama Conclusión]', ha='center', va='center')
    ax_conclusion.axis('off')

    plt.savefig(SLIDES_DIR / 'slide38_future.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def slide_final_thanks():
    """Slide adicional: Gracias / Preguntas."""
    print("  -> Generando slide39_gracias.png")

    fig, ax = plt.subplots(figsize=(14, 7.5), facecolor=COLORS['background'])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    # Título principal
    ax.text(7, 5.5, '¡Gracias!', ha='center', va='center', fontsize=48,
           fontweight='bold', color=COLORS['accent_primary'])

    # Subtítulo
    ax.text(7, 4.2, 'Predicción Automática de Landmarks Anatómicos\nen Radiografías de Tórax con Deep Learning',
           ha='center', va='center', fontsize=14, color=COLORS['text_secondary'])

    # Línea decorativa
    ax.plot([3, 11], [3.5, 3.5], '-', color=COLORS['accent_primary'], linewidth=2)

    # Métricas destacadas
    highlights = [
        ('3.71 px', 'Error Final'),
        ('59%', 'Mejora'),
        ('15', 'Landmarks'),
    ]

    for i, (value, label) in enumerate(highlights):
        x = 4 + i * 3
        ax.text(x, 2.5, value, ha='center', fontsize=24, fontweight='bold',
               color=COLORS['success'])
        ax.text(x, 1.9, label, ha='center', fontsize=11, color=COLORS['text_secondary'])

    # Preguntas
    ax.text(7, 0.8, '¿Preguntas?', ha='center', fontsize=16,
           color=COLORS['accent_secondary'], style='italic')

    plt.savefig(SLIDES_DIR / 'slide39_gracias.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("BLOQUE 8 - CONCLUSIONES (Slides 36-38 + Cierre)")
    print("=" * 70)

    create_directories()

    print("\n[1/2] Generando ASSETS individuales...")
    asset_automation_comparison()
    asset_contributions()
    asset_future_applications()
    asset_summary_metrics()
    asset_conclusion_diagram()

    print("\n[2/2] Generando COMPOSICIONES de slides...")
    slide36_automation()
    slide37_contributions()
    slide38_future()
    slide_final_thanks()

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
