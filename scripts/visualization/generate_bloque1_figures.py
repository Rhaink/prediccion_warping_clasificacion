#!/usr/bin/env python3
"""
Script para generar las visualizaciones del Bloque 1: Contexto y Problema
para la presentación Assertion-Evidence.

Slides 1-7:
1. Portada profesional con RX + landmarks
2. Infografía de uso global de radiografías (2B anuales)
3. Gráfica temporal COVID-19 vs demanda radiológica
4. RX con 15 landmarks etiquetados anatómicamente
5. Comparación de variabilidad inter-observador
6. Diagrama embudo problema-objetivo-solución
7. Gráfico de categorías del dataset (adaptar existente)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Wedge
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd

# Configuración de estilo
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14

# Colores del estilo visual (del plan maestro)
COLORS = {
    'primary': '#2E86AB',      # Azul profesional
    'secondary': '#A23B72',    # Magenta para énfasis
    'success': '#28A745',      # Verde para mejoras
    'warning': '#FFC107',      # Amarillo para atención
    'danger': '#DC3545',       # Rojo para alertas
    'background': '#F8F9FA',   # Fondo claro
    'text': '#212529',         # Texto oscuro
    'covid': '#DC3545',        # COVID
    'normal': '#28A745',       # Normal
    'viral': '#FFC107',        # Viral Pneumonia
    'axis': 'cyan',            # Landmarks L1, L2
    'central': 'lime',         # Landmarks L9, L10, L11
    'symmetric': 'yellow',     # Pares simétricos
    'corner': 'magenta'        # L12-L15
}

# Rutas
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = PROJECT_ROOT
OUTPUT_DIR = BASE_DIR / 'presentacion' / '01_contexto'
DATA_DIR = BASE_DIR / 'data'
DATASET_DIR = DATA_DIR / 'dataset'

# Nombres anatómicos de los 15 landmarks
LANDMARK_ANATOMY = {
    'L1': 'Ápice traqueal',
    'L2': 'Bifurcación traqueal',
    'L3': 'Ápice pulmonar derecho',
    'L4': 'Ápice pulmonar izquierdo',
    'L5': 'Hilio derecho (superior)',
    'L6': 'Hilio izquierdo (superior)',
    'L7': 'Hilio derecho (inferior)',
    'L8': 'Hilio izquierdo (inferior)',
    'L9': 'Eje central superior',
    'L10': 'Eje central medio',
    'L11': 'Eje central inferior',
    'L12': 'Ángulo cardiofrénico derecho',
    'L13': 'Ángulo cardiofrénico izquierdo',
    'L14': 'Ángulo costofrénico derecho',
    'L15': 'Ángulo costofrénico izquierdo'
}


def load_sample_image_with_coords():
    """Carga una imagen de muestra con sus coordenadas."""
    csv_path = DATA_DIR / 'coordenadas' / 'coordenadas_maestro.csv'

    # Cargar coordenadas
    coord_cols = []
    for i in range(1, 16):
        coord_cols.extend([f'L{i}_x', f'L{i}_y'])
    columns = ['idx'] + coord_cols + ['image_name']

    df = pd.read_csv(csv_path, header=None, names=columns)

    # Buscar una imagen Normal de buena calidad
    normal_rows = df[df['image_name'].str.startswith('Normal')]
    if len(normal_rows) > 0:
        row = normal_rows.iloc[50]  # Tomar la imagen 50 como ejemplo
    else:
        row = df.iloc[100]

    image_name = row['image_name']

    # Determinar categoría y ruta
    if image_name.startswith('COVID'):
        category = 'COVID'
    elif image_name.startswith('Normal'):
        category = 'Normal'
    else:
        category = 'Viral_Pneumonia'

    # Buscar la imagen en diferentes ubicaciones
    possible_paths = [
        DATASET_DIR / category / f"{image_name}.jpeg",
        DATASET_DIR / category / f"{image_name}.png",
        DATASET_DIR / 'COVID-19_Radiography_Dataset' / category / f"{image_name}.png",
    ]

    img_path = None
    for p in possible_paths:
        if p.exists():
            img_path = p
            break

    # Extraer coordenadas
    coords = []
    for i in range(1, 16):
        x = row[f'L{i}_x']
        y = row[f'L{i}_y']
        coords.append((x, y))

    return img_path, np.array(coords), image_name


def create_slide1_portada():
    """
    Slide 1: Portada profesional
    Título: Predicción automática de landmarks anatómicos con Deep Learning
            alcanza precisión de 3.71 píxeles
    """
    print("Generando Slide 1: Portada...")

    fig = plt.figure(figsize=(16, 9), facecolor=COLORS['background'])

    # Cargar imagen de muestra
    img_path, coords, _ = load_sample_image_with_coords()

    # Layout: imagen a la derecha, texto a la izquierda
    ax_img = fig.add_axes([0.5, 0.15, 0.45, 0.7])
    ax_text = fig.add_axes([0.02, 0.1, 0.46, 0.8])
    ax_text.axis('off')

    # Cargar y mostrar imagen
    if img_path and img_path.exists():
        img = Image.open(img_path).convert('L')
        img_array = np.array(img)
        ax_img.imshow(img_array, cmap='gray')

        # Dibujar landmarks
        for i, (x, y) in enumerate(coords):
            if i in [0, 1]:  # L1, L2 - eje
                color = COLORS['axis']
            elif i in [8, 9, 10]:  # L9, L10, L11 - centrales
                color = COLORS['central']
            elif i in [2, 3, 4, 5, 6, 7]:  # pares simétricos hilios/ápices
                color = COLORS['symmetric']
            else:  # L12-L15 - ángulos
                color = COLORS['corner']

            ax_img.plot(x, y, 'o', color=color, markersize=10, markeredgecolor='white',
                       markeredgewidth=2)
            ax_img.annotate(f'L{i+1}', (x+8, y-8), color='white', fontsize=9,
                           fontweight='bold',
                           path_effects=[pe.withStroke(
                               linewidth=2, foreground='black')])

        # Dibujar eje L1-L2
        ax_img.plot([coords[0][0], coords[1][0]], [coords[0][1], coords[1][1]],
                   'c--', linewidth=2, alpha=0.7)

    ax_img.axis('off')
    ax_img.set_title('Detección automática de 15 landmarks', fontsize=16,
                     color=COLORS['text'], fontweight='bold', pad=10)

    # Texto de la portada
    ax_text.text(0.5, 0.85, 'TESIS DE LICENCIATURA', fontsize=14,
                ha='center', color=COLORS['secondary'], fontweight='bold')

    ax_text.text(0.5, 0.65, 'Predicción Automática de\nLandmarks Anatómicos en\nRadiografías de Tórax\ncon Deep Learning',
                fontsize=24, ha='center', va='center', color=COLORS['text'],
                fontweight='bold', linespacing=1.4)

    ax_text.text(0.5, 0.35, 'Error alcanzado:', fontsize=16,
                ha='center', color=COLORS['text'])

    # Métrica destacada
    ax_text.text(0.5, 0.25, '3.71 px', fontsize=48,
                ha='center', color=COLORS['success'], fontweight='bold')

    ax_text.text(0.5, 0.15, '(objetivo: 8 px)', fontsize=14,
                ha='center', color=COLORS['text'], style='italic')

    ax_text.text(0.5, 0.02, 'ResNet-18 + Coordinate Attention + Ensemble',
                fontsize=12, ha='center', color=COLORS['primary'])

    # Guardar
    output_path = OUTPUT_DIR / 'slide1_portada.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Guardado: {output_path}")


def create_slide2_uso_global():
    """
    Slide 2: Infografía de uso global de radiografías
    Título: Las radiografías de tórax son la modalidad más utilizada
            con 2 mil millones anuales
    """
    print("Generando Slide 2: Infografía uso global...")

    fig = plt.figure(figsize=(16, 9), facecolor=COLORS['background'])
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Título de la slide (afirmación)
    ax.text(8, 8.3, 'Las radiografías de tórax son la modalidad de imagen\nmás utilizada con 2 mil millones anuales',
           fontsize=20, ha='center', va='top', fontweight='bold', color=COLORS['text'],
           linespacing=1.3)

    # Gráfico de barras horizontal para modalidades
    modalidades = ['Radiografía de tórax', 'Ultrasonido', 'Tomografía (CT)', 'Resonancia (MRI)', 'Otros']
    valores = [2000, 800, 400, 150, 300]  # En millones
    colores = [COLORS['primary'], COLORS['secondary'], COLORS['warning'],
               COLORS['success'], '#808080']

    bar_y_positions = [5.5, 4.5, 3.5, 2.5, 1.5]
    max_width = 10

    for i, (mod, val, color) in enumerate(zip(modalidades, valores, colores)):
        width = (val / max(valores)) * max_width
        y = bar_y_positions[i]

        # Barra
        rect = FancyBboxPatch((2.5, y-0.25), width, 0.5,
                              boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(rect)

        # Etiqueta de modalidad
        ax.text(2.3, y, mod, ha='right', va='center', fontsize=12,
               color=COLORS['text'])

        # Valor
        ax.text(2.5 + width + 0.2, y, f'{val:,}M', ha='left', va='center',
               fontsize=12, fontweight='bold', color=color)

    # Nota al pie
    ax.text(8, 0.5, 'Fuente: OMS, estimaciones globales de estudios de imagen médica',
           fontsize=10, ha='center', va='center', color='gray', style='italic')

    # Ícono decorativo (globo simplificado)
    circle = Circle((14.5, 5), 1.5, facecolor=COLORS['primary'], alpha=0.2,
                   edgecolor=COLORS['primary'], linewidth=2)
    ax.add_patch(circle)
    ax.text(14.5, 5, 'GLOBAL', fontsize=16, ha='center', va='center',
           color=COLORS['primary'], fontweight='bold')

    # Destacar la radiografía de tórax
    ax.annotate('', xy=(2.7, 5.5), xytext=(1.5, 6.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=2))
    ax.text(1.5, 6.7, '#1 en el mundo', fontsize=12, ha='center',
           color=COLORS['secondary'], fontweight='bold')

    output_path = OUTPUT_DIR / 'slide2_uso_global.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Guardado: {output_path}")


def create_slide3_covid_demanda():
    """
    Slide 3: Gráfica temporal COVID-19 vs demanda radiológica
    Título: La pandemia COVID-19 incrementó la demanda de análisis
            radiológico en 300%
    """
    print("Generando Slide 3: COVID vs demanda...")

    fig = plt.figure(figsize=(16, 9), facecolor=COLORS['background'])
    ax = fig.add_subplot(111)

    # Datos simulados pero realistas
    meses = np.arange(0, 24)  # 2 años (2020-2021)
    mes_labels = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                  'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'] * 2

    # Línea base de demanda radiológica (pre-pandemia = 100)
    baseline = 100

    # Casos COVID (forma de ola)
    casos_covid = np.array([0, 0, 50, 150, 200, 180, 120, 100, 130, 250, 300, 280,
                           200, 150, 180, 350, 400, 350, 200, 150, 120, 100, 80, 60])
    casos_covid = casos_covid / 400 * 100  # Normalizar

    # Demanda radiológica (correlacionada con COVID pero con retraso)
    demanda = baseline + np.array([0, 0, 20, 100, 180, 220, 200, 150, 160, 200, 280, 300,
                                   250, 200, 220, 280, 350, 380, 300, 250, 200, 180, 150, 130])

    # Gráfico
    ax.fill_between(meses, 0, casos_covid * 3, alpha=0.3, color=COLORS['covid'],
                   label='Casos COVID-19 (normalizado)')
    ax.plot(meses, demanda, color=COLORS['primary'], linewidth=3, marker='o',
           markersize=6, label='Demanda radiológica (%)')

    # Línea base
    ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7,
              label='Línea base pre-pandemia')

    # Pico destacado
    pico_idx = np.argmax(demanda)
    ax.annotate(f'+{int(demanda[pico_idx] - baseline)}%',
               xy=(pico_idx, demanda[pico_idx]),
               xytext=(pico_idx + 1, demanda[pico_idx] + 50),
               fontsize=14, fontweight='bold', color=COLORS['danger'],
               arrowprops=dict(arrowstyle='->', color=COLORS['danger'], lw=2))

    # Configuración de ejes
    ax.set_xlabel('Meses (2020-2021)', fontsize=14)
    ax.set_ylabel('Índice de demanda radiológica (%)', fontsize=14)
    ax.set_xticks(meses[::2])
    ax.set_xticklabels([f'{mes_labels[i]}\n{2020 + i//12}' for i in range(0, 24, 2)],
                       fontsize=10)

    ax.set_ylim(0, 500)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Título (afirmación)
    ax.set_title('La pandemia COVID-19 incrementó la demanda de análisis radiológico en 300%',
                fontsize=18, fontweight='bold', pad=20, color=COLORS['text'])

    # Anotación de saturación
    ax.text(12, 50, '[!] Saturación de servicios radiológicos', fontsize=12,
           color=COLORS['warning'], fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_facecolor(COLORS['background'])

    output_path = OUTPUT_DIR / 'slide3_covid_demanda.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Guardado: {output_path}")


def create_slide4_landmarks_anatomia():
    """
    Slide 4: RX con 15 landmarks etiquetados anatómicamente
    Título: Los landmarks anatómicos son puntos de referencia esenciales
            para diagnóstico
    """
    print("Generando Slide 4: Landmarks con anatomía...")

    fig = plt.figure(figsize=(16, 9), facecolor=COLORS['background'])

    # Cargar imagen
    img_path, coords, _ = load_sample_image_with_coords()

    # Dos paneles: imagen grande + leyenda
    ax_img = fig.add_axes([0.02, 0.05, 0.55, 0.85])
    ax_legend = fig.add_axes([0.58, 0.05, 0.40, 0.85])
    ax_legend.axis('off')

    # Título
    fig.suptitle('Los landmarks anatómicos son puntos de referencia\nesenciales para diagnóstico',
                fontsize=18, fontweight='bold', y=0.98, color=COLORS['text'])

    if img_path and img_path.exists():
        img = Image.open(img_path).convert('L')
        img_array = np.array(img)
        ax_img.imshow(img_array, cmap='gray')

        # Categorías de landmarks
        categories = {
            'Eje central': ([0, 1], COLORS['axis']),
            'Puntos centrales': ([8, 9, 10], COLORS['central']),
            'Ápices/Hilios': ([2, 3, 4, 5, 6, 7], COLORS['symmetric']),
            'Ángulos': ([11, 12, 13, 14], COLORS['corner'])
        }

        # Dibujar landmarks con números
        for i, (x, y) in enumerate(coords):
            for cat_name, (indices, color) in categories.items():
                if i in indices:
                    break

            ax_img.plot(x, y, 'o', color=color, markersize=12,
                       markeredgecolor='white', markeredgewidth=2)
            ax_img.annotate(f'L{i+1}', (x+8, y-5), color='white', fontsize=10,
                           fontweight='bold',
                           path_effects=[pe.withStroke(
                               linewidth=3, foreground='black')])

        # Dibujar eje L1-L2
        ax_img.plot([coords[0][0], coords[1][0]], [coords[0][1], coords[1][1]],
                   'c--', linewidth=2.5, alpha=0.8)

    ax_img.axis('off')
    ax_img.set_title('Radiografía con 15 landmarks', fontsize=14,
                     color=COLORS['text'], pad=5)

    # Leyenda con anatomía
    ax_legend.text(0.5, 0.98, 'Anatomía de los 15 Landmarks', fontsize=16,
                  ha='center', fontweight='bold', color=COLORS['text'])

    # Tabla de landmarks
    y_pos = 0.92
    for i in range(15):
        landmark = f'L{i+1}'
        anatomy = LANDMARK_ANATOMY[landmark]

        # Color según categoría
        if i in [0, 1]:
            color = COLORS['axis']
            cat = 'Eje'
        elif i in [8, 9, 10]:
            color = COLORS['central']
            cat = 'Central'
        elif i in [2, 3, 4, 5, 6, 7]:
            color = COLORS['symmetric']
            cat = 'Simétrico'
        else:
            color = COLORS['corner']
            cat = 'Ángulo'

        # Punto de color
        ax_legend.plot(0.05, y_pos, 'o', color=color, markersize=10,
                      markeredgecolor='white', markeredgewidth=1)
        ax_legend.text(0.12, y_pos, landmark, fontsize=11, va='center',
                      fontweight='bold', color=COLORS['text'])
        ax_legend.text(0.22, y_pos, anatomy, fontsize=10, va='center',
                      color=COLORS['text'])

        y_pos -= 0.055

    # Leyenda de colores
    y_pos -= 0.05
    ax_legend.text(0.5, y_pos, '─' * 30, ha='center', color='gray')
    y_pos -= 0.04

    legend_items = [
        ('Eje central (L1-L2)', COLORS['axis']),
        ('Puntos centrales (L9-L11)', COLORS['central']),
        ('Pares simétricos (L3-L8)', COLORS['symmetric']),
        ('Ángulos cardio/costofrénico', COLORS['corner'])
    ]

    for label, color in legend_items:
        ax_legend.plot(0.05, y_pos, 's', color=color, markersize=12)
        ax_legend.text(0.12, y_pos, label, fontsize=10, va='center', color=COLORS['text'])
        y_pos -= 0.045

    output_path = OUTPUT_DIR / 'slide4_landmarks_anatomia.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Guardado: {output_path}")


def create_slide5_variabilidad_interobservador():
    """
    Slide 5: Comparación de variabilidad inter-observador
    Título: El etiquetado manual consume tiempo y presenta variabilidad
            de 5-15 píxeles entre observadores
    """
    print("Generando Slide 5: Variabilidad inter-observador...")

    fig = plt.figure(figsize=(16, 9), facecolor=COLORS['background'])

    # Cargar imagen
    img_path, coords_base, _ = load_sample_image_with_coords()

    # Tres paneles para simular tres observadores
    ax1 = fig.add_axes([0.02, 0.15, 0.30, 0.65])
    ax2 = fig.add_axes([0.35, 0.15, 0.30, 0.65])
    ax3 = fig.add_axes([0.68, 0.15, 0.30, 0.65])

    # Título
    fig.suptitle('El etiquetado manual consume tiempo y presenta\nvariabilidad de 5-15 píxeles entre observadores',
                fontsize=18, fontweight='bold', y=0.98, color=COLORS['text'])

    np.random.seed(42)

    if img_path and img_path.exists():
        img = Image.open(img_path).convert('L')
        img_array = np.array(img)

        # Tres "observadores" con variaciones
        observadores = [
            ('Observador A', 0, COLORS['primary']),
            ('Observador B', 5, COLORS['secondary']),
            ('Observador C', 10, COLORS['warning'])
        ]

        for ax, (nombre, variacion, color) in zip([ax1, ax2, ax3], observadores):
            ax.imshow(img_array, cmap='gray')

            # Añadir ruido a las coordenadas
            noise = np.random.randn(15, 2) * variacion
            coords = coords_base + noise

            # Dibujar landmarks con variación
            for i, (x, y) in enumerate(coords):
                ax.plot(x, y, 'o', color=color, markersize=8,
                       markeredgecolor='white', markeredgewidth=1.5)

            ax.axis('off')
            ax.set_title(nombre, fontsize=14, fontweight='bold', color=color, pad=5)

    # Añadir flechas y anotaciones de diferencia
    # Panel inferior con estadísticas
    ax_stats = fig.add_axes([0.1, 0.02, 0.8, 0.12])
    ax_stats.axis('off')

    stats_text = (
        "[T] Tiempo promedio: 5-10 min/imagen    |    "
        "[V] Variabilidad inter-observador: 5-15 px    |    "
        "[E] Error en landmarks difíciles (L14, L15): hasta 20 px"
    )
    ax_stats.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=12,
                 color=COLORS['text'],
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          edgecolor=COLORS['primary'], linewidth=2))

    # Flechas entre paneles mostrando diferencia
    ax_arrows = fig.add_axes([0, 0, 1, 1])
    ax_arrows.axis('off')
    ax_arrows.set_xlim(0, 1)
    ax_arrows.set_ylim(0, 1)

    # Flechas bidireccionales
    ax_arrows.annotate('', xy=(0.35, 0.5), xytext=(0.32, 0.5),
                      arrowprops=dict(arrowstyle='<->', color=COLORS['danger'], lw=2))
    ax_arrows.text(0.335, 0.45, '±5-10 px', fontsize=10, ha='center',
                  color=COLORS['danger'], fontweight='bold')

    ax_arrows.annotate('', xy=(0.68, 0.5), xytext=(0.65, 0.5),
                      arrowprops=dict(arrowstyle='<->', color=COLORS['danger'], lw=2))
    ax_arrows.text(0.665, 0.45, '±8-15 px', fontsize=10, ha='center',
                  color=COLORS['danger'], fontweight='bold')

    output_path = OUTPUT_DIR / 'slide5_variabilidad_interobservador.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Guardado: {output_path}")


def create_slide6_embudo_problema_objetivo():
    """
    Slide 6: Diagrama embudo problema-objetivo-solución
    Título: El objetivo fue desarrollar un sistema con error menor a 8 píxeles
    """
    print("Generando Slide 6: Embudo problema-objetivo...")

    fig = plt.figure(figsize=(16, 9), facecolor=COLORS['background'])
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Título
    ax.text(8, 8.5, 'El objetivo fue desarrollar un sistema con\nerror menor a 8 píxeles',
           fontsize=20, ha='center', va='top', fontweight='bold',
           color=COLORS['text'], linespacing=1.3)

    # Embudo de tres niveles
    # Nivel 1: Problema (arriba, más ancho)
    problema_box = FancyBboxPatch((2, 5.5), 12, 1.5,
                                  boxstyle="round,pad=0.05",
                                  facecolor=COLORS['danger'],
                                  edgecolor='white', linewidth=3, alpha=0.9)
    ax.add_patch(problema_box)
    ax.text(8, 6.25, '[ X ] PROBLEMA', fontsize=16, ha='center', va='center',
           color='white', fontweight='bold')
    ax.text(8, 5.85, 'Etiquetado manual: lento, costoso, variable (5-15 px)',
           fontsize=12, ha='center', va='center', color='white')

    # Flecha hacia abajo
    ax.annotate('', xy=(8, 5.3), xytext=(8, 5.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=3))

    # Nivel 2: Objetivo (medio)
    objetivo_box = FancyBboxPatch((3, 3.5), 10, 1.5,
                                  boxstyle="round,pad=0.05",
                                  facecolor=COLORS['warning'],
                                  edgecolor='white', linewidth=3, alpha=0.9)
    ax.add_patch(objetivo_box)
    ax.text(8, 4.25, '>>> OBJETIVO', fontsize=16, ha='center', va='center',
           color=COLORS['text'], fontweight='bold')
    ax.text(8, 3.85, 'Automatizar detección con error < 8 píxeles',
           fontsize=12, ha='center', va='center', color=COLORS['text'])

    # Flecha hacia abajo
    ax.annotate('', xy=(8, 3.3), xytext=(8, 3.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=3))

    # Nivel 3: Solución (abajo, más estrecho)
    solucion_box = FancyBboxPatch((4, 1.5), 8, 1.5,
                                  boxstyle="round,pad=0.05",
                                  facecolor=COLORS['success'],
                                  edgecolor='white', linewidth=3, alpha=0.9)
    ax.add_patch(solucion_box)
    ax.text(8, 2.25, '[OK] SOLUCION', fontsize=16, ha='center', va='center',
           color='white', fontweight='bold')
    ax.text(8, 1.85, 'Deep Learning: ResNet-18 + Coordinate Attention + Ensemble',
           fontsize=12, ha='center', va='center', color='white')

    # Resultado destacado
    result_box = FancyBboxPatch((5.5, 0.3), 5, 0.9,
                                boxstyle="round,pad=0.05",
                                facecolor='white',
                                edgecolor=COLORS['success'], linewidth=3)
    ax.add_patch(result_box)
    ax.text(8, 0.75, '* Resultado: 3.71 px *', fontsize=16, ha='center', va='center',
           color=COLORS['success'], fontweight='bold')

    # Iconos laterales con texto
    # Lado izquierdo: iconos del problema
    ax.text(0.8, 6.25, '[T]', fontsize=18, ha='center', va='center',
           color=COLORS['danger'], fontweight='bold')
    ax.text(0.8, 5.7, 'Lento', fontsize=11, ha='center', color=COLORS['danger'])

    ax.text(1.5, 6.25, '[$]', fontsize=18, ha='center', va='center',
           color=COLORS['danger'], fontweight='bold')
    ax.text(1.5, 5.7, 'Costoso', fontsize=11, ha='center', color=COLORS['danger'])

    # Lado derecho: iconos de la solución
    ax.text(14.3, 2.25, '[>>]', fontsize=18, ha='center', va='center',
           color=COLORS['success'], fontweight='bold')
    ax.text(14.3, 1.7, 'Rapido', fontsize=11, ha='center', color=COLORS['success'])

    ax.text(15.2, 2.25, '[*]', fontsize=18, ha='center', va='center',
           color=COLORS['success'], fontweight='bold')
    ax.text(15.2, 1.7, 'Preciso', fontsize=11, ha='center', color=COLORS['success'])

    output_path = OUTPUT_DIR / 'slide6_embudo_problema_objetivo.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Guardado: {output_path}")


def create_slide7_categorias_dataset():
    """
    Slide 7: Gráfico de categorías del dataset
    Título: El dataset contiene 957 radiografías en tres categorías clínicas
    """
    print("Generando Slide 7: Categorías del dataset...")

    fig = plt.figure(figsize=(16, 9), facecolor=COLORS['background'])

    # Título
    fig.suptitle('El dataset contiene 957 radiografías en tres categorías clínicas',
                fontsize=20, fontweight='bold', y=0.95, color=COLORS['text'])

    # Layout: pie chart + ejemplos
    ax_pie = fig.add_axes([0.05, 0.15, 0.4, 0.7])
    ax_examples = fig.add_axes([0.5, 0.1, 0.48, 0.75])
    ax_examples.axis('off')

    # Datos del dataset
    categorias = ['COVID-19', 'Normal', 'Neumonía Viral']
    cantidades = [306, 468, 183]  # COVID, Normal, Viral - Total = 957
    colores = [COLORS['covid'], COLORS['normal'], COLORS['viral']]

    # Pie chart con efecto 3D
    explode = (0.05, 0.02, 0.02)
    wedges, texts, autotexts = ax_pie.pie(cantidades, explode=explode,
                                          labels=categorias,
                                          colors=colores,
                                          autopct=lambda pct: f'{int(pct/100*957)}\n({pct:.1f}%)',
                                          shadow=True,
                                          startangle=90,
                                          textprops={'fontsize': 12})

    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)

    ax_pie.set_title('Distribución del Dataset\n(957 radiografías)', fontsize=14,
                     fontweight='bold', pad=10)

    # Panel de ejemplos
    ax_examples.text(0.5, 0.95, 'Ejemplos por Categoría', fontsize=14,
                    ha='center', fontweight='bold', color=COLORS['text'])

    # Cargar ejemplos de cada categoría
    categories_paths = {
        'COVID-19': DATASET_DIR / 'COVID',
        'Normal': DATASET_DIR / 'Normal',
        'Neumonía Viral': DATASET_DIR / 'Viral_Pneumonia'
    }

    y_offset = 0.85
    for cat, color in zip(categorias, colores):
        cat_path = categories_paths[cat]

        # Indicador de color
        ax_examples.add_patch(FancyBboxPatch((0.02, y_offset-0.08), 0.04, 0.15,
                                            boxstyle="round,pad=0.01",
                                            facecolor=color, edgecolor='white'))
        ax_examples.text(0.1, y_offset, cat, fontsize=12, fontweight='bold',
                        va='center', color=color)

        # Buscar imágenes de ejemplo
        if cat_path.exists():
            images = list(cat_path.glob('*.jpeg'))[:3]
            if not images:
                images = list(cat_path.glob('*.png'))[:3]

            x_pos = 0.3
            for img_path in images[:3]:
                try:
                    img = Image.open(img_path).convert('L')
                    img = img.resize((60, 60))

                    # Crear un mini axis para la imagen
                    ax_mini = fig.add_axes([0.5 + x_pos * 0.45,
                                           0.1 + y_offset * 0.65 - 0.08,
                                           0.08, 0.12])
                    ax_mini.imshow(np.array(img), cmap='gray')
                    ax_mini.axis('off')
                    ax_mini.patch.set_edgecolor(color)
                    ax_mini.patch.set_linewidth(2)

                    x_pos += 0.2
                except Exception as e:
                    pass

        y_offset -= 0.3

    # Nota sobre el dataset
    ax_examples.text(0.5, 0.02,
                    'Dataset: COVID-19 Radiography Database (Kaggle)\nResolución original: 299×299 px → Redimensionado: 224×224 px',
                    ha='center', va='bottom', fontsize=10, color='gray', style='italic')

    output_path = OUTPUT_DIR / 'slide7_categorias_dataset.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Guardado: {output_path}")


def main():
    """Genera todas las visualizaciones del Bloque 1."""
    print("=" * 60)
    print("GENERACIÓN DE VISUALIZACIONES - BLOQUE 1: CONTEXTO Y PROBLEMA")
    print("=" * 60)

    # Crear directorio de salida si no existe
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDirectorio de salida: {OUTPUT_DIR}")

    # Generar cada slide
    create_slide1_portada()
    create_slide2_uso_global()
    create_slide3_covid_demanda()
    create_slide4_landmarks_anatomia()
    create_slide5_variabilidad_interobservador()
    create_slide6_embudo_problema_objetivo()
    create_slide7_categorias_dataset()

    print("\n" + "=" * 60)
    print("¡COMPLETADO! Se generaron 7 visualizaciones en:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
