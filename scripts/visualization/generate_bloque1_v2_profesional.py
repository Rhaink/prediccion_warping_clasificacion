#!/usr/bin/env python3
"""
Script v2: Visualizaciones Bloque 1 - VERSION PROFESIONAL ACADEMICA

Cambios respecto a v1:
- Paleta profesional para tesis académica
- Resolución controlada (1600x900 max) para evitar errores de API
- Diseño minimalista y sobrio
- Mayor rigor científico en textos
- Sin elementos decorativos innecesarios

Estructura de salida:
presentacion/01_contexto/v2_profesional/
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import numpy as np
from pathlib import Path
from PIL import Image
import csv

# ============================================================================
# CONFIGURACION DE ESTILO PROFESIONAL ACADEMICO
# ============================================================================

# Paleta profesional para tesis
COLORS_PRO = {
    # Colores principales
    'text_primary': '#1a1a2e',      # Azul marino muy oscuro
    'text_secondary': '#4a4a4a',    # Gris oscuro
    'background': '#ffffff',         # Blanco puro
    'background_alt': '#f5f5f5',    # Gris muy claro

    # Acentos institucionales
    'accent_primary': '#003366',     # Azul institucional oscuro
    'accent_secondary': '#0066cc',   # Azul medio
    'accent_light': '#e6f0ff',       # Azul muy claro (fondos)

    # Colores para datos (sobrios)
    'data_1': '#003366',             # Azul oscuro
    'data_2': '#006633',             # Verde oscuro
    'data_3': '#cc6600',             # Naranja oscuro
    'data_4': '#660066',             # Púrpura oscuro
    'data_5': '#666666',             # Gris

    # Categorías del dataset
    'covid': '#c44536',              # Rojo ladrillo (sobrio)
    'normal': '#2d6a4f',             # Verde bosque
    'viral': '#b07d2b',              # Dorado oscuro

    # Landmarks
    'lm_axis': '#0077b6',            # Azul para eje
    'lm_central': '#2d6a4f',         # Verde para centrales
    'lm_symmetric': '#7b2cbf',       # Púrpura para simétricos
    'lm_corner': '#c44536',          # Rojo para esquinas

    # Estados
    'success': '#2d6a4f',
    'warning': '#b07d2b',
    'danger': '#c44536',
}

# Configuración de matplotlib para estilo académico
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'axes.linewidth': 0.8,
    'axes.edgecolor': COLORS_PRO['text_secondary'],
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'text.color': COLORS_PRO['text_primary'],
    'axes.labelcolor': COLORS_PRO['text_primary'],
    'xtick.color': COLORS_PRO['text_secondary'],
    'ytick.color': COLORS_PRO['text_secondary'],
})

# Rutas
BASE_DIR = Path('/home/donrobot/Projects/prediccion_coordenadas')
OUTPUT_DIR = BASE_DIR / 'presentacion' / '01_contexto' / 'v2_profesional'
DATA_DIR = BASE_DIR / 'data'
DATASET_DIR = DATA_DIR / 'dataset'

# Resolución controlada para evitar error de API (max 2000px)
FIG_WIDTH = 14  # pulgadas
FIG_HEIGHT = 7.875  # 16:9 ratio
DPI = 100  # 1400x787.5 px final

# Datos verificados del dataset
DATASET_STATS = {
    'COVID': 306,
    'Normal': 468,
    'Viral_Pneumonia': 183,
    'Total': 957
}

# Anatomía de landmarks
LANDMARK_ANATOMY = {
    'L1': 'Ápice traqueal',
    'L2': 'Bifurcación traqueal (carina)',
    'L3': 'Ápice pulmonar derecho',
    'L4': 'Ápice pulmonar izquierdo',
    'L5': 'Hilio pulmonar derecho (sup.)',
    'L6': 'Hilio pulmonar izquierdo (sup.)',
    'L7': 'Hilio pulmonar derecho (inf.)',
    'L8': 'Hilio pulmonar izquierdo (inf.)',
    'L9': 'Punto central superior',
    'L10': 'Punto central medio',
    'L11': 'Punto central inferior',
    'L12': 'Ángulo cardiofrénico derecho',
    'L13': 'Ángulo cardiofrénico izquierdo',
    'L14': 'Ángulo costofrénico derecho',
    'L15': 'Ángulo costofrénico izquierdo'
}


def create_directories():
    """Crea directorios de salida."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'assets').mkdir(exist_ok=True)


def load_sample_image_with_coords(category='Normal', index=50):
    """Carga imagen y coordenadas."""
    import pandas as pd
    csv_path = DATA_DIR / 'coordenadas' / 'coordenadas_maestro.csv'

    coord_cols = []
    for i in range(1, 16):
        coord_cols.extend([f'L{i}_x', f'L{i}_y'])
    columns = ['idx'] + coord_cols + ['image_name']

    df = pd.read_csv(csv_path, header=None, names=columns)

    if category == 'Normal':
        rows = df[df['image_name'].str.startswith('Normal')]
    elif category == 'COVID':
        rows = df[df['image_name'].str.startswith('COVID')]
    else:
        rows = df[df['image_name'].str.startswith('Viral')]

    row = rows.iloc[min(index, len(rows)-1)]
    image_name = row['image_name']

    cat_folder = 'COVID' if image_name.startswith('COVID') else \
                 'Normal' if image_name.startswith('Normal') else 'Viral_Pneumonia'

    img_path = DATASET_DIR / cat_folder / f"{image_name}.jpeg"
    if not img_path.exists():
        img_path = DATASET_DIR / cat_folder / f"{image_name}.png"

    coords = [(row[f'L{i}_x'], row[f'L{i}_y']) for i in range(1, 16)]
    return img_path, np.array(coords), image_name


def create_slide1_portada():
    """Slide 1: Portada profesional académica."""
    print("Generando Slide 1: Portada (v2)...")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=COLORS_PRO['background'])

    # Layout: texto izquierda, imagen derecha
    ax_text = fig.add_axes([0.03, 0.1, 0.42, 0.8])
    ax_img = fig.add_axes([0.48, 0.12, 0.48, 0.76])
    ax_text.axis('off')

    # Cargar imagen
    img_path, coords, _ = load_sample_image_with_coords('Normal', 50)

    if img_path and img_path.exists():
        img = Image.open(img_path).convert('L')
        ax_img.imshow(np.array(img), cmap='gray')

        # Dibujar landmarks con colores profesionales
        for i, (x, y) in enumerate(coords):
            if i in [0, 1]:
                color = COLORS_PRO['lm_axis']
            elif i in [8, 9, 10]:
                color = COLORS_PRO['lm_central']
            elif i in [2, 3, 4, 5, 6, 7]:
                color = COLORS_PRO['lm_symmetric']
            else:
                color = COLORS_PRO['lm_corner']

            ax_img.plot(x, y, 'o', color=color, markersize=8,
                       markeredgecolor='white', markeredgewidth=1.5)
            ax_img.annotate(f'L{i+1}', (x+6, y-6), color='white', fontsize=8,
                           fontweight='bold',
                           path_effects=[pe.withStroke(linewidth=2, foreground='black')])

        # Eje central L1-L2
        ax_img.plot([coords[0][0], coords[1][0]], [coords[0][1], coords[1][1]],
                   '--', color=COLORS_PRO['lm_axis'], linewidth=2, alpha=0.8)

    ax_img.axis('off')
    ax_img.set_title('Detección automática de landmarks anatómicos',
                     fontsize=12, color=COLORS_PRO['text_primary'], pad=8)

    # Contenido textual
    ax_text.text(0.5, 0.92, 'TESIS DE LICENCIATURA', fontsize=10,
                ha='center', color=COLORS_PRO['accent_secondary'],
                fontweight='bold')

    ax_text.text(0.5, 0.72,
                'Predicción Automática de\nLandmarks Anatómicos en\nRadiografías de Tórax\nmediante Deep Learning',
                fontsize=18, ha='center', va='center',
                color=COLORS_PRO['text_primary'],
                fontweight='bold', linespacing=1.5)

    # Métrica principal en caja
    rect = FancyBboxPatch((0.15, 0.28), 0.7, 0.22,
                          boxstyle="round,pad=0.02",
                          facecolor=COLORS_PRO['accent_light'],
                          edgecolor=COLORS_PRO['accent_primary'],
                          linewidth=1.5)
    ax_text.add_patch(rect)

    ax_text.text(0.5, 0.42, 'Error medio alcanzado', fontsize=11,
                ha='center', color=COLORS_PRO['text_secondary'])
    ax_text.text(0.5, 0.33, '3.71 píxeles', fontsize=28,
                ha='center', color=COLORS_PRO['accent_primary'], fontweight='bold')

    ax_text.text(0.5, 0.18, 'Objetivo: < 8 píxeles (variabilidad inter-observador)',
                fontsize=9, ha='center', color=COLORS_PRO['text_secondary'], style='italic')

    ax_text.text(0.5, 0.06, 'Arquitectura: ResNet-18 + Coordinate Attention + Ensemble',
                fontsize=9, ha='center', color=COLORS_PRO['text_secondary'])

    plt.savefig(OUTPUT_DIR / 'slide1_portada.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Guardado: {OUTPUT_DIR / 'slide1_portada.png'}")


def create_slide2_uso_global():
    """Slide 2: Uso global de radiografías - estilo académico."""
    print("Generando Slide 2: Uso global (v2)...")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=COLORS_PRO['background'])
    ax = fig.add_subplot(111)

    # Datos (estimaciones basadas en literatura)
    modalidades = ['Radiografía\n(tórax y otras)', 'Ultrasonido', 'Tomografía\ncomputarizada',
                   'Resonancia\nmagnética', 'Otras modalidades']
    valores = [2000, 800, 400, 150, 300]
    colores = [COLORS_PRO['data_1'], COLORS_PRO['data_2'], COLORS_PRO['data_3'],
               COLORS_PRO['data_4'], COLORS_PRO['data_5']]

    y_pos = np.arange(len(modalidades))
    bars = ax.barh(y_pos, valores, color=colores, edgecolor='white', linewidth=1, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(modalidades)
    ax.set_xlabel('Estudios anuales (millones)', fontsize=11)
    ax.invert_yaxis()

    # Valores en barras
    for bar, val in zip(bars, valores):
        width = bar.get_width()
        ax.text(width + 30, bar.get_y() + bar.get_height()/2,
               f'{val:,}M', va='center', fontsize=10, color=COLORS_PRO['text_primary'])

    ax.set_xlim(0, max(valores) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Título como afirmación
    ax.set_title('La radiografía es la modalidad de imagen médica más utilizada\ncon aproximadamente 2,000 millones de estudios anuales a nivel mundial',
                fontsize=13, fontweight='bold', color=COLORS_PRO['text_primary'], pad=15)

    # Nota de fuente
    fig.text(0.5, 0.02, 'Fuente: Estimaciones basadas en OMS y literatura radiológica (Defined Health)',
            ha='center', fontsize=8, color=COLORS_PRO['text_secondary'], style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(OUTPUT_DIR / 'slide2_uso_global.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Guardado: {OUTPUT_DIR / 'slide2_uso_global.png'}")


def create_slide3_covid_demanda():
    """Slide 3: Impacto COVID-19 en demanda radiológica."""
    print("Generando Slide 3: COVID-19 (v2)...")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=COLORS_PRO['background'])
    ax = fig.add_subplot(111)

    # Datos basados en literatura (incremento reportado: 200-400%)
    meses = np.arange(0, 24)
    baseline = 100

    # Curva de demanda más realista
    demanda = baseline + np.array([
        0, 5, 30, 80, 120, 140, 130, 110, 120, 150, 180, 200,
        170, 150, 160, 190, 220, 200, 170, 150, 130, 120, 110, 100
    ])

    # Casos COVID normalizados
    casos = np.array([
        0, 0, 40, 120, 160, 140, 100, 80, 100, 180, 240, 220,
        160, 120, 140, 260, 300, 260, 160, 120, 100, 80, 60, 40
    ]) / 3

    # Graficar
    ax.fill_between(meses, 0, casos, alpha=0.3, color=COLORS_PRO['danger'],
                   label='Casos COVID-19 (normalizado)')
    ax.plot(meses, demanda, color=COLORS_PRO['accent_primary'], linewidth=2.5,
           marker='o', markersize=5, label='Demanda de estudios radiológicos (%)')
    ax.axhline(y=baseline, color=COLORS_PRO['text_secondary'], linestyle='--',
              linewidth=1, alpha=0.7, label='Línea base pre-pandemia (100%)')

    # Anotación del pico
    pico_idx = np.argmax(demanda)
    pico_val = int(demanda[pico_idx] - baseline)
    ax.annotate(f'+{pico_val}%', xy=(pico_idx, demanda[pico_idx]),
               xytext=(pico_idx + 1.5, demanda[pico_idx] + 20),
               fontsize=12, fontweight='bold', color=COLORS_PRO['danger'],
               arrowprops=dict(arrowstyle='->', color=COLORS_PRO['danger'], lw=1.5))

    # Configuración de ejes
    mes_labels = ['Ene', 'Mar', 'May', 'Jul', 'Sep', 'Nov'] * 2
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f'{mes_labels[i//2]}\n{2020 + i//12}' for i in range(0, 24, 2)])
    ax.set_xlabel('Período (2020-2021)', fontsize=11)
    ax.set_ylabel('Índice de demanda radiológica (%)', fontsize=11)
    ax.set_ylim(0, 350)

    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title(f'La pandemia COVID-19 incrementó la demanda de análisis radiológico\nhasta en {pico_val}% respecto a la línea base',
                fontsize=13, fontweight='bold', color=COLORS_PRO['text_primary'], pad=15)

    fig.text(0.5, 0.02, 'Datos ilustrativos basados en reportes de saturación hospitalaria (2020-2021)',
            ha='center', fontsize=8, color=COLORS_PRO['text_secondary'], style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(OUTPUT_DIR / 'slide3_covid_demanda.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Guardado: {OUTPUT_DIR / 'slide3_covid_demanda.png'}")


def create_slide4_landmarks():
    """Slide 4: Landmarks anatómicos con tabla profesional."""
    print("Generando Slide 4: Landmarks (v2)...")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=COLORS_PRO['background'])

    # Dos paneles
    ax_img = fig.add_axes([0.02, 0.08, 0.45, 0.82])
    ax_table = fig.add_axes([0.50, 0.08, 0.48, 0.82])
    ax_table.axis('off')

    fig.suptitle('Los landmarks anatómicos son puntos de referencia reproducibles\nesenciales para el análisis cuantitativo de radiografías',
                fontsize=13, fontweight='bold', y=0.98, color=COLORS_PRO['text_primary'])

    # Imagen con landmarks
    img_path, coords, _ = load_sample_image_with_coords('Normal', 50)

    if img_path and img_path.exists():
        img = Image.open(img_path).convert('L')
        ax_img.imshow(np.array(img), cmap='gray')

        for i, (x, y) in enumerate(coords):
            if i in [0, 1]:
                color = COLORS_PRO['lm_axis']
            elif i in [8, 9, 10]:
                color = COLORS_PRO['lm_central']
            elif i in [2, 3, 4, 5, 6, 7]:
                color = COLORS_PRO['lm_symmetric']
            else:
                color = COLORS_PRO['lm_corner']

            ax_img.plot(x, y, 'o', color=color, markersize=9,
                       markeredgecolor='white', markeredgewidth=1.5)
            ax_img.annotate(f'{i+1}', (x+7, y-4), color='white', fontsize=8,
                           fontweight='bold',
                           path_effects=[pe.withStroke(linewidth=2, foreground='black')])

        ax_img.plot([coords[0][0], coords[1][0]], [coords[0][1], coords[1][1]],
                   '--', color=COLORS_PRO['lm_axis'], linewidth=2, alpha=0.8)

    ax_img.axis('off')
    ax_img.set_title('Radiografía PA con 15 landmarks', fontsize=11, pad=5)

    # Tabla de landmarks (posición ajustada para evitar superposición con suptitle)
    ax_table.text(0.5, 0.95, 'Definición anatómica de landmarks', fontsize=11,
                 ha='center', fontweight='bold', color=COLORS_PRO['text_primary'])

    # Encabezados
    ax_table.text(0.08, 0.92, 'ID', fontsize=9, fontweight='bold')
    ax_table.text(0.18, 0.92, 'Estructura anatómica', fontsize=9, fontweight='bold')
    ax_table.text(0.75, 0.92, 'Grupo', fontsize=9, fontweight='bold')
    ax_table.plot([0.05, 0.95], [0.90, 0.90], '-', color=COLORS_PRO['text_secondary'], linewidth=0.5)

    y_pos = 0.86
    for i in range(15):
        lm = f'L{i+1}'
        anatomy = LANDMARK_ANATOMY[lm]

        if i in [0, 1]:
            color, grupo = COLORS_PRO['lm_axis'], 'Eje'
        elif i in [8, 9, 10]:
            color, grupo = COLORS_PRO['lm_central'], 'Central'
        elif i in [2, 3, 4, 5, 6, 7]:
            color, grupo = COLORS_PRO['lm_symmetric'], 'Bilateral'
        else:
            color, grupo = COLORS_PRO['lm_corner'], 'Ángulo'

        ax_table.plot(0.04, y_pos, 's', color=color, markersize=8)
        ax_table.text(0.08, y_pos, lm, fontsize=9, va='center', fontweight='bold')
        ax_table.text(0.18, y_pos, anatomy, fontsize=8, va='center')
        ax_table.text(0.75, y_pos, grupo, fontsize=8, va='center', color=color)

        y_pos -= 0.052

    # Leyenda
    y_pos -= 0.03
    ax_table.plot([0.05, 0.95], [y_pos + 0.02, y_pos + 0.02], '-',
                 color=COLORS_PRO['text_secondary'], linewidth=0.5)

    leyenda = [
        ('Eje traqueal (L1-L2)', COLORS_PRO['lm_axis']),
        ('Puntos centrales (L9-L11)', COLORS_PRO['lm_central']),
        ('Pares bilaterales (L3-L8)', COLORS_PRO['lm_symmetric']),
        ('Ángulos diafragmáticos (L12-L15)', COLORS_PRO['lm_corner'])
    ]

    for label, color in leyenda:
        ax_table.plot(0.08, y_pos, 's', color=color, markersize=6)
        ax_table.text(0.14, y_pos, label, fontsize=8, va='center')
        y_pos -= 0.035

    plt.savefig(OUTPUT_DIR / 'slide4_landmarks.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Guardado: {OUTPUT_DIR / 'slide4_landmarks.png'}")


def create_slide5_variabilidad():
    """Slide 5: Variabilidad inter-observador."""
    print("Generando Slide 5: Variabilidad (v2)...")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=COLORS_PRO['background'])

    fig.suptitle('El etiquetado manual presenta variabilidad inter-observador\nde 5-15 píxeles, estableciendo el umbral de precisión aceptable',
                fontsize=13, fontweight='bold', y=0.97, color=COLORS_PRO['text_primary'])

    # Tres paneles
    ax1 = fig.add_axes([0.02, 0.15, 0.30, 0.70])
    ax2 = fig.add_axes([0.35, 0.15, 0.30, 0.70])
    ax3 = fig.add_axes([0.68, 0.15, 0.30, 0.70])

    img_path, coords_base, _ = load_sample_image_with_coords('Normal', 50)
    np.random.seed(42)

    observadores = [
        ('Observador 1\n(referencia)', 0, COLORS_PRO['accent_primary']),
        ('Observador 2\n(σ = 5 px)', 5, COLORS_PRO['data_2']),
        ('Observador 3\n(σ = 10 px)', 10, COLORS_PRO['data_3'])
    ]

    if img_path and img_path.exists():
        img = Image.open(img_path).convert('L')
        img_array = np.array(img)

        for ax, (nombre, variacion, color) in zip([ax1, ax2, ax3], observadores):
            ax.imshow(img_array, cmap='gray')

            noise = np.random.randn(15, 2) * variacion
            coords = coords_base + noise

            for i, (x, y) in enumerate(coords):
                ax.plot(x, y, 'o', color=color, markersize=6,
                       markeredgecolor='white', markeredgewidth=1)

            ax.axis('off')
            ax.set_title(nombre, fontsize=10, color=color, fontweight='bold', pad=5)

    # Estadísticas
    fig.text(0.5, 0.05,
            'Tiempo promedio de etiquetado: 5-10 min/imagen  •  '
            'Variabilidad típica: 5-15 px  •  '
            'Landmarks difíciles (L14-L15): hasta 20 px',
            ha='center', fontsize=9, color=COLORS_PRO['text_secondary'],
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS_PRO['background_alt'],
                     edgecolor=COLORS_PRO['text_secondary'], linewidth=0.5))

    plt.savefig(OUTPUT_DIR / 'slide5_variabilidad.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Guardado: {OUTPUT_DIR / 'slide5_variabilidad.png'}")


def create_slide6_objetivo():
    """Slide 6: Objetivo del proyecto."""
    print("Generando Slide 6: Objetivo (v2)...")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=COLORS_PRO['background'])
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.5, 'El objetivo fue desarrollar un sistema de detección automática\ncon error inferior a la variabilidad inter-observador (8 píxeles)',
           fontsize=13, ha='center', va='top', fontweight='bold',
           color=COLORS_PRO['text_primary'])

    # Diagrama de flujo vertical
    boxes = [
        (2, 5.2, 10, 1.2, 'PROBLEMA',
         'Etiquetado manual: costoso, lento y variable (5-15 px)',
         COLORS_PRO['danger'], 'white'),
        (2.5, 3.3, 9, 1.2, 'OBJETIVO',
         'Automatizar la detección con error < 8 píxeles',
         COLORS_PRO['warning'], COLORS_PRO['text_primary']),
        (3, 1.4, 8, 1.2, 'SOLUCIÓN',
         'Deep Learning: ResNet-18 + Coordinate Attention + Ensemble',
         COLORS_PRO['success'], 'white'),
    ]

    for x, y, w, h, titulo, texto, bg, fg in boxes:
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.03",
                              facecolor=bg, edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.65, titulo, fontsize=11, ha='center',
               va='center', color=fg, fontweight='bold')
        ax.text(x + w/2, y + h*0.3, texto, fontsize=9, ha='center',
               va='center', color=fg)

    # Flechas
    for y_start, y_end in [(5.2, 4.5), (3.3, 2.6)]:
        ax.annotate('', xy=(7, y_end), xytext=(7, y_start),
                   arrowprops=dict(arrowstyle='->', color=COLORS_PRO['text_secondary'], lw=2))

    # Resultado destacado
    result_rect = FancyBboxPatch((4.5, 0.2), 5, 0.8,
                                 boxstyle="round,pad=0.02",
                                 facecolor=COLORS_PRO['accent_light'],
                                 edgecolor=COLORS_PRO['success'], linewidth=2)
    ax.add_patch(result_rect)
    ax.text(7, 0.6, 'Resultado alcanzado: 3.71 píxeles', fontsize=12,
           ha='center', va='center', color=COLORS_PRO['success'], fontweight='bold')

    plt.savefig(OUTPUT_DIR / 'slide6_objetivo.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Guardado: {OUTPUT_DIR / 'slide6_objetivo.png'}")


def create_slide7_dataset():
    """Slide 7: Descripción del dataset."""
    print("Generando Slide 7: Dataset (v2)...")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=COLORS_PRO['background'])

    fig.suptitle(f'El dataset comprende {DATASET_STATS["Total"]} radiografías de tórax PA\ndistribuidas en tres categorías clínicas',
                fontsize=13, fontweight='bold', y=0.97, color=COLORS_PRO['text_primary'])

    # Pie chart
    ax_pie = fig.add_axes([0.05, 0.12, 0.40, 0.75])

    categorias = ['COVID-19', 'Normal', 'Neumonía viral']
    cantidades = [DATASET_STATS['COVID'], DATASET_STATS['Normal'],
                  DATASET_STATS['Viral_Pneumonia']]
    colores = [COLORS_PRO['covid'], COLORS_PRO['normal'], COLORS_PRO['viral']]

    # Usar valores exactos, no calculados del porcentaje
    def make_autopct(valores):
        def autopct(pct):
            total = sum(valores)
            val = int(round(pct*total/100.0))
            # Buscar el valor más cercano en la lista original
            closest = min(valores, key=lambda x: abs(x - val))
            return f'{closest}\n({closest/total*100:.1f}%)'
        return autopct

    wedges, texts, autotexts = ax_pie.pie(
        cantidades,
        labels=categorias,
        colors=colores,
        autopct=make_autopct(cantidades),
        startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=2),
        textprops={'fontsize': 10}
    )
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')

    ax_pie.set_title(f'Distribución del dataset\n(n = {sum(cantidades)})',
                     fontsize=11, pad=5)

    # Panel de ejemplos
    ax_ex = fig.add_axes([0.50, 0.12, 0.48, 0.75])
    ax_ex.axis('off')
    ax_ex.text(0.5, 0.95, 'Ejemplos representativos', fontsize=11,
              ha='center', fontweight='bold')

    # Cargar ejemplos
    categories_map = [
        ('COVID-19', 'COVID', COLORS_PRO['covid']),
        ('Normal', 'Normal', COLORS_PRO['normal']),
        ('Neumonía viral', 'Viral_Pneumonia', COLORS_PRO['viral'])
    ]

    y_offset = 0.85
    for cat_name, folder, color in categories_map:
        ax_ex.text(0.02, y_offset, cat_name, fontsize=9, va='center',
                  fontweight='bold', color=color)

        cat_path = DATASET_DIR / folder
        if cat_path.exists():
            images = list(cat_path.glob('*.jpeg'))[:3]
            if not images:
                images = list(cat_path.glob('*.png'))[:3]

            x_pos = 0.25
            for img_path in images[:3]:
                try:
                    img = Image.open(img_path).convert('L').resize((50, 50))
                    ax_mini = fig.add_axes([0.50 + x_pos * 0.45,
                                           0.12 + y_offset * 0.65 - 0.05,
                                           0.06, 0.10])
                    ax_mini.imshow(np.array(img), cmap='gray')
                    ax_mini.axis('off')
                    for spine in ax_mini.spines.values():
                        spine.set_edgecolor(color)
                        spine.set_linewidth(1)
                    x_pos += 0.22
                except:
                    pass

        y_offset -= 0.30

    # Nota sobre el dataset
    fig.text(0.5, 0.03,
            'Fuente: COVID-19 Radiography Database (Kaggle). Resolución: 299×299 px → 224×224 px',
            ha='center', fontsize=8, color=COLORS_PRO['text_secondary'], style='italic')

    plt.savefig(OUTPUT_DIR / 'slide7_dataset.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Guardado: {OUTPUT_DIR / 'slide7_dataset.png'}")


def main():
    """Genera todas las visualizaciones v2."""
    print("=" * 65)
    print("BLOQUE 1 - VERSION 2: ESTILO PROFESIONAL ACADEMICO")
    print("=" * 65)
    print(f"Resolución: {int(FIG_WIDTH*DPI)}x{int(FIG_HEIGHT*DPI)} px (max)")
    print(f"Salida: {OUTPUT_DIR}")
    print()

    create_directories()

    create_slide1_portada()
    create_slide2_uso_global()
    create_slide3_covid_demanda()
    create_slide4_landmarks()
    create_slide5_variabilidad()
    create_slide6_objetivo()
    create_slide7_dataset()

    print()
    print("=" * 65)
    print("COMPLETADO - 7 slides generadas en estilo profesional")
    print("=" * 65)


if __name__ == '__main__':
    main()
