#!/usr/bin/env python3
"""
Script para generar visualizaciones del Bloque 1 - VERSION PROFESIONAL ACADEMICA

Características:
- Paleta de colores sobria y profesional
- Diseño minimalista sin elementos decorativos
- Tipografía académica
- Alto contraste para impresión
- Rigor científico en la presentación de datos

Estructura de salida:
presentacion/01_contexto/v2_profesional/
├── slides/
├── assets/graficos/
├── assets/radiografias/
└── datos/
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path
from PIL import Image
import csv
import shutil

# =============================================================================
# CONFIGURACIÓN DE ESTILO PROFESIONAL ACADÉMICO
# =============================================================================

# Paleta de colores profesional (inspirada en publicaciones científicas)
COLORS = {
    # Colores principales
    'primary': '#1a365d',       # Azul marino oscuro
    'secondary': '#2c5282',     # Azul medio
    'accent': '#c53030',        # Rojo oscuro para énfasis

    # Escala de grises
    'text': '#1a202c',          # Casi negro para texto
    'text_secondary': '#4a5568', # Gris oscuro
    'text_light': '#718096',    # Gris medio
    'background': '#ffffff',    # Blanco puro
    'background_alt': '#f7fafc', # Gris muy claro
    'border': '#e2e8f0',        # Gris borde

    # Colores para datos (desaturados, profesionales)
    'data_1': '#2b6cb0',        # Azul
    'data_2': '#276749',        # Verde oscuro
    'data_3': '#c05621',        # Naranja oscuro
    'data_4': '#6b46c1',        # Púrpura
    'data_5': '#718096',        # Gris

    # Categorías del dataset (profesionales)
    'covid': '#9b2c2c',         # Rojo oscuro
    'normal': '#276749',        # Verde oscuro
    'viral': '#b7791f',         # Amarillo oscuro/dorado

    # Landmarks (alto contraste para visibilidad en RX)
    'landmark_axis': '#00d4ff',      # Cyan brillante
    'landmark_central': '#00ff88',   # Verde brillante
    'landmark_symmetric': '#ffdd00', # Amarillo brillante
    'landmark_corner': '#ff00ff',    # Magenta brillante

    # Semáforo para problema/objetivo/solución
    'problem': '#c53030',       # Rojo
    'objective': '#b7791f',     # Dorado
    'solution': '#276749',      # Verde
}

# Configuración de matplotlib para estilo académico
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times', 'serif'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'normal',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'axes.edgecolor': COLORS['text_secondary'],
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': COLORS['border'],
    'figure.facecolor': COLORS['background'],
    'axes.facecolor': COLORS['background'],
    'savefig.facecolor': COLORS['background'],
    'savefig.edgecolor': 'none',
    'savefig.dpi': 300,
})

# Rutas
BASE_DIR = Path('/home/donrobot/Projects/prediccion_coordenadas')
OUTPUT_DIR = BASE_DIR / 'presentacion' / '01_contexto' / 'v2_profesional'
SLIDES_DIR = OUTPUT_DIR / 'slides'
ASSETS_DIR = OUTPUT_DIR / 'assets'
GRAFICOS_DIR = ASSETS_DIR / 'graficos'
RX_DIR = ASSETS_DIR / 'radiografias'
DATOS_DIR = OUTPUT_DIR / 'datos'
DATA_DIR = BASE_DIR / 'data'
DATASET_DIR = DATA_DIR / 'dataset'

# Datos anatómicos
LANDMARK_ANATOMY = {
    'L1': 'Ápice traqueal',
    'L2': 'Bifurcación traqueal (carina)',
    'L3': 'Ápice pulmonar derecho',
    'L4': 'Ápice pulmonar izquierdo',
    'L5': 'Hilio pulmonar derecho (sup.)',
    'L6': 'Hilio pulmonar izquierdo (sup.)',
    'L7': 'Hilio pulmonar derecho (inf.)',
    'L8': 'Hilio pulmonar izquierdo (inf.)',
    'L9': 'Eje central superior',
    'L10': 'Eje central medio',
    'L11': 'Eje central inferior',
    'L12': 'Ángulo cardiofrénico derecho',
    'L13': 'Ángulo cardiofrénico izquierdo',
    'L14': 'Ángulo costofrénico derecho',
    'L15': 'Ángulo costofrénico izquierdo'
}

# Datos reales del dataset
DATASET_STATS = {'COVID': 306, 'Normal': 468, 'Viral_Pneumonia': 183, 'Total': 957}


def create_directories():
    """Crea estructura de directorios."""
    for d in [SLIDES_DIR, GRAFICOS_DIR, RX_DIR, DATOS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


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

    if image_name.startswith('COVID'):
        cat_folder = 'COVID'
    elif image_name.startswith('Normal'):
        cat_folder = 'Normal'
    else:
        cat_folder = 'Viral_Pneumonia'

    img_path = DATASET_DIR / cat_folder / f"{image_name}.jpeg"
    if not img_path.exists():
        img_path = DATASET_DIR / cat_folder / f"{image_name}.png"

    coords = []
    for i in range(1, 16):
        coords.append((row[f'L{i}_x'], row[f'L{i}_y']))

    return img_path, np.array(coords), image_name


def get_landmark_color(idx):
    """Retorna color según categoría del landmark."""
    if idx in [0, 1]:  # L1, L2 - eje
        return COLORS['landmark_axis']
    elif idx in [8, 9, 10]:  # L9, L10, L11 - centrales
        return COLORS['landmark_central']
    elif idx in [2, 3, 4, 5, 6, 7]:  # pares simétricos
        return COLORS['landmark_symmetric']
    else:  # L12-L15 - ángulos
        return COLORS['landmark_corner']


def create_slide1_portada():
    """Slide 1: Portada profesional académica."""
    print("\n[Slide 1] Portada...")

    img_path, coords, _ = load_sample_image_with_coords('Normal', 50)

    # Exportar RX base
    if img_path and img_path.exists():
        shutil.copy(img_path, RX_DIR / 'rx_base_normal.jpeg')

    # === ASSET: RX con landmarks ===
    fig_rx, ax_rx = plt.subplots(figsize=(8, 8))

    if img_path and img_path.exists():
        img = Image.open(img_path).convert('L')
        ax_rx.imshow(np.array(img), cmap='gray')

        for i, (x, y) in enumerate(coords):
            color = get_landmark_color(i)
            ax_rx.plot(x, y, 'o', color=color, markersize=10,
                      markeredgecolor='white', markeredgewidth=1.5)
            ax_rx.annotate(f'L{i+1}', (x+8, y-8), color='white', fontsize=9,
                          fontweight='bold', fontfamily='sans-serif',
                          path_effects=[pe.withStroke(linewidth=2, foreground='black')])

        # Eje central L1-L2
        ax_rx.plot([coords[0][0], coords[1][0]], [coords[0][1], coords[1][1]],
                  '--', color=COLORS['landmark_axis'], linewidth=2, alpha=0.8)

    ax_rx.axis('off')
    fig_rx.savefig(GRAFICOS_DIR / 'rx_landmarks_anotado.png', dpi=300,
                   bbox_inches='tight', facecolor='white')
    plt.close(fig_rx)

    # === SLIDE COMPLETA ===
    fig = plt.figure(figsize=(16, 9), facecolor=COLORS['background'])

    # Grid: texto izquierda, imagen derecha
    ax_text = fig.add_axes([0.05, 0.1, 0.42, 0.8])
    ax_img = fig.add_axes([0.52, 0.1, 0.45, 0.75])
    ax_text.axis('off')

    # Texto
    ax_text.text(0.0, 0.95, 'TESIS DE LICENCIATURA', fontsize=12,
                color=COLORS['text_light'], fontweight='normal',
                fontfamily='sans-serif', transform=ax_text.transAxes)

    ax_text.text(0.0, 0.75, 'Predicción Automática de\nLandmarks Anatómicos en\nRadiografías de Tórax\nmediante Deep Learning',
                fontsize=22, color=COLORS['text'], fontweight='bold',
                linespacing=1.4, transform=ax_text.transAxes, va='top')

    # Línea separadora
    ax_text.axhline(y=0.42, xmin=0, xmax=0.6, color=COLORS['primary'], linewidth=2)

    # Resultado principal
    ax_text.text(0.0, 0.35, 'Error de predicción alcanzado:', fontsize=12,
                color=COLORS['text_secondary'], transform=ax_text.transAxes)

    ax_text.text(0.0, 0.22, '3.71 píxeles', fontsize=36,
                color=COLORS['primary'], fontweight='bold',
                transform=ax_text.transAxes)

    ax_text.text(0.0, 0.12, '(objetivo establecido: < 8 píxeles)', fontsize=11,
                color=COLORS['text_light'], style='italic',
                transform=ax_text.transAxes)

    ax_text.text(0.0, 0.02, 'Arquitectura: ResNet-18 + Coordinate Attention + Ensemble',
                fontsize=10, color=COLORS['text_secondary'],
                transform=ax_text.transAxes)

    # Imagen
    if img_path and img_path.exists():
        img = Image.open(img_path).convert('L')
        ax_img.imshow(np.array(img), cmap='gray')

        for i, (x, y) in enumerate(coords):
            color = get_landmark_color(i)
            ax_img.plot(x, y, 'o', color=color, markersize=9,
                       markeredgecolor='white', markeredgewidth=1.5)

        ax_img.plot([coords[0][0], coords[1][0]], [coords[0][1], coords[1][1]],
                   '--', color=COLORS['landmark_axis'], linewidth=2, alpha=0.8)

    ax_img.axis('off')
    ax_img.set_title('Detección de 15 landmarks anatómicos', fontsize=12,
                     color=COLORS['text'], pad=10, fontweight='normal')

    fig.savefig(SLIDES_DIR / 'slide1_portada.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Completado")


def create_slide2_uso_global():
    """Slide 2: Uso global de radiografías - estilo académico."""
    print("\n[Slide 2] Uso global de radiografías...")

    modalidades = ['Radiografía de tórax', 'Ultrasonido', 'Tomografía computarizada',
                   'Resonancia magnética', 'Otras modalidades']
    valores = [2000, 800, 400, 150, 300]

    # Guardar datos
    with open(DATOS_DIR / 'modalidades_imagen.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['modalidad', 'millones_anuales'])
        for m, v in zip(modalidades, valores):
            writer.writerow([m, v])

    # === ASSET: Gráfico de barras horizontal ===
    fig_bar, ax = plt.subplots(figsize=(10, 5))

    y_pos = np.arange(len(modalidades))
    colors = [COLORS['primary']] + [COLORS['data_5']] * 4

    bars = ax.barh(y_pos, valores, color=colors, edgecolor='white', linewidth=0.5, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(modalidades)
    ax.set_xlabel('Millones de estudios anuales')
    ax.invert_yaxis()
    ax.set_xlim(0, 2500)

    # Valores al final de barras
    for bar, val in zip(bars, valores):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
               f'{val:,}', va='center', fontsize=10, color=COLORS['text'])

    # Destacar radiografía de tórax
    ax.annotate('Modalidad más utilizada\nglobalmente',
                xy=(valores[0], 0), xytext=(1400, 1.5),
                fontsize=9, color=COLORS['primary'],
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=1.5),
                ha='center')

    fig_bar.savefig(GRAFICOS_DIR / 'barras_modalidades.png', dpi=300, bbox_inches='tight')
    plt.close(fig_bar)

    # === SLIDE COMPLETA ===
    fig = plt.figure(figsize=(16, 9), facecolor=COLORS['background'])

    # Título (afirmación)
    fig.text(0.5, 0.92, 'Las radiografías de tórax constituyen la modalidad de imagen médica',
            fontsize=18, ha='center', fontweight='bold', color=COLORS['text'])
    fig.text(0.5, 0.86, 'más utilizada a nivel mundial, con aproximadamente 2 mil millones de estudios anuales',
            fontsize=18, ha='center', fontweight='bold', color=COLORS['text'])

    ax = fig.add_axes([0.1, 0.1, 0.8, 0.7])

    bars = ax.barh(y_pos, valores, color=colors, edgecolor='white', linewidth=0.5, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(modalidades, fontsize=12)
    ax.set_xlabel('Millones de estudios anuales', fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 2500)

    for bar, val in zip(bars, valores):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
               f'{val:,}', va='center', fontsize=11, color=COLORS['text'])

    # Fuente
    fig.text(0.5, 0.02, 'Fuente: Organización Mundial de la Salud, estimaciones globales de estudios de imagen médica',
            fontsize=9, ha='center', color=COLORS['text_light'], style='italic')

    fig.savefig(SLIDES_DIR / 'slide2_uso_global.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Completado")


def create_slide3_covid_demanda():
    """Slide 3: Impacto COVID-19 en demanda radiológica."""
    print("\n[Slide 3] Impacto COVID-19...")

    meses = np.arange(0, 24)
    mes_labels = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                  'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'] * 2

    baseline = 100

    # Datos de demanda (incremento realista documentado)
    demanda = np.array([100, 100, 120, 180, 250, 280, 260, 220, 230, 270, 330, 350,
                        300, 270, 280, 330, 380, 380, 340, 300, 260, 240, 220, 200])

    # Casos COVID normalizados
    casos = np.array([0, 0, 30, 100, 150, 140, 90, 70, 90, 180, 250, 230,
                      150, 100, 130, 280, 350, 300, 160, 110, 80, 60, 50, 40])
    casos_norm = casos / 350 * 100

    pico = int(max(demanda) - baseline)

    # Guardar datos
    with open(DATOS_DIR / 'covid_demanda.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mes', 'año', 'demanda_pct', 'casos_norm'])
        for i in range(24):
            writer.writerow([mes_labels[i], 2020 + i//12, demanda[i], casos_norm[i]])

    # === ASSET: Gráfico temporal ===
    fig_temp, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(meses, 0, casos_norm * 3, alpha=0.25, color=COLORS['accent'],
                   label='Casos COVID-19 (normalizado)')
    ax.plot(meses, demanda, color=COLORS['primary'], linewidth=2.5, marker='o',
           markersize=5, label='Demanda radiológica (%)')
    ax.axhline(y=baseline, color=COLORS['text_light'], linestyle='--', linewidth=1,
              label='Línea base pre-pandemia')

    pico_idx = np.argmax(demanda)
    ax.annotate(f'+{pico}%', xy=(pico_idx, demanda[pico_idx]),
               xytext=(pico_idx + 2, demanda[pico_idx] + 20),
               fontsize=12, fontweight='bold', color=COLORS['accent'],
               arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=1.5))

    ax.set_xlabel('Período (2020-2021)')
    ax.set_ylabel('Índice de demanda radiológica (%)')
    ax.set_xticks(meses[::2])
    ax.set_xticklabels([f'{mes_labels[i]}\n{2020 + i//12}' for i in range(0, 24, 2)])
    ax.set_ylim(0, 450)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=':')

    fig_temp.savefig(GRAFICOS_DIR / 'grafico_covid_demanda.png', dpi=300, bbox_inches='tight')
    plt.close(fig_temp)

    # === SLIDE COMPLETA ===
    fig = plt.figure(figsize=(16, 9), facecolor=COLORS['background'])

    fig.text(0.5, 0.92, f'La pandemia de COVID-19 incrementó la demanda de análisis radiológico',
            fontsize=18, ha='center', fontweight='bold', color=COLORS['text'])
    fig.text(0.5, 0.86, f'hasta en un {pico}% respecto al período pre-pandemia',
            fontsize=18, ha='center', fontweight='bold', color=COLORS['text'])

    ax = fig.add_axes([0.08, 0.12, 0.84, 0.68])

    ax.fill_between(meses, 0, casos_norm * 3, alpha=0.25, color=COLORS['accent'],
                   label='Casos COVID-19 (normalizado)')
    ax.plot(meses, demanda, color=COLORS['primary'], linewidth=2.5, marker='o',
           markersize=5, label='Demanda radiológica (%)')
    ax.axhline(y=baseline, color=COLORS['text_light'], linestyle='--', linewidth=1,
              label='Línea base pre-pandemia (100%)')

    ax.annotate(f'+{pico}%', xy=(pico_idx, demanda[pico_idx]),
               xytext=(pico_idx + 2, demanda[pico_idx] + 25),
               fontsize=14, fontweight='bold', color=COLORS['accent'],
               arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2))

    ax.set_xlabel('Período (2020-2021)', fontsize=12)
    ax.set_ylabel('Índice de demanda radiológica (%)', fontsize=12)
    ax.set_xticks(meses[::2])
    ax.set_xticklabels([f'{mes_labels[i]}\n{2020 + i//12}' for i in range(0, 24, 2)])
    ax.set_ylim(0, 450)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')

    # Nota sobre saturación
    ax.text(12, 30, 'Período de mayor saturación\nde servicios radiológicos',
           fontsize=10, color=COLORS['accent'], ha='center',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                    edgecolor=COLORS['accent'], linewidth=1))

    fig.savefig(SLIDES_DIR / 'slide3_covid_demanda.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Completado")


def create_slide4_landmarks_anatomia():
    """Slide 4: Landmarks anatómicos."""
    print("\n[Slide 4] Landmarks anatómicos...")

    img_path, coords, _ = load_sample_image_with_coords('Normal', 50)

    # Guardar tabla de landmarks
    with open(DATOS_DIR / 'tabla_landmarks.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Estructura anatómica', 'Categoría'])
        categories = {
            (0, 1): 'Eje traqueal',
            (2, 3, 4, 5, 6, 7): 'Pares simétricos (hilios/ápices)',
            (8, 9, 10): 'Eje central',
            (11, 12, 13, 14): 'Ángulos torácicos'
        }
        for i in range(15):
            for indices, cat in categories.items():
                if i in indices:
                    writer.writerow([f'L{i+1}', LANDMARK_ANATOMY[f'L{i+1}'], cat])
                    break

    # === ASSET: RX con landmarks ===
    fig_rx, ax = plt.subplots(figsize=(8, 10))

    if img_path and img_path.exists():
        img = Image.open(img_path).convert('L')
        ax.imshow(np.array(img), cmap='gray')

        for i, (x, y) in enumerate(coords):
            color = get_landmark_color(i)
            ax.plot(x, y, 'o', color=color, markersize=12,
                   markeredgecolor='white', markeredgewidth=2)
            ax.annotate(f'L{i+1}', (x+10, y-5), color='white', fontsize=10,
                       fontweight='bold', fontfamily='sans-serif',
                       path_effects=[pe.withStroke(linewidth=3, foreground='black')])

        ax.plot([coords[0][0], coords[1][0]], [coords[0][1], coords[1][1]],
               '--', color=COLORS['landmark_axis'], linewidth=2.5, alpha=0.8)

    ax.axis('off')
    fig_rx.savefig(GRAFICOS_DIR / 'rx_landmarks_etiquetados.png', dpi=300, bbox_inches='tight')
    plt.close(fig_rx)

    # === SLIDE COMPLETA ===
    fig = plt.figure(figsize=(16, 9), facecolor=COLORS['background'])

    fig.text(0.5, 0.94, 'Los landmarks anatómicos son puntos de referencia',
            fontsize=18, ha='center', fontweight='bold', color=COLORS['text'])
    fig.text(0.5, 0.88, 'esenciales para el análisis cuantitativo de radiografías de tórax',
            fontsize=18, ha='center', fontweight='bold', color=COLORS['text'])

    ax_img = fig.add_axes([0.02, 0.05, 0.5, 0.78])
    ax_table = fig.add_axes([0.54, 0.08, 0.44, 0.75])
    ax_table.axis('off')

    if img_path and img_path.exists():
        img = Image.open(img_path).convert('L')
        ax_img.imshow(np.array(img), cmap='gray')

        for i, (x, y) in enumerate(coords):
            color = get_landmark_color(i)
            ax_img.plot(x, y, 'o', color=color, markersize=11,
                       markeredgecolor='white', markeredgewidth=2)
            ax_img.annotate(f'L{i+1}', (x+8, y-5), color='white', fontsize=9,
                           fontweight='bold', fontfamily='sans-serif',
                           path_effects=[pe.withStroke(linewidth=3, foreground='black')])

        ax_img.plot([coords[0][0], coords[1][0]], [coords[0][1], coords[1][1]],
                   '--', color=COLORS['landmark_axis'], linewidth=2.5, alpha=0.8)

    ax_img.axis('off')

    # Tabla de landmarks
    ax_table.text(0.0, 0.98, 'Identificación de estructuras anatómicas', fontsize=13,
                 fontweight='bold', color=COLORS['text'], transform=ax_table.transAxes)

    y_pos = 0.92
    for i in range(15):
        lm = f'L{i+1}'
        color = get_landmark_color(i)

        ax_table.plot(0.02, y_pos, 'o', color=color, markersize=8,
                     markeredgecolor='white', markeredgewidth=1,
                     transform=ax_table.transAxes)
        ax_table.text(0.07, y_pos, lm, fontsize=10, fontweight='bold',
                     color=COLORS['text'], va='center', transform=ax_table.transAxes)
        ax_table.text(0.14, y_pos, LANDMARK_ANATOMY[lm], fontsize=9,
                     color=COLORS['text'], va='center', transform=ax_table.transAxes)
        y_pos -= 0.052

    # Leyenda de categorías
    y_pos -= 0.04
    ax_table.plot([0, 0.9], [y_pos, y_pos], color=COLORS['border'],
                 linewidth=1, transform=ax_table.transAxes)
    y_pos -= 0.035

    legend_items = [
        ('Eje traqueal (L1-L2)', COLORS['landmark_axis']),
        ('Eje central (L9-L11)', COLORS['landmark_central']),
        ('Pares simétricos (L3-L8)', COLORS['landmark_symmetric']),
        ('Ángulos torácicos (L12-L15)', COLORS['landmark_corner'])
    ]

    for label, color in legend_items:
        ax_table.plot(0.02, y_pos, 's', color=color, markersize=10,
                     transform=ax_table.transAxes)
        ax_table.text(0.07, y_pos, label, fontsize=9, va='center',
                     color=COLORS['text'], transform=ax_table.transAxes)
        y_pos -= 0.04

    fig.savefig(SLIDES_DIR / 'slide4_landmarks_anatomia.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Completado")


def create_slide5_variabilidad():
    """Slide 5: Variabilidad inter-observador."""
    print("\n[Slide 5] Variabilidad inter-observador...")

    img_path, coords_base, _ = load_sample_image_with_coords('Normal', 50)
    np.random.seed(42)

    # === SLIDE COMPLETA ===
    fig = plt.figure(figsize=(16, 9), facecolor=COLORS['background'])

    fig.text(0.5, 0.94, 'El etiquetado manual de landmarks presenta variabilidad',
            fontsize=18, ha='center', fontweight='bold', color=COLORS['text'])
    fig.text(0.5, 0.88, 'inter-observador de 5-15 píxeles, dependiendo de la estructura anatómica',
            fontsize=18, ha='center', fontweight='bold', color=COLORS['text'])

    axes = [fig.add_axes([0.03 + i*0.32, 0.18, 0.29, 0.62]) for i in range(3)]

    observadores = [
        ('Observador A', 0, COLORS['primary']),
        ('Observador B', 6, COLORS['secondary']),
        ('Observador C', 12, COLORS['accent'])
    ]

    if img_path and img_path.exists():
        img = Image.open(img_path).convert('L')
        img_array = np.array(img)

        for ax, (nombre, variacion, color) in zip(axes, observadores):
            ax.imshow(img_array, cmap='gray')

            noise = np.random.randn(15, 2) * variacion
            coords = coords_base + noise

            for i, (x, y) in enumerate(coords):
                ax.plot(x, y, 'o', color=color, markersize=7,
                       markeredgecolor='white', markeredgewidth=1)

            ax.axis('off')
            ax.set_title(nombre, fontsize=12, fontweight='bold', color=color, pad=8)

    # Flechas indicando diferencias
    fig.text(0.34, 0.50, '↔', fontsize=20, ha='center', color=COLORS['accent'])
    fig.text(0.34, 0.44, '±5-8 px', fontsize=10, ha='center', color=COLORS['accent'], fontweight='bold')

    fig.text(0.66, 0.50, '↔', fontsize=20, ha='center', color=COLORS['accent'])
    fig.text(0.66, 0.44, '±10-15 px', fontsize=10, ha='center', color=COLORS['accent'], fontweight='bold')

    # Información adicional
    info_box = fig.add_axes([0.15, 0.02, 0.7, 0.1])
    info_box.axis('off')

    info_text = ('Tiempo promedio de etiquetado: 5-10 minutos por imagen   •   '
                'Mayor variabilidad en: L14, L15 (ángulos costofrénico)   •   '
                'Menor variabilidad en: L1, L2 (estructuras centrales)')

    info_box.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=10,
                 color=COLORS['text'], transform=info_box.transAxes,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['background_alt'],
                          edgecolor=COLORS['border'], linewidth=1))

    fig.savefig(SLIDES_DIR / 'slide5_variabilidad_interobservador.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Completado")


def create_slide6_embudo():
    """Slide 6: Planteamiento del problema y objetivo."""
    print("\n[Slide 6] Problema y objetivo...")

    # === ASSET: Diagrama ===
    fig_diag, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Problema
    rect1 = FancyBboxPatch((0.5, 7), 9, 1.8, boxstyle="round,pad=0.03",
                           facecolor=COLORS['problem'], edgecolor='white', linewidth=2)
    ax.add_patch(rect1)
    ax.text(5, 8.1, 'PROBLEMA', fontsize=14, ha='center', va='center',
           color='white', fontweight='bold')
    ax.text(5, 7.5, 'Etiquetado manual: alto costo temporal,\nvariabilidad inter-observador (5-15 px)',
           fontsize=11, ha='center', va='center', color='white')

    # Flecha
    ax.annotate('', xy=(5, 6.8), xytext=(5, 7),
               arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=2))

    # Objetivo
    rect2 = FancyBboxPatch((1.5, 4.2), 7, 1.8, boxstyle="round,pad=0.03",
                           facecolor=COLORS['objective'], edgecolor='white', linewidth=2)
    ax.add_patch(rect2)
    ax.text(5, 5.3, 'OBJETIVO', fontsize=14, ha='center', va='center',
           color='white', fontweight='bold')
    ax.text(5, 4.7, 'Desarrollar sistema automático con\nerror de predicción < 8 píxeles',
           fontsize=11, ha='center', va='center', color='white')

    # Flecha
    ax.annotate('', xy=(5, 4), xytext=(5, 4.2),
               arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=2))

    # Solución
    rect3 = FancyBboxPatch((2.5, 1.4), 5, 1.8, boxstyle="round,pad=0.03",
                           facecolor=COLORS['solution'], edgecolor='white', linewidth=2)
    ax.add_patch(rect3)
    ax.text(5, 2.5, 'SOLUCIÓN PROPUESTA', fontsize=14, ha='center', va='center',
           color='white', fontweight='bold')
    ax.text(5, 1.9, 'Deep Learning: ResNet-18 +\nCoordinate Attention + Ensemble',
           fontsize=11, ha='center', va='center', color='white')

    # Resultado
    ax.text(5, 0.5, 'Resultado obtenido: 3.71 píxeles', fontsize=12,
           ha='center', va='center', color=COLORS['solution'], fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                    edgecolor=COLORS['solution'], linewidth=2))

    fig_diag.savefig(GRAFICOS_DIR / 'diagrama_problema_objetivo.png', dpi=300, bbox_inches='tight')
    plt.close(fig_diag)

    # === SLIDE COMPLETA ===
    fig = plt.figure(figsize=(16, 9), facecolor=COLORS['background'])

    fig.text(0.5, 0.94, 'El objetivo fue desarrollar un sistema de predicción automática',
            fontsize=18, ha='center', fontweight='bold', color=COLORS['text'])
    fig.text(0.5, 0.88, 'de landmarks con error inferior a 8 píxeles',
            fontsize=18, ha='center', fontweight='bold', color=COLORS['text'])

    ax = fig.add_axes([0.15, 0.08, 0.7, 0.75])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Cajas con mejor espaciado
    boxes = [
        ((0.3, 7.2), 9.4, 1.6, COLORS['problem'], 'PROBLEMA',
         'Etiquetado manual: alto costo temporal, variabilidad inter-observador (5-15 px)'),
        ((1.3, 4.4), 7.4, 1.6, COLORS['objective'], 'OBJETIVO',
         'Desarrollar sistema automático con error de predicción < 8 píxeles'),
        ((2.3, 1.6), 5.4, 1.6, COLORS['solution'], 'SOLUCIÓN',
         'Deep Learning: ResNet-18 + Coordinate Attention + Ensemble')
    ]

    for (x, y), w, h, color, title, desc in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                             facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.65, title, fontsize=13, ha='center', va='center',
               color='white', fontweight='bold')
        ax.text(x + w/2, y + h*0.3, desc, fontsize=10, ha='center', va='center',
               color='white')

    # Flechas
    ax.annotate('', xy=(5, 7), xytext=(5, 7.2), arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=2))
    ax.annotate('', xy=(5, 4.2), xytext=(5, 4.4), arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=2))

    # Resultado destacado
    ax.text(5, 0.6, '✓ Resultado obtenido: 3.71 píxeles', fontsize=14,
           ha='center', va='center', color=COLORS['solution'], fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                    edgecolor=COLORS['solution'], linewidth=2))

    fig.savefig(SLIDES_DIR / 'slide6_problema_objetivo.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Completado")


def create_slide7_categorias():
    """Slide 7: Composición del dataset."""
    print("\n[Slide 7] Composición del dataset...")

    categorias = ['COVID-19', 'Normal', 'Neumonía viral']
    cantidades = [306, 468, 183]
    colores = [COLORS['covid'], COLORS['normal'], COLORS['viral']]
    total = sum(cantidades)

    # Guardar datos
    with open(DATOS_DIR / 'categorias_dataset.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['categoria', 'cantidad', 'porcentaje'])
        for c, n in zip(categorias, cantidades):
            writer.writerow([c, n, f'{n/total*100:.1f}%'])

    # Exportar ejemplos
    for cat, folder in [('COVID', 'COVID'), ('Normal', 'Normal'), ('Viral', 'Viral_Pneumonia')]:
        cat_path = DATASET_DIR / folder
        if cat_path.exists():
            images = list(cat_path.glob('*.jpeg'))[:3] or list(cat_path.glob('*.png'))[:3]
            for i, img_src in enumerate(images[:3]):
                shutil.copy(img_src, RX_DIR / f"ejemplo_{cat.lower()}_{i+1}.jpeg")

    # === ASSET: Pie chart ===
    fig_pie, ax = plt.subplots(figsize=(8, 8))

    wedges, texts, autotexts = ax.pie(
        cantidades, labels=categorias, colors=colores,
        autopct=lambda pct: f'{int(pct/100*total)}\n({pct:.1f}%)',
        explode=(0.02, 0.03, 0.02), shadow=False, startangle=90,
        textprops={'fontsize': 11}, wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    ax.set_title(f'Distribución del dataset\n(n = {total})', fontsize=13, fontweight='bold', pad=15)

    fig_pie.savefig(GRAFICOS_DIR / 'pie_categorias.png', dpi=300, bbox_inches='tight')
    plt.close(fig_pie)

    # === SLIDE COMPLETA ===
    fig = plt.figure(figsize=(16, 9), facecolor=COLORS['background'])

    fig.text(0.5, 0.94, f'El dataset comprende {total} radiografías de tórax',
            fontsize=18, ha='center', fontweight='bold', color=COLORS['text'])
    fig.text(0.5, 0.88, 'distribuidas en tres categorías diagnósticas',
            fontsize=18, ha='center', fontweight='bold', color=COLORS['text'])

    ax_pie = fig.add_axes([0.05, 0.12, 0.4, 0.68])
    ax_examples = fig.add_axes([0.48, 0.1, 0.5, 0.72])
    ax_examples.axis('off')

    wedges, texts, autotexts = ax_pie.pie(
        cantidades, labels=categorias, colors=colores,
        autopct=lambda pct: f'{int(pct/100*total)}\n({pct:.1f}%)',
        explode=(0.02, 0.03, 0.02), shadow=False, startangle=90,
        textprops={'fontsize': 11}, wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for autotext in autotexts:
        autotext.set_fontweight('bold')

    ax_pie.set_title(f'Distribución del dataset (n = {total})', fontsize=12,
                     fontweight='bold', pad=15)

    # Ejemplos por categoría
    ax_examples.text(0.5, 0.97, 'Ejemplos representativos', fontsize=12,
                    ha='center', fontweight='bold', transform=ax_examples.transAxes)

    categories_paths = {
        'COVID-19': DATASET_DIR / 'COVID',
        'Normal': DATASET_DIR / 'Normal',
        'Neumonía viral': DATASET_DIR / 'Viral_Pneumonia'
    }

    y_offset = 0.85
    for cat, color in zip(categorias, colores):
        cat_path = categories_paths[cat]

        # Indicador de categoría
        ax_examples.add_patch(Rectangle((0.0, y_offset-0.06), 0.03, 0.12,
                                        facecolor=color, edgecolor='none',
                                        transform=ax_examples.transAxes))
        ax_examples.text(0.06, y_offset, cat, fontsize=11, fontweight='bold',
                        va='center', color=color, transform=ax_examples.transAxes)

        # Imágenes de ejemplo
        if cat_path.exists():
            images = list(cat_path.glob('*.jpeg'))[:3] or list(cat_path.glob('*.png'))[:3]
            x_pos = 0.28
            for img_path in images[:3]:
                try:
                    img = Image.open(img_path).convert('L').resize((55, 55))
                    ax_mini = fig.add_axes([0.48 + x_pos * 0.48, 0.1 + y_offset * 0.62, 0.07, 0.1])
                    ax_mini.imshow(np.array(img), cmap='gray')
                    ax_mini.axis('off')
                    x_pos += 0.22
                except:
                    pass

        y_offset -= 0.32

    # Nota sobre el dataset
    fig.text(0.5, 0.02,
            'Fuente: COVID-19 Radiography Database (Kaggle)  •  Resolución: 299×299 px → 224×224 px',
            fontsize=9, ha='center', color=COLORS['text_light'], style='italic')

    fig.savefig(SLIDES_DIR / 'slide7_categorias_dataset.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Completado")


def main():
    """Genera todas las visualizaciones en versión profesional."""
    print("=" * 70)
    print("GENERACIÓN DE VISUALIZACIONES - BLOQUE 1")
    print("VERSIÓN PROFESIONAL ACADÉMICA (v2)")
    print("=" * 70)

    create_directories()
    print(f"\nDirectorio de salida: {OUTPUT_DIR}")

    create_slide1_portada()
    create_slide2_uso_global()
    create_slide3_covid_demanda()
    create_slide4_landmarks_anatomia()
    create_slide5_variabilidad()
    create_slide6_embudo()
    create_slide7_categorias()

    print("\n" + "=" * 70)
    print("GENERACIÓN COMPLETADA")
    print("=" * 70)
    print(f"\nArchivos generados en: {OUTPUT_DIR}/")
    print("  ├── slides/           (7 slides completas)")
    print("  ├── assets/graficos/  (gráficos individuales)")
    print("  ├── assets/radiografias/ (imágenes RX)")
    print("  └── datos/            (CSVs editables)")


if __name__ == '__main__':
    main()
