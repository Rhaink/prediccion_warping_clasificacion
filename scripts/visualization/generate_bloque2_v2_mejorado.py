#!/usr/bin/env python3
"""
Script: Visualizaciones Bloque 2 - VERSION 2 MEJORADA
Correcciones de errores y mejoras visuales profesionales

CORRECCIONES REALIZADAS:
- Slide 8: Símbolo elipsis, layout, landmarks en imagen
- Slide 9: Mejoras menores de visualización
- Slide 10: CORREGIDO ratio incorrecto (era 3×, real es ~1.8×)
- Slide 11: Mejoras en líneas de referencia
- Slide 12: CORREGIDO promedio incorrecto (era 6.3, real es ~8 px), labels cortados
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

# ============================================================================
# CONFIGURACIÓN ESTILO v2_PROFESIONAL MEJORADO
# ============================================================================

COLORS_PRO = {
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
    'data_5': '#666666',
    'covid': '#c44536',
    'normal': '#2d6a4f',
    'viral': '#b07d2b',
    'lm_axis': '#0077b6',
    'lm_central': '#2d6a4f',
    'lm_symmetric': '#7b2cbf',
    'lm_corner': '#c44536',
    'success': '#2d6a4f',
    'warning': '#b07d2b',
    'danger': '#c44536',
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'serif'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'axes.linewidth': 0.8,
    'axes.edgecolor': COLORS_PRO['text_secondary'],
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'text.color': COLORS_PRO['text_primary'],
    'axes.labelcolor': COLORS_PRO['text_primary'],
    'xtick.color': COLORS_PRO['text_secondary'],
    'ytick.color': COLORS_PRO['text_secondary'],
    'figure.facecolor': COLORS_PRO['background'],
    'axes.facecolor': COLORS_PRO['background'],
    'savefig.facecolor': COLORS_PRO['background'],
})

# Rutas
BASE_DIR = Path('/home/donrobot/Projects/prediccion_coordenadas')
OUTPUT_DIR = BASE_DIR / 'presentacion' / '02_metodologia_datos' / 'v2_mejorado'
DATA_DIR = BASE_DIR / 'data'
DATASET_DIR = DATA_DIR / 'dataset'

# Resolución controlada
FIG_WIDTH = 14
FIG_HEIGHT = 7.5
DPI = 100

# Datos del dataset
DATASET_STATS = {
    'COVID': 306,
    'Normal': 468,
    'Viral_Pneumonia': 183,
    'Total': 957
}

SYMMETRIC_PAIRS = [(2, 3), (4, 5), (6, 7), (11, 12), (13, 14)]
CENTRAL_LANDMARKS = [8, 9, 10]

LANDMARK_NAMES_SHORT = {
    1: 'L1', 2: 'L2', 3: 'L3', 4: 'L4', 5: 'L5',
    6: 'L6', 7: 'L7', 8: 'L8', 9: 'L9', 10: 'L10',
    11: 'L11', 12: 'L12', 13: 'L13', 14: 'L14', 15: 'L15'
}


def create_directories():
    """Crea directorios de salida."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_coordinates():
    """Carga coordenadas del CSV maestro."""
    csv_path = DATA_DIR / 'coordenadas' / 'coordenadas_maestro.csv'
    coord_cols = []
    for i in range(1, 16):
        coord_cols.extend([f'L{i}_x', f'L{i}_y'])
    columns = ['idx'] + coord_cols + ['image_name']
    df = pd.read_csv(csv_path, header=None, names=columns)
    return df


def load_sample_image_with_coords(category='Normal', index=50):
    """Carga imagen y coordenadas."""
    df = load_coordinates()

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


def draw_arrow_between_boxes(ax, start, end, color, lw=2):
    """Dibuja flecha profesional entre dos puntos."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                               mutation_scale=15))


def create_slide8_regresion_v2():
    """Slide 8: Regresión - VERSIÓN MEJORADA."""
    print("Generando Slide 8 v2: Formulación como regresión...")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=COLORS_PRO['background'])

    # Título principal
    fig.suptitle('El problema se formula como regresión directa:\nuna imagen produce 30 coordenadas normalizadas [0,1]',
                fontsize=14, fontweight='bold', y=0.96, color=COLORS_PRO['text_primary'])

    # Layout con tres zonas
    ax_input = fig.add_axes([0.02, 0.12, 0.28, 0.72])
    ax_model = fig.add_axes([0.35, 0.12, 0.30, 0.72])
    ax_output = fig.add_axes([0.70, 0.12, 0.28, 0.72])

    # === ENTRADA ===
    ax_input.set_xlim(0, 10)
    ax_input.set_ylim(0, 10)
    ax_input.axis('off')

    # Caja entrada
    input_box = FancyBboxPatch((0.5, 0.5), 9, 9,
                                boxstyle="round,pad=0.02",
                                facecolor=COLORS_PRO['accent_light'],
                                edgecolor=COLORS_PRO['accent_primary'],
                                linewidth=2)
    ax_input.add_patch(input_box)
    ax_input.text(5, 9.2, 'ENTRADA', fontsize=12, ha='center', fontweight='bold',
                 color=COLORS_PRO['accent_primary'])

    # Cargar imagen con landmarks
    img_path, coords, _ = load_sample_image_with_coords('Normal', 50)
    if img_path and img_path.exists():
        ax_img = fig.add_axes([0.05, 0.22, 0.20, 0.50])
        img = Image.open(img_path).convert('L')
        ax_img.imshow(np.array(img), cmap='gray')

        # Dibujar algunos landmarks representativos
        key_landmarks = [0, 1, 8, 9, 10, 13, 14]  # L1, L2, L9, L10, L11, L14, L15
        for i in key_landmarks:
            x, y = coords[i]
            color = COLORS_PRO['lm_central'] if i in [8, 9, 10] else COLORS_PRO['accent_primary']
            ax_img.plot(x, y, 'o', color=color, markersize=5,
                       markeredgecolor='white', markeredgewidth=0.8)

        ax_img.axis('off')

    ax_input.text(5, 1.2, 'Radiografia 224x224 px\n(tensor 1x3x224x224)',
                 fontsize=10, ha='center', color=COLORS_PRO['text_secondary'])

    # === MODELO ===
    ax_model.set_xlim(0, 10)
    ax_model.set_ylim(0, 10)
    ax_model.axis('off')

    # Caja modelo
    model_box = FancyBboxPatch((0.5, 0.5), 9, 9,
                                boxstyle="round,pad=0.02",
                                facecolor=COLORS_PRO['background_alt'],
                                edgecolor=COLORS_PRO['data_2'],
                                linewidth=2)
    ax_model.add_patch(model_box)
    ax_model.text(5, 9.2, 'MODELO', fontsize=12, ha='center', fontweight='bold',
                 color=COLORS_PRO['data_2'])

    # Componentes del modelo con flechas
    components = [
        ('ResNet-18\n(backbone pre-entrenado)', 7.5),
        ('Coordinate Attention\n(captura dependencias espaciales)', 5.0),
        ('Cabeza de Regresion\n(FC + GroupNorm + Dropout)', 2.5)
    ]

    for i, (label, y) in enumerate(components):
        comp_box = FancyBboxPatch((1, y - 0.9), 8, 1.8,
                                   boxstyle="round,pad=0.02",
                                   facecolor='white',
                                   edgecolor=COLORS_PRO['data_2'],
                                   linewidth=1)
        ax_model.add_patch(comp_box)
        ax_model.text(5, y, label, fontsize=9, ha='center', va='center',
                     color=COLORS_PRO['text_primary'])

        # Flechas entre componentes
        if i < len(components) - 1:
            ax_model.annotate('', xy=(5, y - 1.2), xytext=(5, y - 0.9),
                            arrowprops=dict(arrowstyle='-|>', color=COLORS_PRO['data_2'],
                                          lw=1.5, mutation_scale=12))

    # === SALIDA ===
    ax_output.set_xlim(0, 10)
    ax_output.set_ylim(0, 10)
    ax_output.axis('off')

    # Caja salida
    output_box = FancyBboxPatch((0.5, 0.5), 9, 9,
                                 boxstyle="round,pad=0.02",
                                 facecolor=COLORS_PRO['accent_light'],
                                 edgecolor=COLORS_PRO['accent_primary'],
                                 linewidth=2)
    ax_output.add_patch(output_box)
    ax_output.text(5, 9.2, 'SALIDA', fontsize=12, ha='center', fontweight='bold',
                  color=COLORS_PRO['accent_primary'])

    ax_output.text(5, 7.8, '30 coordenadas\nnormalizadas [0,1]', fontsize=11,
                  ha='center', fontweight='bold', color=COLORS_PRO['text_primary'])

    # Vector de coordenadas (sin caracteres problemáticos)
    coord_lines = [
        'L1:  (x₁,  y₁)',
        'L2:  (x₂,  y₂)',
        'L3:  (x₃,  y₃)',
        '      ...',
        'L15: (x₁₅, y₁₅)'
    ]
    y_pos = 6.0
    for line in coord_lines:
        ax_output.text(5, y_pos, line, fontsize=9, ha='center',
                      family='monospace', color=COLORS_PRO['text_secondary'])
        y_pos -= 0.7

    # Cálculo
    calc_box = FancyBboxPatch((1.5, 1.5), 7, 1.5,
                               boxstyle="round,pad=0.02",
                               facecolor='white',
                               edgecolor=COLORS_PRO['accent_secondary'],
                               linewidth=1)
    ax_output.add_patch(calc_box)
    ax_output.text(5, 2.25, '15 landmarks x 2 = 30', fontsize=10, ha='center',
                  color=COLORS_PRO['accent_secondary'], fontweight='bold')

    # Flechas principales entre secciones (en figure coordinates)
    # Flecha ENTRADA -> MODELO
    fig.patches.extend([
        FancyArrowPatch((0.305, 0.48), (0.345, 0.48),
                       arrowstyle='-|>', mutation_scale=20,
                       color=COLORS_PRO['accent_primary'], lw=2,
                       transform=fig.transFigure, figure=fig)
    ])
    # Flecha MODELO -> SALIDA
    fig.patches.extend([
        FancyArrowPatch((0.655, 0.48), (0.695, 0.48),
                       arrowstyle='-|>', mutation_scale=20,
                       color=COLORS_PRO['accent_primary'], lw=2,
                       transform=fig.transFigure, figure=fig)
    ])

    # Nota inferior
    fig.text(0.5, 0.03,
            'Funcion de perdida: Wing Loss | Desnormalizacion: (x_px, y_px) = (x_norm x 224, y_norm x 224)',
            ha='center', fontsize=9, color=COLORS_PRO['text_secondary'],
            style='italic')

    plt.savefig(OUTPUT_DIR / 'slide8_regresion_v2.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor(), pad_inches=0.1)
    plt.close()
    print(f"  Guardado: {OUTPUT_DIR / 'slide8_regresion_v2.png'}")


def create_slide9_split_v2():
    """Slide 9: Split estratificado - VERSIÓN MEJORADA."""
    print("Generando Slide 9 v2: Split estratificado...")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=COLORS_PRO['background'])

    fig.suptitle('El dataset se dividio estratificadamente preservando\nla proporcion de categorias en cada conjunto',
                fontsize=14, fontweight='bold', y=0.96, color=COLORS_PRO['text_primary'])

    # Datos del split 70/15/15
    total = 957
    covid, normal, viral = 306, 468, 183

    splits = {
        'Entrenamiento': {'pct': 70, 'COVID': 214, 'Normal': 328, 'Viral': 128, 'total': 670},
        'Validacion': {'pct': 15, 'COVID': 46, 'Normal': 70, 'Viral': 27, 'total': 143},
        'Prueba': {'pct': 15, 'COVID': 46, 'Normal': 70, 'Viral': 28, 'total': 144}
    }

    # Panel izquierdo: Diagrama de flujo
    ax_flow = fig.add_axes([0.02, 0.10, 0.50, 0.78])
    ax_flow.set_xlim(0, 10)
    ax_flow.set_ylim(0, 10)
    ax_flow.axis('off')

    # Dataset completo
    total_box = FancyBboxPatch((2.5, 8.2), 5, 1.2,
                                boxstyle="round,pad=0.03",
                                facecolor=COLORS_PRO['accent_light'],
                                edgecolor=COLORS_PRO['accent_primary'],
                                linewidth=2)
    ax_flow.add_patch(total_box)
    ax_flow.text(5, 8.8, f'Dataset Completo: n = {total}', fontsize=12,
                ha='center', va='center', fontweight='bold',
                color=COLORS_PRO['accent_primary'])

    # Flechas hacia abajo
    for x in [1.7, 5, 8.3]:
        ax_flow.annotate('', xy=(x, 6.8), xytext=(5, 8.2),
                        arrowprops=dict(arrowstyle='-|>', color=COLORS_PRO['text_secondary'],
                                       lw=1.5, mutation_scale=12))

    # Cajas de splits
    split_info = [
        ('Entrenamiento', 0.2, COLORS_PRO['success'], '70%'),
        ('Validacion', 3.5, COLORS_PRO['warning'], '15%'),
        ('Prueba', 6.8, COLORS_PRO['danger'], '15%')
    ]

    for name, x, color, pct in split_info:
        data = splits[name]

        # Caja principal
        box = FancyBboxPatch((x, 2.8), 3, 4,
                              boxstyle="round,pad=0.02",
                              facecolor='white',
                              edgecolor=color,
                              linewidth=2)
        ax_flow.add_patch(box)

        # Título
        ax_flow.text(x + 1.5, 6.5, f'{name}\n({pct})', fontsize=10, ha='center',
                    fontweight='bold', color=color)

        # Total
        ax_flow.text(x + 1.5, 5.5, f'n = {data["total"]}', fontsize=11, ha='center',
                    fontweight='bold', color=COLORS_PRO['text_primary'])

        # Desglose
        ax_flow.text(x + 1.5, 4.6, f'COVID: {data["COVID"]}', fontsize=9,
                    ha='center', color=COLORS_PRO['covid'])
        ax_flow.text(x + 1.5, 4.0, f'Normal: {data["Normal"]}', fontsize=9,
                    ha='center', color=COLORS_PRO['normal'])
        ax_flow.text(x + 1.5, 3.4, f'Viral: {data["Viral"]}', fontsize=9,
                    ha='center', color=COLORS_PRO['viral'])

    # Nota
    ax_flow.text(5, 1.8, 'Division estratificada: cada conjunto\nmantiene ~32% COVID, 49% Normal, 19% Viral',
                fontsize=9, ha='center', color=COLORS_PRO['text_secondary'], style='italic')

    # Panel derecho: Gráfico de barras apiladas
    ax_bars = fig.add_axes([0.58, 0.15, 0.38, 0.70])

    categories = ['COVID', 'Normal', 'Viral']
    colors_cat = [COLORS_PRO['covid'], COLORS_PRO['normal'], COLORS_PRO['viral']]

    x_pos = np.arange(3)
    width = 0.5

    bottom = np.zeros(3)
    for cat, color in zip(categories, colors_cat):
        values = [splits['Entrenamiento'][cat], splits['Validacion'][cat], splits['Prueba'][cat]]
        ax_bars.bar(x_pos, values, bottom=bottom, color=color, edgecolor='white',
                   linewidth=1.5, label=cat, width=width)
        bottom += values

    ax_bars.set_xticks(x_pos)
    ax_bars.set_xticklabels(['Entrenamiento\n(70%)', 'Validacion\n(15%)', 'Prueba\n(15%)'],
                           fontsize=9)
    ax_bars.set_ylabel('Numero de imagenes', fontsize=10)
    ax_bars.legend(loc='upper right', framealpha=0.95, fontsize=9)
    ax_bars.set_title('Distribucion por conjunto', fontsize=11, fontweight='bold', pad=10)

    ax_bars.spines['top'].set_visible(False)
    ax_bars.spines['right'].set_visible(False)
    ax_bars.set_ylim(0, 750)

    # Totales encima de barras
    totals = [670, 143, 144]
    for i, t in enumerate(totals):
        ax_bars.text(i, t + 15, str(t), ha='center', fontsize=10, fontweight='bold',
                    color=COLORS_PRO['text_primary'])

    plt.savefig(OUTPUT_DIR / 'slide9_split_v2.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor(), pad_inches=0.1)
    plt.close()
    print(f"  Guardado: {OUTPUT_DIR / 'slide9_split_v2.png'}")


def create_slide10_variabilidad_v2():
    """Slide 10: Variabilidad - VERSIÓN CORREGIDA (ratio real ~1.8x, no 3x)."""
    print("Generando Slide 10 v2: Variabilidad por landmark (CORREGIDO)...")

    df = load_coordinates()

    # Calcular desviación estándar por landmark
    std_per_landmark = []
    for i in range(1, 16):
        std_x = df[f'L{i}_x'].std()
        std_y = df[f'L{i}_y'].std()
        std_combined = np.sqrt(std_x**2 + std_y**2)
        std_per_landmark.append(std_combined)

    max_std = max(std_per_landmark)
    min_std = min(std_per_landmark)
    ratio = max_std / min_std
    max_idx = np.argmax(std_per_landmark)
    min_idx = np.argmin(std_per_landmark)

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=COLORS_PRO['background'])

    # TÍTULO CORREGIDO con ratio real
    fig.suptitle(f'La variabilidad por landmark guio la seleccion de la funcion de perdida:\n'
                f'los angulos (L14: {max_std:.1f}px) tienen {ratio:.1f}x mas variabilidad que los centrales (L9: {min_std:.1f}px)',
                fontsize=13, fontweight='bold', y=0.96, color=COLORS_PRO['text_primary'])

    ax = fig.add_axes([0.08, 0.12, 0.88, 0.75])

    # Colores por tipo
    colors = []
    for i in range(15):
        if i in [0, 1]:
            colors.append(COLORS_PRO['lm_axis'])
        elif i in [8, 9, 10]:
            colors.append(COLORS_PRO['lm_central'])
        elif i in [2, 3, 4, 5, 6, 7]:
            colors.append(COLORS_PRO['lm_symmetric'])
        else:
            colors.append(COLORS_PRO['lm_corner'])

    x_pos = np.arange(15)
    bars = ax.bar(x_pos, std_per_landmark, color=colors, edgecolor='white', linewidth=1.5, width=0.7)

    # Destacar máximo y mínimo
    bars[max_idx].set_edgecolor(COLORS_PRO['danger'])
    bars[max_idx].set_linewidth(3)
    bars[min_idx].set_edgecolor(COLORS_PRO['success'])
    bars[min_idx].set_linewidth(3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'L{i+1}' for i in range(15)], fontsize=10)
    ax.set_xlabel('Landmark', fontsize=11)
    ax.set_ylabel('Desviacion estandar combinada (pixeles)', fontsize=11)

    # Línea de media
    mean_std = np.mean(std_per_landmark)
    ax.axhline(y=mean_std, color=COLORS_PRO['text_secondary'], linestyle='--',
               linewidth=1.5, alpha=0.7)
    ax.text(14.5, mean_std + 0.5, f'Media: {mean_std:.1f}px', fontsize=9,
           ha='right', color=COLORS_PRO['text_secondary'])

    # Anotaciones mejoradas (sin flechas que crucen)
    ax.annotate(f'MAX\n{max_std:.1f}px',
               xy=(max_idx, max_std), xytext=(max_idx, max_std + 4),
               fontsize=9, ha='center', fontweight='bold', color=COLORS_PRO['danger'],
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=COLORS_PRO['danger']))

    ax.annotate(f'MIN\n{min_std:.1f}px',
               xy=(min_idx, min_std), xytext=(min_idx, min_std + 6),
               fontsize=9, ha='center', fontweight='bold', color=COLORS_PRO['success'],
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=COLORS_PRO['success']))

    # Leyenda
    legend_elements = [
        mpatches.Patch(facecolor=COLORS_PRO['lm_axis'], edgecolor='white', label='Eje traqueal (L1-L2)'),
        mpatches.Patch(facecolor=COLORS_PRO['lm_symmetric'], edgecolor='white', label='Pares bilaterales (L3-L8)'),
        mpatches.Patch(facecolor=COLORS_PRO['lm_central'], edgecolor='white', label='Centrales (L9-L11)'),
        mpatches.Patch(facecolor=COLORS_PRO['lm_corner'], edgecolor='white', label='Angulos (L12-L15)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.95, fontsize=9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max_std * 1.35)

    # Nota inferior CORREGIDA
    fig.text(0.5, 0.02,
            f'Ratio max/min: {ratio:.2f}x | Wing Loss: sensible a errores pequenos, robusta ante outliers',
            ha='center', fontsize=9, color=COLORS_PRO['text_secondary'], style='italic')

    plt.savefig(OUTPUT_DIR / 'slide10_variabilidad_v2.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor(), pad_inches=0.1)
    plt.close()
    print(f"  Guardado: {OUTPUT_DIR / 'slide10_variabilidad_v2.png'}")


def create_slide11_geometria_v2():
    """Slide 11: Geometría eje central - VERSIÓN MEJORADA."""
    print("Generando Slide 11 v2: Geometria del eje central...")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=COLORS_PRO['background'])

    fig.suptitle('Descubrimiento geometrico: L9, L10 y L11 dividen el segmento L1-L2\n'
                'en cuatro partes aproximadamente iguales (t = 0.25, 0.50, 0.75)',
                fontsize=13, fontweight='bold', y=0.96, color=COLORS_PRO['text_primary'])

    # Dos paneles
    ax_img = fig.add_axes([0.02, 0.08, 0.42, 0.80])
    ax_diagram = fig.add_axes([0.48, 0.08, 0.50, 0.80])

    # === Panel izquierdo: Radiografía ===
    img_path, coords, _ = load_sample_image_with_coords('Normal', 50)

    if img_path and img_path.exists():
        img = Image.open(img_path).convert('L')
        ax_img.imshow(np.array(img), cmap='gray')

        L1, L2 = coords[0], coords[1]

        # Eje principal L1-L2
        ax_img.plot([L1[0], L2[0]], [L1[1], L2[1]], '-',
                   color=COLORS_PRO['lm_axis'], linewidth=3, alpha=0.9)

        # L1 y L2
        for lm, label in [(L1, 'L1'), (L2, 'L2')]:
            ax_img.plot(lm[0], lm[1], 'o', color=COLORS_PRO['lm_axis'], markersize=14,
                       markeredgecolor='white', markeredgewidth=2, zorder=10)
            ax_img.annotate(label, (lm[0] + 12, lm[1]), color='white', fontsize=11,
                           fontweight='bold', zorder=11,
                           path_effects=[pe.withStroke(linewidth=3, foreground=COLORS_PRO['lm_axis'])])

        # L9, L10, L11 con líneas horizontales sutiles
        central_labels = ['L9', 'L10', 'L11']
        for i, idx in enumerate([8, 9, 10]):
            lm = coords[idx]
            ax_img.plot(lm[0], lm[1], 's', color=COLORS_PRO['lm_central'], markersize=12,
                       markeredgecolor='white', markeredgewidth=2, zorder=10)
            ax_img.annotate(central_labels[i], (lm[0] + 12, lm[1]), color='white', fontsize=10,
                           fontweight='bold', zorder=11,
                           path_effects=[pe.withStroke(linewidth=3, foreground=COLORS_PRO['lm_central'])])
            # Línea horizontal sutil
            ax_img.axhline(y=lm[1], color=COLORS_PRO['lm_central'], linestyle=':',
                          alpha=0.4, linewidth=1, zorder=1)

    ax_img.axis('off')
    ax_img.set_title('Eje traqueal y landmarks centrales', fontsize=11, pad=8,
                    color=COLORS_PRO['text_primary'])

    # === Panel derecho: Diagrama esquemático ===
    ax_diagram.set_xlim(0, 10)
    ax_diagram.set_ylim(0, 10)
    ax_diagram.axis('off')

    # Eje vertical
    x_axis = 2.5
    y_top, y_bottom = 9, 1

    ax_diagram.plot([x_axis, x_axis], [y_top, y_bottom], '-',
                   color=COLORS_PRO['lm_axis'], linewidth=4, solid_capstyle='round')

    # Puntos en el eje
    points = [
        ('L1', y_top, COLORS_PRO['lm_axis'], 't = 0', 'o'),
        ('L9', y_top - 2, COLORS_PRO['lm_central'], 't = 0.25', 's'),
        ('L10', (y_top + y_bottom) / 2, COLORS_PRO['lm_central'], 't = 0.50', 's'),
        ('L11', y_bottom + 2, COLORS_PRO['lm_central'], 't = 0.75', 's'),
        ('L2', y_bottom, COLORS_PRO['lm_axis'], 't = 1', 'o'),
    ]

    for label, y, color, t_val, marker in points:
        ax_diagram.plot(x_axis, y, marker, color=color, markersize=16,
                       markeredgecolor='white', markeredgewidth=2, zorder=10)
        ax_diagram.text(x_axis - 0.8, y, label, fontsize=12, ha='right', va='center',
                       fontweight='bold', color=color)
        ax_diagram.text(x_axis + 0.7, y, t_val, fontsize=10, ha='left', va='center',
                       color=COLORS_PRO['text_secondary'])

    # Fórmula paramétrica
    formula_box = FancyBboxPatch((4.5, 6.5), 5, 2.5,
                                  boxstyle="round,pad=0.03",
                                  facecolor=COLORS_PRO['accent_light'],
                                  edgecolor=COLORS_PRO['accent_primary'],
                                  linewidth=1.5)
    ax_diagram.add_patch(formula_box)

    ax_diagram.text(7, 8.5, 'Formula parametrica:', fontsize=10, ha='center',
                   fontweight='bold', color=COLORS_PRO['text_primary'])
    ax_diagram.text(7, 7.5, 'P(t) = L1 + t(L2 - L1)', fontsize=12, ha='center',
                   family='monospace', color=COLORS_PRO['accent_primary'], fontweight='bold')
    ax_diagram.text(7, 6.8, 'donde t e [0, 1]', fontsize=9, ha='center',
                   color=COLORS_PRO['text_secondary'])

    # Tabla de valores observados
    ax_diagram.text(7, 5.5, 'Valores observados:', fontsize=10, ha='center',
                   fontweight='bold', color=COLORS_PRO['text_primary'])

    table_box = FancyBboxPatch((4.8, 3.2), 4.4, 2,
                                boxstyle="round,pad=0.02",
                                facecolor='white',
                                edgecolor=COLORS_PRO['text_secondary'],
                                linewidth=0.5)
    ax_diagram.add_patch(table_box)

    table_data = [('L9', '0.25 +/- 0.05'), ('L10', '0.50 +/- 0.03'), ('L11', '0.75 +/- 0.05')]
    y_table = 4.8
    for lm, t_val in table_data:
        ax_diagram.text(5.5, y_table, lm, fontsize=10, ha='center',
                       color=COLORS_PRO['lm_central'], fontweight='bold')
        ax_diagram.text(8.2, y_table, f't = {t_val}', fontsize=9, ha='center',
                       color=COLORS_PRO['text_secondary'])
        y_table -= 0.5

    # Implicación
    impl_box = FancyBboxPatch((4.5, 1.2), 5, 1.5,
                               boxstyle="round,pad=0.02",
                               facecolor=COLORS_PRO['background_alt'],
                               edgecolor=COLORS_PRO['success'],
                               linewidth=1)
    ax_diagram.add_patch(impl_box)
    ax_diagram.text(7, 1.95, 'Implicacion: la colinealidad permite', fontsize=9, ha='center',
                   color=COLORS_PRO['text_primary'])
    ax_diagram.text(7, 1.45, 'validar predicciones sin GT adicional', fontsize=9, ha='center',
                   color=COLORS_PRO['success'], fontweight='bold')

    plt.savefig(OUTPUT_DIR / 'slide11_geometria_v2.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor(), pad_inches=0.1)
    plt.close()
    print(f"  Guardado: {OUTPUT_DIR / 'slide11_geometria_v2.png'}")


def create_slide12_asimetria_v2():
    """Slide 12: Asimetría - VERSIÓN CORREGIDA (promedio real ~8px, no 6.3px)."""
    print("Generando Slide 12 v2: Asimetria natural (CORREGIDO)...")

    df = load_coordinates()

    # Calcular asimetrías reales
    pair_names = {
        (2, 3): 'Apices\n(L3-L4)',
        (4, 5): 'Hilios sup.\n(L5-L6)',
        (6, 7): 'Hilios inf.\n(L7-L8)',
        (11, 12): 'Cardiofrenicos\n(L12-L13)',
        (13, 14): 'Costofrenicos\n(L14-L15)'
    }

    asimetrias = {}
    all_asym_values = []

    for (i, j), name in pair_names.items():
        # Diferencia vertical (asimetría en Y)
        dist_y = np.abs(df[f'L{i+1}_y'] - df[f'L{j+1}_y'])
        mean_asym = dist_y.mean()
        std_asym = dist_y.std()
        asimetrias[name] = {'mean': mean_asym, 'std': std_asym}
        all_asym_values.append(mean_asym)

    # Calcular media real
    media_real = np.mean(all_asym_values)

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=COLORS_PRO['background'])

    # TÍTULO CORREGIDO con valor real
    fig.suptitle(f'La asimetria natural del cuerpo humano (~{media_real:.1f} px promedio)\n'
                'implica que no se debe forzar simetria perfecta en la perdida',
                fontsize=13, fontweight='bold', y=0.96, color=COLORS_PRO['text_primary'])

    # Dos paneles
    ax_img = fig.add_axes([0.02, 0.12, 0.38, 0.75])
    ax_bars = fig.add_axes([0.48, 0.15, 0.48, 0.70])

    # === Panel izquierdo: Radiografía con pares ===
    img_path, coords, _ = load_sample_image_with_coords('Normal', 50)

    pair_colors = ['#e63946', '#f4a261', '#2a9d8f', '#457b9d', '#6d597a']

    if img_path and img_path.exists():
        img = Image.open(img_path).convert('L')
        ax_img.imshow(np.array(img), cmap='gray')

        for (i, j), color in zip(SYMMETRIC_PAIRS, pair_colors):
            lm_i, lm_j = coords[i], coords[j]

            # Línea conectando
            ax_img.plot([lm_i[0], lm_j[0]], [lm_i[1], lm_j[1]], '--',
                       color=color, linewidth=2.5, alpha=0.8)

            # Puntos
            ax_img.plot(lm_i[0], lm_i[1], 'o', color=color, markersize=10,
                       markeredgecolor='white', markeredgewidth=1.5, zorder=10)
            ax_img.plot(lm_j[0], lm_j[1], 'o', color=color, markersize=10,
                       markeredgecolor='white', markeredgewidth=1.5, zorder=10)

            # Etiquetas
            ax_img.annotate(f'L{i+1}', (lm_i[0] - 18, lm_i[1]), color='white', fontsize=8,
                           fontweight='bold', ha='right', zorder=11,
                           path_effects=[pe.withStroke(linewidth=2, foreground=color)])
            ax_img.annotate(f'L{j+1}', (lm_j[0] + 8, lm_j[1]), color='white', fontsize=8,
                           fontweight='bold', zorder=11,
                           path_effects=[pe.withStroke(linewidth=2, foreground=color)])

    ax_img.axis('off')
    ax_img.set_title('Pares simetricos bilaterales', fontsize=11, pad=8,
                    color=COLORS_PRO['text_primary'])

    # === Panel derecho: Barras horizontales (sin barras de error grandes) ===
    pair_labels = list(pair_names.values())
    y_values = [asimetrias[name]['mean'] for name in pair_labels]

    y_pos = np.arange(len(pair_labels))
    bars = ax_bars.barh(y_pos, y_values, color=pair_colors,
                        edgecolor='white', linewidth=1.5, height=0.6)

    ax_bars.set_yticks(y_pos)
    ax_bars.set_yticklabels(pair_labels, fontsize=9)
    ax_bars.set_xlabel('Asimetria vertical promedio (pixeles)', fontsize=10)
    ax_bars.invert_yaxis()

    # Línea de media
    ax_bars.axvline(x=media_real, color=COLORS_PRO['text_secondary'], linestyle='--',
                   linewidth=2, alpha=0.8)
    ax_bars.text(media_real + 0.3, 4.7, f'Media: {media_real:.1f} px', fontsize=9,
                color=COLORS_PRO['text_secondary'], fontweight='bold')

    # Valores en barras
    for bar, val in zip(bars, y_values):
        ax_bars.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}', va='center', fontsize=9, fontweight='bold',
                    color=COLORS_PRO['text_primary'])

    ax_bars.spines['top'].set_visible(False)
    ax_bars.spines['right'].set_visible(False)
    ax_bars.set_xlim(0, max(y_values) * 1.25)
    ax_bars.set_title('Diferencia vertical en altura entre pares', fontsize=11,
                     fontweight='bold', pad=10)

    # Nota inferior
    fig.text(0.5, 0.02,
            'Conclusion: usar perdida de simetria suave (peso bajo) | '
            'Flip horizontal requiere intercambiar landmarks de cada par',
            ha='center', fontsize=9, color=COLORS_PRO['text_secondary'], style='italic')

    plt.savefig(OUTPUT_DIR / 'slide12_asimetria_v2.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor(), pad_inches=0.1)
    plt.close()
    print(f"  Guardado: {OUTPUT_DIR / 'slide12_asimetria_v2.png'}")


def main():
    """Genera todas las visualizaciones mejoradas del Bloque 2."""
    print("=" * 70)
    print("BLOQUE 2 - VERSION 2 MEJORADA")
    print("Correcciones: ratios, promedios, símbolos, layout")
    print("=" * 70)
    print(f"Resolucion: {int(FIG_WIDTH*DPI)}x{int(FIG_HEIGHT*DPI)} px (max)")
    print(f"Salida: {OUTPUT_DIR}")
    print()

    create_directories()

    create_slide8_regresion_v2()
    create_slide9_split_v2()
    create_slide10_variabilidad_v2()
    create_slide11_geometria_v2()
    create_slide12_asimetria_v2()

    print()
    print("=" * 70)
    print("COMPLETADO - 5 slides mejoradas generadas")
    print("=" * 70)
    print()
    print("CORRECCIONES REALIZADAS:")
    print("- Slide 8: Simbolo elipsis corregido, landmarks en imagen")
    print("- Slide 10: Ratio corregido (era 3x, ahora muestra valor real ~1.8x)")
    print("- Slide 12: Promedio corregido (era 6.3px, ahora muestra valor real)")
    print("- Todas: Mejoras de layout y legibilidad")


if __name__ == '__main__':
    main()
