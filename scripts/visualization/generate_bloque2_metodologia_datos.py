#!/usr/bin/env python3
"""
Script: Visualizaciones Bloque 2 - METODOLOGÍA DE DATOS (Slides 8-12)
Estilo: v2_profesional (resolución max 1400x800, paleta académica)

Slides:
- 8: El problema se formula como regresión: imagen → 30 coordenadas
- 9: Dataset dividido estratificadamente: 70/15/15
- 10: Variabilidad por landmark guió selección de loss
- 11: Geometría L9-L10-L11 define eje central
- 12: Asimetría natural requiere tratamiento de pares
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Arrow
from matplotlib.collections import PatchCollection
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

# ============================================================================
# CONFIGURACIÓN ESTILO v2_PROFESIONAL (idéntico al Bloque 1)
# ============================================================================

COLORS_PRO = {
    'text_primary': '#1a1a2e',
    'text_secondary': '#4a4a4a',
    'background': '#ffffff',
    'background_alt': '#f5f5f5',
    'accent_primary': '#003366',
    'accent_secondary': '#0066cc',
    'accent_light': '#e6f0ff',
    'data_1': '#003366',
    'data_2': '#006633',
    'data_3': '#cc6600',
    'data_4': '#660066',
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = PROJECT_ROOT
OUTPUT_DIR = BASE_DIR / 'presentacion' / '02_metodologia_datos'
DATA_DIR = BASE_DIR / 'data'
DATASET_DIR = DATA_DIR / 'dataset'

# Resolución controlada
FIG_WIDTH = 14
FIG_HEIGHT = 7.875
DPI = 100

# Datos del dataset
DATASET_STATS = {
    'COVID': 306,
    'Normal': 468,
    'Viral_Pneumonia': 183,
    'Total': 957
}

# Pares simétricos (0-indexed)
SYMMETRIC_PAIRS = [(2, 3), (4, 5), (6, 7), (11, 12), (13, 14)]
CENTRAL_LANDMARKS = [8, 9, 10]  # L9, L10, L11

# Anatomía de landmarks
LANDMARK_ANATOMY = {
    1: 'Ápice traqueal',
    2: 'Bifurcación traqueal',
    3: 'Ápice pulmonar der.',
    4: 'Ápice pulmonar izq.',
    5: 'Hilio derecho sup.',
    6: 'Hilio izquierdo sup.',
    7: 'Hilio derecho inf.',
    8: 'Hilio izquierdo inf.',
    9: 'Central superior (L9)',
    10: 'Central medio (L10)',
    11: 'Central inferior (L11)',
    12: 'Cardiofrénico der.',
    13: 'Cardiofrénico izq.',
    14: 'Costofrénico der.',
    15: 'Costofrénico izq.'
}


def create_directories():
    """Crea directorios de salida."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'assets').mkdir(exist_ok=True)


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


def create_slide8_regresion():
    """Slide 8: El problema se formula como regresión: imagen → 30 coordenadas."""
    print("Generando Slide 8: Formulación como regresión...")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=COLORS_PRO['background'])
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Título (afirmación)
    ax.text(7, 7.6, 'El problema se formula como regresión directa:\nuna imagen produce 30 coordenadas normalizadas',
           fontsize=14, ha='center', va='top', fontweight='bold',
           color=COLORS_PRO['text_primary'])

    # === INPUT: Radiografía ===
    # Caja de entrada
    input_rect = FancyBboxPatch((0.5, 2.5), 3.5, 4,
                                 boxstyle="round,pad=0.03",
                                 facecolor=COLORS_PRO['accent_light'],
                                 edgecolor=COLORS_PRO['accent_primary'],
                                 linewidth=2)
    ax.add_patch(input_rect)
    ax.text(2.25, 6.3, 'ENTRADA', fontsize=11, ha='center', fontweight='bold',
           color=COLORS_PRO['accent_primary'])

    # Cargar imagen ejemplo
    img_path, coords, _ = load_sample_image_with_coords('Normal', 50)
    if img_path and img_path.exists():
        img = Image.open(img_path).convert('L').resize((80, 80))
        ax_img = fig.add_axes([0.08, 0.38, 0.17, 0.32])
        ax_img.imshow(np.array(img), cmap='gray')
        ax_img.axis('off')
        ax_img.set_title('Radiografía\n224×224 px', fontsize=9, pad=3)

    # Flecha tensor
    ax.annotate('', xy=(4.8, 4.5), xytext=(4.2, 4.5),
               arrowprops=dict(arrowstyle='->', color=COLORS_PRO['accent_primary'], lw=2))
    ax.text(4.5, 4.1, 'Tensor\n[1, 3, 224, 224]', fontsize=8, ha='center',
           color=COLORS_PRO['text_secondary'])

    # === MODELO: ResNet-18 + CA ===
    model_rect = FancyBboxPatch((5, 2.8), 4, 3.5,
                                boxstyle="round,pad=0.03",
                                facecolor=COLORS_PRO['background_alt'],
                                edgecolor=COLORS_PRO['data_2'],
                                linewidth=2)
    ax.add_patch(model_rect)
    ax.text(7, 6.1, 'MODELO', fontsize=11, ha='center', fontweight='bold',
           color=COLORS_PRO['data_2'])

    # Componentes del modelo
    model_parts = [
        ('ResNet-18\n(backbone)', 5.8),
        ('Coordinate\nAttention', 4.8),
        ('Cabeza de\nRegresión', 3.8)
    ]

    for label, y in model_parts:
        rect = FancyBboxPatch((5.3, y - 0.35), 3.4, 0.7,
                              boxstyle="round,pad=0.02",
                              facecolor='white',
                              edgecolor=COLORS_PRO['data_2'],
                              linewidth=1)
        ax.add_patch(rect)
        ax.text(7, y, label, fontsize=9, ha='center', va='center',
               color=COLORS_PRO['text_primary'])

    # Flechas internas
    for y_start, y_end in [(5.45, 5.15), (4.45, 4.15)]:
        ax.annotate('', xy=(7, y_end), xytext=(7, y_start),
                   arrowprops=dict(arrowstyle='->', color=COLORS_PRO['text_secondary'], lw=1))

    # === OUTPUT: 30 coordenadas ===
    ax.annotate('', xy=(10.2, 4.5), xytext=(9.3, 4.5),
               arrowprops=dict(arrowstyle='->', color=COLORS_PRO['accent_primary'], lw=2))

    output_rect = FancyBboxPatch((10.3, 2.5), 3.2, 4,
                                  boxstyle="round,pad=0.03",
                                  facecolor=COLORS_PRO['accent_light'],
                                  edgecolor=COLORS_PRO['accent_primary'],
                                  linewidth=2)
    ax.add_patch(output_rect)
    ax.text(11.9, 6.3, 'SALIDA', fontsize=11, ha='center', fontweight='bold',
           color=COLORS_PRO['accent_primary'])

    # Vector de coordenadas
    ax.text(11.9, 5.8, '30 coordenadas\nnormalizadas [0,1]', fontsize=10, ha='center',
           color=COLORS_PRO['text_primary'], fontweight='bold')

    # Mostrar estructura del vector
    coord_text = """L1: (x₁, y₁)
L2: (x₂, y₂)
   ⋮
L15: (x₁₅, y₁₅)"""
    ax.text(11.9, 4.5, coord_text, fontsize=9, ha='center', va='center',
           family='monospace', color=COLORS_PRO['text_secondary'])

    ax.text(11.9, 3.0, '15 landmarks\n× 2 coords = 30', fontsize=9, ha='center',
           color=COLORS_PRO['accent_secondary'], style='italic')

    # Nota inferior: fórmula de desnormalización
    ax.text(7, 1.6, 'Coordenadas en píxeles: (x_px, y_px) = (x_norm × 224, y_norm × 224)',
           fontsize=10, ha='center', color=COLORS_PRO['text_secondary'],
           bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS_PRO['background_alt'],
                    edgecolor=COLORS_PRO['text_secondary'], linewidth=0.5))

    # Nota de función de pérdida
    ax.text(7, 0.8, 'Función de pérdida: Wing Loss (logarítmica para errores pequeños, lineal para grandes)',
           fontsize=9, ha='center', color=COLORS_PRO['text_secondary'], style='italic')

    plt.savefig(OUTPUT_DIR / 'slide8_regresion.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Guardado: {OUTPUT_DIR / 'slide8_regresion.png'}")


def create_slide9_split():
    """Slide 9: Dataset dividido estratificadamente: 70/15/15."""
    print("Generando Slide 9: Split estratificado...")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=COLORS_PRO['background'])

    fig.suptitle('El dataset se dividió estratificadamente preservando\nla proporción de categorías en cada conjunto',
                fontsize=14, fontweight='bold', y=0.97, color=COLORS_PRO['text_primary'])

    # Split: 70/15/15
    total = 957
    train_pct, val_pct, test_pct = 0.70, 0.15, 0.15

    train_total = int(total * train_pct)  # 669
    val_total = int(total * val_pct)      # 143
    test_total = total - train_total - val_total  # 145

    # Por categoría
    covid, normal, viral = 306, 468, 183

    # Distribución estratificada
    splits = {
        'Entrenamiento\n(70%)': {
            'total': train_total,
            'COVID': int(covid * train_pct),
            'Normal': int(normal * train_pct),
            'Viral': int(viral * train_pct)
        },
        'Validación\n(15%)': {
            'total': val_total,
            'COVID': int(covid * val_pct),
            'Normal': int(normal * val_pct),
            'Viral': int(viral * val_pct)
        },
        'Prueba\n(15%)': {
            'total': test_total,
            'COVID': covid - int(covid * train_pct) - int(covid * val_pct),
            'Normal': normal - int(normal * train_pct) - int(normal * val_pct),
            'Viral': viral - int(viral * train_pct) - int(viral * val_pct)
        }
    }

    # Layout
    ax_main = fig.add_axes([0.05, 0.15, 0.55, 0.72])
    ax_bars = fig.add_axes([0.65, 0.15, 0.32, 0.72])

    # === Panel izquierdo: Diagrama de flujo ===
    ax_main.set_xlim(0, 10)
    ax_main.set_ylim(0, 8)
    ax_main.axis('off')

    # Caja dataset completo
    full_rect = FancyBboxPatch((3, 6.5), 4, 1,
                                boxstyle="round,pad=0.03",
                                facecolor=COLORS_PRO['accent_light'],
                                edgecolor=COLORS_PRO['accent_primary'],
                                linewidth=2)
    ax_main.add_patch(full_rect)
    ax_main.text(5, 7, f'Dataset Completo\nn = {total}', fontsize=12, ha='center', va='center',
                fontweight='bold', color=COLORS_PRO['accent_primary'])

    # Flechas hacia abajo
    split_positions = [(1.5, 'train'), (5, 'val'), (8.5, 'test')]
    colors_split = [COLORS_PRO['success'], COLORS_PRO['warning'], COLORS_PRO['danger']]

    for i, ((x, name), color) in enumerate(zip(split_positions, colors_split)):
        # Flecha
        ax_main.annotate('', xy=(x, 5.3), xytext=(5, 6.5),
                        arrowprops=dict(arrowstyle='->', color=COLORS_PRO['text_secondary'], lw=1.5))

    # Cajas de cada split
    split_boxes = [
        ('Entrenamiento\n(70%)', 0.3, splits['Entrenamiento\n(70%)'], COLORS_PRO['success']),
        ('Validación\n(15%)', 3.8, splits['Validación\n(15%)'], COLORS_PRO['warning']),
        ('Prueba\n(15%)', 7.3, splits['Prueba\n(15%)'], COLORS_PRO['danger'])
    ]

    for label, x, data, color in split_boxes:
        rect = FancyBboxPatch((x, 2.2), 2.4, 3,
                               boxstyle="round,pad=0.03",
                               facecolor='white',
                               edgecolor=color,
                               linewidth=2)
        ax_main.add_patch(rect)

        ax_main.text(x + 1.2, 5.0, label, fontsize=10, ha='center', va='center',
                    fontweight='bold', color=color)
        ax_main.text(x + 1.2, 4.4, f'n = {data["total"]}', fontsize=11, ha='center',
                    fontweight='bold', color=COLORS_PRO['text_primary'])

        # Desglose por categoría
        ax_main.text(x + 1.2, 3.7, f'COVID: {data["COVID"]}', fontsize=9, ha='center',
                    color=COLORS_PRO['covid'])
        ax_main.text(x + 1.2, 3.2, f'Normal: {data["Normal"]}', fontsize=9, ha='center',
                    color=COLORS_PRO['normal'])
        ax_main.text(x + 1.2, 2.7, f'Viral: {data["Viral"]}', fontsize=9, ha='center',
                    color=COLORS_PRO['viral'])

    # Nota sobre estratificación
    ax_main.text(5, 1.4, 'División estratificada: cada conjunto mantiene\nla proporción original de categorías (~32% COVID, 49% Normal, 19% Viral)',
                fontsize=9, ha='center', color=COLORS_PRO['text_secondary'], style='italic')

    # === Panel derecho: Barras apiladas ===
    categories = ['COVID', 'Normal', 'Viral']
    colors_cat = [COLORS_PRO['covid'], COLORS_PRO['normal'], COLORS_PRO['viral']]

    splits_names = ['Train', 'Val', 'Test']
    x_pos = np.arange(3)

    bottom = np.zeros(3)
    for cat, color in zip(categories, colors_cat):
        values = [splits['Entrenamiento\n(70%)'][cat],
                  splits['Validación\n(15%)'][cat],
                  splits['Prueba\n(15%)'][cat]]
        ax_bars.bar(x_pos, values, bottom=bottom, color=color, edgecolor='white',
                   linewidth=1, label=cat, width=0.6)
        bottom += values

    ax_bars.set_xticks(x_pos)
    ax_bars.set_xticklabels(['Entrenamiento\n(70%)', 'Validación\n(15%)', 'Prueba\n(15%)'])
    ax_bars.set_ylabel('Número de imágenes')
    ax_bars.legend(loc='upper right', framealpha=0.9)
    ax_bars.set_title('Distribución por conjunto', fontsize=11, pad=10)

    ax_bars.spines['top'].set_visible(False)
    ax_bars.spines['right'].set_visible(False)

    # Valores totales encima
    totals = [splits['Entrenamiento\n(70%)']['total'],
              splits['Validación\n(15%)']['total'],
              splits['Prueba\n(15%)']['total']]
    for i, (x, total) in enumerate(zip(x_pos, totals)):
        ax_bars.text(x, total + 10, str(total), ha='center', fontsize=10, fontweight='bold')

    plt.savefig(OUTPUT_DIR / 'slide9_split.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Guardado: {OUTPUT_DIR / 'slide9_split.png'}")


def create_slide10_variabilidad():
    """Slide 10: Variabilidad por landmark guió selección de loss."""
    print("Generando Slide 10: Variabilidad por landmark...")

    # Cargar datos reales
    df = load_coordinates()

    # Calcular desviación estándar por landmark
    std_per_landmark = []
    for i in range(1, 16):
        std_x = df[f'L{i}_x'].std()
        std_y = df[f'L{i}_y'].std()
        std_combined = np.sqrt(std_x**2 + std_y**2)
        std_per_landmark.append(std_combined)

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=COLORS_PRO['background'])

    fig.suptitle('La variabilidad por landmark guió la selección de la función de pérdida:\nlos costofrenicos (L14-L15) tienen 3× más variabilidad que los centrales',
                fontsize=14, fontweight='bold', y=0.97, color=COLORS_PRO['text_primary'])

    ax = fig.add_subplot(111)

    # Asignar colores por tipo de landmark
    colors = []
    for i in range(15):
        if i in [0, 1]:  # L1, L2 - eje
            colors.append(COLORS_PRO['lm_axis'])
        elif i in [8, 9, 10]:  # L9, L10, L11 - centrales
            colors.append(COLORS_PRO['lm_central'])
        elif i in [2, 3, 4, 5, 6, 7]:  # pares simétricos
            colors.append(COLORS_PRO['lm_symmetric'])
        else:  # L12-L15 - ángulos
            colors.append(COLORS_PRO['lm_corner'])

    x_pos = np.arange(15)
    bars = ax.bar(x_pos, std_per_landmark, color=colors, edgecolor='white', linewidth=1)

    # Etiquetas
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'L{i+1}' for i in range(15)], fontsize=9)
    ax.set_xlabel('Landmark', fontsize=11)
    ax.set_ylabel('Desviación estándar combinada (píxeles)', fontsize=11)

    # Líneas de referencia
    mean_std = np.mean(std_per_landmark)
    ax.axhline(y=mean_std, color=COLORS_PRO['text_secondary'], linestyle='--',
               linewidth=1, alpha=0.7, label=f'Media: {mean_std:.1f} px')

    # Destacar máximo y mínimo
    max_idx = np.argmax(std_per_landmark)
    min_idx = np.argmin(std_per_landmark)

    ax.annotate(f'Máx: {std_per_landmark[max_idx]:.1f} px',
               xy=(max_idx, std_per_landmark[max_idx]),
               xytext=(max_idx, std_per_landmark[max_idx] + 8),
               fontsize=9, ha='center', color=COLORS_PRO['danger'],
               arrowprops=dict(arrowstyle='->', color=COLORS_PRO['danger'], lw=1))

    ax.annotate(f'Mín: {std_per_landmark[min_idx]:.1f} px',
               xy=(min_idx, std_per_landmark[min_idx]),
               xytext=(min_idx + 1.5, std_per_landmark[min_idx] + 15),
               fontsize=9, ha='center', color=COLORS_PRO['success'],
               arrowprops=dict(arrowstyle='->', color=COLORS_PRO['success'], lw=1))

    # Leyenda de tipos
    legend_elements = [
        mpatches.Patch(facecolor=COLORS_PRO['lm_axis'], label='Eje traqueal (L1-L2)'),
        mpatches.Patch(facecolor=COLORS_PRO['lm_symmetric'], label='Pares bilaterales (L3-L8)'),
        mpatches.Patch(facecolor=COLORS_PRO['lm_central'], label='Centrales (L9-L11)'),
        mpatches.Patch(facecolor=COLORS_PRO['lm_corner'], label='Ángulos (L12-L15)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max(std_per_landmark) * 1.3)

    # Nota sobre Wing Loss
    ratio = std_per_landmark[max_idx] / std_per_landmark[min_idx]
    fig.text(0.5, 0.02,
            f'Ratio máx/mín: {ratio:.1f}× → Wing Loss: robusta ante outliers, sensible a errores pequeños',
            ha='center', fontsize=9, color=COLORS_PRO['text_secondary'], style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.savefig(OUTPUT_DIR / 'slide10_variabilidad.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Guardado: {OUTPUT_DIR / 'slide10_variabilidad.png'}")


def create_slide11_geometria_central():
    """Slide 11: Geometría L9-L10-L11 define eje central."""
    print("Generando Slide 11: Geometría del eje central...")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=COLORS_PRO['background'])

    fig.suptitle('Descubrimiento geométrico: L9, L10 y L11 dividen el segmento L1-L2\nen cuatro partes aproximadamente iguales (t ≈ 0.25, 0.50, 0.75)',
                fontsize=14, fontweight='bold', y=0.97, color=COLORS_PRO['text_primary'])

    # Dos paneles
    ax_img = fig.add_axes([0.02, 0.08, 0.45, 0.80])
    ax_diagram = fig.add_axes([0.52, 0.08, 0.46, 0.80])

    # === Panel izquierdo: Radiografía con eje ===
    img_path, coords, _ = load_sample_image_with_coords('Normal', 50)

    if img_path and img_path.exists():
        img = Image.open(img_path).convert('L')
        ax_img.imshow(np.array(img), cmap='gray')

        # Dibujar eje L1-L2 prominente
        L1 = coords[0]
        L2 = coords[1]
        ax_img.plot([L1[0], L2[0]], [L1[1], L2[1]], '-',
                   color=COLORS_PRO['lm_axis'], linewidth=3, alpha=0.8)

        # Marcar L1 y L2
        ax_img.plot(L1[0], L1[1], 'o', color=COLORS_PRO['lm_axis'], markersize=12,
                   markeredgecolor='white', markeredgewidth=2)
        ax_img.annotate('L1', (L1[0]+10, L1[1]-5), color='white', fontsize=11,
                       fontweight='bold',
                       path_effects=[pe.withStroke(linewidth=3, foreground=COLORS_PRO['lm_axis'])])

        ax_img.plot(L2[0], L2[1], 'o', color=COLORS_PRO['lm_axis'], markersize=12,
                   markeredgecolor='white', markeredgewidth=2)
        ax_img.annotate('L2', (L2[0]+10, L2[1]+5), color='white', fontsize=11,
                       fontweight='bold',
                       path_effects=[pe.withStroke(linewidth=3, foreground=COLORS_PRO['lm_axis'])])

        # Marcar L9, L10, L11 (centrales)
        central_colors = ['#1a936f', '#2d6a4f', '#40916c']
        for i, idx in enumerate([8, 9, 10]):  # L9, L10, L11
            lm = coords[idx]
            ax_img.plot(lm[0], lm[1], 's', color=central_colors[i], markersize=12,
                       markeredgecolor='white', markeredgewidth=2)
            ax_img.annotate(f'L{idx+1}', (lm[0]+10, lm[1]), color='white', fontsize=10,
                           fontweight='bold',
                           path_effects=[pe.withStroke(linewidth=3, foreground=central_colors[i])])

            # Línea horizontal de referencia
            ax_img.axhline(y=lm[1], color=central_colors[i], linestyle=':', alpha=0.5, linewidth=1)

    ax_img.axis('off')
    ax_img.set_title('Eje traqueal y landmarks centrales', fontsize=11, pad=5)

    # === Panel derecho: Diagrama geométrico ===
    ax_diagram.set_xlim(0, 10)
    ax_diagram.set_ylim(0, 10)
    ax_diagram.axis('off')

    # Eje vertical con puntos
    x_axis = 3
    y_top = 9
    y_bottom = 1

    # Línea del eje
    ax_diagram.plot([x_axis, x_axis], [y_top, y_bottom], '-',
                   color=COLORS_PRO['lm_axis'], linewidth=4)

    # Puntos
    points = [
        ('L1', y_top, COLORS_PRO['lm_axis'], 't = 0'),
        ('L9', y_top - 2, COLORS_PRO['lm_central'], 't ≈ 0.25'),
        ('L10', (y_top + y_bottom) / 2, COLORS_PRO['lm_central'], 't ≈ 0.50'),
        ('L11', y_bottom + 2, COLORS_PRO['lm_central'], 't ≈ 0.75'),
        ('L2', y_bottom, COLORS_PRO['lm_axis'], 't = 1'),
    ]

    for label, y, color, t_val in points:
        ax_diagram.plot(x_axis, y, 'o' if 'L1' in label or 'L2' in label else 's',
                       color=color, markersize=14, markeredgecolor='white', markeredgewidth=2)
        ax_diagram.text(x_axis - 0.8, y, label, fontsize=12, ha='right', va='center',
                       fontweight='bold', color=color)
        ax_diagram.text(x_axis + 0.6, y, t_val, fontsize=10, ha='left', va='center',
                       color=COLORS_PRO['text_secondary'])

    # Fórmula paramétrica
    ax_diagram.text(7, 8.5, 'Fórmula paramétrica:', fontsize=11, fontweight='bold',
                   color=COLORS_PRO['text_primary'])

    formula_box = FancyBboxPatch((4.5, 6), 5, 2,
                                  boxstyle="round,pad=0.03",
                                  facecolor=COLORS_PRO['background_alt'],
                                  edgecolor=COLORS_PRO['text_secondary'],
                                  linewidth=1)
    ax_diagram.add_patch(formula_box)

    ax_diagram.text(7, 7.3, 'P(t) = L1 + t × (L2 − L1)', fontsize=12, ha='center',
                   family='monospace', color=COLORS_PRO['accent_primary'])
    ax_diagram.text(7, 6.5, 'donde t ∈ [0, 1]', fontsize=10, ha='center',
                   color=COLORS_PRO['text_secondary'])

    # Tabla de valores
    ax_diagram.text(7, 5.2, 'Valores observados:', fontsize=11, fontweight='bold',
                   color=COLORS_PRO['text_primary'])

    table_data = [
        ('L9', '0.25 ± 0.05'),
        ('L10', '0.50 ± 0.03'),
        ('L11', '0.75 ± 0.05')
    ]

    y_table = 4.5
    for lm, t_val in table_data:
        ax_diagram.text(5.5, y_table, lm, fontsize=10, ha='center',
                       color=COLORS_PRO['lm_central'], fontweight='bold')
        ax_diagram.text(8, y_table, f't = {t_val}', fontsize=10, ha='center',
                       color=COLORS_PRO['text_secondary'])
        y_table -= 0.6

    # Nota sobre implicación
    ax_diagram.text(7, 1.8, 'Implicación: pérdida de simetría se puede\nmonitorear sin ground truth adicional',
                   fontsize=9, ha='center', color=COLORS_PRO['text_secondary'], style='italic',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS_PRO['accent_light'],
                            edgecolor=COLORS_PRO['accent_secondary'], linewidth=0.5))

    plt.savefig(OUTPUT_DIR / 'slide11_geometria_central.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Guardado: {OUTPUT_DIR / 'slide11_geometria_central.png'}")


def create_slide12_asimetria():
    """Slide 12: Asimetría natural requiere tratamiento de pares."""
    print("Generando Slide 12: Asimetría natural...")

    # Cargar datos
    df = load_coordinates()

    # Calcular asimetrías para cada par
    asimetrias = {}
    pair_names = {
        (2, 3): 'Ápices (L3-L4)',
        (4, 5): 'Hilios sup. (L5-L6)',
        (6, 7): 'Hilios inf. (L7-L8)',
        (11, 12): 'Cardiofrénicos (L12-L13)',
        (13, 14): 'Costofrénicos (L14-L15)'
    }

    for (i, j), name in pair_names.items():
        # Distancia horizontal entre pares (asimetría)
        dist_x = np.abs(df[f'L{i+1}_x'] - df[f'L{j+1}_x'])
        # Diferencia vertical
        dist_y = np.abs(df[f'L{i+1}_y'] - df[f'L{j+1}_y'])
        asimetrias[name] = {
            'x_mean': dist_x.mean(),
            'x_std': dist_x.std(),
            'y_mean': dist_y.mean(),
            'y_std': dist_y.std()
        }

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=COLORS_PRO['background'])

    fig.suptitle('La asimetría natural del cuerpo humano (~6.3 px promedio)\nimplica que no se debe forzar simetría perfecta en la pérdida',
                fontsize=14, fontweight='bold', y=0.97, color=COLORS_PRO['text_primary'])

    # Dos paneles
    ax_img = fig.add_axes([0.02, 0.12, 0.40, 0.75])
    ax_bars = fig.add_axes([0.50, 0.12, 0.47, 0.75])

    # === Panel izquierdo: Radiografía con pares ===
    img_path, coords, _ = load_sample_image_with_coords('Normal', 50)

    if img_path and img_path.exists():
        img = Image.open(img_path).convert('L')
        ax_img.imshow(np.array(img), cmap='gray')

        # Dibujar pares simétricos con líneas de conexión
        pair_colors = ['#e63946', '#f4a261', '#2a9d8f', '#264653', '#9b2226']

        for (i, j), color in zip(SYMMETRIC_PAIRS, pair_colors):
            lm_i = coords[i]
            lm_j = coords[j]

            # Línea conectando el par
            ax_img.plot([lm_i[0], lm_j[0]], [lm_i[1], lm_j[1]], '--',
                       color=color, linewidth=2, alpha=0.8)

            # Puntos
            ax_img.plot(lm_i[0], lm_i[1], 'o', color=color, markersize=10,
                       markeredgecolor='white', markeredgewidth=1.5)
            ax_img.plot(lm_j[0], lm_j[1], 'o', color=color, markersize=10,
                       markeredgecolor='white', markeredgewidth=1.5)

            # Etiquetas
            ax_img.annotate(f'L{i+1}', (lm_i[0]-15, lm_i[1]), color='white', fontsize=8,
                           fontweight='bold', ha='right',
                           path_effects=[pe.withStroke(linewidth=2, foreground=color)])
            ax_img.annotate(f'L{j+1}', (lm_j[0]+10, lm_j[1]), color='white', fontsize=8,
                           fontweight='bold',
                           path_effects=[pe.withStroke(linewidth=2, foreground=color)])

    ax_img.axis('off')
    ax_img.set_title('Pares simétricos bilaterales', fontsize=11, pad=5)

    # === Panel derecho: Barras de asimetría ===
    pair_labels = list(pair_names.values())
    y_asym = [asimetrias[name]['y_mean'] for name in pair_labels]
    y_std = [asimetrias[name]['y_std'] for name in pair_labels]

    y_pos = np.arange(len(pair_labels))
    bars = ax_bars.barh(y_pos, y_asym, xerr=y_std, color=pair_colors,
                        edgecolor='white', linewidth=1, height=0.6,
                        capsize=3, error_kw={'linewidth': 1})

    ax_bars.set_yticks(y_pos)
    ax_bars.set_yticklabels(pair_labels)
    ax_bars.set_xlabel('Asimetría vertical promedio (píxeles)', fontsize=11)
    ax_bars.invert_yaxis()

    # Línea de referencia
    mean_asym = np.mean(y_asym)
    ax_bars.axvline(x=mean_asym, color=COLORS_PRO['text_secondary'], linestyle='--',
                   linewidth=1.5, alpha=0.7, label=f'Media: {mean_asym:.1f} px')

    ax_bars.legend(loc='lower right')
    ax_bars.spines['top'].set_visible(False)
    ax_bars.spines['right'].set_visible(False)
    ax_bars.set_title('Diferencia vertical en altura entre pares', fontsize=11, pad=10)

    # Valores en barras
    for bar, val in zip(bars, y_asym):
        ax_bars.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}', va='center', fontsize=9, color=COLORS_PRO['text_secondary'])

    # Nota inferior
    fig.text(0.5, 0.02,
            'Conclusión: usar pérdida de simetría suave (peso bajo) en lugar de forzar simetría exacta\n'
            'Flip horizontal en augmentación requiere intercambiar landmarks de cada par',
            ha='center', fontsize=9, color=COLORS_PRO['text_secondary'], style='italic')

    plt.savefig(OUTPUT_DIR / 'slide12_asimetria.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Guardado: {OUTPUT_DIR / 'slide12_asimetria.png'}")


def main():
    """Genera todas las visualizaciones del Bloque 2."""
    print("=" * 65)
    print("BLOQUE 2 - METODOLOGÍA DE DATOS (Slides 8-12)")
    print("Estilo: v2_profesional")
    print("=" * 65)
    print(f"Resolución: {int(FIG_WIDTH*DPI)}x{int(FIG_HEIGHT*DPI)} px (max)")
    print(f"Salida: {OUTPUT_DIR}")
    print()

    create_directories()

    create_slide8_regresion()
    create_slide9_split()
    create_slide10_variabilidad()
    create_slide11_geometria_central()
    create_slide12_asimetria()

    print()
    print("=" * 65)
    print("COMPLETADO - 5 slides generadas")
    print("=" * 65)


if __name__ == '__main__':
    main()
