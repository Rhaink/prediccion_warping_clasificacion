#!/usr/bin/env python3
"""
Script: Visualizaciones Bloque 3 - PREPROCESAMIENTO (Slides 13-16)
Estilo: v2_profesional

Slides:
- 13: El preprocesamiento transforma radiografías crudas en tensores normalizados
- 14: CLAHE mejora el contraste local especialmente en consolidaciones COVID
- 15: El flip horizontal requiere intercambiar landmarks de pares simétricos
- 16: La normalización ImageNet permite aprovechar transfer learning
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Arrow
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2

# ============================================================================
# CONFIGURACIÓN ESTILO v2_PROFESIONAL
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
})

# Rutas
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = PROJECT_ROOT
OUTPUT_DIR = BASE_DIR / 'presentacion' / '03_preprocesamiento' / 'v2_profesional'
DATA_DIR = BASE_DIR / 'data'
DATASET_DIR = DATA_DIR / 'dataset'

# Resolución
FIG_WIDTH = 14
FIG_HEIGHT = 7.5
DPI = 100

# Pares simétricos (0-indexed)
SYMMETRIC_PAIRS = [(2, 3), (4, 5), (6, 7), (11, 12), (13, 14)]

# ImageNet stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def create_directories():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_coordinates():
    csv_path = DATA_DIR / 'coordenadas' / 'coordenadas_maestro.csv'
    coord_cols = []
    for i in range(1, 16):
        coord_cols.extend([f'L{i}_x', f'L{i}_y'])
    columns = ['idx'] + coord_cols + ['image_name']
    df = pd.read_csv(csv_path, header=None, names=columns)
    return df


def load_sample_image_with_coords(category='Normal', index=50):
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


def apply_clahe(img_array, clip_limit=2.0, tile_size=4):
    """Aplica CLAHE a una imagen."""
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img_gray)


def create_slide13_pipeline():
    """Slide 13: Pipeline de preprocesamiento."""
    print("Generando Slide 13: Pipeline de preprocesamiento...")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=COLORS_PRO['background'])

    fig.suptitle('El preprocesamiento transforma radiografias crudas en tensores\nnormalizados listos para la red neuronal',
                fontsize=14, fontweight='bold', y=0.96, color=COLORS_PRO['text_primary'])

    # Cargar imagen
    img_path, coords, _ = load_sample_image_with_coords('COVID', 10)

    if img_path and img_path.exists():
        img_original = np.array(Image.open(img_path))
    else:
        img_original = np.random.randint(0, 255, (299, 299), dtype=np.uint8)

    # Pasos del pipeline
    steps = []

    # 1. Original
    if len(img_original.shape) == 2:
        step1 = img_original
    else:
        step1 = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)
    steps.append(('1. Original\n(299x299)', step1, 'gray'))

    # 2. RGB (3 canales)
    step2 = np.stack([step1, step1, step1], axis=-1)
    steps.append(('2. RGB\n(3 canales)', step2, None))

    # 3. CLAHE
    step3 = apply_clahe(step1)
    steps.append(('3. CLAHE\n(clip=2.0)', step3, 'gray'))

    # 4. Resize
    step4 = cv2.resize(step3, (224, 224))
    steps.append(('4. Resize\n(224x224)', step4, 'gray'))

    # 5. Normalizado [0,1]
    step5 = step4.astype(np.float32) / 255.0
    steps.append(('5. Normalizado\n[0, 1]', step5, 'gray'))

    # 6. ImageNet norm
    step6 = (step5 - 0.485) / 0.229  # Aproximación para grayscale
    steps.append(('6. ImageNet\n(tensor)', step6, 'gray'))

    # Crear subplots
    n_steps = len(steps)
    axes = []
    img_width = 0.12
    spacing = 0.02
    start_x = 0.05

    for i, (title, img, cmap) in enumerate(steps):
        ax = fig.add_axes([start_x + i * (img_width + spacing + 0.02), 0.25, img_width, 0.50])
        axes.append(ax)

        if cmap == 'gray':
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)

        ax.axis('off')
        ax.set_title(title, fontsize=9, pad=5, color=COLORS_PRO['text_primary'])

        # Flechas entre pasos
        if i < n_steps - 1:
            arrow_x = start_x + (i + 1) * (img_width + spacing + 0.02) - 0.02
            fig.patches.append(FancyArrowPatch(
                (arrow_x - 0.01, 0.50), (arrow_x + 0.01, 0.50),
                arrowstyle='-|>', mutation_scale=15,
                color=COLORS_PRO['accent_primary'], lw=2,
                transform=fig.transFigure, figure=fig
            ))

    # Descripción de cada paso
    descriptions = [
        'Imagen JPEG\noriginal',
        'Conversion a\n3 canales',
        'Mejora de\ncontraste local',
        'Redimension\nestandar',
        'Escalado a\nrango [0,1]',
        'Normalizacion\nImageNet'
    ]

    for i, desc in enumerate(descriptions):
        fig.text(start_x + i * (img_width + spacing + 0.02) + img_width/2, 0.15,
                desc, ha='center', fontsize=8, color=COLORS_PRO['text_secondary'],
                style='italic')

    # Nota inferior
    fig.text(0.5, 0.04,
            'Parametros CLAHE: clip_limit=2.0, tile_grid_size=4x4 | ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]',
            ha='center', fontsize=9, color=COLORS_PRO['text_secondary'])

    plt.savefig(OUTPUT_DIR / 'slide13_pipeline.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor(), pad_inches=0.1)
    plt.close()
    print(f"  Guardado: {OUTPUT_DIR / 'slide13_pipeline.png'}")


def create_slide14_clahe():
    """Slide 14: CLAHE mejora contraste en COVID."""
    print("Generando Slide 14: CLAHE en COVID...")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=COLORS_PRO['background'])

    fig.suptitle('CLAHE mejora el contraste local, revelando detalles\nen consolidaciones COVID que el modelo debe detectar',
                fontsize=13, fontweight='bold', y=0.99, color=COLORS_PRO['text_primary'])

    # Cargar imagenes de diferentes categorias
    categories = [
        ('COVID', 'COVID-19\n(consolidaciones)', COLORS_PRO['covid']),
        ('Normal', 'Normal\n(pulmones claros)', COLORS_PRO['normal']),
        ('Viral_Pneumonia', 'Neumonia viral\n(infiltrados)', COLORS_PRO['viral'])
    ]

    for row, (cat, label, color) in enumerate(categories):
        # Cargar imagen
        img_path, _, _ = load_sample_image_with_coords(cat, 15)

        if img_path and img_path.exists():
            img = np.array(Image.open(img_path).convert('L'))
        else:
            img = np.random.randint(50, 200, (299, 299), dtype=np.uint8)

        # Resize para uniformidad
        img = cv2.resize(img, (224, 224))
        img_clahe = apply_clahe(img)

        # Diferencia (realzada para visualizacion)
        diff = np.abs(img.astype(np.float32) - img_clahe.astype(np.float32))
        diff = (diff / diff.max() * 255).astype(np.uint8)

        # Subplot: Original
        ax1 = fig.add_axes([0.05, 0.68 - row * 0.28, 0.22, 0.24])
        ax1.imshow(img, cmap='gray')
        ax1.axis('off')
        if row == 0:
            ax1.set_title('Original', fontsize=11, color=COLORS_PRO['text_primary'], pad=5)

        # Etiqueta de categoria
        ax1.text(-0.15, 0.5, label, transform=ax1.transAxes, fontsize=10,
                va='center', ha='right', color=color, fontweight='bold')

        # Subplot: Con CLAHE
        ax2 = fig.add_axes([0.32, 0.68 - row * 0.28, 0.22, 0.24])
        ax2.imshow(img_clahe, cmap='gray')
        ax2.axis('off')
        if row == 0:
            ax2.set_title('Con CLAHE', fontsize=11, color=COLORS_PRO['text_primary'], pad=5)

        # Subplot: Diferencia
        ax3 = fig.add_axes([0.59, 0.68 - row * 0.28, 0.22, 0.24])
        ax3.imshow(diff, cmap='hot')
        ax3.axis('off')
        if row == 0:
            ax3.set_title('Diferencia\n(zonas mejoradas)', fontsize=11,
                         color=COLORS_PRO['text_primary'], pad=5)

    # Panel derecho: Explicacion
    ax_text = fig.add_axes([0.82, 0.15, 0.16, 0.70])
    ax_text.axis('off')

    ax_text.text(0.5, 0.95, 'CLAHE', fontsize=12, ha='center', fontweight='bold',
                color=COLORS_PRO['accent_primary'])

    explanation = """Contrast Limited
Adaptive Histogram
Equalization

Divide la imagen
en bloques (tiles)
y ecualiza cada
uno localmente.

Beneficios:
- Mejora detalles
  en zonas oscuras
- Revela bordes
  de consolidaciones
- Mantiene
  estructuras globales

Parametros:
clip = 2.0
tile = 4x4"""

    ax_text.text(0.5, 0.75, explanation, fontsize=8, ha='center', va='top',
                color=COLORS_PRO['text_secondary'], linespacing=1.3)

    # Nota inferior
    fig.text(0.5, 0.03,
            'El mapa de calor muestra donde CLAHE tiene mayor impacto: bordes, consolidaciones y zonas de bajo contraste',
            ha='center', fontsize=9, color=COLORS_PRO['text_secondary'], style='italic')

    plt.savefig(OUTPUT_DIR / 'slide14_clahe.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor(), pad_inches=0.1)
    plt.close()
    print(f"  Guardado: {OUTPUT_DIR / 'slide14_clahe.png'}")


def create_slide15_flip():
    """Slide 15: Flip horizontal con intercambio de landmarks."""
    print("Generando Slide 15: Flip con intercambio...")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=COLORS_PRO['background'])

    fig.suptitle('El flip horizontal requiere intercambiar las coordenadas\nde los pares simetricos para mantener consistencia anatomica',
                fontsize=14, fontweight='bold', y=0.96, color=COLORS_PRO['text_primary'])

    # Cargar imagen y coordenadas
    img_path, coords, _ = load_sample_image_with_coords('Normal', 50)

    if img_path and img_path.exists():
        img = np.array(Image.open(img_path).convert('L'))
    else:
        img = np.random.randint(50, 200, (299, 299), dtype=np.uint8)

    img_width_px = img.shape[1]

    # Crear flip
    img_flipped = np.fliplr(img)

    # Coordenadas flippeadas (sin intercambio - INCORRECTO)
    coords_flip_wrong = coords.copy()
    coords_flip_wrong[:, 0] = img_width_px - coords_flip_wrong[:, 0]

    # Coordenadas flippeadas (con intercambio - CORRECTO)
    coords_flip_correct = coords_flip_wrong.copy()
    for i, j in SYMMETRIC_PAIRS:
        coords_flip_correct[i], coords_flip_correct[j] = \
            coords_flip_correct[j].copy(), coords_flip_correct[i].copy()

    # Colores para landmarks
    def get_lm_color(idx):
        if idx in [0, 1]:
            return COLORS_PRO['accent_primary']  # Eje
        elif idx in [8, 9, 10]:
            return COLORS_PRO['data_2']  # Centrales
        else:
            return COLORS_PRO['data_4']  # Pares simetricos

    # Tres paneles
    panels = [
        ('Original', img, coords, COLORS_PRO['accent_primary']),
        ('Flip INCORRECTO\n(sin intercambio)', img_flipped, coords_flip_wrong, COLORS_PRO['danger']),
        ('Flip CORRECTO\n(con intercambio)', img_flipped, coords_flip_correct, COLORS_PRO['success'])
    ]

    for col, (title, img_show, coords_show, border_color) in enumerate(panels):
        ax = fig.add_axes([0.03 + col * 0.32, 0.18, 0.28, 0.65])
        ax.imshow(img_show, cmap='gray')

        # Dibujar landmarks
        for i, (x, y) in enumerate(coords_show):
            color = get_lm_color(i)
            # Destacar pares simetricos
            if i in [idx for pair in SYMMETRIC_PAIRS for idx in pair]:
                marker = 'o'
                size = 8
            else:
                marker = 's'
                size = 6

            ax.plot(x, y, marker, color=color, markersize=size,
                   markeredgecolor='white', markeredgewidth=1)
            ax.annotate(f'L{i+1}', (x+5, y-3), color='white', fontsize=7,
                       fontweight='bold',
                       path_effects=[pe.withStroke(linewidth=2, foreground=color)])

        ax.axis('off')

        # Borde de color segun estado
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(3)

        ax.set_title(title, fontsize=11, color=border_color, fontweight='bold', pad=8)

    # Diagrama de intercambio
    ax_diagram = fig.add_axes([0.35, 0.02, 0.30, 0.12])
    ax_diagram.axis('off')

    ax_diagram.text(0.5, 0.9, 'Pares simetricos intercambiados:', fontsize=9,
                   ha='center', fontweight='bold', color=COLORS_PRO['text_primary'])

    pairs_text = 'L3<->L4   L5<->L6   L7<->L8   L12<->L13   L14<->L15'
    ax_diagram.text(0.5, 0.3, pairs_text, fontsize=9, ha='center',
                   family='monospace', color=COLORS_PRO['data_4'],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS_PRO['background_alt'],
                            edgecolor=COLORS_PRO['data_4'], linewidth=1))

    # Nota sobre augmentacion
    fig.text(0.5, 0.02,
            'Data augmentation: flip horizontal con p=0.5 | Sin intercambio, la anatomia izquierda/derecha queda inconsistente',
            ha='center', fontsize=9, color=COLORS_PRO['text_secondary'], style='italic')

    plt.savefig(OUTPUT_DIR / 'slide15_flip.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor(), pad_inches=0.1)
    plt.close()
    print(f"  Guardado: {OUTPUT_DIR / 'slide15_flip.png'}")


def create_slide16_normalizacion():
    """Slide 16: Normalización ImageNet."""
    print("Generando Slide 16: Normalizacion ImageNet...")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=COLORS_PRO['background'])

    fig.suptitle('La normalización con estadísticas ImageNet permite\naprovechar el conocimiento del backbone pre-entrenado',
                fontsize=14, fontweight='bold', y=0.97, color=COLORS_PRO['text_primary'])

    # Cargar imagen
    img_path, _, _ = load_sample_image_with_coords('Normal', 50)

    if img_path and img_path.exists():
        img = np.array(Image.open(img_path).convert('L'))
    else:
        img = np.random.randint(50, 200, (224, 224), dtype=np.uint8)

    img = cv2.resize(img, (224, 224))

    # Diferentes normalizaciones
    img_raw = img.astype(np.float32)  # [0, 255]
    img_01 = img_raw / 255.0  # [0, 1]
    img_imagenet = (img_01 - 0.485) / 0.229  # ImageNet (canal gris aproximado)

    # Panel izquierdo: Histogramas
    ax_hist1 = fig.add_axes([0.05, 0.55, 0.25, 0.30])
    ax_hist2 = fig.add_axes([0.05, 0.15, 0.25, 0.30])

    # Histograma antes
    ax_hist1.hist(img_01.flatten(), bins=50, color=COLORS_PRO['accent_secondary'],
                  alpha=0.7, edgecolor='white')
    ax_hist1.set_title('Antes: Rango [0, 1]', fontsize=10, color=COLORS_PRO['text_primary'])
    ax_hist1.set_xlabel('Valor de pixel', fontsize=9)
    ax_hist1.set_ylabel('Frecuencia', fontsize=9)
    ax_hist1.axvline(x=img_01.mean(), color=COLORS_PRO['danger'], linestyle='--',
                    label=f'Media: {img_01.mean():.3f}')
    ax_hist1.legend(fontsize=8)
    ax_hist1.spines['top'].set_visible(False)
    ax_hist1.spines['right'].set_visible(False)

    # Histograma despues
    ax_hist2.hist(img_imagenet.flatten(), bins=50, color=COLORS_PRO['data_2'],
                  alpha=0.7, edgecolor='white')
    ax_hist2.set_title('Despues: Normalizado ImageNet', fontsize=10, color=COLORS_PRO['text_primary'])
    ax_hist2.set_xlabel('Valor normalizado', fontsize=9)
    ax_hist2.set_ylabel('Frecuencia', fontsize=9)
    ax_hist2.axvline(x=img_imagenet.mean(), color=COLORS_PRO['danger'], linestyle='--',
                    label=f'Media: {img_imagenet.mean():.3f}')
    ax_hist2.axvline(x=0, color=COLORS_PRO['text_secondary'], linestyle=':', alpha=0.5)
    ax_hist2.legend(fontsize=8)
    ax_hist2.spines['top'].set_visible(False)
    ax_hist2.spines['right'].set_visible(False)

    # Panel central: Formula
    ax_formula = fig.add_axes([0.35, 0.40, 0.30, 0.45])
    ax_formula.axis('off')

    ax_formula.text(0.5, 0.95, 'Formula de normalizacion', fontsize=11, ha='center',
                   fontweight='bold', color=COLORS_PRO['text_primary'])

    # Caja con formula
    formula_box = FancyBboxPatch((0.05, 0.55), 0.90, 0.30,
                                  boxstyle="round,pad=0.02",
                                  facecolor=COLORS_PRO['accent_light'],
                                  edgecolor=COLORS_PRO['accent_primary'],
                                  linewidth=1.5)
    ax_formula.add_patch(formula_box)

    ax_formula.text(0.5, 0.72, 'x_norm = (x - mean) / std', fontsize=14, ha='center',
                   family='monospace', color=COLORS_PRO['accent_primary'], fontweight='bold')

    # Valores ImageNet
    ax_formula.text(0.5, 0.45, 'Estadisticas ImageNet:', fontsize=10, ha='center',
                   fontweight='bold', color=COLORS_PRO['text_primary'])

    stats_text = """mean = [0.485, 0.456, 0.406]  (RGB)
std  = [0.229, 0.224, 0.225]  (RGB)"""
    ax_formula.text(0.5, 0.28, stats_text, fontsize=9, ha='center',
                   family='monospace', color=COLORS_PRO['text_secondary'])

    # Nota sobre grayscale
    ax_formula.text(0.5, 0.08, 'Para grayscale: se replica el canal\ny se aplica la misma normalizacion',
                   fontsize=8, ha='center', style='italic', color=COLORS_PRO['text_secondary'])

    # Panel derecho: Por que funciona
    ax_why = fig.add_axes([0.68, 0.15, 0.30, 0.70])
    ax_why.axis('off')

    ax_why.text(0.5, 0.95, 'Por que ImageNet?', fontsize=11, ha='center',
               fontweight='bold', color=COLORS_PRO['accent_primary'])

    benefits = [
        ('Transfer Learning', 'ResNet-18 fue entrenada\ncon estas estadisticas'),
        ('Convergencia', 'Valores cercanos a 0\nfacilitan optimizacion'),
        ('Estabilidad', 'Gradientes mas estables\ndurante entrenamiento'),
        ('Compatibilidad', 'Mismo rango que otros\nmodelos pre-entrenados')
    ]

    y_pos = 0.82
    for title, desc in benefits:
        # Icono circular
        circle = plt.Circle((0.08, y_pos), 0.03, color=COLORS_PRO['success'],
                            transform=ax_why.transAxes)
        ax_why.add_patch(circle)
        ax_why.text(0.08, y_pos, '✓', fontsize=10, ha='center', va='center',
                   color='white', fontweight='bold')

        ax_why.text(0.18, y_pos + 0.02, title, fontsize=10, fontweight='bold',
                   color=COLORS_PRO['text_primary'])
        ax_why.text(0.18, y_pos - 0.08, desc, fontsize=8,
                   color=COLORS_PRO['text_secondary'])
        y_pos -= 0.22

    # Nota inferior
    fig.text(0.5, 0.03,
            'Sin normalizacion ImageNet, el backbone pre-entrenado recibiria datos fuera de su distribucion esperada',
            ha='center', fontsize=9, color=COLORS_PRO['text_secondary'], style='italic')

    plt.savefig(OUTPUT_DIR / 'slide16_normalizacion.png', dpi=DPI,
                bbox_inches='tight', facecolor=fig.get_facecolor(), pad_inches=0.1)
    plt.close()
    print(f"  Guardado: {OUTPUT_DIR / 'slide16_normalizacion.png'}")


def main():
    print("=" * 70)
    print("BLOQUE 3 - PREPROCESAMIENTO (Slides 13-16)")
    print("Estilo: v2_profesional")
    print("=" * 70)
    print(f"Resolucion: {int(FIG_WIDTH*DPI)}x{int(FIG_HEIGHT*DPI)} px")
    print(f"Salida: {OUTPUT_DIR}")
    print()

    create_directories()

    create_slide13_pipeline()
    create_slide14_clahe()
    create_slide15_flip()
    create_slide16_normalizacion()

    print()
    print("=" * 70)
    print("COMPLETADO - 4 slides generadas")
    print("=" * 70)


if __name__ == '__main__':
    main()
