#!/usr/bin/env python3
"""
Generador de visualizaciones detalladas del pipeline de procesamiento.
Sesion 17: Visualizaciones Detalladas del Pipeline.

Este script genera visualizaciones de:
1. Pipeline de preprocesamiento (cada paso)
2. Pipeline de data augmentation
3. Pipeline de inferencia completo (ensemble + TTA)
4. Comparacion por categoria (Normal, COVID, Viral)

Las visualizaciones son para usar en la tesis y presentacion de defensa.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from pathlib import Path
import torch
from PIL import Image
import pandas as pd

# Agregar src_v2 al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src_v2.data.transforms import apply_clahe, SYMMETRIC_PAIRS
from src_v2.data.dataset import LandmarkDataset


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

# Colores consistentes para todo el proyecto
COLORS = {
    'primary': '#2E86AB',      # Azul principal
    'secondary': '#A23B72',    # Rosa/magenta
    'accent': '#F18F01',       # Naranja
    'success': '#C73E1D',      # Rojo ladrillo
    'neutral': '#3B3B3B',      # Gris oscuro
    'light': '#E8E8E8',        # Gris claro
    'gt': '#00FF00',           # Verde - Ground Truth
    'pred': '#FF0000',         # Rojo - Prediccion
    'central': '#FFFF00',      # Amarillo - Landmarks centrales
    'axis': '#00FFFF',         # Cyan - Eje L1-L2
    'covid': '#E63946',        # Rojo COVID
    'normal': '#2A9D8F',       # Verde Normal
    'viral': '#E9C46A',        # Amarillo Viral
}

LANDMARK_NAMES = [
    'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8',
    'L9', 'L10', 'L11', 'L12', 'L13', 'L14', 'L15'
]


def get_sample_images(data_dir='data/dataset', num_per_category=1):
    """Obtiene imagenes de ejemplo de cada categoria."""
    samples = {}
    categories = ['COVID', 'Normal', 'Viral_Pneumonia']

    for category in categories:
        cat_dir = os.path.join(data_dir, category)
        if os.path.exists(cat_dir):
            images = sorted([f for f in os.listdir(cat_dir) if f.endswith('.png')])
            if images:
                # Seleccionar imagen del medio para variedad
                idx = min(len(images) // 2, len(images) - 1)
                img_path = os.path.join(cat_dir, images[idx])
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    samples[category] = {
                        'image': img,
                        'path': img_path,
                        'name': images[idx]
                    }
                    print(f"  Cargada imagen: {img_path}")

    return samples


def load_landmarks_for_image(image_name, csv_path='data/coordenadas/coordenadas_maestro.csv'):
    """Carga landmarks para una imagen especifica."""
    # El CSV no tiene headers, la ultima columna es el nombre de imagen
    df = pd.read_csv(csv_path, header=None)

    # La ultima columna (31) tiene el nombre de imagen
    image_col = df.columns[-1]
    base_name = os.path.splitext(image_name)[0]

    # Buscar la imagen
    row = df[df[image_col] == base_name]

    if len(row) == 0:
        # Intentar con variaciones
        row = df[df[image_col].str.contains(base_name, na=False)]

    if len(row) > 0:
        # Columnas 1-30 son las coordenadas (0 es indice)
        coords = row.iloc[0, 1:31].values.astype(float)
        landmarks = coords.reshape(15, 2)
        return landmarks
    return None


def add_arrow_annotation(ax, text, xy_start, xy_end, color='black'):
    """Agrega una flecha con texto entre dos puntos."""
    ax.annotate(
        text,
        xy=xy_end,
        xytext=xy_start,
        fontsize=9,
        ha='center',
        va='center',
        arrowprops=dict(
            arrowstyle='->',
            color=color,
            lw=1.5,
            connectionstyle='arc3,rad=0'
        )
    )


def draw_step_box(ax, x, y, width, height, title, color, text_color='white'):
    """Dibuja una caja de paso del pipeline."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2),
        width, height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color,
        edgecolor='black',
        linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(x, y, title, ha='center', va='center',
            fontsize=9, fontweight='bold', color=text_color)


# ============================================================================
# PARTE 1: PIPELINE DE PREPROCESAMIENTO
# ============================================================================

def generate_preprocessing_pipeline(output_dir):
    """
    Genera visualizacion del pipeline de preprocesamiento paso a paso.

    Pasos:
    1. Imagen original (299x299, escala de grises)
    2. Conversion a RGB (3 canales)
    3. CLAHE aplicado
    4. Resize a 224x224
    5. Normalizacion ImageNet
    6. Conversion a tensor
    """
    print("\n" + "="*60)
    print("Generando Pipeline de Preprocesamiento...")
    print("="*60)

    output_path = os.path.join(output_dir, 'preprocessing')
    os.makedirs(output_path, exist_ok=True)

    # Obtener imagen de ejemplo
    samples = get_sample_images()
    if 'Normal' not in samples:
        print("Error: No se encontraron imagenes")
        return None

    sample = samples['Normal']
    img_original = sample['image']

    # Paso 1: Imagen original
    print("  Paso 1: Imagen original")
    step1 = img_original.copy()

    # Paso 2: Verificar/convertir a RGB (ya esta en RGB)
    print("  Paso 2: Conversion a RGB")
    step2 = step1.copy()  # Ya es RGB despues de cv2.cvtColor

    # Paso 3: CLAHE
    print("  Paso 3: Aplicando CLAHE")
    step3_pil = Image.fromarray(step2)
    step3_clahe = apply_clahe(step3_pil, clip_limit=2.0, tile_grid_size=(4, 4))
    step3 = np.array(step3_clahe)

    # Paso 4: Resize a 224x224
    print("  Paso 4: Resize a 224x224")
    step4 = cv2.resize(step3, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Paso 5: Normalizacion ImageNet
    print("  Paso 5: Normalizacion ImageNet")
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    step5 = (step4 / 255.0 - mean) / std
    # Para visualizacion, desnormalizar
    step5_vis = (step5 * std + mean)
    step5_vis = np.clip(step5_vis, 0, 1)

    # Paso 6: Conversion a tensor (representacion visual)
    print("  Paso 6: Conversion a tensor")
    step6 = step5_vis.copy()  # Visualmente igual

    # Guardar pasos individuales
    steps = [
        (step1, "step1_original.png", "1. Original\n(299x299)"),
        (step2, "step2_rgb.png", "2. RGB\n(3 canales)"),
        (step3, "step3_clahe.png", "3. CLAHE\n(clip=2.0, tile=4)"),
        (step4, "step4_resize.png", "4. Resize\n(224x224)"),
        (step5_vis, "step5_normalized.png", "5. Normalizado\n(ImageNet)"),
        (step6, "step6_tensor.png", "6. Tensor\n(C, H, W)")
    ]

    for img, filename, _ in steps:
        filepath = os.path.join(output_path, filename)
        if img.max() <= 1.0:
            img_save = (img * 255).astype(np.uint8)
        else:
            img_save = img.astype(np.uint8)
        cv2.imwrite(filepath, cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR))

    # Crear figura de montaje horizontal
    fig, axes = plt.subplots(1, 6, figsize=(18, 4))
    fig.suptitle('Pipeline de Preprocesamiento', fontsize=14, fontweight='bold', y=1.05)

    for idx, (img, _, title) in enumerate(steps):
        ax = axes[idx]
        if img.max() <= 1.0:
            ax.imshow(img)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

        # Agregar flechas entre pasos
        if idx < 5:
            # Posicion de la flecha (en coordenadas de figura)
            pass  # Las flechas se agregan despues

    # Agregar flechas
    for i in range(5):
        fig.text(
            (i + 1) / 6 - 0.005, 0.5,
            '\u2192',  # Flecha Unicode
            fontsize=20,
            ha='center', va='center',
            transform=fig.transFigure
        )

    plt.tight_layout()

    pipeline_path = os.path.join(output_path, 'preprocessing_pipeline.png')
    plt.savefig(pipeline_path, bbox_inches='tight', pad_inches=0.2)
    plt.close()

    print(f"  Guardado: {pipeline_path}")

    # Crear version con detalles tecnicos
    create_preprocessing_detailed(output_path, steps)

    return pipeline_path


def create_preprocessing_detailed(output_path, steps):
    """Crea version detallada del pipeline con anotaciones."""

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(2, 6, height_ratios=[3, 1], hspace=0.3)

    # Fila superior: imagenes
    for idx, (img, _, title) in enumerate(steps):
        ax = fig.add_subplot(gs[0, idx])
        if img.max() <= 1.0:
            ax.imshow(img)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

        # Info de shape
        if img.ndim == 3:
            h, w, c = img.shape
            shape_text = f"Shape: ({h}, {w}, {c})"
        else:
            h, w = img.shape
            shape_text = f"Shape: ({h}, {w})"

        # Rango de valores
        vmin, vmax = img.min(), img.max()
        range_text = f"Range: [{vmin:.2f}, {vmax:.2f}]"

        ax.text(0.5, -0.08, shape_text, transform=ax.transAxes,
                ha='center', fontsize=8, color='gray')
        ax.text(0.5, -0.15, range_text, transform=ax.transAxes,
                ha='center', fontsize=8, color='gray')

    # Fila inferior: descripcion de operaciones
    descriptions = [
        "Carga desde disco\nFormat: PNG\n8 bits/canal",
        "Conversion BGR\u2192RGB\nVerificar 3 canales",
        "CLAHE en espacio LAB\nclip_limit=2.0\ntile_grid=(4,4)",
        "Interpolacion bilinear\nAspect ratio 1:1\n299\u2192224 px",
        "mean=[0.485,0.456,0.406]\nstd=[0.229,0.224,0.225]\n(x-mean)/std",
        "torch.Tensor\nShape: (3, 224, 224)\ndtype: float32"
    ]

    for idx, desc in enumerate(descriptions):
        ax = fig.add_subplot(gs[1, idx])
        ax.text(0.5, 0.5, desc, ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor=COLORS['light'],
                         edgecolor=COLORS['neutral'], alpha=0.8))
        ax.axis('off')

    # Titulo general
    fig.suptitle('Pipeline de Preprocesamiento - Detalle Tecnico',
                 fontsize=14, fontweight='bold', y=0.98)

    detailed_path = os.path.join(output_path, 'preprocessing_pipeline_detailed.png')
    plt.savefig(detailed_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"  Guardado: {detailed_path}")


# ============================================================================
# PARTE 2: PIPELINE DE DATA AUGMENTATION
# ============================================================================

def generate_augmentation_pipeline(output_dir):
    """
    Genera visualizacion del pipeline de data augmentation.

    Muestra:
    - Imagen base
    - Flip horizontal (con intercambio de landmarks)
    - Rotacion +/-10 grados
    - Color jitter (brillo/contraste)
    - Combinacion de augmentations
    """
    print("\n" + "="*60)
    print("Generando Pipeline de Data Augmentation...")
    print("="*60)

    output_path = os.path.join(output_dir, 'augmentation')
    os.makedirs(output_path, exist_ok=True)

    # Obtener imagen de ejemplo con landmarks
    samples = get_sample_images()
    if 'Normal' not in samples:
        print("Error: No se encontraron imagenes")
        return None

    sample = samples['Normal']
    img_original = sample['image'].copy()

    # Cargar landmarks
    landmarks = load_landmarks_for_image(sample['name'])

    # Aplicar CLAHE y resize primero
    img_pil = Image.fromarray(img_original)
    img_clahe = apply_clahe(img_pil, clip_limit=2.0, tile_grid_size=(4, 4))
    img_base = np.array(img_clahe)
    img_base = cv2.resize(img_base, (224, 224))

    # Escalar landmarks
    if landmarks is not None:
        scale = 224 / 299
        landmarks_scaled = landmarks * scale
    else:
        landmarks_scaled = None

    # Crear augmentations
    augmentations = []

    # 1. Original
    augmentations.append({
        'image': img_base.copy(),
        'landmarks': landmarks_scaled.copy() if landmarks_scaled is not None else None,
        'title': 'Original',
        'desc': 'Imagen preprocesada\nsin augmentation'
    })

    # 2. Flip horizontal
    img_flip = cv2.flip(img_base, 1)
    if landmarks_scaled is not None:
        lm_flip = landmarks_scaled.copy()
        lm_flip[:, 0] = 224 - lm_flip[:, 0]
        # Intercambiar pares
        for left, right in SYMMETRIC_PAIRS:
            lm_flip[left], lm_flip[right] = lm_flip[right].copy(), lm_flip[left].copy()
    else:
        lm_flip = None

    augmentations.append({
        'image': img_flip,
        'landmarks': lm_flip,
        'title': 'Flip Horizontal',
        'desc': 'p=0.5\nIntercambio de pares\nsimetricos'
    })

    # 3. Rotacion +10
    M_rot = cv2.getRotationMatrix2D((112, 112), 10, 1.0)
    img_rot = cv2.warpAffine(img_base, M_rot, (224, 224))
    if landmarks_scaled is not None:
        lm_rot = landmarks_scaled.copy()
        ones = np.ones((15, 1))
        lm_hom = np.hstack([lm_rot, ones])
        lm_rot = (M_rot @ lm_hom.T).T
    else:
        lm_rot = None

    augmentations.append({
        'image': img_rot,
        'landmarks': lm_rot,
        'title': 'Rotacion +10\u00b0',
        'desc': 'Rango: \u00b110\u00b0\nCentro: (112, 112)'
    })

    # 4. Color jitter - brillo
    img_bright = np.clip(img_base * 1.2, 0, 255).astype(np.uint8)
    augmentations.append({
        'image': img_bright,
        'landmarks': landmarks_scaled.copy() if landmarks_scaled is not None else None,
        'title': 'Brillo +20%',
        'desc': 'Rango: 0.8-1.2\nNo afecta landmarks'
    })

    # 5. Combinacion
    img_comb = cv2.flip(img_base, 1)
    M_rot2 = cv2.getRotationMatrix2D((112, 112), -5, 1.0)
    img_comb = cv2.warpAffine(img_comb, M_rot2, (224, 224))
    img_comb = np.clip(img_comb * 1.1, 0, 255).astype(np.uint8)

    if landmarks_scaled is not None:
        lm_comb = landmarks_scaled.copy()
        lm_comb[:, 0] = 224 - lm_comb[:, 0]
        for left, right in SYMMETRIC_PAIRS:
            lm_comb[left], lm_comb[right] = lm_comb[right].copy(), lm_comb[left].copy()
        ones = np.ones((15, 1))
        lm_hom = np.hstack([lm_comb, ones])
        lm_comb = (M_rot2 @ lm_hom.T).T
    else:
        lm_comb = None

    augmentations.append({
        'image': img_comb,
        'landmarks': lm_comb,
        'title': 'Combinacion',
        'desc': 'Flip + Rot(-5\u00b0)\n+ Brillo(1.1)'
    })

    # Crear figura
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))

    for idx, aug in enumerate(augmentations):
        # Fila superior: imagen con landmarks
        ax = axes[0, idx]
        ax.imshow(aug['image'])

        if aug['landmarks'] is not None:
            lm = aug['landmarks']
            # Dibujar landmarks
            for i in range(15):
                color = COLORS['central'] if i in [0, 1, 8, 9, 10] else COLORS['gt']
                ax.scatter(lm[i, 0], lm[i, 1], s=40, c=color,
                          edgecolors='black', linewidths=0.5, zorder=5)

            # Dibujar eje L1-L2
            ax.plot([lm[0, 0], lm[1, 0]], [lm[0, 1], lm[1, 1]],
                   color=COLORS['axis'], linewidth=1.5, alpha=0.7)

        ax.set_title(aug['title'], fontsize=11, fontweight='bold')
        ax.axis('off')

        # Fila inferior: descripcion
        ax_desc = axes[1, idx]
        ax_desc.text(0.5, 0.5, aug['desc'], ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor=COLORS['light'],
                             edgecolor=COLORS['neutral']))
        ax_desc.axis('off')

    # Titulo y leyenda
    fig.suptitle('Pipeline de Data Augmentation', fontsize=14, fontweight='bold', y=0.98)

    # Leyenda de landmarks
    legend_elements = [
        plt.scatter([], [], c=COLORS['gt'], s=40, label='Landmarks laterales'),
        plt.scatter([], [], c=COLORS['central'], s=40, label='Landmarks centrales (L1,L2,L9-11)'),
        Line2D([0], [0], color=COLORS['axis'], linewidth=2, label='Eje central (L1-L2)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9,
              bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    pipeline_path = os.path.join(output_path, 'augmentation_pipeline.png')
    plt.savefig(pipeline_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"  Guardado: {pipeline_path}")

    # Crear visualizacion de intercambio de pares
    create_flip_pair_visualization(output_path, img_base, landmarks_scaled)

    return pipeline_path


def create_flip_pair_visualization(output_path, img_base, landmarks):
    """Crea visualizacion detallada del intercambio de pares en flip."""

    if landmarks is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    ax = axes[0]
    ax.imshow(img_base)
    for i in range(15):
        color = COLORS['central'] if i in [0, 1, 8, 9, 10] else COLORS['gt']
        ax.scatter(landmarks[i, 0], landmarks[i, 1], s=60, c=color,
                  edgecolors='black', linewidths=1, zorder=5)
        ax.annotate(f'L{i+1}', (landmarks[i, 0] + 5, landmarks[i, 1] - 5),
                   fontsize=7, color='white',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='green', alpha=0.7))
    ax.set_title('Original', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Flip sin intercambio (INCORRECTO)
    ax = axes[1]
    img_flip = cv2.flip(img_base, 1)
    lm_wrong = landmarks.copy()
    lm_wrong[:, 0] = 224 - lm_wrong[:, 0]
    ax.imshow(img_flip)
    for i in range(15):
        color = '#FF6B6B'  # Rojo claro para indicar error
        ax.scatter(lm_wrong[i, 0], lm_wrong[i, 1], s=60, c=color,
                  edgecolors='black', linewidths=1, zorder=5)
        ax.annotate(f'L{i+1}', (lm_wrong[i, 0] + 5, lm_wrong[i, 1] - 5),
                   fontsize=7, color='white',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.7))
    ax.set_title('Flip INCORRECTO\n(sin intercambio)', fontsize=12, fontweight='bold', color='red')
    ax.axis('off')

    # Flip con intercambio (CORRECTO)
    ax = axes[2]
    lm_correct = landmarks.copy()
    lm_correct[:, 0] = 224 - lm_correct[:, 0]
    for left, right in SYMMETRIC_PAIRS:
        lm_correct[left], lm_correct[right] = lm_correct[right].copy(), lm_correct[left].copy()
    ax.imshow(img_flip)
    for i in range(15):
        color = COLORS['central'] if i in [0, 1, 8, 9, 10] else COLORS['gt']
        ax.scatter(lm_correct[i, 0], lm_correct[i, 1], s=60, c=color,
                  edgecolors='black', linewidths=1, zorder=5)
        ax.annotate(f'L{i+1}', (lm_correct[i, 0] + 5, lm_correct[i, 1] - 5),
                   fontsize=7, color='white',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='green', alpha=0.7))
    ax.set_title('Flip CORRECTO\n(con intercambio)', fontsize=12, fontweight='bold', color='green')
    ax.axis('off')

    # Agregar nota sobre pares
    pairs_text = "Pares simetricos intercambiados:\n"
    pairs_text += "L3\u2194L4, L5\u2194L6, L7\u2194L8, L12\u2194L13, L14\u2194L15"
    fig.text(0.5, 0.02, pairs_text, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

    fig.suptitle('Importancia del Intercambio de Pares en Flip Horizontal',
                fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    flip_path = os.path.join(output_path, 'flip_pairs_detail.png')
    plt.savefig(flip_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"  Guardado: {flip_path}")


# ============================================================================
# PARTE 3: PIPELINE DE INFERENCIA
# ============================================================================

def generate_inference_pipeline(output_dir):
    """
    Genera visualizacion del pipeline de inferencia completo.

    Muestra:
    - Imagen original
    - Preprocesamiento
    - 4 modelos prediciendo
    - TTA (original + flip)
    - Ensemble promedio
    - Landmarks finales
    """
    print("\n" + "="*60)
    print("Generando Pipeline de Inferencia...")
    print("="*60)

    output_path = os.path.join(output_dir, 'inference')
    os.makedirs(output_path, exist_ok=True)

    # Crear diagrama de flujo
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Titulo
    ax.text(8, 9.5, 'Pipeline de Inferencia: Ensemble + TTA',
           ha='center', va='center', fontsize=16, fontweight='bold')

    # Definir posiciones
    box_h = 0.8
    box_w = 2.0

    # 1. Imagen de entrada
    draw_step_box(ax, 2, 7.5, box_w, box_h, 'Imagen\nOriginal', COLORS['primary'])

    # 2. Preprocesamiento
    draw_step_box(ax, 5, 7.5, box_w, box_h, 'Preproc.\n(CLAHE)', COLORS['secondary'])
    ax.annotate('', xy=(4, 7.5), xytext=(3, 7.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # 3. Los 4 modelos
    models_y = [6.0, 5.0, 4.0, 3.0]
    for i, y in enumerate(models_y):
        draw_step_box(ax, 8, y, box_w, 0.7, f'Modelo {i+1}\n(seed={[123, 456, 321, 789][i]})',
                     COLORS['accent'])
        # Flecha desde preprocesamiento
        ax.annotate('', xy=(7, y), xytext=(6, 7.5),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray',
                                 connectionstyle='arc3,rad=-0.1'))

    # 4. TTA para cada modelo
    for i, y in enumerate(models_y):
        draw_step_box(ax, 11, y, box_w, 0.7, 'TTA\n(flip)', COLORS['success'])
        ax.annotate('', xy=(10, y), xytext=(9, y),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # 5. Promedio Ensemble
    draw_step_box(ax, 14, 4.5, box_w, box_h, 'Ensemble\nPromedio', '#4CAF50')
    for y in models_y:
        ax.annotate('', xy=(13, 4.5), xytext=(12, y),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray',
                                 connectionstyle='arc3,rad=0'))

    # 6. Salida
    ax.text(14, 2.5, '15 Landmarks\n(x, y) en pixeles', ha='center', va='center',
           fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green'))
    ax.annotate('', xy=(14, 3.1), xytext=(14, 4.1),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'))

    # Anotaciones explicativas
    explanations = [
        (2, 6.5, "299x299 px\nEscala grises"),
        (5, 6.5, "224x224 px\n3 canales RGB"),
        (8, 1.8, "ResNet-18\n+ CoordAttention\n+ DeepHead"),
        (11, 1.8, "Original + Flip\nPromedio"),
        (14, 1.5, "Promedio de\n8 predicciones"),
    ]

    for x, y, text in explanations:
        ax.text(x, y, text, ha='center', va='center', fontsize=8,
               color='gray', style='italic')

    # Guardar
    pipeline_path = os.path.join(output_path, 'inference_pipeline.png')
    plt.savefig(pipeline_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"  Guardado: {pipeline_path}")

    return pipeline_path


# ============================================================================
# PARTE 4: COMPARACION POR CATEGORIA
# ============================================================================

def generate_category_comparison(output_dir):
    """
    Genera comparacion del pipeline por categoria.

    Muestra como CLAHE ayuda especialmente en COVID.
    """
    print("\n" + "="*60)
    print("Generando Comparacion por Categoria...")
    print("="*60)

    output_path = os.path.join(output_dir, 'categories')
    os.makedirs(output_path, exist_ok=True)

    # Obtener imagenes de cada categoria
    samples = get_sample_images()

    categories_info = {
        'Normal': {'color': COLORS['normal'], 'error': '7.00 px'},
        'COVID': {'color': COLORS['covid'], 'error': '9.03 px'},
        'Viral_Pneumonia': {'color': COLORS['viral'], 'error': '7.98 px'}
    }

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    for row, (category, info) in enumerate(categories_info.items()):
        if category not in samples:
            continue

        sample = samples[category]
        img = sample['image']

        # Columna 1: Original
        ax = axes[row, 0]
        ax.imshow(img)
        ax.set_title('Original', fontsize=10, fontweight='bold')
        ax.axis('off')
        if row == 0:
            ax.set_ylabel('Normal', fontsize=12, fontweight='bold', color=info['color'])
        elif row == 1:
            ax.set_ylabel('COVID', fontsize=12, fontweight='bold', color=info['color'])
        else:
            ax.set_ylabel('Viral', fontsize=12, fontweight='bold', color=info['color'])

        # Columna 2: CLAHE
        ax = axes[row, 1]
        img_pil = Image.fromarray(img)
        img_clahe = apply_clahe(img_pil, clip_limit=2.0, tile_grid_size=(4, 4))
        ax.imshow(np.array(img_clahe))
        ax.set_title('Con CLAHE', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Columna 3: Diferencia
        ax = axes[row, 2]
        diff = np.abs(img.astype(float) - np.array(img_clahe).astype(float))
        diff_normalized = diff / diff.max() if diff.max() > 0 else diff
        ax.imshow(diff_normalized)
        ax.set_title('Diferencia\n(areas realzadas)', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Columna 4: Histograma de luminancia
        ax = axes[row, 3]
        # Convertir a LAB y obtener canal L
        lab_orig = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab_clahe = cv2.cvtColor(np.array(img_clahe), cv2.COLOR_RGB2LAB)

        ax.hist(lab_orig[:,:,0].ravel(), bins=50, alpha=0.5, label='Original', color='blue')
        ax.hist(lab_clahe[:,:,0].ravel(), bins=50, alpha=0.5, label='CLAHE', color='orange')
        ax.set_title('Histograma\n(Luminancia)', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 255)

        # Agregar error en la fila
        ax.text(1.1, 0.5, f"Error: {info['error']}", transform=ax.transAxes,
               fontsize=10, fontweight='bold', color=info['color'],
               va='center')

    # Titulo
    fig.suptitle('Efecto de CLAHE por Categoria de Patologia\n' +
                '(COVID muestra mayor mejora en areas con consolidaciones)',
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 0.95, 0.95])

    comparison_path = os.path.join(output_path, 'category_comparison.png')
    plt.savefig(comparison_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"  Guardado: {comparison_path}")

    return comparison_path


# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

def main():
    """Genera todas las visualizaciones del pipeline."""

    output_base = 'outputs/pipeline_viz'
    os.makedirs(output_base, exist_ok=True)

    print("="*60)
    print(" GENERADOR DE VISUALIZACIONES DEL PIPELINE")
    print(" Sesion 17 - Tesis de Maestria")
    print("="*60)

    figures = []

    # 1. Pipeline de Preprocesamiento
    try:
        result = generate_preprocessing_pipeline(output_base)
        if result:
            figures.append(result)
    except Exception as e:
        print(f"Error en preprocesamiento: {e}")
        import traceback
        traceback.print_exc()

    # 2. Pipeline de Augmentation
    try:
        result = generate_augmentation_pipeline(output_base)
        if result:
            figures.append(result)
    except Exception as e:
        print(f"Error en augmentation: {e}")
        import traceback
        traceback.print_exc()

    # 3. Pipeline de Inferencia
    try:
        result = generate_inference_pipeline(output_base)
        if result:
            figures.append(result)
    except Exception as e:
        print(f"Error en inferencia: {e}")
        import traceback
        traceback.print_exc()

    # 4. Comparacion por Categoria
    try:
        result = generate_category_comparison(output_base)
        if result:
            figures.append(result)
    except Exception as e:
        print(f"Error en comparacion: {e}")
        import traceback
        traceback.print_exc()

    # Resumen
    print("\n" + "="*60)
    print(" RESUMEN DE VISUALIZACIONES GENERADAS")
    print("="*60)

    for fig_path in figures:
        print(f"  \u2713 {fig_path}")

    print(f"\nTotal: {len(figures)} visualizaciones")
    print(f"Directorio: {output_base}/")

    return figures


if __name__ == '__main__':
    main()
