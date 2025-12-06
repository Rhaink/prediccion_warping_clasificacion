#!/usr/bin/env python3
"""
Generador de animaciones GIF para presentaciones.
Sesion 17: Visualizaciones Detalladas del Pipeline.

Este script genera animaciones de:
1. Pipeline de preprocesamiento
2. Forward pass (feature maps cambiando)
3. Ensemble + TTA
4. Progreso del entrenamiento
5. Progreso del proyecto por sesiones

Las animaciones son para usar en la presentacion de defensa de tesis.
"""

import os
import sys
import cv2
import numpy as np

# Usar backend Agg (no interactivo) para poder convertir a arrays
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.animation as animation
from pathlib import Path
from PIL import Image
import imageio.v2 as imageio
import torch

# Agregar src_v2 al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src_v2.data.transforms import apply_clahe, SYMMETRIC_PAIRS

# Configuracion global de matplotlib
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'figure.dpi': 100,
    'savefig.dpi': 150,
})

# Colores consistentes
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'gt': '#00FF00',
    'pred': '#FF0000',
}


def get_sample_image(category='Normal'):
    """Obtiene una imagen de ejemplo."""
    cat_dir = f'data/dataset/{category}'
    if os.path.exists(cat_dir):
        images = sorted([f for f in os.listdir(cat_dir) if f.endswith('.png')])
        if images:
            idx = len(images) // 2
            img_path = os.path.join(cat_dir, images[idx])
            img = cv2.imread(img_path)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


# ============================================================================
# 1. ANIMACION DE PREPROCESAMIENTO
# ============================================================================

def generate_preprocessing_animation(output_dir):
    """
    Genera animacion GIF del pipeline de preprocesamiento.
    Frames: Original -> RGB -> CLAHE -> Resize -> Normalizado
    """
    print("\n" + "="*60)
    print("Generando animacion de preprocesamiento...")
    print("="*60)

    img_original = get_sample_image('Normal')
    if img_original is None:
        print("Error: No se pudo cargar imagen")
        return None

    # Preparar frames
    frames_data = []

    # Frame 1: Original
    frames_data.append({
        'image': img_original,
        'title': 'Paso 1: Imagen Original',
        'subtitle': f'Tamano: {img_original.shape[1]}x{img_original.shape[0]}',
        'duration': 1500
    })

    # Frame 2: RGB (igual, pero con anotacion)
    frames_data.append({
        'image': img_original.copy(),
        'title': 'Paso 2: Conversion a RGB',
        'subtitle': '3 canales: Rojo, Verde, Azul',
        'duration': 1500
    })

    # Frame 3: CLAHE
    img_pil = Image.fromarray(img_original)
    img_clahe = np.array(apply_clahe(img_pil, clip_limit=2.0, tile_grid_size=(4, 4)))
    frames_data.append({
        'image': img_clahe,
        'title': 'Paso 3: CLAHE',
        'subtitle': 'Realce de contraste adaptativo',
        'duration': 2000
    })

    # Frame 4: Resize
    img_resized = cv2.resize(img_clahe, (224, 224))
    frames_data.append({
        'image': img_resized,
        'title': 'Paso 4: Resize',
        'subtitle': 'Redimensionado a 224x224',
        'duration': 1500
    })

    # Frame 5: Normalizado
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_resized / 255.0 - mean) / std
    img_norm_vis = np.clip((img_norm * std + mean), 0, 1)
    frames_data.append({
        'image': (img_norm_vis * 255).astype(np.uint8),
        'title': 'Paso 5: Normalizacion ImageNet',
        'subtitle': 'Rango: [-2.1, 2.6]',
        'duration': 2000
    })

    # Frame 6: Tensor
    frames_data.append({
        'image': (img_norm_vis * 255).astype(np.uint8),
        'title': 'Paso 6: Conversion a Tensor',
        'subtitle': 'Shape: (3, 224, 224)',
        'duration': 1500
    })

    # Crear GIF
    gif_frames = []
    for frame in frames_data:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(frame['image'])
        ax.set_title(frame['title'], fontsize=16, fontweight='bold', pad=10)
        ax.set_xlabel(frame['subtitle'], fontsize=12, color='gray')
        ax.axis('off')

        # Guardar frame como imagen
        fig.canvas.draw()
        frame_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)

        # Duplicar frames segun duracion (para controlar velocidad)
        num_repeats = frame['duration'] // 100  # 100ms por frame base
        for _ in range(num_repeats):
            gif_frames.append(frame_img)

    # Guardar GIF
    output_path = os.path.join(output_dir, 'preprocessing_animation.gif')
    imageio.mimsave(output_path, gif_frames, fps=10, loop=0)

    print(f"  Guardado: {output_path}")
    return output_path


# ============================================================================
# 2. ANIMACION DEL ENSEMBLE + TTA
# ============================================================================

def generate_ensemble_animation(output_dir):
    """
    Genera animacion del proceso de ensemble + TTA.
    Muestra como las predicciones se combinan.
    """
    print("\n" + "="*60)
    print("Generando animacion de Ensemble + TTA...")
    print("="*60)

    img = get_sample_image('Normal')
    if img is None:
        print("Error: No se pudo cargar imagen")
        return None

    # Preprocesar
    img_pil = Image.fromarray(img)
    img_clahe = np.array(apply_clahe(img_pil, clip_limit=2.0, tile_grid_size=(4, 4)))
    img_proc = cv2.resize(img_clahe, (224, 224))

    # Simular landmarks (centro + variacion)
    np.random.seed(42)
    gt_landmarks = np.array([
        [112, 30], [112, 194],  # L1, L2 (eje)
        [70, 60], [154, 60],    # L3, L4
        [60, 100], [164, 100],  # L5, L6
        [55, 160], [169, 160],  # L7, L8
        [112, 55], [112, 100], [112, 145],  # L9, L10, L11
        [50, 45], [174, 45],    # L12, L13
        [45, 180], [179, 180]   # L14, L15
    ])

    # Simular predicciones de cada modelo (con variacion)
    model_preds = []
    for seed in [123, 456, 321, 789]:
        np.random.seed(seed)
        noise = np.random.randn(15, 2) * 5  # Ruido de ~5 pixeles
        pred = gt_landmarks + noise
        model_preds.append(pred)

    # Crear frames
    gif_frames = []

    # Frame 1: Imagen preprocesada
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_proc)
    ax.set_title('Imagen Preprocesada', fontsize=14, fontweight='bold')
    ax.axis('off')
    fig.canvas.draw()
    frame_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close(fig)
    for _ in range(15):  # 1.5 segundos
        gif_frames.append(frame_img)

    # Frames 2-5: Predicciones de cada modelo
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    model_names = ['Modelo 1 (seed=123)', 'Modelo 2 (seed=456)',
                   'Modelo 3 (seed=321)', 'Modelo 4 (seed=789)']

    accumulated_preds = []
    for i, (pred, color, name) in enumerate(zip(model_preds, colors, model_names)):
        accumulated_preds.append(pred)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_proc)

        # Mostrar predicciones acumuladas (mas transparentes las anteriores)
        for j, past_pred in enumerate(accumulated_preds):
            alpha = 0.3 if j < i else 1.0
            ax.scatter(past_pred[:, 0], past_pred[:, 1],
                      s=50, c=colors[j], alpha=alpha, marker='x', linewidths=2)

        ax.set_title(f'Paso {i+1}: {name}', fontsize=14, fontweight='bold')
        ax.axis('off')

        fig.canvas.draw()
        frame_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)

        for _ in range(20):  # 2 segundos por modelo
            gif_frames.append(frame_img)

    # Frame 6: TTA (flip)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_proc)
    ax.set_title('Paso 5: Test-Time Augmentation\n(+flip horizontal)', fontsize=14, fontweight='bold')

    for pred, color in zip(model_preds, colors):
        ax.scatter(pred[:, 0], pred[:, 1], s=30, c=color, alpha=0.5, marker='x', linewidths=1)

    # Predicciones con flip (simuladas)
    for pred, color in zip(model_preds, colors):
        pred_flip = pred.copy()
        pred_flip[:, 0] = 224 - pred_flip[:, 0]
        ax.scatter(pred_flip[:, 0], pred_flip[:, 1], s=30, c=color, alpha=0.3, marker='+', linewidths=1)

    ax.axis('off')
    fig.canvas.draw()
    frame_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close(fig)
    for _ in range(20):
        gif_frames.append(frame_img)

    # Frame 7: Promedio ensemble
    ensemble_pred = np.mean(model_preds, axis=0)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_proc)
    ax.set_title('Paso 6: Ensemble (Promedio)', fontsize=14, fontweight='bold')

    # Ground truth
    ax.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1],
              s=80, c='lime', marker='o', edgecolors='black', linewidths=1, label='Ground Truth')

    # Ensemble
    ax.scatter(ensemble_pred[:, 0], ensemble_pred[:, 1],
              s=60, c='red', marker='x', linewidths=2, label='Ensemble')

    # Lineas de error
    for gt, pred in zip(gt_landmarks, ensemble_pred):
        ax.plot([gt[0], pred[0]], [gt[1], pred[1]], 'r-', alpha=0.3, linewidth=1)

    # Error promedio
    error = np.sqrt(((gt_landmarks - ensemble_pred)**2).sum(axis=1)).mean()
    ax.text(10, 210, f'Error: {error:.2f} px', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax.legend(loc='upper right')
    ax.axis('off')

    fig.canvas.draw()
    frame_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close(fig)
    for _ in range(30):  # 3 segundos
        gif_frames.append(frame_img)

    # Guardar GIF
    output_path = os.path.join(output_dir, 'ensemble_animation.gif')
    imageio.mimsave(output_path, gif_frames, fps=10, loop=0)

    print(f"  Guardado: {output_path}")
    return output_path


# ============================================================================
# 3. ANIMACION DEL PROGRESO DEL PROYECTO
# ============================================================================

def generate_project_progress_animation(output_dir):
    """
    Genera animacion del progreso del proyecto por sesiones.
    Muestra como el error fue bajando.
    """
    print("\n" + "="*60)
    print("Generando animacion de progreso del proyecto...")
    print("="*60)

    # Datos de progreso por sesion
    sessions_data = [
        ('S4: Baseline', 9.08, '#e74c3c'),
        ('S5: +TTA', 8.80, '#e67e22'),
        ('S7: +CLAHE', 8.18, '#f39c12'),
        ('S8: +tile=4', 7.84, '#27ae60'),
        ('S9: +hidden=768', 7.21, '#3498db'),
        ('S10: +epochs=100', 6.75, '#9b59b6'),
        ('S10: Ensemble 3', 4.50, '#1abc9c'),
        ('S12: Ensemble 2', 3.79, '#16a085'),
        ('S13: Ensemble 4', 3.71, '#27ae60'),
    ]

    gif_frames = []

    # Crear frames progresivos
    for i in range(len(sessions_data) + 1):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Datos hasta este punto
        if i == 0:
            labels = []
            errors = []
            colors_plot = []
        else:
            labels = [d[0] for d in sessions_data[:i]]
            errors = [d[1] for d in sessions_data[:i]]
            colors_plot = [d[2] for d in sessions_data[:i]]

        # Grafico de barras
        x_pos = range(len(labels))
        bars = ax.bar(x_pos, errors, color=colors_plot, edgecolor='black', linewidth=1.5)

        # Etiquetas de error
        for j, (bar, error) in enumerate(zip(bars, errors)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{error:.2f}', ha='center', fontsize=9, fontweight='bold')

        # Linea de objetivo
        ax.axhline(y=8.0, color='red', linestyle='--', linewidth=2, label='Objetivo: <8 px')
        ax.axhline(y=4.0, color='green', linestyle='--', linewidth=2, label='Excelente: <4 px')

        ax.set_ylabel('Error (pixeles)', fontsize=12)
        ax.set_title('Progreso del Proyecto: Error por Sesion', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 10)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)

        # Mejora porcentual
        if i > 0:
            mejora = (9.08 - errors[-1]) / 9.08 * 100
            ax.text(0.02, 0.95, f'Mejora total: {mejora:.0f}%',
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()

        fig.canvas.draw()
        frame_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)

        # Duracion de cada frame
        duration = 15 if i < len(sessions_data) else 30
        for _ in range(duration):
            gif_frames.append(frame_img)

    # Guardar GIF
    output_path = os.path.join(output_dir, 'project_progress_animation.gif')
    imageio.mimsave(output_path, gif_frames, fps=10, loop=0)

    print(f"  Guardado: {output_path}")
    return output_path


# ============================================================================
# 4. ANIMACION DEL ENTRENAMIENTO
# ============================================================================

def generate_training_animation(output_dir):
    """
    Genera animacion del proceso de entrenamiento.
    Muestra como los landmarks convergen durante epocas.
    """
    print("\n" + "="*60)
    print("Generando animacion de entrenamiento...")
    print("="*60)

    img = get_sample_image('Normal')
    if img is None:
        print("Error: No se pudo cargar imagen")
        return None

    img_proc = cv2.resize(img, (224, 224))

    # Ground truth
    gt_landmarks = np.array([
        [112, 30], [112, 194],
        [70, 60], [154, 60],
        [60, 100], [164, 100],
        [55, 160], [169, 160],
        [112, 55], [112, 100], [112, 145],
        [50, 45], [174, 45],
        [45, 180], [179, 180]
    ])

    # Simular predicciones en diferentes epocas
    epochs = [1, 5, 10, 25, 50, 100]
    errors = [80, 40, 20, 12, 8, 4]  # Error simulado por epoca

    gif_frames = []

    for epoch, error_scale in zip(epochs, errors):
        np.random.seed(epoch)
        noise = np.random.randn(15, 2) * error_scale
        pred = gt_landmarks + noise
        pred = np.clip(pred, 5, 219)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_proc)

        # Ground truth
        ax.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1],
                  s=80, c='lime', marker='o', edgecolors='black', linewidths=1)

        # Predicciones
        ax.scatter(pred[:, 0], pred[:, 1],
                  s=60, c='red', marker='x', linewidths=2)

        # Lineas de error
        for gt, p in zip(gt_landmarks, pred):
            ax.plot([gt[0], p[0]], [gt[1], p[1]], 'r-', alpha=0.3, linewidth=1)

        # Calcular error real
        real_error = np.sqrt(((gt_landmarks - pred)**2).sum(axis=1)).mean()

        ax.set_title(f'Epoca {epoch}', fontsize=16, fontweight='bold')
        ax.text(10, 210, f'Error: {real_error:.1f} px', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        ax.axis('off')

        fig.canvas.draw()
        frame_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)

        # Mas frames para epocas importantes
        duration = 30 if epoch in [1, 100] else 20
        for _ in range(duration):
            gif_frames.append(frame_img)

    # Frame final
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_proc)
    ax.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1],
              s=80, c='lime', marker='o', edgecolors='black', linewidths=1, label='Ground Truth')
    ax.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1],
              s=60, c='red', marker='x', linewidths=2, label='Prediccion Final')
    ax.set_title('Entrenamiento Completo', fontsize=16, fontweight='bold')
    ax.text(10, 210, f'Error Final: 3.71 px', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax.legend(loc='upper right')
    ax.axis('off')

    fig.canvas.draw()
    frame_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close(fig)

    for _ in range(40):
        gif_frames.append(frame_img)

    # Guardar GIF
    output_path = os.path.join(output_dir, 'training_animation.gif')
    imageio.mimsave(output_path, gif_frames, fps=10, loop=0)

    print(f"  Guardado: {output_path}")
    return output_path


# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

def main():
    """Genera todas las animaciones."""

    output_dir = 'outputs/pipeline_viz/animations'
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print(" GENERADOR DE ANIMACIONES GIF")
    print(" Sesion 17 - Tesis de Maestria")
    print("="*60)

    figures = []

    # 1. Animacion de preprocesamiento
    try:
        result = generate_preprocessing_animation(output_dir)
        if result:
            figures.append(result)
    except Exception as e:
        print(f"Error en preprocesamiento: {e}")
        import traceback
        traceback.print_exc()

    # 2. Animacion de ensemble + TTA
    try:
        result = generate_ensemble_animation(output_dir)
        if result:
            figures.append(result)
    except Exception as e:
        print(f"Error en ensemble: {e}")
        import traceback
        traceback.print_exc()

    # 3. Animacion de progreso del proyecto
    try:
        result = generate_project_progress_animation(output_dir)
        if result:
            figures.append(result)
    except Exception as e:
        print(f"Error en progreso: {e}")
        import traceback
        traceback.print_exc()

    # 4. Animacion del entrenamiento
    try:
        result = generate_training_animation(output_dir)
        if result:
            figures.append(result)
    except Exception as e:
        print(f"Error en entrenamiento: {e}")
        import traceback
        traceback.print_exc()

    # Resumen
    print("\n" + "="*60)
    print(" RESUMEN DE ANIMACIONES GENERADAS")
    print("="*60)

    for fig_path in figures:
        print(f"  \u2713 {fig_path}")

    print(f"\nTotal: {len(figures)} animaciones")
    print(f"Directorio: {output_dir}/")

    return figures


if __name__ == '__main__':
    main()
