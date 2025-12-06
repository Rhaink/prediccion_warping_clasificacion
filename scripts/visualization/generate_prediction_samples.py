#!/usr/bin/env python3
"""
Generador de ejemplos de predicciones para tesis.
Sesión 15: Documentación final y visualizaciones.

Genera:
1. Ejemplos de predicciones vs Ground Truth (por categoría)
2. Visualización del efecto de CLAHE (antes/después)
3. Casos buenos y casos difíciles
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

# Agregar src_v2 al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src_v2.data.dataset import LandmarkDataset, create_dataloaders
from src_v2.data.transforms import apply_clahe

# Configuración
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Colores para landmarks
LANDMARK_COLORS = {
    'gt': '#00FF00',      # Verde - Ground Truth
    'pred': '#FF0000',    # Rojo - Predicción
    'central': '#FFFF00', # Amarillo - Landmarks centrales
    'axis': '#00FFFF',    # Cyan - Eje L1-L2
}

LANDMARK_NAMES = [
    'L1-Superior', 'L2-Inferior', 'L3-Apex Izq', 'L4-Apex Der',
    'L5-Hilio Izq', 'L6-Hilio Der', 'L7-Base Izq', 'L8-Base Der',
    'L9-Centro Sup', 'L10-Centro Med', 'L11-Centro Inf',
    'L12-Borde Izq', 'L13-Borde Der', 'L14-Costof Izq', 'L15-Costof Der'
]


def load_ensemble_models(device='cuda'):
    """Carga los 4 modelos del ensemble."""
    from src_v2.models.resnet_landmark import ResNet18Landmarks

    model_paths = [
        'checkpoints/session10/ensemble/seed123/final_model.pt',
        'checkpoints/session10/ensemble/seed456/final_model.pt',
        'checkpoints/session13/seed321/final_model.pt',
        'checkpoints/session13/seed789/final_model.pt',
    ]

    models = []
    for path in model_paths:
        if os.path.exists(path):
            model = ResNet18Landmarks(
                num_landmarks=15,
                pretrained=False,
                use_coord_attention=True,
                deep_head=True,
                hidden_dim=768,
                dropout_rate=0.3
            )
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            models.append(model)
            print(f"  ✓ Cargado: {path}")
        else:
            print(f"  ✗ No encontrado: {path}")

    return models


def predict_with_ensemble(models, image_tensor, device='cuda'):
    """Realiza predicción con ensemble y TTA."""
    image_tensor = image_tensor.unsqueeze(0).to(device)

    predictions = []
    for model in models:
        with torch.no_grad():
            # Predicción original
            pred = model(image_tensor)
            predictions.append(pred)

            # Predicción con flip (TTA)
            flipped = torch.flip(image_tensor, dims=[3])
            pred_flip = model(flipped)

            # Corregir coordenadas del flip
            pred_flip = pred_flip.view(-1, 15, 2)
            pred_flip[:, :, 0] = 1 - pred_flip[:, :, 0]

            # Intercambiar pares simétricos
            pairs = [(2, 3), (4, 5), (6, 7), (11, 12), (13, 14)]
            for l, r in pairs:
                pred_flip[:, [l, r]] = pred_flip[:, [r, l]]

            pred_flip = pred_flip.view(-1, 30)
            predictions.append(pred_flip)

    # Promediar todas las predicciones
    avg_pred = torch.stack(predictions).mean(dim=0)
    return avg_pred.view(15, 2).cpu().numpy()


def draw_landmarks_on_image(image, gt_landmarks, pred_landmarks=None, title=""):
    """Dibuja landmarks en una imagen."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Mostrar imagen
    ax.imshow(image)

    # Dibujar eje L1-L2
    if gt_landmarks is not None:
        l1, l2 = gt_landmarks[0], gt_landmarks[1]
        ax.plot([l1[0], l2[0]], [l1[1], l2[1]], '-', color=LANDMARK_COLORS['axis'],
                linewidth=1.5, alpha=0.7, label='Eje central')

    # Dibujar Ground Truth
    if gt_landmarks is not None:
        for i, (x, y) in enumerate(gt_landmarks):
            color = LANDMARK_COLORS['central'] if i in [0, 1, 8, 9, 10] else LANDMARK_COLORS['gt']
            ax.scatter(x, y, s=80, c=color, marker='o', edgecolors='black', linewidths=1)
            ax.text(x + 5, y - 5, f'{i+1}', fontsize=7, color='white',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='green', alpha=0.7))

    # Dibujar Predicciones
    if pred_landmarks is not None:
        for i, (x, y) in enumerate(pred_landmarks):
            ax.scatter(x, y, s=60, c=LANDMARK_COLORS['pred'], marker='x', linewidths=2)

        # Líneas de error
        for i in range(len(gt_landmarks)):
            gx, gy = gt_landmarks[i]
            px, py = pred_landmarks[i]
            ax.plot([gx, px], [gy, py], '-', color=LANDMARK_COLORS['pred'],
                   linewidth=0.5, alpha=0.5)

    # Leyenda
    if pred_landmarks is not None:
        ax.scatter([], [], s=80, c=LANDMARK_COLORS['gt'], marker='o', label='Ground Truth')
        ax.scatter([], [], s=60, c=LANDMARK_COLORS['pred'], marker='x', label='Predicción')
        ax.legend(loc='upper right', fontsize=9)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')

    return fig


def generate_clahe_comparison(output_dir):
    """Genera visualización del efecto de CLAHE."""
    print("\nGenerando visualización de CLAHE...")

    # Cargar una imagen de cada categoría
    categories = ['COVID', 'Normal', 'Viral_Pneumonia']

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for row, category in enumerate(categories):
        # Buscar una imagen
        img_dir = f'data/{category}'
        if os.path.exists(img_dir):
            images = [f for f in os.listdir(img_dir) if f.endswith('.png')]
            if images:
                img_path = os.path.join(img_dir, images[5])  # Tomar la 6ta imagen
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Sin CLAHE
                axes[row, 0].imshow(image)
                axes[row, 0].set_title(f'{category}\nOriginal', fontsize=10)
                axes[row, 0].axis('off')

                # Con CLAHE tile=8 (original)
                clahe_8 = apply_clahe(image, clip_limit=2.0, tile_size=8)
                axes[row, 1].imshow(clahe_8)
                axes[row, 1].set_title(f'CLAHE\ntile=8', fontsize=10)
                axes[row, 1].axis('off')

                # Con CLAHE tile=4 (óptimo)
                clahe_4 = apply_clahe(image, clip_limit=2.0, tile_size=4)
                axes[row, 2].imshow(clahe_4)
                axes[row, 2].set_title(f'CLAHE\ntile=4 (óptimo)', fontsize=10)
                axes[row, 2].axis('off')

    plt.suptitle('Efecto de CLAHE en Radiografías por Categoría',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'clahe_comparison.png')
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Comparación CLAHE guardada: {output_path}")
    return output_path


def generate_prediction_examples(output_dir, num_samples=4, use_models=False):
    """Genera ejemplos de predicciones por categoría."""
    print("\nGenerando ejemplos de predicciones...")

    # Cargar dataset
    _, _, test_loader = create_dataloaders(
        'data/coordenadas/coordenadas_maestro.csv',
        'data',
        batch_size=1,
        use_clahe=True,
        clahe_clip_limit=2.0,
        clahe_tile_size=4
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Cargar modelos si se especifica
    models = None
    if use_models:
        print("\nCargando modelos del ensemble...")
        models = load_ensemble_models(device)
        if not models:
            print("No se pudieron cargar los modelos, usando solo GT")
            models = None

    # Agrupar por categoría
    samples_by_category = {'COVID': [], 'Normal': [], 'Viral_Pneumonia': []}

    for batch in test_loader:
        image, landmarks, meta = batch
        category = meta[0]['category']

        if len(samples_by_category.get(category, [])) < num_samples:
            samples_by_category.setdefault(category, []).append({
                'image': image[0],
                'landmarks': landmarks[0],
                'filename': meta[0]['image_name']
            })

        # Verificar si tenemos suficientes muestras
        if all(len(v) >= num_samples for v in samples_by_category.values()):
            break

    # Generar figura con ejemplos
    fig, axes = plt.subplots(3, num_samples, figsize=(4*num_samples, 12))

    for row, (category, samples) in enumerate(samples_by_category.items()):
        for col, sample in enumerate(samples[:num_samples]):
            ax = axes[row, col]

            # Convertir tensor a imagen
            img = sample['image'].permute(1, 2, 0).numpy()
            # Denormalizar
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)

            # Ground Truth en píxeles
            gt = sample['landmarks'].view(15, 2).numpy() * 224

            # Predicciones si hay modelos
            pred = None
            if models:
                pred = predict_with_ensemble(models, sample['image'], device) * 224

            ax.imshow(img)

            # Dibujar landmarks
            for i, (x, y) in enumerate(gt):
                color = 'yellow' if i in [0, 1, 8, 9, 10] else 'lime'
                ax.scatter(x, y, s=50, c=color, marker='o', edgecolors='black', linewidths=0.5)

            if pred is not None:
                for i, (x, y) in enumerate(pred):
                    ax.scatter(x, y, s=30, c='red', marker='x', linewidths=1.5)
                    # Línea de error
                    ax.plot([gt[i, 0], x], [gt[i, 1], y], 'r-', linewidth=0.3, alpha=0.5)

                # Calcular error
                error = np.sqrt(((gt - pred) ** 2).sum(axis=1)).mean()
                ax.set_title(f'{sample["filename"][:15]}...\nError: {error:.2f} px', fontsize=8)
            else:
                ax.set_title(f'{sample["filename"][:15]}...', fontsize=8)

            ax.axis('off')

            # Etiqueta de categoría en primera columna
            if col == 0:
                ax.set_ylabel(category, fontsize=12, fontweight='bold')

    # Leyenda
    fig.text(0.5, 0.02, '○ Ground Truth  × Predicción  (Amarillo: landmarks centrales)',
            ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    plt.suptitle('Ejemplos de Predicciones por Categoría\n(Ensemble 4 Modelos + TTA)',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    output_path = os.path.join(output_dir, 'prediction_examples.png')
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Ejemplos de predicciones guardados: {output_path}")
    return output_path


def generate_best_worst_cases(output_dir, num_each=3, use_models=False):
    """Genera visualización de mejores y peores casos."""
    print("\nGenerando casos mejores/peores...")

    # Cargar dataset
    _, _, test_loader = create_dataloaders(
        'data/coordenadas/coordenadas_maestro.csv',
        'data',
        batch_size=1,
        use_clahe=True,
        clahe_clip_limit=2.0,
        clahe_tile_size=4
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Cargar modelos
    models = None
    if use_models:
        models = load_ensemble_models(device)
        if not models:
            print("No se pudieron cargar los modelos, generando solo GT")
            # Generar visualización simple sin predicciones
            return None

    # Calcular errores para todas las muestras
    all_samples = []
    for batch in test_loader:
        image, landmarks, meta = batch
        gt = landmarks[0].view(15, 2).numpy() * 224

        if models:
            pred = predict_with_ensemble(models, image[0], device) * 224
            error = np.sqrt(((gt - pred) ** 2).sum(axis=1)).mean()
        else:
            pred = None
            error = 0

        all_samples.append({
            'image': image[0],
            'landmarks': landmarks[0],
            'filename': meta[0]['image_name'],
            'category': meta[0]['category'],
            'error': error,
            'pred': pred,
            'gt': gt
        })

    if not models:
        print("Sin modelos cargados, omitiendo casos mejores/peores")
        return None

    # Ordenar por error
    all_samples.sort(key=lambda x: x['error'])
    best = all_samples[:num_each]
    worst = all_samples[-num_each:]

    # Crear figura
    fig, axes = plt.subplots(2, num_each, figsize=(4*num_each, 8))

    for col, sample in enumerate(best):
        ax = axes[0, col]
        img = sample['image'].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)

        ax.imshow(img)
        for i in range(15):
            ax.scatter(sample['gt'][i, 0], sample['gt'][i, 1], s=40, c='lime',
                      marker='o', edgecolors='black', linewidths=0.5)
            ax.scatter(sample['pred'][i, 0], sample['pred'][i, 1], s=25, c='red',
                      marker='x', linewidths=1)

        ax.set_title(f'{sample["category"]}\nError: {sample["error"]:.2f} px', fontsize=9)
        ax.axis('off')
        if col == 0:
            ax.text(-20, 112, 'MEJORES\nCASOS', fontsize=11, fontweight='bold',
                   rotation=90, va='center', color='green')

    for col, sample in enumerate(worst):
        ax = axes[1, col]
        img = sample['image'].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)

        ax.imshow(img)
        for i in range(15):
            ax.scatter(sample['gt'][i, 0], sample['gt'][i, 1], s=40, c='lime',
                      marker='o', edgecolors='black', linewidths=0.5)
            ax.scatter(sample['pred'][i, 0], sample['pred'][i, 1], s=25, c='red',
                      marker='x', linewidths=1)

        ax.set_title(f'{sample["category"]}\nError: {sample["error"]:.2f} px', fontsize=9)
        ax.axis('off')
        if col == 0:
            ax.text(-20, 112, 'PEORES\nCASOS', fontsize=11, fontweight='bold',
                   rotation=90, va='center', color='red')

    plt.suptitle('Mejores y Peores Casos del Ensemble', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'best_worst_cases.png')
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Casos mejores/peores guardados: {output_path}")
    return output_path


def main():
    """Genera todas las visualizaciones de predicciones."""
    output_dir = 'outputs/thesis_figures'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Generando visualizaciones de predicciones para tesis")
    print("=" * 60)

    figures = []

    # 1. Comparación CLAHE (no necesita modelos)
    figures.append(generate_clahe_comparison(output_dir))

    # 2. Ejemplos de predicciones (con modelos si están disponibles)
    try:
        result = generate_prediction_examples(output_dir, num_samples=4, use_models=True)
        if result:
            figures.append(result)
    except Exception as e:
        print(f"Error generando ejemplos de predicciones: {e}")
        # Intentar sin modelos
        try:
            result = generate_prediction_examples(output_dir, num_samples=4, use_models=False)
            if result:
                figures.append(result)
        except Exception as e2:
            print(f"Error: {e2}")

    # 3. Mejores/peores casos (requiere modelos)
    try:
        result = generate_best_worst_cases(output_dir, num_each=3, use_models=True)
        if result:
            figures.append(result)
    except Exception as e:
        print(f"Nota: No se pudieron generar casos mejores/peores: {e}")

    print("\n" + "=" * 60)
    print("Visualizaciones generadas:")
    print("=" * 60)
    for f in figures:
        if f:
            print(f"  • {f}")

    print(f"\nTotal: {len([f for f in figures if f])} figuras")


if __name__ == '__main__':
    main()
