#!/usr/bin/env python3
"""
Generar visualizaciones de predicciones del ensemble vs ground truth.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image

from src_v2.data.dataset import create_dataloaders
from src_v2.models.resnet_landmark import create_model


def load_model(checkpoint_path, device):
    model = create_model(
        pretrained=True, freeze_backbone=True, dropout_rate=0.3, hidden_dim=768,
        use_coord_attention=True, deep_head=True, device=device
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict_with_tta(model, image, device):
    """Prediccion con Test-Time Augmentation."""
    model.eval()
    with torch.no_grad():
        pred1 = model(image)
        image_flip = torch.flip(image, dims=[3])
        pred2 = model(image_flip)
        pred2 = pred2.view(-1, 15, 2)
        pred2[:, :, 0] = 1 - pred2[:, :, 0]
        SYMMETRIC_PAIRS = [(2, 3), (4, 5), (6, 7), (11, 12), (13, 14)]
        for left, right in SYMMETRIC_PAIRS:
            pred2[:, [left, right]] = pred2[:, [right, left]]
        pred2 = pred2.view(-1, 30)
        return (pred1 + pred2) / 2


def predict_ensemble(models, image, device):
    """Prediccion del ensemble."""
    preds = []
    for model in models:
        pred = predict_with_tta(model, image, device)
        preds.append(pred)
    return torch.stack(preds).mean(dim=0)


def denormalize_image(tensor):
    """Desnormalizar imagen para visualizacion."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def visualize_sample(image, pred, target, meta, save_path, image_size=224):
    """Visualizar predicciones vs ground truth."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Imagen
    img = denormalize_image(image)
    ax.imshow(img)

    # Convertir a pixeles
    pred_px = pred.cpu().numpy() * image_size
    target_px = target.cpu().numpy() * image_size

    # Colores por grupo
    colors_gt = {
        'axis': '#00FF00',      # L1, L2 - verde
        'central': '#00FFFF',   # L9, L10, L11 - cyan
        'lateral': '#FFFF00',   # L3-L8 - amarillo
        'border': '#FF00FF',    # L12, L13 - magenta
        'costal': '#FF0000',    # L14, L15 - rojo
    }
    colors_pred = {
        'axis': '#008800',
        'central': '#008888',
        'lateral': '#888800',
        'border': '#880088',
        'costal': '#880000',
    }

    groups = {
        0: 'axis', 1: 'axis',           # L1, L2
        2: 'lateral', 3: 'lateral',     # L3, L4
        4: 'lateral', 5: 'lateral',     # L5, L6
        6: 'lateral', 7: 'lateral',     # L7, L8
        8: 'central', 9: 'central', 10: 'central',  # L9, L10, L11
        11: 'border', 12: 'border',     # L12, L13
        13: 'costal', 14: 'costal',     # L14, L15
    }

    # Dibujar landmarks
    for i in range(15):
        group = groups[i]

        # Ground truth (circulos grandes)
        ax.scatter(target_px[i, 0], target_px[i, 1],
                   c=colors_gt[group], s=120, marker='o',
                   edgecolors='white', linewidths=1.5, alpha=0.9)

        # Prediccion (X pequena)
        ax.scatter(pred_px[i, 0], pred_px[i, 1],
                   c=colors_pred[group], s=80, marker='x',
                   linewidths=2, alpha=0.9)

        # Linea conectando
        ax.plot([target_px[i, 0], pred_px[i, 0]],
                [target_px[i, 1], pred_px[i, 1]],
                'w-', linewidth=0.5, alpha=0.5)

        # Etiqueta
        ax.annotate(f'L{i+1}', (target_px[i, 0]+3, target_px[i, 1]-3),
                    fontsize=7, color='white', weight='bold')

    # Error por muestra
    errors = np.sqrt(np.sum((pred_px - target_px)**2, axis=1))
    mean_error = errors.mean()

    # Titulo
    title = f"{meta['category']} - {meta['image_name']}\n"
    title += f"Mean Error: {mean_error:.2f} px | "
    title += f"Min: {errors.min():.1f} | Max: {errors.max():.1f} px"
    ax.set_title(title, fontsize=12, color='white', pad=10)

    ax.axis('off')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Leyenda
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#00FF00',
                   markersize=10, label='GT - Axis (L1,L2)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#00FFFF',
                   markersize=10, label='GT - Central (L9-L11)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFFF00',
                   markersize=10, label='GT - Lateral (L3-L8)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF0000',
                   markersize=10, label='GT - Costal (L14,L15)'),
        plt.Line2D([0], [0], marker='x', color='#008800', markersize=10,
                   markeredgewidth=2, linestyle='None', label='Prediction'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
              facecolor='black', edgecolor='white', labelcolor='white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
    plt.close()

    return mean_error


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Cargar modelos
    print("Loading ensemble models...")
    model_paths = [
        'checkpoints/session10/exp4_epochs100/final_model.pt',
        'checkpoints/session10/ensemble/seed123/final_model.pt',
        'checkpoints/session10/ensemble/seed456/final_model.pt',
    ]
    models = [load_model(PROJECT_ROOT / p, device) for p in model_paths]

    # Cargar datos
    _, _, test_loader = create_dataloaders(
        csv_path=str(PROJECT_ROOT / 'data/coordenadas/coordenadas_maestro.csv'),
        data_root=str(PROJECT_ROOT / 'data/'),
        batch_size=1, num_workers=0, random_state=42,
        use_clahe=True, clahe_clip_limit=2.0, clahe_tile_size=4,
    )

    # Crear directorio de salida
    output_dir = PROJECT_ROOT / 'outputs' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generar visualizaciones
    print(f"\nGenerating visualizations for {min(20, len(test_loader))} samples...")

    all_errors = []
    for i, batch in enumerate(test_loader):
        if i >= 20:
            break

        image = batch[0].to(device)
        target = batch[1][0].view(15, 2)
        meta = batch[2][0]

        # Prediccion del ensemble
        pred = predict_ensemble(models, image, device)
        pred = pred[0].view(15, 2)

        # Visualizar
        save_path = output_dir / f'sample_{i:02d}_{meta["category"]}_{meta["image_name"]}.png'
        error = visualize_sample(image[0], pred, target, meta, save_path)
        all_errors.append(error)

        print(f"  [{i+1:2d}] {meta['category']:<20} Error: {error:.2f} px")

    # Resumen
    print("\n" + "="*50)
    print("RESUMEN DE VISUALIZACIONES")
    print("="*50)
    print(f"Samples visualized: {len(all_errors)}")
    print(f"Mean error: {np.mean(all_errors):.2f} px")
    print(f"Std error:  {np.std(all_errors):.2f} px")
    print(f"Min error:  {np.min(all_errors):.2f} px")
    print(f"Max error:  {np.max(all_errors):.2f} px")
    print(f"\nOutput directory: {output_dir}")

    # Crear grid de mejores y peores
    print("\nCreating summary grid...")
    errors_sorted = sorted(enumerate(all_errors), key=lambda x: x[1])

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Ensemble Predictions: Best (top) and Worst (bottom) 5 samples',
                 fontsize=14, color='white')

    # Los mejores 5
    for j, (idx, err) in enumerate(errors_sorted[:5]):
        path = list(output_dir.glob(f'sample_{idx:02d}_*.png'))[0]
        img = Image.open(path)
        axes[0, j].imshow(img)
        axes[0, j].set_title(f'Error: {err:.2f} px', color='white', fontsize=10)
        axes[0, j].axis('off')

    # Los peores 5
    for j, (idx, err) in enumerate(errors_sorted[-5:]):
        path = list(output_dir.glob(f'sample_{idx:02d}_*.png'))[0]
        img = Image.open(path)
        axes[1, j].imshow(img)
        axes[1, j].set_title(f'Error: {err:.2f} px', color='white', fontsize=10)
        axes[1, j].axis('off')

    fig.patch.set_facecolor('black')
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor('black')

    plt.tight_layout()
    summary_path = output_dir / 'summary_best_worst.png'
    plt.savefig(summary_path, dpi=150, facecolor='black', bbox_inches='tight')
    plt.close()
    print(f"Summary saved: {summary_path}")


if __name__ == '__main__':
    main()
