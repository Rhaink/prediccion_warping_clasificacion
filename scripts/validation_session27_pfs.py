#!/usr/bin/env python3
"""
Sesión 27: Validación PFS con modelo de 98.02% accuracy

Compara Pulmonary Focus Score entre:
- Modelo Session 27 (98.02% accuracy, 15K imágenes)
- Modelo Original (sin warping)
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import json
from collections import defaultdict
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class GradCAM:
    """Implementación de Grad-CAM para ResNet-18."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, target_class=None):
        """Genera mapa de calor Grad-CAM."""
        self.model.eval()

        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        self.model.zero_grad()
        output[0, target_class].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, target_class, confidence


def load_model_session27(model_path, device):
    """Carga modelo Session 27 (98.02% accuracy)."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 3)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint['class_names']


def load_model_original(model_path, device):
    """Carga modelo entrenado en imágenes originales."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 3)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint['class_names']


def get_transform_warped():
    """Transform para imágenes warped (grayscale -> RGB)."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_transform_original():
    """Transform para imágenes originales."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def calculate_pulmonary_focus_score(gradcam_map, lung_mask):
    """
    Cuantifica qué porcentaje de la atención del modelo
    está dentro de la región pulmonar.
    """
    mask_resized = cv2.resize(lung_mask.astype(np.float32),
                               (gradcam_map.shape[1], gradcam_map.shape[0]))
    mask_binary = (mask_resized > 0.5).astype(np.float32)

    attention_in_lung = np.sum(gradcam_map * mask_binary)
    total_attention = np.sum(gradcam_map)

    if total_attention == 0:
        return 0.0

    return attention_in_lung / total_attention


def run_pfs_comparison(device, output_dir):
    """
    Compara Pulmonary Focus Score entre modelo Session 27 y Original.
    """
    print("\n" + "=" * 70)
    print("SESIÓN 27: Pulmonary Focus Score con Modelo de 98.02%")
    print("=" * 70)

    # Cargar modelos
    print("\nCargando modelos...")
    model_s27, class_names = load_model_session27(
        PROJECT_ROOT / 'outputs/session27_models/resnet18_expanded_15k_best.pt',
        device
    )
    print(f"  Modelo Session 27: 98.02% accuracy")

    model_original, _ = load_model_original(
        PROJECT_ROOT / 'outputs/classifier_comparison/resnet18_original/best_model.pt',
        device
    )
    print(f"  Modelo Original cargado")

    # Transforms
    transform_warped = get_transform_warped()
    transform_original = get_transform_original()

    # Dataset paths
    dataset_base = PROJECT_ROOT / 'data/dataset/COVID-19_Radiography_Dataset'
    warped_base = PROJECT_ROOT / 'outputs/full_warped_dataset/test'

    results = {
        'session27': {'scores': [], 'by_class': defaultdict(list)},
        'original': {'scores': [], 'by_class': defaultdict(list)}
    }

    class_folder_map = {
        'COVID': 'COVID',
        'Normal': 'Normal',
        'Viral_Pneumonia': 'Viral Pneumonia'
    }

    sample_count = 0
    max_samples_per_class = 50

    for class_name in class_names:
        class_folder = class_folder_map[class_name]
        warped_class_dir = warped_base / class_name
        original_images_dir = dataset_base / class_folder / 'images'
        masks_dir = dataset_base / class_folder / 'masks'

        if not warped_class_dir.exists():
            continue

        warped_files = list(warped_class_dir.glob('*.png'))[:max_samples_per_class]
        print(f"\nProcesando {len(warped_files)} imágenes de {class_name}...")

        for warped_path in tqdm(warped_files, desc=f"  {class_name}"):
            base_name = warped_path.stem
            if base_name.endswith('_warped'):
                original_name = base_name[:-7]
            else:
                original_name = base_name

            original_path = original_images_dir / f'{original_name}.png'
            mask_path = masks_dir / f'{original_name}.png'

            if not original_path.exists() or not mask_path.exists():
                continue

            # Cargar imágenes
            img_warped = Image.open(warped_path)
            img_original = Image.open(original_path).convert('RGB')
            mask = np.array(Image.open(mask_path).convert('L'))

            # Tensores
            tensor_warped = transform_warped(img_warped).unsqueeze(0).to(device)
            tensor_original = transform_original(img_original).unsqueeze(0).to(device)

            # Grad-CAM para modelo Session 27
            gradcam_s27 = GradCAM(model_s27, model_s27.layer4[-1])
            heatmap_s27, _, _ = gradcam_s27(tensor_warped)

            # Grad-CAM para modelo original
            gradcam_original = GradCAM(model_original, model_original.layer4[-1])
            heatmap_original, _, _ = gradcam_original(tensor_original)

            # Calcular Pulmonary Focus Score
            score_s27 = calculate_pulmonary_focus_score(heatmap_s27, mask)
            score_original = calculate_pulmonary_focus_score(heatmap_original, mask)

            results['session27']['scores'].append(score_s27)
            results['session27']['by_class'][class_name].append(score_s27)
            results['original']['scores'].append(score_original)
            results['original']['by_class'][class_name].append(score_original)

            sample_count += 1

    # Estadísticas
    print("\n" + "-" * 50)
    print("RESULTADOS: Pulmonary Focus Score")
    print("-" * 50)

    stats = {}
    for model_type in ['session27', 'original']:
        scores = np.array(results[model_type]['scores'])
        stats[model_type] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'median': float(np.median(scores)),
            'by_class': {}
        }
        label = "SESSION 27 (98.02%)" if model_type == 'session27' else "ORIGINAL"
        print(f"\n{label}:")
        print(f"  Media: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        print(f"  Mediana: {np.median(scores):.4f}")

        for class_name in class_names:
            class_scores = np.array(results[model_type]['by_class'][class_name])
            if len(class_scores) > 0:
                stats[model_type]['by_class'][class_name] = float(np.mean(class_scores))
                print(f"  {class_name}: {np.mean(class_scores):.4f}")

    # Diferencia
    diff = stats['session27']['mean'] - stats['original']['mean']
    print(f"\nDiferencia (Session27 - Original): {diff:+.4f}")
    print(f"Mejora relativa: {100*diff/stats['original']['mean']:+.1f}%")

    # Guardar resultados
    stats['difference'] = diff
    stats['samples_analyzed'] = sample_count

    with open(output_dir / 'session27_pfs_results.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # Crear visualización
    create_pfs_visualization(results, class_names, output_dir)

    return stats


def create_pfs_visualization(results, class_names, output_dir):
    """Crea visualización de Pulmonary Focus Score."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Box plot comparativo
    ax1 = axes[0]
    data = [results['original']['scores'], results['session27']['scores']]
    bp = ax1.boxplot(data, labels=['Original\n(sin warping)', 'Session 27\n(98.02% acc)'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightgreen')
    ax1.set_ylabel('Pulmonary Focus Score')
    ax1.set_title('Atención en Región Pulmonar\n(Mayor = Mejor)')
    ax1.grid(True, alpha=0.3)

    # 2. Por clase
    ax2 = axes[1]
    x = np.arange(len(class_names))
    width = 0.35

    means_orig = [np.mean(results['original']['by_class'][c]) for c in class_names]
    means_s27 = [np.mean(results['session27']['by_class'][c]) for c in class_names]

    ax2.bar(x - width/2, means_orig, width, label='Original', color='lightcoral')
    ax2.bar(x + width/2, means_s27, width, label='Session 27 (98.02%)', color='lightgreen')
    ax2.set_ylabel('Pulmonary Focus Score')
    ax2.set_title('Pulmonary Focus Score por Clase')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'session27_pfs_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nVisualización guardada: {output_dir / 'session27_pfs_comparison.png'}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")

    # Crear directorio de salida
    output_dir = PROJECT_ROOT / 'outputs/session27_models'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ejecutar comparación PFS
    results = run_pfs_comparison(device, output_dir)

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN SESIÓN 27 - PULMONARY FOCUS SCORE")
    print("=" * 70)
    print(f"\nModelo Session 27 (98.02% accuracy):")
    print(f"  PFS: {results['session27']['mean']:.4f}")
    print(f"\nModelo Original:")
    print(f"  PFS: {results['original']['mean']:.4f}")
    print(f"\nMejora: {100*results['difference']/results['original']['mean']:+.1f}%")
    print("=" * 70)


if __name__ == '__main__':
    main()
