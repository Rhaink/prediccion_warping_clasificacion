#!/usr/bin/env python3
"""
Sesion 24: Grad-CAM Multi-Arquitectura

Genera visualizaciones Grad-CAM para múltiples arquitecturas CNN,
comparando modelos entrenados en dataset original vs warped.

Arquitecturas soportadas:
- MobileNetV2, DenseNet-121, VGG-16, ResNet-18/50, EfficientNet-B0
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
import matplotlib.cm as cm
import cv2
import pandas as pd
import argparse

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class GradCAM:
    """Implementacion generalizada de Grad-CAM para cualquier arquitectura."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []

        # Registrar hooks
        self.handles.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        self.handles.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, target_class=None):
        """Genera mapa de calor Grad-CAM."""
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()

        # Calcular pesos (global average pooling de gradientes)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Combinacion ponderada de activaciones
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # ReLU y normalizar
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalizar a [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, target_class, confidence

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()


def get_target_layer(model, model_name):
    """Obtiene la capa objetivo para Grad-CAM segun la arquitectura."""
    if model_name in ['resnet18', 'resnet50']:
        return model.layer4[-1]
    elif model_name == 'mobilenet_v2':
        return model.features[-1]
    elif model_name == 'efficientnet_b0':
        return model.features[-1]
    elif model_name == 'densenet121':
        return model.features.denseblock4
    elif model_name == 'vgg16':
        return model.features[-1]
    elif model_name == 'alexnet':
        return model.features[-1]
    else:
        raise ValueError(f"Arquitectura no soportada: {model_name}")


def create_model(model_name, num_classes=3):
    """Crea modelo con estructura de clasificacion."""
    if model_name == 'alexnet':
        model = models.alexnet(weights=None)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    elif model_name == 'resnet18':
        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    elif model_name == 'densenet121':
        model = models.densenet121(weights=None)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    elif model_name == 'vgg16':
        model = models.vgg16(weights=None)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    return model


def load_model(model_path, model_name, device):
    """Carga modelo entrenado."""
    checkpoint = torch.load(model_path, map_location=device)

    model = create_model(model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    class_names = checkpoint.get('class_names', ['COVID', 'Normal', 'Viral_Pneumonia'])
    return model, class_names


def get_transform():
    """Transform para imagenes."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def overlay_heatmap(image, heatmap, alpha=0.5):
    """Superpone heatmap sobre imagen."""
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    if len(image.shape) == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image

    if image_rgb.max() <= 1:
        image_rgb = (image_rgb * 255).astype(np.uint8)

    overlay = (alpha * heatmap_colored + (1 - alpha) * image_rgb).astype(np.uint8)
    return overlay


def generate_gradcam_for_model(model_name, output_dir, device, n_samples=4):
    """Genera visualizaciones Grad-CAM para una arquitectura."""

    print(f"\n{'='*60}")
    print(f"Generando Grad-CAM para: {model_name.upper()}")
    print(f"{'='*60}")

    # Rutas de modelos
    model_orig_path = PROJECT_ROOT / f'outputs/classifier_comparison/{model_name}_original/best_model.pt'
    model_warp_path = PROJECT_ROOT / f'outputs/classifier_comparison/{model_name}_warped/best_model.pt'

    if not model_orig_path.exists() or not model_warp_path.exists():
        print(f"  Modelos no encontrados para {model_name}")
        return None

    # Cargar modelos
    print(f"  Cargando modelo original: {model_orig_path.name}")
    model_orig, class_names = load_model(model_orig_path, model_name, device)

    print(f"  Cargando modelo warped: {model_warp_path.name}")
    model_warp, _ = load_model(model_warp_path, model_name, device)

    # Obtener capas objetivo
    target_layer_orig = get_target_layer(model_orig, model_name)
    target_layer_warp = get_target_layer(model_warp, model_name)

    # Leer imagenes de test
    test_csv = PROJECT_ROOT / 'outputs/warped_dataset/test/images.csv'
    df = pd.read_csv(test_csv)

    # Seleccionar ejemplos de cada clase
    selected = []
    for category in ['COVID', 'Normal', 'Viral_Pneumonia']:
        cat_samples = df[df['category'] == category]['image_name'].tolist()
        selected.extend([(name, category) for name in cat_samples[:n_samples]])

    transform = get_transform()
    results = []

    # Crear directorio de salida
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Generar visualizaciones
    for i, (image_name, category) in enumerate(selected):
        # Rutas
        warped_path = PROJECT_ROOT / f'outputs/warped_dataset/test/{category}/{image_name}_warped.png'
        original_path = PROJECT_ROOT / f'data/dataset/{category}/{image_name}.png'

        if not warped_path.exists() or not original_path.exists():
            continue

        # Cargar imagenes
        img_warped = Image.open(warped_path).convert('RGB')
        img_original = Image.open(original_path).convert('RGB')

        img_warped_np = np.array(img_warped)
        img_original_np = np.array(img_original)

        # Tensors
        tensor_warped = transform(img_warped).unsqueeze(0).to(device)
        tensor_original = transform(img_original).unsqueeze(0).to(device)

        # Grad-CAM
        gradcam_orig = GradCAM(model_orig, target_layer_orig)
        heatmap_orig, pred_orig, conf_orig = gradcam_orig(tensor_original)
        gradcam_orig.remove_hooks()

        gradcam_warp = GradCAM(model_warp, target_layer_warp)
        heatmap_warp, pred_warp, conf_warp = gradcam_warp(tensor_warped)
        gradcam_warp.remove_hooks()

        # Overlays
        overlay_orig = overlay_heatmap(img_original_np, heatmap_orig)
        overlay_warp = overlay_heatmap(img_warped_np, heatmap_warp)

        results.append({
            'image': image_name,
            'category': category,
            'pred_original': class_names[pred_orig],
            'conf_original': conf_orig,
            'pred_warped': class_names[pred_warp],
            'conf_warped': conf_warp,
            'correct_original': class_names[pred_orig] == category,
            'correct_warped': class_names[pred_warp] == category,
            'overlay_orig': overlay_orig,
            'overlay_warp': overlay_warp,
            'img_original': img_original_np,
            'img_warped': img_warped_np,
        })

    # Crear grilla comparativa
    create_comparison_grid(results, model_name, class_names, model_output_dir)

    # Guardar resultados CSV
    df_results = pd.DataFrame([{k: v for k, v in r.items()
                               if k not in ['overlay_orig', 'overlay_warp', 'img_original', 'img_warped']}
                              for r in results])
    df_results.to_csv(model_output_dir / 'gradcam_results.csv', index=False)

    # Estadisticas
    correct_orig = sum(r['correct_original'] for r in results)
    correct_warp = sum(r['correct_warped'] for r in results)
    total = len(results)

    print(f"\n  Imagenes analizadas: {total}")
    print(f"  Accuracy Original: {correct_orig}/{total} ({100*correct_orig/total:.1f}%)")
    print(f"  Accuracy Warped: {correct_warp}/{total} ({100*correct_warp/total:.1f}%)")

    return results


def create_comparison_grid(results, model_name, class_names, output_dir):
    """Crea grilla comparativa 4x3 de Grad-CAM."""

    n_samples = min(len(results), 12)
    n_cols = 4
    n_rows = 3  # 3 filas para cada par (original arriba, warped abajo)

    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(16, 12))

    for idx, result in enumerate(results[:n_samples]):
        col = idx % n_cols
        row_pair = idx // n_cols
        row_orig = row_pair * 2
        row_warp = row_pair * 2 + 1

        # Imagen original + Grad-CAM
        axes[row_orig, col].imshow(result['overlay_orig'])
        pred_orig = result['pred_original']
        conf_orig = result['conf_original']
        color_orig = 'green' if result['correct_original'] else 'red'
        axes[row_orig, col].set_title(f"ORIG: {pred_orig}\n({conf_orig*100:.0f}%)",
                                      fontsize=9, color=color_orig)
        axes[row_orig, col].axis('off')

        # Imagen warped + Grad-CAM
        axes[row_warp, col].imshow(result['overlay_warp'])
        pred_warp = result['pred_warped']
        conf_warp = result['conf_warped']
        color_warp = 'green' if result['correct_warped'] else 'red'
        axes[row_warp, col].set_title(f"WARP: {pred_warp}\n({conf_warp*100:.0f}%)",
                                      fontsize=9, color=color_warp)
        axes[row_warp, col].axis('off')

        # Etiqueta GT
        if col == 0:
            axes[row_orig, col].set_ylabel(f"GT: {result['category'][:6]}",
                                           fontsize=9, rotation=0, labelpad=40)

    # Ocultar ejes vacios
    for idx in range(n_samples, n_rows * n_cols):
        col = idx % n_cols
        row_pair = idx // n_cols
        axes[row_pair * 2, col].axis('off')
        axes[row_pair * 2 + 1, col].axis('off')

    display_name = {
        'mobilenet_v2': 'MobileNetV2',
        'densenet121': 'DenseNet-121',
        'vgg16': 'VGG-16',
        'resnet18': 'ResNet-18',
        'resnet50': 'ResNet-50',
        'efficientnet_b0': 'EfficientNet-B0',
        'alexnet': 'AlexNet',
    }

    plt.suptitle(f'Grad-CAM: {display_name.get(model_name, model_name)}\n'
                 'Original (arriba) vs Warped (abajo) | Verde=Correcto, Rojo=Error',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / f'gradcam_grid_{model_name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Grilla guardada: {save_path}")


def create_multi_arch_summary(all_results, output_dir):
    """Crea resumen comparativo entre arquitecturas."""

    n_models = len(all_results)
    if n_models == 0:
        return

    fig, axes = plt.subplots(n_models, 8, figsize=(24, 4 * n_models))

    if n_models == 1:
        axes = axes.reshape(1, -1)

    display_names = {
        'mobilenet_v2': 'MobileNetV2',
        'densenet121': 'DenseNet-121',
        'vgg16': 'VGG-16',
        'resnet18': 'ResNet-18',
    }

    for model_idx, (model_name, results) in enumerate(all_results.items()):
        # Mostrar 4 ejemplos (1 de cada clase + 1 extra)
        for sample_idx, result in enumerate(results[:4]):
            col_orig = sample_idx * 2
            col_warp = sample_idx * 2 + 1

            # Original
            axes[model_idx, col_orig].imshow(result['overlay_orig'])
            color_o = 'green' if result['correct_original'] else 'red'
            axes[model_idx, col_orig].set_title(f"ORIG: {result['pred_original'][:3]}",
                                                fontsize=8, color=color_o)
            axes[model_idx, col_orig].axis('off')

            # Warped
            axes[model_idx, col_warp].imshow(result['overlay_warp'])
            color_w = 'green' if result['correct_warped'] else 'red'
            axes[model_idx, col_warp].set_title(f"WARP: {result['pred_warped'][:3]}",
                                                fontsize=8, color=color_w)
            axes[model_idx, col_warp].axis('off')

        # Label de modelo
        axes[model_idx, 0].set_ylabel(display_names.get(model_name, model_name),
                                      fontsize=11, rotation=0, labelpad=60,
                                      fontweight='bold')

    plt.suptitle('Grad-CAM Multi-Arquitectura: Original vs Warped\n'
                 'Comparacion de donde "mira" cada modelo',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / 'gradcam_multi_architecture_summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nResumen multi-arquitectura guardado: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM Multi-Arquitectura')
    parser.add_argument('--models', nargs='+',
                       default=['mobilenet_v2', 'densenet121'],
                       help='Lista de modelos a analizar')
    parser.add_argument('--n-samples', type=int, default=4,
                       help='Muestras por clase')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    output_dir = PROJECT_ROOT / 'outputs/gradcam_multi'
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_name in args.models:
        results = generate_gradcam_for_model(
            model_name, output_dir, device, args.n_samples
        )
        if results:
            all_results[model_name] = results

    # Crear resumen multi-arquitectura
    if len(all_results) > 0:
        create_multi_arch_summary(all_results, output_dir)

    print("\n" + "="*60)
    print("GRAD-CAM MULTI-ARQUITECTURA COMPLETADO")
    print("="*60)
    print(f"Arquitecturas analizadas: {list(all_results.keys())}")
    print(f"Salida: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
