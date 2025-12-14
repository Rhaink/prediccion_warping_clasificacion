#!/usr/bin/env python3
"""
Sesión 26 v2: Validación Avanzada - Test de Artefactos Mejorado

Mejoras:
- Artefactos más agresivos (marcas grandes, shortcuts visuales)
- Visualizaciones Grad-CAM comparativas detalladas
- Test en dataset warped vs original para mismo modelo
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import json
from collections import defaultdict
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Model Loading
# =============================================================================

def load_model_original(model_path, device):
    """Carga modelo entrenado en imágenes originales."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(num_features, 3)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint['class_names']


def get_transform_original():
    """Transform para imágenes originales."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# =============================================================================
# Artefactos Agresivos
# =============================================================================

def inject_aggressive_artifact(image, class_label, artifact_type='watermark'):
    """
    Inyecta artefacto agresivo correlacionado con clase.

    Tipos:
    - 'watermark': Texto grande semi-transparente
    - 'corner_box': Caja de color en esquina
    - 'border': Borde de color específico
    """
    img = image.copy().convert('RGBA')
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Mapeo de clase a color/símbolo
    class_config = {
        'COVID': {'color': (255, 0, 0), 'letter': 'C', 'text': 'COVID'},
        'Normal': {'color': (0, 255, 0), 'letter': 'N', 'text': 'NORMAL'},
        'Viral_Pneumonia': {'color': (0, 0, 255), 'letter': 'V', 'text': 'VP'}
    }

    config = class_config.get(class_label, class_config['COVID'])

    if artifact_type == 'watermark':
        # Texto grande semi-transparente en el centro
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
        except:
            font = ImageFont.load_default()

        # Overlay semi-transparente
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)

        text = config['text']
        bbox = draw_overlay.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (width - text_width) // 2
        y = (height - text_height) // 2

        color_with_alpha = (*config['color'], 128)  # 50% transparente
        draw_overlay.text((x, y), text, fill=color_with_alpha, font=font)

        img = Image.alpha_composite(img, overlay)

    elif artifact_type == 'corner_box':
        # Caja grande en esquina superior derecha
        box_size = 80
        draw.rectangle([width-box_size-10, 10, width-10, box_size+10],
                       fill=(*config['color'], 200))
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 50)
        except:
            font = ImageFont.load_default()
        draw.text((width-box_size, 20), config['letter'], fill=(255, 255, 255, 255), font=font)

    elif artifact_type == 'border':
        # Borde grueso de color específico
        border_width = 15
        draw.rectangle([0, 0, width-1, height-1],
                       outline=(*config['color'], 255), width=border_width)

    return img.convert('RGB')


def run_aggressive_artifact_test(device, output_dir):
    """
    Test de robustez con artefactos más agresivos.
    """
    print("\n" + "=" * 70)
    print("TEST DE ARTEFACTOS AGRESIVOS")
    print("=" * 70)

    # Cargar modelo
    print("\nCargando modelo original...")
    model_original, class_names = load_model_original(
        PROJECT_ROOT / 'outputs/classifier_comparison/resnet18_original/best_model.pt',
        device
    )

    transform = get_transform_original()

    # Dataset original
    dataset_base = PROJECT_ROOT / 'data/dataset/COVID-19_Radiography_Dataset'

    class_folder_map = {
        'COVID': 'COVID',
        'Normal': 'Normal',
        'Viral_Pneumonia': 'Viral Pneumonia'
    }

    artifact_types = ['watermark', 'corner_box', 'border']
    results = {atype: {'correct': 0, 'total': 0} for atype in artifact_types}
    results['clean'] = {'correct': 0, 'total': 0}

    max_samples_per_class = 100

    for class_idx, class_name in enumerate(class_names):
        class_folder = class_folder_map[class_name]
        images_dir = dataset_base / class_folder / 'images'

        if not images_dir.exists():
            continue

        image_files = list(images_dir.glob('*.png'))[:max_samples_per_class]
        print(f"\nProcesando {len(image_files)} imágenes de {class_name}...")

        for img_path in tqdm(image_files, desc=f"  {class_name}"):
            img_clean = Image.open(img_path).convert('RGB')

            # Test con imagen limpia
            tensor_clean = transform(img_clean).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_clean = model_original(tensor_clean).argmax(dim=1).item()

            results['clean']['total'] += 1
            if pred_clean == class_idx:
                results['clean']['correct'] += 1

            # Test con cada tipo de artefacto
            for atype in artifact_types:
                img_artifact = inject_aggressive_artifact(img_clean, class_name, atype)
                tensor_artifact = transform(img_artifact).unsqueeze(0).to(device)

                with torch.no_grad():
                    pred_artifact = model_original(tensor_artifact).argmax(dim=1).item()

                results[atype]['total'] += 1
                if pred_artifact == class_idx:
                    results[atype]['correct'] += 1

    # Estadísticas
    print("\n" + "-" * 50)
    print("RESULTADOS: Test de Artefactos Agresivos")
    print("-" * 50)

    stats = {}
    for condition, data in results.items():
        acc = 100 * data['correct'] / data['total'] if data['total'] > 0 else 0
        stats[condition] = acc
        print(f"  {condition:15}: {acc:.2f}% ({data['correct']}/{data['total']})")

    clean_acc = stats['clean']
    print(f"\nDegradación por artefacto:")
    for atype in artifact_types:
        deg = clean_acc - stats[atype]
        print(f"  {atype:15}: {deg:+.2f} puntos")

    # Guardar
    with open(output_dir / 'aggressive_artifact_results.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # Visualización
    create_artifact_comparison_viz(results, artifact_types, output_dir)

    # Crear ejemplos visuales
    create_artifact_visual_examples(class_names, dataset_base, class_folder_map, output_dir)

    return stats


def create_artifact_comparison_viz(results, artifact_types, output_dir):
    """Crea visualización comparativa de artefactos."""
    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = ['clean'] + artifact_types
    accuracies = [100 * results[c]['correct'] / results[c]['total'] for c in conditions]
    colors = ['green'] + ['red'] * len(artifact_types)

    bars = ax.bar(conditions, accuracies, color=colors, alpha=0.7)

    ax.axhline(y=accuracies[0], color='green', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Robustez a Artefactos Sintéticos Agresivos\n(Modelo Original)')
    ax.set_ylim(0, 105)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'aggressive_artifact_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nVisualización guardada: {output_dir / 'aggressive_artifact_comparison.png'}")


def create_artifact_visual_examples(class_names, dataset_base, class_folder_map, output_dir):
    """Crea ejemplos visuales de los artefactos."""
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))

    artifact_types = ['clean', 'watermark', 'corner_box', 'border']

    for row, class_name in enumerate(class_names):
        class_folder = class_folder_map[class_name]
        images_dir = dataset_base / class_folder / 'images'

        sample_path = list(images_dir.glob('*.png'))[0]
        img_clean = Image.open(sample_path).convert('RGB')

        for col, atype in enumerate(artifact_types):
            ax = axes[row, col]

            if atype == 'clean':
                ax.imshow(img_clean)
            else:
                img_artifact = inject_aggressive_artifact(img_clean, class_name, atype)
                ax.imshow(img_artifact)

            ax.axis('off')
            if row == 0:
                ax.set_title(atype.replace('_', ' ').title(), fontsize=12)

        axes[row, 0].set_ylabel(class_name, fontsize=12, rotation=0, labelpad=60)

    plt.suptitle('Ejemplos de Artefactos Sintéticos Agresivos por Clase', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'artifact_examples_aggressive.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Ejemplos guardados: {output_dir / 'artifact_examples_aggressive.png'}")


# =============================================================================
# Grad-CAM Comparativo Detallado
# =============================================================================

class GradCAM:
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


def create_gradcam_grid(device, output_dir):
    """
    Crea grilla de Grad-CAM comparando modelo original vs imágenes con/sin artefactos.
    """
    print("\n" + "=" * 70)
    print("GRAD-CAM: Visualización de Artefactos")
    print("=" * 70)

    model, class_names = load_model_original(
        PROJECT_ROOT / 'outputs/classifier_comparison/resnet18_original/best_model.pt',
        device
    )

    transform = get_transform_original()
    dataset_base = PROJECT_ROOT / 'data/dataset/COVID-19_Radiography_Dataset'

    class_folder_map = {
        'COVID': 'COVID',
        'Normal': 'Normal',
        'Viral_Pneumonia': 'Viral Pneumonia'
    }

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    for row, class_name in enumerate(class_names):
        class_folder = class_folder_map[class_name]
        images_dir = dataset_base / class_folder / 'images'

        sample_path = list(images_dir.glob('*.png'))[5]  # Tomar el 6to ejemplo
        img_clean = Image.open(sample_path).convert('RGB')
        img_np_clean = np.array(img_clean)

        # Columna 1: Imagen original
        ax = axes[row, 0]
        ax.imshow(img_np_clean)
        ax.set_title('Original' if row == 0 else '', fontsize=11)
        ax.axis('off')
        if row == 0:
            ax.set_title('Imagen Original', fontsize=11)

        # Columna 2: Grad-CAM en imagen limpia
        tensor_clean = transform(img_clean).unsqueeze(0).to(device)
        gradcam = GradCAM(model, model.layer4[-1])
        heatmap, pred_class, conf = gradcam(tensor_clean)
        overlay = overlay_heatmap(img_np_clean, heatmap)

        ax = axes[row, 1]
        ax.imshow(overlay)
        pred_name = class_names[pred_class]
        color = 'green' if pred_name == class_name else 'red'
        ax.set_title(f'{pred_name} ({conf*100:.0f}%)' if row > 0 else f'Grad-CAM Limpia\n{pred_name} ({conf*100:.0f}%)',
                     fontsize=10, color=color)
        ax.axis('off')

        # Columna 3: Imagen con watermark
        img_wm = inject_aggressive_artifact(img_clean, class_name, 'watermark')
        ax = axes[row, 2]
        ax.imshow(img_wm)
        ax.set_title('Con Watermark' if row == 0 else '', fontsize=11)
        ax.axis('off')

        # Columna 4: Grad-CAM con watermark
        tensor_wm = transform(img_wm).unsqueeze(0).to(device)
        gradcam_wm = GradCAM(model, model.layer4[-1])
        heatmap_wm, pred_wm, conf_wm = gradcam_wm(tensor_wm)
        img_wm_np = np.array(img_wm)
        overlay_wm = overlay_heatmap(img_wm_np, heatmap_wm)

        ax = axes[row, 3]
        ax.imshow(overlay_wm)
        pred_name_wm = class_names[pred_wm]
        color_wm = 'green' if pred_name_wm == class_name else 'red'
        ax.set_title(f'{pred_name_wm} ({conf_wm*100:.0f}%)' if row > 0 else f'Grad-CAM Watermark\n{pred_name_wm} ({conf_wm*100:.0f}%)',
                     fontsize=10, color=color_wm)
        ax.axis('off')

        # Etiqueta de clase
        axes[row, 0].text(-0.15, 0.5, class_name.replace('_', '\n'),
                          transform=axes[row, 0].transAxes,
                          fontsize=12, fontweight='bold',
                          va='center', ha='right')

    plt.suptitle('Grad-CAM: ¿A dónde mira el modelo con artefactos?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'gradcam_artifact_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nGrad-CAM guardado: {output_dir / 'gradcam_artifact_comparison.png'}")


# =============================================================================
# Análisis de Invarianza: Warped vs Original
# =============================================================================

def analyze_warping_invariance(device, output_dir):
    """
    Analiza si el warping proporciona invarianza geométrica.
    Compara predicciones para la misma imagen en formato original vs warped.
    """
    print("\n" + "=" * 70)
    print("ANÁLISIS DE INVARIANZA POR WARPING")
    print("=" * 70)

    model, class_names = load_model_original(
        PROJECT_ROOT / 'outputs/classifier_comparison/resnet18_original/best_model.pt',
        device
    )

    transform = get_transform_original()

    # Rutas
    original_base = PROJECT_ROOT / 'data/dataset/COVID-19_Radiography_Dataset'
    warped_base = PROJECT_ROOT / 'outputs/full_warped_dataset/test'

    class_folder_map = {
        'COVID': 'COVID',
        'Normal': 'Normal',
        'Viral_Pneumonia': 'Viral Pneumonia'
    }

    results = {
        'original': {'correct': 0, 'total': 0, 'confidences': []},
        'warped': {'correct': 0, 'total': 0, 'confidences': []},
        'agreement': 0,
        'disagreement': 0
    }

    max_samples = 100

    for class_idx, class_name in enumerate(class_names):
        class_folder = class_folder_map[class_name]
        original_dir = original_base / class_folder / 'images'
        warped_dir = warped_base / class_name

        if not warped_dir.exists():
            continue

        warped_files = list(warped_dir.glob('*.png'))[:max_samples]
        print(f"\nComparando {len(warped_files)} imágenes de {class_name}...")

        for warped_path in tqdm(warped_files, desc=f"  {class_name}"):
            # Extraer nombre original
            base_name = warped_path.stem
            if base_name.endswith('_warped'):
                original_name = base_name[:-7]
            else:
                original_name = base_name

            original_path = original_dir / f'{original_name}.png'

            if not original_path.exists():
                continue

            # Cargar imágenes
            img_original = Image.open(original_path).convert('RGB')
            img_warped = Image.open(warped_path).convert('RGB')

            # Predicciones
            tensor_orig = transform(img_original).unsqueeze(0).to(device)
            tensor_warp = transform(img_warped).unsqueeze(0).to(device)

            with torch.no_grad():
                out_orig = model(tensor_orig)
                out_warp = model(tensor_warp)

                pred_orig = out_orig.argmax(dim=1).item()
                pred_warp = out_warp.argmax(dim=1).item()

                conf_orig = F.softmax(out_orig, dim=1)[0, pred_orig].item()
                conf_warp = F.softmax(out_warp, dim=1)[0, pred_warp].item()

            # Registrar
            results['original']['total'] += 1
            results['warped']['total'] += 1
            results['original']['confidences'].append(conf_orig)
            results['warped']['confidences'].append(conf_warp)

            if pred_orig == class_idx:
                results['original']['correct'] += 1
            if pred_warp == class_idx:
                results['warped']['correct'] += 1

            if pred_orig == pred_warp:
                results['agreement'] += 1
            else:
                results['disagreement'] += 1

    # Estadísticas
    print("\n" + "-" * 50)
    print("RESULTADOS: Invarianza por Warping")
    print("-" * 50)

    acc_orig = 100 * results['original']['correct'] / results['original']['total']
    acc_warp = 100 * results['warped']['correct'] / results['warped']['total']
    agreement_rate = 100 * results['agreement'] / (results['agreement'] + results['disagreement'])

    print(f"\nAccuracy en imágenes ORIGINALES: {acc_orig:.2f}%")
    print(f"Accuracy en imágenes WARPED: {acc_warp:.2f}%")
    print(f"Diferencia: {acc_warp - acc_orig:+.2f} puntos")
    print(f"\nTasa de acuerdo (pred_orig == pred_warp): {agreement_rate:.2f}%")
    print(f"Desacuerdos: {results['disagreement']} de {results['agreement'] + results['disagreement']}")

    conf_orig_mean = np.mean(results['original']['confidences'])
    conf_warp_mean = np.mean(results['warped']['confidences'])
    print(f"\nConfianza media en originales: {conf_orig_mean:.3f}")
    print(f"Confianza media en warped: {conf_warp_mean:.3f}")

    stats = {
        'accuracy_original': acc_orig,
        'accuracy_warped': acc_warp,
        'agreement_rate': agreement_rate,
        'confidence_original': conf_orig_mean,
        'confidence_warped': conf_warp_mean,
        'total_samples': results['original']['total']
    }

    with open(output_dir / 'invariance_analysis.json', 'w') as f:
        json.dump(stats, f, indent=2)

    return stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")

    output_dir = PROJECT_ROOT / 'outputs/session26_validation'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 70)
    print("# SESIÓN 26 v2: Validación Avanzada")
    print("#" * 70)

    # Test de artefactos agresivos
    artifact_results = run_aggressive_artifact_test(device, output_dir)

    # Grad-CAM con artefactos
    create_gradcam_grid(device, output_dir)

    # Análisis de invarianza
    invariance_results = analyze_warping_invariance(device, output_dir)

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN SESIÓN 26 v2")
    print("=" * 70)

    print("\n1. TEST DE ARTEFACTOS AGRESIVOS:")
    for k, v in artifact_results.items():
        print(f"   {k}: {v:.2f}%")

    print("\n2. ANÁLISIS DE INVARIANZA:")
    print(f"   Accuracy Original: {invariance_results['accuracy_original']:.2f}%")
    print(f"   Accuracy Warped: {invariance_results['accuracy_warped']:.2f}%")
    print(f"   Tasa de Acuerdo: {invariance_results['agreement_rate']:.2f}%")

    print(f"\nResultados en: {output_dir}")


if __name__ == '__main__':
    main()
