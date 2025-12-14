#!/usr/bin/env python3
"""
Sesión 26: Validación Avanzada del Pipeline de Warping

Experimentos:
1. Grad-CAM con Pulmonary Focus Score - Verifica que el modelo mira regiones pulmonares
2. Test de Artefactos Sintéticos - Demuestra robustez a shortcuts
3. Análisis de Errores - Identifica patrones en clasificaciones incorrectas
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
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Grad-CAM Implementation
# =============================================================================

class GradCAM:
    """Implementación de Grad-CAM para ResNet-18."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Registrar hooks
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


# =============================================================================
# Model Loading
# =============================================================================

def load_model_warped(model_path, device):
    """Carga modelo entrenado en dataset warped (margin 1.05)."""
    state_dict = torch.load(model_path, map_location=device, weights_only=False)

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 3)

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    class_names = ['COVID', 'Normal', 'Viral_Pneumonia']
    return model, class_names


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


# =============================================================================
# Transforms
# =============================================================================

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


# =============================================================================
# EXPERIMENTO 1: Pulmonary Focus Score
# =============================================================================

def calculate_pulmonary_focus_score(gradcam_map, lung_mask):
    """
    Cuantifica qué porcentaje de la atención del modelo
    está dentro de la región pulmonar.

    Score = sum(gradcam * mask) / sum(gradcam)
    """
    # Redimensionar máscara al tamaño del gradcam
    mask_resized = cv2.resize(lung_mask.astype(np.float32),
                               (gradcam_map.shape[1], gradcam_map.shape[0]))
    mask_binary = (mask_resized > 0.5).astype(np.float32)

    attention_in_lung = np.sum(gradcam_map * mask_binary)
    total_attention = np.sum(gradcam_map)

    if total_attention == 0:
        return 0.0

    return attention_in_lung / total_attention


def run_pulmonary_focus_experiment(device, output_dir):
    """
    Experimento 1: Comparar Pulmonary Focus Score entre modelos.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTO 1: Pulmonary Focus Score")
    print("=" * 70)

    # Cargar modelos
    print("\nCargando modelos...")
    model_warped, class_names = load_model_warped(
        PROJECT_ROOT / 'outputs/margin_experiment/margin_1.05/models/resnet18_best.pt',
        device
    )
    model_original, _ = load_model_original(
        PROJECT_ROOT / 'outputs/classifier_comparison/resnet18_original/best_model.pt',
        device
    )

    # Transforms
    transform_warped = get_transform_warped()
    transform_original = get_transform_original()

    # Dataset paths
    dataset_base = PROJECT_ROOT / 'data/dataset/COVID-19_Radiography_Dataset'
    warped_base = PROJECT_ROOT / 'outputs/full_warped_dataset/test'

    results = {
        'warped': {'scores': [], 'by_class': defaultdict(list)},
        'original': {'scores': [], 'by_class': defaultdict(list)}
    }

    # Mapeo de nombres de carpetas
    class_folder_map = {
        'COVID': 'COVID',
        'Normal': 'Normal',
        'Viral_Pneumonia': 'Viral Pneumonia'
    }

    # Procesar imágenes de test
    sample_count = 0
    max_samples_per_class = 50  # Limitar para no tardar demasiado

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
            # Extraer nombre original (quitar _warped.png)
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

            # Grad-CAM para modelo warped
            gradcam_warped = GradCAM(model_warped, model_warped.layer4[-1])
            heatmap_warped, _, _ = gradcam_warped(tensor_warped)

            # Grad-CAM para modelo original
            gradcam_original = GradCAM(model_original, model_original.layer4[-1])
            heatmap_original, _, _ = gradcam_original(tensor_original)

            # Calcular Pulmonary Focus Score
            score_warped = calculate_pulmonary_focus_score(heatmap_warped, mask)
            score_original = calculate_pulmonary_focus_score(heatmap_original, mask)

            results['warped']['scores'].append(score_warped)
            results['warped']['by_class'][class_name].append(score_warped)
            results['original']['scores'].append(score_original)
            results['original']['by_class'][class_name].append(score_original)

            sample_count += 1

    # Estadísticas
    print("\n" + "-" * 50)
    print("RESULTADOS: Pulmonary Focus Score")
    print("-" * 50)

    stats = {}
    for model_type in ['warped', 'original']:
        scores = np.array(results[model_type]['scores'])
        stats[model_type] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'median': float(np.median(scores)),
            'by_class': {}
        }
        print(f"\n{model_type.upper()}:")
        print(f"  Media: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        print(f"  Mediana: {np.median(scores):.4f}")

        for class_name in class_names:
            class_scores = np.array(results[model_type]['by_class'][class_name])
            if len(class_scores) > 0:
                stats[model_type]['by_class'][class_name] = float(np.mean(class_scores))
                print(f"  {class_name}: {np.mean(class_scores):.4f}")

    # Diferencia
    diff = stats['warped']['mean'] - stats['original']['mean']
    print(f"\nDiferencia (Warped - Original): {diff:+.4f}")
    print(f"Mejora relativa: {100*diff/stats['original']['mean']:+.1f}%")

    # Guardar resultados
    stats['difference'] = diff
    stats['samples_analyzed'] = sample_count

    with open(output_dir / 'pulmonary_focus_scores.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # Crear visualización
    create_pfs_visualization(results, class_names, output_dir)

    return stats


def create_pfs_visualization(results, class_names, output_dir):
    """Crea visualización de Pulmonary Focus Score."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Box plot comparativo
    ax1 = axes[0]
    data = [results['original']['scores'], results['warped']['scores']]
    bp = ax1.boxplot(data, labels=['Original', 'Warped'], patch_artist=True)
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
    means_warp = [np.mean(results['warped']['by_class'][c]) for c in class_names]

    ax2.bar(x - width/2, means_orig, width, label='Original', color='lightcoral')
    ax2.bar(x + width/2, means_warp, width, label='Warped', color='lightgreen')
    ax2.set_ylabel('Pulmonary Focus Score')
    ax2.set_title('Pulmonary Focus Score por Clase')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'pulmonary_focus_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nVisualización guardada: {output_dir / 'pulmonary_focus_comparison.png'}")


# =============================================================================
# EXPERIMENTO 2: Test de Artefactos Sintéticos
# =============================================================================

def inject_synthetic_artifact(image, class_label, artifact_type='letter'):
    """
    Inyecta artefacto sintético correlacionado con clase.

    Args:
        image: PIL Image
        class_label: 'COVID', 'Normal', o 'Viral_Pneumonia'
        artifact_type: 'letter' o 'shape'

    Returns:
        Imagen con artefacto inyectado
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Mapeo de clase a artefacto
    artifacts = {
        'COVID': ('C', (255, 255, 255)),
        'Normal': ('N', (255, 255, 255)),
        'Viral_Pneumonia': ('V', (255, 255, 255))
    }

    letter, color = artifacts.get(class_label, ('?', (255, 255, 255)))

    # Posición: esquina inferior derecha
    img_width, img_height = img.size
    x = img_width - 40
    y = img_height - 40

    # Dibujar letra grande
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    except:
        font = ImageFont.load_default()

    draw.text((x, y), letter, fill=color, font=font)

    return img


def run_synthetic_artifact_experiment(device, output_dir):
    """
    Experimento 2: Test de robustez a artefactos sintéticos.

    Hipótesis: El modelo warped debe ser INMUNE a artefactos porque
    el warping los elimina. El modelo original debe CONFUNDIRSE.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTO 2: Test de Artefactos Sintéticos")
    print("=" * 70)

    # Cargar modelos
    print("\nCargando modelos...")
    model_original, class_names = load_model_original(
        PROJECT_ROOT / 'outputs/classifier_comparison/resnet18_original/best_model.pt',
        device
    )

    transform = get_transform_original()

    # Dataset original
    dataset_base = PROJECT_ROOT / 'data/dataset/COVID-19_Radiography_Dataset'

    # Mapeo de carpetas
    class_folder_map = {
        'COVID': 'COVID',
        'Normal': 'Normal',
        'Viral_Pneumonia': 'Viral Pneumonia'
    }

    results = {
        'clean': {'correct': 0, 'total': 0, 'by_class': defaultdict(lambda: {'correct': 0, 'total': 0})},
        'with_artifact': {'correct': 0, 'total': 0, 'by_class': defaultdict(lambda: {'correct': 0, 'total': 0})},
        'confusions': []  # Casos donde el artefacto cambió la predicción
    }

    max_samples_per_class = 100

    for class_idx, class_name in enumerate(class_names):
        class_folder = class_folder_map[class_name]
        images_dir = dataset_base / class_folder / 'images'

        if not images_dir.exists():
            continue

        image_files = list(images_dir.glob('*.png'))[:max_samples_per_class]
        print(f"\nProcesando {len(image_files)} imágenes de {class_name}...")

        for img_path in tqdm(image_files, desc=f"  {class_name}"):
            # Cargar imagen original
            img_clean = Image.open(img_path).convert('RGB')

            # Crear versión con artefacto
            img_artifact = inject_synthetic_artifact(img_clean, class_name)

            # Tensores
            tensor_clean = transform(img_clean).unsqueeze(0).to(device)
            tensor_artifact = transform(img_artifact).unsqueeze(0).to(device)

            # Predicciones
            with torch.no_grad():
                pred_clean = model_original(tensor_clean).argmax(dim=1).item()
                pred_artifact = model_original(tensor_artifact).argmax(dim=1).item()

            # Registrar resultados para imagen limpia
            correct_clean = (pred_clean == class_idx)
            results['clean']['total'] += 1
            results['clean']['by_class'][class_name]['total'] += 1
            if correct_clean:
                results['clean']['correct'] += 1
                results['clean']['by_class'][class_name]['correct'] += 1

            # Registrar resultados para imagen con artefacto
            correct_artifact = (pred_artifact == class_idx)
            results['with_artifact']['total'] += 1
            results['with_artifact']['by_class'][class_name]['total'] += 1
            if correct_artifact:
                results['with_artifact']['correct'] += 1
                results['with_artifact']['by_class'][class_name]['correct'] += 1

            # Registrar confusiones
            if pred_clean != pred_artifact:
                results['confusions'].append({
                    'image': img_path.name,
                    'true_class': class_name,
                    'pred_clean': class_names[pred_clean],
                    'pred_artifact': class_names[pred_artifact]
                })

    # Estadísticas
    print("\n" + "-" * 50)
    print("RESULTADOS: Test de Artefactos Sintéticos")
    print("-" * 50)

    acc_clean = 100 * results['clean']['correct'] / results['clean']['total']
    acc_artifact = 100 * results['with_artifact']['correct'] / results['with_artifact']['total']

    print(f"\nModelo Original en imágenes LIMPIAS: {acc_clean:.2f}%")
    print(f"Modelo Original con ARTEFACTOS: {acc_artifact:.2f}%")
    print(f"Degradación por artefactos: {acc_clean - acc_artifact:+.2f} puntos")

    print(f"\nConfusiones por artefacto: {len(results['confusions'])} de {results['clean']['total']}")
    print(f"Tasa de confusión: {100*len(results['confusions'])/results['clean']['total']:.2f}%")

    # Por clase
    print("\nPor clase:")
    for class_name in class_names:
        clean_acc = 100 * results['clean']['by_class'][class_name]['correct'] / results['clean']['by_class'][class_name]['total']
        art_acc = 100 * results['with_artifact']['by_class'][class_name]['correct'] / results['with_artifact']['by_class'][class_name]['total']
        print(f"  {class_name}: Limpia={clean_acc:.1f}%, Artefacto={art_acc:.1f}%, Δ={clean_acc-art_acc:+.1f}")

    # Guardar resultados
    stats = {
        'accuracy_clean': acc_clean,
        'accuracy_with_artifact': acc_artifact,
        'degradation': acc_clean - acc_artifact,
        'confusion_rate': 100 * len(results['confusions']) / results['clean']['total'],
        'total_samples': results['clean']['total'],
        'confusions': results['confusions'][:20]  # Solo primeras 20
    }

    with open(output_dir / 'artifact_test_results.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # Crear visualización de ejemplos de confusión
    create_artifact_examples(results['confusions'][:6], class_names, dataset_base,
                             class_folder_map, output_dir)

    return stats


def create_artifact_examples(confusions, class_names, dataset_base, class_folder_map, output_dir):
    """Crea visualización de ejemplos de confusión por artefacto."""
    if not confusions:
        return

    n_examples = min(6, len(confusions))
    fig, axes = plt.subplots(2, n_examples, figsize=(3*n_examples, 6))

    for idx, conf in enumerate(confusions[:n_examples]):
        class_folder = class_folder_map[conf['true_class']]
        img_path = dataset_base / class_folder / 'images' / conf['image']

        if not img_path.exists():
            continue

        img_clean = Image.open(img_path).convert('RGB')
        img_artifact = inject_synthetic_artifact(img_clean, conf['true_class'])

        # Plot limpia
        ax_clean = axes[0, idx] if n_examples > 1 else axes[0]
        ax_clean.imshow(img_clean)
        ax_clean.set_title(f"Pred: {conf['pred_clean']}", fontsize=9)
        ax_clean.axis('off')

        # Plot con artefacto
        ax_art = axes[1, idx] if n_examples > 1 else axes[1]
        ax_art.imshow(img_artifact)
        ax_art.set_title(f"Pred: {conf['pred_artifact']}", fontsize=9,
                        color='red' if conf['pred_artifact'] != conf['true_class'] else 'green')
        ax_art.axis('off')

    axes[0, 0].set_ylabel('Limpia', fontsize=10)
    axes[1, 0].set_ylabel('Con Artefacto', fontsize=10)

    plt.suptitle('Confusiones Causadas por Artefactos Sintéticos\n(El artefacto cambia la predicción)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'artifact_confusion_examples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nEjemplos guardados: {output_dir / 'artifact_confusion_examples.png'}")


# =============================================================================
# EXPERIMENTO 3: Análisis de Errores
# =============================================================================

def run_error_analysis(device, output_dir):
    """
    Experimento 3: Análisis detallado de errores del modelo warped.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTO 3: Análisis de Errores")
    print("=" * 70)

    # Cargar modelo warped
    print("\nCargando modelo warped...")
    model_warped, class_names = load_model_warped(
        PROJECT_ROOT / 'outputs/margin_experiment/margin_1.05/models/resnet18_best.pt',
        device
    )

    transform = get_transform_warped()

    # Dataset warped test
    warped_base = PROJECT_ROOT / 'outputs/full_warped_dataset/test'

    # Recopilar errores
    errors = []
    all_predictions = []
    confusion_matrix = np.zeros((3, 3), dtype=int)

    for class_idx, class_name in enumerate(class_names):
        class_dir = warped_base / class_name

        if not class_dir.exists():
            continue

        image_files = list(class_dir.glob('*.png'))
        print(f"\nEvaluando {len(image_files)} imágenes de {class_name}...")

        for img_path in tqdm(image_files, desc=f"  {class_name}"):
            img = Image.open(img_path)
            tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model_warped(tensor)
                probs = F.softmax(output, dim=1)
                pred = output.argmax(dim=1).item()
                confidence = probs[0, pred].item()

            confusion_matrix[class_idx, pred] += 1

            prediction_info = {
                'image': img_path.name,
                'true_class': class_name,
                'pred_class': class_names[pred],
                'confidence': confidence,
                'correct': pred == class_idx
            }
            all_predictions.append(prediction_info)

            if pred != class_idx:
                errors.append(prediction_info)

    # Estadísticas
    print("\n" + "-" * 50)
    print("RESULTADOS: Análisis de Errores")
    print("-" * 50)

    total = len(all_predictions)
    correct = total - len(errors)
    accuracy = 100 * correct / total

    print(f"\nTotal imágenes: {total}")
    print(f"Correctas: {correct}")
    print(f"Errores: {len(errors)}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Matriz de confusión
    print("\nMatriz de Confusión:")
    print(f"{'':15} {'COVID':>10} {'Normal':>10} {'VP':>10}")
    for i, class_name in enumerate(class_names):
        row = confusion_matrix[i]
        print(f"{class_name:15} {row[0]:>10} {row[1]:>10} {row[2]:>10}")

    # Análisis de patrones de error
    print("\nPatrones de Error:")
    error_patterns = defaultdict(int)
    for err in errors:
        pattern = f"{err['true_class']} -> {err['pred_class']}"
        error_patterns[pattern] += 1

    for pattern, count in sorted(error_patterns.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(errors) if errors else 0
        print(f"  {pattern}: {count} ({pct:.1f}% de errores)")

    # Errores con baja confianza vs alta confianza
    if errors:
        confidences = [e['confidence'] for e in errors]
        low_conf = [e for e in errors if e['confidence'] < 0.7]
        high_conf = [e for e in errors if e['confidence'] >= 0.7]

        print(f"\nErrores con baja confianza (<70%): {len(low_conf)} ({100*len(low_conf)/len(errors):.1f}%)")
        print(f"Errores con alta confianza (>=70%): {len(high_conf)} ({100*len(high_conf)/len(errors):.1f}%)")

    # Guardar resultados
    stats = {
        'total': total,
        'correct': correct,
        'errors': len(errors),
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix.tolist(),
        'error_patterns': dict(error_patterns),
        'sample_errors': errors[:20]  # Primeros 20 errores
    }

    with open(output_dir / 'error_analysis.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # Visualización
    create_error_visualization(confusion_matrix, class_names, error_patterns, output_dir)

    return stats


def create_error_visualization(confusion_matrix, class_names, error_patterns, output_dir):
    """Crea visualización del análisis de errores."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Matriz de confusión
    ax1 = axes[0]
    im = ax1.imshow(confusion_matrix, cmap='Blues')

    ax1.set_xticks(range(len(class_names)))
    ax1.set_yticks(range(len(class_names)))
    ax1.set_xticklabels(['COVID', 'Normal', 'VP'])
    ax1.set_yticklabels(['COVID', 'Normal', 'VP'])
    ax1.set_xlabel('Predicho')
    ax1.set_ylabel('Real')
    ax1.set_title('Matriz de Confusión')

    # Anotar valores
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax1.text(j, i, confusion_matrix[i, j],
                           ha="center", va="center",
                           color="white" if confusion_matrix[i, j] > confusion_matrix.max()/2 else "black")

    # 2. Patrones de error
    ax2 = axes[1]
    if error_patterns:
        patterns = list(error_patterns.keys())
        counts = list(error_patterns.values())
        colors = ['red' if 'COVID' in p.split(' -> ')[0] else 'orange' if 'Normal' in p.split(' -> ')[0] else 'purple'
                  for p in patterns]
        ax2.barh(patterns, counts, color=colors)
        ax2.set_xlabel('Número de errores')
        ax2.set_title('Patrones de Error')
        ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_dir / 'error_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nVisualización guardada: {output_dir / 'error_analysis.png'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")

    # Crear directorio de salida
    output_dir = PROJECT_ROOT / 'outputs/session26_validation'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ejecutar experimentos
    results = {}

    # Experimento 1: Pulmonary Focus Score
    print("\n" + "#" * 70)
    print("# INICIANDO EXPERIMENTOS DE VALIDACIÓN - SESIÓN 26")
    print("#" * 70)

    results['pulmonary_focus'] = run_pulmonary_focus_experiment(device, output_dir)

    # Experimento 2: Artefactos Sintéticos
    results['artifact_test'] = run_synthetic_artifact_experiment(device, output_dir)

    # Experimento 3: Análisis de Errores
    results['error_analysis'] = run_error_analysis(device, output_dir)

    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN FINAL - SESIÓN 26")
    print("=" * 70)

    print("\n1. PULMONARY FOCUS SCORE:")
    pf = results['pulmonary_focus']
    print(f"   Warped: {pf['warped']['mean']:.4f} ± {pf['warped']['std']:.4f}")
    print(f"   Original: {pf['original']['mean']:.4f} ± {pf['original']['std']:.4f}")
    print(f"   Mejora: {100*pf['difference']/pf['original']['mean']:+.1f}%")

    print("\n2. TEST DE ARTEFACTOS:")
    af = results['artifact_test']
    print(f"   Accuracy limpia: {af['accuracy_clean']:.2f}%")
    print(f"   Accuracy con artefacto: {af['accuracy_with_artifact']:.2f}%")
    print(f"   Tasa de confusión: {af['confusion_rate']:.2f}%")

    print("\n3. ANÁLISIS DE ERRORES:")
    ea = results['error_analysis']
    print(f"   Accuracy modelo warped: {ea['accuracy']:.2f}%")
    print(f"   Errores totales: {ea['errors']}")
    print(f"   Patrón más común: {max(ea['error_patterns'].items(), key=lambda x: x[1]) if ea['error_patterns'] else 'N/A'}")

    # Guardar resumen
    with open(output_dir / 'session26_summary.json', 'w') as f:
        json.dump({
            'pulmonary_focus': {
                'warped_mean': pf['warped']['mean'],
                'original_mean': pf['original']['mean'],
                'improvement': pf['difference']
            },
            'artifact_test': {
                'clean_accuracy': af['accuracy_clean'],
                'artifact_accuracy': af['accuracy_with_artifact'],
                'confusion_rate': af['confusion_rate']
            },
            'error_analysis': {
                'accuracy': ea['accuracy'],
                'total_errors': ea['errors'],
                'error_patterns': ea['error_patterns']
            }
        }, f, indent=2)

    print(f"\nResultados guardados en: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
