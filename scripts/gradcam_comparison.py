#!/usr/bin/env python3
"""
Sesion 22: Grad-CAM para comparar atencion de modelos warpeado vs original.

Demuestra visualmente que:
- Modelo original: mira artefactos (flechas, equipos medicos, bordes)
- Modelo warpeado: mira patrones pulmonares reales
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class GradCAM:
    """Implementacion de Grad-CAM para ResNet-18."""

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
        """
        Genera mapa de calor Grad-CAM.

        Args:
            input_tensor: Tensor de entrada (1, 3, H, W)
            target_class: Clase objetivo (None = clase predicha)

        Returns:
            heatmap: Mapa de calor normalizado (H, W)
            pred_class: Clase predicha
            confidence: Confianza de la prediccion
        """
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


def load_model_warped(model_path, device):
    """Carga modelo entrenado en dataset warpeado."""
    checkpoint = torch.load(model_path, map_location=device)

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


def load_model_original(model_path, device):
    """Carga modelo entrenado en imagenes originales."""
    checkpoint = torch.load(model_path, map_location=device)

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


def get_transform_warped():
    """Transform para imagenes warpeadas (grayscale -> RGB)."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_transform_original():
    """Transform para imagenes originales."""
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
    # Redimensionar heatmap al tamano de la imagen
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Aplicar colormap
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]  # RGB
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Convertir imagen a RGB si es grayscale
    if len(image.shape) == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image

    # Asegurar que imagen esta en rango [0, 255]
    if image_rgb.max() <= 1:
        image_rgb = (image_rgb * 255).astype(np.uint8)

    # Superponer
    overlay = (alpha * heatmap_colored + (1 - alpha) * image_rgb).astype(np.uint8)

    return overlay


def generate_comparison(image_name, category, output_dir, device,
                        model_warped, model_original, class_names):
    """Genera comparacion Grad-CAM para una imagen."""

    # Rutas
    warped_path = PROJECT_ROOT / f'outputs/warped_dataset/test/{category}/{image_name}_warped.png'
    original_path = PROJECT_ROOT / f'data/dataset/{category}/{image_name}.png'

    if not warped_path.exists() or not original_path.exists():
        print(f"  Imagen no encontrada: {image_name}")
        return None

    # Cargar imagenes
    img_warped = Image.open(warped_path)
    img_original = Image.open(original_path).convert('RGB')

    # Arrays para visualizacion
    img_warped_np = np.array(img_warped)
    img_original_np = np.array(img_original)

    # Transforms
    transform_warped = get_transform_warped()
    transform_original = get_transform_original()

    tensor_warped = transform_warped(img_warped).unsqueeze(0).to(device)
    tensor_original = transform_original(img_original).unsqueeze(0).to(device)

    # Grad-CAM para modelo warpeado (ultima capa convolucional: layer4)
    gradcam_warped = GradCAM(model_warped, model_warped.layer4[-1])
    heatmap_warped, pred_warped, conf_warped = gradcam_warped(tensor_warped)

    # Grad-CAM para modelo original
    gradcam_original = GradCAM(model_original, model_original.layer4[-1])
    heatmap_original, pred_original, conf_original = gradcam_original(tensor_original)

    # Superponer heatmaps
    overlay_warped = overlay_heatmap(img_warped_np, heatmap_warped)
    overlay_original = overlay_heatmap(img_original_np, heatmap_original)

    # Crear figura comparativa
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Fila 1: Modelo Original
    axes[0, 0].imshow(img_original_np)
    axes[0, 0].set_title('Original: Imagen', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(heatmap_original, cmap='jet')
    axes[0, 1].set_title('Original: Heatmap', fontsize=12)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(overlay_original)
    pred_name_orig = class_names[pred_original]
    axes[0, 2].set_title(f'Original: {pred_name_orig} ({conf_original*100:.1f}%)', fontsize=12)
    axes[0, 2].axis('off')

    # Fila 2: Modelo Warpeado
    axes[1, 0].imshow(img_warped_np, cmap='gray')
    axes[1, 0].set_title('Warpeado: Imagen', fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(heatmap_warped, cmap='jet')
    axes[1, 1].set_title('Warpeado: Heatmap', fontsize=12)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(overlay_warped)
    pred_name_warp = class_names[pred_warped]
    axes[1, 2].set_title(f'Warpeado: {pred_name_warp} ({conf_warped*100:.1f}%)', fontsize=12)
    axes[1, 2].axis('off')

    # Titulo general
    gt_label = category.replace('_', ' ')
    fig.suptitle(f'Grad-CAM: {image_name}\nGround Truth: {gt_label}', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Guardar
    save_path = output_dir / f'gradcam_{image_name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'image': image_name,
        'category': category,
        'pred_original': class_names[pred_original],
        'conf_original': conf_original,
        'pred_warped': class_names[pred_warped],
        'conf_warped': conf_warped,
        'correct_original': class_names[pred_original] == category,
        'correct_warped': class_names[pred_warped] == category,
    }


def main():
    import pandas as pd

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # Crear directorio de salida
    output_dir = PROJECT_ROOT / 'outputs/gradcam'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar modelos
    print("\nCargando modelos...")
    model_warped, class_names = load_model_warped(
        PROJECT_ROOT / 'outputs/classifier/best_classifier.pt', device
    )
    model_original, _ = load_model_original(
        PROJECT_ROOT / 'outputs/classifier_original/best_classifier_original.pt', device
    )
    print(f"Clases: {class_names}")

    # Leer imagenes de test
    test_csv = PROJECT_ROOT / 'outputs/warped_dataset/test/images.csv'
    df = pd.read_csv(test_csv)

    # Seleccionar ejemplos representativos de cada clase
    # Priorizar casos donde los modelos difieren o hay errores
    selected = []

    for category in ['COVID', 'Normal', 'Viral_Pneumonia']:
        cat_samples = df[df['category'] == category]['image_name'].tolist()
        # Tomar hasta 4 de cada categoria
        selected.extend([(name, category) for name in cat_samples[:4]])

    print(f"\nGenerando Grad-CAM para {len(selected)} imagenes...")

    results = []
    for i, (image_name, category) in enumerate(selected):
        print(f"  [{i+1}/{len(selected)}] {image_name} ({category})")
        result = generate_comparison(
            image_name, category, output_dir, device,
            model_warped, model_original, class_names
        )
        if result:
            results.append(result)

    # Crear grilla resumen
    print("\nGenerando grilla resumen...")
    create_summary_grid(selected[:12], output_dir, device,
                        model_warped, model_original, class_names)

    # Guardar resultados
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'gradcam_results.csv', index=False)

    # Estadisticas
    print("\n" + "="*60)
    print("ESTADISTICAS GRAD-CAM")
    print("="*60)

    correct_orig = sum(r['correct_original'] for r in results)
    correct_warp = sum(r['correct_warped'] for r in results)
    total = len(results)

    print(f"Imagenes analizadas: {total}")
    print(f"Accuracy Original: {correct_orig}/{total} ({100*correct_orig/total:.1f}%)")
    print(f"Accuracy Warpeado: {correct_warp}/{total} ({100*correct_warp/total:.1f}%)")

    # Casos donde difieren
    differs = [r for r in results if r['correct_original'] != r['correct_warped']]
    print(f"\nCasos donde difieren: {len(differs)}")
    for r in differs:
        print(f"  {r['image']} ({r['category']}): "
              f"Orig={r['pred_original']}({'OK' if r['correct_original'] else 'ERR'}), "
              f"Warp={r['pred_warped']}({'OK' if r['correct_warped'] else 'ERR'})")

    print(f"\nResultados guardados en: {output_dir}")
    print("="*60)


def create_summary_grid(samples, output_dir, device, model_warped, model_original, class_names):
    """Crea grilla resumen de Grad-CAM."""

    n_samples = min(len(samples), 12)
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(16, 4 * n_rows))

    transform_warped = get_transform_warped()
    transform_original = get_transform_original()

    for idx, (image_name, category) in enumerate(samples[:n_samples]):
        col = idx % n_cols
        row_orig = (idx // n_cols) * 2
        row_warp = row_orig + 1

        # Rutas
        warped_path = PROJECT_ROOT / f'outputs/warped_dataset/test/{category}/{image_name}_warped.png'
        original_path = PROJECT_ROOT / f'data/dataset/{category}/{image_name}.png'

        if not warped_path.exists() or not original_path.exists():
            continue

        # Cargar imagenes
        img_warped = Image.open(warped_path)
        img_original = Image.open(original_path).convert('RGB')

        img_warped_np = np.array(img_warped)
        img_original_np = np.array(img_original)

        # Tensors
        tensor_warped = transform_warped(img_warped).unsqueeze(0).to(device)
        tensor_original = transform_original(img_original).unsqueeze(0).to(device)

        # Grad-CAM
        gradcam_warped = GradCAM(model_warped, model_warped.layer4[-1])
        heatmap_warped, pred_warped, conf_warped = gradcam_warped(tensor_warped)

        gradcam_original = GradCAM(model_original, model_original.layer4[-1])
        heatmap_original, pred_original, conf_original = gradcam_original(tensor_original)

        # Overlays
        overlay_warped = overlay_heatmap(img_warped_np, heatmap_warped)
        overlay_original = overlay_heatmap(img_original_np, heatmap_original)

        # Plot
        ax_orig = axes[row_orig, col] if n_rows > 1 else axes[row_orig] if n_cols > 1 else axes
        ax_warp = axes[row_warp, col] if n_rows > 1 else axes[row_warp] if n_cols > 1 else axes

        ax_orig.imshow(overlay_original)
        pred_orig_name = class_names[pred_original]
        color_orig = 'green' if pred_orig_name == category else 'red'
        ax_orig.set_title(f'ORIG: {pred_orig_name}\n({conf_original*100:.0f}%)',
                         fontsize=9, color=color_orig)
        ax_orig.axis('off')

        ax_warp.imshow(overlay_warped)
        pred_warp_name = class_names[pred_warped]
        color_warp = 'green' if pred_warp_name == category else 'red'
        ax_warp.set_title(f'WARP: {pred_warp_name}\n({conf_warped*100:.0f}%)',
                         fontsize=9, color=color_warp)
        ax_warp.axis('off')

        # Etiqueta de GT en el lado
        if col == 0:
            ax_orig.set_ylabel(f'GT: {category}', fontsize=10, rotation=0, labelpad=50)

    # Ocultar ejes vacios
    for idx in range(n_samples, n_rows * n_cols):
        col = idx % n_cols
        row_orig = (idx // n_cols) * 2
        row_warp = row_orig + 1
        if n_rows > 1:
            axes[row_orig, col].axis('off')
            axes[row_warp, col].axis('off')

    plt.suptitle('Grad-CAM: Original (arriba) vs Warpeado (abajo)\n'
                 'Verde=Correcto, Rojo=Error', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / 'gradcam_summary_grid.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grilla guardada en: {save_path}")


if __name__ == '__main__':
    main()
