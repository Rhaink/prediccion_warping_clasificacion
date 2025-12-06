#!/usr/bin/env python3
"""
SESIÓN 29: Análisis Grad-CAM Comparativo con Pulmonary Focus Score (PFS)

OBJETIVO: Cuantificar que el modelo warped se enfoca más en el parénquima pulmonar.

METODOLOGÍA:
1. Generar Grad-CAM para ambos modelos
2. Cargar máscaras pulmonares del dataset
3. Calcular PFS = (gradcam * mask).sum() / gradcam.sum()
4. Comparar PFS estadísticamente (t-test)

MÉTRICA DE ÉXITO: PFS_warped > PFS_original con p < 0.05

Autor: Proyecto Tesis Maestría
Fecha: 29-Nov-2024
Sesión: 29
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch.nn.functional as F
from PIL import Image
from collections import defaultdict
import json
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuración
BATCH_SIZE = 1  # Para Grad-CAM procesamos de a uno
SEED = 42
NUM_SAMPLES = 300  # Muestras por modelo para análisis estadístico

# Paths
WARPED_DATASET_DIR = PROJECT_ROOT / "outputs" / "full_warped_dataset"
ORIGINAL_DATASET_DIR = PROJECT_ROOT / "data" / "dataset" / "COVID-19_Radiography_Dataset"
WARPED_MODEL_PATH = PROJECT_ROOT / "outputs" / "session27_models" / "resnet18_expanded_15k_best.pt"
ORIGINAL_MODEL_PATH = PROJECT_ROOT / "outputs" / "session28_baseline_original" / "resnet18_original_15k_best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "session29_robustness"

# Clases
CLASS_NAMES = ['COVID', 'Normal', 'Viral_Pneumonia']
ORIGINAL_CLASS_MAPPING = {
    'COVID': 'COVID',
    'Normal': 'Normal',
    'Viral Pneumonia': 'Viral_Pneumonia'
}
REVERSE_CLASS_MAPPING = {v: k for k, v in ORIGINAL_CLASS_MAPPING.items()}


class GrayscaleToRGB:
    def __call__(self, img):
        if img.mode == 'RGB':
            return img
        return img.convert('RGB')


class GradCAM:
    """Implementación de Grad-CAM para ResNet."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Registrar hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """Genera mapa Grad-CAM."""
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()

        # Calcular pesos
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Ponderar activaciones
        cam = torch.sum(self.activations * pooled_gradients, dim=1, keepdim=True)

        # ReLU y normalizar
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze()

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy(), target_class


def create_resnet18(num_classes=3):
    """Crea ResNet-18 con la misma arquitectura que los modelos entrenados."""
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, num_classes)
    )
    return model


def load_model(checkpoint_path, device):
    """Carga un modelo desde un checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_resnet18(num_classes=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    return model


def get_base_transform():
    """Transformación base sin perturbaciones."""
    return transforms.Compose([
        GrayscaleToRGB(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_original_test_set_with_masks():
    """Carga el test set del dataset original junto con las máscaras."""
    np.random.seed(SEED)

    all_images = []
    for class_name, mapped_name in ORIGINAL_CLASS_MAPPING.items():
        class_dir = ORIGINAL_DATASET_DIR / class_name / "images"
        mask_dir = ORIGINAL_DATASET_DIR / class_name / "masks"
        if class_dir.exists():
            for img_path in class_dir.glob("*.png"):
                mask_path = mask_dir / img_path.name
                if mask_path.exists():
                    all_images.append((img_path, mask_path, mapped_name))

    # Agrupar por clase
    by_class = defaultdict(list)
    for img_path, mask_path, class_name in all_images:
        by_class[class_name].append((img_path, mask_path, class_name))

    # Mismo split que en entrenamiento
    train_ratio, val_ratio = 0.75, 0.15
    test_items = []
    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    for class_name, items in sorted(by_class.items()):
        np.random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        test_images = items[n_train + n_val:]
        for img_path, mask_path, cn in test_images:
            test_items.append({
                'image_path': img_path,
                'mask_path': mask_path,
                'class_name': cn,
                'class_idx': class_to_idx[cn]
            })

    np.random.shuffle(test_items)
    return test_items


def load_warped_test_set_with_masks():
    """Carga el test set warped y busca las máscaras correspondientes del original."""
    test_dir = WARPED_DATASET_DIR / "test"
    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    test_items = []
    for class_name in CLASS_NAMES:
        class_dir = test_dir / class_name
        original_class = REVERSE_CLASS_MAPPING.get(class_name, class_name)
        mask_dir = ORIGINAL_DATASET_DIR / original_class / "masks"

        if class_dir.exists():
            for img_path in class_dir.glob("*.png"):
                # El nombre del archivo warped tiene "_warped" al final
                # Necesitamos quitarlo para encontrar la máscara original
                original_name = img_path.stem.replace("_warped", "") + ".png"
                mask_path = mask_dir / original_name
                if mask_path.exists():
                    test_items.append({
                        'image_path': img_path,
                        'mask_path': mask_path,
                        'class_name': class_name,
                        'class_idx': class_to_idx[class_name]
                    })

    return test_items


def calculate_pfs(gradcam, mask):
    """
    Calcula Pulmonary Focus Score.
    PFS = (gradcam * mask).sum() / gradcam.sum()
    """
    gradcam = np.clip(gradcam, 0, 1)

    if gradcam.sum() == 0:
        return 0.0

    # Convertir máscara a escala de grises si es RGB
    if len(mask.shape) == 3:
        mask = np.mean(mask, axis=2)

    # Normalizar máscara a 0-1
    mask = mask / 255.0 if mask.max() > 1 else mask

    # Calcular PFS
    overlap = (gradcam * mask).sum()
    total = gradcam.sum()

    return overlap / total


def run_gradcam_analysis():
    """Ejecuta el análisis Grad-CAM comparativo."""
    print("=" * 80)
    print("SESIÓN 29: ANÁLISIS GRAD-CAM COMPARATIVO")
    print("=" * 80)
    print("Calculando Pulmonary Focus Score (PFS) para comparar enfoque anatómico")
    print()

    # Setup
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar modelos
    print("\n1. Cargando modelos...")
    model_original = load_model(ORIGINAL_MODEL_PATH, device)
    model_warped = load_model(WARPED_MODEL_PATH, device)
    print("   ✓ Modelos cargados")

    # Crear Grad-CAM extractors (última capa convolucional de ResNet)
    gradcam_original = GradCAM(model_original, model_original.layer4[-1])
    gradcam_warped = GradCAM(model_warped, model_warped.layer4[-1])

    # Cargar test sets con máscaras
    print("\n2. Cargando test sets con máscaras...")
    original_test_items = load_original_test_set_with_masks()
    warped_test_items = load_warped_test_set_with_masks()

    # Limitar a NUM_SAMPLES por modelo
    np.random.shuffle(original_test_items)
    np.random.shuffle(warped_test_items)
    original_test_items = original_test_items[:NUM_SAMPLES]
    warped_test_items = warped_test_items[:NUM_SAMPLES]

    print(f"   Original: {len(original_test_items)} imágenes")
    print(f"   Warped: {len(warped_test_items)} imágenes")

    # Transformación
    transform = get_base_transform()

    # Calcular PFS para cada imagen
    print("\n3. Calculando Pulmonary Focus Score...")

    pfs_original_list = []
    pfs_warped_list = []
    pfs_by_class_original = defaultdict(list)
    pfs_by_class_warped = defaultdict(list)

    # PFS para modelo original
    print("   Procesando modelo original...")
    for i, item in enumerate(original_test_items):
        if i % 50 == 0:
            print(f"      {i}/{len(original_test_items)}")

        # Cargar imagen y máscara
        img = Image.open(item['image_path'])
        mask = np.array(Image.open(item['mask_path']).resize((224, 224)))

        # Transformar imagen
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Generar Grad-CAM
        cam, pred_class = gradcam_original.generate(img_tensor)

        # Calcular PFS
        pfs = calculate_pfs(cam, mask)
        pfs_original_list.append(pfs)
        pfs_by_class_original[item['class_name']].append(pfs)

    # PFS para modelo warped
    print("   Procesando modelo warped...")
    for i, item in enumerate(warped_test_items):
        if i % 50 == 0:
            print(f"      {i}/{len(warped_test_items)}")

        # Cargar imagen warped
        img = Image.open(item['image_path'])
        # Cargar máscara original (no warped)
        mask = np.array(Image.open(item['mask_path']).resize((224, 224)))

        # Transformar imagen
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Generar Grad-CAM
        cam, pred_class = gradcam_warped.generate(img_tensor)

        # Calcular PFS
        pfs = calculate_pfs(cam, mask)
        pfs_warped_list.append(pfs)
        pfs_by_class_warped[item['class_name']].append(pfs)

    # Estadísticas
    print("\n4. Análisis Estadístico...")
    print("-" * 80)

    pfs_original = np.array(pfs_original_list)
    pfs_warped = np.array(pfs_warped_list)

    mean_original = np.mean(pfs_original)
    std_original = np.std(pfs_original)
    mean_warped = np.mean(pfs_warped)
    std_warped = np.std(pfs_warped)

    print(f"\nPFS Global:")
    print(f"   Original: {mean_original:.4f} ± {std_original:.4f}")
    print(f"   Warped:   {mean_warped:.4f} ± {std_warped:.4f}")

    # T-test
    t_stat, p_value = stats.ttest_ind(pfs_warped, pfs_original)
    print(f"\nT-test (Warped vs Original):")
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value: {p_value:.6f}")

    # PFS por clase
    print(f"\nPFS por clase:")
    print(f"{'Clase':<20} {'Original':<20} {'Warped':<20}")
    print("-" * 60)
    for class_name in CLASS_NAMES:
        orig_mean = np.mean(pfs_by_class_original.get(class_name, [0]))
        orig_std = np.std(pfs_by_class_original.get(class_name, [0]))
        warp_mean = np.mean(pfs_by_class_warped.get(class_name, [0]))
        warp_std = np.std(pfs_by_class_warped.get(class_name, [0]))
        print(f"{class_name:<20} {orig_mean:.4f} ± {orig_std:.4f}    {warp_mean:.4f} ± {warp_std:.4f}")

    # Conclusión
    print("\n" + "=" * 80)
    print("CONCLUSIÓN")
    print("=" * 80)

    if p_value < 0.05:
        if mean_warped > mean_original:
            conclusion = f"HIPÓTESIS CONFIRMADA: PFS_warped ({mean_warped:.4f}) > PFS_original ({mean_original:.4f}) con p={p_value:.6f}"
            hypothesis_confirmed = True
        else:
            conclusion = f"HIPÓTESIS REFUTADA: PFS_original ({mean_original:.4f}) > PFS_warped ({mean_warped:.4f}) con p={p_value:.6f}"
            hypothesis_confirmed = False
    else:
        conclusion = f"RESULTADO NO SIGNIFICATIVO: p={p_value:.6f} > 0.05"
        hypothesis_confirmed = None

    print(conclusion)

    # Interpretación adicional
    if hypothesis_confirmed == True:
        print("\n-> El modelo warped se enfoca MÁS en el parénquima pulmonar.")
        print("   Esto sugiere que aprende características más anatómicamente relevantes.")
    elif hypothesis_confirmed == False:
        print("\n-> El modelo original se enfoca MÁS en el parénquima pulmonar.")
    else:
        print("\n-> No hay diferencia significativa en el enfoque pulmonar.")

    # Guardar resultados
    results = {
        'pfs_global': {
            'original': {'mean': float(mean_original), 'std': float(std_original)},
            'warped': {'mean': float(mean_warped), 'std': float(std_warped)}
        },
        'pfs_by_class': {
            'original': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in pfs_by_class_original.items()},
            'warped': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in pfs_by_class_warped.items()}
        },
        'statistical_test': {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        },
        'n_samples': {
            'original': len(pfs_original),
            'warped': len(pfs_warped)
        },
        'conclusion': conclusion,
        'hypothesis_confirmed': hypothesis_confirmed,
        'timestamp': datetime.now().isoformat()
    }

    results_path = OUTPUT_DIR / 'gradcam_pfs_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados guardados: {results_path}")

    return results


if __name__ == "__main__":
    run_gradcam_analysis()
