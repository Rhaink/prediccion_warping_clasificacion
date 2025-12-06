#!/usr/bin/env python3
"""
SESIÓN 29: Test de Robustez a Artefactos de Imagen

OBJETIVO: Simular variabilidad entre hospitales/equipos mediante
          perturbaciones de imagen (ruido, blur, contraste, brillo, compresión).

HIPÓTESIS:
- Si degradación_warped < degradación_original → Warped es más robusto
- Si degradación_warped > degradación_original → Original es más robusto

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
from torchvision.transforms import functional as TF
from PIL import Image, ImageFilter
from collections import defaultdict
import json
from datetime import datetime
from sklearn.metrics import accuracy_score
import io
import warnings
warnings.filterwarnings('ignore')

# Configuración
BATCH_SIZE = 32
SEED = 42

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


class GrayscaleToRGB:
    def __call__(self, img):
        if img.mode == 'RGB':
            return img
        return img.convert('RGB')


class GaussianNoise:
    """Añade ruido gaussiano a la imagen."""
    def __init__(self, std=0.02):
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0, 1)


class AdjustContrast:
    """Ajusta el contraste de la imagen."""
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img):
        return TF.adjust_contrast(img, self.factor)


class AdjustBrightness:
    """Ajusta el brillo de la imagen."""
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img):
        # factor > 0 aumenta brillo, < 0 lo reduce
        return TF.adjust_brightness(img, 1 + self.factor)


class JPEGCompression:
    """Simula compresión JPEG con pérdida de calidad."""
    def __init__(self, quality=50):
        self.quality = quality

    def __call__(self, img):
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')


class SimpleDataset(Dataset):
    """Dataset simple que carga imágenes de una lista de paths."""

    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, class_idx = self.image_paths[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, class_idx


def load_warped_test_set():
    """Carga el test set del dataset warped."""
    test_dir = WARPED_DATASET_DIR / "test"
    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    image_paths = []
    for class_name in CLASS_NAMES:
        class_dir = test_dir / class_name
        if class_dir.exists():
            for img_path in class_dir.glob("*.png"):
                image_paths.append((img_path, class_to_idx[class_name]))

    return image_paths


def load_original_test_set():
    """Carga el test set del dataset original con el mismo split que se usó en entrenamiento."""
    np.random.seed(SEED)

    all_images = []
    for class_name, mapped_name in ORIGINAL_CLASS_MAPPING.items():
        class_dir = ORIGINAL_DATASET_DIR / class_name / "images"
        if class_dir.exists():
            images = list(class_dir.glob("*.png"))
            all_images.extend([(img, mapped_name) for img in images])

    # Agrupar por clase
    by_class = defaultdict(list)
    for path, class_name in all_images:
        by_class[class_name].append(path)

    # Mismo split que en entrenamiento
    train_ratio, val_ratio = 0.75, 0.15
    test_paths = []
    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    for class_name, paths in sorted(by_class.items()):
        np.random.shuffle(paths)
        n = len(paths)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        test_images = paths[n_train + n_val:]
        test_paths.extend([(p, class_to_idx[class_name]) for p in test_images])

    np.random.shuffle(test_paths)
    return test_paths


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
    model.eval()
    return model


def get_base_transform():
    """Transformación base sin perturbaciones."""
    return transforms.Compose([
        GrayscaleToRGB(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_artifact_transforms():
    """Define todas las perturbaciones de artefactos a probar."""
    base_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    perturbations = {}

    # Ruido Gaussiano
    for std, label in [(0.02, 'leve'), (0.05, 'fuerte')]:
        perturbations[f'ruido_gaussiano_{label}'] = transforms.Compose([
            GrayscaleToRGB(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            GaussianNoise(std=std),
            base_normalize
        ])

    # Blur Gaussiano
    for kernel, label in [(3, 'leve'), (5, 'fuerte')]:
        perturbations[f'blur_{label}'] = transforms.Compose([
            GrayscaleToRGB(),
            transforms.Resize((224, 224)),
            transforms.GaussianBlur(kernel_size=kernel, sigma=(kernel/3, kernel/3)),
            transforms.ToTensor(),
            base_normalize
        ])

    # Contraste
    for factor, label in [(0.7, 'bajo'), (1.3, 'alto')]:
        perturbations[f'contraste_{label}'] = transforms.Compose([
            GrayscaleToRGB(),
            transforms.Resize((224, 224)),
            AdjustContrast(factor),
            transforms.ToTensor(),
            base_normalize
        ])

    # Brillo
    for factor, label in [(-0.1, 'bajo'), (0.1, 'alto')]:
        perturbations[f'brillo_{label}'] = transforms.Compose([
            GrayscaleToRGB(),
            transforms.Resize((224, 224)),
            AdjustBrightness(factor),
            transforms.ToTensor(),
            base_normalize
        ])

    # Compresión JPEG
    perturbations['jpeg_q50'] = transforms.Compose([
        GrayscaleToRGB(),
        transforms.Resize((224, 224)),
        JPEGCompression(quality=50),
        transforms.ToTensor(),
        base_normalize
    ])

    perturbations['jpeg_q30'] = transforms.Compose([
        GrayscaleToRGB(),
        transforms.Resize((224, 224)),
        JPEGCompression(quality=30),
        transforms.ToTensor(),
        base_normalize
    ])

    # Combinación: ruido + blur
    perturbations['ruido_blur_combo'] = transforms.Compose([
        GrayscaleToRGB(),
        transforms.Resize((224, 224)),
        transforms.GaussianBlur(kernel_size=3, sigma=1.0),
        transforms.ToTensor(),
        GaussianNoise(std=0.02),
        base_normalize
    ])

    return perturbations


def evaluate_model(model, dataloader, device):
    """Evalúa el modelo y devuelve accuracy."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds) * 100


def run_artifact_robustness_test():
    """Ejecuta el test de robustez a artefactos de imagen."""
    print("=" * 80)
    print("SESIÓN 29: TEST DE ROBUSTEZ A ARTEFACTOS DE IMAGEN")
    print("=" * 80)
    print("Simulando variabilidad entre hospitales/equipos: ORIGINAL vs WARPED")
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

    # Cargar test sets
    print("\n2. Cargando test sets...")
    original_test_paths = load_original_test_set()
    warped_test_paths = load_warped_test_set()
    print(f"   Original test set: {len(original_test_paths)} imágenes")
    print(f"   Warped test set: {len(warped_test_paths)} imágenes")

    # Evaluar baselines (sin perturbaciones)
    print("\n3. Evaluando baselines (sin perturbaciones)...")
    base_transform = get_base_transform()

    original_dataset_base = SimpleDataset(original_test_paths, transform=base_transform)
    warped_dataset_base = SimpleDataset(warped_test_paths, transform=base_transform)

    original_loader_base = DataLoader(original_dataset_base, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    warped_loader_base = DataLoader(warped_dataset_base, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    baseline_original = evaluate_model(model_original, original_loader_base, device)
    baseline_warped = evaluate_model(model_warped, warped_loader_base, device)

    print(f"   Baseline Original: {baseline_original:.2f}%")
    print(f"   Baseline Warped: {baseline_warped:.2f}%")

    # Obtener perturbaciones
    perturbations = get_artifact_transforms()

    # Evaluar con cada perturbación
    print("\n4. Evaluando con artefactos de imagen...")
    print("-" * 85)

    results = {
        'baseline': {
            'original': baseline_original,
            'warped': baseline_warped
        },
        'perturbations': {}
    }

    header = f"{'Perturbación':<25} {'Orig Acc':<12} {'Warp Acc':<12} {'Orig Deg':<12} {'Warp Deg':<12} {'Ganador':<10}"
    print(header)
    print("-" * 85)

    wins_warped = 0
    wins_original = 0
    ties = 0

    for pert_name, pert_transform in perturbations.items():
        # Crear datasets con perturbación
        original_dataset_pert = SimpleDataset(original_test_paths, transform=pert_transform)
        warped_dataset_pert = SimpleDataset(warped_test_paths, transform=pert_transform)

        original_loader_pert = DataLoader(original_dataset_pert, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        warped_loader_pert = DataLoader(warped_dataset_pert, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        # Evaluar
        acc_original = evaluate_model(model_original, original_loader_pert, device)
        acc_warped = evaluate_model(model_warped, warped_loader_pert, device)

        # Calcular degradación (baseline - perturbado)
        deg_original = baseline_original - acc_original
        deg_warped = baseline_warped - acc_warped

        # Determinar ganador (menor degradación = más robusto)
        if abs(deg_original - deg_warped) < 0.5:  # Empate si diferencia < 0.5%
            winner = "EMPATE"
            ties += 1
        elif deg_warped < deg_original:
            winner = "WARPED"
            wins_warped += 1
        else:
            winner = "ORIGINAL"
            wins_original += 1

        results['perturbations'][pert_name] = {
            'acc_original': acc_original,
            'acc_warped': acc_warped,
            'degradation_original': deg_original,
            'degradation_warped': deg_warped,
            'winner': winner
        }

        print(f"{pert_name:<25} {acc_original:<12.2f} {acc_warped:<12.2f} {deg_original:<12.2f} {deg_warped:<12.2f} {winner:<10}")

    # Resumen
    print("\n" + "=" * 85)
    print("RESUMEN")
    print("=" * 85)

    total = len(perturbations)
    print(f"\nVictorias WARPED: {wins_warped}/{total} ({wins_warped/total*100:.1f}%)")
    print(f"Victorias ORIGINAL: {wins_original}/{total} ({wins_original/total*100:.1f}%)")
    print(f"Empates: {ties}/{total} ({ties/total*100:.1f}%)")

    # Calcular degradación promedio
    avg_deg_original = np.mean([r['degradation_original'] for r in results['perturbations'].values()])
    avg_deg_warped = np.mean([r['degradation_warped'] for r in results['perturbations'].values()])

    print(f"\nDegradación promedio:")
    print(f"   Original: {avg_deg_original:.2f}%")
    print(f"   Warped: {avg_deg_warped:.2f}%")

    # Conclusión
    print("\n" + "=" * 85)
    print("CONCLUSIÓN")
    print("=" * 85)

    if wins_warped >= 7:
        conclusion = "HIPÓTESIS CONFIRMADA: Warped es MÁS ROBUSTO a artefactos (gana en ≥70% de perturbaciones)"
        hypothesis_confirmed = True
    elif wins_original >= 7:
        conclusion = "HIPÓTESIS REFUTADA: Original es MÁS ROBUSTO a artefactos (gana en ≥70% de perturbaciones)"
        hypothesis_confirmed = False
    else:
        if avg_deg_warped < avg_deg_original:
            conclusion = f"RESULTADO MIXTO: Warped degrada {avg_deg_original - avg_deg_warped:.2f}% menos en promedio"
            hypothesis_confirmed = "partial"
        else:
            conclusion = f"RESULTADO MIXTO: Original degrada {avg_deg_warped - avg_deg_original:.2f}% menos en promedio"
            hypothesis_confirmed = "partial"

    print(conclusion)

    # Guardar resultados
    results['summary'] = {
        'wins_warped': wins_warped,
        'wins_original': wins_original,
        'ties': ties,
        'total_perturbations': total,
        'avg_degradation_original': avg_deg_original,
        'avg_degradation_warped': avg_deg_warped,
        'conclusion': conclusion,
        'hypothesis_confirmed': hypothesis_confirmed,
        'timestamp': datetime.now().isoformat()
    }

    results_path = OUTPUT_DIR / 'artifact_robustness_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados guardados: {results_path}")

    return results


if __name__ == "__main__":
    run_artifact_robustness_test()
