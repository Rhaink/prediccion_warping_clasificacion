#!/usr/bin/env python3
"""
Sesión 31: Cross-Evaluation Multi-Arquitectura

Evalúa cada modelo entrenado en AMBOS datasets (original y warped) para medir
generalización de cada arquitectura con diferentes margins.

Autor: Proyecto Tesis Maestría
Fecha: 29-Nov-2024
Sesión: 31
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image
from collections import defaultdict
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import argparse

# Configuración
BATCH_SIZE = 32
SEED = 42

# Paths
WARPED_105_DIR = PROJECT_ROOT / "outputs" / "full_warped_dataset"
WARPED_125_DIR = PROJECT_ROOT / "outputs" / "session31_multi_arch" / "datasets" / "full_warped_margin125"
ORIGINAL_DATASET_DIR = PROJECT_ROOT / "data" / "dataset" / "COVID-19_Radiography_Dataset"
MODELS_DIR = PROJECT_ROOT / "outputs" / "session31_multi_arch" / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "session31_multi_arch" / "cross_evaluation"

CLASSES = ['COVID', 'Normal', 'Viral Pneumonia']
CLASS_MAPPING = {'COVID': 'COVID', 'Normal': 'Normal', 'Viral Pneumonia': 'Viral_Pneumonia'}


class GrayscaleToRGB:
    def __call__(self, img):
        if img.mode == 'RGB':
            return img
        return img.convert('RGB')


class OriginalDataset(Dataset):
    """Dataset para imágenes originales."""
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


class WarpedDataset(Dataset):
    """Wrapper para ImageFolder."""
    def __init__(self, image_folder_dataset):
        self.dataset = image_folder_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label


def get_eval_transform():
    return transforms.Compose([
        GrayscaleToRGB(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def create_original_test_split(seed=42, train_ratio=0.75, val_ratio=0.15):
    """Crear split de test idéntico al usado en entrenamiento."""
    np.random.seed(seed)

    all_images = []
    for class_name in CLASSES:
        class_dir = ORIGINAL_DATASET_DIR / class_name / "images"
        if class_dir.exists():
            images = list(class_dir.glob("*.png"))
            mapped_class = CLASS_MAPPING[class_name]
            all_images.extend([(img, mapped_class) for img in images])

    by_class = defaultdict(list)
    for path, class_name in all_images:
        by_class[class_name].append(path)

    test_paths = []
    for class_name, paths in sorted(by_class.items()):
        np.random.shuffle(paths)
        n = len(paths)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        test_paths.extend([(p, class_name) for p in paths[n_train+n_val:]])

    np.random.shuffle(test_paths)
    return test_paths


def create_model(arch_name, num_classes=3):
    """Crea modelo según arquitectura (sin pesos)."""
    if arch_name == 'alexnet':
        model = models.alexnet(weights=None)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    elif arch_name == 'mobilenet':
        model = models.mobilenet_v2(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    elif arch_name == 'efficientnet':
        model = models.efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    elif arch_name == 'densenet':
        model = models.densenet121(weights=None)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    elif arch_name == 'resnet18':
        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    else:
        raise ValueError(f"Arquitectura no soportada: {arch_name}")
    return model


def evaluate_model(model, dataloader, device, class_names):
    """Evalúa un modelo y retorna métricas."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1_macro = f1_score(all_labels, all_preds, average='macro') * 100
    cm = confusion_matrix(all_labels, all_preds)

    class_accuracies = {}
    for idx, name in enumerate(class_names):
        class_acc = cm[idx, idx] / cm[idx].sum() * 100 if cm[idx].sum() > 0 else 0
        class_accuracies[name] = class_acc

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'confusion_matrix': cm.tolist(),
        'class_accuracies': class_accuracies,
        'n_samples': len(all_labels)
    }


def main():
    parser = argparse.ArgumentParser(description='Cross-Evaluation Multi-Arquitectura')
    parser.add_argument('--arch', type=str, default='all',
                       choices=['all', 'alexnet', 'mobilenet', 'efficientnet', 'densenet'])
    args = parser.parse_args()

    print("="*70)
    print("SESIÓN 31: CROSS-EVALUATION MULTI-ARQUITECTURA")
    print("="*70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eval_transform = get_eval_transform()
    class_names = ['COVID', 'Normal', 'Viral_Pneumonia']

    # Preparar datasets de test
    print("\n1. Preparando datasets de test...")

    # Original test set
    original_test_paths = create_original_test_split(seed=SEED)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    original_test_indexed = [(p, class_to_idx[c]) for p, c in original_test_paths]
    original_test_dataset = OriginalDataset(original_test_indexed, transform=eval_transform)
    original_test_loader = DataLoader(original_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"   Test Original: {len(original_test_dataset)} imágenes")

    # Warped 1.05 test set
    warped_105_test = datasets.ImageFolder(WARPED_105_DIR / 'test', transform=eval_transform)
    warped_105_loader = DataLoader(WarpedDataset(warped_105_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"   Test Warped 1.05: {len(warped_105_test)} imágenes")

    # Warped 1.25 test set
    if (WARPED_125_DIR / 'test').exists():
        warped_125_test = datasets.ImageFolder(WARPED_125_DIR / 'test', transform=eval_transform)
        warped_125_loader = DataLoader(WarpedDataset(warped_125_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        print(f"   Test Warped 1.25: {len(warped_125_test)} imágenes")
    else:
        warped_125_loader = None
        print("   Test Warped 1.25: NO DISPONIBLE")

    # Definir arquitecturas
    if args.arch == 'all':
        architectures = ['alexnet', 'mobilenet', 'efficientnet', 'densenet']
    else:
        architectures = [args.arch]

    dataset_types = ['original', 'warped_105', 'warped_125']
    test_loaders = {
        'original': original_test_loader,
        'warped_105': warped_105_loader,
        'warped_125': warped_125_loader
    }

    # Ejecutar cross-evaluation
    print("\n2. Ejecutando cross-evaluation...")

    all_results = {}

    for arch in architectures:
        print(f"\n{'='*60}")
        print(f"ARQUITECTURA: {arch.upper()}")
        print(f"{'='*60}")

        arch_results = {}

        for train_dataset in dataset_types:
            model_path = MODELS_DIR / f'{arch}_{train_dataset}_best.pt'

            if not model_path.exists():
                print(f"   SKIP: {model_path.name} no existe")
                continue

            print(f"\n   Modelo: {arch}_{train_dataset}")

            # Cargar modelo
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model = create_model(arch, num_classes=3)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()

            model_results = {}

            for test_name, test_loader in test_loaders.items():
                if test_loader is None:
                    continue

                result = evaluate_model(model, test_loader, device, class_names)
                model_results[f'test_{test_name}'] = result
                print(f"      → Test {test_name}: {result['accuracy']:.2f}%")

            arch_results[f'train_{train_dataset}'] = model_results

        all_results[arch] = arch_results

    # Calcular gaps de generalización
    print("\n" + "="*70)
    print("3. ANÁLISIS DE GENERALIZACIÓN")
    print("="*70)

    summary_data = []

    for arch, arch_results in all_results.items():
        print(f"\n{arch.upper()}:")

        for train_dataset, model_results in arch_results.items():
            # Baseline: accuracy en su propio test set
            if train_dataset == 'train_original':
                baseline_key = 'test_original'
            elif train_dataset == 'train_warped_105':
                baseline_key = 'test_warped_105'
            else:
                baseline_key = 'test_warped_125'

            if baseline_key not in model_results:
                continue

            baseline_acc = model_results[baseline_key]['accuracy']

            # Cross-evaluation
            for test_name, test_result in model_results.items():
                if test_name == baseline_key:
                    continue

                cross_acc = test_result['accuracy']
                gap = baseline_acc - cross_acc

                summary_data.append({
                    'arch': arch,
                    'train_dataset': train_dataset.replace('train_', ''),
                    'test_dataset': test_name.replace('test_', ''),
                    'accuracy': cross_acc,
                    'baseline': baseline_acc,
                    'gap': gap
                })

                print(f"   {train_dataset} → {test_name}: {cross_acc:.2f}% (gap: {gap:+.2f}%)")

    # Guardar resultados
    print("\n4. Guardando resultados...")

    results_json = {
        'experiment': 'Cross-Evaluation Multi-Arquitectura Session 31',
        'timestamp': datetime.now().isoformat(),
        'detailed_results': {},
        'summary': summary_data
    }

    # Convertir results para JSON
    for arch, arch_results in all_results.items():
        results_json['detailed_results'][arch] = {}
        for train_ds, model_results in arch_results.items():
            results_json['detailed_results'][arch][train_ds] = model_results

    json_path = OUTPUT_DIR / 'cross_evaluation_multi_arch.json'
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"   Resultados guardados: {json_path}")

    # Tabla resumen
    print("\n" + "="*70)
    print("TABLA RESUMEN")
    print("="*70)
    print(f"\n{'Arch':<12} {'Train':<12} {'Test':<12} {'Acc':>8} {'Gap':>8}")
    print("-"*56)

    for row in summary_data:
        print(f"{row['arch']:<12} {row['train_dataset']:<12} {row['test_dataset']:<12} "
              f"{row['accuracy']:>7.2f}% {row['gap']:>+7.2f}%")

    print("\n" + "="*70)
    print("CROSS-EVALUATION COMPLETADO")
    print("="*70)

    return results_json


if __name__ == "__main__":
    main()
