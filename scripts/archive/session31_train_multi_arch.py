#!/usr/bin/env python3
"""
Sesión 31: Entrenamiento Multi-Arquitectura

Este script entrena múltiples arquitecturas (AlexNet, MobileNetV2, EfficientNet-B0, DenseNet-121)
en tres datasets: Original 15K, Warped 1.05, Warped 1.25.

Configuración idéntica a ResNet-18:
- BATCH_SIZE = 32
- MAX_EPOCHS = 50
- PATIENCE = 10
- LEARNING_RATE = 1e-4
- optimizer = AdamW(weight_decay=0.01)
- scheduler = ReduceLROnPlateau(mode='max', factor=0.5, patience=5)

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
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from collections import Counter
import json
import time
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import argparse

# Configuración
BATCH_SIZE = 32
MAX_EPOCHS = 50
PATIENCE = 10
LEARNING_RATE = 1e-4
SEED = 42

# Paths
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "session31_multi_arch" / "models"

# Dataset paths
DATASETS = {
    'original': PROJECT_ROOT / "data" / "dataset" / "COVID-19_Radiography_Dataset",
    'warped_105': PROJECT_ROOT / "outputs" / "full_warped_dataset",
    'warped_125': PROJECT_ROOT / "outputs" / "session31_multi_arch" / "datasets" / "full_warped_margin125"
}


class GrayscaleToRGB:
    """Convierte imagen grayscale a RGB (3 canales)."""
    def __call__(self, img):
        if img.mode == 'RGB':
            return img
        return img.convert('RGB')


def get_transforms(train=True):
    """Transformaciones para train/val/test."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if train:
        return transforms.Compose([
            GrayscaleToRGB(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)
            ),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            GrayscaleToRGB(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])


def get_class_weights(dataset):
    """Calcula pesos de clase inversamente proporcionales."""
    labels = [dataset.targets[i] for i in range(len(dataset))]
    class_counts = Counter(labels)
    n_samples = len(labels)
    n_classes = len(class_counts)

    weights = []
    for i in range(n_classes):
        count = class_counts.get(i, 1)
        weight = n_samples / (n_classes * count)
        weights.append(weight)

    return torch.FloatTensor(weights)


def create_model(arch_name, num_classes=3):
    """Crea el modelo según la arquitectura especificada."""
    if arch_name == 'alexnet':
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    elif arch_name == 'mobilenet':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    elif arch_name == 'efficientnet':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    elif arch_name == 'densenet':
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    elif arch_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    else:
        raise ValueError(f"Arquitectura no soportada: {arch_name}")

    return model


def load_datasets(dataset_type):
    """Carga los datasets según el tipo."""
    base_path = DATASETS[dataset_type]

    if dataset_type == 'original':
        # Para el dataset original, necesitamos crear los splits manualmente
        # usando la misma semilla que warped para asegurar consistencia
        train_dir = base_path
        val_dir = base_path
        test_dir = base_path

        # Crear dataset con splits manuales (75/15/10)
        return load_original_dataset(base_path)
    else:
        # Para warped datasets, ya tienen la estructura train/val/test
        train_dir = base_path / 'train'
        val_dir = base_path / 'val'
        test_dir = base_path / 'test'

        train_transform = get_transforms(train=True)
        eval_transform = get_transforms(train=False)

        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
        test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

        return train_dataset, val_dataset, test_dataset


def load_original_dataset(base_path):
    """
    Carga el dataset original con splits idénticos a los warped.
    Usa la misma semilla y proporciones para consistencia.
    """
    from collections import defaultdict
    from torch.utils.data import Subset

    # Cargar todas las imágenes
    classes = ['COVID', 'Normal', 'Viral Pneumonia']
    class_mapping = {'COVID': 'COVID', 'Normal': 'Normal', 'Viral Pneumonia': 'Viral_Pneumonia'}

    all_images = []
    for class_name in classes:
        class_dir = base_path / class_name / "images"
        if class_dir.exists():
            images = list(class_dir.glob("*.png"))
            mapped_class = class_mapping[class_name]
            all_images.extend([(str(img), mapped_class) for img in images])

    # Crear splits con seed=42
    np.random.seed(42)

    by_class = defaultdict(list)
    for path, class_name in all_images:
        by_class[class_name].append(path)

    splits = {'train': [], 'val': [], 'test': []}
    train_ratio, val_ratio = 0.75, 0.15

    for class_name, paths in by_class.items():
        np.random.shuffle(paths)
        n = len(paths)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits['train'].extend([(p, class_name) for p in paths[:n_train]])
        splits['val'].extend([(p, class_name) for p in paths[n_train:n_train+n_val]])
        splits['test'].extend([(p, class_name) for p in paths[n_train+n_val:]])

    # Shuffle each split
    for split in splits.values():
        np.random.shuffle(split)

    # Create custom datasets
    train_transform = get_transforms(train=True)
    eval_transform = get_transforms(train=False)

    train_dataset = OriginalDataset(splits['train'], transform=train_transform)
    val_dataset = OriginalDataset(splits['val'], transform=eval_transform)
    test_dataset = OriginalDataset(splits['test'], transform=eval_transform)

    return train_dataset, val_dataset, test_dataset


class OriginalDataset(torch.utils.data.Dataset):
    """Dataset personalizado para las imágenes originales."""
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform
        self.classes = ['COVID', 'Normal', 'Viral_Pneumonia']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.targets = [self.class_to_idx[c] for _, c in image_list]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path, class_name = self.image_list[idx]
        from PIL import Image
        img = Image.open(img_path).convert('L')  # Grayscale

        if self.transform:
            img = self.transform(img)

        label = self.class_to_idx[class_name]
        return img, label


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entrena una época."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    """Evalúa el modelo."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(all_labels)
    epoch_acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    return epoch_loss, epoch_acc, f1_macro, all_preds, all_labels


def train_model(arch_name, dataset_type, device):
    """Entrena un modelo específico en un dataset específico."""
    print(f"\n{'='*60}")
    print(f"ENTRENANDO: {arch_name.upper()} en {dataset_type.upper()}")
    print(f"{'='*60}")

    # Cargar datasets
    train_dataset, val_dataset, test_dataset = load_datasets(dataset_type)

    class_names = ['COVID', 'Normal', 'Viral_Pneumonia']
    class_weights = get_class_weights(train_dataset)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Crear modelo
    model = create_model(arch_name, num_classes=len(class_names))
    model = model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_acc = 0
    patience_counter = 0
    best_state = None
    best_epoch = 0

    start_time = time.time()

    for epoch in range(MAX_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        print(f"  Epoch {epoch+1:2d}: Train Acc={train_acc*100:.2f}% | "
              f"Val Acc={val_acc*100:.2f}% F1={val_f1*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping en epoch {epoch+1}")
                break

    elapsed = time.time() - start_time
    print(f"\nEntrenamiento completado en {elapsed:.1f}s")
    print(f"Mejor modelo en epoch {best_epoch} con val_acc={best_val_acc*100:.2f}%")

    # Cargar mejor modelo
    model.load_state_dict(best_state)

    # Evaluar en test
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\n*** Test Accuracy: {test_acc*100:.2f}% | F1: {test_f1*100:.2f}% ***")

    # Matriz de confusión
    cm = confusion_matrix(test_labels, test_preds)

    # Guardar modelo
    model_name = f"{arch_name}_{dataset_type}"
    checkpoint = {
        'model_state_dict': best_state,
        'class_names': class_names,
        'model_name': arch_name,
        'dataset_type': dataset_type,
        'val_accuracy': best_val_acc * 100,
        'test_accuracy': test_acc * 100,
        'test_f1': test_f1 * 100,
        'best_epoch': best_epoch,
        'training_time': elapsed,
        'confusion_matrix': cm.tolist(),
        'timestamp': datetime.now().isoformat(),
        'seed': SEED,
        'dataset_size': {
            'train': len(train_dataset),
            'val': len(val_dataset),
            'test': len(test_dataset)
        }
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / f'{model_name}_best.pt'
    torch.save(checkpoint, save_path)
    print(f"Modelo guardado: {save_path}")

    # Guardar métricas JSON
    metrics = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
    metrics_path = OUTPUT_DIR / f'{model_name}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return {
        'arch': arch_name,
        'dataset': dataset_type,
        'val_acc': best_val_acc * 100,
        'test_acc': test_acc * 100,
        'test_f1': test_f1 * 100,
        'confusion_matrix': cm.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description='Entrenamiento Multi-Arquitectura Sesión 31')
    parser.add_argument('--arch', type=str, default='all',
                       choices=['all', 'alexnet', 'mobilenet', 'efficientnet', 'densenet', 'resnet18'],
                       help='Arquitectura a entrenar')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['all', 'original', 'warped_105', 'warped_125'],
                       help='Dataset a usar')
    args = parser.parse_args()

    print("="*70)
    print("SESIÓN 31: ENTRENAMIENTO MULTI-ARQUITECTURA")
    print("="*70)

    # Setup
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Definir arquitecturas y datasets a entrenar
    if args.arch == 'all':
        architectures = ['alexnet', 'mobilenet', 'efficientnet', 'densenet']
    else:
        architectures = [args.arch]

    if args.dataset == 'all':
        dataset_types = ['original', 'warped_105', 'warped_125']
    else:
        dataset_types = [args.dataset]

    print(f"\nArquitecturas: {architectures}")
    print(f"Datasets: {dataset_types}")

    # Entrenar todos los modelos
    all_results = []

    for arch in architectures:
        for dataset in dataset_types:
            try:
                result = train_model(arch, dataset, device)
                all_results.append(result)
            except Exception as e:
                print(f"\nERROR entrenando {arch} en {dataset}: {e}")
                import traceback
                traceback.print_exc()

    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL - SESIÓN 31")
    print("="*70)

    # Crear tabla de resultados
    print("\n{:<15} {:<15} {:<12} {:<12} {:<12}".format(
        "Arquitectura", "Dataset", "Val Acc", "Test Acc", "Test F1"))
    print("-"*66)

    for r in all_results:
        print("{:<15} {:<15} {:<12.2f} {:<12.2f} {:<12.2f}".format(
            r['arch'], r['dataset'], r['val_acc'], r['test_acc'], r['test_f1']))

    # Guardar resumen
    summary_path = OUTPUT_DIR.parent / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': all_results
        }, f, indent=2)
    print(f"\nResumen guardado: {summary_path}")

    return all_results


if __name__ == "__main__":
    main()
