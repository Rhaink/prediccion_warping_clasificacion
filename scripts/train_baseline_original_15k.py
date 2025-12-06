#!/usr/bin/env python3
"""
SESIÓN 28: Baseline con Dataset Original (15K imágenes)

OBJETIVO: Entrenar ResNet-18 con las MISMAS 15K imágenes originales (sin warping)
          para comparación justa con el modelo warped (98.02% accuracy).

METODOLOGÍA:
1. Usar EXACTAMENTE el mismo split train/val/test (seed=42)
2. Misma configuración de entrenamiento que el modelo warped
3. Comparar accuracy, F1, y métricas por clase

HIPÓTESIS:
- Si warped > original: warping ayuda eliminando artefactos
- Si warped ≈ original: no hay diferencia significativa
- Si warped < original: warping pierde información útil

Autor: Proyecto Tesis Maestría
Fecha: 29-Nov-2024
Sesión: 28
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from collections import Counter, defaultdict
import json
import time
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Configuración (IDÉNTICA al modelo warped)
BATCH_SIZE = 32
MAX_EPOCHS = 50
PATIENCE = 10
LEARNING_RATE = 1e-4
SEED = 42

# Paths
ORIGINAL_DATASET_DIR = PROJECT_ROOT / "data" / "dataset" / "COVID-19_Radiography_Dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "session28_baseline_original"

# Clases (mismas que warped)
CLASSES = ['COVID', 'Normal', 'Viral Pneumonia']
CLASS_MAPPING = {
    'COVID': 'COVID',
    'Normal': 'Normal',
    'Viral Pneumonia': 'Viral_Pneumonia'
}


class GrayscaleToRGB:
    def __call__(self, img):
        if img.mode == 'RGB':
            return img
        return img.convert('RGB')


class OriginalDataset(Dataset):
    """Dataset que carga imágenes originales del COVID-19 Radiography Dataset."""

    def __init__(self, image_paths, transform=None):
        """
        image_paths: Lista de tuplas (path, class_idx)
        """
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


def create_splits_identical_to_warped(seed=42, train_ratio=0.75, val_ratio=0.15):
    """
    Crear splits IDÉNTICOS a los usados en generate_full_warped_dataset.py
    Esto garantiza comparación justa.
    """
    np.random.seed(seed)

    all_images = []
    for class_name in CLASSES:
        class_dir = ORIGINAL_DATASET_DIR / class_name / "images"
        if class_dir.exists():
            images = list(class_dir.glob("*.png"))
            mapped_class = CLASS_MAPPING[class_name]
            all_images.extend([(img, mapped_class) for img in images])

    # Agrupar por clase (replicando lógica de generate_full_warped_dataset.py)
    by_class = defaultdict(list)
    for path, class_name in all_images:
        by_class[class_name].append(path)

    splits = {'train': [], 'val': [], 'test': []}

    # Mismo orden de procesamiento que el script original
    for class_name, paths in sorted(by_class.items()):
        np.random.shuffle(paths)
        n = len(paths)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits['train'].extend([(p, class_name) for p in paths[:n_train]])
        splits['val'].extend([(p, class_name) for p in paths[n_train:n_train+n_val]])
        splits['test'].extend([(p, class_name) for p in paths[n_train+n_val:]])

    # Shuffle cada split
    for split in splits.values():
        np.random.shuffle(split)

    return splits


def get_transforms(train=True):
    """Transformaciones IDÉNTICAS al modelo warped."""
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


def get_class_weights(labels, num_classes):
    """Calcula pesos de clase inversamente proporcionales."""
    class_counts = Counter(labels)
    n_samples = len(labels)

    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        weight = n_samples / (num_classes * count)
        weights.append(weight)

    return torch.FloatTensor(weights)


def create_resnet18(num_classes=3):
    """Crea ResNet-18 con transfer learning (IDÉNTICO al modelo warped)."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, num_classes)
    )
    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
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


def main():
    print("="*70)
    print("SESIÓN 28: BASELINE ORIGINAL (15K IMÁGENES)")
    print("="*70)
    print("Objetivo: Entrenar ResNet-18 en imágenes ORIGINALES para comparación")
    print(f"Dataset: {ORIGINAL_DATASET_DIR}")

    # Setup
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Crear splits idénticos al warped
    print("\n1. Creando splits idénticos al dataset warped...")
    splits = create_splits_identical_to_warped(seed=SEED)

    # Mapear clase nombres a índices
    class_names = ['COVID', 'Normal', 'Viral_Pneumonia']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    # Convertir paths a formato (path, class_idx)
    train_paths = [(p, class_to_idx[c]) for p, c in splits['train']]
    val_paths = [(p, class_to_idx[c]) for p, c in splits['val']]
    test_paths = [(p, class_to_idx[c]) for p, c in splits['test']]

    print(f"   Train: {len(train_paths)}")
    print(f"   Val: {len(val_paths)}")
    print(f"   Test: {len(test_paths)}")

    # Distribución por clase en train
    train_labels = [c for _, c in train_paths]
    train_counts = Counter(train_labels)
    print("\n   Distribución en train:")
    for idx, name in enumerate(class_names):
        print(f"      {name}: {train_counts[idx]}")

    # Crear datasets
    train_transform = get_transforms(train=True)
    eval_transform = get_transforms(train=False)

    train_dataset = OriginalDataset(train_paths, transform=train_transform)
    val_dataset = OriginalDataset(val_paths, transform=eval_transform)
    test_dataset = OriginalDataset(test_paths, transform=eval_transform)

    # Calcular pesos de clase
    class_weights = get_class_weights(train_labels, len(class_names))

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Crear modelo
    print("\n2. Creando modelo ResNet-18...")
    model = create_resnet18(num_classes=len(class_names))
    model = model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_acc = 0
    patience_counter = 0
    best_state = None
    best_epoch = 0

    # Entrenamiento
    print("\n3. Entrenando modelo...")
    start_time = time.time()

    for epoch in range(MAX_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        print(f"  Epoch {epoch+1:2d}: Train Loss={train_loss:.4f} Acc={train_acc*100:.2f}% | "
              f"Val Loss={val_loss:.4f} Acc={val_acc*100:.2f}% F1={val_f1*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
            print(f"    -> Nuevo mejor modelo!")
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
    print("\n4. Evaluación en Test...")
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test F1 Macro: {test_f1*100:.2f}%")

    # Matriz de confusión
    cm = confusion_matrix(test_labels, test_preds)
    print("\nMatriz de Confusión:")
    print(f"{'':>12} {'COVID':>8} {'Normal':>8} {'Viral':>8}")
    for idx, name in enumerate(class_names):
        print(f"{name:>12} {cm[idx,0]:>8} {cm[idx,1]:>8} {cm[idx,2]:>8}")

    # Accuracy por clase
    print("\nAccuracy por clase:")
    class_accuracies = {}
    for idx, name in enumerate(class_names):
        class_acc = cm[idx, idx] / cm[idx].sum() * 100 if cm[idx].sum() > 0 else 0
        class_accuracies[name] = class_acc
        print(f"  {name}: {class_acc:.2f}%")

    # GUARDAR MODELO
    checkpoint = {
        'model_state_dict': best_state,
        'class_names': class_names,
        'model_name': 'resnet18_original',
        'val_accuracy': best_val_acc * 100,
        'test_accuracy': test_acc * 100,
        'test_f1': test_f1 * 100,
        'best_epoch': best_epoch,
        'training_time': elapsed,
        'confusion_matrix': cm.tolist(),
        'class_accuracies': class_accuracies,
        'timestamp': datetime.now().isoformat(),
        'seed': SEED,
        'dataset_size': {
            'train': len(train_dataset),
            'val': len(val_dataset),
            'test': len(test_dataset)
        }
    }

    save_path = OUTPUT_DIR / 'resnet18_original_15k_best.pt'
    torch.save(checkpoint, save_path)
    print(f"\n*** MODELO GUARDADO: {save_path} ***")

    # Guardar métricas en JSON
    metrics = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
    metrics_path = OUTPUT_DIR / 'resnet18_original_15k_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Métricas guardadas: {metrics_path}")

    # =====================================================
    # COMPARACIÓN CON MODELO WARPED
    # =====================================================
    print("\n" + "="*70)
    print("COMPARACIÓN: ORIGINAL vs WARPED")
    print("="*70)

    # Cargar métricas del modelo warped
    warped_metrics_path = PROJECT_ROOT / "outputs" / "session27_models" / "resnet18_expanded_15k_metrics.json"
    with open(warped_metrics_path) as f:
        warped_metrics = json.load(f)

    print(f"\n{'Métrica':<25} {'Original':<15} {'Warped':<15} {'Diferencia':<15}")
    print("-" * 70)

    orig_acc = test_acc * 100
    warp_acc = warped_metrics['test_accuracy']
    diff_acc = warp_acc - orig_acc
    sign = "+" if diff_acc > 0 else ""
    print(f"{'Test Accuracy':<25} {orig_acc:<15.2f} {warp_acc:<15.2f} {sign}{diff_acc:<15.2f}")

    orig_f1 = test_f1 * 100
    warp_f1 = warped_metrics['test_f1']
    diff_f1 = warp_f1 - orig_f1
    sign = "+" if diff_f1 > 0 else ""
    print(f"{'Test F1 Macro':<25} {orig_f1:<15.2f} {warp_f1:<15.2f} {sign}{diff_f1:<15.2f}")

    # Conclusión
    print("\n" + "="*70)
    print("CONCLUSIÓN:")
    print("="*70)

    if diff_acc > 2.0:
        conclusion = f"WARPED SUPERIOR: +{diff_acc:.2f}% accuracy"
        print(f"✓ {conclusion}")
        print("  El warping AYUDA eliminando artefactos hospitalarios.")
    elif diff_acc < -2.0:
        conclusion = f"ORIGINAL SUPERIOR: {diff_acc:.2f}% accuracy"
        print(f"⚠️ {conclusion}")
        print("  El warping PIERDE información útil.")
    else:
        conclusion = f"SIN DIFERENCIA SIGNIFICATIVA: {diff_acc:+.2f}% accuracy"
        print(f"= {conclusion}")
        print("  Ambos modelos tienen rendimiento similar.")

    # Guardar comparación
    comparison = {
        'original': {
            'test_accuracy': orig_acc,
            'test_f1': orig_f1,
            'val_accuracy': best_val_acc * 100,
            'class_accuracies': class_accuracies
        },
        'warped': {
            'test_accuracy': warp_acc,
            'test_f1': warp_f1,
            'val_accuracy': warped_metrics['val_accuracy']
        },
        'difference': {
            'test_accuracy': diff_acc,
            'test_f1': diff_f1
        },
        'conclusion': conclusion,
        'timestamp': datetime.now().isoformat()
    }

    comparison_path = OUTPUT_DIR / 'comparison_original_vs_warped.json'
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparación guardada: {comparison_path}")

    return test_acc


if __name__ == "__main__":
    main()
