#!/usr/bin/env python3
"""
Sesión 27: Re-entrenamiento ResNet-18 en Dataset Expandido

OBJETIVO: Recuperar el modelo de 97.76% accuracy que no fue guardado
          en la Sesión 25.

Autor: Proyecto Tesis Maestría
Fecha: 29-Nov-2024
Sesión: 27
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

# Configuración
BATCH_SIZE = 32
MAX_EPOCHS = 50
PATIENCE = 10
LEARNING_RATE = 1e-4
SEED = 42

# Paths
DATASET_DIR = PROJECT_ROOT / "outputs" / "full_warped_dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "session27_models"


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


def create_resnet18(num_classes=3):
    """Crea ResNet-18 con transfer learning."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, num_classes)
    )
    return model


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


def main():
    print("="*70)
    print("SESIÓN 27: RE-ENTRENAMIENTO RESNET-18 EN DATASET EXPANDIDO")
    print("="*70)
    print(f"Objetivo: Recuperar modelo de ~97.76% accuracy")
    print(f"Dataset: {DATASET_DIR}")

    # Setup
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar datasets
    train_dir = DATASET_DIR / 'train'
    val_dir = DATASET_DIR / 'val'
    test_dir = DATASET_DIR / 'test'

    train_transform = get_transforms(train=True)
    eval_transform = get_transforms(train=False)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    class_names = train_dataset.classes
    class_weights = get_class_weights(train_dataset)

    print(f"\nClases: {class_names}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # Distribución por clase
    train_counts = Counter([train_dataset.targets[i] for i in range(len(train_dataset))])
    for idx, name in enumerate(class_names):
        print(f"  {name}: {train_counts[idx]}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Crear modelo
    print(f"\n--- Entrenando ResNet-18 ---")
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
    print("\n--- Evaluación en Test ---")
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
    for idx, name in enumerate(class_names):
        class_acc = cm[idx, idx] / cm[idx].sum() * 100 if cm[idx].sum() > 0 else 0
        print(f"  {name}: {class_acc:.2f}%")

    # GUARDAR MODELO
    checkpoint = {
        'model_state_dict': best_state,
        'class_names': class_names,
        'model_name': 'resnet18',
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

    save_path = OUTPUT_DIR / 'resnet18_expanded_15k_best.pt'
    torch.save(checkpoint, save_path)
    print(f"\n*** MODELO GUARDADO: {save_path} ***")

    # También guardar un JSON con métricas
    metrics_path = OUTPUT_DIR / 'resnet18_expanded_15k_metrics.json'
    metrics = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Métricas guardadas: {metrics_path}")

    print("\n" + "="*70)
    print(f"RESULTADO FINAL: Test Accuracy = {test_acc*100:.2f}%")
    print("="*70)

    return test_acc


if __name__ == "__main__":
    main()
