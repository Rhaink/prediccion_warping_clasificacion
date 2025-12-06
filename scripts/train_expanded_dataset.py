#!/usr/bin/env python3
"""
Sesión 25: Entrenamiento en Dataset Expandido (~15K imágenes)

Compara:
- Baseline: 957 imágenes (dataset etiquetado)
- Expandido: ~15K imágenes (landmarks predichos)

Modelos canario: AlexNet + ResNet-18
Margin scale: 1.05 (óptimo encontrado)

Autor: Proyecto Tesis Maestría
Fecha: 29-Nov-2024
Sesión: 25
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
EXPANDED_DATASET_DIR = PROJECT_ROOT / "outputs" / "full_warped_dataset"
BASELINE_DATASET_DIR = PROJECT_ROOT / "outputs" / "margin_experiment" / "margin_1.05"  # Baseline con mismo margen
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "expanded_experiment"


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


def create_model(model_name, num_classes=3):
    """Crea modelo con transfer learning."""
    if model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    elif model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    else:
        raise ValueError(f"Modelo {model_name} no soportado")

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


def train_model(model, train_loader, val_loader, class_weights, device, model_name, max_epochs=50, patience=10):
    """Entrena un modelo completo con early stopping."""
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_acc = 0
    patience_counter = 0
    best_state = None
    best_epoch = 0

    start_time = time.time()

    for epoch in range(max_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"      Early stopping en epoch {epoch+1}")
                break

    # Cargar mejor modelo
    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - start_time
    return model, best_val_acc, best_epoch, elapsed


def run_experiment(dataset_dir, experiment_name, device):
    """Ejecuta experimento completo en un dataset."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO: {experiment_name}")
    print(f"Dataset: {dataset_dir}")
    print(f"{'='*60}")

    # Cargar datasets
    train_dir = dataset_dir / 'train'
    val_dir = dataset_dir / 'val'
    test_dir = dataset_dir / 'test'

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

    results = {}
    criterion = nn.CrossEntropyLoss()

    for model_name in ['alexnet', 'resnet18']:
        print(f"\n  --- Entrenando {model_name} ---")

        # Crear y entrenar modelo
        model = create_model(model_name, num_classes=len(class_names))
        model = model.to(device)

        model, best_val_acc, best_epoch, train_time = train_model(
            model, train_loader, val_loader, class_weights, device, model_name,
            max_epochs=MAX_EPOCHS, patience=PATIENCE
        )

        # Evaluar en test
        test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
            model, test_loader, criterion, device
        )

        # Métricas por clase
        cm = confusion_matrix(test_labels, test_preds)
        per_class_acc = {}
        for idx, name in enumerate(class_names):
            if cm[idx].sum() > 0:
                per_class_acc[name] = cm[idx, idx] / cm[idx].sum()
            else:
                per_class_acc[name] = 0.0

        results[model_name] = {
            'val_accuracy': best_val_acc * 100,
            'test_accuracy': test_acc * 100,
            'test_f1': test_f1 * 100,
            'epochs': best_epoch,
            'time_seconds': train_time,
            'per_class_accuracy': per_class_acc,
            'confusion_matrix': cm.tolist()
        }

        # GUARDAR MODELO - Sesión 27
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'class_names': class_names,
            'model_name': model_name,
            'val_accuracy': best_val_acc * 100,
            'test_accuracy': test_acc * 100,
            'test_f1': test_f1 * 100,
            'best_epoch': best_epoch,
            'confusion_matrix': cm.tolist()
        }
        save_path = OUTPUT_DIR / f'{model_name}_{experiment_name.replace(" ", "_").replace("(", "").replace(")", "").replace("~", "")}_best.pt'
        torch.save(checkpoint, save_path)
        print(f"    Modelo guardado: {save_path}")

        print(f"    Val Acc: {best_val_acc*100:.2f}%")
        print(f"    Test Acc: {test_acc*100:.2f}%")
        print(f"    F1 Macro: {test_f1*100:.2f}%")
        print(f"    Tiempo: {train_time:.1f}s")

        for name, acc in per_class_acc.items():
            print(f"      {name}: {acc*100:.1f}%")

    return results


def main():
    print("="*70)
    print("SESIÓN 25: COMPARACIÓN BASELINE vs DATASET EXPANDIDO")
    print("="*70)
    print(f"\nModelos canario: AlexNet + ResNet-18")
    print(f"Margin scale: 1.05 (óptimo)")

    # Setup
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Experimento 1: Dataset expandido (~15K imágenes)
    if EXPANDED_DATASET_DIR.exists():
        all_results['expanded_15k'] = run_experiment(
            EXPANDED_DATASET_DIR,
            "DATASET EXPANDIDO (~15K imágenes)",
            device
        )
    else:
        print(f"\nWARNING: Dataset expandido no encontrado en {EXPANDED_DATASET_DIR}")

    # Experimento 2: Baseline (957 imágenes) - si existe con margin 1.05
    if BASELINE_DATASET_DIR.exists():
        all_results['baseline_957'] = run_experiment(
            BASELINE_DATASET_DIR,
            "BASELINE (957 imágenes, margin=1.05)",
            device
        )
    else:
        print(f"\nWARNING: Dataset baseline no encontrado en {BASELINE_DATASET_DIR}")

    # Resumen comparativo
    print("\n" + "="*70)
    print("RESUMEN COMPARATIVO")
    print("="*70)

    if 'expanded_15k' in all_results and 'baseline_957' in all_results:
        print("\n{:<15} {:>12} {:>12} {:>10} {:>10}".format(
            "Modelo", "Baseline", "Expandido", "Mejora", "Factor"
        ))
        print("-"*60)

        for model_name in ['alexnet', 'resnet18']:
            baseline_acc = all_results['baseline_957'][model_name]['test_accuracy']
            expanded_acc = all_results['expanded_15k'][model_name]['test_accuracy']
            improvement = expanded_acc - baseline_acc
            factor = len(list((EXPANDED_DATASET_DIR / 'train').glob('*/*.png'))) / 717  # approx

            print("{:<15} {:>11.2f}% {:>11.2f}% {:>+9.2f}% {:>9.1f}x".format(
                model_name,
                baseline_acc,
                expanded_acc,
                improvement,
                factor
            ))

        print("\n" + "-"*60)
        print("Factor = ratio de imágenes de entrenamiento (expandido/baseline)")

    # Guardar resultados
    results_file = OUTPUT_DIR / "expanded_vs_baseline_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'seed': SEED,
            'margin_scale': 1.05,
            'results': all_results
        }, f, indent=2)

    print(f"\nResultados guardados en: {results_file}")

    # CSV para fácil visualización
    csv_file = OUTPUT_DIR / "expanded_vs_baseline_results.csv"
    with open(csv_file, 'w') as f:
        f.write("experiment,model,val_accuracy,test_accuracy,test_f1,epochs,time_seconds\n")
        for exp_name, exp_results in all_results.items():
            for model_name, metrics in exp_results.items():
                f.write(f"{exp_name},{model_name},{metrics['val_accuracy']:.2f},"
                       f"{metrics['test_accuracy']:.2f},{metrics['test_f1']:.2f},"
                       f"{metrics['epochs']},{metrics['time_seconds']:.1f}\n")

    print(f"CSV guardado en: {csv_file}")

    return all_results


if __name__ == "__main__":
    main()
