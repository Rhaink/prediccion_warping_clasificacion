#!/usr/bin/env python3
"""
Sesion 23: Script generalizado para entrenar multiples arquitecturas CNN
en ambos datasets (original y warpeado).

Objetivo: Demostrar que el shortcut learning es CONSISTENTE independientemente
de la arquitectura CNN utilizada.

Arquitecturas soportadas (comparables a MVTec HALCON):
- alexnet
- resnet18
- resnet50
- mobilenet_v2
- efficientnet_b0
- densenet121
- vgg16

Datasets:
- warped: outputs/warped_dataset/ (imagenes con warping anatomico)
- original: data/dataset/ (imagenes originales 299x299)

Uso:
    python scripts/archive/classification/train_all_architectures.py --model resnet50 --dataset warped
    python scripts/archive/classification/train_all_architectures.py --model alexnet --dataset original
    python scripts/archive/classification/train_all_architectures.py --run-all  # Ejecuta todos los experimentos
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Agregar el directorio raiz al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Arquitecturas soportadas y sus funciones de creacion
SUPPORTED_MODELS = {
    'alexnet': 'AlexNet',
    'resnet18': 'ResNet-18',
    'resnet50': 'ResNet-50',
    'mobilenet_v2': 'MobileNetV2',
    'efficientnet_b0': 'EfficientNet-B0',
    'densenet121': 'DenseNet-121',
    'vgg16': 'VGG-16',
}


class OriginalDataset(Dataset):
    """Dataset que carga imagenes originales usando splits del dataset warpeado."""

    def __init__(self, split_csv, original_data_dir, transform=None):
        """
        Args:
            split_csv: Ruta al CSV del split (e.g., train/images.csv)
            original_data_dir: Directorio con imagenes originales (data/dataset)
            transform: Transformaciones a aplicar
        """
        self.df = pd.read_csv(split_csv)
        self.original_data_dir = Path(original_data_dir)
        self.transform = transform

        # Mapeo de categorias a indices
        self.class_to_idx = {'COVID': 0, 'Normal': 1, 'Viral_Pneumonia': 2}
        self.classes = ['COVID', 'Normal', 'Viral_Pneumonia']

        # Construir lista de rutas e etiquetas
        self.samples = []
        for _, row in self.df.iterrows():
            image_name = row['image_name']
            category = row['category']

            # Construir ruta a imagen original
            img_path = self.original_data_dir / category / f"{image_name}.png"

            if img_path.exists():
                self.samples.append((img_path, self.class_to_idx[category]))
            # Si no existe, intentar con espacio en lugar de guion bajo
            else:
                alt_category = category.replace('_', ' ')
                alt_name = image_name.replace('_', ' ')
                img_path_alt = self.original_data_dir / category / f"{alt_name}.png"
                if img_path_alt.exists():
                    self.samples.append((img_path_alt, self.class_to_idx[category]))

        self.targets = [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class GrayscaleToRGB:
    """Convierte imagen grayscale a RGB (3 canales)."""
    def __call__(self, img):
        if img.mode == 'RGB':
            return img
        return img.convert('RGB')


def get_class_weights(dataset):
    """Calcula pesos de clase inversamente proporcionales a la frecuencia."""
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


def get_transforms(img_size=224, train=True):
    """Retorna transformaciones para train/val/test."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if train:
        return transforms.Compose([
            GrayscaleToRGB(),
            transforms.Resize((img_size, img_size)),
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
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])


def create_model(model_name, num_classes=3, pretrained=True):
    """Crea modelo con transfer learning para cualquier arquitectura soportada."""

    if model_name == 'alexnet':
        if pretrained:
            model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        else:
            model = models.alexnet(weights=None)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    elif model_name == 'resnet18':
        if pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    elif model_name == 'resnet50':
        if pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    elif model_name == 'mobilenet_v2':
        if pretrained:
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            model = models.mobilenet_v2(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    elif model_name == 'efficientnet_b0':
        if pretrained:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    elif model_name == 'densenet121':
        if pretrained:
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        else:
            model = models.densenet121(weights=None)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    elif model_name == 'vgg16':
        if pretrained:
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        else:
            model = models.vgg16(weights=None)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    else:
        raise ValueError(f"Modelo '{model_name}' no soportado. "
                        f"Opciones: {list(SUPPORTED_MODELS.keys())}")

    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entrena una epoca."""
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

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Evalua el modelo."""
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
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    return epoch_loss, epoch_acc, f1_macro, f1_weighted, all_preds, all_labels


def plot_confusion_matrix(cm, class_names, save_path, title='Matriz de Confusion'):
    """Genera y guarda matriz de confusion."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Prediccion')
    plt.ylabel('Real')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_history(history, save_path, title='Historial de Entrenamiento'):
    """Grafica historial de entrenamiento."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=14)

    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss por Epoca')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy por Epoca')
    axes[1].legend()
    axes[1].grid(True)

    # F1 Score
    axes[2].plot(history['val_f1_macro'], label='F1 Macro')
    axes[2].plot(history['val_f1_weighted'], label='F1 Weighted')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('F1 Score (Validation)')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_single_experiment(model_name, dataset_type, output_base_dir,
                           epochs=50, batch_size=32, lr=1e-4, patience=10, seed=42):
    """
    Entrena un solo experimento (una arquitectura en un dataset).

    Args:
        model_name: Nombre de la arquitectura (e.g., 'resnet50')
        dataset_type: 'warped' o 'original'
        output_base_dir: Directorio base para guardar resultados
        epochs: Numero maximo de epocas
        batch_size: Tamano del batch
        lr: Learning rate
        patience: Paciencia para early stopping
        seed: Semilla aleatoria

    Returns:
        dict con resultados del experimento
    """
    start_time = time.time()

    # Configurar semilla
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Directorio de salida especifico
    output_dir = Path(output_base_dir) / f"{model_name}_{dataset_type}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Rutas del dataset
    warped_data_dir = PROJECT_ROOT / 'outputs' / 'warped_dataset'
    original_data_dir = PROJECT_ROOT / 'data' / 'dataset'

    # Transformaciones
    train_transform = get_transforms(train=True)
    eval_transform = get_transforms(train=False)

    print(f"\n{'='*60}")
    print(f"Experimento: {SUPPORTED_MODELS[model_name]} en dataset {dataset_type}")
    print(f"{'='*60}")

    # Datasets
    if dataset_type == 'warped':
        train_dir = warped_data_dir / 'train'
        val_dir = warped_data_dir / 'val'
        test_dir = warped_data_dir / 'test'

        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
        test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)
    else:
        # Para dataset original, usar OriginalDataset con los CSVs del warped
        train_dataset = OriginalDataset(
            warped_data_dir / 'train' / 'images.csv',
            original_data_dir,
            transform=train_transform
        )
        val_dataset = OriginalDataset(
            warped_data_dir / 'val' / 'images.csv',
            original_data_dir,
            transform=eval_transform
        )
        test_dataset = OriginalDataset(
            warped_data_dir / 'test' / 'images.csv',
            original_data_dir,
            transform=eval_transform
        )

    class_names = train_dataset.classes
    print(f"Clases: {class_names}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=4)

    # Modelo
    print(f"Creando modelo {SUPPORTED_MODELS[model_name]}...")
    model = create_model(model_name, num_classes=len(class_names), pretrained=True)
    model = model.to(device)

    # Contar parametros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametros totales: {total_params:,}")
    print(f"Parametros entrenables: {trainable_params:,}")

    # Class weights
    class_weights = get_class_weights(train_dataset).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")

    # Loss y optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=False
    )

    # Historial
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_f1_macro': [], 'val_f1_weighted': []
    }

    # Early stopping
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None

    print(f"Iniciando entrenamiento ({epochs} epocas max)...")

    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_f1_macro, val_f1_weighted, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        # Scheduler
        scheduler.step(val_f1_macro)

        # Guardar historial
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1_macro'].append(val_f1_macro)
        history['val_f1_weighted'].append(val_f1_weighted)

        # Early stopping
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping en epoch {epoch+1}")
                break

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}: Val Acc={val_acc:.4f}, F1={val_f1_macro:.4f}")

    # Cargar mejor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Guardar modelo
    model_path = output_dir / 'best_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'model_name': model_name,
        'best_val_f1': best_val_f1,
    }, model_path)

    # Evaluacion final en TEST
    test_loss, test_acc, test_f1_macro, test_f1_weighted, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nResultados Test:")
    print(f"  Accuracy: {test_acc*100:.2f}%")
    print(f"  F1 Macro: {test_f1_macro*100:.2f}%")
    print(f"  F1 Weighted: {test_f1_weighted*100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    cm_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(cm, class_names, cm_path,
                         title=f'{SUPPORTED_MODELS[model_name]} - {dataset_type}')

    # Training history
    history_path = output_dir / 'training_history.png'
    plot_training_history(history, history_path,
                         title=f'{SUPPORTED_MODELS[model_name]} - {dataset_type}')

    # Tiempo de entrenamiento
    training_time = time.time() - start_time

    # Resultados
    results = {
        'model': model_name,
        'model_display_name': SUPPORTED_MODELS[model_name],
        'dataset': dataset_type,
        'epochs_trained': len(history['train_loss']),
        'best_val_f1': float(best_val_f1),
        'test_metrics': {
            'accuracy': float(test_acc),
            'f1_macro': float(test_f1_macro),
            'f1_weighted': float(test_f1_weighted),
        },
        'per_class_metrics': classification_report(
            test_labels, test_preds, target_names=class_names, output_dict=True
        ),
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'training_time_seconds': training_time,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'timestamp': datetime.now().isoformat(),
    }

    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Tiempo de entrenamiento: {training_time/60:.1f} min")

    return results


def run_all_experiments(output_base_dir, epochs=50, batch_size=32, lr=1e-4,
                       patience=10, seed=42, skip_existing=True):
    """
    Ejecuta todos los experimentos (todas las arquitecturas en ambos datasets).

    Args:
        output_base_dir: Directorio base para resultados
        skip_existing: Si True, salta experimentos que ya tienen resultados

    Returns:
        Lista de diccionarios con resultados
    """
    all_results = []
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    models = list(SUPPORTED_MODELS.keys())
    datasets = ['original', 'warped']

    total_experiments = len(models) * len(datasets)
    completed = 0

    print(f"\n{'='*60}")
    print(f"EJECUTANDO {total_experiments} EXPERIMENTOS")
    print(f"{'='*60}")
    print(f"Arquitecturas: {list(SUPPORTED_MODELS.values())}")
    print(f"Datasets: {datasets}")
    print(f"{'='*60}\n")

    for model_name in models:
        for dataset_type in datasets:
            completed += 1
            exp_name = f"{model_name}_{dataset_type}"
            exp_dir = output_base / exp_name
            results_path = exp_dir / 'results.json'

            print(f"\n[{completed}/{total_experiments}] {SUPPORTED_MODELS[model_name]} en {dataset_type}")

            # Verificar si ya existe
            if skip_existing and results_path.exists():
                print(f"  -> Ya existe, cargando resultados previos...")
                with open(results_path) as f:
                    results = json.load(f)
                all_results.append(results)
                continue

            # Ejecutar experimento
            try:
                results = train_single_experiment(
                    model_name=model_name,
                    dataset_type=dataset_type,
                    output_base_dir=output_base_dir,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    patience=patience,
                    seed=seed
                )
                all_results.append(results)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    return all_results


def generate_comparison_report(results_list, output_dir):
    """
    Genera tabla comparativa y graficos de todos los resultados.

    Args:
        results_list: Lista de diccionarios con resultados
        output_dir: Directorio donde guardar los reportes
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Crear DataFrame
    import pandas as pd

    data = []
    for r in results_list:
        data.append({
            'Model': r['model_display_name'],
            'Dataset': r['dataset'],
            'Accuracy': r['test_metrics']['accuracy'] * 100,
            'F1_Macro': r['test_metrics']['f1_macro'] * 100,
            'F1_Weighted': r['test_metrics']['f1_weighted'] * 100,
            'Epochs': r['epochs_trained'],
            'Time_min': r.get('training_time_seconds', 0) / 60,
            'Params': r.get('total_params', 0),
        })

    df = pd.DataFrame(data)

    # Guardar CSV
    csv_path = output_dir / 'comparison_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nTabla guardada en: {csv_path}")

    # Crear tabla pivote para comparacion
    pivot_acc = df.pivot(index='Model', columns='Dataset', values='Accuracy')
    pivot_f1 = df.pivot(index='Model', columns='Dataset', values='F1_Macro')

    # Calcular diferencia Original - Warped
    if 'original' in pivot_acc.columns and 'warped' in pivot_acc.columns:
        pivot_acc['Diferencia'] = pivot_acc['original'] - pivot_acc['warped']
        pivot_f1['Diferencia'] = pivot_f1['original'] - pivot_f1['warped']

    print("\n" + "="*80)
    print("TABLA COMPARATIVA DE ACCURACY (%)")
    print("="*80)
    print(pivot_acc.round(2).to_string())

    print("\n" + "="*80)
    print("TABLA COMPARATIVA DE F1-MACRO (%)")
    print("="*80)
    print(pivot_f1.round(2).to_string())

    # Grafico de barras comparativo
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy
    models = df['Model'].unique()
    x = np.arange(len(models))
    width = 0.35

    original_acc = df[df['Dataset'] == 'original'].set_index('Model')['Accuracy'].reindex(models).values
    warped_acc = df[df['Dataset'] == 'warped'].set_index('Model')['Accuracy'].reindex(models).values

    bars1 = axes[0].bar(x - width/2, original_acc, width, label='Original', color='#e74c3c', alpha=0.8)
    bars2 = axes[0].bar(x + width/2, warped_acc, width, label='Warped', color='#3498db', alpha=0.8)

    axes[0].set_xlabel('Arquitectura')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy: Original vs Warped')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim(80, 100)

    # Etiquetas de valor
    for bar in bars1:
        height = bar.get_height()
        axes[0].annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        axes[0].annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    # F1 Macro
    original_f1 = df[df['Dataset'] == 'original'].set_index('Model')['F1_Macro'].reindex(models).values
    warped_f1 = df[df['Dataset'] == 'warped'].set_index('Model')['F1_Macro'].reindex(models).values

    bars3 = axes[1].bar(x - width/2, original_f1, width, label='Original', color='#e74c3c', alpha=0.8)
    bars4 = axes[1].bar(x + width/2, warped_f1, width, label='Warped', color='#3498db', alpha=0.8)

    axes[1].set_xlabel('Arquitectura')
    axes[1].set_ylabel('F1-Macro (%)')
    axes[1].set_title('F1-Macro: Original vs Warped')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim(80, 100)

    for bar in bars3:
        height = bar.get_height()
        axes[1].annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    for bar in bars4:
        height = bar.get_height()
        axes[1].annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    chart_path = output_dir / 'comparison_chart.png'
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Grafico guardado en: {chart_path}")

    # Grafico de diferencias
    fig, ax = plt.subplots(figsize=(10, 6))

    diff_acc = original_acc - warped_acc
    colors = ['#e74c3c' if d > 0 else '#3498db' for d in diff_acc]

    bars = ax.barh(models, diff_acc, color=colors, alpha=0.8)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Diferencia Accuracy (Original - Warped) [%]')
    ax.set_title('Gap de Accuracy entre Original y Warped\n(Valores positivos = shortcuts en original)')
    ax.grid(axis='x', alpha=0.3)

    # Etiquetas
    for bar, val in zip(bars, diff_acc):
        width = bar.get_width()
        ax.annotate(f'{val:.1f}%',
                   xy=(width, bar.get_y() + bar.get_height()/2),
                   xytext=(5 if width > 0 else -5, 0),
                   textcoords="offset points",
                   ha='left' if width > 0 else 'right',
                   va='center')

    plt.tight_layout()
    diff_path = output_dir / 'accuracy_difference.png'
    plt.savefig(diff_path, dpi=150)
    plt.close()
    print(f"Grafico de diferencias guardado en: {diff_path}")

    # Guardar resumen JSON
    summary = {
        'total_experiments': len(results_list),
        'architectures': list(models),
        'mean_accuracy_original': float(np.mean(original_acc)),
        'mean_accuracy_warped': float(np.mean(warped_acc)),
        'mean_difference': float(np.mean(diff_acc)),
        'std_difference': float(np.std(diff_acc)),
        'hypothesis_validated': {
            'H1_original_high': bool(np.mean(original_acc) > 90),
            'H2_warped_honest': bool(np.mean(warped_acc) < 92),
            'H3_consistent_gap': bool(np.std(diff_acc) < 5),
            'H4_data_problem': bool(np.mean(diff_acc) > 5),
        },
        'timestamp': datetime.now().isoformat(),
    }

    summary_path = output_dir / 'summary_report.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Resumen guardado en: {summary_path}")

    # Imprimir conclusiones
    print("\n" + "="*80)
    print("CONCLUSIONES")
    print("="*80)
    print(f"Accuracy promedio Original: {np.mean(original_acc):.1f}%")
    print(f"Accuracy promedio Warped:   {np.mean(warped_acc):.1f}%")
    print(f"Diferencia promedio:        {np.mean(diff_acc):.1f}% (±{np.std(diff_acc):.1f}%)")
    print()
    print("Validacion de Hipotesis:")
    print(f"  H1: Todas las redes en original dan >90%? {'SI' if np.all(original_acc > 90) else 'NO'}")
    print(f"  H2: Todas las redes en warped dan 85-92%? {'SI' if np.all((warped_acc > 82) & (warped_acc < 93)) else 'NO'}")
    print(f"  H3: Gap consistente (<5% std)?            {'SI' if np.std(diff_acc) < 5 else 'NO'}")
    print(f"  H4: Problema es de DATOS (gap >5%)?       {'SI' if np.mean(diff_acc) > 5 else 'NO'}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Entrenar clasificadores CNN - Sesion 23',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Entrenar un modelo especifico
  python scripts/archive/classification/train_all_architectures.py --model resnet50 --dataset warped

  # Ejecutar todos los experimentos
  python scripts/archive/classification/train_all_architectures.py --run-all

  # Solo generar reporte (si ya existen resultados)
  python scripts/archive/classification/train_all_architectures.py --report-only
        """
    )

    parser.add_argument('--model', type=str, choices=list(SUPPORTED_MODELS.keys()),
                        help='Arquitectura a entrenar')
    parser.add_argument('--dataset', type=str, choices=['warped', 'original'],
                        help='Dataset a usar')
    parser.add_argument('--run-all', action='store_true',
                        help='Ejecutar todos los experimentos')
    parser.add_argument('--report-only', action='store_true',
                        help='Solo generar reporte de resultados existentes')
    parser.add_argument('--output-dir', type=str,
                        default='outputs/classifier_comparison',
                        help='Directorio de salida')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-skip', action='store_true',
                        help='No saltar experimentos existentes')

    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir

    if args.report_only:
        # Solo generar reporte
        print("Generando reporte de resultados existentes...")
        results = []
        for model_name in SUPPORTED_MODELS.keys():
            for dataset_type in ['original', 'warped']:
                results_path = output_dir / f"{model_name}_{dataset_type}" / 'results.json'
                if results_path.exists():
                    with open(results_path) as f:
                        results.append(json.load(f))

        if results:
            generate_comparison_report(results, output_dir)
        else:
            print("No se encontraron resultados.")
        return

    if args.run_all:
        # Ejecutar todos los experimentos
        results = run_all_experiments(
            output_base_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            seed=args.seed,
            skip_existing=not args.no_skip
        )
        generate_comparison_report(results, output_dir)

    elif args.model and args.dataset:
        # Ejecutar un solo experimento
        results = train_single_experiment(
            model_name=args.model,
            dataset_type=args.dataset,
            output_base_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            seed=args.seed
        )
        print(f"\nResultados guardados en: {output_dir / f'{args.model}_{args.dataset}'}")

    else:
        parser.print_help()
        print("\nError: Especifica --model y --dataset, o usa --run-all")


if __name__ == '__main__':
    main()
