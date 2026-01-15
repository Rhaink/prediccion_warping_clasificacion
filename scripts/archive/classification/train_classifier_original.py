#!/usr/bin/env python3
"""
Sesion 22: Entrenamiento de clasificador CNN en imagenes ORIGINALES
para comparacion con dataset warpeado.

Usa los MISMOS splits que el dataset warpeado para comparacion justa.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
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
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


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
        self.category_to_dir = {
            'COVID': 'COVID',
            'Normal': 'Normal',
            'Viral_Pneumonia': 'Viral Pneumonia',
        }
        self.image_subdir = "images"

        # Construir lista de rutas e etiquetas
        self.samples = []
        for _, row in self.df.iterrows():
            image_name = row['image_name']
            category = row['category']

            img_path = self._find_image_path(category, image_name)
            if img_path is not None:
                self.samples.append((img_path, self.class_to_idx[category]))
            else:
                print(f"Warning: No encontrada {category}/{image_name}")

        self.targets = [s[1] for s in self.samples]

    def _find_image_path(self, category, image_name):
        """Encuentra la ruta a la imagen original con fallback de extensiones."""
        dir_name = self.category_to_dir.get(category, category)
        base_dir = self.original_data_dir / dir_name
        candidates = [
            base_dir / self.image_subdir / f"{image_name}.png",
            base_dir / self.image_subdir / f"{image_name}.jpg",
            base_dir / self.image_subdir / f"{image_name}.jpeg",
            base_dir / f"{image_name}.png",
            base_dir / f"{image_name}.jpg",
            base_dir / f"{image_name}.jpeg",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_class_weights(dataset):
    """Calcula pesos de clase inversamente proporcionales a la frecuencia."""
    class_counts = Counter(dataset.targets)
    n_samples = len(dataset.targets)
    n_classes = len(class_counts)

    weights = []
    for i in range(n_classes):
        count = class_counts.get(i, 1)
        weight = n_samples / (n_classes * count)
        weights.append(weight)

    return torch.FloatTensor(weights)


def get_transforms(img_size=224, train=True):
    """Retorna transformaciones para train/val/test."""
    # Normalizacion de ImageNet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if train:
        return transforms.Compose([
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
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])


def create_model(model_name='resnet18', num_classes=3, pretrained=True):
    """Crea modelo con transfer learning."""
    if model_name == 'resnet18':
        if pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18(weights=None)

        # Modificar capa final
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    elif model_name == 'efficientnet_b0':
        if pretrained:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_b0(weights=None)

        # Modificar clasificador
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    else:
        raise ValueError(f"Modelo {model_name} no soportado")

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


def plot_confusion_matrix(cm, class_names, save_path):
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
    plt.title('Matriz de Confusion (Imagenes Originales)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_history(history, save_path):
    """Grafica historial de entrenamiento."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

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


def main():
    parser = argparse.ArgumentParser(description='Entrenar clasificador en imagenes originales')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file with default values')
    parser.add_argument('--warped-data-dir', type=str,
                        default='outputs/warped_lung_best/session_warping',
                        help='Directorio del dataset warpeado (para leer splits)')
    parser.add_argument('--original-data-dir', type=str,
                        default='data/dataset/COVID-19_Radiography_Dataset',
                        help='Directorio de imagenes originales')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'efficientnet_b0'],
                        help='Arquitectura del modelo')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Numero de epocas')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Tamano del batch')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--use-class-weights', action='store_true', default=True,
                        help='Usar pesos de clase')
    parser.add_argument('--output-dir', type=str,
                        default='outputs/classifier_original',
                        help='Directorio de salida')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla aleatoria')

    args, _ = parser.parse_known_args()
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_file():
            parser.error(f"Config file not found: {config_path}")
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        if not isinstance(config_data, dict):
            parser.error("Config file must be a JSON object with flat key/value pairs.")
        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(config_data) - valid_keys)
        if unknown_keys:
            parser.error(f"Unknown config keys: {', '.join(unknown_keys)}")
        parser.set_defaults(**config_data)

    args = parser.parse_args()

    # Configurar semilla
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Crear directorio de salida
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # Rutas
    warped_dir = PROJECT_ROOT / args.warped_data_dir
    original_dir = PROJECT_ROOT / args.original_data_dir

    # Transformaciones
    train_transform = get_transforms(train=True)
    eval_transform = get_transforms(train=False)

    # Datasets (usando mismos splits que dataset warpeado)
    print("\nCargando datasets originales con splits del dataset warpeado...")
    train_dataset = OriginalDataset(
        warped_dir / 'train' / 'images.csv',
        original_dir,
        transform=train_transform
    )
    val_dataset = OriginalDataset(
        warped_dir / 'val' / 'images.csv',
        original_dir,
        transform=eval_transform
    )
    test_dataset = OriginalDataset(
        warped_dir / 'test' / 'images.csv',
        original_dir,
        transform=eval_transform
    )

    class_names = train_dataset.classes
    print(f"Clases: {class_names}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Distribucion de clases
    train_counts = Counter(train_dataset.targets)
    print(f"\nDistribucion de train:")
    for idx, name in enumerate(class_names):
        print(f"  {name}: {train_counts[idx]}")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4)

    # Modelo
    print(f"\nCreando modelo {args.model}...")
    model = create_model(args.model, num_classes=len(class_names), pretrained=True)
    model = model.to(device)

    # Class weights
    class_weights = None
    if args.use_class_weights:
        class_weights = get_class_weights(train_dataset).to(device)
        print(f"Class weights: {class_weights.cpu().numpy()}")

    # Loss y optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Historial de entrenamiento
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_f1_macro': [], 'val_f1_weighted': []
    }

    # Early stopping
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None

    print(f"\n{'='*60}")
    print(f"Iniciando entrenamiento (IMAGENES ORIGINALES): {args.epochs} epocas")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_f1_macro, val_f1_weighted, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        # Scheduler step
        scheduler.step(val_f1_macro)

        # Guardar historial
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1_macro'].append(val_f1_macro)
        history['val_f1_weighted'].append(val_f1_weighted)

        print(f"Epoch {epoch+1:3d}/{args.epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, "
              f"F1={val_f1_macro:.4f}")

        # Early stopping
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  -> Nuevo mejor modelo: F1 = {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping en epoch {epoch+1}")
                break

    # Cargar mejor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Guardar modelo
    model_path = output_dir / 'best_classifier_original.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'model_name': args.model,
        'best_val_f1': best_val_f1,
    }, model_path)
    print(f"\nModelo guardado en: {model_path}")

    # Evaluacion final en TEST
    print(f"\n{'='*60}")
    print("EVALUACION FINAL EN TEST SET (IMAGENES ORIGINALES)")
    print(f"{'='*60}\n")

    test_loss, test_acc, test_f1_macro, test_f1_weighted, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Macro: {test_f1_macro:.4f}")
    print(f"Test F1 Weighted: {test_f1_weighted:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print("Confusion Matrix:")
    print(cm)

    # Guardar confusion matrix
    cm_path = output_dir / 'confusion_matrix_original.png'
    plot_confusion_matrix(cm, class_names, cm_path)
    print(f"\nConfusion matrix guardada en: {cm_path}")

    # Guardar training history
    history_path = output_dir / 'training_history_original.png'
    plot_training_history(history, history_path)
    print(f"Training history guardada en: {history_path}")

    # Guardar resultados JSON
    results = {
        'model': args.model,
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
        'dataset_type': 'original',
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'timestamp': datetime.now().isoformat(),
    }

    results_path = output_dir / 'results_original.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Resultados guardados en: {results_path}")

    print(f"\n{'='*60}")
    print("RESUMEN FINAL (IMAGENES ORIGINALES)")
    print(f"{'='*60}")
    print(f"Modelo: {args.model}")
    print(f"Dataset: ORIGINAL (299x299, imagen completa)")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test F1 Macro: {test_f1_macro*100:.2f}%")
    print(f"Test F1 Weighted: {test_f1_weighted*100:.2f}%")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
