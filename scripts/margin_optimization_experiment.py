#!/usr/bin/env python3
"""
Sesión 25: Experimento de Optimización de Margen

Este script genera datasets warpeados con diferentes valores de margin_scale
y entrena los modelos "canario" (AlexNet y ResNet-18) para encontrar el
margen óptimo.

Hipótesis:
- margin_scale > 1.0: Expande el ROI, captura más del pulmón
- margin_scale < 1.0: Contrae el ROI, puede cortar información
- margin_scale = 1.0: Comportamiento actual (baseline)

Si el contorno actual está cortando parte del pulmón, expandir debería
mejorar la accuracy de clasificación.

Autor: Proyecto Tesis Maestría
Fecha: 29-Nov-2024
Sesión: 25
"""

import numpy as np
import cv2
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
from typing import Dict, Tuple, List
from collections import defaultdict
import time
import sys
from PIL import Image
import pandas as pd

# Configuración de paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "margin_experiment"
SHAPE_ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "shape_analysis"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
DATA_DIR = PROJECT_ROOT / "data" / "dataset"

# Importar funciones de warping
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from piecewise_affine_warp import (
    load_canonical_shape,
    load_delaunay_triangles,
    piecewise_affine_warp
)


# =============================================================================
# PARTE 1: FUNCIONES DE ESCALADO DE LANDMARKS
# =============================================================================

def scale_landmarks_from_centroid(landmarks: np.ndarray,
                                   scale: float = 1.0) -> np.ndarray:
    """
    Escalar landmarks desde su centroide.

    Args:
        landmarks: Array (15, 2) con coordenadas de landmarks
        scale: Factor de escala
            - scale > 1.0: Expande (landmarks se alejan del centroide)
            - scale < 1.0: Contrae (landmarks se acercan al centroide)
            - scale = 1.0: Sin cambio

    Returns:
        scaled_landmarks: Array (15, 2) con landmarks escalados
    """
    # Calcular centroide
    centroid = landmarks.mean(axis=0)

    # Escalar desde el centroide
    scaled = centroid + (landmarks - centroid) * scale

    return scaled


def clip_landmarks_to_image(landmarks: np.ndarray,
                            image_size: int = 224,
                            margin: int = 2) -> np.ndarray:
    """
    Asegurar que landmarks estén dentro de la imagen.

    Args:
        landmarks: Array (15, 2)
        image_size: Tamaño de la imagen
        margin: Margen mínimo desde el borde

    Returns:
        clipped_landmarks: Landmarks dentro de límites
    """
    clipped = np.clip(landmarks, margin, image_size - margin - 1)
    return clipped


# =============================================================================
# PARTE 2: GENERACIÓN DE DATASET CON MARGEN
# =============================================================================

def load_all_landmarks() -> Dict:
    """Cargar landmarks GT de todo el dataset."""
    npz_path = PREDICTIONS_DIR / "all_landmarks.npz"
    data = np.load(npz_path, allow_pickle=True)

    return {
        'train': {
            'landmarks': data['train_landmarks'],
            'image_names': data['train_image_names'],
            'categories': data['train_categories']
        },
        'val': {
            'landmarks': data['val_landmarks'],
            'image_names': data['val_image_names'],
            'categories': data['val_categories']
        },
        'test': {
            'landmarks': data['test_landmarks'],
            'image_names': data['test_image_names'],
            'categories': data['test_categories']
        }
    }


def generate_warped_dataset_with_margin(margin_scale: float,
                                         output_base_dir: Path) -> Dict:
    """
    Generar dataset warpeado con un margin_scale específico.

    Args:
        margin_scale: Factor de escala para landmarks
        output_base_dir: Directorio base de salida

    Returns:
        Dict con estadísticas del procesamiento
    """
    output_dir = output_base_dir / f"margin_{margin_scale:.2f}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar datos
    canonical_shape, image_size = load_canonical_shape()
    triangles = load_delaunay_triangles()
    data = load_all_landmarks()

    print(f"\n{'='*60}")
    print(f"Generando dataset con margin_scale = {margin_scale}")
    print(f"{'='*60}")

    all_stats = {}

    for split_name in ['train', 'val', 'test']:
        split_data = data[split_name]
        split_dir = output_dir / split_name

        # Crear directorios
        for cat in ['Normal', 'COVID', 'Viral_Pneumonia']:
            (split_dir / cat).mkdir(parents=True, exist_ok=True)

        landmarks = split_data['landmarks']
        image_names = split_data['image_names']
        categories = split_data['categories']

        n_images = len(landmarks)
        stats = {'processed': 0, 'failed': 0, 'fill_rates': []}

        print(f"\n  {split_name}: {n_images} imágenes...")

        for i in range(n_images):
            lm = landmarks[i]
            name = str(image_names[i])
            cat = str(categories[i])

            # Cargar imagen
            img_path = DATA_DIR / cat / f"{name}.png"
            if not img_path.exists():
                stats['failed'] += 1
                continue

            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                stats['failed'] += 1
                continue

            # Resize
            if image.shape[0] != 224 or image.shape[1] != 224:
                image = cv2.resize(image, (224, 224))

            # APLICAR MARGIN_SCALE a los landmarks fuente
            scaled_landmarks = scale_landmarks_from_centroid(lm, margin_scale)
            scaled_landmarks = clip_landmarks_to_image(scaled_landmarks)

            # Aplicar warping
            try:
                warped = piecewise_affine_warp(
                    image=image,
                    source_landmarks=scaled_landmarks,
                    target_landmarks=canonical_shape,
                    triangles=triangles,
                    use_full_coverage=False
                )
            except Exception as e:
                stats['failed'] += 1
                continue

            # Calcular fill rate
            black_pixels = np.sum(warped == 0)
            fill_rate = 1 - (black_pixels / warped.size)
            stats['fill_rates'].append(fill_rate)

            # Guardar
            output_path = split_dir / cat / f"{name}_warped.png"
            cv2.imwrite(str(output_path), warped)
            stats['processed'] += 1

        print(f"    Procesadas: {stats['processed']}/{n_images}")
        all_stats[split_name] = stats

    # Guardar configuración
    fill_rates = []
    for split_stats in all_stats.values():
        fill_rates.extend(split_stats['fill_rates'])
    fill_rates = np.array(fill_rates)

    config = {
        'margin_scale': margin_scale,
        'fill_rate_mean': float(fill_rates.mean()),
        'fill_rate_std': float(fill_rates.std()),
        'total_processed': sum(s['processed'] for s in all_stats.values()),
        'total_failed': sum(s['failed'] for s in all_stats.values())
    }

    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n  Fill rate: {fill_rates.mean()*100:.1f}% ± {fill_rates.std()*100:.1f}%")

    return config


# =============================================================================
# PARTE 3: DATASET Y DATALOADER
# =============================================================================

class WarpedDataset(Dataset):
    """Dataset para imágenes warpeadas."""

    def __init__(self, data_dir: Path, split: str, transform=None):
        self.data_dir = data_dir / split
        self.transform = transform
        self.classes = ['COVID', 'Normal', 'Viral_Pneumonia']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Recolectar imágenes
        self.samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.png"):
                    self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Cargar imagen
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(data_dir: Path, batch_size: int = 16) -> Dict:
    """Crear dataloaders para train/val/test."""

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = WarpedDataset(data_dir, 'train', train_transform)
    val_dataset = WarpedDataset(data_dir, 'val', eval_transform)
    test_dataset = WarpedDataset(data_dir, 'test', eval_transform)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset)
    }


# =============================================================================
# PARTE 4: MODELOS CANARIO
# =============================================================================

def create_model(model_name: str, num_classes: int = 3, pretrained: bool = True):
    """Crear modelo canario (AlexNet o ResNet-18)."""

    if model_name == 'alexnet':
        model = models.alexnet(pretrained=pretrained)
        model.classifier[-1] = nn.Linear(4096, num_classes)
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    return model


def train_model(model, train_loader, val_loader, device,
                epochs: int = 30, patience: int = 7) -> Dict:
    """
    Entrenar modelo con early stopping.

    Returns:
        Dict con métricas de entrenamiento
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                      factor=0.5, patience=3)

    model = model.to(device)
    best_val_acc = 0
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total

        # Actualizar scheduler
        scheduler.step(val_acc)

        # Guardar historia
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"      Early stopping en epoch {epoch+1}")
            break

    # Restaurar mejor modelo
    model.load_state_dict(best_state)

    return {
        'best_val_acc': best_val_acc,
        'epochs_trained': epoch + 1,
        'history': history
    }


def evaluate_model(model, test_loader, device) -> Dict:
    """Evaluar modelo en test set."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = 100. * (all_preds == all_labels).sum() / len(all_labels)

    # Accuracy por clase
    classes = ['COVID', 'Normal', 'Viral_Pneumonia']
    class_acc = {}
    for i, class_name in enumerate(classes):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc[class_name] = 100. * (all_preds[mask] == all_labels[mask]).sum() / mask.sum()

    return {
        'accuracy': accuracy,
        'class_accuracy': class_acc,
        'predictions': all_preds,
        'labels': all_labels
    }


# =============================================================================
# PARTE 5: EXPERIMENTO PRINCIPAL
# =============================================================================

def run_margin_experiment(margin_scales: List[float] = None,
                          models_to_test: List[str] = None,
                          generate_datasets: bool = True) -> pd.DataFrame:
    """
    Ejecutar experimento completo de optimización de margen.

    Args:
        margin_scales: Lista de valores de margin_scale a probar
        models_to_test: Lista de modelos ('alexnet', 'resnet18')
        generate_datasets: Si True, genera datasets. Si False, usa existentes.

    Returns:
        DataFrame con resultados
    """
    if margin_scales is None:
        margin_scales = [0.95, 1.0, 1.05, 1.10, 1.15]

    if models_to_test is None:
        models_to_test = ['alexnet', 'resnet18']

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Detectar dispositivo
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Usando MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("Usando CPU")

    # 1. Generar datasets (si es necesario)
    if generate_datasets:
        print("\n" + "="*60)
        print("FASE 1: GENERACIÓN DE DATASETS CON DIFERENTES MÁRGENES")
        print("="*60)

        for margin in margin_scales:
            generate_warped_dataset_with_margin(margin, OUTPUT_DIR)

    # 2. Entrenar y evaluar modelos
    print("\n" + "="*60)
    print("FASE 2: ENTRENAMIENTO DE MODELOS CANARIO")
    print("="*60)

    results = []

    for margin in margin_scales:
        data_dir = OUTPUT_DIR / f"margin_{margin:.2f}"

        if not data_dir.exists():
            print(f"\n[!] Dataset para margin={margin} no encontrado. Saltando...")
            continue

        print(f"\n{'='*60}")
        print(f"MARGIN_SCALE = {margin}")
        print(f"{'='*60}")

        # Cargar dataloaders
        loaders = get_dataloaders(data_dir, batch_size=16)
        print(f"  Train: {loaders['train_size']} | Val: {loaders['val_size']} | Test: {loaders['test_size']}")

        for model_name in models_to_test:
            print(f"\n  Entrenando {model_name}...")

            start_time = time.time()

            # Crear modelo
            model = create_model(model_name)

            # Entrenar
            train_result = train_model(
                model,
                loaders['train'],
                loaders['val'],
                device,
                epochs=50,
                patience=10
            )

            # Evaluar
            test_result = evaluate_model(model, loaders['test'], device)

            elapsed = time.time() - start_time

            print(f"    Val Acc: {train_result['best_val_acc']:.2f}%")
            print(f"    Test Acc: {test_result['accuracy']:.2f}%")
            print(f"    Tiempo: {elapsed:.1f}s")

            # Guardar resultado
            result = {
                'margin_scale': margin,
                'model': model_name,
                'val_accuracy': train_result['best_val_acc'],
                'test_accuracy': test_result['accuracy'],
                'epochs': train_result['epochs_trained'],
                'time_seconds': elapsed
            }

            # Añadir accuracy por clase
            for class_name, acc in test_result['class_accuracy'].items():
                result[f'acc_{class_name}'] = acc

            results.append(result)

            # Guardar modelo
            model_dir = OUTPUT_DIR / f"margin_{margin:.2f}" / "models"
            model_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), model_dir / f"{model_name}_best.pt")

    # 3. Crear DataFrame de resultados
    df = pd.DataFrame(results)

    # Guardar resultados
    results_path = OUTPUT_DIR / "margin_experiment_results.csv"
    df.to_csv(results_path, index=False)
    print(f"\nResultados guardados: {results_path}")

    # 4. Mostrar resumen
    print("\n" + "="*60)
    print("RESUMEN DE RESULTADOS")
    print("="*60)

    # Tabla pivote
    pivot = df.pivot(index='margin_scale', columns='model', values='test_accuracy')
    print("\nTest Accuracy por margin_scale y modelo:")
    print(pivot.to_string())

    # Mejor configuración por modelo
    print("\nMejor margin_scale por modelo:")
    for model_name in models_to_test:
        model_df = df[df['model'] == model_name]
        best_row = model_df.loc[model_df['test_accuracy'].idxmax()]
        print(f"  {model_name}: margin={best_row['margin_scale']:.2f} → {best_row['test_accuracy']:.2f}%")

    return df


def main():
    """Punto de entrada principal."""
    print("="*60)
    print("SESIÓN 25: EXPERIMENTO DE OPTIMIZACIÓN DE MARGEN")
    print("="*60)
    print("\nModelos canario: AlexNet (mejora con warped) + ResNet-18 (mayor gap)")
    print("Margin scales: [0.95, 1.0, 1.05, 1.10, 1.15]")

    # Ejecutar experimento
    results = run_margin_experiment(
        margin_scales=[0.95, 1.0, 1.05, 1.10, 1.15],
        models_to_test=['alexnet', 'resnet18'],
        generate_datasets=True
    )

    return results


if __name__ == "__main__":
    main()
