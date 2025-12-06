#!/usr/bin/env python3
"""
SESIÓN 28: Experimento de Márgenes Extendidos

OBJETIVO: Encontrar el margen óptimo real con 15K imágenes.
          El margen 1.05 fue óptimo con 957 imágenes; ¿cambia con más datos?

MÁRGENES A PROBAR: 1.05, 1.10, 1.15, 1.20, 1.25, 1.30

METODOLOGÍA:
1. Generar dataset warped para cada margen
2. Entrenar ResNet-18 con configuración estándar
3. Comparar accuracy en test set

Autor: Proyecto Tesis Maestría
Fecha: 29-Nov-2024
Sesión: 28
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2
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
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from scripts.predict import EnsemblePredictor
from scripts.piecewise_affine_warp import (
    load_canonical_shape,
    load_delaunay_triangles,
    piecewise_affine_warp
)

# Configuración
BATCH_SIZE = 32
MAX_EPOCHS = 30  # Reducido para ahorrar tiempo
PATIENCE = 7
LEARNING_RATE = 1e-4
SEED = 42
IMAGE_SIZE = 224

# Márgenes a probar
MARGINS = [1.05, 1.10, 1.15, 1.20, 1.25, 1.30]

# Paths
ORIGINAL_DATASET_DIR = PROJECT_ROOT / "data" / "dataset" / "COVID-19_Radiography_Dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "session28_margin_experiment"

# Clases
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


def scale_landmarks_from_centroid(landmarks, scale):
    centroid = landmarks.mean(axis=0)
    return centroid + (landmarks - centroid) * scale


def clip_landmarks_to_image(landmarks, image_size=224, margin=2):
    return np.clip(landmarks, margin, image_size - margin - 1)


def create_splits(seed=42, train_ratio=0.75, val_ratio=0.15):
    """Crear splits idénticos a otros experimentos."""
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

    splits = {'train': [], 'val': [], 'test': []}

    for class_name, paths in sorted(by_class.items()):
        np.random.shuffle(paths)
        n = len(paths)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits['train'].extend([(p, class_name) for p in paths[:n_train]])
        splits['val'].extend([(p, class_name) for p in paths[n_train:n_train+n_val]])
        splits['test'].extend([(p, class_name) for p in paths[n_train+n_val:]])

    for split in splits.values():
        np.random.shuffle(split)

    return splits


class WarpedOnFlyDataset(Dataset):
    """Dataset que aplica warping on-the-fly con un margen específico."""

    def __init__(self, image_paths, predictor, canonical_shape, triangles,
                 margin_scale, transform=None, cache_landmarks=True):
        self.image_paths = image_paths
        self.predictor = predictor
        self.canonical_shape = canonical_shape
        self.triangles = triangles
        self.margin_scale = margin_scale
        self.transform = transform

        # Cache de landmarks para evitar predicciones repetidas
        self.landmarks_cache = {}
        self.cache_landmarks = cache_landmarks

    def __len__(self):
        return len(self.image_paths)

    def _get_landmarks(self, img_path):
        if self.cache_landmarks and str(img_path) in self.landmarks_cache:
            return self.landmarks_cache[str(img_path)]

        landmarks_norm = self.predictor.predict(img_path, return_normalized=True)
        landmarks = landmarks_norm * IMAGE_SIZE

        if self.cache_landmarks:
            self.landmarks_cache[str(img_path)] = landmarks

        return landmarks

    def __getitem__(self, idx):
        img_path, class_idx = self.image_paths[idx]

        # Cargar imagen
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image.shape[0] != IMAGE_SIZE or image.shape[1] != IMAGE_SIZE:
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        # Obtener landmarks
        landmarks = self._get_landmarks(img_path)

        # Aplicar margin_scale
        scaled = scale_landmarks_from_centroid(landmarks, self.margin_scale)
        scaled = clip_landmarks_to_image(scaled)

        # Aplicar warping
        warped = piecewise_affine_warp(
            image=image,
            source_landmarks=scaled,
            target_landmarks=self.canonical_shape,
            triangles=self.triangles,
            use_full_coverage=False
        )

        # Convertir a PIL para transforms
        img = Image.fromarray(warped)

        if self.transform:
            img = self.transform(img)

        return img, class_idx


def get_transforms(train=True):
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


def create_resnet18(num_classes=3):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model


def train_and_evaluate(margin, splits, predictor, canonical_shape, triangles, device):
    """Entrena y evalúa un modelo con un margen específico."""
    print(f"\n{'='*60}")
    print(f"MARGEN: {margin}")
    print(f"{'='*60}")

    # Preparar class indices
    class_names = ['COVID', 'Normal', 'Viral_Pneumonia']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    train_paths = [(p, class_to_idx[c]) for p, c in splits['train']]
    val_paths = [(p, class_to_idx[c]) for p, c in splits['val']]
    test_paths = [(p, class_to_idx[c]) for p, c in splits['test']]

    # Crear datasets
    train_transform = get_transforms(train=True)
    eval_transform = get_transforms(train=False)

    # Usar subconjunto para acelerar (2000 train, 500 val, 500 test)
    # Para experimento completo, comentar estas líneas
    np.random.seed(SEED)
    train_subset = np.random.choice(len(train_paths), min(3000, len(train_paths)), replace=False)
    train_paths = [train_paths[i] for i in train_subset]

    train_dataset = WarpedOnFlyDataset(
        train_paths, predictor, canonical_shape, triangles,
        margin, train_transform
    )
    val_dataset = WarpedOnFlyDataset(
        val_paths, predictor, canonical_shape, triangles,
        margin, eval_transform
    )
    test_dataset = WarpedOnFlyDataset(
        test_paths, predictor, canonical_shape, triangles,
        margin, eval_transform
    )

    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # Class weights
    train_labels = [c for _, c in train_paths]
    class_counts = Counter(train_labels)
    n_samples = len(train_labels)
    weights = torch.FloatTensor([
        n_samples / (3 * class_counts.get(i, 1)) for i in range(3)
    ])

    # DataLoaders (num_workers=0 para evitar problemas de CUDA con multiprocessing)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Modelo
    model = create_resnet18()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    best_val_acc = 0
    patience_counter = 0
    best_state = None

    # Entrenamiento
    for epoch in range(MAX_EPOCHS):
        # Train
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.numpy())

        val_acc = accuracy_score(val_labels, val_preds)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}: Val Acc={val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    # Cargar mejor modelo
    model.load_state_dict(best_state)

    # Test
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='macro')

    print(f"  RESULTADO: Test Acc={test_acc*100:.2f}% | F1={test_f1*100:.2f}%")

    return {
        'margin': margin,
        'val_accuracy': best_val_acc * 100,
        'test_accuracy': test_acc * 100,
        'test_f1': test_f1 * 100
    }


def main():
    print("="*70)
    print("SESIÓN 28: EXPERIMENTO DE MÁRGENES EXTENDIDOS")
    print("="*70)
    print(f"Márgenes a probar: {MARGINS}")

    # Setup
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Crear splits
    print("\n1. Creando splits...")
    splits = create_splits(seed=SEED)
    print(f"   Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")

    # Cargar predictor y configuración
    print("\n2. Cargando modelo de landmarks...")
    predictor = EnsemblePredictor(use_clahe=True)
    canonical_shape, _ = load_canonical_shape()
    triangles = load_delaunay_triangles()

    # Ejecutar experimentos
    print("\n3. Ejecutando experimentos por margen...")
    results = []

    start_time = time.time()

    for margin in MARGINS:
        result = train_and_evaluate(
            margin, splits, predictor, canonical_shape, triangles, device
        )
        results.append(result)

    elapsed = time.time() - start_time

    # Resumen
    print("\n" + "="*70)
    print("RESUMEN DE RESULTADOS")
    print("="*70)

    print(f"\n{'Margen':<10} {'Val Acc':<12} {'Test Acc':<12} {'Test F1':<12}")
    print("-" * 46)

    best_margin = None
    best_acc = 0

    for r in results:
        print(f"{r['margin']:<10.2f} {r['val_accuracy']:<12.2f} {r['test_accuracy']:<12.2f} {r['test_f1']:<12.2f}")
        if r['test_accuracy'] > best_acc:
            best_acc = r['test_accuracy']
            best_margin = r['margin']

    print(f"\n*** MEJOR MARGEN: {best_margin} con {best_acc:.2f}% accuracy ***")

    # Guardar resultados
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'margins_tested': MARGINS,
        'results': results,
        'best_margin': best_margin,
        'best_accuracy': best_acc,
        'elapsed_minutes': elapsed / 60
    }

    results_path = OUTPUT_DIR / 'margin_experiment_results.json'
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResultados guardados: {results_path}")

    return best_margin


if __name__ == "__main__":
    main()
