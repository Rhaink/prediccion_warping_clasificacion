#!/usr/bin/env python3
"""
SESIÓN 30: Cross-Evaluation - Evaluación Cruzada de Modelos

OBJETIVO: Evaluar cada modelo en AMBOS datasets para medir generalización.

Experimentos:
- Modelo Original → Test Original = 98.84% (baseline, ya conocido)
- Modelo Original → Test Warped   = ??? (cross-evaluation)
- Modelo Warped   → Test Warped   = 98.02% (baseline, ya conocido)
- Modelo Warped   → Test Original = ??? (cross-evaluation)

HIPÓTESIS:
- Si el modelo warped generaliza mejor, debería mantener accuracy alto en imágenes originales
- El modelo original podría degradar más al evaluar en imágenes warped

Autor: Proyecto Tesis Maestría
Fecha: 29-Nov-2024
Sesión: 30
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

# Configuración
BATCH_SIZE = 32
SEED = 42

# Paths
WARPED_DATASET_DIR = PROJECT_ROOT / "outputs" / "full_warped_dataset"
ORIGINAL_DATASET_DIR = PROJECT_ROOT / "data" / "dataset" / "COVID-19_Radiography_Dataset"
MODEL_ORIGINAL_PATH = PROJECT_ROOT / "outputs" / "session28_baseline_original" / "resnet18_original_15k_best.pt"
MODEL_WARPED_PATH = PROJECT_ROOT / "outputs" / "session27_models" / "resnet18_expanded_15k_best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "session30_analysis"

# Clases
CLASSES = ['COVID', 'Normal', 'Viral Pneumonia']
CLASS_MAPPING = {
    'COVID': 'COVID',
    'Normal': 'Normal',
    'Viral Pneumonia': 'Viral_Pneumonia'
}


class GrayscaleToRGB:
    """Convierte imagen grayscale a RGB (3 canales)."""
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
        return img, class_idx, str(img_path)


def get_eval_transform():
    """Transformación de evaluación (IDÉNTICA a ambos modelos)."""
    return transforms.Compose([
        GrayscaleToRGB(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def create_original_test_split(seed=42, train_ratio=0.75, val_ratio=0.15):
    """
    Crear split de test IDÉNTICO al usado en entrenamiento.
    Replica la lógica de generate_full_warped_dataset.py
    """
    np.random.seed(seed)

    all_images = []
    for class_name in CLASSES:
        class_dir = ORIGINAL_DATASET_DIR / class_name / "images"
        if class_dir.exists():
            images = list(class_dir.glob("*.png"))
            mapped_class = CLASS_MAPPING[class_name]
            all_images.extend([(img, mapped_class) for img in images])

    # Agrupar por clase
    by_class = defaultdict(list)
    for path, class_name in all_images:
        by_class[class_name].append(path)

    test_paths = []

    # Mismo orden de procesamiento que el script original
    for class_name, paths in sorted(by_class.items()):
        np.random.shuffle(paths)
        n = len(paths)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # Solo nos interesa el test split
        test_paths.extend([(p, class_name) for p in paths[n_train+n_val:]])

    # Shuffle
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


def evaluate_model(model, dataloader, device, class_names):
    """Evalúa un modelo y retorna métricas detalladas."""
    model.eval()
    all_preds = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for inputs, labels, paths in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)

    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1_macro = f1_score(all_labels, all_preds, average='macro') * 100
    cm = confusion_matrix(all_labels, all_preds)

    # Accuracy por clase
    class_accuracies = {}
    for idx, name in enumerate(class_names):
        class_acc = cm[idx, idx] / cm[idx].sum() * 100 if cm[idx].sum() > 0 else 0
        class_accuracies[name] = class_acc

    # Identificar errores
    errors = []
    for i, (pred, label, path) in enumerate(zip(all_preds, all_labels, all_paths)):
        if pred != label:
            errors.append({
                'path': path,
                'true_class': class_names[label],
                'pred_class': class_names[pred],
                'true_idx': int(label),
                'pred_idx': int(pred)
            })

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'confusion_matrix': cm.tolist(),
        'class_accuracies': class_accuracies,
        'errors': errors,
        'n_samples': len(all_labels),
        'n_errors': len(errors)
    }


class WarpedDatasetWithPaths(Dataset):
    """Wrapper para ImageFolder que también retorna paths."""

    def __init__(self, image_folder_dataset):
        self.dataset = image_folder_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        path = self.dataset.samples[idx][0]
        return img, label, path


def main():
    print("="*70)
    print("SESIÓN 30: CROSS-EVALUATION")
    print("="*70)
    print("Objetivo: Evaluar cada modelo en AMBOS datasets")

    # Setup
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "error_analysis").mkdir(exist_ok=True)

    # =====================================================
    # 1. CARGAR MODELOS
    # =====================================================
    print("\n1. Cargando modelos...")

    # Modelo Original
    checkpoint_orig = torch.load(MODEL_ORIGINAL_PATH, map_location=device, weights_only=False)
    model_original = create_resnet18(num_classes=3)
    model_original.load_state_dict(checkpoint_orig['model_state_dict'])
    model_original = model_original.to(device)
    model_original.eval()
    print(f"   Modelo Original cargado: {MODEL_ORIGINAL_PATH.name}")

    # Modelo Warped
    checkpoint_warp = torch.load(MODEL_WARPED_PATH, map_location=device, weights_only=False)
    model_warped = create_resnet18(num_classes=3)
    model_warped.load_state_dict(checkpoint_warp['model_state_dict'])
    model_warped = model_warped.to(device)
    model_warped.eval()
    print(f"   Modelo Warped cargado: {MODEL_WARPED_PATH.name}")

    class_names = checkpoint_orig['class_names']
    print(f"   Clases: {class_names}")

    # =====================================================
    # 2. PREPARAR DATASETS
    # =====================================================
    print("\n2. Preparando datasets...")

    eval_transform = get_eval_transform()

    # Test set WARPED (usar ImageFolder)
    warped_test_dir = WARPED_DATASET_DIR / 'test'
    warped_test_base = datasets.ImageFolder(warped_test_dir, transform=eval_transform)
    warped_test_dataset = WarpedDatasetWithPaths(warped_test_base)
    warped_test_loader = DataLoader(warped_test_dataset, batch_size=BATCH_SIZE,
                                     shuffle=False, num_workers=4)
    print(f"   Test Warped: {len(warped_test_dataset)} imágenes")

    # Test set ORIGINAL (crear con splits idénticos)
    original_test_paths = create_original_test_split(seed=SEED)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    original_test_paths_indexed = [(p, class_to_idx[c]) for p, c in original_test_paths]
    original_test_dataset = OriginalDataset(original_test_paths_indexed, transform=eval_transform)
    original_test_loader = DataLoader(original_test_dataset, batch_size=BATCH_SIZE,
                                       shuffle=False, num_workers=4)
    print(f"   Test Original: {len(original_test_dataset)} imágenes")

    # =====================================================
    # 3. EVALUACIÓN CRUZADA
    # =====================================================
    print("\n3. Ejecutando evaluación cruzada...")

    results = {}

    # 3.1 Modelo Original → Test Original (baseline)
    print("\n   [1/4] Modelo Original → Test Original...")
    results['original_on_original'] = evaluate_model(
        model_original, original_test_loader, device, class_names
    )
    print(f"         Accuracy: {results['original_on_original']['accuracy']:.2f}%")

    # 3.2 Modelo Original → Test Warped (CROSS)
    print("\n   [2/4] Modelo Original → Test Warped (CROSS)...")
    results['original_on_warped'] = evaluate_model(
        model_original, warped_test_loader, device, class_names
    )
    print(f"         Accuracy: {results['original_on_warped']['accuracy']:.2f}%")

    # 3.3 Modelo Warped → Test Warped (baseline)
    print("\n   [3/4] Modelo Warped → Test Warped...")
    results['warped_on_warped'] = evaluate_model(
        model_warped, warped_test_loader, device, class_names
    )
    print(f"         Accuracy: {results['warped_on_warped']['accuracy']:.2f}%")

    # 3.4 Modelo Warped → Test Original (CROSS)
    print("\n   [4/4] Modelo Warped → Test Original (CROSS)...")
    results['warped_on_original'] = evaluate_model(
        model_warped, original_test_loader, device, class_names
    )
    print(f"         Accuracy: {results['warped_on_original']['accuracy']:.2f}%")

    # =====================================================
    # 4. CALCULAR GAPS DE GENERALIZACIÓN
    # =====================================================
    print("\n" + "="*70)
    print("4. RESULTADOS DE CROSS-EVALUATION")
    print("="*70)

    # Tabla de resultados
    print(f"\n{'Evaluación':<35} {'Accuracy':>10} {'F1 Macro':>10}")
    print("-" * 60)

    for key in ['original_on_original', 'original_on_warped',
                'warped_on_warped', 'warped_on_original']:
        r = results[key]
        label = key.replace('_', ' ').title()
        print(f"{label:<35} {r['accuracy']:>9.2f}% {r['f1_macro']:>9.2f}%")

    # Calcular gaps
    gap_original = results['original_on_original']['accuracy'] - results['original_on_warped']['accuracy']
    gap_warped = results['warped_on_warped']['accuracy'] - results['warped_on_original']['accuracy']

    print("\n" + "-"*60)
    print(f"{'Gap Original (O→O vs O→W)':<35} {gap_original:>+9.2f}%")
    print(f"{'Gap Warped (W→W vs W→O)':<35} {gap_warped:>+9.2f}%")

    # Generalización cruzada
    cross_original_to_warped = results['original_on_warped']['accuracy']
    cross_warped_to_original = results['warped_on_original']['accuracy']

    print("\n" + "="*70)
    print("5. ANÁLISIS DE GENERALIZACIÓN")
    print("="*70)

    print(f"""
┌─────────────────────────────────────────────────────────────────┐
│ MATRIZ DE CROSS-EVALUATION                                      │
├─────────────────────────────────────────────────────────────────┤
│                     Test Original    Test Warped                │
│ Modelo Original     {results['original_on_original']['accuracy']:>6.2f}%         {results['original_on_warped']['accuracy']:>6.2f}%           │
│ Modelo Warped       {results['warped_on_original']['accuracy']:>6.2f}%         {results['warped_on_warped']['accuracy']:>6.2f}%           │
├─────────────────────────────────────────────────────────────────┤
│ Gap (baseline - cross)                                          │
│ Original: {gap_original:>+6.2f}%    Warped: {gap_warped:>+6.2f}%                           │
└─────────────────────────────────────────────────────────────────┘
""")

    # Determinar ganador en generalización
    if gap_warped < gap_original:
        generalization_winner = "WARPED"
        generalization_msg = f"El modelo Warped generaliza MEJOR (gap {gap_warped:.2f}% vs {gap_original:.2f}%)"
    elif gap_original < gap_warped:
        generalization_winner = "ORIGINAL"
        generalization_msg = f"El modelo Original generaliza MEJOR (gap {gap_original:.2f}% vs {gap_warped:.2f}%)"
    else:
        generalization_winner = "EMPATE"
        generalization_msg = "Ambos modelos generalizan igual"

    print(f"CONCLUSIÓN: {generalization_msg}")

    # Analizar rendimiento cruzado absoluto
    print(f"\nRendimiento en datos 'ajenos':")
    print(f"  - Original en datos warped: {cross_original_to_warped:.2f}%")
    print(f"  - Warped en datos originales: {cross_warped_to_original:.2f}%")

    if cross_warped_to_original > cross_original_to_warped:
        print(f"  → Warped tiene MEJOR rendimiento absoluto en cross-evaluation")
    elif cross_original_to_warped > cross_warped_to_original:
        print(f"  → Original tiene MEJOR rendimiento absoluto en cross-evaluation")
    else:
        print(f"  → Rendimiento cruzado idéntico")

    # =====================================================
    # 6. GUARDAR RESULTADOS
    # =====================================================
    print("\n6. Guardando resultados...")

    # Preparar resultados para JSON (sin errores detallados en el archivo principal)
    results_summary = {}
    for key, val in results.items():
        results_summary[key] = {
            'accuracy': val['accuracy'],
            'f1_macro': val['f1_macro'],
            'confusion_matrix': val['confusion_matrix'],
            'class_accuracies': val['class_accuracies'],
            'n_samples': val['n_samples'],
            'n_errors': val['n_errors']
        }

    cross_eval_results = {
        'experiment': 'Cross-Evaluation Session 30',
        'timestamp': datetime.now().isoformat(),
        'models': {
            'original': str(MODEL_ORIGINAL_PATH),
            'warped': str(MODEL_WARPED_PATH)
        },
        'results': results_summary,
        'gaps': {
            'original_gap': gap_original,
            'warped_gap': gap_warped,
            'generalization_winner': generalization_winner
        },
        'cross_performance': {
            'original_on_warped': cross_original_to_warped,
            'warped_on_original': cross_warped_to_original,
            'winner_absolute': 'warped' if cross_warped_to_original > cross_original_to_warped else
                               ('original' if cross_original_to_warped > cross_warped_to_original else 'tie')
        },
        'conclusion': generalization_msg
    }

    # Guardar JSON principal
    json_path = OUTPUT_DIR / 'cross_evaluation_results.json'
    with open(json_path, 'w') as f:
        json.dump(cross_eval_results, f, indent=2)
    print(f"   Resultados guardados: {json_path}")

    # Guardar errores detallados en archivos separados
    for key in ['original_on_warped', 'warped_on_original']:
        errors_path = OUTPUT_DIR / "error_analysis" / f"errors_{key}.json"
        with open(errors_path, 'w') as f:
            json.dump({
                'evaluation': key,
                'n_errors': results[key]['n_errors'],
                'n_total': results[key]['n_samples'],
                'error_rate': results[key]['n_errors'] / results[key]['n_samples'] * 100,
                'errors': results[key]['errors']
            }, f, indent=2)
        print(f"   Errores {key}: {errors_path}")

    print("\n" + "="*70)
    print("CROSS-EVALUATION COMPLETADO")
    print("="*70)

    return cross_eval_results


if __name__ == "__main__":
    main()
