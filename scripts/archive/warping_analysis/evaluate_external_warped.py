#!/usr/bin/env python3
"""
Sesion 37: Evaluacion de modelos WARPED en Dataset3 warpeado.

Este script evalua los clasificadores WARPED en el Dataset3 warpeado
para una comparacion justa con los modelos ORIGINAL.

Comparacion justa:
- Modelos ORIGINAL -> evaluados en Dataset3 original (ya hecho en Sesion 36)
- Modelos WARPED -> evaluados en Dataset3 warpeado (este script)

Metricas:
- Accuracy
- Sensibilidad COVID (Recall de positivos)
- Especificidad (Recall de negativos)
- AUC-ROC
- F1-Score

Autor: Proyecto Tesis Maestria
Fecha: 03-Dic-2024
Sesion: 37
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class WarpedExternalDataset(Dataset):
    """Dataset para cargar imagenes warpeadas de Dataset3."""

    def __init__(self, data_dir, split='test', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.class_to_idx = {'positive': 1, 'negative': 0}
        self.classes = ['negative', 'positive']
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        split_dir = self.data_dir / self.split
        for label in ['positive', 'negative']:
            label_dir = split_dir / label
            if not label_dir.exists():
                continue
            for img_path in label_dir.glob('*.png'):
                self.samples.append((img_path, self.class_to_idx[label]))
        self.targets = [s[1] for s in self.samples]
        print(f"  Cargadas {len(self.samples)} imagenes warpeadas para {self.split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class GrayscaleToRGB:
    def __call__(self, img):
        if img.mode == 'RGB':
            return img
        return img.convert('RGB')


def get_transforms(img_size=224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    return transforms.Compose([
        GrayscaleToRGB(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])


def create_model(model_name, num_classes=3, checkpoint_path=None):
    """Crea y carga un modelo."""
    if model_name == 'resnet18':
        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_features, num_classes))
    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_features, num_classes))
    elif model_name == 'densenet121':
        model = models.densenet121(weights=None)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_features, num_classes))
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_features, num_classes))
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_features, num_classes))
    elif model_name == 'alexnet':
        model = models.alexnet(weights=None)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_features, num_classes))
    elif model_name == 'vgg16':
        model = models.vgg16(weights=None)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_features, num_classes))
    else:
        raise ValueError(f"Modelo '{model_name}' no soportado")

    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    return model


def evaluate_binary(model, dataloader, device):
    """
    Evalua modelo de 3 clases en dataset binario.

    Mapeo de probabilidades:
    - P(positive) = P(COVID) = softmax[0]
    - P(negative) = P(Normal) + P(Viral_Pneumonia) = softmax[1] + softmax[2]
    """
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluando", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1).cpu().numpy()

            prob_positive = probs[:, 0]  # P(COVID)
            preds = (prob_positive > 0.5).astype(int)

            all_probs.extend(prob_positive)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    sensitivity = recall_score(all_labels, all_preds, pos_label=1)
    specificity = recall_score(all_labels, all_preds, pos_label=0)
    precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    f1 = f1_score(all_labels, all_preds, pos_label=1)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': float(accuracy),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'f1_score': float(f1),
        'auc_roc': float(auc),
        'confusion_matrix': cm.tolist(),
        'n_samples': len(all_labels),
        'n_positive': int(sum(all_labels)),
        'n_negative': int(len(all_labels) - sum(all_labels))
    }


def find_warped_models(classifier_dir):
    """Encuentra modelos WARPED disponibles."""
    models = []
    classifier_path = Path(classifier_dir)

    if not classifier_path.exists():
        return models

    for model_dir in classifier_path.iterdir():
        if not model_dir.is_dir():
            continue

        # Solo modelos warped
        if not model_dir.name.endswith('_warped'):
            continue

        checkpoint = model_dir / 'best_model.pt'
        results_file = model_dir / 'results.json'

        if not checkpoint.exists():
            continue

        parts = model_dir.name.rsplit('_', 1)
        arch_name = parts[0] if len(parts) == 2 else model_dir.name

        train_metrics = {}
        if results_file.exists():
            with open(results_file) as f:
                train_results = json.load(f)
                train_metrics = train_results.get('test_metrics', {})

        models.append({
            'name': model_dir.name,
            'architecture': arch_name,
            'dataset_type': 'warped',
            'checkpoint': str(checkpoint),
            'train_accuracy': train_metrics.get('accuracy', 0),
            'train_f1': train_metrics.get('f1_macro', 0)
        })

    return models


def main():
    parser = argparse.ArgumentParser(
        description='Evaluar modelos WARPED en Dataset3 warpeado'
    )
    parser.add_argument('--warped-data-dir', type=str,
                       default='outputs/external_validation/dataset3_warped',
                       help='Directorio con Dataset3 warpeado')
    parser.add_argument('--classifier-dir', type=str,
                       default='outputs/classifier_comparison',
                       help='Directorio con modelos entrenados')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/external_validation',
                       help='Directorio para guardar resultados')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("EVALUACION DE MODELOS WARPED EN DATASET3 WARPEADO")
    print("=" * 70)
    print(f"\nDispositivo: {device}")

    warped_data_dir = PROJECT_ROOT / args.warped_data_dir
    classifier_dir = PROJECT_ROOT / args.classifier_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verificar que existe Dataset3 warpeado
    if not (warped_data_dir / 'test').exists():
        print(f"\nError: No se encuentra Dataset3 warpeado en {warped_data_dir}")
        print("Ejecuta primero: python scripts/warp_dataset3.py")
        return

    # Cargar dataset warpeado
    print(f"\nCargando Dataset3 warpeado desde: {warped_data_dir}")
    transform = get_transforms(img_size=224)
    warped_dataset = WarpedExternalDataset(
        data_dir=warped_data_dir,
        split='test',
        transform=transform
    )

    warped_loader = DataLoader(
        warped_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    print(f"Total imagenes warpeadas: {len(warped_dataset)}")
    print(f"  Positive (COVID): {sum(warped_dataset.targets)}")
    print(f"  Negative: {len(warped_dataset) - sum(warped_dataset.targets)}")

    # Encontrar modelos WARPED
    print(f"\nBuscando modelos WARPED en: {classifier_dir}")
    warped_models = find_warped_models(classifier_dir)

    if not warped_models:
        print("No se encontraron modelos WARPED para evaluar")
        return

    print(f"Modelos WARPED a evaluar: {len(warped_models)}")
    for m in warped_models:
        print(f"  - {m['name']} (train acc: {m['train_accuracy']*100:.1f}%)")

    # Evaluar cada modelo
    results = []

    for model_info in warped_models:
        print(f"\n{'='*60}")
        print(f"Evaluando: {model_info['name']}")
        print(f"{'='*60}")

        try:
            model = create_model(
                model_name=model_info['architecture'],
                num_classes=3,
                checkpoint_path=model_info['checkpoint']
            )
            model = model.to(device)

            metrics = evaluate_binary(model, warped_loader, device)

            print(f"\nResultados en Dataset3 WARPEADO:")
            print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
            print(f"  Sensibilidad (COVID): {metrics['sensitivity']*100:.2f}%")
            print(f"  Especificidad: {metrics['specificity']*100:.2f}%")
            print(f"  AUC-ROC: {metrics['auc_roc']*100:.2f}%")
            print(f"  F1-Score: {metrics['f1_score']*100:.2f}%")

            train_acc = model_info['train_accuracy']
            ext_acc = metrics['accuracy']
            gap = train_acc - ext_acc

            print(f"\n  Gap de generalizacion:")
            print(f"    Train acc: {train_acc*100:.2f}%")
            print(f"    Test acc (warped externo): {ext_acc*100:.2f}%")
            print(f"    Gap: {gap*100:.2f}%")

            results.append({
                'model_name': model_info['name'],
                'architecture': model_info['architecture'],
                'dataset_type': 'warped',
                'eval_dataset': 'dataset3_warped',
                'train_accuracy': train_acc,
                'train_f1': model_info['train_f1'],
                'metrics': metrics,
                'gap': gap
            })

        except Exception as e:
            print(f"Error evaluando {model_info['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Ordenar por accuracy
    results.sort(key=lambda x: x['metrics']['accuracy'], reverse=True)

    # Guardar resultados
    results_path = output_dir / 'warped_on_warped_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'evaluation_type': 'warped_models_on_warped_dataset3',
            'n_samples': len(warped_dataset),
            'results': results
        }, f, indent=2)
    print(f"\nResultados guardados en: {results_path}")

    # Cargar resultados baseline para comparacion
    baseline_path = output_dir / 'baseline_results.json'
    baseline_results = {}
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline_data = json.load(f)
            for r in baseline_data.get('results', []):
                baseline_results[r['model_name']] = r

    # Tabla resumen comparativa
    print("\n" + "=" * 90)
    print("COMPARACION: WARPED en D3_warped vs WARPED en D3_original")
    print("=" * 90)
    print(f"\n{'Modelo':<25} {'D3_orig%':>10} {'D3_warp%':>10} {'Diff%':>8} | {'Gap_orig%':>10} {'Gap_warp%':>10}")
    print("-" * 90)

    for r in results:
        model_name = r['model_name']
        acc_warped = r['metrics']['accuracy'] * 100
        gap_warped = r['gap'] * 100

        # Buscar resultado en D3 original
        if model_name in baseline_results:
            acc_original = baseline_results[model_name]['metrics']['accuracy'] * 100
            gap_original = baseline_results[model_name]['gap'] * 100
            diff = acc_warped - acc_original
            print(f"{model_name:<25} {acc_original:>10.1f} {acc_warped:>10.1f} {diff:>+8.1f} | {gap_original:>10.1f} {gap_warped:>10.1f}")
        else:
            print(f"{model_name:<25} {'N/A':>10} {acc_warped:>10.1f} {'N/A':>8} | {'N/A':>10} {gap_warped:>10.1f}")

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)

    avg_acc = np.mean([r['metrics']['accuracy'] for r in results])
    avg_gap = np.mean([r['gap'] for r in results])

    print(f"\nModelos WARPED en Dataset3 WARPEADO:")
    print(f"  Accuracy promedio: {avg_acc*100:.1f}%")
    print(f"  Gap promedio: {avg_gap*100:.1f}%")

    if baseline_results:
        warped_baseline = [v for k, v in baseline_results.items() if k.endswith('_warped')]
        if warped_baseline:
            avg_acc_orig = np.mean([r['metrics']['accuracy'] for r in warped_baseline])
            avg_gap_orig = np.mean([r['gap'] for r in warped_baseline])

            print(f"\nModelos WARPED en Dataset3 ORIGINAL (Sesion 36):")
            print(f"  Accuracy promedio: {avg_acc_orig*100:.1f}%")
            print(f"  Gap promedio: {avg_gap_orig*100:.1f}%")

            improvement = avg_acc - avg_acc_orig
            print(f"\nMejora al usar Dataset3 warpeado: {improvement*100:+.1f}%")

    print("=" * 70)


if __name__ == '__main__':
    main()
