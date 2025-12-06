#!/usr/bin/env python3
"""
Sesion 37: Analisis de mapeo de clases y estrategias alternativas.

Este script:
1. Analiza las matrices de confusion de modelos entrenados (3 clases)
   para entender cuanto confunden COVID vs Viral_Pneumonia vs Normal
2. Evalua diferentes estrategias de mapeo para clasificacion binaria
3. Compara resultados en Dataset3 con cada estrategia

Estrategias de mapeo evaluadas:
- Opcion A: P(negative) = P(Normal) + P(Viral_Pneumonia)  [actual]
- Opcion B: P(negative) = P(Normal) solamente
- Opcion C: Excluir predicciones donde P(Viral_Pneumonia) > umbral
- Opcion D: Ponderar: P(negative) = P(Normal) + alpha*P(Viral_Pneumonia)
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

# Agregar directorio raiz al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def analyze_confusion_matrices():
    """
    Analiza las matrices de confusion de modelos entrenados para entender
    la confusion entre COVID, Viral_Pneumonia y Normal.
    """
    print("=" * 70)
    print("ANALISIS DE CONFUSION ENTRE CLASES (Modelos entrenados, 3 clases)")
    print("=" * 70)

    classifier_dir = PROJECT_ROOT / 'outputs' / 'classifier_comparison'

    # Recopilar matrices de confusion
    confusion_data = []

    for model_dir in sorted(classifier_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        results_file = model_dir / 'results.json'
        if not results_file.exists():
            continue

        with open(results_file) as f:
            data = json.load(f)

        cm = np.array(data['confusion_matrix'])
        class_names = data.get('class_names', ['COVID', 'Normal', 'Viral_Pneumonia'])

        confusion_data.append({
            'model': model_dir.name,
            'dataset': data.get('dataset', 'unknown'),
            'confusion_matrix': cm,
            'class_names': class_names
        })

    # Analizar patrones de confusion
    print("\n" + "=" * 70)
    print("PATRON DE CONFUSION POR MODELO")
    print("=" * 70)

    # Orden de clases: [COVID, Normal, Viral_Pneumonia]
    # cm[i,j] = predichos como j cuando el real es i

    total_stats = {
        'covid_as_normal': 0,
        'covid_as_viral': 0,
        'normal_as_covid': 0,
        'normal_as_viral': 0,
        'viral_as_covid': 0,
        'viral_as_normal': 0,
        'total_covid': 0,
        'total_normal': 0,
        'total_viral': 0
    }

    print(f"\n{'Modelo':<28} | COVID->Normal | COVID->Viral | Normal->COVID | Normal->Viral | Viral->COVID | Viral->Normal")
    print("-" * 140)

    for item in confusion_data:
        cm = item['confusion_matrix']
        model = item['model']

        # Extraer confusiones (asumiendo orden [COVID, Normal, Viral_Pneumonia])
        covid_as_normal = cm[0, 1]  # COVID predicho como Normal
        covid_as_viral = cm[0, 2]   # COVID predicho como Viral_Pneumonia
        normal_as_covid = cm[1, 0]  # Normal predicho como COVID
        normal_as_viral = cm[1, 2]  # Normal predicho como Viral_Pneumonia
        viral_as_covid = cm[2, 0]   # Viral predicho como COVID
        viral_as_normal = cm[2, 1]  # Viral predicho como Normal

        total_covid = cm[0].sum()
        total_normal = cm[1].sum()
        total_viral = cm[2].sum()

        # Acumular
        total_stats['covid_as_normal'] += covid_as_normal
        total_stats['covid_as_viral'] += covid_as_viral
        total_stats['normal_as_covid'] += normal_as_covid
        total_stats['normal_as_viral'] += normal_as_viral
        total_stats['viral_as_covid'] += viral_as_covid
        total_stats['viral_as_normal'] += viral_as_normal
        total_stats['total_covid'] += total_covid
        total_stats['total_normal'] += total_normal
        total_stats['total_viral'] += total_viral

        print(f"{model:<28} | {covid_as_normal:>6}/{total_covid:<3} | "
              f"{covid_as_viral:>6}/{total_covid:<3} | "
              f"{normal_as_covid:>6}/{total_normal:<3} | "
              f"{normal_as_viral:>6}/{total_normal:<3} | "
              f"{viral_as_covid:>6}/{total_viral:<3} | "
              f"{viral_as_normal:>6}/{total_viral:<3}")

    # Resumen agregado
    print("\n" + "=" * 70)
    print("RESUMEN AGREGADO DE CONFUSIONES")
    print("=" * 70)

    print("\n1. CONFUSION COVID -> otras clases:")
    total_covid = total_stats['total_covid']
    covid_to_normal_pct = 100 * total_stats['covid_as_normal'] / total_covid
    covid_to_viral_pct = 100 * total_stats['covid_as_viral'] / total_covid
    print(f"   COVID -> Normal:           {total_stats['covid_as_normal']:>4}/{total_covid} ({covid_to_normal_pct:.1f}%)")
    print(f"   COVID -> Viral_Pneumonia:  {total_stats['covid_as_viral']:>4}/{total_covid} ({covid_to_viral_pct:.1f}%)")

    print("\n2. CONFUSION otras clases -> COVID (falsos positivos):")
    total_normal = total_stats['total_normal']
    total_viral = total_stats['total_viral']
    normal_to_covid_pct = 100 * total_stats['normal_as_covid'] / total_normal
    viral_to_covid_pct = 100 * total_stats['viral_as_covid'] / total_viral
    print(f"   Normal -> COVID:           {total_stats['normal_as_covid']:>4}/{total_normal} ({normal_to_covid_pct:.1f}%)")
    print(f"   Viral_Pneumonia -> COVID:  {total_stats['viral_as_covid']:>4}/{total_viral} ({viral_to_covid_pct:.1f}%)")

    print("\n3. INTERPRETACION PARA MAPEO BINARIO:")

    if viral_to_covid_pct > normal_to_covid_pct * 1.5:
        print("   ⚠️  ALERTA: Viral_Pneumonia se confunde SIGNIFICATIVAMENTE mas con COVID que Normal")
        print(f"      Viral->COVID ({viral_to_covid_pct:.1f}%) vs Normal->COVID ({normal_to_covid_pct:.1f}%)")
        print("      -> El mapeo P(neg)=P(Normal)+P(Viral) puede ser PROBLEMATICO")
        print("      -> Considerar solo P(neg)=P(Normal) o ponderacion")
        recommendation = 'B'  # Solo Normal
    elif viral_to_covid_pct > normal_to_covid_pct:
        print("   ⚠️  Viral_Pneumonia se confunde algo mas con COVID que Normal")
        print(f"      Viral->COVID ({viral_to_covid_pct:.1f}%) vs Normal->COVID ({normal_to_covid_pct:.1f}%)")
        print("      -> Considerar estrategia D (ponderacion)")
        recommendation = 'D'  # Ponderacion
    else:
        print("   ✓  La confusion Viral->COVID NO es mayor que Normal->COVID")
        print(f"      Viral->COVID ({viral_to_covid_pct:.1f}%) vs Normal->COVID ({normal_to_covid_pct:.1f}%)")
        print("      -> El mapeo actual P(neg)=P(Normal)+P(Viral) es RAZONABLE")
        recommendation = 'A'  # Actual

    return {
        'total_stats': total_stats,
        'covid_to_normal_pct': covid_to_normal_pct,
        'covid_to_viral_pct': covid_to_viral_pct,
        'normal_to_covid_pct': normal_to_covid_pct,
        'viral_to_covid_pct': viral_to_covid_pct,
        'recommendation': recommendation
    }


class ExternalDataset(Dataset):
    """Dataset para cargar imagenes preprocesadas de Dataset3."""

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


def get_raw_probabilities(model, dataloader, device):
    """
    Obtiene probabilidades raw de 3 clases para todas las muestras.

    Returns:
        probs: array (N, 3) con probabilidades [P(COVID), P(Normal), P(Viral)]
        labels: array (N,) con etiquetas reales (0=negative, 1=positive)
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Extrayendo probs", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(labels.numpy())

    return np.vstack(all_probs), np.array(all_labels)


def evaluate_mapping_strategy(probs, labels, strategy, alpha=0.5, threshold=0.3):
    """
    Evalua una estrategia de mapeo.

    Estrategias:
    - 'A': P(neg) = P(Normal) + P(Viral)  [actual]
    - 'B': P(neg) = P(Normal) solamente
    - 'C': Excluir donde P(Viral) > threshold
    - 'D': P(neg) = P(Normal) + alpha * P(Viral)

    Returns:
        dict con metricas y estadisticas
    """
    # probs[:, 0] = P(COVID)
    # probs[:, 1] = P(Normal)
    # probs[:, 2] = P(Viral_Pneumonia)

    if strategy == 'A':
        # Opcion A: P(negative) = P(Normal) + P(Viral_Pneumonia)
        prob_positive = probs[:, 0]
        prob_negative = probs[:, 1] + probs[:, 2]
        mask = np.ones(len(labels), dtype=bool)

    elif strategy == 'B':
        # Opcion B: P(negative) = P(Normal) solamente
        # Renormalizar: P(pos) = P(COVID)/(P(COVID)+P(Normal))
        prob_covid = probs[:, 0]
        prob_normal = probs[:, 1]
        total = prob_covid + prob_normal
        prob_positive = prob_covid / (total + 1e-8)
        prob_negative = prob_normal / (total + 1e-8)
        mask = np.ones(len(labels), dtype=bool)

    elif strategy == 'C':
        # Opcion C: Excluir donde P(Viral) > threshold
        mask = probs[:, 2] <= threshold
        prob_positive = probs[:, 0]
        prob_negative = probs[:, 1] + probs[:, 2]

    elif strategy == 'D':
        # Opcion D: Ponderar P(negative) = P(Normal) + alpha*P(Viral)
        prob_positive = probs[:, 0]
        prob_negative = probs[:, 1] + alpha * probs[:, 2]
        # Renormalizar
        total = prob_positive + prob_negative
        prob_positive = prob_positive / (total + 1e-8)
        prob_negative = prob_negative / (total + 1e-8)
        mask = np.ones(len(labels), dtype=bool)

    else:
        raise ValueError(f"Estrategia '{strategy}' no reconocida")

    # Aplicar mascara
    prob_pos = prob_positive[mask]
    prob_neg = prob_negative[mask]
    y_true = labels[mask]
    n_excluded = (~mask).sum()

    # Prediccion: positive si P(COVID) > 0.5
    y_pred = (prob_pos > 0.5).astype(int)

    # Calcular metricas
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    try:
        auc = roc_auc_score(y_true, prob_pos)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': float(accuracy),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'f1_score': float(f1),
        'auc_roc': float(auc),
        'confusion_matrix': cm.tolist() if len(cm) > 0 else [],
        'n_samples': int(mask.sum()),
        'n_excluded': int(n_excluded),
        'n_positive': int(y_true.sum()),
        'n_negative': int(len(y_true) - y_true.sum())
    }


def evaluate_all_strategies(probs, labels, model_name):
    """Evalua todas las estrategias de mapeo para un modelo."""

    strategies = {
        'A': {'desc': 'P(neg) = P(Normal) + P(Viral)', 'params': {}},
        'B': {'desc': 'P(neg) = P(Normal) only', 'params': {}},
        'C_0.2': {'desc': 'Excluir P(Viral) > 0.2', 'params': {'threshold': 0.2}},
        'C_0.3': {'desc': 'Excluir P(Viral) > 0.3', 'params': {'threshold': 0.3}},
        'C_0.4': {'desc': 'Excluir P(Viral) > 0.4', 'params': {'threshold': 0.4}},
        'D_0.3': {'desc': 'P(neg) = P(Normal) + 0.3*P(Viral)', 'params': {'alpha': 0.3}},
        'D_0.5': {'desc': 'P(neg) = P(Normal) + 0.5*P(Viral)', 'params': {'alpha': 0.5}},
        'D_0.7': {'desc': 'P(neg) = P(Normal) + 0.7*P(Viral)', 'params': {'alpha': 0.7}},
    }

    results = {}
    for strat_name, strat_info in strategies.items():
        base_strat = strat_name.split('_')[0]
        params = strat_info['params']

        metrics = evaluate_mapping_strategy(probs, labels, base_strat, **params)
        metrics['description'] = strat_info['desc']
        results[strat_name] = metrics

    return results


def find_best_models():
    """Encuentra los mejores modelos original y warped."""
    classifier_dir = PROJECT_ROOT / 'outputs' / 'classifier_comparison'

    best_original = None
    best_warped = None
    best_original_f1 = 0
    best_warped_f1 = 0

    for model_dir in classifier_dir.iterdir():
        if not model_dir.is_dir():
            continue

        results_file = model_dir / 'results.json'
        checkpoint = model_dir / 'best_model.pt'

        if not results_file.exists() or not checkpoint.exists():
            continue

        with open(results_file) as f:
            data = json.load(f)

        f1 = data.get('test_metrics', {}).get('f1_macro', 0)
        dataset_type = data.get('dataset', 'unknown')

        parts = model_dir.name.rsplit('_', 1)
        arch_name = parts[0] if len(parts) == 2 else model_dir.name

        model_info = {
            'name': model_dir.name,
            'architecture': arch_name,
            'checkpoint': str(checkpoint),
            'f1': f1
        }

        if dataset_type == 'original' and f1 > best_original_f1:
            best_original = model_info
            best_original_f1 = f1
        elif dataset_type == 'warped' and f1 > best_warped_f1:
            best_warped = model_info
            best_warped_f1 = f1

    return best_original, best_warped


def main():
    parser = argparse.ArgumentParser(description='Analisis de mapeo de clases')
    parser.add_argument('--external-data-dir', type=str,
                       default='outputs/external_validation/dataset3',
                       help='Directorio con Dataset3 procesado')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/external_validation',
                       help='Directorio para guardar resultados')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--skip-confusion-analysis', action='store_true',
                       help='Omitir analisis de matrices de confusion')
    args = parser.parse_args()

    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")

    # 1. Analizar matrices de confusion de modelos entrenados
    if not args.skip_confusion_analysis:
        confusion_analysis = analyze_confusion_matrices()
    else:
        confusion_analysis = {'recommendation': 'A'}

    # 2. Evaluar estrategias de mapeo en Dataset3
    print("\n" + "=" * 70)
    print("EVALUACION DE ESTRATEGIAS DE MAPEO EN DATASET3")
    print("=" * 70)

    external_data_dir = PROJECT_ROOT / args.external_data_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (external_data_dir / 'test').exists():
        print(f"Error: No se encuentra Dataset3 en {external_data_dir}")
        print("Ejecuta primero: python scripts/prepare_dataset3.py")
        return

    # Cargar dataset
    print(f"\nCargando Dataset3 desde: {external_data_dir}")
    transform = get_transforms(img_size=224)
    dataset = ExternalDataset(external_data_dir, split='test', transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"  Total: {len(dataset)} imagenes")
    print(f"  Positive: {sum(dataset.targets)}, Negative: {len(dataset) - sum(dataset.targets)}")

    # Encontrar mejores modelos
    best_original, best_warped = find_best_models()

    models_to_evaluate = []
    if best_original:
        models_to_evaluate.append(('original', best_original))
    if best_warped:
        models_to_evaluate.append(('warped', best_warped))

    # Evaluar cada modelo con todas las estrategias
    all_results = {}

    for dtype, model_info in models_to_evaluate:
        print(f"\n{'='*60}")
        print(f"Evaluando: {model_info['name']}")
        print(f"{'='*60}")

        model = create_model(
            model_info['architecture'],
            num_classes=3,
            checkpoint_path=model_info['checkpoint']
        )
        model = model.to(device)

        # Obtener probabilidades
        probs, labels = get_raw_probabilities(model, dataloader, device)

        # Evaluar todas las estrategias
        strategy_results = evaluate_all_strategies(probs, labels, model_info['name'])
        all_results[model_info['name']] = strategy_results

        # Mostrar resultados
        print(f"\n{'Estrategia':<35} | {'Acc%':>6} | {'Sens%':>6} | {'Spec%':>6} | {'AUC%':>6} | {'Excl':>5}")
        print("-" * 80)
        for strat, metrics in strategy_results.items():
            print(f"{metrics['description']:<35} | "
                  f"{metrics['accuracy']*100:>6.1f} | "
                  f"{metrics['sensitivity']*100:>6.1f} | "
                  f"{metrics['specificity']*100:>6.1f} | "
                  f"{metrics['auc_roc']*100:>6.1f} | "
                  f"{metrics['n_excluded']:>5}")

    # Guardar resultados
    results_path = output_dir / 'mapping_analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'confusion_analysis': {
                k: v for k, v in confusion_analysis.items()
                if k != 'total_stats' or isinstance(v, (int, float, str))
            },
            'strategy_comparison': all_results
        }, f, indent=2, default=str)
    print(f"\nResultados guardados en: {results_path}")

    # Analisis final y recomendacion
    print("\n" + "=" * 70)
    print("ANALISIS FINAL Y RECOMENDACION")
    print("=" * 70)

    # Encontrar mejor estrategia por modelo
    for model_name, strategies in all_results.items():
        best_strat = max(strategies.items(), key=lambda x: x[1]['auc_roc'])
        baseline_a = strategies['A']

        print(f"\n{model_name}:")
        print(f"  Baseline (A): AUC={baseline_a['auc_roc']*100:.1f}%, "
              f"Acc={baseline_a['accuracy']*100:.1f}%")
        print(f"  Mejor estrategia: {best_strat[0]} ({best_strat[1]['description']})")
        print(f"    AUC={best_strat[1]['auc_roc']*100:.1f}%, "
              f"Acc={best_strat[1]['accuracy']*100:.1f}%")

        improvement = best_strat[1]['auc_roc'] - baseline_a['auc_roc']
        if improvement > 0.01:
            print(f"    -> Mejora de {improvement*100:.1f}% en AUC vs baseline")
        else:
            print(f"    -> Sin mejora significativa vs baseline")

    print("\n" + "=" * 70)
    print("CONCLUSION:")

    # Basado en el analisis de confusion
    rec = confusion_analysis.get('recommendation', 'A')
    if rec == 'A':
        print("  El mapeo actual (A) es VALIDO y apropiado.")
        print("  P(negative) = P(Normal) + P(Viral_Pneumonia) es correcto")
        print("  porque Viral_Pneumonia no se confunde mas con COVID que Normal.")
    elif rec == 'B':
        print("  Se recomienda usar estrategia B: P(negative) = P(Normal) solamente")
        print("  porque Viral_Pneumonia se confunde significativamente con COVID.")
    else:
        print(f"  Se recomienda evaluar estrategia D (ponderacion)")
        print("  porque hay cierta confusion entre Viral_Pneumonia y COVID.")

    print("=" * 70)


if __name__ == '__main__':
    main()
