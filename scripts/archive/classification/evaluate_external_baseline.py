#!/usr/bin/env python3
"""
Sesion 36: Evaluacion de modelos existentes en Dataset3 (FedCOVIDx).

Este script evalua clasificadores de 3 clases en un dataset externo de 2 clases.

Estrategia de mapeo (3 clases -> 2 clases):
- P(positive) = P(COVID)
- P(negative) = P(Normal) + P(Viral_Pneumonia)

Modelos a evaluar:
- Original (ResNet-18, entrenado en 15K imagenes originales)
- Warped (DenseNet-121, entrenado en imagenes warpeadas con margin 1.05)
- Otros modelos del directorio classifier_comparison

Metricas:
- Accuracy
- Sensibilidad COVID (Recall de positivos)
- Especificidad (Recall de negativos)
- AUC-ROC
- F1-Score

Output:
- outputs/external_validation/baseline_results.json
- outputs/external_validation/baseline_comparison.png
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
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Agregar directorio raiz al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class ExternalDataset(Dataset):
    """Dataset para cargar imagenes preprocesadas de Dataset3."""

    def __init__(self, data_dir, split='test', transform=None):
        """
        Args:
            data_dir: Directorio con imagenes procesadas (outputs/external_validation/dataset3)
            split: 'train', 'val', o 'test'
            transform: Transformaciones a aplicar
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        # Clases del dataset externo (binario)
        self.class_to_idx = {'positive': 1, 'negative': 0}
        self.classes = ['negative', 'positive']  # Orden por indice

        # Cargar metadata si existe
        metadata_path = self.data_dir / split / 'metadata.csv'
        if metadata_path.exists():
            self.df = pd.read_csv(metadata_path)
            self.use_metadata = True
        else:
            self.use_metadata = False

        # Construir lista de muestras
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        """Carga las rutas de imagenes y etiquetas."""
        split_dir = self.data_dir / self.split

        for label in ['positive', 'negative']:
            label_dir = split_dir / label
            if not label_dir.exists():
                continue

            for img_path in label_dir.glob('*.png'):
                self.samples.append((img_path, self.class_to_idx[label]))

        self.targets = [s[1] for s in self.samples]
        print(f"  Cargadas {len(self.samples)} imagenes para {self.split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path)

        # Convertir a RGB para compatibilidad con modelos ImageNet
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class GrayscaleToRGB:
    """Convierte imagen grayscale a RGB."""
    def __call__(self, img):
        if img.mode == 'RGB':
            return img
        return img.convert('RGB')


def get_transforms(img_size=224):
    """Transformaciones para evaluacion (sin augmentacion)."""
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
    """
    Crea y carga un modelo.

    Args:
        model_name: Nombre de la arquitectura
        num_classes: Numero de clases de salida
        checkpoint_path: Ruta al checkpoint del modelo

    Returns:
        Modelo cargado
    """
    # Crear arquitectura base
    if model_name == 'resnet18':
        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    elif model_name == 'densenet121':
        model = models.densenet121(weights=None)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    elif model_name == 'alexnet':
        model = models.alexnet(weights=None)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    elif model_name == 'vgg16':
        model = models.vgg16(weights=None)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    else:
        raise ValueError(f"Modelo '{model_name}' no soportado")

    # Cargar pesos si se proporciona checkpoint
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"  Cargado checkpoint: {checkpoint_path}")

    return model


def evaluate_binary(model, dataloader, device):
    """
    Evalua modelo de 3 clases en dataset binario.

    Mapeo de probabilidades:
    - P(positive) = P(COVID) = softmax[0]
    - P(negative) = P(Normal) + P(Viral_Pneumonia) = softmax[1] + softmax[2]

    Returns:
        dict con metricas de evaluacion
    """
    model.eval()

    all_probs = []  # Probabilidades de clase positiva (COVID)
    all_preds = []  # Predicciones binarias
    all_labels = []  # Etiquetas reales

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluando", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Aplicar softmax para obtener probabilidades
            probs = F.softmax(outputs, dim=1).cpu().numpy()

            # Mapeo 3 clases -> 2 clases
            # Clase 0 del modelo = COVID = positive (1)
            # Clases 1,2 del modelo = Normal + Viral_Pneumonia = negative (0)
            prob_positive = probs[:, 0]  # P(COVID)
            prob_negative = probs[:, 1] + probs[:, 2]  # P(Normal) + P(Viral)

            # Prediccion: positive si P(COVID) > 0.5
            preds = (prob_positive > 0.5).astype(int)

            all_probs.extend(prob_positive)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calcular metricas
    accuracy = accuracy_score(all_labels, all_preds)
    sensitivity = recall_score(all_labels, all_preds, pos_label=1)  # COVID recall
    specificity = recall_score(all_labels, all_preds, pos_label=0)  # Non-COVID recall
    precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    f1 = f1_score(all_labels, all_preds, pos_label=1)

    # AUC-ROC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': float(accuracy),
        'sensitivity': float(sensitivity),  # COVID recall
        'specificity': float(specificity),  # Non-COVID recall
        'precision': float(precision),
        'f1_score': float(f1),
        'auc_roc': float(auc),
        'confusion_matrix': cm.tolist(),
        'n_samples': len(all_labels),
        'n_positive': int(sum(all_labels)),
        'n_negative': int(len(all_labels) - sum(all_labels))
    }


def find_available_models(classifier_dir):
    """
    Encuentra modelos disponibles en el directorio de clasificadores.

    Returns:
        Lista de dicts con info de cada modelo
    """
    models = []
    classifier_path = Path(classifier_dir)

    if not classifier_path.exists():
        return models

    for model_dir in classifier_path.iterdir():
        if not model_dir.is_dir():
            continue

        checkpoint = model_dir / 'best_model.pt'
        results_file = model_dir / 'results.json'

        if not checkpoint.exists():
            continue

        # Extraer nombre de arquitectura y tipo de dataset
        dir_name = model_dir.name
        parts = dir_name.rsplit('_', 1)

        if len(parts) == 2:
            arch_name = parts[0]
            dataset_type = parts[1]  # 'original' o 'warped'
        else:
            arch_name = dir_name
            dataset_type = 'unknown'

        # Leer resultados del entrenamiento si existen
        train_metrics = {}
        if results_file.exists():
            with open(results_file) as f:
                train_results = json.load(f)
                train_metrics = train_results.get('test_metrics', {})

        models.append({
            'name': dir_name,
            'architecture': arch_name,
            'dataset_type': dataset_type,
            'checkpoint': str(checkpoint),
            'train_accuracy': train_metrics.get('accuracy', 0),
            'train_f1': train_metrics.get('f1_macro', 0)
        })

    return models


def plot_comparison(results, save_path, title='Comparacion de Modelos en Dataset Externo'):
    """Genera grafico comparativo de resultados."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14)

    # Datos para graficos
    model_names = [r['model_name'] for r in results]
    dataset_types = [r['dataset_type'] for r in results]

    # Colores por tipo de dataset
    colors = ['#2ecc71' if t == 'original' else '#e74c3c' for t in dataset_types]

    # 1. Accuracy
    ax1 = axes[0]
    accuracies = [r['metrics']['accuracy'] * 100 for r in results]
    bars1 = ax1.barh(model_names, accuracies, color=colors)
    ax1.set_xlabel('Accuracy (%)')
    ax1.set_title('Accuracy en Dataset Externo')
    ax1.set_xlim(0, 100)
    for bar, acc in zip(bars1, accuracies):
        ax1.text(acc + 1, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}%', va='center', fontsize=9)

    # 2. Sensibilidad (COVID Recall)
    ax2 = axes[1]
    sensitivities = [r['metrics']['sensitivity'] * 100 for r in results]
    bars2 = ax2.barh(model_names, sensitivities, color=colors)
    ax2.set_xlabel('Sensibilidad (%)')
    ax2.set_title('Sensibilidad COVID')
    ax2.set_xlim(0, 100)
    for bar, sens in zip(bars2, sensitivities):
        ax2.text(sens + 1, bar.get_y() + bar.get_height()/2,
                f'{sens:.1f}%', va='center', fontsize=9)

    # 3. AUC-ROC
    ax3 = axes[2]
    aucs = [r['metrics']['auc_roc'] * 100 for r in results]
    bars3 = ax3.barh(model_names, aucs, color=colors)
    ax3.set_xlabel('AUC-ROC (%)')
    ax3.set_title('AUC-ROC')
    ax3.set_xlim(0, 100)
    for bar, auc in zip(bars3, aucs):
        ax3.text(auc + 1, bar.get_y() + bar.get_height()/2,
                f'{auc:.1f}%', va='center', fontsize=9)

    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Original'),
        Patch(facecolor='#e74c3c', label='Warped')
    ]
    fig.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grafico guardado: {save_path}")


def plot_gap_analysis(results, save_path):
    """Genera grafico de gap de generalizacion."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Datos
    model_names = []
    train_accs = []
    external_accs = []
    gaps = []
    colors = []

    for r in results:
        model_names.append(r['model_name'])
        train_acc = r.get('train_accuracy', 0) * 100
        ext_acc = r['metrics']['accuracy'] * 100
        train_accs.append(train_acc)
        external_accs.append(ext_acc)
        gaps.append(train_acc - ext_acc)
        colors.append('#2ecc71' if r['dataset_type'] == 'original' else '#e74c3c')

    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, train_accs, width, label='Acc. Train (propio)',
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, external_accs, width, label='Acc. Test (externo)',
                   color=colors, alpha=0.8)

    # Linea de gap
    for i, (train, ext, gap) in enumerate(zip(train_accs, external_accs, gaps)):
        ax.annotate('', xy=(i + width/2, ext), xytext=(i - width/2, train),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
        ax.text(i, max(train, ext) + 2, f'Gap: {gap:.1f}%',
               ha='center', fontsize=8, color='gray')

    ax.set_xlabel('Modelo')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Gap de Generalizacion: Entrenamiento vs Dataset Externo')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grafico de gap guardado: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluar modelos de clasificacion en Dataset3 externo'
    )
    parser.add_argument(
        '--external-data-dir',
        type=str,
        default='outputs/external_validation/dataset3',
        help='Directorio con Dataset3 procesado'
    )
    parser.add_argument(
        '--classifier-dir',
        type=str,
        default='outputs/classifier_comparison',
        help='Directorio con modelos entrenados'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/external_validation',
        help='Directorio para guardar resultados'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help='Modelos especificos a evaluar (e.g., resnet18_original densenet121_warped)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Tamano del batch'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Dispositivo (cuda, cpu, o auto)'
    )

    args = parser.parse_args()

    # Configurar dispositivo
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("EVALUACION EN DATASET EXTERNO (Dataset3 - FedCOVIDx)")
    print("=" * 60)
    print(f"\nDispositivo: {device}")

    # Rutas
    external_data_dir = PROJECT_ROOT / args.external_data_dir
    classifier_dir = PROJECT_ROOT / args.classifier_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verificar que existe el dataset externo
    if not (external_data_dir / 'test').exists():
        print(f"\nError: No se encuentra el dataset procesado en {external_data_dir}")
        print("Ejecuta primero: python scripts/archive/classification/prepare_dataset3.py")
        sys.exit(1)

    # Cargar dataset externo
    print(f"\nCargando Dataset3 desde: {external_data_dir}")
    transform = get_transforms(img_size=224)
    external_dataset = ExternalDataset(
        data_dir=external_data_dir,
        split='test',
        transform=transform
    )

    external_loader = DataLoader(
        external_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    print(f"Total imagenes test: {len(external_dataset)}")
    print(f"  Positive (COVID): {sum(external_dataset.targets)}")
    print(f"  Negative: {len(external_dataset) - sum(external_dataset.targets)}")

    # Encontrar modelos disponibles
    print(f"\nBuscando modelos en: {classifier_dir}")
    available_models = find_available_models(classifier_dir)

    if not available_models:
        print("No se encontraron modelos para evaluar")
        sys.exit(1)

    # Filtrar modelos si se especifica
    if args.models:
        available_models = [m for m in available_models if m['name'] in args.models]

    print(f"Modelos a evaluar: {len(available_models)}")
    for m in available_models:
        print(f"  - {m['name']} (train acc: {m['train_accuracy']*100:.1f}%)")

    # Evaluar cada modelo
    results = []

    for model_info in available_models:
        print(f"\n{'='*60}")
        print(f"Evaluando: {model_info['name']}")
        print(f"{'='*60}")

        try:
            # Crear y cargar modelo
            model = create_model(
                model_name=model_info['architecture'],
                num_classes=3,
                checkpoint_path=model_info['checkpoint']
            )
            model = model.to(device)

            # Evaluar
            metrics = evaluate_binary(model, external_loader, device)

            print(f"\nResultados en Dataset3:")
            print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
            print(f"  Sensibilidad (COVID): {metrics['sensitivity']*100:.2f}%")
            print(f"  Especificidad: {metrics['specificity']*100:.2f}%")
            print(f"  AUC-ROC: {metrics['auc_roc']*100:.2f}%")
            print(f"  F1-Score: {metrics['f1_score']*100:.2f}%")

            # Calcular gap
            train_acc = model_info['train_accuracy']
            ext_acc = metrics['accuracy']
            gap = train_acc - ext_acc

            print(f"\n  Gap de generalizacion:")
            print(f"    Train acc (propio): {train_acc*100:.2f}%")
            print(f"    Test acc (externo): {ext_acc*100:.2f}%")
            print(f"    Gap: {gap*100:.2f}%")

            results.append({
                'model_name': model_info['name'],
                'architecture': model_info['architecture'],
                'dataset_type': model_info['dataset_type'],
                'train_accuracy': train_acc,
                'train_f1': model_info['train_f1'],
                'metrics': metrics,
                'gap': gap
            })

        except Exception as e:
            print(f"Error evaluando {model_info['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Ordenar resultados por accuracy externa
    results.sort(key=lambda x: x['metrics']['accuracy'], reverse=True)

    # Guardar resultados
    results_path = output_dir / 'baseline_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'external_dataset': 'Dataset3_FedCOVIDx',
            'n_samples': len(external_dataset),
            'results': results
        }, f, indent=2)
    print(f"\nResultados guardados en: {results_path}")

    # Generar graficos
    if results:
        comparison_path = output_dir / 'baseline_comparison.png'
        plot_comparison(results, comparison_path)

        gap_path = output_dir / 'gap_analysis.png'
        plot_gap_analysis(results, gap_path)

    # Tabla resumen
    print("\n" + "=" * 80)
    print("RESUMEN DE RESULTADOS")
    print("=" * 80)
    print(f"\n{'Modelo':<25} {'Tipo':<10} {'Train%':<8} {'Ext%':<8} {'Gap%':<8} {'Sens%':<8} {'AUC%':<8}")
    print("-" * 80)

    for r in results:
        print(f"{r['model_name']:<25} "
              f"{r['dataset_type']:<10} "
              f"{r['train_accuracy']*100:>6.1f}  "
              f"{r['metrics']['accuracy']*100:>6.1f}  "
              f"{r['gap']*100:>6.1f}  "
              f"{r['metrics']['sensitivity']*100:>6.1f}  "
              f"{r['metrics']['auc_roc']*100:>6.1f}")

    # Analisis comparativo Original vs Warped
    print("\n" + "=" * 60)
    print("ANALISIS COMPARATIVO: ORIGINAL vs WARPED")
    print("=" * 60)

    original_results = [r for r in results if r['dataset_type'] == 'original']
    warped_results = [r for r in results if r['dataset_type'] == 'warped']

    if original_results and warped_results:
        avg_gap_original = np.mean([r['gap'] for r in original_results])
        avg_gap_warped = np.mean([r['gap'] for r in warped_results])
        avg_acc_original = np.mean([r['metrics']['accuracy'] for r in original_results])
        avg_acc_warped = np.mean([r['metrics']['accuracy'] for r in warped_results])

        print(f"\nPromedio modelos ORIGINAL:")
        print(f"  Accuracy externa: {avg_acc_original*100:.2f}%")
        print(f"  Gap promedio: {avg_gap_original*100:.2f}%")

        print(f"\nPromedio modelos WARPED:")
        print(f"  Accuracy externa: {avg_acc_warped*100:.2f}%")
        print(f"  Gap promedio: {avg_gap_warped*100:.2f}%")

        if avg_gap_warped < avg_gap_original:
            print("\n-> Los modelos WARPED tienen MENOR gap de generalizacion")
        elif avg_gap_warped > avg_gap_original:
            print("\n-> Los modelos ORIGINAL tienen MENOR gap de generalizacion")
        else:
            print("\n-> No hay diferencia significativa en el gap")

    print(f"\n{'='*60}")
    print("Evaluacion completada")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
