#!/usr/bin/env python3
"""
Script para evaluar ensemble de modelos.
Sesion 12: Soporta diferentes combinaciones y pesos.

NOTA: Los valores de referencia validados están en GROUND_TRUTH.json
El ensemble óptimo de 4 modelos logra 3.71 px (validado Sesión 43).
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from tqdm import tqdm

from src_v2.data.dataset import create_dataloaders
from src_v2.models.resnet_landmark import create_model
from src_v2.evaluation.metrics import evaluate_model, generate_evaluation_report


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate ensemble of models')
    parser.add_argument('--exclude-42', action='store_true',
                        help='Exclude seed=42 model from ensemble')
    parser.add_argument('--weighted', action='store_true',
                        help='Use weighted ensemble (inverse of validation error)')
    parser.add_argument('--custom-weights', type=str, default=None,
                        help='Custom weights as comma-separated values (e.g. "0.2,0.4,0.4")')
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Cargar modelo desde checkpoint."""
    model = create_model(
        pretrained=True,
        freeze_backbone=True,
        dropout_rate=0.3,
        hidden_dim=768,
        use_coord_attention=True,
        deep_head=True,
        device=device
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict_with_tta(model, image, device):
    """Prediccion con Test-Time Augmentation."""
    model.eval()
    with torch.no_grad():
        # Original
        pred1 = model(image)

        # Flip horizontal
        image_flip = torch.flip(image, dims=[3])
        pred2 = model(image_flip)

        # Invertir flip en predicciones
        pred2 = pred2.view(-1, 15, 2)
        pred2[:, :, 0] = 1 - pred2[:, :, 0]  # Invertir X

        # Intercambiar pares simetricos
        SYMMETRIC_PAIRS = [(2, 3), (4, 5), (6, 7), (11, 12), (13, 14)]
        for left, right in SYMMETRIC_PAIRS:
            pred2[:, [left, right]] = pred2[:, [right, left]]

        pred2 = pred2.view(-1, 30)

        # Promediar
        return (pred1 + pred2) / 2


def evaluate_ensemble(models, dataloader, device, image_size=224, weights=None):
    """Evaluar ensemble promediando predicciones.

    Args:
        models: Lista de modelos
        dataloader: DataLoader con datos de test
        device: CPU o GPU
        image_size: Tamaño de imagen para convertir a pixeles
        weights: Lista de pesos para cada modelo (None = pesos iguales)
    """
    all_preds = []
    all_targets = []
    all_categories = []

    # Normalizar pesos
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    for batch in tqdm(dataloader, desc="Evaluating ensemble"):
        images = batch[0].to(device)
        targets = batch[1].to(device)

        if len(batch) > 2:
            metadata = batch[2]
            if isinstance(metadata, list) and len(metadata) > 0 and isinstance(metadata[0], dict):
                categories = [m['category'] for m in metadata]
            elif isinstance(metadata, dict):
                categories = metadata['category']
            else:
                categories = ['Unknown'] * len(images)
        else:
            categories = ['Unknown'] * len(images)

        # Predicciones de cada modelo con TTA
        preds = []
        for model in models:
            pred = predict_with_tta(model, images, device)
            preds.append(pred)

        # Promediar predicciones con pesos
        preds_stack = torch.stack(preds)  # (n_models, batch, 30)
        weights_tensor = torch.tensor(weights, device=device).view(-1, 1, 1)
        ensemble_pred = (preds_stack * weights_tensor).sum(dim=0)

        all_preds.append(ensemble_pred.cpu())
        all_targets.append(targets.cpu())
        all_categories.extend(categories)

    # Concatenar
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calcular metricas
    all_preds = all_preds.view(-1, 15, 2)
    all_targets = all_targets.view(-1, 15, 2)

    # Error por pixel
    errors_px = torch.norm((all_preds - all_targets) * image_size, dim=-1)

    # Metricas generales
    metrics = {
        'overall': {
            'mean': float(errors_px.mean()),
            'std': float(errors_px.std()),
            'median': float(errors_px.median()),
        },
        'percentiles': {
            'p50': float(torch.quantile(errors_px, 0.50)),
            'p75': float(torch.quantile(errors_px, 0.75)),
            'p90': float(torch.quantile(errors_px, 0.90)),
            'p95': float(torch.quantile(errors_px, 0.95)),
        },
        'per_landmark': {},
        'per_category': {}
    }

    # Error por landmark
    for i in range(15):
        landmark_name = f'L{i+1}'
        landmark_errors = errors_px[:, i]
        metrics['per_landmark'][landmark_name] = {
            'mean': float(landmark_errors.mean()),
            'std': float(landmark_errors.std()),
            'median': float(landmark_errors.median()),
            'max': float(landmark_errors.max()),
        }

    # Error por categoria
    errors_flat = errors_px.mean(dim=1)
    for cat in set(all_categories):
        cat_mask = [c == cat for c in all_categories]
        cat_errors = errors_flat[cat_mask]
        metrics['per_category'][cat] = {
            'mean': float(cat_errors.mean()),
            'std': float(cat_errors.std()),
            'count': int(sum(cat_mask)),
        }

    return metrics


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Definir modelos disponibles con sus errores de validación
    available_models = {
        'seed42': {
            'path': 'checkpoints/session10/exp4_epochs100/final_model.pt',
            'val_error': 7.22,   # Error en validación sin TTA
            'test_error': 4.10,  # Error en test CON TTA (actualizado Sesion 46)
        },
        'seed123': {
            'path': 'checkpoints/session10/ensemble/seed123/final_model.pt',
            'val_error': 5.05,
            'test_error': 4.05,
        },
        'seed456': {
            'path': 'checkpoints/session10/ensemble/seed456/final_model.pt',
            'val_error': 5.21,
            'test_error': 4.04,
        },
    }

    # Seleccionar modelos a usar
    if args.exclude_42:
        model_keys = ['seed123', 'seed456']
        print("\n*** Excluding seed=42 from ensemble ***")
    else:
        model_keys = ['seed42', 'seed123', 'seed456']

    # Calcular pesos
    weights = None
    if args.custom_weights:
        weights = [float(w) for w in args.custom_weights.split(',')]
        if len(weights) != len(model_keys):
            raise ValueError(f"Custom weights length ({len(weights)}) != models ({len(model_keys)})")
        print(f"\n*** Using custom weights: {weights} ***")
    elif args.weighted:
        # Pesos inversamente proporcionales al error de validación
        val_errors = [available_models[k]['val_error'] for k in model_keys]
        weights = [1.0 / e for e in val_errors]
        total = sum(weights)
        weights = [w / total for w in weights]
        print(f"\n*** Using weighted ensemble (inverse val error): ***")
        for k, w, e in zip(model_keys, weights, val_errors):
            print(f"    {k}: weight={w:.3f} (val_error={e:.2f})")

    # Cargar modelos
    print("\nLoading models...")
    models = []
    for key in model_keys:
        path = available_models[key]['path']
        print(f"  Loading {key}: {path}")
        model = load_model(PROJECT_ROOT / path, device)
        models.append(model)

    # Cargar datos
    print("\nLoading data...")
    _, _, test_loader = create_dataloaders(
        csv_path=str(PROJECT_ROOT / 'data/coordenadas/coordenadas_maestro.csv'),
        data_root=str(PROJECT_ROOT / 'data/'),
        batch_size=16,
        num_workers=4,
        random_state=42,
        use_clahe=True,
        clahe_clip_limit=2.0,
        clahe_tile_size=4,
    )

    # Evaluar ensemble
    print("\nEvaluating ensemble...")
    metrics = evaluate_ensemble(models, test_loader, device, weights=weights)

    # Mostrar resultados
    print("\n" + "=" * 60)
    ensemble_name = f"ENSEMBLE ({len(models)} models)"
    if args.exclude_42:
        ensemble_name += " [excl. seed=42]"
    if args.weighted:
        ensemble_name += " [weighted]"
    print(f"{ensemble_name} EVALUATION REPORT")
    print("=" * 60)

    print(f"\n--- Overall Metrics ---")
    print(f"Mean Error:   {metrics['overall']['mean']:.2f} px")
    print(f"Std Error:    {metrics['overall']['std']:.2f} px")
    print(f"Median Error: {metrics['overall']['median']:.2f} px")

    print(f"\n--- Percentiles ---")
    for k, v in metrics['percentiles'].items():
        print(f"  {k}: {v:.2f} px")

    print(f"\n--- Error per Landmark ---")
    print(f"{'Landmark':<10} {'Mean':>8} {'Std':>8} {'Median':>8} {'Max':>8}")
    print("-" * 44)

    # Ordenar por error medio
    sorted_landmarks = sorted(
        metrics['per_landmark'].items(),
        key=lambda x: x[1]['mean']
    )
    for landmark, data in sorted_landmarks:
        print(f"{landmark:<10} {data['mean']:>8.2f} {data['std']:>8.2f} {data['median']:>8.2f} {data['max']:>8.2f}")

    print(f"\nBest:  {sorted_landmarks[0][0]} ({sorted_landmarks[0][1]['mean']:.2f} px)")
    print(f"Worst: {sorted_landmarks[-1][0]} ({sorted_landmarks[-1][1]['mean']:.2f} px)")

    print(f"\n--- Error per Category ---")
    for cat, data in sorted(metrics['per_category'].items(), key=lambda x: x[1]['mean']):
        print(f"{cat}: {data['mean']:.2f} +/- {data['std']:.2f} px (n={data['count']})")

    print("\n" + "=" * 60)

    # Comparar con modelos individuales
    print("\n--- Comparison with Individual Models (TEST with TTA) ---")
    for k in ['seed42', 'seed123', 'seed456']:
        status = "✓ included" if k.replace('seed', 'seed') in model_keys else "✗ excluded"
        print(f"  {k}: {available_models[k]['test_error']:.2f} px ({status})")
    print(f"\n  ENSEMBLE: {metrics['overall']['mean']:.2f} px")

    # Mostrar mejora/empeoramiento vs baseline
    # NOTA: El ensemble óptimo de 4 modelos logra 3.71 px (GROUND_TRUTH.json)
    baseline_ensemble = 3.71  # Ensemble de 4 modelos + TTA (Sesión 43)
    diff = metrics['overall']['mean'] - baseline_ensemble
    if diff < 0:
        print(f"\n  *** MEJORA: {-diff:.2f} px respecto a ensemble óptimo (3.71 px) ***")
    elif diff > 0:
        print(f"\n  *** EMPEORA: +{diff:.2f} px respecto a ensemble óptimo (3.71 px) ***")
    else:
        print(f"\n  *** SIN CAMBIO respecto a ensemble óptimo (3.71 px) ***")


if __name__ == '__main__':
    main()
