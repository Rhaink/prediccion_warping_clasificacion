#!/usr/bin/env python3
"""
Script para verificar modelos individuales del ensemble.
Sesion 11: Verificacion de resultados.

NOTA: Los valores de referencia validados están en GROUND_TRUTH.json
Los valores esperados en este script son de Sesión 11 (históricos).
El ensemble óptimo de 4 modelos logra 3.71 px (validado Sesión 43).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from tqdm import tqdm

from src_v2.data.dataset import create_dataloaders
from src_v2.models.resnet_landmark import create_model


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


def evaluate_single_model(model, dataloader, device, use_tta=True, image_size=224):
    """Evaluar un solo modelo."""
    all_preds = []
    all_targets = []
    all_categories = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        images = batch[0].to(device)
        targets = batch[1].to(device)

        if len(batch) > 2:
            metadata = batch[2]
            if isinstance(metadata, list) and len(metadata) > 0 and isinstance(metadata[0], dict):
                categories = [m['category'] for m in metadata]
            else:
                categories = ['Unknown'] * len(images)
        else:
            categories = ['Unknown'] * len(images)

        with torch.no_grad():
            if use_tta:
                pred = predict_with_tta(model, images, device)
            else:
                pred = model(images)

        all_preds.append(pred.cpu())
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

    return {
        'mean': float(errors_px.mean()),
        'std': float(errors_px.std()),
        'median': float(errors_px.median()),
        'per_category': {
            cat: float(errors_px.mean(dim=1)[[c == cat for c in all_categories]].mean())
            for cat in set(all_categories)
        }
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("="*60)
    print("VERIFICACION DE MODELOS INDIVIDUALES")
    print("="*60)

    # Cargar datos (mismo split que entrenamiento)
    print("\nLoading test data...")
    _, _, test_loader = create_dataloaders(
        csv_path=str(PROJECT_ROOT / 'data/coordenadas/coordenadas_maestro.csv'),
        data_root=str(PROJECT_ROOT / 'data/'),
        batch_size=16,
        num_workers=4,
        random_state=42,  # MISMO seed que entrenamiento
        use_clahe=True,
        clahe_clip_limit=2.0,
        clahe_tile_size=4,
    )
    print(f"Test samples: {len(test_loader.dataset)}")

    # Modelos a verificar
    # Valores actualizados según GROUND_TRUTH.json (con TTA)
    models_to_verify = [
        ('seed=42 (exp4_epochs100)', 'checkpoints/session10/exp4_epochs100/final_model.pt', 4.10),
        ('seed=123 (ensemble)', 'checkpoints/session10/ensemble/seed123/final_model.pt', 4.05),
        ('seed=456 (ensemble)', 'checkpoints/session10/ensemble/seed456/final_model.pt', 4.04),
    ]

    print("\n" + "-"*60)
    print("Evaluando cada modelo con TTA...")
    print("-"*60)

    results = []
    for name, path, expected in models_to_verify:
        print(f"\n[{name}]")
        print(f"  Path: {path}")
        print(f"  Expected: ~{expected:.2f} px")

        model = load_model(PROJECT_ROOT / path, device)
        metrics = evaluate_single_model(model, test_loader, device, use_tta=True)

        actual = metrics['mean']
        diff = actual - expected
        status = "OK" if abs(diff) < 0.5 else "WARNING"

        print(f"  Actual:   {actual:.2f} px")
        print(f"  Diff:     {diff:+.2f} px [{status}]")
        print(f"  Std:      {metrics['std']:.2f} px")
        print(f"  Median:   {metrics['median']:.2f} px")
        print(f"  Per category:")
        for cat, err in sorted(metrics['per_category'].items()):
            print(f"    {cat}: {err:.2f} px")

        results.append({
            'name': name,
            'expected': expected,
            'actual': actual,
            'diff': diff,
            'status': status
        })

        del model
        torch.cuda.empty_cache()

    # Resumen
    print("\n" + "="*60)
    print("RESUMEN DE VERIFICACION")
    print("="*60)
    print(f"\n{'Model':<30} {'Expected':>10} {'Actual':>10} {'Diff':>10} {'Status':>10}")
    print("-"*70)
    for r in results:
        print(f"{r['name']:<30} {r['expected']:>10.2f} {r['actual']:>10.2f} {r['diff']:>+10.2f} {r['status']:>10}")

    # Verificar si los resultados son consistentes
    all_ok = all(r['status'] == 'OK' for r in results)
    print("\n" + "-"*60)
    if all_ok:
        print("VERIFICACION EXITOSA: Todos los modelos dentro del margen esperado")
    else:
        print("ADVERTENCIA: Algunos modelos fuera del margen esperado (>0.5 px)")
    print("-"*60)


if __name__ == '__main__':
    main()
