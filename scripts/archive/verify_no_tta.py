#!/usr/bin/env python3
"""
Verificar modelos SIN TTA para entender discrepancia.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm import tqdm

from src_v2.data.dataset import create_dataloaders
from src_v2.models.resnet_landmark import create_model


def load_model(checkpoint_path, device):
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


def evaluate_no_tta(model, dataloader, device, image_size=224):
    """Evaluar SIN TTA."""
    all_preds = []
    all_targets = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        images = batch[0].to(device)
        targets = batch[1].to(device)

        with torch.no_grad():
            pred = model(images)

        all_preds.append(pred.cpu())
        all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0).view(-1, 15, 2)
    all_targets = torch.cat(all_targets, dim=0).view(-1, 15, 2)
    errors_px = torch.norm((all_preds - all_targets) * image_size, dim=-1)

    return float(errors_px.mean())


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("="*60)
    print("COMPARACION: CON TTA vs SIN TTA")
    print("="*60)

    # Cargar datos
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

    models = [
        ('seed=42', 'checkpoints/session10/exp4_epochs100/final_model.pt', 6.75),
        ('seed=123', 'checkpoints/session10/ensemble/seed123/final_model.pt', 7.16),
        ('seed=456', 'checkpoints/session10/ensemble/seed456/final_model.pt', 7.20),
    ]

    print(f"\n{'Model':<15} {'Reported':>10} {'No TTA':>10} {'Match?':>10}")
    print("-"*50)

    for name, path, expected in models:
        model = load_model(PROJECT_ROOT / path, device)
        no_tta = evaluate_no_tta(model, test_loader, device)
        diff = abs(no_tta - expected)
        match = "YES" if diff < 0.5 else "NO"
        print(f"{name:<15} {expected:>10.2f} {no_tta:>10.2f} {match:>10}")
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
