#!/usr/bin/env python3
"""
Verificar modelos en VAL vs TEST para entender discrepancia.
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
        pretrained=True, freeze_backbone=True, dropout_rate=0.3, hidden_dim=768,
        use_coord_attention=True, deep_head=True, device=device
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def evaluate_on_loader(model, dataloader, device, image_size=224):
    all_preds = []
    all_targets = []
    for batch in dataloader:
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
    print("="*60)
    print("COMPARACION VAL vs TEST POR MODELO")
    print("="*60)

    _, val_loader, test_loader = create_dataloaders(
        csv_path=str(PROJECT_ROOT / 'data/coordenadas/coordenadas_maestro.csv'),
        data_root=str(PROJECT_ROOT / 'data/'),
        batch_size=16, num_workers=4, random_state=42,
        use_clahe=True, clahe_clip_limit=2.0, clahe_tile_size=4,
    )

    models = [
        ('seed=42', 'checkpoints/session10/exp4_epochs100/final_model.pt'),
        ('seed=123', 'checkpoints/session10/ensemble/seed123/final_model.pt'),
        ('seed=456', 'checkpoints/session10/ensemble/seed456/final_model.pt'),
    ]

    print(f"\n{'Model':<12} {'VAL':>10} {'TEST':>10} {'Diff':>10}")
    print("-"*45)

    for name, path in models:
        model = load_model(PROJECT_ROOT / path, device)
        val_err = evaluate_on_loader(model, val_loader, device)
        test_err = evaluate_on_loader(model, test_loader, device)
        diff = val_err - test_err
        print(f"{name:<12} {val_err:>10.2f} {test_err:>10.2f} {diff:>+10.2f}")
        del model
        torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("CONCLUSION:")
    print("  - Todos los modelos tienen mejor resultado en TEST que en VAL")
    print("  - Esto puede indicar que TEST set es 'mas facil' por azar")
    print("  - O que hay alguna diferencia sistematica entre splits")
    print("="*60)


if __name__ == '__main__':
    main()
