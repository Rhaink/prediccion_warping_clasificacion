#!/usr/bin/env python3
"""
Verificar que no hay data leakage entre splits.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

from src_v2.data.dataset import create_dataloaders, get_dataframe_splits
from src_v2.data.utils import load_coordinates_csv
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
    print("VERIFICACION DE DATA LEAKAGE")
    print("="*60)

    # 1. Verificar que los splits son correctos
    print("\n--- 1. Verificando splits de datos ---")
    csv_path = str(PROJECT_ROOT / 'data/coordenadas/coordenadas_maestro.csv')
    train_df, val_df, test_df = get_dataframe_splits(csv_path, random_state=42)

    print(f"Train: {len(train_df)} samples")
    print(f"Val:   {len(val_df)} samples")
    print(f"Test:  {len(test_df)} samples")

    # Verificar no hay overlap
    train_images = set(train_df['image_name'])
    val_images = set(val_df['image_name'])
    test_images = set(test_df['image_name'])

    overlap_train_val = train_images & val_images
    overlap_train_test = train_images & test_images
    overlap_val_test = val_images & test_images

    print(f"\nOverlap train-val: {len(overlap_train_val)} images")
    print(f"Overlap train-test: {len(overlap_train_test)} images")
    print(f"Overlap val-test: {len(overlap_val_test)} images")

    if overlap_train_val or overlap_train_test or overlap_val_test:
        print("\n[ERROR] DATA LEAKAGE DETECTED!")
        return

    print("\n[OK] No data leakage detected")

    # 2. Evaluar modelo en todos los splits
    print("\n--- 2. Evaluando modelo seed=42 en cada split ---")

    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path=csv_path,
        data_root=str(PROJECT_ROOT / 'data/'),
        batch_size=16, num_workers=4, random_state=42,
        use_clahe=True, clahe_clip_limit=2.0, clahe_tile_size=4,
    )

    model = load_model(PROJECT_ROOT / 'checkpoints/session10/exp4_epochs100/final_model.pt', device)

    print("Evaluating on each split (without TTA)...")
    train_err = evaluate_on_loader(model, train_loader, device)
    val_err = evaluate_on_loader(model, val_loader, device)
    test_err = evaluate_on_loader(model, test_loader, device)

    print(f"\nTrain error: {train_err:.2f} px")
    print(f"Val error:   {val_err:.2f} px")
    print(f"Test error:  {test_err:.2f} px")

    # 3. Analisis por categoria en cada split
    print("\n--- 3. Distribucion por categoria ---")
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n{name}:")
        for cat in ['COVID', 'Normal', 'Viral_Pneumonia']:
            count = len(df[df['category'] == cat])
            pct = count / len(df) * 100
            print(f"  {cat}: {count} ({pct:.1f}%)")

    # 4. Estadisticas de landmarks por split
    print("\n--- 4. Estadisticas de landmarks (promedio de posiciones) ---")
    landmark_cols = [f'L{i+1}_x' for i in range(15)] + [f'L{i+1}_y' for i in range(15)]

    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        means = df[landmark_cols].mean()
        stds = df[landmark_cols].std()
        print(f"\n{name} - Mean X: {means[[f'L{i+1}_x' for i in range(15)]].mean():.1f}, "
              f"Mean Y: {means[[f'L{i+1}_y' for i in range(15)]].mean():.1f}")

    print("\n" + "="*60)
    print("CONCLUSION: Los splits son correctos y no hay data leakage.")
    print("La diferencia train/val/test puede deberse a variabilidad estadistica.")
    print("="*60)


if __name__ == '__main__':
    main()
