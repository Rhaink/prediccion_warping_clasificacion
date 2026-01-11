#!/usr/bin/env python
"""
Script de entrenamiento para el modelo jerárquico.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_v2.data.dataset import create_dataloaders
from src_v2.models.hierarchical import HierarchicalLandmarkModel, AxisLoss
from src_v2.models.losses import WingLoss
from src_v2.evaluation.metrics import evaluate_model_with_tta


def train_epoch(model, loader, optimizer, criterion, axis_criterion, device, axis_weight=0.5):
    """Entrenar una epoca."""
    model.train()
    total_loss = 0
    total_error = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch[0].to(device)
        targets = batch[1].to(device)

        optimizer.zero_grad()

        outputs = model(images)

        # Loss principal
        main_loss = criterion(outputs, targets)

        # Loss adicional para el eje
        axis_loss = axis_criterion(outputs, targets)

        loss = main_loss + axis_weight * axis_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calcular error en pixeles
        with torch.no_grad():
            diff = (outputs - targets) * 224
            diff = diff.view(-1, 15, 2)
            errors = torch.sqrt((diff ** 2).sum(dim=2))
            total_error += errors.mean().item()

    return total_loss / len(loader), total_error / len(loader)


def validate(model, loader, criterion, device):
    """Validar modelo."""
    model.eval()
    total_loss = 0
    total_error = 0

    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            targets = batch[1].to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item()

            diff = (outputs - targets) * 224
            diff = diff.view(-1, 15, 2)
            errors = torch.sqrt((diff ** 2).sum(dim=2))
            total_error += errors.mean().item()

    return total_loss / len(loader), total_error / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Train Hierarchical Landmark Model")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file with default values")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--phase1-epochs", type=int, default=15)
    parser.add_argument("--phase2-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--axis-weight", type=float, default=0.5)
    parser.add_argument("--save-dir", type=str, default="checkpoints/hierarchical")
    parser.add_argument("--clahe", action="store_true", default=True)
    args, _ = parser.parse_known_args()
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_file():
            parser.error(f"Config file not found: {config_path}")
        with open(config_path, "r") as f:
            config_data = json.load(f)
        if not isinstance(config_data, dict):
            parser.error("Config file must be a JSON object with flat key/value pairs.")
        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(config_data) - valid_keys)
        if unknown_keys:
            parser.error(f"Unknown config keys: {', '.join(unknown_keys)}")
        parser.set_defaults(**config_data)

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "phase1"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "phase2"), exist_ok=True)

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path="data/coordenadas/coordenadas_maestro.csv",
        data_root="data",
        batch_size=16,
        use_clahe=args.clahe,
        clahe_tile_size=4,
        random_state=args.seed
    )
    print(f"Data loaded: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")

    # Model
    model = HierarchicalLandmarkModel(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss
    criterion = WingLoss(normalized=True)
    axis_criterion = AxisLoss(weight=1.0)

    # Phase 1: Backbone frozen
    print("\n" + "="*60)
    print("PHASE 1: Training with frozen backbone")
    print("="*60)

    model.freeze_backbone()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3
    )

    best_val_error = float("inf")
    patience_counter = 0

    for epoch in range(args.phase1_epochs):
        train_loss, train_error = train_epoch(
            model, train_loader, optimizer, criterion, axis_criterion, device, args.axis_weight
        )
        val_loss, val_error = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{args.phase1_epochs} - "
              f"Train: {train_error:.2f}px, Val: {val_error:.2f}px")

        if val_error < best_val_error:
            best_val_error = val_error
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_error": val_error
            }, os.path.join(args.save_dir, "phase1", "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Phase 1 best: {best_val_error:.2f}px")

    # Phase 2: Fine-tuning
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning with unfrozen backbone")
    print("="*60)

    # Load best phase 1 model
    checkpoint = torch.load(os.path.join(args.save_dir, "phase1", "best_model.pt"))
    model.load_state_dict(checkpoint["model_state_dict"])

    model.unfreeze_backbone()
    optimizer = torch.optim.Adam(model.get_trainable_params(backbone_lr=2e-5, head_lr=2e-4))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.phase2_epochs)

    best_val_error = float("inf")
    patience_counter = 0

    for epoch in range(args.phase2_epochs):
        train_loss, train_error = train_epoch(
            model, train_loader, optimizer, criterion, axis_criterion, device, args.axis_weight
        )
        val_loss, val_error = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.phase2_epochs} - "
              f"Train: {train_error:.2f}px, Val: {val_error:.2f}px")

        if val_error < best_val_error:
            best_val_error = val_error
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_error": val_error
            }, os.path.join(args.save_dir, "phase2", "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Phase 2 best: {best_val_error:.2f}px")

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    # Load best model
    checkpoint = torch.load(os.path.join(args.save_dir, "phase2", "best_model.pt"))
    model.load_state_dict(checkpoint["model_state_dict"])

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": vars(args)
    }, os.path.join(args.save_dir, "final_model.pt"))

    # Evaluate
    results = evaluate_model_with_tta(model, test_loader, device)

    print(f"\nTest Results (with TTA):")
    print(f"  Mean Error: {results['overall']['mean']:.2f} px")
    print(f"  Std Error: {results['overall']['std']:.2f} px")
    print(f"\nPer Category:")
    for cat, data in results["per_category"].items():
        print(f"  {cat}: {data['mean']:.2f} px (n={data['count']})")

    # Save results
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump({
            "config": vars(args),
            "results": {
                "mean": results["overall"]["mean"],
                "std": results["overall"]["std"],
                "per_category": {k: v["mean"] for k, v in results["per_category"].items()}
            }
        }, f, indent=2)

    print(f"\nResults saved to {args.save_dir}")


if __name__ == "__main__":
    main()
