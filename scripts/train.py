#!/usr/bin/env python3
"""
Script principal de entrenamiento para landmark prediction.

Uso:
    python scripts/train.py                     # Entrenamiento completo
    python scripts/train.py --test              # Prueba corta (2 epocas)
    python scripts/train.py --phase1-only       # Solo phase 1
    python scripts/train.py --phase2-only       # Solo phase 2 (requiere checkpoint)
"""

import sys
from pathlib import Path

# Agregar src_v2 al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch
import json
from datetime import datetime

from src_v2.data.dataset import create_dataloaders
from src_v2.models.resnet_landmark import create_model, count_parameters
from src_v2.models.losses import (
    WingLoss, WeightedWingLoss, CombinedLandmarkLoss, get_landmark_weights
)
from src_v2.training.trainer import LandmarkTrainer
from src_v2.evaluation.metrics import (
    evaluate_model, evaluate_model_with_tta, generate_evaluation_report
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train landmark prediction model')

    # Paths
    parser.add_argument('--csv-path', type=str,
                       default='data/coordenadas/coordenadas_maestro.csv',
                       help='Path to coordinates CSV')
    parser.add_argument('--data-root', type=str, default='data/',
                       help='Root directory for data')
    parser.add_argument('--save-dir', type=str, default='checkpoints/',
                       help='Directory to save checkpoints')
    parser.add_argument('--output-dir', type=str, default='outputs/',
                       help='Directory for outputs and logs')

    # Training mode
    parser.add_argument('--test', action='store_true',
                       help='Quick test run (2 epochs each phase)')
    parser.add_argument('--phase1-only', action='store_true',
                       help='Only run phase 1')
    parser.add_argument('--phase2-only', action='store_true',
                       help='Only run phase 2 (requires --checkpoint)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')

    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--phase1-epochs', type=int, default=15,
                       help='Number of epochs for phase 1')
    parser.add_argument('--phase2-epochs', type=int, default=50,
                       help='Number of epochs for phase 2')
    parser.add_argument('--phase1-lr', type=float, default=1e-3,
                       help='Learning rate for phase 1')
    parser.add_argument('--phase2-backbone-lr', type=float, default=2e-5,
                       help='Backbone learning rate for phase 2')
    parser.add_argument('--phase2-head-lr', type=float, default=2e-4,
                       help='Head learning rate for phase 2')
    parser.add_argument('--phase1-patience', type=int, default=5,
                       help='Early stopping patience for phase 1')
    parser.add_argument('--phase2-patience', type=int, default=10,
                       help='Early stopping patience for phase 2')

    # Data augmentation
    parser.add_argument('--flip-prob', type=float, default=0.5,
                       help='Probability of horizontal flip')
    parser.add_argument('--rotation', type=float, default=10.0,
                       help='Max rotation in degrees')

    # CLAHE preprocessing (Session 7 - COVID improvement)
    parser.add_argument('--clahe', action='store_true',
                       help='Use CLAHE for contrast enhancement (helps with COVID)')
    parser.add_argument('--clahe-clip', type=float, default=2.0,
                       help='CLAHE clip limit (higher = more contrast)')
    parser.add_argument('--clahe-tile', type=int, default=8,
                       help='CLAHE tile size')

    # Category weighting (Session 7 - COVID oversampling)
    parser.add_argument('--category-weights', action='store_true',
                       help='Use weighted sampling to oversample COVID')
    parser.add_argument('--covid-weight', type=float, default=2.0,
                       help='Weight for COVID samples (default: 2.0)')
    parser.add_argument('--normal-weight', type=float, default=1.0,
                       help='Weight for Normal samples (default: 1.0)')
    parser.add_argument('--viral-weight', type=float, default=1.2,
                       help='Weight for Viral Pneumonia samples (default: 1.2)')

    # Loss configuration
    parser.add_argument('--loss', type=str, default='wing',
                       choices=['wing', 'weighted_wing', 'combined'],
                       help='Loss function to use')
    parser.add_argument('--weight-strategy', type=str, default='uniform',
                       choices=['uniform', 'inverse_variance', 'custom'],
                       help='Landmark weight strategy')
    parser.add_argument('--central-weight', type=float, default=1.0,
                       help='Weight for central alignment loss (combined only)')
    parser.add_argument('--symmetry-weight', type=float, default=0.5,
                       help='Weight for symmetry loss (combined only)')

    # Architecture options
    parser.add_argument('--coord-attention', action='store_true',
                       help='Use Coordinate Attention module')
    parser.add_argument('--deep-head', action='store_true',
                       help='Use deeper head with BatchNorm')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension in head')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')

    # Test-Time Augmentation
    parser.add_argument('--tta', action='store_true',
                       help='Use Test-Time Augmentation for evaluation')

    # Other
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()


def setup_device():
    """Setup and return the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def create_loss_function(args, device):
    """Create loss function based on args."""
    if args.loss == 'wing':
        # Wing Loss con parametros normalizados para coordenadas en [0,1]
        return WingLoss(omega=10.0, epsilon=2.0, normalized=True, image_size=224)

    elif args.loss == 'weighted_wing':
        weights = get_landmark_weights(args.weight_strategy)
        return WeightedWingLoss(
            omega=10.0, epsilon=2.0,
            weights=weights.to(device),
            normalized=True, image_size=224
        )

    elif args.loss == 'combined':
        weights = get_landmark_weights(args.weight_strategy)
        return CombinedLandmarkLoss(
            wing_omega=10.0,
            wing_epsilon=2.0,
            landmark_weights=weights.to(device),
            central_weight=args.central_weight,
            symmetry_weight=args.symmetry_weight,
            symmetry_margin=6.0,
            image_size=224
        )

    raise ValueError(f"Unknown loss: {args.loss}")


def save_training_config(args, output_dir):
    """Save training configuration to JSON."""
    config = vars(args).copy()
    config['timestamp'] = datetime.now().isoformat()

    config_path = output_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Setup directories
    save_dir = Path(args.save_dir)
    output_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Adjust epochs for test mode
    if args.test:
        print("\n" + "=" * 60)
        print("TEST MODE: Running with reduced epochs")
        print("=" * 60)
        args.phase1_epochs = 2
        args.phase2_epochs = 2
        args.phase1_patience = 2
        args.phase2_patience = 2

    # Setup device
    device = setup_device()

    # Create dataloaders
    print("\nLoading data...")
    if args.clahe:
        print(f"Using CLAHE: clip_limit={args.clahe_clip}, tile_size={args.clahe_tile}")

    # Pesos por categoria
    category_weights = None
    if args.category_weights:
        category_weights = {
            'COVID': args.covid_weight,
            'Normal': args.normal_weight,
            'Viral_Pneumonia': args.viral_weight,
        }

    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path=str(PROJECT_ROOT / args.csv_path),
        data_root=str(PROJECT_ROOT / args.data_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_state=args.seed,
        flip_prob=args.flip_prob,
        rotation_degrees=args.rotation,
        use_clahe=args.clahe,
        clahe_clip_limit=args.clahe_clip,
        clahe_tile_size=args.clahe_tile,
        use_category_weights=args.category_weights,
        category_weights=category_weights,
    )

    # Create model
    print("\nCreating model...")
    arch_info = []
    if args.coord_attention:
        arch_info.append("CoordAttention")
    if args.deep_head:
        arch_info.append("DeepHead")
    if arch_info:
        print(f"Architecture: {' + '.join(arch_info)}")

    model = create_model(
        pretrained=True,
        freeze_backbone=True,
        dropout_rate=args.dropout,
        hidden_dim=args.hidden_dim,
        use_coord_attention=args.coord_attention,
        deep_head=args.deep_head,
        device=device
    )

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Load checkpoint if specified
    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Create loss function
    print(f"\nUsing loss: {args.loss}")
    criterion = create_loss_function(args, device)

    # Create trainer
    trainer = LandmarkTrainer(
        model=model,
        device=device,
        save_dir=str(save_dir),
        image_size=224
    )

    # Save config
    save_training_config(args, output_dir)

    # Training
    history = {}

    if args.phase2_only:
        if not args.checkpoint:
            print("ERROR: --phase2-only requires --checkpoint")
            return

        history['phase2'] = trainer.train_phase2(
            train_loader, val_loader, criterion,
            epochs=args.phase2_epochs,
            backbone_lr=args.phase2_backbone_lr,
            head_lr=args.phase2_head_lr,
            patience=args.phase2_patience
        )

    elif args.phase1_only:
        history['phase1'] = trainer.train_phase1(
            train_loader, val_loader, criterion,
            epochs=args.phase1_epochs,
            lr=args.phase1_lr,
            patience=args.phase1_patience
        )

    else:
        # Full training
        history = trainer.train_full(
            train_loader, val_loader, criterion,
            phase1_epochs=args.phase1_epochs,
            phase2_epochs=args.phase2_epochs,
            phase1_lr=args.phase1_lr,
            phase2_backbone_lr=args.phase2_backbone_lr,
            phase2_head_lr=args.phase2_head_lr,
            phase1_patience=args.phase1_patience,
            phase2_patience=args.phase2_patience
        )

    # Final evaluation on test set
    print("\n" + "=" * 60)
    if args.tta:
        print("FINAL EVALUATION ON TEST SET (with TTA)")
    else:
        print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)

    if args.tta:
        metrics = evaluate_model_with_tta(model, test_loader, device, image_size=224)
    else:
        metrics = evaluate_model(model, test_loader, device, image_size=224)
    report = generate_evaluation_report(metrics)
    print(report)

    # Save report
    report_path = output_dir / f'evaluation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # Save final model
    final_model_path = save_dir / 'final_model.pt'
    trainer.save_model(str(final_model_path))

    # Save history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        # Convert to serializable format
        history_serializable = {}
        for phase, phase_history in history.items():
            history_serializable[phase] = {
                k: [float(v) for v in vals] if isinstance(vals, list) else vals
                for k, vals in phase_history.items()
            }
        json.dump(history_serializable, f, indent=2)
    print(f"History saved to {history_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final test error: {metrics['overall']['mean']:.2f} +/- {metrics['overall']['std']:.2f} px")


if __name__ == '__main__':
    main()
