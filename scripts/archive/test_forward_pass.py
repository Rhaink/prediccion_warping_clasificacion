#!/usr/bin/env python3
"""
Script para verificar forward pass del modelo con datos reales.
Sesion 2: Verificacion de modelo y losses.
"""

import sys
from pathlib import Path

# Agregar paths
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src_v2.data.dataset import LandmarkDataset, create_dataloaders
from src_v2.data.transforms import get_train_transforms, get_val_transforms
from src_v2.models.resnet_landmark import (
    ResNet18Landmarks,
    create_model,
    count_parameters
)
from src_v2.models.losses import (
    WingLoss,
    WeightedWingLoss,
    CentralAlignmentLoss,
    SoftSymmetryLoss,
    CombinedLandmarkLoss,
    get_landmark_weights
)

# Dispositivo global
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_model_creation():
    """Prueba creación del modelo."""
    print("=" * 60)
    print("TEST 1: Creación del modelo")
    print("=" * 60)

    print(f"  Dispositivo: {DEVICE}")

    # Modelo con backbone congelado (Phase 1)
    model = create_model(
        num_landmarks=15,
        pretrained=True,
        freeze_backbone=True,
        dropout_rate=0.5,
        hidden_dim=256,
        device=DEVICE
    )

    total, trainable = count_parameters(model)
    print(f"Modelo creado exitosamente")
    print(f"  Parámetros totales: {total:,}")
    print(f"  Parámetros entrenables: {trainable:,}")
    print(f"  Backbone congelado: {total - trainable:,} params")

    # Verificar output shape
    dummy_input = torch.rand(4, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

    assert output.shape == (4, 30), f"Expected (4, 30), got {output.shape}"
    assert output.min() >= 0 and output.max() <= 1, "Output should be in [0, 1]"

    print("✓ Test 1 PASADO\n")
    return model


def test_unfreeze_backbone():
    """Prueba descongelar backbone (Phase 2)."""
    print("=" * 60)
    print("TEST 2: Descongelar backbone")
    print("=" * 60)

    model = create_model(freeze_backbone=True, device=DEVICE)
    _, trainable_frozen = count_parameters(model)

    model.unfreeze_backbone()
    _, trainable_unfrozen = count_parameters(model)

    print(f"  Params entrenables (congelado): {trainable_frozen:,}")
    print(f"  Params entrenables (descongelado): {trainable_unfrozen:,}")
    print(f"  Backbone activado: {trainable_unfrozen - trainable_frozen:,} params")

    assert trainable_unfrozen > trainable_frozen
    print("✓ Test 2 PASADO\n")


def test_differentiated_lr_groups():
    """Prueba grupos de parámetros para LR diferenciado."""
    print("=" * 60)
    print("TEST 3: Grupos de LR diferenciado")
    print("=" * 60)

    model = create_model(freeze_backbone=False, device=DEVICE)
    param_groups = model.get_trainable_params()

    print(f"  Número de grupos: {len(param_groups)}")

    for group in param_groups:
        num_params = sum(p.numel() for p in group['params'])
        print(f"    - {group['name']}: {num_params:,} params")

    # Crear optimizador con LR diferenciado
    optimizer = torch.optim.Adam([
        {'params': param_groups[0]['params'], 'lr': 2e-5},   # backbone
        {'params': param_groups[1]['params'], 'lr': 2e-4},   # head
    ])

    print(f"  Optimizador creado con LRs: backbone=2e-5, head=2e-4")
    print("✓ Test 3 PASADO\n")


def test_dataset_loading():
    """Prueba carga del dataset real."""
    print("=" * 60)
    print("TEST 4: Carga del dataset")
    print("=" * 60)

    # Usar create_dataloaders para crear los loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path='data/coordenadas/coordenadas_maestro.csv',
        data_root='data',
        batch_size=8,
        val_split=0.15,
        test_split=0.10,
        num_workers=0  # 0 para evitar problemas de multiprocessing
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Obtener una muestra del val_loader (retorna imagen, landmarks, metadata)
    batch_images, batch_landmarks, batch_meta = next(iter(val_loader))

    print(f"  Batch image shape: {batch_images.shape}")
    print(f"  Batch image dtype: {batch_images.dtype}")
    print(f"  Batch landmarks shape: {batch_landmarks.shape}")
    print(f"  Landmarks range: [{batch_landmarks.min():.3f}, {batch_landmarks.max():.3f}]")

    # Verificar normalización ImageNet
    print(f"  Image mean (aprox): {batch_images.mean():.3f}")
    print(f"  Image std (aprox): {batch_images.std():.3f}")

    assert batch_images.shape[1:] == (3, 224, 224)
    assert batch_landmarks.shape[1] == 30

    print("✓ Test 4 PASADO\n")
    return val_loader


def test_forward_pass_real_data(model, val_loader):
    """Forward pass con datos reales en GPU."""
    print("=" * 60)
    print("TEST 5: Forward pass con datos reales (GPU)")
    print("=" * 60)

    batch_images, batch_landmarks, batch_meta = next(iter(val_loader))

    # Mover a GPU
    batch_images = batch_images.to(DEVICE)
    batch_landmarks = batch_landmarks.to(DEVICE)

    print(f"  Batch images shape: {batch_images.shape}")
    print(f"  Batch images device: {batch_images.device}")
    print(f"  Batch landmarks shape: {batch_landmarks.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(batch_images)

    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Predictions device: {predictions.device}")
    print(f"  Predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")

    # Calcular error inicial (sin entrenar)
    pred_px = predictions * 224  # Desnormalizar
    target_px = batch_landmarks * 224

    errors = (pred_px - target_px).view(-1, 15, 2)
    errors_per_landmark = torch.sqrt((errors ** 2).sum(dim=-1))  # Euclidean
    mean_error = errors_per_landmark.mean().item()

    print(f"  Error promedio (sin entrenar): {mean_error:.1f} px")
    print(f"  (Esperado ~50-100 px para modelo no entrenado)")

    print("✓ Test 5 PASADO\n")
    return batch_images, batch_landmarks, predictions


def test_losses_real_data(predictions, targets):
    """Prueba losses con datos reales en GPU."""
    print("=" * 60)
    print("TEST 6: Losses con datos reales (GPU)")
    print("=" * 60)

    # WingLoss básico
    wing = WingLoss()
    wing_loss = wing(predictions, targets)
    print(f"  WingLoss: {wing_loss.item():.4f}")

    # WeightedWingLoss
    weights = get_landmark_weights('inverse_variance').to(DEVICE)
    weighted_wing = WeightedWingLoss(weights=weights).to(DEVICE)
    weighted_loss = weighted_wing(predictions, targets)
    print(f"  WeightedWingLoss: {weighted_loss.item():.4f}")

    # CentralAlignmentLoss
    central = CentralAlignmentLoss(image_size=224)
    central_loss = central(predictions)
    print(f"  CentralAlignmentLoss: {central_loss.item():.2f} px")

    # SoftSymmetryLoss
    symmetry = SoftSymmetryLoss(margin=6.0, image_size=224)
    symmetry_loss = symmetry(predictions)
    print(f"  SoftSymmetryLoss: {symmetry_loss.item():.2f} px²")

    # CombinedLandmarkLoss
    combined = CombinedLandmarkLoss(
        central_weight=0.3,
        symmetry_weight=0.1,
        symmetry_margin=6.0
    ).to(DEVICE)
    combined_result = combined(predictions, targets)
    print(f"  CombinedLoss:")
    print(f"    - Total: {combined_result['total'].item():.4f}")
    print(f"    - Wing: {combined_result['wing'].item():.4f}")
    print(f"    - Central: {combined_result['central'].item():.2f}")
    print(f"    - Symmetry: {combined_result['symmetry'].item():.2f}")

    print("✓ Test 6 PASADO\n")


def test_backward_pass():
    """Prueba backward pass completo en GPU."""
    print("=" * 60)
    print("TEST 7: Backward pass y gradientes (GPU)")
    print("=" * 60)

    model = create_model(freeze_backbone=True, device=DEVICE)
    model.train()

    # Crear batch en GPU
    images = torch.rand(4, 3, 224, 224).to(DEVICE)
    targets = torch.rand(4, 30).to(DEVICE)

    # Forward
    predictions = model(images)

    # Loss combinado
    loss_fn = CombinedLandmarkLoss().to(DEVICE)
    losses = loss_fn(predictions, targets)

    # Backward
    losses['total'].backward()

    # Verificar gradientes en head
    head_grads = []
    for name, param in model.head.named_parameters():
        if param.grad is not None:
            head_grads.append(param.grad.abs().mean().item())

    print(f"  Loss total: {losses['total'].item():.4f}")
    print(f"  Gradientes en head: {len(head_grads)} tensores")
    print(f"  Gradiente promedio: {sum(head_grads)/len(head_grads):.6f}")

    # Verificar que backbone no tiene gradientes (está congelado)
    backbone_grads = sum(1 for p in model.backbone.parameters() if p.grad is not None)
    print(f"  Tensores con gradiente en backbone (congelado): {backbone_grads}")

    assert backbone_grads == 0, "Backbone debería estar congelado"

    print("✓ Test 7 PASADO\n")


def test_training_step():
    """Simula un paso de entrenamiento completo en GPU."""
    print("=" * 60)
    print("TEST 8: Paso de entrenamiento completo (GPU)")
    print("=" * 60)

    # Setup
    model = create_model(freeze_backbone=True, device=DEVICE)
    model.train()

    optimizer = torch.optim.Adam(
        model.get_trainable_params()[1]['params'],  # Solo head
        lr=1e-3
    )
    loss_fn = CombinedLandmarkLoss().to(DEVICE)

    # Datos en GPU
    images = torch.rand(8, 3, 224, 224).to(DEVICE)
    targets = torch.rand(8, 30).to(DEVICE)

    # Paso de entrenamiento
    optimizer.zero_grad()
    predictions = model(images)
    losses = loss_fn(predictions, targets)
    losses['total'].backward()
    optimizer.step()

    print(f"  Paso de entrenamiento ejecutado exitosamente")
    print(f"  Loss: {losses['total'].item():.4f}")

    # Verificar que los pesos cambiaron
    predictions_after = model(images)
    losses_after = loss_fn(predictions_after, targets)

    print(f"  Loss después: {losses_after['total'].item():.4f}")

    # El loss debería ser diferente (probablemente menor)
    assert losses['total'].item() != losses_after['total'].item(), \
        "Los pesos deberían haber cambiado"

    print("✓ Test 8 PASADO\n")


def test_predict_landmarks_method():
    """Prueba método predict_landmarks en GPU."""
    print("=" * 60)
    print("TEST 9: Método predict_landmarks (GPU)")
    print("=" * 60)

    model = create_model(device=DEVICE)
    model.eval()

    images = torch.rand(4, 3, 224, 224).to(DEVICE)

    with torch.no_grad():
        # Método estándar
        output_flat = model(images)  # (4, 30) en [0,1]

        # Método predict_landmarks
        landmarks_px = model.predict_landmarks(images, image_size=224)  # (4, 15, 2) en px

    print(f"  Output flat shape: {output_flat.shape}")
    print(f"  Landmarks px shape: {landmarks_px.shape}")
    print(f"  Output flat range: [{output_flat.min():.3f}, {output_flat.max():.3f}]")
    print(f"  Landmarks px range: [{landmarks_px.min():.1f}, {landmarks_px.max():.1f}]")

    assert landmarks_px.shape == (4, 15, 2)
    assert landmarks_px.max() <= 224

    print("✓ Test 9 PASADO\n")


def main():
    """Ejecuta todas las pruebas."""
    print("\n" + "=" * 60)
    print("SESIÓN 2: Verificación de Modelo y Loss Functions")
    print("=" * 60 + "\n")

    print(f"Dispositivo: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    try:
        # Tests de modelo
        model = test_model_creation()
        test_unfreeze_backbone()
        test_differentiated_lr_groups()

        # Tests con datos reales
        val_loader = test_dataset_loading()
        images, targets, predictions = test_forward_pass_real_data(model, val_loader)

        # Tests de losses
        test_losses_real_data(predictions, targets)

        # Tests de entrenamiento
        test_backward_pass()
        test_training_step()
        test_predict_landmarks_method()

        print("=" * 60)
        print("✓ TODOS LOS TESTS PASARON EXITOSAMENTE")
        print("=" * 60)
        print("\nResumen:")
        print("  - ResNet18Landmarks: OK")
        print("  - Freeze/Unfreeze backbone: OK")
        print("  - LR diferenciado: OK")
        print("  - Dataset carga: OK")
        print("  - Forward pass (GPU): OK")
        print("  - WingLoss: OK")
        print("  - WeightedWingLoss: OK")
        print("  - CentralAlignmentLoss: OK")
        print("  - SoftSymmetryLoss: OK")
        print("  - CombinedLandmarkLoss: OK")
        print("  - Backward pass (GPU): OK")
        print("  - Training step (GPU): OK")
        print("\nListo para Sesión 3: Training Pipeline")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
