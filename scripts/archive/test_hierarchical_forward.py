#!/usr/bin/env python
"""
Test del forward pass del modelo jerárquico con datos reales.
Compara predicciones con GT para identificar dónde está el bug.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src_v2.data.dataset import create_dataloaders
from src_v2.models.hierarchical import HierarchicalLandmarkModel

def test_forward():
    """Test forward pass y analizar predicciones."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Cargar datos
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path="data/coordenadas/coordenadas_maestro.csv",
        data_root="data",
        batch_size=4,
        use_clahe=True,
        clahe_tile_size=4,
        random_state=42
    )

    # Modelo SIN entrenar (random weights)
    model = HierarchicalLandmarkModel().to(device)
    model.eval()

    # Tomar un batch
    batch = next(iter(test_loader))
    images = batch[0].to(device)
    targets = batch[1].to(device)  # (B, 30)

    print(f"\nImages shape: {images.shape}")
    print(f"Targets shape: {targets.shape}")

    # Forward pass
    with torch.no_grad():
        preds = model(images)

    print(f"Predictions shape: {preds.shape}")
    print(f"Predictions range: [{preds.min():.4f}, {preds.max():.4f}]")

    # Reshape
    preds = preds.view(-1, 15, 2)
    targets = targets.view(-1, 15, 2)

    # Analizar errores por landmark
    print("\n" + "="*60)
    print("ERRORES POR LANDMARK (modelo sin entrenar)")
    print("="*60)

    errors_per_landmark = []
    for i in range(15):
        error = torch.norm(preds[:, i] - targets[:, i], dim=1) * 224
        errors_per_landmark.append(error.mean().item())
        print(f"L{i+1}: {error.mean():.2f} px")

    print(f"\nError promedio: {np.mean(errors_per_landmark):.2f} px")

    # Ahora cargar modelo ENTRENADO si existe
    checkpoint_path = "checkpoints/session13/hierarchical/final_model.pt"
    if os.path.exists(checkpoint_path):
        print("\n" + "="*60)
        print("CARGANDO MODELO ENTRENADO")
        print("="*60)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():
            preds_trained = model(images)

        preds_trained = preds_trained.view(-1, 15, 2)

        print("\nERRORES POR LANDMARK (modelo entrenado)")
        errors_trained = []
        for i in range(15):
            error = torch.norm(preds_trained[:, i] - targets[:, i], dim=1) * 224
            errors_trained.append(error.mean().item())
            print(f"L{i+1}: {error.mean():.2f} px")

        print(f"\nError promedio: {np.mean(errors_trained):.2f} px")

        # Comparar predicciones vs targets para L1, L2
        print("\n" + "="*60)
        print("ANÁLISIS DEL EJE (L1, L2)")
        print("="*60)

        for i in range(min(4, preds_trained.shape[0])):
            print(f"\nSample {i}:")
            print(f"  L1 pred:   ({preds_trained[i,0,0]:.4f}, {preds_trained[i,0,1]:.4f})")
            print(f"  L1 target: ({targets[i,0,0]:.4f}, {targets[i,0,1]:.4f})")
            print(f"  L2 pred:   ({preds_trained[i,1,0]:.4f}, {preds_trained[i,1,1]:.4f})")
            print(f"  L2 target: ({targets[i,1,0]:.4f}, {targets[i,1,1]:.4f})")

            # Error del eje
            error_L1 = torch.norm(preds_trained[i,0] - targets[i,0]) * 224
            error_L2 = torch.norm(preds_trained[i,1] - targets[i,1]) * 224
            print(f"  Error L1: {error_L1:.2f} px, Error L2: {error_L2:.2f} px")

        # Verificar si los bilaterales están en las posiciones correctas
        print("\n" + "="*60)
        print("VERIFICACIÓN DE BILATERALES")
        print("="*60)

        BILATERAL_PAIRS = [
            (2, 3, "L3,L4"),
            (4, 5, "L5,L6"),
            (6, 7, "L7,L8"),
            (11, 12, "L12,L13"),
            (13, 14, "L14,L15"),
        ]

        for left_idx, right_idx, name in BILATERAL_PAIRS:
            error_left = torch.norm(preds_trained[:, left_idx] - targets[:, left_idx], dim=1) * 224
            error_right = torch.norm(preds_trained[:, right_idx] - targets[:, right_idx], dim=1) * 224
            print(f"{name}: Left={error_left.mean():.2f}px, Right={error_right.mean():.2f}px")

        # Análisis detallado de un sample
        print("\n" + "="*60)
        print("ANÁLISIS DETALLADO SAMPLE 0")
        print("="*60)

        sample_idx = 0
        pred = preds_trained[sample_idx]
        target = targets[sample_idx]

        print("\n{:<6} {:>20} {:>20} {:>10}".format("Land", "Predicción", "Target", "Error(px)"))
        print("-" * 60)
        for i in range(15):
            error = torch.norm(pred[i] - target[i]) * 224
            print(f"L{i+1:<4} ({pred[i,0]:.3f}, {pred[i,1]:.3f})  ({target[i,0]:.3f}, {target[i,1]:.3f})  {error:.1f}")

    else:
        print(f"\nNo se encontró checkpoint en {checkpoint_path}")
        print("Ejecuta train_hierarchical.py primero.")


def analyze_reconstruction_directly():
    """
    Analizar la reconstrucción directamente con parámetros conocidos del GT.
    """
    print("\n" + "="*60)
    print("ANÁLISIS DE RECONSTRUCCIÓN CON PARÁMETROS DEL GT")
    print("="*60)

    device = torch.device("cpu")

    # Cargar datos
    train_loader, _, _ = create_dataloaders(
        csv_path="data/coordenadas/coordenadas_maestro.csv",
        data_root="data",
        batch_size=1,
        use_clahe=True,
        random_state=42
    )

    batch = next(iter(train_loader))
    targets = batch[1]  # (1, 30)
    targets = targets.view(1, 15, 2)

    # Extraer parámetros reales del GT
    L1_gt = targets[:, 0]  # (1, 2)
    L2_gt = targets[:, 1]  # (1, 2)

    axis_vec = L2_gt - L1_gt
    axis_len = torch.norm(axis_vec, dim=1, keepdim=True)
    axis_unit = axis_vec / (axis_len + 1e-8)
    perp_unit = torch.stack([-axis_unit[:, 1], axis_unit[:, 0]], dim=1)

    print(f"\nEje L1-L2:")
    print(f"  L1: ({L1_gt[0,0]:.4f}, {L1_gt[0,1]:.4f})")
    print(f"  L2: ({L2_gt[0,0]:.4f}, {L2_gt[0,1]:.4f})")
    print(f"  axis_len: {axis_len.item():.4f}")

    # Reconstruir centrales manualmente
    print(f"\nLandmarks centrales:")
    for idx, t_base in [(8, 0.25), (9, 0.50), (10, 0.75)]:
        # Landmark del GT
        gt = targets[:, idx]

        # Reconstruido con t_base
        reconstructed = L1_gt + t_base * axis_vec

        error = torch.norm(reconstructed - gt) * 224
        print(f"  L{idx+1}: GT=({gt[0,0]:.4f}, {gt[0,1]:.4f}), Rec=({reconstructed[0,0]:.4f}, {reconstructed[0,1]:.4f}), Error={error:.2f}px")

    # Reconstruir bilaterales manualmente
    print(f"\nLandmarks bilaterales:")
    BILATERAL_PAIRS = [
        (2, 3, 0.25, "L3,L4"),
        (4, 5, 0.50, "L5,L6"),
        (6, 7, 0.75, "L7,L8"),
        (11, 12, 0.10, "L12,L13 (hardcoded=0.10)"),
        (13, 14, 0.90, "L14,L15"),
    ]

    for left_idx, right_idx, t_hard, name in BILATERAL_PAIRS:
        gt_left = targets[:, left_idx]
        gt_right = targets[:, right_idx]

        # Calcular t real
        center = (gt_left + gt_right) / 2
        vec_to_center = center - L1_gt
        t_real = (vec_to_center * axis_unit).sum(dim=1) / (axis_len.squeeze() + 1e-8)

        # Calcular distancias perpendiculares reales
        base_point_real = L1_gt + t_real.unsqueeze(1) * axis_vec
        d_left_real = torch.norm(gt_left - base_point_real)
        d_right_real = torch.norm(gt_right - base_point_real)

        # Reconstruir con t hardcoded
        base_point_hard = L1_gt + t_hard * axis_vec

        # Usar distancias reales pero con t hardcoded
        # Signo de la distancia (izq vs der)
        sign_left = torch.sign((gt_left - base_point_real) @ perp_unit.T).squeeze()
        sign_right = torch.sign((gt_right - base_point_real) @ perp_unit.T).squeeze()

        reconstructed_left = base_point_hard - sign_left * d_left_real * perp_unit
        reconstructed_right = base_point_hard + sign_right * d_right_real * perp_unit

        error_left = torch.norm(reconstructed_left - gt_left) * 224
        error_right = torch.norm(reconstructed_right - gt_right) * 224

        print(f"\n  {name}:")
        print(f"    t_real={t_real.item():.4f}, t_hard={t_hard:.4f}, diff={abs(t_real.item()-t_hard):.4f}")
        print(f"    d_left={d_left_real.item():.4f}, d_right={d_right_real.item():.4f}")
        print(f"    Error con t_hard: Left={error_left:.2f}px, Right={error_right:.2f}px")


if __name__ == "__main__":
    test_forward()
    print("\n\n")
    analyze_reconstruction_directly()
