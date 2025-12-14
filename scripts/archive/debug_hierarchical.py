#!/usr/bin/env python
"""
Script de debug para analizar los parámetros geométricos reales del dataset
y compararlos con los valores hardcodeados en el modelo jerárquico.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src_v2.data.dataset import create_dataloaders

def analyze_geometric_params():
    """Analizar parámetros geométricos reales de los landmarks."""

    # Cargar datos
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path="data/coordenadas/coordenadas_maestro.csv",
        data_root="data",
        batch_size=32,
        use_clahe=True,
        clahe_tile_size=4,
        random_state=42
    )

    # Recolectar todos los landmarks
    all_landmarks = []
    for batch in train_loader:
        landmarks = batch[1]  # (B, 30)
        landmarks = landmarks.view(-1, 15, 2)  # (B, 15, 2)
        all_landmarks.append(landmarks)

    all_landmarks = torch.cat(all_landmarks, dim=0)  # (N, 15, 2)
    print(f"Total samples: {all_landmarks.shape[0]}")

    # Extraer eje
    L1 = all_landmarks[:, 0]  # (N, 2)
    L2 = all_landmarks[:, 1]  # (N, 2)

    axis_vec = L2 - L1  # (N, 2)
    axis_len = torch.norm(axis_vec, dim=1, keepdim=True)  # (N, 1)
    axis_unit = axis_vec / (axis_len + 1e-8)  # (N, 2)

    # Vector perpendicular
    perp_unit = torch.stack([-axis_unit[:, 1], axis_unit[:, 0]], dim=1)  # (N, 2)

    print("\n" + "="*60)
    print("ANÁLISIS DE PARÁMETROS GEOMÉTRICOS")
    print("="*60)

    print(f"\nLongitud del eje (L1-L2):")
    print(f"  Mean: {axis_len.mean():.4f}")
    print(f"  Std:  {axis_len.std():.4f}")
    print(f"  Min:  {axis_len.min():.4f}")
    print(f"  Max:  {axis_len.max():.4f}")

    # Pares bilaterales
    BILATERAL_PAIRS = [
        (2, 3),   # L3, L4 - Apices
        (4, 5),   # L5, L6 - Hilios
        (6, 7),   # L7, L8 - Bases
        (11, 12), # L12, L13 - Bordes superiores
        (13, 14), # L14, L15 - Costofrenicos
    ]

    PAIR_NAMES = ["L3,L4 (Apices)", "L5,L6 (Hilios)", "L7,L8 (Bases)",
                  "L12,L13 (Bordes sup)", "L14,L15 (Costofrenicos)"]

    print("\n" + "="*60)
    print("PARÁMETRO t (posición relativa en el eje)")
    print("="*60)
    print("(t=0 en L1, t=1 en L2)")

    bilateral_t_real = []

    for pair_idx, (left_idx, right_idx) in enumerate(BILATERAL_PAIRS):
        # Centro del par bilateral
        left = all_landmarks[:, left_idx]
        right = all_landmarks[:, right_idx]
        center = (left + right) / 2  # (N, 2)

        # Proyección sobre el eje para encontrar t
        vec_to_center = center - L1  # (N, 2)

        # t = (vec_to_center · axis_unit) / axis_len
        proj_len = (vec_to_center * axis_unit).sum(dim=1)  # (N,)
        t = proj_len / (axis_len.squeeze() + 1e-8)  # (N,)

        t_mean = t.mean().item()
        t_std = t.std().item()
        bilateral_t_real.append(t_mean)

        print(f"\n{PAIR_NAMES[pair_idx]}:")
        print(f"  t_mean: {t_mean:.4f} ± {t_std:.4f}")
        print(f"  t_range: [{t.min():.4f}, {t.max():.4f}]")

    print("\n" + "="*60)
    print("COMPARACIÓN CON VALORES HARDCODEADOS")
    print("="*60)

    hardcoded_t = [0.25, 0.50, 0.75, 0.10, 0.90]

    print("\n{:<25} {:>10} {:>10} {:>10}".format("Par", "Real", "Hardcoded", "Error"))
    print("-" * 60)
    for i, name in enumerate(PAIR_NAMES):
        real = bilateral_t_real[i]
        hard = hardcoded_t[i]
        error = abs(real - hard)
        status = "✓" if error < 0.1 else "✗ BUG!"
        print(f"{name:<25} {real:>10.4f} {hard:>10.4f} {error:>10.4f} {status}")

    print("\n" + "="*60)
    print("DISTANCIAS PERPENDICULARES")
    print("="*60)

    for pair_idx, (left_idx, right_idx) in enumerate(BILATERAL_PAIRS):
        left = all_landmarks[:, left_idx]
        right = all_landmarks[:, right_idx]

        # Centro del par
        center = (left + right) / 2
        vec_to_center = center - L1

        # Proyección sobre el eje
        t = (vec_to_center * axis_unit).sum(dim=1, keepdim=True)
        base_point = L1 + t * axis_unit

        # Distancia perpendicular izquierda y derecha
        d_left = torch.norm(left - base_point, dim=1)  # (N,)
        d_right = torch.norm(right - base_point, dim=1)  # (N,)

        # También calcular como fracción del eje
        d_left_norm = d_left / (axis_len.squeeze() + 1e-8)
        d_right_norm = d_right / (axis_len.squeeze() + 1e-8)

        print(f"\n{PAIR_NAMES[pair_idx]}:")
        print(f"  d_left:  {d_left.mean():.4f} ± {d_left.std():.4f} (abs)")
        print(f"  d_left:  {d_left_norm.mean():.4f} ± {d_left_norm.std():.4f} (norm por axis_len)")
        print(f"  d_right: {d_right.mean():.4f} ± {d_right.std():.4f} (abs)")
        print(f"  d_right: {d_right_norm.mean():.4f} ± {d_right_norm.std():.4f} (norm por axis_len)")

    print("\n" + "="*60)
    print("LANDMARKS CENTRALES (L9, L10, L11)")
    print("="*60)

    for idx, name in [(8, "L9"), (9, "L10"), (10, "L11")]:
        point = all_landmarks[:, idx]
        vec = point - L1
        t = (vec * axis_unit).sum(dim=1) / (axis_len.squeeze() + 1e-8)

        # Distancia perpendicular al eje
        proj = t.unsqueeze(1) * axis_unit
        perp = vec - proj
        perp_dist = torch.norm(perp, dim=1)

        print(f"\n{name}:")
        print(f"  t_mean: {t.mean():.4f} ± {t.std():.4f} (teórico: {0.25 if idx==8 else 0.50 if idx==9 else 0.75})")
        print(f"  perp_dist: {perp_dist.mean():.6f} (debería ser ~0)")

    print("\n" + "="*60)
    print("RESUMEN DE BUGS")
    print("="*60)

    print("\nBUG 1: bilateral_t_base tiene valores incorrectos")
    print(f"  Hardcoded: {hardcoded_t}")
    print(f"  Real:      {[round(t, 4) for t in bilateral_t_real]}")

    print("\nBUG 2: Las distancias perpendiculares se multiplican por axis_len")
    print("  Esto causa que landmarks se muevan proporcionalmente al tamaño del eje")
    print("  Las distancias deberían ser absolutas o normalizadas diferentemente")

    print("\n" + "="*60)
    print("VALORES CORREGIDOS SUGERIDOS")
    print("="*60)
    print(f"bilateral_t_base = {[round(t, 2) for t in bilateral_t_real]}")

    return bilateral_t_real


if __name__ == "__main__":
    analyze_geometric_params()
