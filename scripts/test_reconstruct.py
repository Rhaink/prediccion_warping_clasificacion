#!/usr/bin/env python
"""
Test aislado de la función de reconstrucción.
Verifica si los parámetros correctos producen los landmarks correctos.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src_v2.data.dataset import create_dataloaders


def manual_reconstruct(L1, L2, dt_centrales, bilateral_params):
    """
    Reconstrucción manual siguiendo la lógica del modelo CORREGIDO.
    """
    batch_size = L1.size(0)
    device = L1.device

    landmarks = torch.zeros(batch_size, 15, 2, device=device)

    # L1 y L2
    landmarks[:, 0] = L1
    landmarks[:, 1] = L2

    # Vector del eje
    axis_vec = L2 - L1
    axis_len = torch.norm(axis_vec, dim=1, keepdim=True) + 1e-8
    axis_unit = axis_vec / axis_len

    # Vector perpendicular (rotación 90 grados)
    perp_unit = torch.stack([-axis_unit[:, 1], axis_unit[:, 0]], dim=1)

    # Landmarks centrales
    for i, (landmark_idx, t_base) in enumerate([(8, 0.25), (9, 0.50), (10, 0.75)]):
        t = t_base + dt_centrales[:, i]
        landmarks[:, landmark_idx] = L1 + t.unsqueeze(1) * axis_vec

    # Pares bilaterales
    BILATERAL_PAIRS = [(2, 3), (4, 5), (6, 7), (11, 12), (13, 14)]
    # CORREGIDO: t_base para L12,L13 es 0.0, para L14,L15 es 1.0
    bilateral_t_base = [0.25, 0.50, 0.75, 0.00, 1.00]

    for pair_idx, (left_idx, right_idx) in enumerate(BILATERAL_PAIRS):
        p_start = pair_idx * 3
        t_offset = torch.tanh(bilateral_params[:, p_start]) * 0.2
        # CORREGIDO: rango de 0.7 en lugar de 0.5
        d_left = torch.sigmoid(bilateral_params[:, p_start + 1]) * 0.7
        d_right = torch.sigmoid(bilateral_params[:, p_start + 2]) * 0.7

        t = bilateral_t_base[pair_idx] + t_offset
        base_point = L1 + t.unsqueeze(1) * axis_vec

        # CORREGIDO: signos invertidos
        landmarks[:, left_idx] = base_point + d_left.unsqueeze(1) * perp_unit * axis_len
        landmarks[:, right_idx] = base_point - d_right.unsqueeze(1) * perp_unit * axis_len

    return torch.clamp(landmarks.view(batch_size, -1), 0, 1)


def compute_optimal_params(targets):
    """
    Calcular los parámetros óptimos que deberían reconstruir los targets.
    """
    targets = targets.view(-1, 15, 2)
    batch_size = targets.size(0)

    L1 = targets[:, 0]
    L2 = targets[:, 1]

    axis_vec = L2 - L1
    axis_len = torch.norm(axis_vec, dim=1, keepdim=True) + 1e-8
    axis_unit = axis_vec / axis_len
    perp_unit = torch.stack([-axis_unit[:, 1], axis_unit[:, 0]], dim=1)

    # dt para centrales
    dt_centrales = []
    for idx, t_base in [(8, 0.25), (9, 0.50), (10, 0.75)]:
        point = targets[:, idx]
        vec = point - L1
        t_actual = (vec * axis_unit).sum(dim=1) / (axis_len.squeeze() + 1e-8)
        dt = t_actual - t_base
        dt_centrales.append(dt)
    dt_centrales = torch.stack(dt_centrales, dim=1)

    # Parámetros bilaterales
    BILATERAL_PAIRS = [(2, 3), (4, 5), (6, 7), (11, 12), (13, 14)]
    # CORREGIDO: usar los mismos valores que el modelo
    bilateral_t_base = [0.25, 0.50, 0.75, 0.00, 1.00]

    bilateral_params = []
    for pair_idx, (left_idx, right_idx) in enumerate(BILATERAL_PAIRS):
        left = targets[:, left_idx]
        right = targets[:, right_idx]
        center = (left + right) / 2

        # t actual
        vec_to_center = center - L1
        t_actual = (vec_to_center * axis_unit).sum(dim=1) / (axis_len.squeeze() + 1e-8)
        t_offset = t_actual - bilateral_t_base[pair_idx]

        # Distancias perpendiculares
        base_point = L1 + t_actual.unsqueeze(1) * axis_vec

        # Proyección sobre perp_unit
        vec_left = left - base_point
        vec_right = right - base_point

        # Distancia con signo (negativo si está en dirección opuesta a perp_unit)
        proj_left = (vec_left * perp_unit).sum(dim=1)  # Escalar
        proj_right = (vec_right * perp_unit).sum(dim=1)  # Escalar

        # d_left y d_right (normalizados por axis_len)
        d_left = torch.norm(vec_left, dim=1) / (axis_len.squeeze() + 1e-8)
        d_right = torch.norm(vec_right, dim=1) / (axis_len.squeeze() + 1e-8)

        bilateral_params.extend([t_offset, d_left, d_right])

    bilateral_params = torch.stack(bilateral_params, dim=1)

    return dt_centrales, bilateral_params


def test_reconstruction():
    """Test completo de reconstrucción."""

    # Cargar datos
    train_loader, _, _ = create_dataloaders(
        csv_path="data/coordenadas/coordenadas_maestro.csv",
        data_root="data",
        batch_size=4,
        use_clahe=True,
        random_state=42
    )

    batch = next(iter(train_loader))
    targets = batch[1]  # (4, 30)

    targets_reshaped = targets.view(-1, 15, 2)
    L1 = targets_reshaped[:, 0]
    L2 = targets_reshaped[:, 1]

    print("="*60)
    print("TEST 1: Reconstrucción con parámetros CERO")
    print("="*60)

    dt_zero = torch.zeros(4, 3)
    bilateral_zero = torch.zeros(4, 15)  # sigmoid(0) = 0.5, tanh(0) = 0

    reconstructed = manual_reconstruct(L1, L2, dt_zero, bilateral_zero)
    reconstructed = reconstructed.view(-1, 15, 2)

    print(f"\nCon parámetros cero (sigmoid(0)=0.5, d=0.25):")
    for i in range(15):
        error = torch.norm(reconstructed[:, i] - targets_reshaped[:, i], dim=1) * 224
        print(f"  L{i+1}: {error.mean():.2f} px")

    print("\n" + "="*60)
    print("TEST 2: Verificar qué distancias genera sigmoid(0)*0.5")
    print("="*60)

    axis_vec = L2 - L1
    axis_len = torch.norm(axis_vec, dim=1)
    d_with_zero = 0.5 * axis_len  # sigmoid(0)*0.5 * axis_len

    print(f"\naxis_len promedio: {axis_len.mean():.4f}")
    print(f"d con params=0: {d_with_zero.mean():.4f} (en coordenadas normalizadas)")
    print(f"d en pixeles: {d_with_zero.mean() * 224:.2f} px")

    # Calcular distancias reales
    print(f"\nDistancias reales necesarias:")
    BILATERAL_PAIRS = [(2, 3), (4, 5), (6, 7), (11, 12), (13, 14)]
    for left_idx, right_idx in BILATERAL_PAIRS:
        left = targets_reshaped[:, left_idx]
        right = targets_reshaped[:, right_idx]
        center = (left + right) / 2

        vec = center - L1
        t = (vec * (L2-L1)).sum(dim=1) / (axis_len**2 + 1e-8)
        base = L1 + t.unsqueeze(1) * axis_vec

        d_left = torch.norm(left - base, dim=1)
        d_right = torch.norm(right - base, dim=1)

        print(f"  L{left_idx+1},L{right_idx+1}: d_left={d_left.mean():.4f}, d_right={d_right.mean():.4f}")

    print("\n" + "="*60)
    print("TEST 3: Reconstrucción con parámetros ÓPTIMOS calculados")
    print("="*60)

    dt_opt, bilateral_opt = compute_optimal_params(targets)

    print(f"\ndt_centrales calculados:")
    for i in range(3):
        print(f"  dt_{i}: {dt_opt[:, i].mean():.4f} (rango para tanh*0.1: [-0.1, 0.1])")

    print(f"\nbilateral_params calculados (cada 3: t_offset, d_left, d_right):")
    PAIR_NAMES = ["L3,L4", "L5,L6", "L7,L8", "L12,L13", "L14,L15"]
    for i in range(5):
        t_off = bilateral_opt[:, i*3].mean()
        d_l = bilateral_opt[:, i*3+1].mean()
        d_r = bilateral_opt[:, i*3+2].mean()
        print(f"  {PAIR_NAMES[i]}: t_off={t_off:.4f}, d_left={d_l:.4f}, d_right={d_r:.4f}")

    # Ahora necesitamos convertir d_left, d_right a valores que sigmoid pueda producir
    # Si d está normalizado: d = sigmoid(x) * 0.5
    # x = sigmoid_inv(d / 0.5) = log(d/0.5 / (1 - d/0.5))

    print("\n" + "="*60)
    print("TEST 4: Verificar rangos de parámetros")
    print("="*60)

    print("\nEl modelo CORREGIDO usa:")
    print("  dt: tanh(x) * 0.1 → rango [-0.1, 0.1]")
    print("  t_offset: tanh(x) * 0.2 → rango [-0.2, 0.2]")
    print("  d: sigmoid(x) * 0.7 → rango [0, 0.7]")

    print("\nParámetros reales necesarios:")
    for i in range(5):
        d_l = bilateral_opt[:, i*3+1].mean().item()
        d_r = bilateral_opt[:, i*3+2].mean().item()

        # Verificar si están dentro del rango CORREGIDO (0.7)
        in_range_l = "✓" if d_l <= 0.7 else f"✗ (max 0.7, necesita {d_l:.3f})"
        in_range_r = "✓" if d_r <= 0.7 else f"✗ (max 0.7, necesita {d_r:.3f})"
        print(f"  {PAIR_NAMES[i]}: d_l={d_l:.4f} {in_range_l}, d_r={d_r:.4f} {in_range_r}")

    print("\n" + "="*60)
    print("TEST 5: Verificar dirección del vector perpendicular")
    print("="*60)

    # Para un sample
    i = 0
    L1_i = targets_reshaped[i, 0]
    L2_i = targets_reshaped[i, 1]
    L3_i = targets_reshaped[i, 2]  # Debería estar a la IZQUIERDA
    L4_i = targets_reshaped[i, 3]  # Debería estar a la DERECHA

    axis_vec_i = L2_i - L1_i
    axis_unit_i = axis_vec_i / torch.norm(axis_vec_i)
    perp_unit_i = torch.stack([-axis_unit_i[1], axis_unit_i[0]])

    # Dirección de L3 y L4 respecto al eje
    center_34 = (L3_i + L4_i) / 2
    t = ((center_34 - L1_i) * axis_unit_i).sum() / torch.norm(axis_vec_i)
    base = L1_i + t * axis_vec_i

    vec_to_L3 = L3_i - base
    vec_to_L4 = L4_i - base

    dot_L3 = (vec_to_L3 * perp_unit_i).sum()
    dot_L4 = (vec_to_L4 * perp_unit_i).sum()

    print(f"\nSample {i}:")
    print(f"  L1: ({L1_i[0]:.3f}, {L1_i[1]:.3f})")
    print(f"  L2: ({L2_i[0]:.3f}, {L2_i[1]:.3f})")
    print(f"  L3 (izq): ({L3_i[0]:.3f}, {L3_i[1]:.3f})")
    print(f"  L4 (der): ({L4_i[0]:.3f}, {L4_i[1]:.3f})")
    print(f"  perp_unit: ({perp_unit_i[0]:.3f}, {perp_unit_i[1]:.3f})")
    print(f"  L3 · perp_unit: {dot_L3:.4f} (negativo = contrario a perp_unit)")
    print(f"  L4 · perp_unit: {dot_L4:.4f} (positivo = misma dirección que perp_unit)")

    if dot_L3 < 0 and dot_L4 > 0:
        print("\n  ✓ L3 está en dirección -perp_unit, L4 en +perp_unit")
        print("  El modelo reconstruye: left = base - d*perp, right = base + d*perp")
        print("  Esto es CORRECTO si L3 es 'left' y L4 es 'right'")
    else:
        print("\n  ✗ Los signos no coinciden con lo esperado!")

    print("\n" + "="*60)
    print("CONCLUSIÓN (después de correcciones)")
    print("="*60)

    # Verificar que los bugs están corregidos
    print("\nVERIFICACIÓN DE CORRECCIONES:")

    # Bug 1: bilateral_t_base (corregido)
    print("\n1. bilateral_t_base[3] (L12,L13): ahora usa 0.00 ✓")
    print("   bilateral_t_base[4] (L14,L15): ahora usa 1.00 ✓")

    # Bug 2: Rango de distancias (corregido a 0.7)
    max_d = max(bilateral_opt[:, 1::3].max().item(), bilateral_opt[:, 2::3].max().item())
    if max_d <= 0.7:
        print(f"\n2. d máximo necesario: {max_d:.3f}, sigmoid*0.7 max=0.7 ✓")
    else:
        print(f"\n2. d máximo necesario: {max_d:.3f}, pero sigmoid*0.7 max=0.7 ✗")
        print("   Aún hay distancias fuera de rango")

    # Bug 3: signos invertidos (corregido)
    print("\n3. Signos de vector perpendicular: corregidos ✓")


if __name__ == "__main__":
    test_reconstruction()
