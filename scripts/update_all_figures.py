#!/usr/bin/env python3
"""
Actualizar todas las figuras con conexiones anatomicas correctas.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shape_analysis" / "figures"

# Cargar datos
data = np.load(PROJECT_ROOT / "outputs" / "predictions" / "all_landmarks.npz", allow_pickle=True)
all_landmarks = data['all_landmarks']
all_categories = data['all_categories']

aligned_data = np.load(PROJECT_ROOT / "outputs" / "shape_analysis" / "aligned_shapes.npz", allow_pickle=True)
aligned_shapes = aligned_data['aligned_shapes']
canonical_shape = aligned_data['canonical_shape']
canonical_pixels = aligned_data['canonical_shape_pixels']

# CONEXIONES ANATOMICAS CORRECTAS (indices 0-based)
EJE_CENTRAL = [0, 8, 9, 10, 1]
CONTORNO_IZQUIERDO = [0, 11, 2, 4, 6, 13, 1]
CONTORNO_DERECHO = [0, 12, 3, 5, 7, 14, 1]

def plot_shape_with_connections(ax, shape, color='blue', alpha=1.0, linewidth=2,
                                  show_points=True, point_size=50, label=None):
    ax.plot(shape[CONTORNO_IZQUIERDO, 0], shape[CONTORNO_IZQUIERDO, 1],
            color=color, alpha=alpha, linewidth=linewidth)
    ax.plot(shape[CONTORNO_DERECHO, 0], shape[CONTORNO_DERECHO, 1],
            color=color, alpha=alpha, linewidth=linewidth)
    ax.plot(shape[EJE_CENTRAL, 0], shape[EJE_CENTRAL, 1],
            color=color, alpha=alpha, linewidth=linewidth, linestyle='--')
    if show_points:
        ax.scatter(shape[:, 0], shape[:, 1], c=color, s=point_size,
                  zorder=5, alpha=alpha, label=label)

LANDMARK_NAMES = [
    "L1 (Superior)", "L2 (Inferior)", "L3 (Apex Izq)", "L4 (Apex Der)",
    "L5 (Hilio Izq)", "L6 (Hilio Der)", "L7 (Base Izq)", "L8 (Base Der)",
    "L9 (Centro Sup)", "L10 (Centro Med)", "L11 (Centro Inf)",
    "L12 (Borde Sup Izq)", "L13 (Borde Sup Der)",
    "L14 (Costof Izq)", "L15 (Costof Der)"
]

# =============================================================================
# ACTUALIZAR: canonical_shape.png
# =============================================================================
print("Actualizando canonical_shape.png...")

fig, ax = plt.subplots(figsize=(10, 10))

# Contornos con colores
ax.plot(canonical_pixels[CONTORNO_IZQUIERDO, 0], canonical_pixels[CONTORNO_IZQUIERDO, 1],
        'b-', linewidth=2, label='Contorno izquierdo')
ax.plot(canonical_pixels[CONTORNO_DERECHO, 0], canonical_pixels[CONTORNO_DERECHO, 1],
        'g-', linewidth=2, label='Contorno derecho')
ax.plot(canonical_pixels[EJE_CENTRAL, 0], canonical_pixels[EJE_CENTRAL, 1],
        'r--', linewidth=2, label='Eje central')

ax.scatter(canonical_pixels[:, 0], canonical_pixels[:, 1],
           c='blue', s=100, zorder=5)

for i, (x, y) in enumerate(canonical_pixels):
    ax.annotate(f'L{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
               fontsize=10, fontweight='bold')

ax.set_xlim(0, 224)
ax.set_ylim(224, 0)
ax.set_aspect('equal')
ax.set_title('Forma Canónica (GPA)\n15 Landmarks Anatómicos', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "canonical_shape.png", dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# ACTUALIZAR: aligned_shapes_sample.png
# =============================================================================
print("Actualizando aligned_shapes_sample.png...")

fig, ax = plt.subplots(figsize=(10, 10))

n_samples = 100
indices = np.random.choice(len(aligned_shapes), n_samples, replace=False)

for idx in indices:
    plot_shape_with_connections(ax, aligned_shapes[idx], color='gray', alpha=0.2,
                                 linewidth=0.5, show_points=False)

plot_shape_with_connections(ax, canonical_shape, color='red', linewidth=2,
                             point_size=60, label='Forma Canónica')

ax.set_aspect('equal')
ax.set_title(f'Formas Alineadas por GPA\n({n_samples} de {len(aligned_shapes)} formas)', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "aligned_shapes_sample.png", dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# ACTUALIZAR: pca_modes_variation.png
# =============================================================================
print("Actualizando pca_modes_variation.png...")

# PCA
n_shapes, n_landmarks, n_dims = aligned_shapes.shape
flat_shapes = aligned_shapes.reshape(n_shapes, -1)
mean_shape = flat_shapes.mean(axis=0)
centered = flat_shapes - mean_shape
cov_matrix = np.cov(centered.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order][:5]
eigenvectors = eigenvectors[:, order][:, :5]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for pc_idx, ax_row in enumerate(axes):
    eigenvec = eigenvectors[:, pc_idx].reshape(n_landmarks, 2)
    std = np.sqrt(eigenvalues[pc_idx])
    var_explained = eigenvalues[pc_idx] / eigenvalues.sum() * 100

    for col_idx, sigma in enumerate([-2, 0, 2]):
        ax = ax_row[col_idx]
        varied_shape = canonical_shape + sigma * std * eigenvec

        plot_shape_with_connections(ax, varied_shape, color='blue', linewidth=1.5, point_size=40)

        ax.set_aspect('equal')
        ax.set_title(f'PC{pc_idx+1} ({var_explained:.1f}%): {sigma:+d}σ', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

plt.suptitle('Modos de Variación PCA\n(sobre formas alineadas por GPA)', fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pca_modes_variation.png", dpi=150, bbox_inches='tight')
plt.close()

print("\nFiguras actualizadas:")
print("  - canonical_shape.png")
print("  - aligned_shapes_sample.png")
print("  - pca_modes_variation.png")
