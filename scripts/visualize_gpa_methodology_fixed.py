#!/usr/bin/env python3
"""
Visualizaciones del proceso GPA para la Metodologia de la Tesis.
VERSION CORREGIDA con conexiones anatomicas correctas.

Conexiones correctas:
- Eje central: 1 → 9 → 10 → 11 → 2
- Contorno izquierdo: 1 → 12 → 3 → 5 → 7 → 14 → 2
- Contorno derecho: 1 → 13 → 4 → 6 → 8 → 15 → 2
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shape_analysis" / "figures" / "methodology"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from scripts.gpa_analysis import (
    center_shape, scale_shape, align_shape, optimal_rotation_matrix,
    LANDMARK_NAMES
)

# CONEXIONES ANATOMICAS CORRECTAS (indices 0-based)
EJE_CENTRAL = [0, 8, 9, 10, 1]  # L1 → L9 → L10 → L11 → L2
CONTORNO_IZQUIERDO = [0, 11, 2, 4, 6, 13, 1]  # L1 → L12 → L3 → L5 → L7 → L14 → L2
CONTORNO_DERECHO = [0, 12, 3, 5, 7, 14, 1]  # L1 → L13 → L4 → L6 → L8 → L15 → L2

def plot_shape_with_connections(ax, shape, color='blue', alpha=1.0, linewidth=2,
                                  show_points=True, point_size=50, label=None):
    """Plotear forma con conexiones anatomicas correctas."""
    # Contorno izquierdo
    ax.plot(shape[CONTORNO_IZQUIERDO, 0], shape[CONTORNO_IZQUIERDO, 1],
            color=color, alpha=alpha, linewidth=linewidth)
    # Contorno derecho
    ax.plot(shape[CONTORNO_DERECHO, 0], shape[CONTORNO_DERECHO, 1],
            color=color, alpha=alpha, linewidth=linewidth)
    # Eje central
    ax.plot(shape[EJE_CENTRAL, 0], shape[EJE_CENTRAL, 1],
            color=color, alpha=alpha, linewidth=linewidth, linestyle='--')

    if show_points:
        ax.scatter(shape[:, 0], shape[:, 1], c=color, s=point_size,
                  zorder=5, alpha=alpha, label=label)

# Cargar datos
print("Cargando datos...")
data = np.load(PROJECT_ROOT / "outputs" / "predictions" / "all_landmarks.npz", allow_pickle=True)
all_landmarks = data['all_landmarks']
all_categories = data['all_categories']

aligned_data = np.load(PROJECT_ROOT / "outputs" / "shape_analysis" / "aligned_shapes.npz", allow_pickle=True)
aligned_shapes = aligned_data['aligned_shapes']
canonical_shape = aligned_data['canonical_shape']
canonical_pixels = aligned_data['canonical_shape_pixels']

# Seleccionar 3 formas representativas
np.random.seed(42)
idx_normal = np.where(all_categories == 'Normal')[0][10]
idx_covid = np.where(all_categories == 'COVID')[0][15]
idx_viral = np.where(all_categories == 'Viral_Pneumonia')[0][5]
sample_indices = [idx_normal, idx_covid, idx_viral]
sample_names = ['Normal', 'COVID', 'Viral']
sample_colors = ['green', 'red', 'blue']

# =============================================================================
# FIGURA 1: EL PROBLEMA - VARIABILIDAD EN FORMAS ORIGINALES
# =============================================================================
print("\n1. Generando: Variabilidad en formas originales...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: Formas originales
ax1 = axes[0]
ax1.set_title('a) Formas Originales\n(coordenadas de imagen)', fontsize=12, fontweight='bold')

for idx, name, color in zip(sample_indices, sample_names, sample_colors):
    shape = all_landmarks[idx]
    plot_shape_with_connections(ax1, shape, color=color, alpha=0.8, linewidth=1.5,
                                 point_size=40, label=name)

ax1.set_xlim(0, 224)
ax1.set_ylim(224, 0)  # Y invertido
ax1.set_xlabel('X (pixeles)', fontsize=11)
ax1.set_ylabel('Y (pixeles)', fontsize=11)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Panel 2: Histograma de centroides
ax2 = axes[1]
ax2.set_title('b) Variabilidad de Centroides\n(traslación)', fontsize=12, fontweight='bold')

centroids = np.array([all_landmarks[i].mean(axis=0) for i in range(len(all_landmarks))])
ax2.scatter(centroids[:, 0], centroids[:, 1], c='blue', alpha=0.3, s=20)
ax2.scatter(centroids.mean(axis=0)[0], centroids.mean(axis=0)[1],
           c='red', s=200, marker='x', linewidths=3, label='Promedio')
ax2.set_xlabel('X centroide (pixeles)', fontsize=11)
ax2.set_ylabel('Y centroide (pixeles)', fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

# Panel 3: Histograma de escalas
ax3 = axes[2]
ax3.set_title('c) Variabilidad de Escalas\n(tamaño)', fontsize=12, fontweight='bold')

scales = []
for i in range(len(all_landmarks)):
    centered, _ = center_shape(all_landmarks[i])
    scale = np.linalg.norm(centered, 'fro')
    scales.append(scale)
scales = np.array(scales)

ax3.hist(scales, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
ax3.axvline(scales.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Media: {scales.mean():.1f}')
ax3.set_xlabel('Escala (norma Frobenius)', fontsize=11)
ax3.set_ylabel('Frecuencia', fontsize=11)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_problema_variabilidad.png", dpi=150, bbox_inches='tight')
print(f"  Guardado: {OUTPUT_DIR / '01_problema_variabilidad.png'}")
plt.close()

# =============================================================================
# FIGURA 2: PASO A PASO DE PROCRUSTES
# =============================================================================
print("\n2. Generando: Proceso Procrustes paso a paso...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Seleccionar formas para demostrar
demo_idx = idx_covid
demo_shape = all_landmarks[demo_idx].copy()
ref_idx = idx_normal
ref_shape = all_landmarks[ref_idx].copy()

# Fila 1: Transformaciones
# Panel 1.1: Original
ax = axes[0, 0]
ax.set_title('1. Forma Original', fontsize=11, fontweight='bold')
plot_shape_with_connections(ax, demo_shape, color='red', linewidth=1.5, point_size=40)
centroid = demo_shape.mean(axis=0)
ax.scatter(centroid[0], centroid[1], c='black', s=100, marker='x', linewidths=2, label='Centroide')
ax.set_xlim(0, 224)
ax.set_ylim(224, 0)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Panel 1.2: Centrada
ax = axes[0, 1]
ax.set_title('2. Centrada\n(traslación eliminada)', fontsize=11, fontweight='bold')
centered, centroid_orig = center_shape(demo_shape)
plot_shape_with_connections(ax, centered, color='red', linewidth=1.5, point_size=40)
ax.scatter(0, 0, c='black', s=100, marker='x', linewidths=2, label='Origen (0,0)')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
# Invertir Y para consistencia visual
ax.invert_yaxis()

# Panel 1.3: Escalada
ax = axes[0, 2]
ax.set_title('3. Escalada\n(norma = 1)', fontsize=11, fontweight='bold')
scaled, scale_orig = scale_shape(centered)
plot_shape_with_connections(ax, scaled, color='red', linewidth=1.5, point_size=40)
ax.scatter(0, 0, c='black', s=100, marker='x', linewidths=2)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.invert_yaxis()

# Panel 1.4: Rotada
ax = axes[0, 3]
ax.set_title('4. Rotada\n(alineada con referencia)', fontsize=11, fontweight='bold')

ref_centered, _ = center_shape(ref_shape)
ref_scaled, _ = scale_shape(ref_centered)
aligned = align_shape(scaled, ref_scaled)

plot_shape_with_connections(ax, aligned, color='red', linewidth=1.5, point_size=40, label='Alineada')
plot_shape_with_connections(ax, ref_scaled, color='green', alpha=0.5, linewidth=1, point_size=30, label='Referencia')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.invert_yaxis()

# Fila 2: Fórmulas
ax = axes[1, 0]
ax.axis('off')
ax.text(0.5, 0.7, 'Centrado:', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
ax.text(0.5, 0.5, r"$\mathbf{x'} = \mathbf{x} - \bar{\mathbf{x}}$", fontsize=14, ha='center', transform=ax.transAxes)
ax.text(0.5, 0.25, f'Centroide:\n({centroid_orig[0]:.1f}, {centroid_orig[1]:.1f})',
       fontsize=10, ha='center', transform=ax.transAxes, color='gray')

ax = axes[1, 1]
ax.axis('off')
ax.text(0.5, 0.7, 'Escalado:', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
ax.text(0.5, 0.5, r"$\mathbf{x''} = \frac{\mathbf{x'}}{||\mathbf{x'}||_F}$", fontsize=14, ha='center', transform=ax.transAxes)
ax.text(0.5, 0.25, f'Escala original:\n{scale_orig:.1f}',
       fontsize=10, ha='center', transform=ax.transAxes, color='gray')

ax = axes[1, 2]
ax.axis('off')
ax.text(0.5, 0.7, 'Rotación Óptima:', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
ax.text(0.5, 0.5, r"$\mathbf{R} = \mathbf{V}\mathbf{U}^T$", fontsize=14, ha='center', transform=ax.transAxes)
ax.text(0.5, 0.3, r"donde $\mathbf{H} = \mathbf{X}^T\mathbf{Y} = \mathbf{U}\mathbf{S}\mathbf{V}^T$",
       fontsize=11, ha='center', transform=ax.transAxes)

ax = axes[1, 3]
ax.axis('off')
ax.text(0.5, 0.7, 'Distancia Procrustes:', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
dist = np.linalg.norm(aligned - ref_scaled, 'fro')
ax.text(0.5, 0.5, f'd = {dist:.4f}', fontsize=14, ha='center', transform=ax.transAxes, color='blue')
ax.text(0.5, 0.25, '(norma Frobenius del\nresiduo después de alinear)',
       fontsize=10, ha='center', transform=ax.transAxes, color='gray')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_procrustes_paso_a_paso.png", dpi=150, bbox_inches='tight')
print(f"  Guardado: {OUTPUT_DIR / '02_procrustes_paso_a_paso.png'}")
plt.close()

# =============================================================================
# FIGURA 3: ALGORITMO GPA ITERATIVO
# =============================================================================
print("\n3. Generando: Algoritmo GPA iterativo...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

n_vis = 30
vis_indices = np.random.choice(len(all_landmarks), n_vis, replace=False)

# Panel 1: Formas normalizadas sin rotar
ax = axes[0, 0]
ax.set_title('Iteración 0:\nFormas normalizadas (sin rotar)', fontsize=11, fontweight='bold')

normalized_shapes = []
for idx in vis_indices:
    centered, _ = center_shape(all_landmarks[idx])
    scaled, _ = scale_shape(centered)
    normalized_shapes.append(scaled)
    plot_shape_with_connections(ax, scaled, color='blue', alpha=0.2, linewidth=0.5, show_points=False)

normalized_shapes = np.array(normalized_shapes)
mean_shape = normalized_shapes.mean(axis=0)
plot_shape_with_connections(ax, mean_shape, color='red', linewidth=2, point_size=30, label='Ref. inicial')
ax.legend(fontsize=8)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.invert_yaxis()

# Simular iteraciones
reference = mean_shape.copy()
ref_scaled, _ = scale_shape(reference)

for iter_num in range(1, 3):
    ax = axes[0, iter_num]
    ax.set_title(f'Iteración {iter_num}:\nFormas alineadas', fontsize=11, fontweight='bold')

    aligned_iter = []
    for shape in normalized_shapes:
        aligned = align_shape(shape, ref_scaled)
        aligned_iter.append(aligned)
        plot_shape_with_connections(ax, aligned, color='green', alpha=0.2, linewidth=0.5, show_points=False)

    aligned_iter = np.array(aligned_iter)
    new_ref = aligned_iter.mean(axis=0)
    ref_scaled, _ = scale_shape(new_ref)

    plot_shape_with_connections(ax, ref_scaled, color='red', linewidth=2, point_size=30, label='Nueva ref.')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

# Panel: Convergencia
ax = axes[1, 0]
ax.set_title('Convergencia del GPA', fontsize=11, fontweight='bold')

reference = normalized_shapes.mean(axis=0)
ref_scaled, _ = scale_shape(reference)
distances = []

for iteration in range(10):
    aligned_all = []
    for shape in normalized_shapes:
        aligned = align_shape(shape, ref_scaled)
        aligned_all.append(aligned)
    aligned_all = np.array(aligned_all)

    new_ref = aligned_all.mean(axis=0)
    new_ref_scaled, _ = scale_shape(new_ref)

    mean_dist = np.mean([np.linalg.norm(aligned_all[i] - new_ref_scaled, 'fro')
                         for i in range(len(aligned_all))])
    distances.append(mean_dist)
    ref_scaled = new_ref_scaled

ax.plot(range(len(distances)), distances, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Iteración', fontsize=11)
ax.set_ylabel('Distancia promedio', fontsize=11)
ax.grid(True, alpha=0.3)

# Panel: Forma canónica final
ax = axes[1, 1]
ax.set_title('Forma Canónica Final', fontsize=11, fontweight='bold')

plot_shape_with_connections(ax, canonical_shape, color='red', linewidth=2, point_size=60)
for i, (x, y) in enumerate(canonical_shape):
    ax.annotate(f'{i+1}', (x, y), xytext=(3, 3), textcoords='offset points', fontsize=8)

ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.invert_yaxis()

# Panel: Todas las formas alineadas
ax = axes[1, 2]
ax.set_title('957 Formas Alineadas', fontsize=11, fontweight='bold')

for i in range(0, len(aligned_shapes), 10):
    plot_shape_with_connections(ax, aligned_shapes[i], color='gray', alpha=0.15,
                                 linewidth=0.3, show_points=False)

plot_shape_with_connections(ax, canonical_shape, color='red', linewidth=2, point_size=40, label='Canónica')
ax.legend(fontsize=8)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_gpa_iterativo.png", dpi=150, bbox_inches='tight')
print(f"  Guardado: {OUTPUT_DIR / '03_gpa_iterativo.png'}")
plt.close()

# =============================================================================
# FIGURA 4: EFECTO DEL GPA - ANTES VS DESPUES
# =============================================================================
print("\n4. Generando: Efecto del GPA (antes vs después)...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: Antes
ax = axes[0]
ax.set_title('ANTES de GPA\n(centradas y escaladas, sin alinear rotación)', fontsize=12, fontweight='bold')

for i in range(0, len(all_landmarks), 5):
    centered, _ = center_shape(all_landmarks[i])
    scaled, _ = scale_shape(centered)
    plot_shape_with_connections(ax, scaled, color='blue', alpha=0.1, linewidth=0.3, show_points=False)

ax.set_xlabel('X (normalizado)', fontsize=11)
ax.set_ylabel('Y (normalizado)', fontsize=11)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.invert_yaxis()

all_normalized = []
for i in range(len(all_landmarks)):
    centered, _ = center_shape(all_landmarks[i])
    scaled, _ = scale_shape(centered)
    all_normalized.append(scaled)
all_normalized = np.array(all_normalized)
std_before = all_normalized.std()

ax.text(0.02, 0.02, f'Dispersión (std): {std_before:.4f}', transform=ax.transAxes,
       fontsize=10, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat'))

# Panel 2: Después
ax = axes[1]
ax.set_title('DESPUÉS de GPA\n(alineadas por rotación)', fontsize=12, fontweight='bold')

for i in range(0, len(aligned_shapes), 5):
    plot_shape_with_connections(ax, aligned_shapes[i], color='green', alpha=0.1,
                                 linewidth=0.3, show_points=False)

plot_shape_with_connections(ax, canonical_shape, color='red', linewidth=2, point_size=50, label='Canónica')

ax.set_xlabel('X (normalizado)', fontsize=11)
ax.set_ylabel('Y (normalizado)', fontsize=11)
ax.legend(loc='upper right', fontsize=10)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.invert_yaxis()

std_after = aligned_shapes.std()
reduction = (1 - std_after/std_before) * 100

ax.text(0.02, 0.02, f'Dispersión (std): {std_after:.4f}\nReducción: {reduction:.1f}%',
       transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
       bbox=dict(boxstyle='round', facecolor='lightgreen'))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_efecto_gpa_antes_despues.png", dpi=150, bbox_inches='tight')
print(f"  Guardado: {OUTPUT_DIR / '04_efecto_gpa_antes_despues.png'}")
plt.close()

# =============================================================================
# FIGURA 5: FORMA CANONICA EN ESCALA DE IMAGEN
# =============================================================================
print("\n5. Generando: Forma canónica en escala de imagen...")

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Panel 1: Forma canónica con contornos
ax = axes[0]
ax.set_title('Forma Canónica GPA\n(escala 224x224 píxeles)', fontsize=12, fontweight='bold')

# Contorno izquierdo
ax.plot(canonical_pixels[CONTORNO_IZQUIERDO, 0], canonical_pixels[CONTORNO_IZQUIERDO, 1],
        'b-', linewidth=2, label='Contorno izquierdo')
# Contorno derecho
ax.plot(canonical_pixels[CONTORNO_DERECHO, 0], canonical_pixels[CONTORNO_DERECHO, 1],
        'g-', linewidth=2, label='Contorno derecho')
# Eje central
ax.plot(canonical_pixels[EJE_CENTRAL, 0], canonical_pixels[EJE_CENTRAL, 1],
        'r--', linewidth=2, label='Eje central')

# Landmarks
ax.scatter(canonical_pixels[:, 0], canonical_pixels[:, 1], c='red', s=100, zorder=5,
          edgecolors='white', linewidths=2)

for i, (x, y) in enumerate(canonical_pixels):
    ax.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
               fontsize=9, fontweight='bold')

ax.set_xlim(0, 224)
ax.set_ylim(224, 0)
ax.set_xlabel('X (píxeles)', fontsize=11)
ax.set_ylabel('Y (píxeles)', fontsize=11)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Panel 2: Tabla de coordenadas
ax = axes[1]
ax.axis('off')
ax.set_title('Coordenadas de la Forma Canónica', fontsize=12, fontweight='bold')

col_labels = ['Landmark', 'X (px)', 'Y (px)', 'Descripción']
cell_text = []
for i, (x, y) in enumerate(canonical_pixels):
    desc = LANDMARK_NAMES[i].split('(')[1].replace(')', '') if '(' in LANDMARK_NAMES[i] else ''
    cell_text.append([f'L{i+1}', f'{x:.1f}', f'{y:.1f}', desc])

table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

for i in range(len(col_labels)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(color='white', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_forma_canonica_imagen.png", dpi=150, bbox_inches='tight')
print(f"  Guardado: {OUTPUT_DIR / '05_forma_canonica_imagen.png'}")
plt.close()

# =============================================================================
# FIGURA 6: DIAGRAMA DE FLUJO (sin cambios)
# =============================================================================
print("\n6. Generando: Diagrama de flujo del algoritmo...")

fig, ax = plt.subplots(figsize=(12, 14))
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)

box_color = '#E6F3FF'

def draw_box(ax, x, y, width, height, text, color=box_color):
    rect = plt.Rectangle((x - width/2, y - height/2), width, height,
                         facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=10, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle='->', color='#333333', lw=2))

ax.text(5, 13.5, 'Algoritmo GPA (Generalized Procrustes Analysis)',
       ha='center', va='center', fontsize=14, fontweight='bold')

y_pos = 12.5
draw_box(ax, 5, y_pos, 6, 0.8, 'ENTRADA: 957 formas (15 landmarks cada una)', '#FFE6E6')

y_pos -= 1.2
draw_arrow(ax, 5, y_pos + 0.8, 5, y_pos + 0.4)
draw_box(ax, 5, y_pos, 6, 0.8, '1. Centrar cada forma: x\' = x - centroide')

y_pos -= 1.2
draw_arrow(ax, 5, y_pos + 0.8, 5, y_pos + 0.4)
draw_box(ax, 5, y_pos, 6, 0.8, '2. Escalar cada forma: x\'\' = x\' / ||x\'||')

y_pos -= 1.2
draw_arrow(ax, 5, y_pos + 0.8, 5, y_pos + 0.4)
draw_box(ax, 5, y_pos, 6, 0.8, '3. Referencia inicial = promedio de formas')

y_pos -= 1.5
draw_arrow(ax, 5, y_pos + 1.1, 5, y_pos + 0.6)
draw_box(ax, 5, y_pos, 7, 1.2, 'ITERACIÓN:\n4. Rotar cada forma para alinear con referencia\n   (usando SVD: R = V * U^T)', '#E6FFE6')

y_pos -= 1.4
draw_arrow(ax, 5, y_pos + 1.0, 5, y_pos + 0.5)
draw_box(ax, 5, y_pos, 6, 0.8, '5. Nueva referencia = promedio de formas alineadas')

y_pos -= 1.2
draw_arrow(ax, 5, y_pos + 0.8, 5, y_pos + 0.4)
draw_box(ax, 5, y_pos, 6, 0.8, '6. Normalizar referencia (escala = 1)')

y_pos -= 1.4
draw_arrow(ax, 5, y_pos + 1.0, 5, y_pos + 0.5)
draw_box(ax, 5, y_pos, 5, 1.0, '¿Cambio < tolerancia?', '#FFFFD0')

ax.annotate('', xy=(1.5, y_pos), xytext=(2.5, y_pos),
           arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.text(2, y_pos + 0.3, 'NO', fontsize=10, color='red', ha='center')
ax.annotate('', xy=(1.5, 7.2), xytext=(1.5, y_pos),
           arrowprops=dict(arrowstyle='->', color='red', lw=2, connectionstyle='arc3,rad=0'))

y_pos -= 1.4
draw_arrow(ax, 5, y_pos + 1.0, 5, y_pos + 0.5)
ax.text(5.5, y_pos + 0.8, 'SÍ', fontsize=10, color='green')
draw_box(ax, 5, y_pos, 6, 0.8, 'SALIDA: Forma canónica + formas alineadas', '#FFE6E6')

ax.text(5, 0.5, 'Convergencia típica: 2-3 iteraciones\nTolerancia: 10⁻⁴',
       ha='center', va='center', fontsize=9, style='italic', color='gray')

plt.savefig(OUTPUT_DIR / "06_diagrama_flujo_gpa.png", dpi=150, bbox_inches='tight')
print(f"  Guardado: {OUTPUT_DIR / '06_diagrama_flujo_gpa.png'}")
plt.close()

# =============================================================================
# RESUMEN
# =============================================================================
print("\n" + "=" * 60)
print("VISUALIZACIONES CORREGIDAS GENERADAS")
print("=" * 60)
print(f"\nDirectorio: {OUTPUT_DIR}")
print("\nConexiones anatómicas usadas:")
print(f"  Eje central: {[i+1 for i in EJE_CENTRAL]}")
print(f"  Contorno izquierdo: {[i+1 for i in CONTORNO_IZQUIERDO]}")
print(f"  Contorno derecho: {[i+1 for i in CONTORNO_DERECHO]}")
