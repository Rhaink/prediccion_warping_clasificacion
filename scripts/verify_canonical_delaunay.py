#!/usr/bin/env python3
"""
Verificar que los triangulos de Delaunay son validos en la forma canonica GPA.
Generar visualizacion comparativa.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from pathlib import Path
from scipy.spatial import Delaunay

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shape_analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Cargar datos
print("Cargando forma canonica GPA...")
with open(OUTPUT_DIR / "canonical_shape_gpa.json") as f:
    gpa_data = json.load(f)
canonical_pixels = np.array(gpa_data['canonical_shape_pixels'])

print("Cargando triangulos de Delaunay existentes...")
with open(PROJECT_ROOT / "outputs" / "predictions" / "delaunay_triangles.json") as f:
    delaunay_data = json.load(f)
triangles_original = np.array(delaunay_data['triangles'])
mean_landmarks_original = np.array(delaunay_data['mean_landmarks'])

# Calcular nueva triangulacion Delaunay sobre forma canonica
print("\nCalculando triangulacion Delaunay sobre forma canonica...")
tri_canonical = Delaunay(canonical_pixels)
triangles_canonical = tri_canonical.simplices

print(f"\nComparacion de triangulaciones:")
print(f"  Triangulos originales (promedio simple): {len(triangles_original)}")
print(f"  Triangulos canonicos (GPA): {len(triangles_canonical)}")

# Verificar si las triangulaciones son iguales
def normalize_triangles(triangles):
    """Normalizar triangulos para comparacion (ordenar vertices)."""
    normalized = []
    for tri in triangles:
        # Rotar para que el menor indice este primero
        tri = list(tri)
        min_idx = tri.index(min(tri))
        tri = tri[min_idx:] + tri[:min_idx]
        normalized.append(tuple(tri))
    return set(normalized)

tri_set_original = normalize_triangles(triangles_original)
tri_set_canonical = normalize_triangles(triangles_canonical)

common = tri_set_original & tri_set_canonical
only_original = tri_set_original - tri_set_canonical
only_canonical = tri_set_canonical - tri_set_original

print(f"\n  Triangulos comunes: {len(common)}")
print(f"  Solo en original: {len(only_original)}")
print(f"  Solo en canonico: {len(only_canonical)}")

if len(only_original) == 0 and len(only_canonical) == 0:
    print("\n  ✓ Las triangulaciones son IDENTICAS")
else:
    print("\n  ⚠ Las triangulaciones difieren ligeramente")
    if only_original:
        print(f"    Triangulos solo en original: {only_original}")
    if only_canonical:
        print(f"    Triangulos solo en canonico: {only_canonical}")

# Calcular diferencia entre landmarks
diff = canonical_pixels - mean_landmarks_original
diff_norm = np.linalg.norm(diff, axis=1)
print(f"\nDiferencia entre forma canonica GPA y promedio simple:")
print(f"  Distancia promedio: {diff_norm.mean():.2f} px")
print(f"  Distancia maxima: {diff_norm.max():.2f} px (L{diff_norm.argmax()+1})")
print(f"  Distancia minima: {diff_norm.min():.2f} px (L{diff_norm.argmin()+1})")

# Crear visualizacion comparativa
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Subplot 1: Promedio simple con triangulacion original
ax1 = axes[0]
ax1.set_title('Promedio Simple + Delaunay Original', fontsize=12)
ax1.scatter(mean_landmarks_original[:, 0], mean_landmarks_original[:, 1],
            c='blue', s=100, zorder=5)
ax1.triplot(mean_landmarks_original[:, 0], mean_landmarks_original[:, 1],
            triangles_original, 'b-', alpha=0.5)
for i, (x, y) in enumerate(mean_landmarks_original):
    ax1.annotate(f'L{i+1}', (x, y), xytext=(3, 3), textcoords='offset points', fontsize=8)
ax1.set_xlim(0, 224)
ax1.set_ylim(224, 0)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# Subplot 2: Forma canonica GPA con triangulacion
ax2 = axes[1]
ax2.set_title('Forma Canonica GPA + Delaunay', fontsize=12)
ax2.scatter(canonical_pixels[:, 0], canonical_pixels[:, 1],
            c='red', s=100, zorder=5)
ax2.triplot(canonical_pixels[:, 0], canonical_pixels[:, 1],
            triangles_canonical, 'r-', alpha=0.5)
for i, (x, y) in enumerate(canonical_pixels):
    ax2.annotate(f'L{i+1}', (x, y), xytext=(3, 3), textcoords='offset points', fontsize=8)
ax2.set_xlim(0, 224)
ax2.set_ylim(224, 0)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

# Subplot 3: Comparacion superpuesta
ax3 = axes[2]
ax3.set_title('Comparacion: Simple (azul) vs GPA (rojo)', fontsize=12)
ax3.scatter(mean_landmarks_original[:, 0], mean_landmarks_original[:, 1],
            c='blue', s=80, alpha=0.5, label='Promedio Simple')
ax3.scatter(canonical_pixels[:, 0], canonical_pixels[:, 1],
            c='red', s=80, alpha=0.5, label='GPA')
# Conectar puntos correspondientes
for i in range(len(canonical_pixels)):
    ax3.plot([mean_landmarks_original[i, 0], canonical_pixels[i, 0]],
             [mean_landmarks_original[i, 1], canonical_pixels[i, 1]],
             'g-', alpha=0.5, linewidth=1)
    ax3.annotate(f'L{i+1}', canonical_pixels[i], fontsize=8)
ax3.set_xlim(0, 224)
ax3.set_ylim(224, 0)
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right')

plt.tight_layout()
output_path = FIGURES_DIR / "canonical_vs_simple_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nVisualizacion guardada en: {output_path}")
plt.close()

# Guardar triangulacion canonica actualizada
canonical_delaunay = {
    'num_triangles': len(triangles_canonical),
    'triangles': triangles_canonical.tolist(),
    'canonical_landmarks': canonical_pixels.tolist(),
    'method': 'GPA + Delaunay',
    'description': 'Triangulacion Delaunay sobre forma canonica GPA'
}

output_json = OUTPUT_DIR / "canonical_delaunay_triangles.json"
with open(output_json, 'w') as f:
    json.dump(canonical_delaunay, f, indent=2)
print(f"Triangulacion canonica guardada en: {output_json}")

print("\n" + "=" * 50)
print("VERIFICACION COMPLETADA")
print("=" * 50)
