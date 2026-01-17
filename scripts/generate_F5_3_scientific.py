#!/usr/bin/env python3
"""
Generador de figura F5.3 científica de alta calidad.
Muestra la forma canónica y su triangulación de Delaunay.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from scipy.spatial import Delaunay

# Configuración de estilo científico
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300

# Colores sutiles para publicación
COLORS = {
    'central': '#D32F2F',   # Rojo suave
    'left': '#1976D2',      # Azul suave
    'right': '#F57C00',     # Naranja suave
    'line': '#37474F',      # Gris oscuro
}

def load_canonical_shape():
    """Carga la forma canónica desde el JSON."""
    canonical_path = Path('outputs/shape_analysis/canonical_shape_gpa.json')
    with open(canonical_path) as f:
        data = json.load(f)
    return np.array(data['canonical_shape_pixels'])

def main():
    # Cargar forma canónica
    landmarks = load_canonical_shape()

    # Crear figura con dos paneles
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Índices de puntos por grupo
    central_idx = [0, 8, 9, 10, 1]  # L1, L9, L10, L11, L2
    left_idx = [11, 2, 4, 6, 13]    # L12, L3, L5, L7, L14
    right_idx = [12, 3, 5, 7, 14]   # L13, L4, L6, L8, L15

    # =========================================================================
    # PANEL A: Forma canónica con contorno
    # =========================================================================
    ax = axes[0]

    # Dibujar líneas de contorno primero (para que queden detrás)
    # Eje central
    ax.plot(landmarks[central_idx, 0], landmarks[central_idx, 1],
            'k-', linewidth=1.5, alpha=0.4, zorder=1)

    # Contorno izquierdo
    left_contour = [11, 2, 4, 6, 13, 10, 11]  # L12→L3→L5→L7→L14→L11→L12
    ax.plot(landmarks[left_contour, 0], landmarks[left_contour, 1],
            'k-', linewidth=1.5, alpha=0.4, zorder=1)

    # Contorno derecho
    right_contour = [12, 3, 5, 7, 14, 10, 12]  # L13→L4→L6→L8→L15→L11→L13
    ax.plot(landmarks[right_contour, 0], landmarks[right_contour, 1],
            'k-', linewidth=1.5, alpha=0.4, zorder=1)

    # Dibujar puntos
    ax.scatter(landmarks[central_idx, 0], landmarks[central_idx, 1],
              c=COLORS['central'], s=100, edgecolor='white', linewidth=1.5,
              label='Eje central', zorder=5, alpha=0.9)
    ax.scatter(landmarks[left_idx, 0], landmarks[left_idx, 1],
              c=COLORS['left'], s=100, edgecolor='white', linewidth=1.5,
              label='Pulmón izquierdo', zorder=5, alpha=0.9)
    ax.scatter(landmarks[right_idx, 0], landmarks[right_idx, 1],
              c=COLORS['right'], s=100, edgecolor='white', linewidth=1.5,
              label='Pulmón derecho', zorder=5, alpha=0.9)

    # Etiquetas de puntos (solo algunos clave para no saturar)
    key_points = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14]  # L1, L2, L3, L4, L9-L15
    for i in key_points:
        x, y = landmarks[i]
        # Ajustar posición del texto según ubicación del punto
        if i in central_idx:
            xytext = (8, 0)
        elif i in left_idx:
            xytext = (-12, 0)
        else:
            xytext = (8, 0)

        ax.annotate(f'L{i+1}', (x, y), xytext=xytext,
                   textcoords='offset points',
                   fontsize=8, fontweight='bold', color='#212121',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='none', alpha=0.7))

    ax.set_xlim(-5, 229)
    ax.set_ylim(229, -5)
    ax.set_aspect('equal')
    ax.set_xlabel('Coordenada X (píxeles)', fontweight='bold')
    ax.set_ylabel('Coordenada Y (píxeles)', fontweight='bold')
    ax.set_title('(a) Forma canónica de referencia', fontweight='bold', pad=10)
    ax.legend(loc='lower left', framealpha=0.95)
    ax.grid(True, alpha=0.2, linestyle='--')

    # =========================================================================
    # PANEL B: Triangulación de Delaunay
    # =========================================================================
    ax = axes[1]

    # Calcular triangulación
    tri = Delaunay(landmarks)

    # Dibujar triángulos
    for simplex in tri.simplices:
        triangle = landmarks[simplex]
        # Cerrar el triángulo
        triangle = np.vstack([triangle, triangle[0]])
        ax.plot(triangle[:, 0], triangle[:, 1],
               'k-', linewidth=1.2, alpha=0.5, zorder=1)

    # Dibujar puntos (más pequeños)
    ax.scatter(landmarks[central_idx, 0], landmarks[central_idx, 1],
              c=COLORS['central'], s=80, edgecolor='white', linewidth=1.2,
              zorder=5, alpha=0.9)
    ax.scatter(landmarks[left_idx, 0], landmarks[left_idx, 1],
              c=COLORS['left'], s=80, edgecolor='white', linewidth=1.2,
              zorder=5, alpha=0.9)
    ax.scatter(landmarks[right_idx, 0], landmarks[right_idx, 1],
              c=COLORS['right'], s=80, edgecolor='white', linewidth=1.2,
              zorder=5, alpha=0.9)

    ax.set_xlim(-5, 229)
    ax.set_ylim(229, -5)
    ax.set_aspect('equal')
    ax.set_xlabel('Coordenada X (píxeles)', fontweight='bold')
    ax.set_ylabel('Coordenada Y (píxeles)', fontweight='bold')
    ax.set_title('(b) Triangulación de Delaunay', fontweight='bold', pad=10)
    ax.grid(True, alpha=0.2, linestyle='--')

    # Agregar texto con información
    n_triangles = len(tri.simplices)
    info_text = f'16 triángulos\n15 puntos de referencia'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                    edgecolor='gray', alpha=0.9))

    # Ajustar layout
    plt.tight_layout()

    # Guardar figura
    output_path = Path('docs/Tesis/Figures/F5.3_forma_canonica.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Figura generada: {output_path}")
    print(f"  Resolución: 300 DPI")
    print(f"  Triángulos: {n_triangles}")

    plt.close()

if __name__ == "__main__":
    main()
