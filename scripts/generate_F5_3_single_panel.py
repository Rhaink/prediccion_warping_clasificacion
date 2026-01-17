#!/usr/bin/env python3
"""
Generador de figura F5.3 - Forma Estándar Pulmonar (UN SOLO PANEL).

Genera figura científica de alta calidad mostrando la forma estándar pulmonar
obtenida del Análisis Procrustes Generalizado sobre 957 radiografías anotadas.

NO incluye triangulación (esa está en F5.4).
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuración de estilo científico
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300

# Colores para publicación científica
COLORS = {
    'central': '#D32F2F',   # Rojo para eje central
    'left': '#1976D2',      # Azul para pulmón izquierdo
    'right': '#F57C00',     # Naranja para pulmón derecho
    'contour': '#37474F',   # Gris oscuro para contornos
}


def load_canonical_shape():
    """Carga la forma estándar pulmonar desde el JSON real del proyecto."""
    canonical_path = Path('outputs/shape_analysis/canonical_shape_gpa.json')
    with open(canonical_path) as f:
        data = json.load(f)

    # Validar datos
    assert data['n_landmarks'] == 15, "Esperaba 15 landmarks"
    assert data['method'] == "Generalized Procrustes Analysis (GPA)", "Método incorrecto"
    assert data['convergence']['n_shapes_used'] == 957, "Número de formas incorrecto"

    return np.array(data['canonical_shape_pixels']), data


def main():
    # Cargar forma estándar pulmonar REAL del proyecto
    landmarks, metadata = load_canonical_shape()

    print(f"Cargando forma estándar pulmonar...")
    print(f"  Método: {metadata['method']}")
    print(f"  Formas usadas: {metadata['convergence']['n_shapes_used']}")
    print(f"  Convergencia: {metadata['convergence']['converged']}")
    print(f"  Iteraciones: {metadata['convergence']['n_iterations']}")

    # Crear figura con UN SOLO PANEL
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Índices de puntos por grupo
    # Basado en src_v2/constants.py
    central_idx = [0, 8, 9, 10, 1]  # L1, L9, L10, L11, L2 (eje central vertical)
    left_idx = [11, 2, 4, 6, 13]    # L12, L3, L5, L7, L14 (pulmón izquierdo)
    right_idx = [12, 3, 5, 7, 14]   # L13, L4, L6, L8, L15 (pulmón derecho)

    # =========================================================================
    # CONTORNOS PULMONARES
    # =========================================================================

    # Eje central (L1 → L9 → L10 → L11 → L2)
    ax.plot(landmarks[central_idx, 0], landmarks[central_idx, 1],
            color=COLORS['contour'], linewidth=2.0, alpha=0.5, zorder=1,
            linestyle='-')

    # Contorno pulmón izquierdo: L12 → L3 → L5 → L7 → L14 → L11 (cerrado)
    left_contour = [11, 2, 4, 6, 13, 10]  # Índices: L12, L3, L5, L7, L14, L11
    ax.plot(landmarks[left_contour, 0], landmarks[left_contour, 1],
            color=COLORS['contour'], linewidth=2.0, alpha=0.5, zorder=1,
            linestyle='-')

    # Contorno pulmón derecho: L13 → L4 → L6 → L8 → L15 → L11 (cerrado)
    right_contour = [12, 3, 5, 7, 14, 10]  # Índices: L13, L4, L6, L8, L15, L11
    ax.plot(landmarks[right_contour, 0], landmarks[right_contour, 1],
            color=COLORS['contour'], linewidth=2.0, alpha=0.5, zorder=1,
            linestyle='-')

    # =========================================================================
    # PUNTOS DE REFERENCIA (LANDMARKS)
    # =========================================================================

    # Dibujar puntos coloreados por grupo
    ax.scatter(landmarks[central_idx, 0], landmarks[central_idx, 1],
              c=COLORS['central'], s=120, edgecolor='white', linewidth=2,
              label='Eje central (5 puntos)', zorder=5, alpha=0.95)

    ax.scatter(landmarks[left_idx, 0], landmarks[left_idx, 1],
              c=COLORS['left'], s=120, edgecolor='white', linewidth=2,
              label='Pulmón izquierdo (5 puntos)', zorder=5, alpha=0.95)

    ax.scatter(landmarks[right_idx, 0], landmarks[right_idx, 1],
              c=COLORS['right'], s=120, edgecolor='white', linewidth=2,
              label='Pulmón derecho (5 puntos)', zorder=5, alpha=0.95)

    # =========================================================================
    # ETIQUETAS DE PUNTOS CLAVE
    # =========================================================================

    # Etiquetar puntos clave para orientación
    key_labels = {
        0: ('L1', (10, -8)),    # Superior
        1: ('L2', (10, 8)),     # Inferior
        2: ('L3', (-15, 0)),    # Apex izq
        3: ('L4', (10, 0)),     # Apex der
        8: ('L9', (10, 0)),     # Centro sup
        9: ('L10', (10, 0)),    # Centro med
        10: ('L11', (10, 0)),   # Centro inf
        11: ('L12', (-15, -8)), # Borde sup izq
        12: ('L13', (10, -8)),  # Borde sup der
        13: ('L14', (-15, 8)),  # Costofrenico izq
        14: ('L15', (10, 8)),   # Costofrenico der
    }

    for idx, (label, offset) in key_labels.items():
        x, y = landmarks[idx]
        ax.annotate(label, (x, y), xytext=offset,
                   textcoords='offset points',
                   fontsize=9, fontweight='bold', color='#212121',
                   bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                            edgecolor='none', alpha=0.8),
                   zorder=6)

    # =========================================================================
    # CONFIGURACIÓN DE EJES Y ESTILO
    # =========================================================================

    ax.set_xlim(-5, 229)
    ax.set_ylim(229, -5)  # Invertir Y para que coincida con coordenadas de imagen
    ax.set_aspect('equal')
    ax.set_xlabel('Coordenada X (píxeles)', fontweight='bold')
    ax.set_ylabel('Coordenada Y (píxeles)', fontweight='bold')
    ax.set_title('Forma estándar pulmonar obtenida por Análisis Procrustes Generalizado',
                fontweight='bold', pad=15)

    # Leyenda
    ax.legend(loc='lower left', framealpha=0.95, edgecolor='gray')

    # Grid sutil
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)

    # Agregar nota con información del método
    info_text = (
        f"n = {metadata['convergence']['n_shapes_used']} radiografías\n"
        f"Iteraciones GPA: {metadata['convergence']['n_iterations']}\n"
        f"15 puntos de referencia"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                    edgecolor='gray', alpha=0.95, linewidth=1))

    # Ajustar layout
    plt.tight_layout()

    # Guardar figura
    output_path = Path('docs/Tesis/Figures/F5.3_forma_canonica.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    print(f"\n✓ Figura F5.3 generada exitosamente:")
    print(f"  Archivo: {output_path}")
    print(f"  Resolución: 300 DPI")
    print(f"  Dimensiones: ~2400x2400 píxeles")
    print(f"  Formato: PNG con fondo blanco")
    print(f"\nCaracterísticas:")
    print(f"  ✓ Forma estándar pulmonar REAL del proyecto")
    print(f"  ✓ 15 puntos de referencia coloreados por grupo")
    print(f"  ✓ Contornos pulmonares visibles")
    print(f"  ✓ SIN triangulación (ya está en F5.4)")
    print(f"  ✓ Calidad científica apropiada para publicación")

    plt.close()


if __name__ == "__main__":
    main()
