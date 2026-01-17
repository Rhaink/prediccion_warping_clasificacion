#!/usr/bin/env python3
"""
Generador de figura F5.3 - Forma Estándar Pulmonar (CORREGIDO).

Genera figura científica mostrando la forma estándar pulmonar REAL
obtenida del Análisis Procrustes Generalizado sobre 957 radiografías.

VERSIÓN CORREGIDA: Usa la misma estructura de contornos que el resto del proyecto.
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
    'left': '#1976D2',      # Azul para pulmón izquierdo
    'right': '#2E7D32',     # Verde para pulmón derecho
    'central': '#D32F2F',   # Rojo para eje central
}

# ESTRUCTURA CORRECTA DE CONTORNOS (igual que en update_all_figures.py)
# Índices de los landmarks (0-indexed)
EJE_CENTRAL = [0, 8, 9, 10, 1]           # L1 → L9 → L10 → L11 → L2
CONTORNO_IZQUIERDO = [0, 11, 2, 4, 6, 13, 1]  # L1 → L12 → L3 → L5 → L7 → L14 → L2
CONTORNO_DERECHO = [0, 12, 3, 5, 7, 14, 1]    # L1 → L13 → L4 → L6 → L8 → L15 → L2


def load_canonical_shape():
    """Carga la forma estándar pulmonar REAL desde el JSON del proyecto."""
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

    # Crear figura con UN SOLO PANEL (cuadrado)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # =========================================================================
    # CONTORNOS PULMONARES (estructura correcta del proyecto)
    # =========================================================================

    # Contorno pulmón izquierdo: L1 → L12 → L3 → L5 → L7 → L14 → L2
    ax.plot(landmarks[CONTORNO_IZQUIERDO, 0], landmarks[CONTORNO_IZQUIERDO, 1],
            color=COLORS['left'], linewidth=2.5, alpha=0.8, zorder=2,
            label='Contorno izquierdo')

    # Contorno pulmón derecho: L1 → L13 → L4 → L6 → L8 → L15 → L2
    ax.plot(landmarks[CONTORNO_DERECHO, 0], landmarks[CONTORNO_DERECHO, 1],
            color=COLORS['right'], linewidth=2.5, alpha=0.8, zorder=2,
            label='Contorno derecho')

    # Eje central (línea discontinua): L1 → L9 → L10 → L11 → L2
    ax.plot(landmarks[EJE_CENTRAL, 0], landmarks[EJE_CENTRAL, 1],
            color=COLORS['central'], linewidth=2.0, alpha=0.7, zorder=1,
            linestyle='--', label='Eje central')

    # =========================================================================
    # PUNTOS DE REFERENCIA (LANDMARKS)
    # =========================================================================

    # Dibujar todos los puntos en azul oscuro
    ax.scatter(landmarks[:, 0], landmarks[:, 1],
              c='#0D47A1', s=120, edgecolor='white', linewidth=2,
              zorder=5, alpha=0.95)

    # =========================================================================
    # ETIQUETAS DE TODOS LOS PUNTOS
    # =========================================================================

    # Etiquetar TODOS los landmarks (L1 a L15)
    # Ajustar posición de texto según ubicación del punto
    offsets = {
        0: (8, -10),   # L1 (Superior)
        1: (8, 8),     # L2 (Inferior)
        2: (-18, 0),   # L3 (Apex Izq)
        3: (10, 0),    # L4 (Apex Der)
        4: (-18, 0),   # L5 (Hilio Izq)
        5: (10, 0),    # L6 (Hilio Der)
        6: (-18, 0),   # L7 (Base Izq)
        7: (10, 0),    # L8 (Base Der)
        8: (10, 0),    # L9 (Centro Sup)
        9: (10, 0),    # L10 (Centro Med)
        10: (10, 0),   # L11 (Centro Inf)
        11: (-5, -12), # L12 (Borde Sup Izq)
        12: (5, -12),  # L13 (Borde Sup Der)
        13: (-18, 8),  # L14 (Costofrenico Izq)
        14: (10, 8),   # L15 (Costofrenico Der)
    }

    for idx in range(15):
        x, y = landmarks[idx]
        offset = offsets.get(idx, (8, 0))
        ax.annotate(f'L{idx+1}', (x, y), xytext=offset,
                   textcoords='offset points',
                   fontsize=9, fontweight='bold', color='#212121',
                   bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                            edgecolor='none', alpha=0.85),
                   zorder=6)

    # =========================================================================
    # CONFIGURACIÓN DE EJES Y ESTILO
    # =========================================================================

    ax.set_xlim(0, 224)
    ax.set_ylim(224, 0)  # Invertir Y para que coincida con coordenadas de imagen
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
    print(f"  Estructura de contornos:")
    print(f"    - Izquierdo: L1 → L12 → L3 → L5 → L7 → L14 → L2")
    print(f"    - Derecho: L1 → L13 → L4 → L6 → L8 → L15 → L2")
    print(f"    - Eje central: L1 → L9 → L10 → L11 → L2")
    print(f"\nCaracterísticas:")
    print(f"  ✓ Forma estándar pulmonar REAL del proyecto (957 radiografías)")
    print(f"  ✓ Contornos cerrados formando siluetas pulmonares")
    print(f"  ✓ 15 puntos de referencia etiquetados")
    print(f"  ✓ SIN triangulación (ya está en F5.4)")
    print(f"  ✓ Calidad científica 300 DPI")

    plt.close()


if __name__ == "__main__":
    main()
