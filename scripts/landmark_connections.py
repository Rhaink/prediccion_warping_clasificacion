#!/usr/bin/env python3
"""
Definicion de conexiones anatomicas correctas de landmarks.

Conexiones para visualizacion de los 15 landmarks en radiografias de torax:
- Eje central: 1 → 9 → 10 → 11 → 2
- Contorno pulmon izquierdo: 1 → 12 → 3 → 5 → 7 → 14 → 2
- Contorno pulmon derecho: 1 → 13 → 4 → 6 → 8 → 15 → 2
"""

# Conexiones como listas de indices (0-based)
# Para usar: landmarks[conexion] da los puntos a conectar

# Eje central (columna vertebral/mediastino)
EJE_CENTRAL = [0, 8, 9, 10, 1]  # L1, L9, L10, L11, L2

# Contorno pulmon izquierdo (sentido horario desde arriba)
PULMON_IZQUIERDO = [0, 11, 2, 4, 6, 13, 1]  # L1, L12, L3, L5, L7, L14, L2

# Contorno pulmon derecho (sentido antihorario desde arriba)
PULMON_DERECHO = [0, 12, 3, 5, 7, 14, 1]  # L1, L13, L4, L6, L8, L15, L2

# Todas las conexiones para dibujar
TODAS_CONEXIONES = {
    'eje_central': EJE_CENTRAL,
    'pulmon_izquierdo': PULMON_IZQUIERDO,
    'pulmon_derecho': PULMON_DERECHO
}

# Colores para cada tipo de conexion
COLORES_CONEXIONES = {
    'eje_central': 'red',
    'pulmon_izquierdo': 'blue',
    'pulmon_derecho': 'green'
}

def plot_landmark_connections(ax, landmarks, show_eje=True, show_pulmones=True,
                              eje_color='red', pulmon_izq_color='blue',
                              pulmon_der_color='green', linewidth=2, alpha=0.7):
    """
    Dibujar conexiones anatomicas correctas entre landmarks.

    Args:
        ax: Matplotlib axes
        landmarks: Array (15, 2) con coordenadas de landmarks
        show_eje: Si mostrar eje central
        show_pulmones: Si mostrar contornos de pulmones
        eje_color, pulmon_izq_color, pulmon_der_color: Colores
        linewidth: Grosor de lineas
        alpha: Transparencia
    """
    if show_eje:
        eje_points = landmarks[EJE_CENTRAL]
        ax.plot(eje_points[:, 0], eje_points[:, 1],
                color=eje_color, linewidth=linewidth, alpha=alpha,
                label='Eje central', linestyle='-')

    if show_pulmones:
        # Pulmon izquierdo
        izq_points = landmarks[PULMON_IZQUIERDO]
        ax.plot(izq_points[:, 0], izq_points[:, 1],
                color=pulmon_izq_color, linewidth=linewidth, alpha=alpha,
                label='Pulmón izquierdo', linestyle='-')

        # Pulmon derecho
        der_points = landmarks[PULMON_DERECHO]
        ax.plot(der_points[:, 0], der_points[:, 1],
                color=pulmon_der_color, linewidth=linewidth, alpha=alpha,
                label='Pulmón derecho', linestyle='-')


if __name__ == "__main__":
    # Test
    import numpy as np
    print("Conexiones de landmarks:")
    print(f"  Eje central: {[i+1 for i in EJE_CENTRAL]}")
    print(f"  Pulmon izquierdo: {[i+1 for i in PULMON_IZQUIERDO]}")
    print(f"  Pulmon derecho: {[i+1 for i in PULMON_DERECHO]}")
