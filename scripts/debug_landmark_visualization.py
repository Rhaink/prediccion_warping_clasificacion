#!/usr/bin/env python3
"""
Debug: Visualizar landmarks numerados para identificar conexiones correctas.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shape_analysis" / "figures" / "debug"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cargar datos
data = np.load(PROJECT_ROOT / "outputs" / "predictions" / "all_landmarks.npz", allow_pickle=True)
all_landmarks = data['all_landmarks']
all_image_names = data['all_image_names']
all_categories = data['all_categories']

# Cargar forma canonica
with open(PROJECT_ROOT / "outputs" / "shape_analysis" / "canonical_shape_gpa.json") as f:
    canonical_data = json.load(f)
canonical_pixels = np.array(canonical_data['canonical_shape_pixels'])

LANDMARK_NAMES = [
    "L1: Superior (mediastino)",
    "L2: Inferior (vertebra)",
    "L3: Apex Izquierdo",
    "L4: Apex Derecho",
    "L5: Hilio Izquierdo",
    "L6: Hilio Derecho",
    "L7: Base Izquierda",
    "L8: Base Derecha",
    "L9: Centro Superior",
    "L10: Centro Medio",
    "L11: Centro Inferior",
    "L12: Borde Sup Izquierdo",
    "L13: Borde Sup Derecho",
    "L14: Costofrenico Izquierdo",
    "L15: Costofrenico Derecho"
]

# =============================================================================
# FIGURA 1: Forma canonica con landmarks numerados (grande y claro)
# =============================================================================
print("Generando visualizacion de landmarks numerados...")

fig, ax = plt.subplots(figsize=(14, 14))

# Plotear landmarks con numeros grandes
colors = plt.cm.tab20(np.linspace(0, 1, 15))

for i, (x, y) in enumerate(canonical_pixels):
    ax.scatter(x, y, c=[colors[i]], s=300, zorder=5, edgecolors='black', linewidths=2)
    ax.annotate(f'{i+1}', (x, y), fontsize=14, fontweight='bold',
                ha='center', va='center', color='white')

# Ejes y grid
ax.set_xlim(0, 224)
ax.set_ylim(0, 224)  # NO invertido para ver orientacion real
ax.set_xlabel('X (pixeles)', fontsize=12)
ax.set_ylabel('Y (pixeles)', fontsize=12)
ax.set_title('Forma Canonica - Landmarks Numerados\n(Y NO invertido - origen abajo-izquierda)', fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Leyenda con nombres
legend_text = '\n'.join([f'{i+1}: {LANDMARK_NAMES[i]}' for i in range(15)])
ax.text(1.02, 0.98, legend_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "landmarks_numerados_y_normal.png", dpi=150, bbox_inches='tight')
print(f"  Guardado: {OUTPUT_DIR / 'landmarks_numerados_y_normal.png'}")
plt.close()

# =============================================================================
# FIGURA 2: Con Y invertido (como imagen)
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 14))

for i, (x, y) in enumerate(canonical_pixels):
    ax.scatter(x, y, c=[colors[i]], s=300, zorder=5, edgecolors='black', linewidths=2)
    ax.annotate(f'{i+1}', (x, y), fontsize=14, fontweight='bold',
                ha='center', va='center', color='white')

ax.set_xlim(0, 224)
ax.set_ylim(224, 0)  # INVERTIDO como coordenadas de imagen
ax.set_xlabel('X (pixeles)', fontsize=12)
ax.set_ylabel('Y (pixeles)', fontsize=12)
ax.set_title('Forma Canonica - Landmarks Numerados\n(Y INVERTIDO - como coordenadas de imagen)', fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

legend_text = '\n'.join([f'{i+1}: {LANDMARK_NAMES[i]}' for i in range(15)])
ax.text(1.02, 0.98, legend_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "landmarks_numerados_y_invertido.png", dpi=150, bbox_inches='tight')
print(f"  Guardado: {OUTPUT_DIR / 'landmarks_numerados_y_invertido.png'}")
plt.close()

# =============================================================================
# FIGURA 3: Sobre imagen real
# =============================================================================
# Buscar una imagen Normal
idx_normal = np.where(all_categories == 'Normal')[0][0]
img_name = all_image_names[idx_normal]
img_path = PROJECT_ROOT / "data" / "dataset" / "Normal" / f"{img_name}.png"

if img_path.exists():
    print(f"\nCargando imagen: {img_path}")
    img = Image.open(img_path).convert('L')
    img_224 = img.resize((224, 224))
    img_arr = np.array(img_224)

    fig, ax = plt.subplots(figsize=(14, 14))

    ax.imshow(img_arr, cmap='gray')

    # Landmarks de esta imagen
    landmarks = all_landmarks[idx_normal]

    for i, (x, y) in enumerate(landmarks):
        ax.scatter(x, y, c=[colors[i]], s=200, zorder=5, edgecolors='white', linewidths=2)
        ax.annotate(f'{i+1}', (x, y), fontsize=12, fontweight='bold',
                    ha='center', va='center', color='white',
                    bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.8))

    ax.set_title(f'Landmarks sobre imagen real: {img_name}\n(para verificar orientacion)', fontsize=14)
    ax.axis('off')

    # Leyenda
    legend_text = '\n'.join([f'{i+1}: {LANDMARK_NAMES[i].split(":")[1].strip()}' for i in range(15)])
    ax.text(1.02, 0.98, legend_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "landmarks_sobre_imagen_real.png", dpi=150, bbox_inches='tight')
    print(f"  Guardado: {OUTPUT_DIR / 'landmarks_sobre_imagen_real.png'}")
    plt.close()

# =============================================================================
# FIGURA 4: Tabla de coordenadas
# =============================================================================
print("\nCoordenadas de forma canonica:")
print("-" * 60)
print(f"{'Landmark':<5} {'Nombre':<30} {'X':>8} {'Y':>8}")
print("-" * 60)
for i, (x, y) in enumerate(canonical_pixels):
    name = LANDMARK_NAMES[i].split(":")[1].strip() if ":" in LANDMARK_NAMES[i] else LANDMARK_NAMES[i]
    print(f"L{i+1:<4} {name:<30} {x:>8.1f} {y:>8.1f}")
print("-" * 60)

print("\n" + "=" * 60)
print("INSTRUCCIONES PARA DEFINIR CONEXIONES")
print("=" * 60)
print("""
Por favor revisa las imagenes generadas:
1. landmarks_numerados_y_invertido.png - Vista como imagen (Y invertido)
2. landmarks_sobre_imagen_real.png - Sobre radiografia real

Luego indica las conexiones anatomicas correctas, por ejemplo:
- Contorno pulmon izquierdo: [3, 12, 5, 7, 14, ...]
- Contorno pulmon derecho: [4, 13, 6, 8, 15, ...]
- Eje central: [1, 9, 10, 11, 2]

Las conexiones actuales son incorrectas (conectan todos en secuencia).
""")
