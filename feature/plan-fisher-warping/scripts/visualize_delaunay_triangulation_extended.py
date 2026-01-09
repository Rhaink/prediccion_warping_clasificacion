"""
Script: visualize_delaunay_triangulation_extended.py
Propósito: Visualizar triangulación de Delaunay extendida (15 landmarks + 8 puntos de borde)
Input:
    - Imagen original (data/dataset/COVID/COVID-269.png)
    - Landmarks predichos (outputs/predictions/all_landmarks.npz)
Output:
    - results/figures/warping_explained/delaunay_triangulation_extended.png

Nota:
    Esta triangulación extendida es solo para referencia visual y
    NO se usa en el pipeline de clasificación.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import Delaunay
import sys
from typing import Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
FEATURE_DIR = PROJECT_ROOT / "feature" / "plan-fisher-warping"
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.piecewise_affine_warp import add_boundary_points


# Input paths
DATA_DIR = PROJECT_ROOT / "data" / "dataset"
LANDMARKS_FILE = PROJECT_ROOT / "outputs" / "predictions" / "all_landmarks.npz"

# Output paths
OUTPUT_DIR = FEATURE_DIR / "results" / "figures" / "warping_explained"
OUTPUT_FILE = OUTPUT_DIR / "delaunay_triangulation_extended.png"

# Imagen de ejemplo
EXAMPLE_IMAGE = "COVID-269"
EXAMPLE_CATEGORY = "COVID"

# Configuración visual
IMAGE_SIZE = 224
TRIANGLE_THICKNESS = 2
LANDMARK_RADIUS = 3
BOUNDARY_RADIUS = 3
LANDMARK_BORDER_THICKNESS = 1
BOUNDARY_BORDER_THICKNESS = 1


def load_image(category: str, image_name: str, target_size: int = 224) -> np.ndarray:
    image_path = DATA_DIR / category / f"{image_name}.png"
    if not image_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(image, (target_size, target_size))


def load_landmarks(image_name: str, landmarks_file: Path) -> np.ndarray:
    if not landmarks_file.exists():
        raise FileNotFoundError(f"No se encontró archivo de landmarks: {landmarks_file}")
    data = np.load(str(landmarks_file), allow_pickle=True)
    all_image_names = data["all_image_names"]
    idx = np.where(all_image_names == image_name)[0]
    if len(idx) == 0:
        raise ValueError(f"Imagen '{image_name}' no encontrada en landmarks")
    return data["all_landmarks"][idx[0]]


def draw_extended_triangulation(
    image: np.ndarray,
    landmarks: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()

    extended = add_boundary_points(landmarks, image_size=IMAGE_SIZE)
    tri = Delaunay(extended)

    extended_rounded = np.round(extended).astype(np.int32)
    landmarks_rounded = np.round(landmarks).astype(np.int32)

    triangle_color = (0, 255, 255)  # Amarillo
    landmark_color = (0, 0, 255)    # Rojo
    boundary_color = (255, 0, 0)    # Azul

    for tri_indices in tri.simplices:
        pts = extended_rounded[tri_indices]
        cv2.polylines(
            image_rgb,
            [pts],
            isClosed=True,
            color=triangle_color,
            thickness=TRIANGLE_THICKNESS,
            lineType=cv2.LINE_AA
        )

    for x, y in landmarks_rounded:
        cv2.circle(image_rgb, (int(x), int(y)), radius=LANDMARK_RADIUS,
                   color=landmark_color, thickness=-1)
        cv2.circle(image_rgb, (int(x), int(y)), radius=LANDMARK_RADIUS,
                   color=(255, 255, 255), thickness=LANDMARK_BORDER_THICKNESS)

    for x, y in extended_rounded[len(landmarks_rounded):]:
        cv2.circle(image_rgb, (int(x), int(y)), radius=BOUNDARY_RADIUS,
                   color=boundary_color, thickness=-1)
        cv2.circle(image_rgb, (int(x), int(y)), radius=BOUNDARY_RADIUS,
                   color=(255, 255, 255), thickness=BOUNDARY_BORDER_THICKNESS)

    return image_rgb, tri.simplices


def create_visualization(
    image_original: np.ndarray,
    image_with_triangulation: np.ndarray,
    n_triangles: int,
    output_path: Path
):
    if len(image_original.shape) == 2:
        image_original_rgb = cv2.cvtColor(image_original, cv2.COLOR_GRAY2RGB)
    else:
        image_original_rgb = image_original

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image_original_rgb)
    axes[0].set_title("(a) Imagen Original", fontsize=11, pad=10)
    axes[0].axis("off")

    image_with_triangulation_rgb = cv2.cvtColor(image_with_triangulation, cv2.COLOR_BGR2RGB)
    axes[1].imshow(image_with_triangulation_rgb)
    axes[1].set_title(
        f"(b) Delaunay extendida\n({n_triangles} triángulos, 23 puntos)",
        fontsize=11,
        pad=10
    )
    axes[1].axis("off")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markersize=6, label="Landmarks (15)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="blue",
               markersize=6, label="Puntos de borde (8)"),
        Line2D([0], [0], color="yellow", linewidth=TRIANGLE_THICKNESS,
               label=f"Triángulos ({n_triangles})")
    ]
    axes[1].legend(handles=legend_elements, loc="upper right",
                   fontsize=9, framealpha=0.95)

    plt.tight_layout(pad=1.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"✓ Visualización guardada en: {output_path}")
    plt.close(fig)


def main():
    print("=" * 80)
    print("VISUALIZACIÓN DE TRIANGULACIÓN DELAUNAY EXTENDIDA")
    print("=" * 80)

    print(f"\n1. Cargando imagen: {EXAMPLE_IMAGE}")
    image_original = load_image(EXAMPLE_CATEGORY, EXAMPLE_IMAGE, target_size=IMAGE_SIZE)
    print(f"   ✓ Imagen cargada: {image_original.shape}")

    print(f"\n2. Cargando landmarks de: {LANDMARKS_FILE.name}")
    landmarks = load_landmarks(EXAMPLE_IMAGE, LANDMARKS_FILE)
    print(f"   ✓ Landmarks cargados: {landmarks.shape}")

    print(f"\n3. Dibujando triangulación extendida")
    image_with_triangulation, triangles = draw_extended_triangulation(
        image_original,
        landmarks
    )
    print(f"   ✓ Triángulos: {len(triangles)}")

    print(f"\n4. Generando visualización comparativa")
    create_visualization(
        image_original,
        image_with_triangulation,
        len(triangles),
        OUTPUT_FILE
    )

    print("\n" + "=" * 80)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 80)


if __name__ == "__main__":
    main()
