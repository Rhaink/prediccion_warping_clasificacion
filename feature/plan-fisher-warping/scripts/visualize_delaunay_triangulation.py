"""
Script: visualize_delaunay_triangulation.py
Propósito: Visualizar triangulación de Delaunay sobre radiografía de tórax
Input:
    - Imagen original (data/dataset/COVID/COVID-269.png)
    - Landmarks predichos (outputs/predictions/all_landmarks.npz)
    - Triangulación pre-calculada (outputs/shape_analysis/canonical_delaunay_triangles.json)
Output:
    - results/figures/warping_explained/delaunay_triangulation_example.png

Descripción:
    Muestra la triangulación de Delaunay calculada sobre la forma canónica
    (18 triángulos usando SOLO los 15 landmarks anatómicos).

IMPORTANTE - Cómo funciona la triangulación en el proyecto:
    1. La triangulación se calculó UNA VEZ sobre la forma canónica (15 landmarks)
    2. Resultado: 18 triángulos pre-calculados (guardados en JSON)
    3. Los MISMOS índices de triángulos se usan para TODAS las imágenes
    4. Solo cambian las coordenadas de los landmarks, NO la topología
    5. NO se usan puntos de borde adicionales

    Este script visualiza esos 18 triángulos aplicados a los landmarks
    de una imagen específica.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import Delaunay


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Rutas base
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
FEATURE_DIR = PROJECT_ROOT / "feature" / "plan-fisher-warping"

# Input paths
DATA_DIR = PROJECT_ROOT / "data" / "dataset"
LANDMARKS_FILE = PROJECT_ROOT / "outputs" / "predictions" / "all_landmarks.npz"

# Output paths
OUTPUT_DIR = FEATURE_DIR / "results" / "figures" / "warping_explained"
OUTPUT_FILE = OUTPUT_DIR / "delaunay_triangulation_example.png"

# Imagen de ejemplo
EXAMPLE_IMAGE = "COVID-269"
EXAMPLE_CATEGORY = "COVID"

# Configuración visual
IMAGE_SIZE = 224


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def load_image(category: str, image_name: str, target_size: int = 224) -> np.ndarray:
    """
    Carga una imagen del dataset y la redimensiona.

    Args:
        category: Categoría de la imagen
        image_name: Nombre de la imagen sin extensión
        target_size: Tamaño objetivo

    Returns:
        Imagen en escala de grises redimensionada
    """
    image_path = DATA_DIR / category / f"{image_name}.png"

    if not image_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (target_size, target_size))

    return image_resized


def load_landmarks(image_name: str, landmarks_file: Path) -> np.ndarray:
    """
    Carga los landmarks predichos para una imagen específica.

    Args:
        image_name: Nombre de la imagen sin extensión
        landmarks_file: Path al archivo .npz con landmarks

    Returns:
        Array de landmarks (15, 2)
    """
    if not landmarks_file.exists():
        raise FileNotFoundError(f"No se encontró archivo de landmarks: {landmarks_file}")

    data = np.load(str(landmarks_file), allow_pickle=True)
    all_image_names = data['all_image_names']
    idx = np.where(all_image_names == image_name)[0]

    if len(idx) == 0:
        raise ValueError(f"Imagen '{image_name}' no encontrada en landmarks")

    landmarks = data['all_landmarks'][idx[0]]

    return landmarks


def load_canonical_triangulation(triangles_file: Path) -> np.ndarray:
    """
    Carga la triangulación pre-calculada sobre la forma canónica.

    Args:
        triangles_file: Path al archivo JSON con triangulación

    Returns:
        Array de índices de triángulos (n_triangles, 3)
    """
    import json

    if not triangles_file.exists():
        raise FileNotFoundError(f"No se encontró archivo de triangulación: {triangles_file}")

    with open(triangles_file) as f:
        data = json.load(f)

    triangles = np.array(data['triangles'], dtype=np.int32)

    return triangles


def draw_delaunay_triangulation(
    image: np.ndarray,
    landmarks: np.ndarray,
    triangles: np.ndarray
) -> np.ndarray:
    """
    Dibuja la triangulación de Delaunay sobre la imagen.

    Args:
        image: Imagen en escala de grises
        landmarks: Landmarks anatómicos (15, 2)
        triangles: Índices de triángulos (n_triangles, 3)

    Returns:
        Imagen RGB con triangulación dibujada
    """
    # Convertir a RGB
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()

    # Colores apropiados para publicaciones científicas (BGR para OpenCV)
    triangle_color = (0, 255, 255)    # Amarillo (alto contraste)
    landmark_color = (0, 0, 255)      # Rojo (estándar para landmarks)

    # Dibujar aristas de todos los triángulos
    for tri_indices in triangles:
        pts = landmarks[tri_indices].astype(np.int32)

        # Dibujar el triángulo (solo bordes, grosor 2 para visibilidad)
        cv2.polylines(
            image_rgb,
            [pts],
            isClosed=True,
            color=triangle_color,
            thickness=2,
            lineType=cv2.LINE_AA
        )

    # Dibujar landmarks anatómicos
    for i in range(len(landmarks)):
        x, y = landmarks[i]
        # Círculo rojo más grande para landmarks
        cv2.circle(
            image_rgb,
            (int(x), int(y)),
            radius=5,
            color=landmark_color,
            thickness=-1
        )
        # Borde blanco para mejor contraste
        cv2.circle(
            image_rgb,
            (int(x), int(y)),
            radius=5,
            color=(255, 255, 255),  # Borde blanco
            thickness=1
        )

    return image_rgb


def create_visualization(
    image_original: np.ndarray,
    image_with_triangulation: np.ndarray,
    n_triangles: int,
    output_path: Path
):
    """
    Crea visualización comparativa de la triangulación.

    Args:
        image_original: Imagen original sin modificar
        image_with_triangulation: Imagen con triangulación dibujada
        n_triangles: Número de triángulos en la triangulación
        output_path: Path donde guardar la figura
    """
    # Convertir imagen original a RGB
    if len(image_original.shape) == 2:
        image_original_rgb = cv2.cvtColor(image_original, cv2.COLOR_GRAY2RGB)
    else:
        image_original_rgb = image_original

    # Crear figura
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Panel 1: Imagen original
    axes[0].imshow(image_original_rgb)
    axes[0].set_title('(a) Imagen Original', fontsize=11, pad=10)
    axes[0].axis('off')

    # Panel 2: Con triangulación
    # Convertir de BGR (OpenCV) a RGB (matplotlib)
    image_with_triangulation_rgb = cv2.cvtColor(image_with_triangulation, cv2.COLOR_BGR2RGB)
    axes[1].imshow(image_with_triangulation_rgb)
    axes[1].set_title(
        f'(b) Triangulación de Delaunay\n({n_triangles} triángulos)',
        fontsize=11,
        pad=10
    )
    axes[1].axis('off')

    # Leyenda para el panel de triangulación
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=8, label='Landmarks anatómicos (15)'),
        Line2D([0], [0], color='yellow', linewidth=2,
               label=f'Triángulos Delaunay ({n_triangles})')
    ]
    axes[1].legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=9,
        framealpha=0.95
    )

    # Ajustar layout
    plt.tight_layout(pad=1.5)

    # Guardar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        str(output_path),
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    print(f"✓ Visualización guardada en: {output_path}")

    plt.close(fig)


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal que ejecuta el pipeline de visualización.
    """
    print("=" * 80)
    print("VISUALIZACIÓN DE TRIANGULACIÓN DE DELAUNAY")
    print("=" * 80)

    # 1. Cargar imagen original
    print(f"\n1. Cargando imagen: {EXAMPLE_IMAGE}")
    image_original = load_image(EXAMPLE_CATEGORY, EXAMPLE_IMAGE, target_size=IMAGE_SIZE)
    print(f"   ✓ Imagen cargada: {image_original.shape}")

    # 2. Cargar landmarks
    print(f"\n2. Cargando landmarks de: {LANDMARKS_FILE.name}")
    landmarks = load_landmarks(EXAMPLE_IMAGE, LANDMARKS_FILE)
    print(f"   ✓ Landmarks cargados: {landmarks.shape}")

    # 3. Cargar triangulación pre-calculada
    print(f"\n3. Cargando triangulación pre-calculada")
    triangles_file = PROJECT_ROOT / "outputs" / "shape_analysis" / "canonical_delaunay_triangles.json"
    triangles = load_canonical_triangulation(triangles_file)
    n_triangles = len(triangles)
    print(f"   ✓ Triangulación cargada: {triangles_file.name}")
    print(f"   ✓ Número de triángulos: {n_triangles}")
    print(f"   ✓ Topología calculada sobre forma canónica (15 landmarks)")

    # 4. Dibujar triangulación sobre imagen
    print(f"\n4. Dibujando triangulación sobre imagen")
    image_with_triangulation = draw_delaunay_triangulation(
        image_original,
        landmarks,
        triangles
    )
    print(f"   ✓ Triangulación dibujada")

    # 5. Crear visualización comparativa
    print(f"\n5. Generando visualización comparativa")
    create_visualization(
        image_original,
        image_with_triangulation,
        n_triangles,
        OUTPUT_FILE
    )

    print("\n" + "=" * 80)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 80)
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Tamaño: {OUTPUT_FILE.stat().st_size / 1024:.2f} KB")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    main()
