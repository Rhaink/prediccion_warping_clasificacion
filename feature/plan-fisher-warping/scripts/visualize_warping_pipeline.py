"""
Script: visualize_warping_pipeline.py
Propósito: Visualizar el pipeline completo de warping paso a paso
Input:
    - Imagen original (data/dataset/COVID/COVID-269.png)
    - Landmarks predichos (outputs/predictions/all_landmarks.npz)
    - Triangulación pre-calculada (outputs/shape_analysis/canonical_delaunay_triangles.json)
    - Imagen warped (outputs/full_warped_dataset/val/COVID/COVID-269_warped.png)
Output:
    - results/figures/warping_explained/warping_step_by_step.png

Descripción:
    Muestra el proceso completo de warping en 4 pasos:
    1. Imagen original
    2. Original + Landmarks anatómicos
    3. Original + Triangulación de Delaunay
    4. Resultado: Imagen warped

    Esto explica visualmente cómo funciona el proceso de normalización geométrica.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Rutas base
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
FEATURE_DIR = PROJECT_ROOT / "feature" / "plan-fisher-warping"

# Input paths
DATA_DIR = PROJECT_ROOT / "data" / "dataset"
LANDMARKS_FILE = PROJECT_ROOT / "outputs" / "predictions" / "all_landmarks.npz"
TRIANGLES_FILE = PROJECT_ROOT / "outputs" / "shape_analysis" / "canonical_delaunay_triangles.json"
WARPED_DIR = PROJECT_ROOT / "outputs" / "full_warped_dataset"

# Output paths
OUTPUT_DIR = FEATURE_DIR / "results" / "figures" / "warping_explained"
OUTPUT_FILE = OUTPUT_DIR / "warping_step_by_step.png"

# Imagen de ejemplo
EXAMPLE_IMAGE = "COVID-269"
EXAMPLE_CATEGORY = "COVID"
EXAMPLE_SPLIT = "val"  # La imagen está en el split de validación

# Configuración visual
IMAGE_SIZE = 224


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def load_image(category: str, image_name: str, target_size: int = 224) -> np.ndarray:
    """Carga una imagen del dataset y la redimensiona."""
    image_path = DATA_DIR / category / f"{image_name}.png"

    if not image_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (target_size, target_size))

    return image_resized


def load_warped_image(split: str, category: str, image_name: str) -> np.ndarray:
    """Carga la imagen warped correspondiente."""
    warped_path = WARPED_DIR / split / category / f"{image_name}_warped.png"

    if not warped_path.exists():
        raise FileNotFoundError(f"No se encontró imagen warped: {warped_path}")

    warped = cv2.imread(str(warped_path), cv2.IMREAD_GRAYSCALE)

    return warped


def load_landmarks(image_name: str, landmarks_file: Path) -> np.ndarray:
    """Carga los landmarks predichos para una imagen específica."""
    if not landmarks_file.exists():
        raise FileNotFoundError(f"No se encontró archivo de landmarks: {landmarks_file}")

    data = np.load(str(landmarks_file), allow_pickle=True)
    all_image_names = data['all_image_names']
    idx = np.where(all_image_names == image_name)[0]

    if len(idx) == 0:
        raise ValueError(f"Imagen '{image_name}' no encontrada en landmarks")

    landmarks = data['all_landmarks'][idx[0]]

    return landmarks


def load_triangulation(triangles_file: Path) -> np.ndarray:
    """Carga la triangulación pre-calculada."""
    if not triangles_file.exists():
        raise FileNotFoundError(f"No se encontró archivo de triangulación: {triangles_file}")

    with open(triangles_file) as f:
        data = json.load(f)

    triangles = np.array(data['triangles'], dtype=np.int32)

    return triangles


def draw_landmarks_on_image(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Dibuja landmarks sobre una imagen.

    Args:
        image: Imagen en escala de grises
        landmarks: Landmarks (15, 2)

    Returns:
        Imagen RGB con landmarks dibujados
    """
    # Convertir a RGB
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()

    # Color estándar para landmarks (BGR)
    landmark_color = (0, 0, 255)  # Rojo

    # Dibujar cada landmark
    for x, y in landmarks:
        x_int, y_int = int(round(x)), int(round(y))

        # Círculo rojo
        cv2.circle(image_rgb, (x_int, y_int), radius=4, color=landmark_color, thickness=-1)

        # Borde blanco para contraste
        cv2.circle(image_rgb, (x_int, y_int), radius=4, color=(255, 255, 255), thickness=1)

    return image_rgb


def draw_triangulation_on_image(
    image: np.ndarray,
    landmarks: np.ndarray,
    triangles: np.ndarray
) -> np.ndarray:
    """
    Dibuja triangulación sobre una imagen.

    Args:
        image: Imagen en escala de grises
        landmarks: Landmarks (15, 2)
        triangles: Índices de triángulos (n_triangles, 3)

    Returns:
        Imagen RGB con triangulación dibujada
    """
    # Convertir a RGB
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()

    # Colores (BGR)
    triangle_color = (0, 255, 255)  # Amarillo
    landmark_color = (0, 0, 255)    # Rojo

    # Dibujar aristas de triángulos
    for tri_indices in triangles:
        pts = landmarks[tri_indices].astype(np.int32)
        cv2.polylines(image_rgb, [pts], isClosed=True,
                     color=triangle_color, thickness=2, lineType=cv2.LINE_AA)

    # Dibujar landmarks
    for x, y in landmarks:
        x_int, y_int = int(round(x)), int(round(y))
        cv2.circle(image_rgb, (x_int, y_int), radius=4, color=landmark_color, thickness=-1)
        cv2.circle(image_rgb, (x_int, y_int), radius=4, color=(255, 255, 255), thickness=1)

    return image_rgb


def create_pipeline_visualization(
    image_original: np.ndarray,
    image_with_landmarks: np.ndarray,
    image_with_triangulation: np.ndarray,
    image_warped: np.ndarray,
    output_path: Path
):
    """
    Crea visualización del pipeline completo en formato 2x2.

    Args:
        image_original: Imagen original limpia
        image_with_landmarks: Imagen con landmarks
        image_with_triangulation: Imagen con triangulación
        image_warped: Imagen warped final
        output_path: Path donde guardar
    """
    # Convertir todas las imágenes a RGB si es necesario
    if len(image_original.shape) == 2:
        img1 = cv2.cvtColor(image_original, cv2.COLOR_GRAY2RGB)
    else:
        img1 = image_original

    # Convertir BGR a RGB para las imágenes con anotaciones
    img2 = cv2.cvtColor(image_with_landmarks, cv2.COLOR_BGR2RGB)
    img3 = cv2.cvtColor(image_with_triangulation, cv2.COLOR_BGR2RGB)

    # Convertir imagen warped a RGB
    if len(image_warped.shape) == 2:
        img4 = cv2.cvtColor(image_warped, cv2.COLOR_GRAY2RGB)
    else:
        img4 = image_warped

    # Crear figura 2x2
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Panel 1: Imagen original
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title('(a) Imagen Original', fontsize=11, pad=10)
    axes[0, 0].axis('off')

    # Panel 2: Con landmarks
    axes[0, 1].imshow(img2)
    axes[0, 1].set_title('(b) Landmarks Detectados', fontsize=11, pad=10)
    axes[0, 1].axis('off')

    # Panel 3: Con triangulación
    axes[1, 0].imshow(img3)
    axes[1, 0].set_title('(c) Triangulación de Delaunay', fontsize=11, pad=10)
    axes[1, 0].axis('off')

    # Panel 4: Resultado warped
    axes[1, 1].imshow(img4)
    axes[1, 1].set_title('(d) Imagen Warped (Normalizada)', fontsize=11, pad=10)
    axes[1, 1].axis('off')

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
    """Función principal que ejecuta el pipeline de visualización."""
    print("=" * 80)
    print("VISUALIZACIÓN DEL PIPELINE DE WARPING PASO A PASO")
    print("=" * 80)

    # 1. Cargar imagen original
    print(f"\n1. Cargando imagen original: {EXAMPLE_IMAGE}")
    image_original = load_image(EXAMPLE_CATEGORY, EXAMPLE_IMAGE, target_size=IMAGE_SIZE)
    print(f"   ✓ Imagen cargada: {image_original.shape}")

    # 2. Cargar landmarks
    print(f"\n2. Cargando landmarks")
    landmarks = load_landmarks(EXAMPLE_IMAGE, LANDMARKS_FILE)
    print(f"   ✓ Landmarks cargados: {landmarks.shape}")

    # 3. Cargar triangulación
    print(f"\n3. Cargando triangulación pre-calculada")
    triangles = load_triangulation(TRIANGLES_FILE)
    print(f"   ✓ Triangulación cargada: {len(triangles)} triángulos")

    # 4. Cargar imagen warped
    print(f"\n4. Cargando imagen warped")
    image_warped = load_warped_image(EXAMPLE_SPLIT, EXAMPLE_CATEGORY, EXAMPLE_IMAGE)
    print(f"   ✓ Imagen warped cargada: {image_warped.shape}")

    # 5. Generar visualizaciones intermedias
    print(f"\n5. Generando visualizaciones de cada paso")

    # Paso 2: Imagen con landmarks
    image_with_landmarks = draw_landmarks_on_image(image_original, landmarks)
    print(f"   ✓ Paso 2: Landmarks dibujados")

    # Paso 3: Imagen con triangulación
    image_with_triangulation = draw_triangulation_on_image(image_original, landmarks, triangles)
    print(f"   ✓ Paso 3: Triangulación dibujada")

    # 6. Crear panel 2x2 del pipeline completo
    print(f"\n6. Creando panel 2x2 del pipeline completo")
    create_pipeline_visualization(
        image_original,
        image_with_landmarks,
        image_with_triangulation,
        image_warped,
        OUTPUT_FILE
    )

    print("\n" + "=" * 80)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 80)
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Tamaño: {OUTPUT_FILE.stat().st_size / 1024:.2f} KB")
    print("\nEl panel muestra el proceso completo:")
    print("  (a) Imagen original de entrada")
    print("  (b) Detección de 15 landmarks anatómicos")
    print("  (c) Triangulación de Delaunay (18 triángulos)")
    print("  (d) Imagen warped a la forma canónica")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    main()
