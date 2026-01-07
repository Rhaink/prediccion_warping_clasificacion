"""
Script: visualize_landmarks_overlay.py
Propósito: Visualizar landmarks anatómicos sobre imagen original de radiografía
Input:
    - Imagen original (data/dataset/COVID/COVID-269.png)
    - Landmarks predichos (outputs/predictions/all_landmarks.npz)
Output:
    - results/figures/warping_explained/landmarks_overlay_example.png

Descripción:
    Carga una imagen del dataset y sus landmarks predichos correspondientes,
    luego dibuja círculos y etiquetas (L1-L15) sobre cada punto anatómico.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Rutas base (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
FEATURE_DIR = PROJECT_ROOT / "feature" / "plan-fisher-warping"

# Input paths
DATA_DIR = PROJECT_ROOT / "data" / "dataset"
LANDMARKS_FILE = PROJECT_ROOT / "outputs" / "predictions" / "all_landmarks.npz"

# Output paths
OUTPUT_DIR = FEATURE_DIR / "results" / "figures" / "warping_explained"
OUTPUT_FILE = OUTPUT_DIR / "landmarks_overlay_example.png"

# Imagen de ejemplo a visualizar
EXAMPLE_IMAGE = "COVID-269"
EXAMPLE_CATEGORY = "COVID"


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def load_image(category: str, image_name: str, target_size: int = 224) -> np.ndarray:
    """
    Carga una imagen del dataset y la redimensiona al tamaño objetivo.

    Args:
        category: Categoría de la imagen (COVID, Normal, Viral_Pneumonia)
        image_name: Nombre de la imagen sin extensión (ej: "COVID-269")
        target_size: Tamaño objetivo para redimensionar (default: 224)

    Returns:
        Imagen en escala de grises redimensionada (target_size, target_size)
    """
    image_path = DATA_DIR / category / f"{image_name}.png"

    if not image_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

    # Cargar en escala de grises
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    # Redimensionar a 224x224 (escala de landmarks)
    image_resized = cv2.resize(image, (target_size, target_size))

    return image_resized


def load_landmarks(image_name: str, landmarks_file: Path) -> np.ndarray:
    """
    Carga los landmarks predichos para una imagen específica.

    Args:
        image_name: Nombre de la imagen sin extensión (ej: "COVID-269")
        landmarks_file: Path al archivo .npz con landmarks

    Returns:
        Array de landmarks (15, 2) con coordenadas (x, y)
    """
    if not landmarks_file.exists():
        raise FileNotFoundError(f"No se encontró archivo de landmarks: {landmarks_file}")

    # Cargar archivo .npz (allow_pickle=True para arrays de strings)
    data = np.load(str(landmarks_file), allow_pickle=True)

    # Buscar índice de la imagen
    all_image_names = data['all_image_names']
    idx = np.where(all_image_names == image_name)[0]

    if len(idx) == 0:
        raise ValueError(f"Imagen '{image_name}' no encontrada en landmarks")

    # Extraer landmarks
    landmarks = data['all_landmarks'][idx[0]]  # (15, 2)

    return landmarks


def draw_landmarks_on_image(
    image: np.ndarray,
    landmarks: np.ndarray,
    circle_radius: int = 3,
    circle_color: tuple = (255, 0, 0),  # Rojo en RGB
    text_color: tuple = (255, 255, 0),  # Amarillo en RGB
    font_scale: float = 0.35,
    font_thickness: int = 1
) -> np.ndarray:
    """
    Dibuja landmarks sobre una imagen con círculos y etiquetas.
    Formato apropiado para publicaciones científicas.

    Args:
        image: Imagen en escala de grises (H, W)
        landmarks: Array de landmarks (15, 2) con coordenadas (x, y)
        circle_radius: Radio de los círculos de landmarks
        circle_color: Color BGR de los círculos (para OpenCV)
        text_color: Color BGR del texto de etiquetas (para OpenCV)
        font_scale: Escala de fuente
        font_thickness: Grosor de fuente

    Returns:
        Imagen RGB con landmarks dibujados
    """
    # Convertir imagen grayscale a RGB para poder usar colores
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()

    # Obtener dimensiones de la imagen
    img_height, img_width = image_rgb.shape[:2]

    # Dibujar cada landmark
    for i, (x, y) in enumerate(landmarks):
        # Convertir coordenadas a enteros
        x_int, y_int = int(round(x)), int(round(y))

        # Dibujar círculo (OpenCV usa BGR, no RGB)
        cv2.circle(
            image_rgb,
            (x_int, y_int),
            radius=circle_radius,
            color=circle_color,
            thickness=-1  # -1 = relleno
        )

        # Dibujar borde del círculo para mejor visibilidad
        cv2.circle(
            image_rgb,
            (x_int, y_int),
            radius=circle_radius,
            color=(255, 255, 255),  # Borde blanco
            thickness=1
        )

        # Dibujar etiqueta (1, 2, ..., 15)
        label = f"{i+1}"

        # Calcular tamaño aproximado del texto
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        # Posicionamiento inteligente para evitar que el texto se salga
        margin = 5
        base_offset = circle_radius + 4

        # Determinar posición horizontal del texto
        if x_int + base_offset + text_width + margin > img_width:
            # Muy cerca del borde derecho -> poner texto a la izquierda
            text_x = x_int - base_offset - text_width
        else:
            # Posición normal -> texto a la derecha
            text_x = x_int + base_offset

        # Determinar posición vertical del texto
        if y_int - text_height - margin < 0:
            # Muy cerca del borde superior -> poner texto abajo
            text_y = y_int + base_offset + text_height
        elif y_int + text_height + margin > img_height:
            # Muy cerca del borde inferior -> poner texto arriba
            text_y = y_int - base_offset
        else:
            # Posición normal -> centrado verticalmente
            text_y = y_int + text_height // 2

        cv2.putText(
            image_rgb,
            label,
            (int(text_x), int(text_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA
        )

    return image_rgb


def create_visualization(
    image_original: np.ndarray,
    image_with_landmarks: np.ndarray,
    image_name: str,
    output_path: Path
):
    """
    Crea una visualización lado a lado de la imagen original y con landmarks.
    Formato apropiado para publicaciones científicas.

    Args:
        image_original: Imagen original sin landmarks (grayscale)
        image_with_landmarks: Imagen con landmarks dibujados (RGB)
        image_name: Nombre de la imagen para el título
        output_path: Path donde guardar la figura
    """
    # Convertir imagen original a RGB para mantener consistencia visual
    if len(image_original.shape) == 2:
        image_original_rgb = cv2.cvtColor(image_original, cv2.COLOR_GRAY2RGB)
    else:
        image_original_rgb = image_original

    # Crear figura con 2 subplots - estilo publicación científica
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Panel 1: Imagen original
    axes[0].imshow(image_original_rgb)
    axes[0].set_title('(a) Imagen Original', fontsize=11, pad=10)
    axes[0].axis('off')

    # Panel 2: Imagen con landmarks
    axes[1].imshow(image_with_landmarks)
    axes[1].set_title('(b) Landmarks Anatómicos', fontsize=11, pad=10)
    axes[1].axis('off')

    # Ajustar layout
    plt.tight_layout(pad=1.5)

    # Guardar figura con configuración para publicaciones
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        str(output_path),
        dpi=300,  # Alta resolución para publicaciones
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    print(f"✓ Visualización guardada en: {output_path}")

    # Cerrar figura
    plt.close(fig)


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal que ejecuta el pipeline de visualización.
    """
    print("=" * 80)
    print("VISUALIZACIÓN DE LANDMARKS SOBRE IMAGEN ORIGINAL")
    print("=" * 80)

    # 1. Cargar imagen original
    print(f"\n1. Cargando imagen: {EXAMPLE_IMAGE}")
    image_original = load_image(EXAMPLE_CATEGORY, EXAMPLE_IMAGE, target_size=224)
    print(f"   ✓ Imagen cargada: {image_original.shape}")

    # 2. Cargar landmarks correspondientes
    print(f"\n2. Cargando landmarks de: {LANDMARKS_FILE.name}")
    landmarks = load_landmarks(EXAMPLE_IMAGE, LANDMARKS_FILE)
    print(f"   ✓ Landmarks cargados: {landmarks.shape}")
    print(f"   ✓ Coordenadas (x, y) en escala 224x224")

    # 3. Dibujar landmarks sobre la imagen
    print(f"\n3. Dibujando landmarks sobre la imagen")
    image_with_landmarks = draw_landmarks_on_image(image_original, landmarks)
    print(f"   ✓ Se dibujaron {len(landmarks)} landmarks (L1-L15)")

    # 4. Crear visualización comparativa
    print(f"\n4. Generando visualización comparativa")
    create_visualization(
        image_original,
        image_with_landmarks,
        EXAMPLE_IMAGE,
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
