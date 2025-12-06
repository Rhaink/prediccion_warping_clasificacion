#!/usr/bin/env python3
"""
Piecewise Affine Warping para Normalizacion Geometrica

Sesion 20: Implementacion de warping piece-wise affine para transformar
imagenes a forma canonica usando PREDICCIONES del modelo (no ground truth).

Pipeline de Inferencia:
1. Imagen entrada -> Modelo predice 15 landmarks
2. Landmarks predichos + Forma canonica + Delaunay -> Warping
3. Imagen warpeada (normalizada geometricamente)

Autor: Proyecto Tesis Maestria
Fecha: 28-Nov-2024
"""

import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from scipy.spatial import Delaunay
import warnings

# Configuracion de paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shape_analysis"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
WARPED_DIR = OUTPUT_DIR / "warped"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Cargar conexiones anatomicas
import sys
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from landmark_connections import EJE_CENTRAL, PULMON_IZQUIERDO, PULMON_DERECHO


# =============================================================================
# PARTE 1: FUNCIONES DE CARGA
# =============================================================================

def load_canonical_shape(json_path: Path = OUTPUT_DIR / "canonical_shape_gpa.json") -> Tuple[np.ndarray, int]:
    """
    Cargar forma canonica desde JSON.

    Returns:
        canonical_shape: Array (15, 2) con coordenadas en pixeles
        image_size: Tamano de imagen (224)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    canonical_shape = np.array(data['canonical_shape_pixels'])
    image_size = data['image_size']

    return canonical_shape, image_size


def load_delaunay_triangles(json_path: Path = OUTPUT_DIR / "canonical_delaunay_triangles.json") -> np.ndarray:
    """
    Cargar triangulacion Delaunay desde JSON.

    Returns:
        triangles: Array (18, 3) con indices de vertices de cada triangulo
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    triangles = np.array(data['triangles'])
    return triangles


def load_test_predictions(npz_path: Path = PREDICTIONS_DIR / "test_predictions.npz") -> Dict:
    """
    Cargar predicciones del ensemble para test set.

    Returns:
        Dict con:
        - predictions: (96, 15, 2) predicciones del ensemble
        - ground_truth: (96, 15, 2) landmarks reales
        - image_names: nombres de imagenes
        - categories: categorias
    """
    data = np.load(npz_path, allow_pickle=True)

    return {
        'predictions': data['predictions'],
        'ground_truth': data['ground_truth'],
        'image_names': data['image_names'],
        'categories': data['categories']
    }


# =============================================================================
# PARTE 2: WARPING PIECE-WISE AFFINE
# =============================================================================

def get_affine_transform_matrix(src_tri: np.ndarray, dst_tri: np.ndarray) -> np.ndarray:
    """
    Calcular matriz de transformacion afin entre dos triangulos.

    Args:
        src_tri: Vertices del triangulo fuente (3, 2)
        dst_tri: Vertices del triangulo destino (3, 2)

    Returns:
        M: Matriz de transformacion afin 2x3
    """
    src = src_tri.astype(np.float32)
    dst = dst_tri.astype(np.float32)

    M = cv2.getAffineTransform(src, dst)
    return M


def create_triangle_mask(shape: Tuple[int, int], triangle: np.ndarray) -> np.ndarray:
    """
    Crear mascara binaria para un triangulo.

    Args:
        shape: (height, width) de la imagen
        triangle: Vertices del triangulo (3, 2)

    Returns:
        mask: Mascara binaria con el triangulo
    """
    mask = np.zeros(shape, dtype=np.uint8)
    pts = triangle.astype(np.int32)
    cv2.fillConvexPoly(mask, pts, 255)
    return mask


def get_bounding_box(triangle: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Obtener bounding box de un triangulo.

    Returns:
        (x, y, w, h) del bounding box
    """
    x_min = max(0, int(np.floor(triangle[:, 0].min())))
    y_min = max(0, int(np.floor(triangle[:, 1].min())))
    x_max = int(np.ceil(triangle[:, 0].max()))
    y_max = int(np.ceil(triangle[:, 1].max()))

    return x_min, y_min, x_max - x_min, y_max - y_min


def warp_triangle(src_img: np.ndarray,
                  dst_img: np.ndarray,
                  src_tri: np.ndarray,
                  dst_tri: np.ndarray) -> None:
    """
    Warping de un triangulo de imagen fuente a imagen destino.

    El warping es IN-PLACE (modifica dst_img directamente).

    Args:
        src_img: Imagen fuente (H, W) o (H, W, C)
        dst_img: Imagen destino (debe tener mismo tamaño que src_img)
        src_tri: Vertices del triangulo fuente (3, 2)
        dst_tri: Vertices del triangulo destino (3, 2)
    """
    # Obtener bounding boxes
    src_rect = get_bounding_box(src_tri)
    dst_rect = get_bounding_box(dst_tri)

    # Extraer regiones de interes
    src_x, src_y, src_w, src_h = src_rect
    dst_x, dst_y, dst_w, dst_h = dst_rect

    # Verificar que los bounding boxes son validos
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        return

    # Ajustar triangulos a coordenadas locales
    src_tri_local = src_tri.copy()
    src_tri_local[:, 0] -= src_x
    src_tri_local[:, 1] -= src_y

    dst_tri_local = dst_tri.copy()
    dst_tri_local[:, 0] -= dst_x
    dst_tri_local[:, 1] -= dst_y

    # Extraer parche de imagen fuente
    src_patch = src_img[src_y:src_y+src_h, src_x:src_x+src_w].copy()

    # Calcular transformacion afin
    M = get_affine_transform_matrix(src_tri_local, dst_tri_local)

    # Aplicar warp al parche
    warped_patch = cv2.warpAffine(src_patch, M, (dst_w, dst_h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT_101)

    # Crear mascara del triangulo destino
    mask = create_triangle_mask((dst_h, dst_w), dst_tri_local)

    # Aplicar mascara y copiar al destino
    # Expandir mascara si la imagen tiene canales de color
    if len(dst_img.shape) == 3:
        mask = mask[:, :, np.newaxis]

    # Mezclar usando la mascara
    dst_region = dst_img[dst_y:dst_y+dst_h, dst_x:dst_x+dst_w]
    np.copyto(dst_region, warped_patch, where=(mask > 0))


def add_boundary_points(landmarks: np.ndarray, image_size: int = 224) -> np.ndarray:
    """
    Agregar puntos de borde a los landmarks para cubrir toda la imagen.

    Agrega 8 puntos: 4 esquinas + 4 puntos medios de los bordes.

    Args:
        landmarks: Array (15, 2) con landmarks originales
        image_size: Tamaño de la imagen

    Returns:
        extended_landmarks: Array (23, 2) con landmarks + puntos de borde
    """
    # 4 esquinas
    corners = np.array([
        [0, 0],                    # Top-left
        [image_size-1, 0],         # Top-right
        [0, image_size-1],         # Bottom-left
        [image_size-1, image_size-1]  # Bottom-right
    ], dtype=np.float64)

    # 4 puntos medios de bordes
    midpoints = np.array([
        [image_size/2, 0],         # Top-center
        [0, image_size/2],         # Left-center
        [image_size-1, image_size/2],  # Right-center
        [image_size/2, image_size-1]   # Bottom-center
    ], dtype=np.float64)

    # Concatenar: landmarks + esquinas + midpoints
    extended = np.vstack([landmarks, corners, midpoints])
    return extended


def compute_extended_triangulation(landmarks: np.ndarray, image_size: int = 224) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcular triangulacion extendida que cubre toda la imagen.

    Args:
        landmarks: Array (15, 2) con landmarks
        image_size: Tamaño de la imagen

    Returns:
        extended_landmarks: Landmarks + puntos de borde (23, 2)
        triangles: Indices de triangulos sobre los puntos extendidos
    """
    extended = add_boundary_points(landmarks, image_size)

    # Calcular Delaunay
    tri = Delaunay(extended)

    return extended, tri.simplices


def piecewise_affine_warp(image: np.ndarray,
                          source_landmarks: np.ndarray,
                          target_landmarks: np.ndarray,
                          triangles: np.ndarray = None,
                          output_size: int = 224,
                          use_full_coverage: bool = True) -> np.ndarray:
    """
    Warping piece-wise affine de imagen usando triangulacion.

    Args:
        image: Imagen fuente (H, W) o (H, W, C)
        source_landmarks: Landmarks de la imagen fuente (15, 2)
        target_landmarks: Landmarks objetivo/canonicos (15, 2)
        triangles: Indices de triangulos (18, 3). Si None y use_full_coverage=True,
                   se calcula triangulacion extendida.
        output_size: Tamaño de imagen de salida
        use_full_coverage: Si True, añade puntos de borde para cobertura completa

    Returns:
        warped_image: Imagen warpeada a la forma canonica
    """
    # Si queremos cobertura completa, extender landmarks y recalcular triangulacion
    if use_full_coverage:
        src_extended = add_boundary_points(source_landmarks, output_size)
        dst_extended = add_boundary_points(target_landmarks, output_size)
        # Calcular nueva triangulacion sobre puntos destino
        tri = Delaunay(dst_extended)
        triangles = tri.simplices
        source_pts = src_extended
        target_pts = dst_extended
    else:
        source_pts = source_landmarks
        target_pts = target_landmarks

    # Crear imagen destino
    if len(image.shape) == 3:
        warped = np.zeros((output_size, output_size, image.shape[2]), dtype=image.dtype)
    else:
        warped = np.zeros((output_size, output_size), dtype=image.dtype)

    # Warpear cada triangulo
    for tri_indices in triangles:
        src_tri = source_pts[tri_indices]
        dst_tri = target_pts[tri_indices]

        # Verificar que los triangulos son validos (no degenerados)
        src_det = np.linalg.det(np.column_stack([src_tri - src_tri[0], [1, 1, 1]]))
        dst_det = np.linalg.det(np.column_stack([dst_tri - dst_tri[0], [1, 1, 1]]))

        if abs(src_det) < 1e-6 or abs(dst_det) < 1e-6:
            continue

        try:
            warp_triangle(image, warped, src_tri, dst_tri)
        except Exception as e:
            warnings.warn(f"Error warping triangle {tri_indices}: {e}")
            continue

    return warped


def inverse_piecewise_affine_warp(image: np.ndarray,
                                   source_landmarks: np.ndarray,
                                   target_landmarks: np.ndarray,
                                   triangles: np.ndarray = None,
                                   output_size: int = 224,
                                   use_full_coverage: bool = True) -> np.ndarray:
    """
    Warping inverso: de forma canonica a forma original.

    Util para generar imagenes con forma variada a partir de forma canonica.
    """
    # Simplemente intercambiar source y target
    return piecewise_affine_warp(image, target_landmarks, source_landmarks,
                                  triangles, output_size, use_full_coverage)


# =============================================================================
# PARTE 3: FUNCION DE ALTO NIVEL PARA INFERENCIA
# =============================================================================

def normalize_image_geometry(image: np.ndarray,
                              predicted_landmarks: np.ndarray,
                              canonical_shape: np.ndarray = None,
                              triangles: np.ndarray = None,
                              use_full_coverage: bool = True) -> np.ndarray:
    """
    Normalizar geometria de una imagen usando landmarks predichos.

    Esta funcion es la interfaz principal para el pipeline de inferencia:
    1. Recibe imagen + predicciones del modelo
    2. Aplica warping piece-wise affine
    3. Devuelve imagen normalizada geometricamente

    Args:
        image: Imagen de entrada (H, W) o (H, W, C)
        predicted_landmarks: Landmarks predichos por el modelo (15, 2)
        canonical_shape: Forma canonica (15, 2). Si None, se carga del JSON.
        triangles: Triangulacion (18, 3). Si None, se usa triangulacion dinamica.
        use_full_coverage: Si True, añade puntos de borde para cobertura completa.

    Returns:
        normalized_image: Imagen con geometria normalizada a forma canonica
    """
    # Cargar forma canonica si no se proporciona
    if canonical_shape is None:
        canonical_shape, _ = load_canonical_shape()

    # Redimensionar imagen si es necesario
    if image.shape[0] != 224 or image.shape[1] != 224:
        image = cv2.resize(image, (224, 224))

    # Aplicar warping con cobertura completa
    normalized = piecewise_affine_warp(
        image, predicted_landmarks, canonical_shape,
        triangles=triangles, output_size=224,
        use_full_coverage=use_full_coverage
    )

    return normalized


# =============================================================================
# PARTE 4: VISUALIZACIONES
# =============================================================================

def plot_landmarks_and_triangles(ax, landmarks: np.ndarray, triangles: np.ndarray,
                                  color='blue', alpha=0.5, label=None):
    """
    Dibujar landmarks y triangulacion en un axes.
    """
    # Dibujar triangulos
    for tri_indices in triangles:
        tri_pts = landmarks[tri_indices]
        # Cerrar el triangulo
        tri_pts = np.vstack([tri_pts, tri_pts[0]])
        ax.plot(tri_pts[:, 0], tri_pts[:, 1],
                color=color, alpha=alpha, linewidth=0.5)

    # Dibujar landmarks
    ax.scatter(landmarks[:, 0], landmarks[:, 1],
               c=color, s=30, zorder=5, label=label)

    # Etiquetar landmarks
    for i, (x, y) in enumerate(landmarks):
        ax.annotate(f'{i+1}', (x, y), fontsize=6, xytext=(2, 2),
                   textcoords='offset points')


def visualize_warp(original_image: np.ndarray,
                   warped_image: np.ndarray,
                   source_landmarks: np.ndarray,
                   canonical_landmarks: np.ndarray,
                   triangles: np.ndarray,
                   title: str = "Piecewise Affine Warp",
                   output_path: Optional[Path] = None):
    """
    Visualizar antes y despues del warping.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Imagen original con landmarks predichos
    ax = axes[0]
    ax.imshow(original_image, cmap='gray')
    plot_landmarks_and_triangles(ax, source_landmarks, triangles,
                                  color='red', alpha=0.7, label='Predichos')
    ax.set_title('Original + Landmarks Predichos', fontsize=12)
    ax.axis('off')

    # Imagen warpeada con forma canonica
    ax = axes[1]
    ax.imshow(warped_image, cmap='gray')
    plot_landmarks_and_triangles(ax, canonical_landmarks, triangles,
                                  color='green', alpha=0.7, label='Canonica')
    ax.set_title('Warped + Forma Canonica', fontsize=12)
    ax.axis('off')

    # Comparacion de formas
    ax = axes[2]
    # Crear imagen compuesta
    if len(original_image.shape) == 3:
        composite = np.zeros_like(original_image)
        composite[:, :, 0] = original_image[:, :, 0] * 0.5  # Original en rojo
        composite[:, :, 1] = warped_image[:, :, 1] * 0.5 if len(warped_image.shape) == 3 else warped_image * 0.5  # Warped en verde
    else:
        composite = np.zeros((original_image.shape[0], original_image.shape[1], 3), dtype=np.uint8)
        composite[:, :, 0] = (original_image * 0.5).astype(np.uint8)
        composite[:, :, 1] = (warped_image * 0.5).astype(np.uint8)

    ax.imshow(composite)
    # Dibujar ambas formas
    for i, (src, dst) in enumerate(zip(source_landmarks, canonical_landmarks)):
        ax.plot([src[0], dst[0]], [src[1], dst[1]], 'w-', alpha=0.5, linewidth=0.5)
    ax.scatter(source_landmarks[:, 0], source_landmarks[:, 1], c='red', s=20, label='Original')
    ax.scatter(canonical_landmarks[:, 0], canonical_landmarks[:, 1], c='green', s=20, label='Canonica')
    ax.set_title('Correspondencia de Landmarks', fontsize=12)
    ax.legend(loc='upper right', fontsize=8)
    ax.axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualizacion guardada: {output_path}")

    plt.close()


def visualize_warp_detailed(original_image: np.ndarray,
                             warped_image: np.ndarray,
                             source_landmarks: np.ndarray,
                             canonical_landmarks: np.ndarray,
                             triangles: np.ndarray,
                             image_name: str = "sample",
                             output_path: Optional[Path] = None):
    """
    Visualizacion detallada del warping con conexiones anatomicas.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Panel 1: Imagen original
    ax = axes[0, 0]
    ax.imshow(original_image, cmap='gray')
    ax.set_title('Imagen Original', fontsize=12)
    ax.axis('off')

    # Panel 2: Original con triangulacion
    ax = axes[0, 1]
    ax.imshow(original_image, cmap='gray')
    plot_landmarks_and_triangles(ax, source_landmarks, triangles, color='cyan', alpha=0.6)
    # Dibujar conexiones anatomicas
    for indices, color in [(EJE_CENTRAL, 'red'),
                           (PULMON_IZQUIERDO, 'blue'),
                           (PULMON_DERECHO, 'green')]:
        pts = source_landmarks[indices]
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2, alpha=0.8)
    ax.set_title('Original + Delaunay + Anatomia', fontsize=12)
    ax.axis('off')

    # Panel 3: Imagen warpeada
    ax = axes[1, 0]
    ax.imshow(warped_image, cmap='gray')
    ax.set_title('Imagen Warped', fontsize=12)
    ax.axis('off')

    # Panel 4: Warped con forma canonica
    ax = axes[1, 1]
    ax.imshow(warped_image, cmap='gray')
    plot_landmarks_and_triangles(ax, canonical_landmarks, triangles, color='lime', alpha=0.6)
    # Dibujar conexiones anatomicas canonicas
    for indices, color in [(EJE_CENTRAL, 'red'),
                           (PULMON_IZQUIERDO, 'blue'),
                           (PULMON_DERECHO, 'green')]:
        pts = canonical_landmarks[indices]
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2, alpha=0.8)
    ax.set_title('Warped + Forma Canonica', fontsize=12)
    ax.axis('off')

    plt.suptitle(f'Piecewise Affine Warp: {image_name}', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualizacion detallada guardada: {output_path}")

    plt.close()


# =============================================================================
# PARTE 5: MAIN - DEMO Y VERIFICACION
# =============================================================================

def get_image_path(image_name: str, category: str) -> Path:
    """Construir path a imagen original."""
    data_dir = PROJECT_ROOT / "data" / "dataset" / category
    return data_dir / f"{image_name}.png"


def process_all_test_images(test_data: Dict,
                             canonical_shape: np.ndarray,
                             triangles: np.ndarray,
                             max_images: int = None) -> Tuple[List[Dict], Dict]:
    """
    Procesar todas las imagenes del test set.

    Returns:
        warped_data: Lista con datos de cada imagen
        stats: Estadisticas de calidad
    """
    predictions = test_data['predictions']
    ground_truth = test_data['ground_truth']
    image_names = test_data['image_names']
    categories = test_data['categories']

    n_images = len(predictions) if max_images is None else min(max_images, len(predictions))

    warped_data = []
    stats = {'fill_rates': [], 'categories': {'Normal': [], 'COVID': [], 'Viral_Pneumonia': []}}

    for i in range(n_images):
        pred = predictions[i]
        gt = ground_truth[i]
        name = image_names[i]
        cat = categories[i]

        # Cargar imagen
        img_path = get_image_path(name, cat)
        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        # Resize si es necesario
        if image.shape[0] != 224 or image.shape[1] != 224:
            image = cv2.resize(image, (224, 224))

        # Aplicar warping usando PREDICCIONES
        warped = piecewise_affine_warp(image, pred, canonical_shape,
                                        use_full_coverage=True)

        # Calcular metricas
        black_pixels = np.sum(warped == 0)
        fill_rate = 1 - (black_pixels / warped.size)

        warped_data.append({
            'index': i,
            'name': name,
            'category': cat,
            'original': image,
            'warped': warped,
            'predicted': pred,
            'gt': gt,
            'fill_rate': fill_rate
        })

        stats['fill_rates'].append(fill_rate)
        stats['categories'][cat].append(fill_rate)

    return warped_data, stats


def main():
    """
    Demo del warping piece-wise affine con predicciones del ensemble.
    """
    print("=" * 60)
    print("SESION 20: Piecewise Affine Warping")
    print("=" * 60)

    # Crear directorio de salida
    WARPED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cargar datos
    print("\n1. Cargando datos...")
    canonical_shape, image_size = load_canonical_shape()
    triangles = load_delaunay_triangles()
    test_data = load_test_predictions()

    print(f"   Forma canonica: {canonical_shape.shape}")
    print(f"   Triangulos: {triangles.shape}")
    print(f"   Predicciones test: {test_data['predictions'].shape}")

    # 2. Procesar TODAS las imagenes del test set
    print("\n2. Procesando todas las imagenes del test set...")
    warped_data, stats = process_all_test_images(test_data, canonical_shape, triangles)
    print(f"   Imagenes procesadas: {len(warped_data)}")

    # 3. Generar visualizaciones de ejemplo (una por categoria)
    print("\n3. Generando visualizaciones de ejemplo...")

    categories_seen = set()
    examples = []

    for data in warped_data:
        if data['category'] not in categories_seen:
            categories_seen.add(data['category'])
            examples.append(data)
        if len(categories_seen) == 3:
            break

    for example in examples:
        print(f"   {example['name']} ({example['category']})")

        # Guardar visualizacion simple
        output_path = WARPED_DIR / f"warp_{example['category']}_{example['name']}.png"
        visualize_warp(example['original'], example['warped'],
                       example['predicted'], canonical_shape,
                       triangles, f"Warping: {example['name']}", output_path)

        # Guardar visualizacion detallada
        output_path_detailed = WARPED_DIR / f"warp_detailed_{example['category']}_{example['name']}.png"
        visualize_warp_detailed(example['original'], example['warped'],
                                 example['predicted'], canonical_shape,
                                 triangles, example['name'], output_path_detailed)

    # 4. Crear grilla comparativa
    print("\n4. Creando grilla comparativa...")

    if len(examples) >= 3:
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))

        for row, data in enumerate(examples[:3]):
            # Original
            axes[row, 0].imshow(data['original'], cmap='gray')
            axes[row, 0].set_title(f"Original\n{data['category']}", fontsize=10)
            axes[row, 0].axis('off')

            # Original con landmarks
            axes[row, 1].imshow(data['original'], cmap='gray')
            axes[row, 1].scatter(data['predicted'][:, 0], data['predicted'][:, 1],
                                c='red', s=20)
            axes[row, 1].set_title("+ Landmarks\nPredichos", fontsize=10)
            axes[row, 1].axis('off')

            # Warped
            axes[row, 2].imshow(data['warped'], cmap='gray')
            axes[row, 2].set_title("Warped", fontsize=10)
            axes[row, 2].axis('off')

            # Warped con forma canonica
            axes[row, 3].imshow(data['warped'], cmap='gray')
            axes[row, 3].scatter(canonical_shape[:, 0], canonical_shape[:, 1],
                                c='green', s=20)
            axes[row, 3].set_title("+ Forma\nCanonica", fontsize=10)
            axes[row, 3].axis('off')

        plt.suptitle('Piecewise Affine Warping: Normalizacion Geometrica', fontsize=14)
        plt.tight_layout()

        grid_path = WARPED_DIR / "warp_comparison_grid.png"
        plt.savefig(grid_path, dpi=150, bbox_inches='tight')
        print(f"   Grilla guardada: {grid_path}")
        plt.close()

    # 5. Verificar calidad del warping (estadisticas globales)
    print("\n5. Verificando calidad del warping...")

    fill_rates = np.array(stats['fill_rates'])
    print(f"   Fill rate global:")
    print(f"      Mean: {fill_rates.mean()*100:.1f}%")
    print(f"      Std:  {fill_rates.std()*100:.1f}%")
    print(f"      Min:  {fill_rates.min()*100:.1f}%")
    print(f"      Max:  {fill_rates.max()*100:.1f}%")

    print(f"\n   Fill rate por categoria:")
    for cat in ['Normal', 'COVID', 'Viral_Pneumonia']:
        cat_rates = np.array(stats['categories'][cat])
        if len(cat_rates) > 0:
            print(f"      {cat}: {cat_rates.mean()*100:.1f}% (n={len(cat_rates)})")

    # 6. Guardar imagenes warpeadas
    print("\n6. Guardando imagenes warpeadas...")

    warped_images_array = np.array([d['warped'] for d in warped_data])
    original_images_array = np.array([d['original'] for d in warped_data])

    np.savez(WARPED_DIR / "warped_test_images.npz",
             warped_images=warped_images_array,
             original_images=original_images_array,
             predictions=np.array([d['predicted'] for d in warped_data]),
             ground_truth=np.array([d['gt'] for d in warped_data]),
             image_names=np.array([d['name'] for d in warped_data]),
             categories=np.array([d['category'] for d in warped_data]),
             fill_rates=fill_rates,
             canonical_shape=canonical_shape)
    print(f"   Guardado: warped_test_images.npz ({len(warped_data)} imagenes)")

    # 7. Guardar configuracion del warping
    config = {
        'canonical_shape_file': str(OUTPUT_DIR / "canonical_shape_gpa.json"),
        'triangles_file': str(OUTPUT_DIR / "canonical_delaunay_triangles.json"),
        'num_triangles': len(triangles),
        'num_landmarks': 15,
        'num_boundary_points': 8,
        'total_points_extended': 23,
        'image_size': 224,
        'interpolation': 'bilinear',
        'border_mode': 'reflect_101',
        'fill_rate_mean': float(fill_rates.mean()),
        'fill_rate_std': float(fill_rates.std()),
        'images_processed': len(warped_data),
        'session': 20,
        'date': '2024-11-28'
    }

    config_path = WARPED_DIR / "warp_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"   Configuracion guardada: {config_path}")

    # 8. Crear histograma de fill rates
    print("\n7. Generando histograma de fill rates...")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(fill_rates * 100, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(fill_rates.mean() * 100, color='red', linestyle='--',
               label=f'Mean: {fill_rates.mean()*100:.1f}%')
    ax.set_xlabel('Fill Rate (%)', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.set_title('Distribucion de Fill Rates del Warping', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    hist_path = WARPED_DIR / "fill_rate_histogram.png"
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Histograma guardado: {hist_path}")

    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN - Sesion 20")
    print("=" * 60)
    print(f"Imagenes procesadas: {len(warped_data)}")
    print(f"Triangulos Delaunay: {len(triangles)} (original)")
    print(f"Puntos extendidos: 23 (15 landmarks + 8 borde)")
    print(f"Fill rate promedio: {fill_rates.mean()*100:.1f}%")
    print(f"\nArchivos generados en: {WARPED_DIR}")
    print("\nArchivos:")
    for f in sorted(WARPED_DIR.glob("*")):
        print(f"  - {f.name}")

    return warped_data, stats


if __name__ == "__main__":
    main()
