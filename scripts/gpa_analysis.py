#!/usr/bin/env python3
"""
Generalized Procrustes Analysis (GPA) para Forma Canonica

Sesion 19: Implementacion de GPA para calcular la forma canonica
de los 15 landmarks anatomicos en radiografias de torax.

El GPA elimina:
- Traslacion (centrar en origen)
- Escala (normalizar a norma unitaria)
- Rotacion (alinear con referencia)

Autor: Proyecto Tesis Maestria
Fecha: 28-Nov-2024
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings

# Configuracion de paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "outputs" / "predictions" / "all_landmarks.npz"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shape_analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Nombres de landmarks para visualizacion
LANDMARK_NAMES = [
    "L1 (Superior)", "L2 (Inferior)", "L3 (Apex Izq)", "L4 (Apex Der)",
    "L5 (Hilio Izq)", "L6 (Hilio Der)", "L7 (Base Izq)", "L8 (Base Der)",
    "L9 (Centro Sup)", "L10 (Centro Med)", "L11 (Centro Inf)",
    "L12 (Borde Sup Izq)", "L13 (Borde Sup Der)",
    "L14 (Costofrenico Izq)", "L15 (Costofrenico Der)"
]


# =============================================================================
# PARTE 1: FUNCIONES BASE DE PROCRUSTES
# =============================================================================

def center_shape(shape: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Centrar una forma en el origen (eliminar traslacion).

    Args:
        shape: Array (n_landmarks, 2) con coordenadas (x, y)

    Returns:
        centered: Forma centrada en el origen
        centroid: Centroide original (para poder revertir)
    """
    centroid = shape.mean(axis=0)
    centered = shape - centroid
    return centered, centroid


def scale_shape(shape: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Escalar una forma a norma unitaria (Frobenius norm = 1).

    Args:
        shape: Array (n_landmarks, 2) centrado en origen

    Returns:
        scaled: Forma con norma unitaria
        scale: Factor de escala original (para poder revertir)
    """
    scale = np.linalg.norm(shape, 'fro')
    if scale < 1e-10:
        warnings.warn("Shape has near-zero scale")
        return shape, 1.0
    scaled = shape / scale
    return scaled, scale


def optimal_rotation_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Calcular la matriz de rotacion optima usando SVD (Procrustes).

    Encuentra R que minimiza ||source @ R - target||^2

    Args:
        source: Forma a rotar (n_landmarks, 2)
        target: Forma de referencia (n_landmarks, 2)

    Returns:
        R: Matriz de rotacion 2x2
    """
    # Correlacion cruzada: H = source^T @ target
    H = source.T @ target

    # SVD: H = U @ S @ Vt
    U, S, Vt = np.linalg.svd(H)

    # Matriz de rotacion optima: R = V @ U^T
    R = Vt.T @ U.T

    # Asegurar rotacion propia (det = +1, no reflexion)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


def align_shape(shape: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Alinear una forma con una referencia (eliminar rotacion).

    Args:
        shape: Forma a alinear (n_landmarks, 2) - ya centrada y escalada
        reference: Forma de referencia (n_landmarks, 2) - ya centrada y escalada

    Returns:
        aligned: Forma rotada para minimizar distancia a referencia
    """
    R = optimal_rotation_matrix(shape, reference)
    aligned = shape @ R
    return aligned


def procrustes_distance(shape1: np.ndarray, shape2: np.ndarray) -> float:
    """
    Calcular distancia Procrustes entre dos formas.

    La distancia se calcula DESPUES de alinear ambas formas.

    Args:
        shape1, shape2: Formas a comparar (n_landmarks, 2)

    Returns:
        distance: Distancia Procrustes (norma Frobenius del residuo)
    """
    # Centrar y escalar ambas
    s1_centered, _ = center_shape(shape1)
    s1_scaled, _ = scale_shape(s1_centered)

    s2_centered, _ = center_shape(shape2)
    s2_scaled, _ = scale_shape(s2_centered)

    # Alinear shape1 con shape2
    s1_aligned = align_shape(s1_scaled, s2_scaled)

    # Distancia = norma de la diferencia
    distance = np.linalg.norm(s1_aligned - s2_scaled, 'fro')
    return distance


def full_procrustes_alignment(shape: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Alineacion Procrustes completa: centrar, escalar, rotar.

    Args:
        shape: Forma a alinear (n_landmarks, 2)
        reference: Forma de referencia (n_landmarks, 2)

    Returns:
        aligned: Forma completamente alineada
        params: Diccionario con parametros de transformacion
    """
    # Paso 1: Centrar
    shape_centered, centroid = center_shape(shape)
    ref_centered, ref_centroid = center_shape(reference)

    # Paso 2: Escalar
    shape_scaled, scale = scale_shape(shape_centered)
    ref_scaled, ref_scale = scale_shape(ref_centered)

    # Paso 3: Rotar
    R = optimal_rotation_matrix(shape_scaled, ref_scaled)
    aligned = shape_scaled @ R

    params = {
        'centroid': centroid,
        'scale': scale,
        'rotation': R,
        'ref_centroid': ref_centroid,
        'ref_scale': ref_scale
    }

    return aligned, params


# =============================================================================
# PARTE 2: GENERALIZED PROCRUSTES ANALYSIS (GPA) ITERATIVO
# =============================================================================

def gpa_iterative(shapes: np.ndarray,
                  max_iterations: int = 100,
                  tolerance: float = 1e-8,
                  verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generalized Procrustes Analysis iterativo.

    Algoritmo:
    1. Centrar y escalar todas las formas
    2. Inicializar referencia = promedio de formas
    3. Repetir hasta convergencia:
       a) Rotar cada forma para alinear con referencia
       b) Calcular nueva referencia = promedio de formas alineadas
       c) Normalizar referencia
       d) Verificar convergencia (cambio < tolerancia)

    Args:
        shapes: Array (n_shapes, n_landmarks, 2) con todas las formas
        max_iterations: Numero maximo de iteraciones
        tolerance: Tolerancia para convergencia
        verbose: Si True, imprime progreso

    Returns:
        canonical_shape: Forma canonica (consenso Procrustes)
        aligned_shapes: Todas las formas alineadas (n_shapes, n_landmarks, 2)
        convergence_info: Informacion de convergencia
    """
    n_shapes, n_landmarks, n_dims = shapes.shape

    if verbose:
        print(f"GPA: {n_shapes} formas, {n_landmarks} landmarks")

    # Paso 1: Centrar y escalar todas las formas
    normalized_shapes = np.zeros_like(shapes)
    original_scales = np.zeros(n_shapes)
    original_centroids = np.zeros((n_shapes, 2))

    for i in range(n_shapes):
        centered, centroid = center_shape(shapes[i])
        scaled, scale = scale_shape(centered)
        normalized_shapes[i] = scaled
        original_scales[i] = scale
        original_centroids[i] = centroid

    # Paso 2: Inicializar referencia como promedio
    reference = normalized_shapes.mean(axis=0)
    reference_scaled, _ = scale_shape(reference)

    # Almacenar historico de convergencia
    distances_history = []

    # Paso 3: Iteracion hasta convergencia
    aligned_shapes = normalized_shapes.copy()

    for iteration in range(max_iterations):
        # 3a) Rotar cada forma para alinear con referencia
        for i in range(n_shapes):
            aligned_shapes[i] = align_shape(normalized_shapes[i], reference_scaled)

        # 3b) Calcular nueva referencia
        new_reference = aligned_shapes.mean(axis=0)

        # 3c) Normalizar referencia
        new_reference_scaled, _ = scale_shape(new_reference)

        # 3d) Calcular cambio (distancia entre referencias)
        change = np.linalg.norm(new_reference_scaled - reference_scaled, 'fro')

        # Calcular distancia promedio al consenso
        mean_distance = np.mean([
            np.linalg.norm(aligned_shapes[i] - new_reference_scaled, 'fro')
            for i in range(n_shapes)
        ])
        distances_history.append(mean_distance)

        if verbose and (iteration < 5 or iteration % 10 == 0):
            print(f"  Iter {iteration}: cambio={change:.2e}, dist_promedio={mean_distance:.6f}")

        # Verificar convergencia
        if change < tolerance:
            if verbose:
                print(f"  Convergencia en iteracion {iteration} (cambio={change:.2e})")
            reference_scaled = new_reference_scaled
            break

        reference_scaled = new_reference_scaled
        # NO copiar aligned_shapes a normalized_shapes - siempre alinear las formas originales normalizadas

    else:
        if verbose:
            print(f"  Alcanzado maximo de iteraciones ({max_iterations})")

    # Forma canonica final
    canonical_shape = reference_scaled

    # Informacion de convergencia
    convergence_info = {
        'n_iterations': iteration + 1,
        'converged': change < tolerance,
        'final_change': float(change),
        'distances_history': distances_history,
        'original_scales': original_scales,
        'original_centroids': original_centroids,
        'n_shapes': n_shapes,
        'n_landmarks': n_landmarks
    }

    return canonical_shape, aligned_shapes, convergence_info


# =============================================================================
# PARTE 3: CONVERSION A ESCALA ORIGINAL Y GUARDADO
# =============================================================================

def scale_canonical_to_image(canonical_shape: np.ndarray,
                             image_size: int = 224,
                             padding: float = 0.1) -> np.ndarray:
    """
    Convertir forma canonica normalizada a escala de imagen.

    Args:
        canonical_shape: Forma canonica (n_landmarks, 2) con norma ~1
        image_size: Tamano de la imagen destino
        padding: Margen relativo (0.1 = 10% de padding)

    Returns:
        scaled_shape: Forma en coordenadas de imagen (pixeles)
    """
    # La forma canonica esta centrada en (0,0) con norma ~1
    # Necesitamos escalarla y trasladarla al centro de la imagen

    # Calcular rango actual
    min_coords = canonical_shape.min(axis=0)
    max_coords = canonical_shape.max(axis=0)
    range_coords = max_coords - min_coords

    # Escalar para que quepa en imagen con padding
    usable_size = image_size * (1 - 2 * padding)
    scale_factor = usable_size / max(range_coords)

    # Escalar y centrar
    scaled = canonical_shape * scale_factor

    # Trasladar al centro de la imagen
    scaled_center = scaled.mean(axis=0)
    image_center = np.array([image_size / 2, image_size / 2])
    scaled_shape = scaled - scaled_center + image_center

    return scaled_shape


def save_canonical_shape(canonical_shape: np.ndarray,
                        canonical_shape_pixels: np.ndarray,
                        convergence_info: Dict,
                        output_path: Path):
    """
    Guardar forma canonica en formato JSON.
    """
    data = {
        'canonical_shape_normalized': canonical_shape.tolist(),
        'canonical_shape_pixels': canonical_shape_pixels.tolist(),
        'image_size': 224,
        'n_landmarks': int(canonical_shape.shape[0]),
        'landmark_names': LANDMARK_NAMES,
        'convergence': {
            'n_iterations': int(convergence_info['n_iterations']),
            'converged': bool(convergence_info['converged']),
            'final_change': float(convergence_info['final_change']),
            'n_shapes_used': int(convergence_info['n_shapes'])
        },
        'method': 'Generalized Procrustes Analysis (GPA)',
        'date': '2024-11-28',
        'session': 19
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Forma canonica guardada en: {output_path}")


# =============================================================================
# PARTE 4: VISUALIZACIONES
# =============================================================================

def plot_convergence(convergence_info: Dict, output_path: Optional[Path] = None):
    """
    Graficar convergencia del GPA.
    """
    distances = convergence_info['distances_history']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(distances, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Iteracion', fontsize=12)
    ax.set_ylabel('Distancia promedio al consenso', fontsize=12)
    ax.set_title('Convergencia del GPA', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(distances) - 1)

    # Marcar convergencia
    if convergence_info['converged']:
        ax.axhline(y=distances[-1], color='g', linestyle='--', alpha=0.5,
                   label=f'Convergencia: {distances[-1]:.6f}')
        ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Grafico de convergencia guardado en: {output_path}")

    plt.close()


def plot_canonical_shape(canonical_shape_pixels: np.ndarray,
                        output_path: Optional[Path] = None):
    """
    Visualizar forma canonica con landmarks etiquetados.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plotear puntos
    ax.scatter(canonical_shape_pixels[:, 0], canonical_shape_pixels[:, 1],
               c='blue', s=100, zorder=5)

    # Etiquetar landmarks
    for i, (x, y) in enumerate(canonical_shape_pixels):
        ax.annotate(f'L{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold')

    # Conectar landmarks del eje central
    eje_indices = [0, 8, 9, 10, 1]  # L1, L9, L10, L11, L2
    eje_points = canonical_shape_pixels[eje_indices]
    ax.plot(eje_points[:, 0], eje_points[:, 1], 'r-', linewidth=2,
            label='Eje central', alpha=0.7)

    # Conectar pares bilaterales
    pairs = [(2, 3), (4, 5), (6, 7), (11, 12), (13, 14)]
    for left, right in pairs:
        ax.plot([canonical_shape_pixels[left, 0], canonical_shape_pixels[right, 0]],
               [canonical_shape_pixels[left, 1], canonical_shape_pixels[right, 1]],
               'g--', alpha=0.5)

    ax.set_xlim(0, 224)
    ax.set_ylim(224, 0)  # Invertir Y para que coincida con coordenadas de imagen
    ax.set_aspect('equal')
    ax.set_title('Forma Canonica (GPA)\n15 Landmarks Anatomicos', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualizacion guardada en: {output_path}")

    plt.close()


def plot_aligned_shapes_sample(aligned_shapes: np.ndarray,
                               canonical_shape: np.ndarray,
                               n_samples: int = 50,
                               output_path: Optional[Path] = None):
    """
    Visualizar muestra de formas alineadas junto con forma canonica.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Seleccionar muestra aleatoria
    n_shapes = aligned_shapes.shape[0]
    if n_samples < n_shapes:
        indices = np.random.choice(n_shapes, n_samples, replace=False)
    else:
        indices = np.arange(n_shapes)

    # Plotear formas alineadas (en gris, muy transparente)
    for idx in indices:
        ax.scatter(aligned_shapes[idx, :, 0], aligned_shapes[idx, :, 1],
                  c='gray', s=10, alpha=0.3)

    # Plotear forma canonica (en rojo, solido)
    ax.scatter(canonical_shape[:, 0], canonical_shape[:, 1],
               c='red', s=100, zorder=5, label='Forma Canonica')

    # Conectar landmarks de forma canonica
    for i in range(len(canonical_shape)):
        ax.annotate(f'{i+1}', canonical_shape[i], fontsize=8)

    ax.set_aspect('equal')
    ax.set_title(f'Formas Alineadas por GPA\n({n_samples} de {n_shapes} formas)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualizacion guardada en: {output_path}")

    plt.close()


# =============================================================================
# PARTE 5: PCA INFORMATIVO (PARA TESIS)
# =============================================================================

def pca_on_aligned_shapes(aligned_shapes: np.ndarray,
                          n_components: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA sobre formas alineadas para analisis de variabilidad.

    Args:
        aligned_shapes: Array (n_shapes, n_landmarks, 2)
        n_components: Numero de componentes a extraer

    Returns:
        eigenvalues: Valores propios (varianza explicada)
        eigenvectors: Vectores propios (modos de variacion)
        scores: Proyeccion de formas en espacio PCA
    """
    n_shapes, n_landmarks, n_dims = aligned_shapes.shape

    # Aplanar: (n_shapes, n_landmarks*2)
    flat_shapes = aligned_shapes.reshape(n_shapes, -1)

    # Centrar
    mean_shape = flat_shapes.mean(axis=0)
    centered = flat_shapes - mean_shape

    # Matriz de covarianza
    cov_matrix = np.cov(centered.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Ordenar por varianza descendente
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order][:n_components]
    eigenvectors = eigenvectors[:, order][:, :n_components]

    # Proyectar formas
    scores = centered @ eigenvectors

    return eigenvalues, eigenvectors, scores


def plot_pca_modes(canonical_shape: np.ndarray,
                   eigenvectors: np.ndarray,
                   eigenvalues: np.ndarray,
                   output_path: Optional[Path] = None):
    """
    Visualizar modos de variacion PC1 y PC2 (±2σ).
    """
    n_landmarks = canonical_shape.shape[0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for pc_idx, ax_row in enumerate(axes):
        eigenvec = eigenvectors[:, pc_idx].reshape(n_landmarks, 2)
        std = np.sqrt(eigenvalues[pc_idx])
        var_explained = eigenvalues[pc_idx] / eigenvalues.sum() * 100

        for col_idx, sigma in enumerate([-2, 0, 2]):
            ax = ax_row[col_idx]

            # Forma con variacion
            varied_shape = canonical_shape + sigma * std * eigenvec

            # Plotear
            ax.scatter(varied_shape[:, 0], varied_shape[:, 1],
                      c='blue', s=80, zorder=5)

            # Conectar landmarks
            for i in range(n_landmarks):
                ax.annotate(f'{i+1}', varied_shape[i], fontsize=8)

            # Conectar eje central
            eje = [0, 8, 9, 10, 1]
            ax.plot(varied_shape[eje, 0], varied_shape[eje, 1], 'r-', alpha=0.5)

            ax.set_aspect('equal')
            ax.set_title(f'PC{pc_idx+1} ({var_explained:.1f}%): {sigma:+d}σ', fontsize=12)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Modos de Variacion PCA\n(sobre formas alineadas por GPA)', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Modos PCA guardados en: {output_path}")

    plt.close()


def plot_pca_scatter_by_category(scores: np.ndarray,
                                 categories: np.ndarray,
                                 eigenvalues: np.ndarray,
                                 output_path: Optional[Path] = None):
    """
    Scatter plot de PC1 vs PC2 coloreado por categoria.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_cats = np.unique(categories)
    colors = {'Normal': 'green', 'COVID': 'red', 'Viral_Pneumonia': 'blue'}

    for cat in unique_cats:
        mask = categories == cat
        ax.scatter(scores[mask, 0], scores[mask, 1],
                  c=colors.get(cat, 'gray'), label=cat, alpha=0.6, s=50)

    var1 = eigenvalues[0] / eigenvalues.sum() * 100
    var2 = eigenvalues[1] / eigenvalues.sum() * 100

    ax.set_xlabel(f'PC1 ({var1:.1f}% varianza)', fontsize=12)
    ax.set_ylabel(f'PC2 ({var2:.1f}% varianza)', fontsize=12)
    ax.set_title('Distribucion de Formas por Categoria\n(espacio PCA)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(0, color='k', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Scatter PCA por categoria guardado en: {output_path}")

    plt.close()


# =============================================================================
# MAIN: EJECUTAR GPA COMPLETO
# =============================================================================

def main():
    """
    Ejecutar GPA completo sobre todo el dataset.
    """
    print("=" * 60)
    print("SESION 19: Generalized Procrustes Analysis (GPA)")
    print("=" * 60)

    # Crear directorios de salida
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar datos
    print("\n1. Cargando datos...")
    data = np.load(DATA_PATH, allow_pickle=True)
    all_landmarks = data['all_landmarks']  # (957, 15, 2)
    image_names = data['all_image_names']
    categories = data['all_categories']

    print(f"   Formas cargadas: {all_landmarks.shape}")
    print(f"   Categorias: {np.unique(categories, return_counts=True)}")

    # Ejecutar GPA
    print("\n2. Ejecutando GPA iterativo...")
    canonical_shape, aligned_shapes, convergence_info = gpa_iterative(
        all_landmarks,
        max_iterations=100,
        tolerance=1e-4,  # Tolerancia practica (1e-10 es demasiado estricta)
        verbose=True
    )

    print(f"\n   Forma canonica (normalizada): shape={canonical_shape.shape}")
    print(f"   Formas alineadas: shape={aligned_shapes.shape}")
    print(f"   Convergencia en {convergence_info['n_iterations']} iteraciones")

    # Convertir a escala de imagen
    print("\n3. Convirtiendo a escala de imagen (224x224)...")
    canonical_shape_pixels = scale_canonical_to_image(canonical_shape, image_size=224)
    print(f"   Rango X: {canonical_shape_pixels[:, 0].min():.1f} - {canonical_shape_pixels[:, 0].max():.1f}")
    print(f"   Rango Y: {canonical_shape_pixels[:, 1].min():.1f} - {canonical_shape_pixels[:, 1].max():.1f}")

    # Guardar forma canonica
    print("\n4. Guardando forma canonica...")
    save_canonical_shape(
        canonical_shape,
        canonical_shape_pixels,
        convergence_info,
        OUTPUT_DIR / "canonical_shape_gpa.json"
    )

    # Guardar formas alineadas
    np.savez(
        OUTPUT_DIR / "aligned_shapes.npz",
        aligned_shapes=aligned_shapes,
        canonical_shape=canonical_shape,
        canonical_shape_pixels=canonical_shape_pixels,
        image_names=image_names,
        categories=categories
    )
    print(f"   Formas alineadas guardadas en: {OUTPUT_DIR / 'aligned_shapes.npz'}")

    # Generar visualizaciones
    print("\n5. Generando visualizaciones...")

    # Convergencia
    plot_convergence(convergence_info, FIGURES_DIR / "gpa_convergence.png")

    # Forma canonica
    plot_canonical_shape(canonical_shape_pixels, FIGURES_DIR / "canonical_shape.png")

    # Formas alineadas
    plot_aligned_shapes_sample(aligned_shapes, canonical_shape,
                               n_samples=100, output_path=FIGURES_DIR / "aligned_shapes_sample.png")

    # PCA informativo
    print("\n6. PCA informativo (para tesis)...")
    eigenvalues, eigenvectors, scores = pca_on_aligned_shapes(aligned_shapes, n_components=5)

    var_explained = eigenvalues / eigenvalues.sum() * 100
    print(f"   Varianza explicada:")
    for i, var in enumerate(var_explained):
        print(f"     PC{i+1}: {var:.2f}%")
    print(f"   PC1 + PC2: {var_explained[0] + var_explained[1]:.2f}%")

    # Visualizaciones PCA
    plot_pca_modes(canonical_shape, eigenvectors, eigenvalues,
                   FIGURES_DIR / "pca_modes_variation.png")

    plot_pca_scatter_by_category(scores, categories, eigenvalues,
                                 FIGURES_DIR / "pca_category_scatter.png")

    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN - Sesion 19 Completada")
    print("=" * 60)
    print(f"Formas procesadas: {all_landmarks.shape[0]}")
    print(f"Iteraciones GPA: {convergence_info['n_iterations']}")
    print(f"Convergencia: {'SI' if convergence_info['converged'] else 'NO'}")
    print(f"Distancia final al consenso: {convergence_info['distances_history'][-1]:.6f}")
    print(f"\nArchivos generados:")
    print(f"  - {OUTPUT_DIR / 'canonical_shape_gpa.json'}")
    print(f"  - {OUTPUT_DIR / 'aligned_shapes.npz'}")
    print(f"  - {FIGURES_DIR / 'gpa_convergence.png'}")
    print(f"  - {FIGURES_DIR / 'canonical_shape.png'}")
    print(f"  - {FIGURES_DIR / 'aligned_shapes_sample.png'}")
    print(f"  - {FIGURES_DIR / 'pca_modes_variation.png'}")
    print(f"  - {FIGURES_DIR / 'pca_category_scatter.png'}")

    return canonical_shape_pixels, aligned_shapes, convergence_info


if __name__ == "__main__":
    main()
