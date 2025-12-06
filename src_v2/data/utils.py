"""
Utilidades para manejo de datos de landmarks
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from PIL import Image


# Pares de landmarks simetricos (indices 0-based)
# L3-L4, L5-L6, L7-L8, L12-L13, L14-L15
SYMMETRIC_PAIRS = [(2, 3), (4, 5), (6, 7), (11, 12), (13, 14)]

# Landmarks centrales (deben estar sobre el eje L1-L2)
CENTRAL_LANDMARKS = [8, 9, 10]  # L9, L10, L11

# Nombres de landmarks
LANDMARK_NAMES = [
    'L1',  'L2',  'L3',  'L4',  'L5',  'L6',  'L7',  'L8',
    'L9',  'L10', 'L11', 'L12', 'L13', 'L14', 'L15'
]


def load_coordinates_csv(csv_path: str) -> pd.DataFrame:
    """
    Carga el CSV de coordenadas y devuelve DataFrame estructurado.

    Estructura del CSV:
    - Columna 0: indice
    - Columnas 1-30: coordenadas (L1_x, L1_y, ..., L15_x, L15_y)
    - Columna 31: nombre de imagen

    Returns:
        DataFrame con columnas: image_name, category, L1_x, L1_y, ..., L15_x, L15_y
    """
    # Nombres de columnas
    coord_cols = []
    for i in range(1, 16):
        coord_cols.extend([f'L{i}_x', f'L{i}_y'])

    columns = ['idx'] + coord_cols + ['image_name']

    df = pd.read_csv(csv_path, header=None, names=columns)

    # Extraer categoria del nombre de imagen
    def extract_category(name: str) -> str:
        if name.startswith('COVID'):
            return 'COVID'
        elif name.startswith('Normal'):
            return 'Normal'
        elif name.startswith('Viral'):
            return 'Viral_Pneumonia'
        else:
            return 'Unknown'

    df['category'] = df['image_name'].apply(extract_category)

    # Eliminar columna de indice original
    df = df.drop('idx', axis=1)

    return df


def get_image_path(image_name: str, category: str, data_root: str) -> Path:
    """
    Construye la ruta completa a una imagen.

    Nota: Las imágenes son .png. El directorio es Viral_Pneumonia pero
    los nombres de archivo usan "Viral Pneumonia" con espacio.
    """
    return Path(data_root) / 'dataset' / category / f'{image_name}.png'


def get_landmarks_array(row: pd.Series) -> np.ndarray:
    """
    Extrae coordenadas de landmarks como array numpy (15, 2).
    """
    coords = []
    for i in range(1, 16):
        x = row[f'L{i}_x']
        y = row[f'L{i}_y']
        coords.append([x, y])
    return np.array(coords, dtype=np.float32)


def landmarks_to_dict(landmarks: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """
    Convierte array de landmarks a diccionario.
    """
    return {LANDMARK_NAMES[i]: tuple(landmarks[i]) for i in range(15)}


def visualize_landmarks(
    image: np.ndarray,
    landmarks: np.ndarray,
    predicted: Optional[np.ndarray] = None,
    title: str = "",
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Visualiza imagen con landmarks superpuestos.

    Args:
        image: Imagen como array numpy (H, W) o (H, W, 3)
        landmarks: Coordenadas ground truth (15, 2) en pixeles
        predicted: Coordenadas predichas opcional (15, 2) en pixeles
        title: Titulo del grafico
        figsize: Tamano de figura
        save_path: Ruta para guardar (opcional)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Mostrar imagen
    if len(image.shape) == 2:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)

    # Colores por tipo de landmark
    colors_gt = {
        'axis': 'cyan',       # L1, L2 (eje central)
        'central': 'lime',    # L9, L10, L11 (puntos centrales)
        'bilateral': 'yellow', # Pares bilaterales
        'corner': 'magenta'   # L12, L13, L14, L15
    }

    def get_color(idx: int) -> str:
        if idx in [0, 1]:
            return colors_gt['axis']
        elif idx in CENTRAL_LANDMARKS:
            return colors_gt['central']
        elif idx in [11, 12, 13, 14]:
            return colors_gt['corner']
        else:
            return colors_gt['bilateral']

    # Dibujar landmarks GT
    for i, (x, y) in enumerate(landmarks):
        color = get_color(i)
        ax.scatter(x, y, c=color, s=100, marker='o', edgecolors='black', linewidth=1)
        ax.annotate(LANDMARK_NAMES[i], (x + 5, y - 5), color='white',
                   fontsize=8, fontweight='bold')

    # Dibujar eje L1-L2
    ax.plot([landmarks[0, 0], landmarks[1, 0]],
            [landmarks[0, 1], landmarks[1, 1]],
            'c--', linewidth=2, alpha=0.7, label='Eje L1-L2')

    # Dibujar predicciones si existen
    if predicted is not None:
        for i, (x, y) in enumerate(predicted):
            ax.scatter(x, y, c='red', s=60, marker='x', linewidth=2)
        ax.scatter([], [], c='red', marker='x', label='Prediccion')

    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def compute_statistics(df: pd.DataFrame) -> Dict:
    """
    Calcula estadisticas del dataset.
    """
    stats = {
        'total_samples': len(df),
        'by_category': df['category'].value_counts().to_dict(),
    }

    # Estadisticas por landmark
    landmark_stats = {}
    for i in range(1, 16):
        x_col = f'L{i}_x'
        y_col = f'L{i}_y'
        landmark_stats[f'L{i}'] = {
            'x_mean': df[x_col].mean(),
            'x_std': df[x_col].std(),
            'y_mean': df[y_col].mean(),
            'y_std': df[y_col].std(),
        }

    stats['landmark_stats'] = landmark_stats

    return stats


def compute_symmetry_error(landmarks: np.ndarray) -> Dict[str, float]:
    """
    Calcula error de simetria para cada par bilateral.

    Args:
        landmarks: Array (15, 2) de coordenadas

    Returns:
        Diccionario con error de simetria por par
    """
    # Centro del eje
    L1, L2 = landmarks[0], landmarks[1]
    eje = L2 - L1
    eje_len = np.linalg.norm(eje)

    if eje_len < 1e-6:
        return {}

    eje_unit = eje / eje_len
    perp = np.array([-eje_unit[1], eje_unit[0]])  # Perpendicular

    errors = {}
    for left_idx, right_idx in SYMMETRIC_PAIRS:
        left = landmarks[left_idx]
        right = landmarks[right_idx]

        # Distancia perpendicular desde L1
        d_left = np.abs(np.dot(left - L1, perp))
        d_right = np.abs(np.dot(right - L1, perp))

        errors[f'L{left_idx+1}-L{right_idx+1}'] = np.abs(d_left - d_right)

    return errors
