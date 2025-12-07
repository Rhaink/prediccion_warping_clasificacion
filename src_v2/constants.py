"""
Constantes centralizadas para el proyecto COVID-19 Detection via Anatomical Landmarks.

Este módulo define todas las constantes del dominio en un solo lugar
para evitar duplicación y facilitar el mantenimiento.

Uso:
    from src_v2.constants import SYMMETRIC_PAIRS, CENTRAL_LANDMARKS, DEFAULT_IMAGE_SIZE
"""

from typing import Tuple, List

# =============================================================================
# LANDMARKS - Configuración de puntos anatómicos
# =============================================================================

# Número de landmarks anatómicos
NUM_LANDMARKS: int = 15

# Número de coordenadas (x, y por cada landmark)
NUM_COORDINATES: int = NUM_LANDMARKS * 2  # 30

# Nombres de landmarks (L1 a L15)
LANDMARK_NAMES: List[str] = [
    'L1',   # Superior mediastinum
    'L2',   # Inferior mediastinum
    'L3',   # Left apex
    'L4',   # Right apex
    'L5',   # Left hilum
    'L6',   # Right hilum
    'L7',   # Left base
    'L8',   # Right base
    'L9',   # Superior central
    'L10',  # Middle central
    'L11',  # Inferior central
    'L12',  # Left upper border
    'L13',  # Right upper border
    'L14',  # Left costophrenic angle
    'L15',  # Right costophrenic angle
]

# Pares de landmarks simétricos (índices 0-based)
# L3-L4, L5-L6, L7-L8, L12-L13, L14-L15
SYMMETRIC_PAIRS: List[Tuple[int, int]] = [
    (2, 3),   # L3-L4: Apexes
    (4, 5),   # L5-L6: Hilums
    (6, 7),   # L7-L8: Bases
    (11, 12), # L12-L13: Upper borders
    (13, 14), # L14-L15: Costophrenic angles
]

# Landmarks centrales que deben estar sobre el eje L1-L2
CENTRAL_LANDMARKS: List[int] = [8, 9, 10]  # L9, L10, L11

# Landmarks del eje vertical (mediastinum)
AXIS_LANDMARKS: List[int] = [0, 1]  # L1, L2

# =============================================================================
# DIMENSIONES DE IMAGEN
# =============================================================================

# Tamaño de imagen por defecto para el modelo
DEFAULT_IMAGE_SIZE: int = 224

# Tamaño original de las imágenes del dataset
ORIGINAL_IMAGE_SIZE: int = 299

# =============================================================================
# NORMALIZACIÓN
# =============================================================================

# Valores de normalización ImageNet (para modelos preentrenados)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# =============================================================================
# CATEGORÍAS DEL DATASET
# =============================================================================

# Categorías de clasificación
CATEGORIES: List[str] = ['COVID', 'Normal', 'Viral_Pneumonia']

# Número de clases
NUM_CLASSES: int = len(CATEGORIES)

# Pesos por defecto por categoría (para balanceo de clases)
DEFAULT_CATEGORY_WEIGHTS = {
    'COVID': 2.0,           # Más peso por ser clase minoritaria importante
    'Normal': 1.0,          # Peso base
    'Viral_Pneumonia': 1.2  # Peso intermedio
}

# =============================================================================
# MODELO
# =============================================================================

# Dimensión de features del backbone ResNet-18
BACKBONE_FEATURE_DIM: int = 512

# Dimensión oculta de la cabeza de regresión
DEFAULT_HIDDEN_DIM: int = 768

# Dropout por defecto
DEFAULT_DROPOUT_RATE: float = 0.3

# =============================================================================
# ENTRENAMIENTO
# =============================================================================

# Batch size por defecto
DEFAULT_BATCH_SIZE: int = 16

# Learning rates por defecto
DEFAULT_PHASE1_LR: float = 1e-3
DEFAULT_PHASE2_BACKBONE_LR: float = 2e-5
DEFAULT_PHASE2_HEAD_LR: float = 2e-4

# Épocas por defecto
DEFAULT_PHASE1_EPOCHS: int = 15
DEFAULT_PHASE2_EPOCHS: int = 100

# =============================================================================
# LOSS
# =============================================================================

# Parámetros de Wing Loss (en píxeles, se normalizan automáticamente)
DEFAULT_WING_OMEGA: float = 10.0
DEFAULT_WING_EPSILON: float = 2.0

# Margen de simetría permitida (en píxeles)
DEFAULT_SYMMETRY_MARGIN: float = 6.0

# =============================================================================
# DATA AUGMENTATION
# =============================================================================

# Probabilidad de flip horizontal
DEFAULT_FLIP_PROB: float = 0.5

# Grados de rotación máxima
DEFAULT_ROTATION_DEGREES: float = 10.0

# Parámetros CLAHE
DEFAULT_CLAHE_CLIP_LIMIT: float = 2.0
DEFAULT_CLAHE_TILE_SIZE: int = 8
