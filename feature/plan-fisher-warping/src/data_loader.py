"""
Módulo de carga de datos para el pipeline Fisher-Warping.

MATEMÁTICAS FUNDAMENTALES
=========================

1. APLANAMIENTO (Flatten)
-------------------------
Cada imagen de tamaño H × W se convierte en un vector de H·W elementos.

    Imagen(224, 224) → Vector(50176,)

Esto es necesario porque PCA y otros algoritmos lineales operan sobre
vectores, no sobre matrices 2D. Conceptualmente, cada pixel se convierte
en una "característica" o dimensión del espacio de datos.

    pixel[0,0], pixel[0,1], ..., pixel[H-1,W-1]

El orden del aplanamiento debe ser consistente para todas las imágenes.
Numpy usa row-major order (C-style): primero recorre filas, luego columnas.

2. NORMALIZACIÓN DE INTENSIDAD
------------------------------
Las imágenes en escala de grises tienen valores en [0, 255].
Para PCA conviene normalizar a [0, 1]:

    pixel_normalizado = pixel / 255.0

Esto evita problemas numéricos y hace que todos los pixeles contribuyan
en la misma escala.

3. ESTRUCTURA DE DATOS
----------------------
Para N imágenes de dimensión D, creamos una matriz X de tamaño (N, D):

    X = | img1[0]  img1[1]  ...  img1[D-1] |  <- fila 1: imagen 1
        | img2[0]  img2[1]  ...  img2[D-1] |  <- fila 2: imagen 2
        |   ...      ...           ...     |
        | imgN[0]  imgN[1]  ...  imgN[D-1] |  <- fila N: imagen N

Cada fila es una imagen aplanada (un punto en el espacio de 50,176 dimensiones).

4. MÁSCARA PARA IMÁGENES WARPED
-------------------------------
Las imágenes warped tienen fondo negro (valor 0) que no contiene información
médica relevante. Para evitar que PCA desperdicie componentes en el fondo:

    máscara = (imagen > umbral)  # típicamente umbral = 0

Solo usamos pixeles donde la máscara es True. Esto reduce dimensiones:

    50,176 pixeles → ~23,000 pixeles (solo contenido médico)

La máscara debe ser COMÚN a todas las imágenes para mantener consistencia.
Calculamos la intersección de máscaras de todas las imágenes.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DatasetSplit:
    """
    Contenedor para un split del dataset (train, val, o test).

    Atributos:
        X: Matriz de datos (N, D) donde N=muestras, D=dimensiones
        y: Vector de etiquetas (N,) con valores numéricos
        ids: Lista de identificadores de imagen
        paths: Lista de rutas a las imágenes
    """
    X: np.ndarray
    y: np.ndarray
    ids: List[str]
    paths: List[str]


@dataclass
class Dataset:
    """
    Dataset completo con los tres splits.

    Contiene train, val, test y metadatos como:
    - image_shape: Dimensiones originales de la imagen (H, W)
    - n_features: Número de características (H * W)
    - class_names: Nombres de las clases
    - label_map: Mapeo de nombre de clase a índice numérico
    """
    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit
    image_shape: Tuple[int, int]
    n_features: int
    class_names: List[str]
    label_map: Dict[str, int]
    mask: Optional[np.ndarray] = None  # Máscara de pixeles válidos (para warped)
    mask_indices: Optional[np.ndarray] = None  # Índices de pixeles en la máscara


def compute_common_mask(
    csv_path: Path,
    base_path: Path,
    threshold: int = 0,
    sample_size: int = 100,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula la máscara común de contenido válido para imágenes warped.

    MATEMÁTICAS:
    Para cada imagen i, creamos una máscara binaria:
        mask_i = (imagen_i > threshold)

    La máscara común es la INTERSECCIÓN de todas las máscaras:
        mask_common = mask_1 AND mask_2 AND ... AND mask_N

    Esto asegura que solo usamos pixeles que tienen contenido en TODAS las imágenes.

    Args:
        csv_path: Ruta al CSV de splits
        base_path: Ruta base del proyecto
        threshold: Umbral para considerar pixel como contenido (default: 0)
        sample_size: Número de imágenes a usar para calcular la máscara
        verbose: Si True, imprime progreso

    Returns:
        Tupla (mask_2d, mask_indices) donde:
        - mask_2d: Máscara 2D booleana (H, W)
        - mask_indices: Índices de pixeles válidos en el vector aplanado
    """
    df = pd.read_csv(csv_path)

    # Usar solo imágenes de training para calcular la máscara
    df_train = df[df['split'] == 'train']

    # Tomar una muestra si hay muchas imágenes
    if len(df_train) > sample_size:
        df_sample = df_train.sample(n=sample_size, random_state=42)
    else:
        df_sample = df_train

    if verbose:
        print(f"Calculando máscara común con {len(df_sample)} imágenes...")

    # Inicializar máscara con todos True
    first_path = base_path / df_sample.iloc[0]['path']
    first_img = cv2.imread(str(first_path), cv2.IMREAD_GRAYSCALE)
    mask_common = np.ones(first_img.shape, dtype=bool)

    # Calcular intersección de máscaras
    for _, row in df_sample.iterrows():
        img_path = base_path / row['path']
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask_i = img > threshold
        mask_common = mask_common & mask_i

    n_valid = np.sum(mask_common)
    total = mask_common.size
    pct = 100 * n_valid / total

    if verbose:
        print(f"  Pixeles válidos: {n_valid}/{total} ({pct:.1f}%)")

    # Obtener índices de pixeles válidos
    mask_indices = np.where(mask_common.flatten())[0]

    return mask_common, mask_indices


def load_image_as_vector(
    path: Path,
    normalize: bool = True,
    mask_indices: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Carga una imagen y la convierte a vector aplanado.

    PROCESO:
    1. Leer imagen en escala de grises
    2. Aplanar a vector 1D
    3. Aplicar máscara si se proporciona (solo pixeles válidos)
    4. Normalizar a [0, 1] (opcional)

    Args:
        path: Ruta a la imagen
        normalize: Si True, divide por 255 para normalizar a [0, 1]
        mask_indices: Índices de pixeles a extraer (si None, usa todos)

    Returns:
        Vector 1D de tipo float32

    EJEMPLO con máscara:
        Sin máscara: imagen 224×224 → vector 50176
        Con máscara: imagen 224×224 → vector ~22000 (solo pixeles de contenido)
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar: {path}")

    # Aplanar: (H, W) -> (H*W,)
    vector = img.flatten().astype(np.float32)

    # Aplicar máscara si se proporciona
    if mask_indices is not None:
        vector = vector[mask_indices]

    # Normalizar intensidades a [0, 1]
    if normalize:
        vector = vector / 255.0

    return vector


def create_label_map(
    classes: List[str],
    scenario: str = "3class"
) -> Dict[str, int]:
    """
    Crea mapeo de nombres de clase a índices numéricos.

    ESCENARIOS:

    1. 3 clases (scenario="3class"):
       COVID -> 0, Normal -> 1, Viral_Pneumonia -> 2

    2. 2 clases (scenario="2class"):
       COVID y Viral_Pneumonia -> 0 (Enfermo)
       Normal -> 1 (Normal/Sano)

       Esto agrupa las patologías pulmonares en una sola clase,
       simplificando el problema a: ¿está enfermo o normal?

    Args:
        classes: Lista de nombres de clase originales
        scenario: "3class" o "2class"

    Returns:
        Diccionario {nombre_clase: índice}
    """
    if scenario == "3class":
        # Mapeo directo: cada clase a su índice
        return {cls: i for i, cls in enumerate(sorted(classes))}

    elif scenario == "2class":
        # Agrupar: COVID y Viral_Pneumonia = Enfermo (0), Normal = Sano (1)
        label_map = {}
        for cls in classes:
            if cls == "Normal":
                label_map[cls] = 1  # Sano
            else:
                label_map[cls] = 0  # Enfermo (COVID o Viral Pneumonia)
        return label_map

    else:
        raise ValueError(f"Escenario no reconocido: {scenario}")


def get_class_names(scenario: str = "3class") -> List[str]:
    """
    Obtiene los nombres de clase según el escenario.

    Args:
        scenario: "3class" o "2class"

    Returns:
        Lista de nombres de clase
    """
    if scenario == "3class":
        return ["COVID", "Normal", "Viral_Pneumonia"]
    elif scenario == "2class":
        return ["Enfermo", "Normal"]
    else:
        raise ValueError(f"Escenario no reconocido: {scenario}")


def load_split_from_csv(
    csv_path: Path,
    base_path: Path,
    split: str,
    label_map: Dict[str, int],
    mask_indices: Optional[np.ndarray] = None,
    verbose: bool = True
) -> DatasetSplit:
    """
    Carga un split del dataset desde CSV.

    PROCESO:
    1. Filtrar filas del CSV por split (train/val/test)
    2. Cargar cada imagen como vector (opcionalmente con máscara)
    3. Aplicar mapeo de etiquetas
    4. Retornar como DatasetSplit

    Args:
        csv_path: Ruta al CSV con columnas [image_id, class, split, path]
        base_path: Ruta base para resolver paths relativos
        split: "train", "val" o "test"
        label_map: Mapeo de clase a índice
        mask_indices: Índices de pixeles a usar (None = todos)
        verbose: Si True, imprime progreso

    Returns:
        DatasetSplit con X, y, ids, paths
    """
    df = pd.read_csv(csv_path)
    df_split = df[df['split'] == split].copy()

    if len(df_split) == 0:
        raise ValueError(f"No hay datos para split '{split}' en {csv_path}")

    # Preparar listas
    vectors = []
    labels = []
    ids = []
    paths = []

    n_total = len(df_split)

    for i, (_, row) in enumerate(df_split.iterrows()):
        img_path = base_path / row['path']

        # Cargar imagen como vector (con máscara si se proporciona)
        vector = load_image_as_vector(img_path, mask_indices=mask_indices)
        vectors.append(vector)

        # Mapear etiqueta
        label = label_map[row['class']]
        labels.append(label)

        ids.append(row['image_id'])
        paths.append(str(row['path']))

        # Progreso
        if verbose and (i + 1) % 500 == 0:
            print(f"  {split}: {i + 1}/{n_total} imágenes cargadas...")

    if verbose:
        print(f"  {split}: {n_total} imágenes cargadas.")

    # Convertir a numpy arrays
    X = np.array(vectors, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    return DatasetSplit(X=X, y=y, ids=ids, paths=paths)


def load_dataset(
    csv_path: Path,
    base_path: Path,
    scenario: str = "2class",
    use_mask: bool = False,
    verbose: bool = True
) -> Dataset:
    """
    Carga el dataset completo desde CSV.

    FLUJO COMPLETO:
    1. Crear mapeo de etiquetas según escenario
    2. (Opcional) Calcular máscara común para imágenes warped
    3. Cargar train, val, test por separado
    4. Extraer metadatos (shape, n_features, etc.)
    5. Retornar Dataset completo

    IMPORTANTE sobre el escenario:
    - "3class": Clasifica COVID vs Normal vs Viral_Pneumonia
    - "2class": Clasifica Enfermo (COVID + Viral) vs Sano (Normal)

    IMPORTANTE sobre la máscara:
    - use_mask=True: Solo usa pixeles de contenido (para imágenes warped)
    - use_mask=False: Usa todos los pixeles (para imágenes originales)

    Args:
        csv_path: Ruta al CSV de splits
        base_path: Ruta base del proyecto
        scenario: "3class" o "2class"
        use_mask: Si True, calcula y aplica máscara de contenido
        verbose: Si True, imprime progreso

    Returns:
        Dataset completo con train, val, test y metadatos

    Ejemplo de uso:
        >>> csv_path = Path("results/metrics/01_full_balanced_3class_warped.csv")
        >>> dataset = load_dataset(csv_path, base_path, scenario="2class", use_mask=True)
        >>> print(f"Train shape: {dataset.train.X.shape}")
        Train shape: (5040, 22000)  # Reducido por máscara
    """
    # Determinar clases originales del CSV
    df = pd.read_csv(csv_path)
    original_classes = sorted(df['class'].unique().tolist())

    if verbose:
        print(f"Cargando dataset: {csv_path.name}")
        print(f"  Escenario: {scenario}")
        print(f"  Usar máscara: {use_mask}")
        print(f"  Clases originales: {original_classes}")

    # Calcular máscara si se solicita
    mask_2d = None
    mask_indices = None
    if use_mask:
        mask_2d, mask_indices = compute_common_mask(csv_path, base_path, verbose=verbose)

    # Crear mapeo de etiquetas
    label_map = create_label_map(original_classes, scenario)
    class_names = get_class_names(scenario)

    if verbose:
        print(f"  Mapeo de etiquetas: {label_map}")
        print(f"  Nombres de clase: {class_names}")
        print()

    # Cargar cada split (con máscara si se calculó)
    train = load_split_from_csv(csv_path, base_path, "train", label_map, mask_indices, verbose)
    val = load_split_from_csv(csv_path, base_path, "val", label_map, mask_indices, verbose)
    test = load_split_from_csv(csv_path, base_path, "test", label_map, mask_indices, verbose)

    # Extraer metadatos
    n_features = train.X.shape[1]

    # Para image_shape, usar la original (224, 224) aunque hayamos aplicado máscara
    # Esto es necesario para poder reconstruir/visualizar
    first_path = base_path / df.iloc[0]['path']
    first_img = cv2.imread(str(first_path), cv2.IMREAD_GRAYSCALE)
    image_shape = first_img.shape

    if verbose:
        print()
        print(f"Dataset cargado:")
        print(f"  Shape de imagen original: {image_shape}")
        print(f"  Número de características: {n_features}")
        if use_mask:
            print(f"    (reducido de {image_shape[0]*image_shape[1]} por máscara)")
        print(f"  Train: {train.X.shape[0]} muestras")
        print(f"  Val: {val.X.shape[0]} muestras")
        print(f"  Test: {test.X.shape[0]} muestras")

        # Distribución de clases (mejorada)
        print(f"\n  Distribución de clases (train):")
        for cls_idx in sorted(set(label_map.values())):
            count = np.sum(train.y == cls_idx)
            cls_name = class_names[cls_idx]
            print(f"    {cls_name} ({cls_idx}): {count}")

    return Dataset(
        train=train,
        val=val,
        test=test,
        image_shape=image_shape,
        n_features=n_features,
        class_names=class_names,
        label_map=label_map,
        mask=mask_2d,
        mask_indices=mask_indices
    )


def get_class_distribution(split: DatasetSplit, class_names: List[str]) -> Dict[str, int]:
    """
    Calcula la distribución de clases en un split.

    Args:
        split: DatasetSplit a analizar
        class_names: Nombres de las clases

    Returns:
        Dict {nombre_clase: conteo}
    """
    unique, counts = np.unique(split.y, return_counts=True)
    return {class_names[i]: int(counts[j]) for j, i in enumerate(unique)}


if __name__ == "__main__":
    # Ejemplo de uso
    base_path = Path(__file__).parent.parent.parent.parent
    csv_path = Path(__file__).parent.parent / "results" / "metrics" / "01_full_balanced_3class_warped.csv"

    print("="*60)
    print("PRUEBA DE DATA LOADER")
    print("="*60)
    print()

    # Cargar dataset en escenario de 2 clases
    dataset = load_dataset(csv_path, base_path, scenario="2class", verbose=True)

    print()
    print("="*60)
    print("VERIFICACIÓN")
    print("="*60)
    print(f"\nForma de X_train: {dataset.train.X.shape}")
    print(f"Forma de y_train: {dataset.train.y.shape}")
    print(f"Rango de valores en X: [{dataset.train.X.min():.3f}, {dataset.train.X.max():.3f}]")
    print(f"Valores únicos en y: {np.unique(dataset.train.y)}")
