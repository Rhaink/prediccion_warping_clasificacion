"""
Modulo de visualizacion para el pipeline Fisher-Warping.

Genera figuras profesionales para publicacion cientifica.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random


def set_publication_style():
    """Configura matplotlib para figuras de publicacion."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


def load_image_grayscale(path: Path) -> np.ndarray:
    """Carga una imagen en escala de grises."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar: {path}")
    return img


def get_matching_images(
    warped_dir: Path,
    original_dirs: Dict[str, Path],
    classes: List[str],
    n_samples: int = 1,
    seed: int = 42
) -> Dict[str, List[Tuple[Path, Path]]]:
    """
    Encuentra pares de imagenes (original, warped) para cada clase.

    Args:
        warped_dir: Directorio con imagenes warped (ej: outputs/warped_dataset/train)
        original_dirs: Dict con rutas de originales por clase
        classes: Lista de clases a procesar
        n_samples: Numero de muestras por clase
        seed: Semilla para reproducibilidad

    Returns:
        Dict con lista de tuplas (original_path, warped_path) por clase
    """
    random.seed(seed)
    result = {}

    for cls in classes:
        warped_class_dir = warped_dir / cls
        original_class_dir = original_dirs[cls]

        # Listar archivos warped
        warped_files = list(warped_class_dir.glob("*_warped.png"))

        # Seleccionar n_samples aleatorios
        if len(warped_files) > n_samples:
            warped_files = random.sample(warped_files, n_samples)

        pairs = []
        for warped_path in warped_files:
            # Obtener nombre original (remover _warped)
            original_name = warped_path.stem.replace("_warped", "") + ".png"
            original_path = original_class_dir / original_name

            if original_path.exists():
                pairs.append((original_path, warped_path))

        result[cls] = pairs

    return result


def create_comparison_panel(
    image_pairs: Dict[str, List[Tuple[Path, Path]]],
    classes: List[str],
    class_labels: Optional[Dict[str, str]] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (10, 7)
) -> plt.Figure:
    """
    Crea un panel comparativo Original vs Warped.

    Formato: 2 filas x 3 columnas
    - Fila 1: Originales
    - Fila 2: Warped

    Args:
        image_pairs: Dict con pares (original, warped) por clase
        classes: Lista de clases en orden
        class_labels: Dict con etiquetas en espanol por clase (opcional)
        output_path: Ruta para guardar (opcional)
        figsize: Tamano de la figura

    Returns:
        Figura de matplotlib
    """
    set_publication_style()

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Etiquetas de fila en espanol
    row_labels = ['(a) Original', '(b) Normalizada']

    # Etiquetas de clase en espanol por defecto
    if class_labels is None:
        class_labels = {
            "COVID": "COVID",
            "Normal": "Normal",
            "Viral_Pneumonia": "Neumonia",
        }

    for col, cls in enumerate(classes):
        pairs = image_pairs.get(cls, [])
        if not pairs:
            continue

        original_path, warped_path = pairs[0]  # Primera muestra

        # Cargar imagenes
        img_original = load_image_grayscale(original_path)
        img_warped = load_image_grayscale(warped_path)

        # Fila 0: Original
        axes[0, col].imshow(img_original, cmap='gray')
        axes[0, col].set_title(class_labels.get(cls, cls), fontweight='bold')
        axes[0, col].axis('off')

        # Fila 1: Warped
        axes[1, col].imshow(img_warped, cmap='gray')
        axes[1, col].axis('off')

    # Agregar etiquetas de fila en el lado izquierdo
    for row, label in enumerate(row_labels):
        axes[row, 0].annotate(
            label,
            xy=(-0.15, 0.5),
            xycoords='axes fraction',
            fontsize=11,
            ha='right',
            va='center',
            fontweight='bold'
        )

    plt.tight_layout()

    # Guardar si se especifica ruta
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Figura guardada en: {output_path}")

    return fig


def generate_manual_dataset_panel(
    base_path: Path,
    output_path: Path,
    seed: int = 42
) -> plt.Figure:
    """
    Genera panel para el dataset manual (warped_dataset).

    Args:
        base_path: Ruta base del proyecto
        output_path: Ruta donde guardar la figura
        seed: Semilla para reproducibilidad
    """
    warped_dir = base_path / "outputs" / "warped_dataset" / "train"

    original_dirs = {
        "COVID": base_path / "data" / "dataset" / "COVID",
        "Normal": base_path / "data" / "dataset" / "Normal",
        "Viral_Pneumonia": base_path / "data" / "dataset" / "Viral_Pneumonia",
    }

    classes = ["COVID", "Normal", "Viral_Pneumonia"]

    pairs = get_matching_images(warped_dir, original_dirs, classes, n_samples=1, seed=seed)

    return create_comparison_panel(
        pairs,
        classes,
        output_path=output_path
    )


def generate_full_dataset_panel(
    base_path: Path,
    output_path: Path,
    seed: int = 42
) -> plt.Figure:
    """
    Genera panel para el dataset completo (full_warped_dataset).

    Args:
        base_path: Ruta base del proyecto
        output_path: Ruta donde guardar la figura
        seed: Semilla para reproducibilidad
    """
    warped_dir = base_path / "outputs" / "full_warped_dataset" / "train"

    radiography_base = base_path / "data" / "dataset" / "COVID-19_Radiography_Dataset"

    original_dirs = {
        "COVID": radiography_base / "COVID" / "images",
        "Normal": radiography_base / "Normal" / "images",
        "Viral_Pneumonia": radiography_base / "Viral Pneumonia" / "images",
    }

    classes = ["COVID", "Normal", "Viral_Pneumonia"]

    pairs = get_matching_images(warped_dir, original_dirs, classes, n_samples=1, seed=seed)

    return create_comparison_panel(
        pairs,
        classes,
        output_path=output_path
    )


def generate_multiple_panels(
    base_path: Path,
    output_dir: Path,
    n_panels: int = 6,
    start_seed: int = 42
):
    """
    Genera multiples paneles con diferentes imagenes.

    Args:
        base_path: Ruta base del proyecto
        output_dir: Directorio de salida
        n_panels: Numero de paneles a generar
        start_seed: Semilla inicial (se incrementa para cada panel)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generando {n_panels} paneles de cada dataset...")

    for i in range(n_panels):
        seed = start_seed + i
        panel_num = i + 1

        # Panel dataset manual
        print(f"\n  Panel {panel_num}/{n_panels} - Dataset manual (seed={seed})...")
        generate_manual_dataset_panel(
            base_path,
            output_dir / f"panel_manual_{panel_num:02d}.png",
            seed=seed
        )

        # Panel dataset completo
        print(f"  Panel {panel_num}/{n_panels} - Dataset completo (seed={seed})...")
        generate_full_dataset_panel(
            base_path,
            output_dir / f"panel_full_{panel_num:02d}.png",
            seed=seed
        )

    print(f"\n{n_panels * 2} paneles generados en: {output_dir}")


if __name__ == "__main__":
    # Ruta base del proyecto
    base_path = Path(__file__).parent.parent.parent.parent
    results_figures = Path(__file__).parent.parent / "results" / "figures"

    print("Generando paneles comparativos...")
    print(f"Base path: {base_path}")
    print(f"Output dir: {results_figures}")

    # Generar 6 paneles de cada dataset
    generate_multiple_panels(base_path, results_figures, n_panels=6)

    print("\nPaneles generados exitosamente.")
