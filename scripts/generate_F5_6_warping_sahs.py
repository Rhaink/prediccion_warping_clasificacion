#!/usr/bin/env python3
"""
Script para generar F5.6: Ejemplos de normalización geométrica con SAHS.

Muestra un grid de 3x4 con ejemplos de imágenes warped + SAHS por clase.

Uso:
    python scripts/generate_F5_6_warping_sahs.py
"""

import logging
from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_warping_examples_figure(
    data_dir: str,
    output_path: str,
    split: str = "test",
    n_per_class: int = 4,
    seed: int = 42,
):
    """
    Genera figura F5.6 con ejemplos de normalización geométrica.

    Args:
        data_dir: Directorio con el dataset warped + SAHS
        output_path: Path de salida para la figura
        split: Split del dataset (test, train, val)
        n_per_class: Número de ejemplos por clase
        seed: Semilla para selección aleatoria
    """
    random.seed(seed)
    np.random.seed(seed)

    data_path = Path(data_dir) / split

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {data_path}")

    # Configuración de clases
    classes = ['COVID', 'Normal', 'Viral_Pneumonia']
    labels_es = {
        'COVID': 'COVID-19',
        'Normal': 'Normal',
        'Viral_Pneumonia': 'Neumonía Viral',
    }
    label_style = {
        "fontsize": 16,
        "fontweight": "bold",
        "rotation": 90,
        "va": "center",
        "ha": "center",
        "color": "black",
        "bbox": {
            "facecolor": "white",
            "edgecolor": "none",
            "linewidth": 0,
            "boxstyle": "round,pad=0.2",
        },
    }

    # Crear figura (3 filas x 4 columnas)
    fig, axes = plt.subplots(3, 4, figsize=(14, 11))

    for row, class_name in enumerate(classes):
        class_dir = data_path / class_name

        if not class_dir.exists():
            logger.warning(f"Directorio no encontrado: {class_dir}")
            continue

        # Obtener imágenes de la clase
        images = list(class_dir.glob("*.png"))

        if len(images) < n_per_class:
            logger.warning(
                f"Clase {class_name}: solo {len(images)} imágenes disponibles "
                f"(se requieren {n_per_class})"
            )

        # Seleccionar imágenes aleatoriamente
        selected_images = random.sample(images, min(n_per_class, len(images)))

        logger.info(f"Clase {class_name}: {len(selected_images)} imágenes seleccionadas")

        # Plotear imágenes
        for col in range(n_per_class):
            ax = axes[row, col]

            if col < len(selected_images):
                # Cargar y mostrar imagen
                img = cv2.imread(str(selected_images[col]), cv2.IMREAD_GRAYSCALE)

                if img is not None:
                    ax.imshow(img, cmap='gray')
                else:
                    logger.warning(f"No se pudo cargar: {selected_images[col]}")

            ax.axis('off')

        # Etiqueta de fila (clase)
        label = labels_es.get(class_name, class_name)
        # Posicionar etiqueta a la izquierda de la primera imagen
        axes[row, 0].text(
            -0.22, 0.5,
            label,
            transform=axes[row, 0].transAxes,
            **label_style,
        )

    plt.tight_layout()

    # Guardar
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"\nFigura guardada en: {output_path}")

    plt.close()


def main():
    """Función principal."""
    # Configuración
    data_dir = "outputs/warped_lung_sahs"
    output_path = "docs/Tesis/Figures/F5.6_ejemplos_warping.png"

    # Verificar que exista el dataset
    if not Path(data_dir).exists():
        logger.error(f"Dataset no encontrado: {data_dir}")
        return

    # Generar figura
    logger.info("Generando figura F5.6 con imágenes warped + SAHS...")
    generate_warping_examples_figure(
        data_dir=data_dir,
        output_path=output_path,
        split="test",
        n_per_class=4,
        seed=42,
    )

    logger.info("\n¡Completado!")


if __name__ == "__main__":
    main()
