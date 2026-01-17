#!/usr/bin/env python3
"""
Script para generar F5.9: Casos mal clasificados del clasificador SAHS.

Usa predicciones reales del clasificador para mostrar ejemplos verdaderos de errores,
con imágenes warped+SAHS y sin repeticiones.

Uso:
    python scripts/generate_F5_9_misclassified.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

# Agregar src_v2 al path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src_v2.models import create_classifier, get_classifier_transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MisclassifiedFinder:
    """Encuentra casos mal clasificados del clasificador."""

    def __init__(
        self,
        checkpoint: str,
        data_dir: str,
        split: str = "test",
        device: str = "cuda",
    ):
        self.checkpoint = Path(checkpoint)
        self.data_dir = Path(data_dir)
        self.split = split
        self.device = device

        # Cargar modelo
        logger.info(f"Cargando modelo desde: {self.checkpoint}")
        self.model = create_classifier(checkpoint=str(self.checkpoint), device=self.device)
        self.model.eval()

        # Preparar dataset
        transform = get_classifier_transforms(train=False)
        split_dir = self.data_dir / split

        self.dataset = datasets.ImageFolder(split_dir, transform=transform)
        self.class_names = self.dataset.classes

        logger.info(f"Dataset: {split_dir}")
        logger.info(f"Clases: {self.class_names}")
        logger.info(f"Total imágenes: {len(self.dataset)}")

    def find_misclassified(self, batch_size: int = 32) -> Dict[str, List[Dict]]:
        """
        Encuentra todas las imágenes mal clasificadas.

        Returns:
            Dict con estructura:
            {
                "COVID->Normal": [{"path": ..., "confidence": 0.85, "true_label": "COVID", "pred_label": "Normal"}, ...],
                ...
            }
        """
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        misclassified = {}
        current_idx = 0

        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Evaluando"):
                images = images.to(self.device)
                labels_np = labels.cpu().numpy()

                # Predicciones
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                confidences, predictions = torch.max(probs, dim=1)

                predictions_np = predictions.cpu().numpy()
                confidences_np = confidences.cpu().numpy()

                # Identificar errores
                for idx in range(len(labels_np)):
                    true_label = labels_np[idx]
                    pred_label = predictions_np[idx]

                    # Obtener path de la imagen original
                    img_path, _ = self.dataset.samples[current_idx]
                    current_idx += 1

                    if true_label != pred_label:
                        true_class = self.class_names[true_label]
                        pred_class = self.class_names[pred_label]

                        error_type = f"{true_class}→{pred_class}"

                        if error_type not in misclassified:
                            misclassified[error_type] = []

                        misclassified[error_type].append({
                            "path": img_path,
                            "confidence": float(confidences_np[idx]),
                            "true_label": true_class,
                            "pred_label": pred_class,
                        })

        return misclassified


def generate_figure(
    misclassified: Dict[str, List[Dict]],
    output_path: str,
    n_samples: int = 6,
    seed: int = 42,
):
    """
    Genera figura F5.9 con casos mal clasificados.

    Args:
        misclassified: Dict con casos mal clasificados por tipo de error
        output_path: Path de salida para la figura
        n_samples: Número de ejemplos a mostrar
        seed: Semilla para selección aleatoria
    """
    random.seed(seed)
    np.random.seed(seed)

    # Configurar figura (3x2 grid para 6 ejemplos)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    # Seleccionar ejemplos variados de diferentes tipos de error
    selected_examples = []
    error_types_available = sorted(misclassified.keys())

    logger.info("\nCasos mal clasificados disponibles:")
    for error_type in error_types_available:
        logger.info(f"  {error_type}: {len(misclassified[error_type])} casos")

    # Distribuir ejemplos entre los tipos de error disponibles
    for error_type in error_types_available:
        if len(selected_examples) >= n_samples:
            break

        cases = misclassified[error_type]
        if cases:
            # Tomar hasta 2 ejemplos de cada tipo de error para variedad
            n_to_take = min(2, len(cases), n_samples - len(selected_examples))
            selected = random.sample(cases, n_to_take)
            selected_examples.extend(selected)

    # Si aún faltan ejemplos, rellenar con los tipos de error más comunes
    while len(selected_examples) < n_samples and error_types_available:
        # Encontrar el tipo de error con más casos
        error_type = max(error_types_available, key=lambda et: len(misclassified[et]))
        remaining_cases = [
            c for c in misclassified[error_type]
            if c not in selected_examples
        ]

        if remaining_cases:
            selected_examples.append(random.choice(remaining_cases))
        else:
            error_types_available.remove(error_type)

    # Verificar que no haya duplicados
    paths_used = set()
    unique_examples = []
    for example in selected_examples:
        if example["path"] not in paths_used:
            unique_examples.append(example)
            paths_used.add(example["path"])

    selected_examples = unique_examples[:n_samples]

    logger.info(f"\nEjemplos seleccionados: {len(selected_examples)}")

    # Mapeo de nombres de clases a español
    labels_es = {
        "COVID": "COVID-19",
        "Normal": "Normal",
        "Viral_Pneumonia": "Neumonía Viral",
    }

    # Plotear ejemplos
    for idx, example in enumerate(selected_examples):
        ax = axes[idx]

        # Cargar imagen
        img = cv2.imread(example["path"], cv2.IMREAD_GRAYSCALE)

        if img is not None:
            ax.imshow(img, cmap='gray')

            # Título con información del error
            true_label_es = labels_es.get(example["true_label"], example["true_label"])
            pred_label_es = labels_es.get(example["pred_label"], example["pred_label"])

            title = f"{true_label_es} → {pred_label_es}\nConf: {example['confidence']:.0%}"
            ax.set_title(title, fontsize=11, pad=8)

        ax.axis('off')

    # Ocultar axes sobrantes si hay menos de n_samples ejemplos
    for idx in range(len(selected_examples), len(axes)):
        axes[idx].axis('off')

    # Título general
    plt.suptitle(
        'Ejemplos de clasificaciones erróneas\n(Verdadero → Predicho)',
        fontsize=14,
        y=0.98,
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
    checkpoint = "outputs/classifier_warped_sahs_masked/best_classifier.pt"
    data_dir = "outputs/warped_lung_sahs"
    output_path = "docs/Tesis/Figures/F5.9_casos_mal_clasificados.png"

    # Verificar que existan los paths
    if not Path(checkpoint).exists():
        logger.error(f"Checkpoint no encontrado: {checkpoint}")
        return

    if not Path(data_dir).exists():
        logger.error(f"Dataset no encontrado: {data_dir}")
        return

    # Encontrar casos mal clasificados
    finder = MisclassifiedFinder(
        checkpoint=checkpoint,
        data_dir=data_dir,
        split="test",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    logger.info("\nBuscando casos mal clasificados...")
    misclassified = finder.find_misclassified(batch_size=32)

    if not misclassified:
        logger.warning("No se encontraron casos mal clasificados!")
        return

    # Generar figura
    logger.info("\nGenerando figura F5.9...")
    generate_figure(
        misclassified=misclassified,
        output_path=output_path,
        n_samples=6,
        seed=42,
    )

    logger.info("\n¡Completado!")


if __name__ == "__main__":
    main()
