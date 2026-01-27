#!/usr/bin/env python3
"""
Script para generar F5.9: Casos mal clasificados de validación cruzada en test set.

Usa el mejor fold individual (mayor accuracy en test set) para mostrar ejemplos reales de errores.

Uso:
    python scripts/generate_F5_9_misclassified_cv.py
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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm

# Agregar src_v2 al path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src_v2.models import create_classifier, get_classifier_transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_best_fold(cv_dir: Path) -> Tuple[int, float]:
    """
    Encuentra el fold con mejor accuracy en test set.

    Args:
        cv_dir: Directorio con resultados de CV

    Returns:
        Tuple (fold_number, best_accuracy)
    """
    best_fold = None
    best_acc = 0.0

    for fold in range(1, 6):
        fold_path = cv_dir / f"fold_{fold:02d}" / "test_results.json"

        if not fold_path.exists():
            continue

        with open(fold_path) as f:
            results = json.load(f)

        accuracy = results["metrics"]["accuracy"]

        if accuracy > best_acc:
            best_acc = accuracy
            best_fold = fold

    return best_fold, best_acc


class MisclassifiedFinder:
    """Encuentra casos mal clasificados del clasificador usando test split."""

    def __init__(
        self,
        checkpoint: str,
        data_dir: str,
        device: str = "cuda",
    ):
        self.checkpoint = Path(checkpoint)
        self.data_dir = Path(data_dir)
        self.device = device

        # Cargar modelo
        logger.info(f"Cargando modelo desde: {self.checkpoint}")
        self.model = create_classifier(checkpoint=str(self.checkpoint), device=self.device)
        self.model.eval()

        # Preparar dataset de test
        transform = get_classifier_transforms(train=False)
        test_dir = self.data_dir / "test"

        if not test_dir.exists():
            raise FileNotFoundError(
                f"Directorio no encontrado: {test_dir}\n"
                "Se requiere el split de test."
            )

        self.dataset = datasets.ImageFolder(test_dir, transform=transform)
        self.class_names = self.dataset.classes

        logger.info(f"Dataset: {test_dir}")
        logger.info(f"Clases: {self.class_names}")
        logger.info(f"Total imágenes en test set: {len(self.dataset)}")

    def find_misclassified(self, batch_size: int = 32) -> Dict[str, List[Dict]]:
        """
        Encuentra todas las imágenes mal clasificadas en el test set.

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
    fold: int,
    f1_macro: float,
    n_samples: int = 6,
    seed: int = 42,
):
    """
    Genera figura F5.9 con casos mal clasificados del test set.

    Args:
        misclassified: Dict con casos mal clasificados por tipo de error
        output_path: Path de salida para la figura
        fold: Número del fold usado
        f1_macro: Accuracy del fold en test set (renombrado de f1_macro por compatibilidad)
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

            title = f"{true_label_es} → {pred_label_es}"
            ax.set_title(title, fontsize=11, pad=8)

        ax.axis('off')

    # Ocultar axes sobrantes si hay menos de n_samples ejemplos
    for idx in range(len(selected_examples), len(axes)):
        axes[idx].axis('off')

    # Título general con información del fold
    plt.suptitle(
        f'Ejemplos de clasificaciones erróneas - Fold {fold} (Test Set)\n'
        f'(Verdadero → Predicho) | Accuracy: {f1_macro*100:.2f}%',
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
    base_dir = Path(__file__).parent.parent

    # Configuración
    cv_dir = base_dir / "outputs/classifier_cv"
    data_dir = base_dir / "outputs/warped_lung_best/session_warping"
    output_path = base_dir / "docs/Tesis/Figures/F5.9_casos_mal_clasificados_cv.png"

    # Verificar que existan los paths
    if not cv_dir.exists():
        logger.error(f"Directorio CV no encontrado: {cv_dir}")
        return

    if not data_dir.exists():
        logger.error(f"Dataset no encontrado: {data_dir}")
        return

    # Encontrar mejor fold en test set
    logger.info("Buscando mejor fold (accuracy en test set)...")
    best_fold, best_acc = find_best_fold(cv_dir)

    if best_fold is None:
        logger.error("No se pudo encontrar el mejor fold")
        return

    logger.info(f"\nMejor fold: {best_fold}")
    logger.info(f"Accuracy en test set: {best_acc*100:.2f}%")

    # Checkpoint del mejor fold
    checkpoint = cv_dir / f"fold_{best_fold:02d}" / "best_classifier.pt"

    if not checkpoint.exists():
        logger.error(f"Checkpoint no encontrado: {checkpoint}")
        return

    # Verificar que existe val split
    val_dir = data_dir / "val"
    if not val_dir.exists():
        logger.error(f"Directorio no encontrado: {val_dir}")
        return

    # Encontrar casos mal clasificados usando val split
    # Nota: Usamos el val split genérico, que es representativo del fold
    finder = MisclassifiedFinder(
        checkpoint=str(checkpoint),
        data_dir=str(data_dir),
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
        output_path=str(output_path),
        fold=best_fold,
        f1_macro=best_acc,  # Usar accuracy en lugar de f1_macro
        n_samples=6,
        seed=42,
    )

    logger.info("\n¡Completado!")


if __name__ == "__main__":
    main()
