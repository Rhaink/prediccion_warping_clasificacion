#!/usr/bin/env python3
"""
Script to generate F5.9: Misclassified cases from the SAHS classifier.

Uses real classifier predictions to show true error examples,
with warped+SAHS images and no repeats.

Usage:
    python scripts/generate_F5_9_misclassified_en.py
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
    """Finds misclassified cases from the classifier."""

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
        logger.info(f"Loading model from: {self.checkpoint}")
        self.model = create_classifier(checkpoint=str(self.checkpoint), device=self.device)
        self.model.eval()

        # Preparar dataset
        transform = get_classifier_transforms(train=False)
        split_dir = self.data_dir / split

        self.dataset = datasets.ImageFolder(split_dir, transform=transform)
        self.class_names = self.dataset.classes

        logger.info(f"Dataset: {split_dir}")
        logger.info(f"Classes: {self.class_names}")
        logger.info(f"Total images: {len(self.dataset)}")

    def find_misclassified(self, batch_size: int = 32) -> Dict[str, List[Dict]]:
        """
        Finds all misclassified images.

        Returns:
            Dict with structure:
            {
                "COVID->Normal": [{"path": ..., "confidence": 0.85, "true_label": "COVID", "pred_label": "Normal"}, ...],
                ...
            }
        """
        # Use single-process data loading to avoid multiprocessing permissions issues.
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        misclassified = {}
        current_idx = 0

        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Evaluating"):
                images = images.to(self.device)
                labels_np = labels.cpu().numpy()

                # Predictions
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                confidences, predictions = torch.max(probs, dim=1)

                predictions_np = predictions.cpu().numpy()
                confidences_np = confidences.cpu().numpy()

                # Identify errors
                for idx in range(len(labels_np)):
                    true_label = labels_np[idx]
                    pred_label = predictions_np[idx]

                    # Obtener path de la imagen original
                    img_path, _ = self.dataset.samples[current_idx]
                    current_idx += 1

                    if true_label != pred_label:
                        true_class = self.class_names[true_label]
                        pred_class = self.class_names[pred_label]

                        error_type = f"{true_class}->{pred_class}"

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
    Generates figure F5.9 with misclassified cases.

    Args:
        misclassified: Dict of misclassified cases by error type
        output_path: Output path for the figure
        n_samples: Number of examples to show
        seed: Random seed for selection
    """
    random.seed(seed)
    np.random.seed(seed)

    # Configure figure (3x2 grid for 6 examples)
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.flatten()
    title_fontsize = 13.5
    suptitle_fontsize = 16

    # Select varied examples from different error types
    selected_examples = []
    error_types_available = sorted(misclassified.keys())

    logger.info("\nAvailable misclassified cases:")
    for error_type in error_types_available:
        logger.info(f"  {error_type}: {len(misclassified[error_type])} cases")

    # Distribute examples across available error types
    for error_type in error_types_available:
        if len(selected_examples) >= n_samples:
            break

        cases = misclassified[error_type]
        if cases:
            # Take up to 2 examples per error type for variety
            n_to_take = min(2, len(cases), n_samples - len(selected_examples))
            selected = random.sample(cases, n_to_take)
            selected_examples.extend(selected)

    # If still missing examples, fill with the most common error types
    while len(selected_examples) < n_samples and error_types_available:
        # Find the error type with the most cases
        error_type = max(error_types_available, key=lambda et: len(misclassified[et]))
        remaining_cases = [
            c for c in misclassified[error_type]
            if c not in selected_examples
        ]

        if remaining_cases:
            selected_examples.append(random.choice(remaining_cases))
        else:
            error_types_available.remove(error_type)

    # Ensure no duplicates
    paths_used = set()
    unique_examples = []
    for example in selected_examples:
        if example["path"] not in paths_used:
            unique_examples.append(example)
            paths_used.add(example["path"])

    selected_examples = unique_examples[:n_samples]

    logger.info(f"\nSelected examples: {len(selected_examples)}")

    # Map class names to English labels
    labels_en = {
        "COVID": "COVID-19",
        "Normal": "Normal",
        "Viral_Pneumonia": "Viral Pneumonia",
    }

    # Plot examples
    for idx, example in enumerate(selected_examples):
        ax = axes[idx]

        # Cargar imagen
        img = cv2.imread(example["path"], cv2.IMREAD_GRAYSCALE)

        if img is not None:
            ax.imshow(img, cmap='gray')

            # Título con información del error
            true_label_en = labels_en.get(example["true_label"], example["true_label"])
            pred_label_en = labels_en.get(example["pred_label"], example["pred_label"])

            title = f"True: {true_label_en}\nPred: {pred_label_en}"
            ax.set_title(title, fontsize=title_fontsize, pad=4)

        ax.axis('off')

    # Hide extra axes if fewer than n_samples examples
    for idx in range(len(selected_examples), len(axes)):
        axes[idx].axis('off')

    # Overall title
    plt.suptitle(
        'Misclassified examples',
        fontsize=suptitle_fontsize,
        y=0.965,
    )

    # Balance title clearance with compact panel spacing.
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.03, top=0.9, wspace=0.14, hspace=0.2)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"\nFigure saved to: {output_path}")

    plt.close()


def main():
    """Main entrypoint."""
    # Configuration
    checkpoint = "outputs/classifier_warped_sahs_masked/best_classifier.pt"
    data_dir = "outputs/warped_lung_sahs"
    output_path = "docs/Tesis/Figures/F5.9_misclassified_cases.png"

    # Ensure required paths exist
    if not Path(checkpoint).exists():
        logger.error(f"Checkpoint not found: {checkpoint}")
        return

    if not Path(data_dir).exists():
        logger.error(f"Dataset not found: {data_dir}")
        return

    # Find misclassified cases
    finder = MisclassifiedFinder(
        checkpoint=checkpoint,
        data_dir=data_dir,
        split="test",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    logger.info("\nSearching for misclassified cases...")
    misclassified = finder.find_misclassified(batch_size=32)

    if not misclassified:
        logger.warning("No misclassified cases found!")
        return

    # Generate figure
    logger.info("\nGenerating figure F5.9...")
    generate_figure(
        misclassified=misclassified,
        output_path=output_path,
        n_samples=6,
        seed=42,
    )

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
