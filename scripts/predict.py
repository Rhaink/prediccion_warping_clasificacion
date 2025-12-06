#!/usr/bin/env python3
"""
Script de inferencia para prediccion de landmarks en radiografias de torax.

Uso:
    python scripts/predict.py imagen.png
    python scripts/predict.py imagen.png --output resultado.json
    python scripts/predict.py imagen.png --visualize
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import torch
import numpy as np
from PIL import Image
import cv2

from src_v2.models.resnet_landmark import create_model


class EnsemblePredictor:
    """
    Predictor de landmarks usando ensemble de modelos.

    Uso:
        predictor = EnsemblePredictor()
        landmarks = predictor.predict("xray.png")
        # landmarks: array (15, 2) con coordenadas en pixeles

    Nota (Sesion 12):
        El ensemble optimo usa solo seed=123 y seed=456.
        El modelo seed=42 (6.75 px) degrada el ensemble de 3.79 a 4.50 px.
    """

    # Ensemble optimo: 2 modelos (sin seed=42)
    # Error: 3.79 px (Sesion 12)
    DEFAULT_MODEL_PATHS = [
        'checkpoints/session10/ensemble/seed123/final_model.pt',
        'checkpoints/session10/ensemble/seed456/final_model.pt',
    ]

    # Ensemble completo (3 modelos) - para referencia
    ALL_MODEL_PATHS = [
        'checkpoints/session10/exp4_epochs100/final_model.pt',  # seed=42 (6.75 px)
        'checkpoints/session10/ensemble/seed123/final_model.pt',  # seed=123 (4.05 px)
        'checkpoints/session10/ensemble/seed456/final_model.pt',  # seed=456 (4.04 px)
    ]

    LANDMARK_NAMES = [
        'L1 (Superior)', 'L2 (Inferior)',
        'L3 (Apex Izq)', 'L4 (Apex Der)',
        'L5 (Hilio Izq)', 'L6 (Hilio Der)',
        'L7 (Base Izq)', 'L8 (Base Der)',
        'L9 (Centro Sup)', 'L10 (Centro Med)', 'L11 (Centro Inf)',
        'L12 (Borde Izq)', 'L13 (Borde Der)',
        'L14 (Costofrenico Izq)', 'L15 (Costofrenico Der)',
    ]

    SYMMETRIC_PAIRS = [(2, 3), (4, 5), (6, 7), (11, 12), (13, 14)]

    def __init__(
        self,
        model_paths=None,
        device=None,
        use_clahe=True,
        clahe_clip_limit=2.0,
        clahe_tile_size=4,
    ):
        """
        Inicializar predictor.

        Args:
            model_paths: Lista de rutas a checkpoints. Si None, usa modelos por defecto.
            device: Dispositivo (cuda/cpu). Si None, detecta automaticamente.
            use_clahe: Aplicar CLAHE a las imagenes.
            clahe_clip_limit: Limite de contraste para CLAHE.
            clahe_tile_size: Tamano de tiles para CLAHE.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size

        # Normalizacion ImageNet
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)

        # Cargar modelos
        model_paths = model_paths or self.DEFAULT_MODEL_PATHS
        self.models = []
        for path in model_paths:
            full_path = PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
            model = self._load_model(full_path)
            self.models.append(model)

        print(f"Loaded {len(self.models)} models on {self.device}")

    def _load_model(self, checkpoint_path):
        """Cargar modelo desde checkpoint."""
        model = create_model(
            pretrained=True,
            freeze_backbone=True,
            dropout_rate=0.3,
            hidden_dim=768,
            use_coord_attention=True,
            deep_head=True,
            device=self.device
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def _apply_clahe(self, image):
        """Aplicar CLAHE en espacio LAB."""
        img_array = np.array(image)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_tile_size, self.clahe_tile_size)
        )
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb)

    def _preprocess(self, image):
        """Preprocesar imagen para el modelo."""
        # Guardar tamano original
        original_size = image.size  # (width, height)

        # Aplicar CLAHE si esta habilitado
        if self.use_clahe:
            image = self._apply_clahe(image)

        # Resize a 224x224
        image = image.resize((224, 224), Image.BILINEAR)

        # Convertir a tensor
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).to(self.device)

        # Normalizar
        img_tensor = (img_tensor - self.mean) / self.std

        return img_tensor.unsqueeze(0), original_size

    def _predict_with_tta(self, model, image):
        """Prediccion con Test-Time Augmentation."""
        with torch.no_grad():
            # Original
            pred1 = model(image)

            # Flip horizontal
            image_flip = torch.flip(image, dims=[3])
            pred2 = model(image_flip)

            # Invertir flip en predicciones
            pred2 = pred2.view(-1, 15, 2)
            pred2[:, :, 0] = 1 - pred2[:, :, 0]

            # Intercambiar pares simetricos
            for left, right in self.SYMMETRIC_PAIRS:
                pred2[:, [left, right]] = pred2[:, [right, left]]

            pred2 = pred2.view(-1, 30)

            # Promediar
            return (pred1 + pred2) / 2

    def predict(self, image_path, return_normalized=False):
        """
        Predecir landmarks para una imagen.

        Args:
            image_path: Ruta a la imagen (str o Path).
            return_normalized: Si True, retorna coordenadas normalizadas [0,1].
                              Si False, retorna coordenadas en pixeles de imagen original.

        Returns:
            landmarks: Array (15, 2) con coordenadas (x, y) de cada landmark.
        """
        # Cargar imagen
        image = Image.open(image_path).convert('RGB')
        img_tensor, original_size = self._preprocess(image)

        # Prediccion del ensemble
        preds = []
        for model in self.models:
            pred = self._predict_with_tta(model, img_tensor)
            preds.append(pred)

        ensemble_pred = torch.stack(preds).mean(dim=0)
        landmarks = ensemble_pred[0].view(15, 2).cpu().numpy()

        if return_normalized:
            return landmarks

        # Convertir a coordenadas de imagen original
        landmarks[:, 0] *= original_size[0]  # width
        landmarks[:, 1] *= original_size[1]  # height

        return landmarks

    def predict_batch(self, image_paths, return_normalized=False):
        """
        Predecir landmarks para multiples imagenes.

        Args:
            image_paths: Lista de rutas a imagenes.
            return_normalized: Si True, retorna coordenadas normalizadas.

        Returns:
            List de arrays (15, 2) con coordenadas.
        """
        return [self.predict(p, return_normalized) for p in image_paths]


def visualize_prediction(image_path, landmarks, output_path=None):
    """Visualizar predicciones sobre la imagen."""
    import matplotlib.pyplot as plt

    image = Image.open(image_path).convert('RGB')
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)

    colors = {
        'axis': '#00FF00',
        'central': '#00FFFF',
        'lateral': '#FFFF00',
        'costal': '#FF0000',
    }

    groups = {
        0: 'axis', 1: 'axis',
        2: 'lateral', 3: 'lateral', 4: 'lateral', 5: 'lateral',
        6: 'lateral', 7: 'lateral',
        8: 'central', 9: 'central', 10: 'central',
        11: 'lateral', 12: 'lateral',
        13: 'costal', 14: 'costal',
    }

    for i in range(15):
        color = colors[groups[i]]
        ax.scatter(landmarks[i, 0], landmarks[i, 1],
                   c=color, s=100, marker='o', edgecolors='white', linewidths=1.5)
        ax.annotate(f'L{i+1}', (landmarks[i, 0]+5, landmarks[i, 1]-5),
                    fontsize=8, color='white', weight='bold')

    ax.axis('off')
    ax.set_title(f'Predicted Landmarks - {Path(image_path).name}')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Predecir landmarks en radiografias de torax'
    )
    parser.add_argument('image', type=str, help='Ruta a la imagen')
    parser.add_argument('--output', '-o', type=str, help='Guardar resultados en JSON')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Visualizar predicciones')
    parser.add_argument('--vis-output', type=str,
                        help='Guardar visualizacion en archivo')
    parser.add_argument('--no-clahe', action='store_true',
                        help='Deshabilitar CLAHE')
    parser.add_argument('--normalized', action='store_true',
                        help='Retornar coordenadas normalizadas [0,1]')

    args = parser.parse_args()

    # Verificar imagen existe
    if not Path(args.image).exists():
        print(f"Error: Imagen no encontrada: {args.image}")
        sys.exit(1)

    # Crear predictor
    predictor = EnsemblePredictor(use_clahe=not args.no_clahe)

    # Predecir
    landmarks = predictor.predict(args.image, return_normalized=args.normalized)

    # Mostrar resultados
    print("\n" + "="*50)
    print("PREDICCION DE LANDMARKS")
    print("="*50)
    print(f"Imagen: {args.image}")
    print(f"Coordenadas: {'normalizadas [0,1]' if args.normalized else 'pixeles'}")
    print("\n" + "-"*50)

    for i, name in enumerate(EnsemblePredictor.LANDMARK_NAMES):
        print(f"{name:<25} ({landmarks[i, 0]:7.2f}, {landmarks[i, 1]:7.2f})")

    # Guardar JSON si se especifica
    if args.output:
        result = {
            'image': str(args.image),
            'normalized': args.normalized,
            'landmarks': {
                f'L{i+1}': {'x': float(landmarks[i, 0]), 'y': float(landmarks[i, 1])}
                for i in range(15)
            }
        }
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResultados guardados: {args.output}")

    # Visualizar si se solicita
    if args.visualize or args.vis_output:
        visualize_prediction(args.image, landmarks, args.vis_output)


if __name__ == '__main__':
    main()
