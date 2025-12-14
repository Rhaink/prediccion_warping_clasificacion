#!/usr/bin/env python3
"""
SESIÓN 28: Experimento Crítico - Verificación de Fondo Negro como Shortcut

OBJETIVO: Demostrar que el modelo NO usa el fondo negro de las imágenes
          warpeadas como atajo para clasificación.

HIPÓTESIS:
- H0: El fondo negro NO afecta la clasificación
- H1: El modelo usa el fondo negro como shortcut

METODOLOGÍA:
1. Análisis estadístico de fill_rate por clase
2. Evaluar modelo con diferentes tratamientos del fondo:
   - Original (fondo negro)
   - Fondo con ruido gaussiano
   - Fondo con valor medio de la imagen
   - Fondo con ruido uniforme

Si accuracy es similar en todas las variantes → H0 (fondo NO es shortcut)

Autor: Proyecto Tesis Maestría
Fecha: 29-Nov-2024
Sesión: 28
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from scipy import stats
from collections import defaultdict
import json
import time
from datetime import datetime

# Configuración
BATCH_SIZE = 32
SEED = 42

# Paths
DATASET_DIR = PROJECT_ROOT / "outputs" / "full_warped_dataset"
MODEL_PATH = PROJECT_ROOT / "outputs" / "session27_models" / "resnet18_expanded_15k_best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "session28_black_background"


class GrayscaleToRGB:
    def __call__(self, img):
        if img.mode == 'RGB':
            return img
        return img.convert('RGB')


class BackgroundModifiedDataset(Dataset):
    """Dataset que modifica el fondo de las imágenes warpeadas."""

    def __init__(self, root_dir, transform=None, background_mode='original'):
        """
        background_mode:
            - 'original': fondo negro sin modificar
            - 'gaussian': ruido gaussiano en fondo negro
            - 'uniform': ruido uniforme en fondo negro
            - 'mean': valor medio de la imagen en fondo negro
            - 'edge_blur': copia borrosa de bordes en el fondo
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.background_mode = background_mode

        self.samples = []
        self.class_to_idx = {}
        self.fill_rates = []  # Para análisis estadístico

        # Cargar imágenes
        for class_idx, class_name in enumerate(sorted(self.root_dir.iterdir())):
            if class_name.is_dir():
                self.class_to_idx[class_name.name] = class_idx
                for img_path in class_name.glob('*.png'):
                    self.samples.append((img_path, class_idx, class_name.name))

        self.classes = list(self.class_to_idx.keys())
        np.random.seed(SEED)

    def __len__(self):
        return len(self.samples)

    def _get_black_mask(self, img_array):
        """Detecta píxeles negros (fondo)."""
        if len(img_array.shape) == 3:
            # RGB: píxel es negro si todos los canales son muy bajos
            return np.all(img_array < 5, axis=2)
        else:
            # Grayscale
            return img_array < 5

    def _fill_background(self, img_array, mask):
        """Rellena el fondo según el modo seleccionado."""
        result = img_array.copy()

        if self.background_mode == 'original':
            return result

        # Obtener valores no-negros para calcular estadísticas
        if len(img_array.shape) == 3:
            non_black = img_array[~mask]
        else:
            non_black = img_array[~mask]

        if len(non_black) == 0:
            return result

        if self.background_mode == 'gaussian':
            # Ruido gaussiano con media y std del área de pulmón
            mean_val = non_black.mean()
            std_val = max(non_black.std(), 1)

            if len(img_array.shape) == 3:
                noise = np.random.normal(mean_val * 0.3, std_val * 0.5, img_array.shape)
                noise = np.clip(noise, 0, 255).astype(np.uint8)
                for c in range(3):
                    result[:, :, c][mask] = noise[:, :, c][mask]
            else:
                noise = np.random.normal(mean_val * 0.3, std_val * 0.5, img_array.shape)
                noise = np.clip(noise, 0, 255).astype(np.uint8)
                result[mask] = noise[mask]

        elif self.background_mode == 'uniform':
            # Ruido uniforme
            if len(img_array.shape) == 3:
                noise = np.random.randint(0, 50, img_array.shape, dtype=np.uint8)
                for c in range(3):
                    result[:, :, c][mask] = noise[:, :, c][mask]
            else:
                noise = np.random.randint(0, 50, img_array.shape, dtype=np.uint8)
                result[mask] = noise[mask]

        elif self.background_mode == 'mean':
            # Valor medio gris del área de pulmón
            mean_val = int(non_black.mean() * 0.3)
            if len(img_array.shape) == 3:
                for c in range(3):
                    result[:, :, c][mask] = mean_val
            else:
                result[mask] = mean_val

        elif self.background_mode == 'high_noise':
            # Ruido muy alto para verificar si afecta
            if len(img_array.shape) == 3:
                noise = np.random.randint(50, 200, img_array.shape, dtype=np.uint8)
                for c in range(3):
                    result[:, :, c][mask] = noise[:, :, c][mask]
            else:
                noise = np.random.randint(50, 200, img_array.shape, dtype=np.uint8)
                result[mask] = noise[mask]

        return result

    def __getitem__(self, idx):
        img_path, class_idx, class_name = self.samples[idx]

        # Cargar imagen
        img = Image.open(img_path)
        img_array = np.array(img)

        # Calcular fill rate (proporción de área NO negra)
        mask = self._get_black_mask(img_array)
        fill_rate = 1 - mask.mean()

        # Modificar fondo
        img_modified = self._fill_background(img_array, mask)
        img = Image.fromarray(img_modified)

        if self.transform:
            img = self.transform(img)

        return img, class_idx, fill_rate, class_name


def analyze_fill_rates(dataset):
    """Analiza estadísticamente los fill rates por clase."""
    print("\n" + "="*70)
    print("ANÁLISIS ESTADÍSTICO DE FILL RATE POR CLASE")
    print("="*70)

    fill_rates_by_class = defaultdict(list)

    # Recolectar fill rates
    for i in range(len(dataset)):
        _, class_idx, fill_rate, class_name = dataset[i]
        fill_rates_by_class[class_name].append(fill_rate)

        if (i + 1) % 500 == 0:
            print(f"  Procesando... {i+1}/{len(dataset)}")

    # Estadísticas descriptivas
    print("\nEstadísticas de Fill Rate por Clase:")
    print("-" * 50)

    results = {}
    all_rates = []

    for class_name, rates in sorted(fill_rates_by_class.items()):
        rates = np.array(rates)
        all_rates.append(rates)
        results[class_name] = {
            'mean': float(np.mean(rates)),
            'std': float(np.std(rates)),
            'min': float(np.min(rates)),
            'max': float(np.max(rates)),
            'count': len(rates)
        }
        print(f"  {class_name:20s}: mean={results[class_name]['mean']:.4f} "
              f"std={results[class_name]['std']:.4f} "
              f"[{results[class_name]['min']:.4f}, {results[class_name]['max']:.4f}]")

    # Test ANOVA: ¿hay diferencia significativa entre clases?
    print("\nTest ANOVA (H0: medias iguales entre clases):")
    f_stat, p_value = stats.f_oneway(*all_rates)
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")

    if p_value < 0.05:
        print("  RESULTADO: Diferencia SIGNIFICATIVA entre clases (p < 0.05)")
        print("  ⚠️  ALERTA: El fill rate PODRÍA ser un shortcut")
    else:
        print("  RESULTADO: NO hay diferencia significativa (p >= 0.05)")
        print("  ✓ El fill rate NO correlaciona con la clase")

    # Test adicional: correlación de Spearman
    print("\nTest de Kruskal-Wallis (no paramétrico):")
    h_stat, kw_p = stats.kruskal(*all_rates)
    print(f"  H-statistic: {h_stat:.4f}")
    print(f"  p-value: {kw_p:.6f}")

    results['anova_f'] = float(f_stat)
    results['anova_p'] = float(p_value)
    results['kruskal_h'] = float(h_stat)
    results['kruskal_p'] = float(kw_p)

    return results


def load_model(model_path, device, num_classes=3):
    """Carga el modelo entrenado."""
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes)
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint['class_names']


def evaluate_with_background(model, dataloader, device):
    """Evalúa el modelo y retorna métricas detalladas."""
    model.eval()

    all_preds = []
    all_labels = []
    all_fill_rates = []
    all_class_names = []

    with torch.no_grad():
        for inputs, labels, fill_rates, class_names in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_fill_rates.extend(fill_rates.numpy())
            all_class_names.extend(class_names)

    # Calcular accuracy
    correct = np.array(all_preds) == np.array(all_labels)
    accuracy = correct.mean() * 100

    # Accuracy por clase
    class_accuracies = {}
    for class_name in set(all_class_names):
        mask = np.array(all_class_names) == class_name
        if mask.sum() > 0:
            class_accuracies[class_name] = correct[mask].mean() * 100

    return {
        'accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'n_samples': len(all_labels)
    }


def collate_fn(batch):
    """Custom collate para incluir fill_rate y class_name."""
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    fill_rates = torch.tensor([item[2] for item in batch])
    class_names = [item[3] for item in batch]
    return images, labels, fill_rates, class_names


def main():
    print("="*70)
    print("SESIÓN 28: EXPERIMENTO CRÍTICO - VERIFICACIÓN FONDO NEGRO")
    print("="*70)
    print(f"Modelo: {MODEL_PATH}")
    print(f"Dataset: {DATASET_DIR}")

    # Setup
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Transformaciones
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    eval_transform = transforms.Compose([
        GrayscaleToRGB(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    # Cargar modelo
    print("\nCargando modelo...")
    model, class_names = load_model(MODEL_PATH, device)
    print(f"Clases: {class_names}")

    # =====================================================
    # PARTE 1: ANÁLISIS DE FILL RATE
    # =====================================================
    print("\n" + "="*70)
    print("PARTE 1: ANÁLISIS DE FILL RATE POR CLASE")
    print("="*70)

    test_dir = DATASET_DIR / 'test'
    dataset_original = BackgroundModifiedDataset(
        test_dir, transform=eval_transform, background_mode='original'
    )

    fill_rate_analysis = analyze_fill_rates(dataset_original)

    # =====================================================
    # PARTE 2: EVALUACIÓN CON DIFERENTES FONDOS
    # =====================================================
    print("\n" + "="*70)
    print("PARTE 2: EVALUACIÓN CON DIFERENTES TRATAMIENTOS DE FONDO")
    print("="*70)

    background_modes = [
        ('original', 'Fondo negro original'),
        ('gaussian', 'Ruido gaussiano en fondo'),
        ('uniform', 'Ruido uniforme (0-50)'),
        ('mean', 'Valor medio gris en fondo'),
        ('high_noise', 'Ruido alto (50-200)'),
    ]

    results = {}

    for mode, description in background_modes:
        print(f"\n--- Evaluando: {description} ({mode}) ---")

        dataset = BackgroundModifiedDataset(
            test_dir, transform=eval_transform, background_mode=mode
        )

        dataloader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, collate_fn=collate_fn
        )

        metrics = evaluate_with_background(model, dataloader, device)
        results[mode] = metrics

        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        for cls, acc in sorted(metrics['class_accuracies'].items()):
            print(f"    {cls}: {acc:.2f}%")

    # =====================================================
    # PARTE 3: ANÁLISIS Y CONCLUSIONES
    # =====================================================
    print("\n" + "="*70)
    print("PARTE 3: RESUMEN Y CONCLUSIONES")
    print("="*70)

    print("\nComparación de Accuracy por Tratamiento de Fondo:")
    print("-" * 50)

    baseline = results['original']['accuracy']

    for mode, description in background_modes:
        acc = results[mode]['accuracy']
        diff = acc - baseline
        status = "✓" if abs(diff) < 2.0 else "⚠️"
        print(f"  {description:30s}: {acc:.2f}% ({diff:+.2f}%) {status}")

    # Determinar si el fondo negro es un shortcut
    max_diff = max(abs(results[mode]['accuracy'] - baseline)
                   for mode, _ in background_modes if mode != 'original')

    print("\n" + "="*70)
    print("CONCLUSIÓN:")
    print("="*70)

    if max_diff < 2.0:
        conclusion = "H0 ACEPTADA: El fondo negro NO es un shortcut"
        is_shortcut = False
        print(f"✓ {conclusion}")
        print(f"  Máxima variación de accuracy: {max_diff:.2f}%")
        print("  El modelo NO depende del fondo negro para clasificar.")
        print("  Las predicciones son robustas a cambios en el fondo.")
    else:
        conclusion = "H1 ACEPTADA: El fondo negro PODRÍA ser un shortcut"
        is_shortcut = True
        print(f"⚠️ {conclusion}")
        print(f"  Máxima variación de accuracy: {max_diff:.2f}%")
        print("  SE REQUIERE investigación adicional.")

    # =====================================================
    # GUARDAR RESULTADOS
    # =====================================================

    final_results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': str(MODEL_PATH),
        'test_samples': len(dataset_original),
        'fill_rate_analysis': fill_rate_analysis,
        'background_experiments': results,
        'max_accuracy_difference': max_diff,
        'is_shortcut': is_shortcut,
        'conclusion': conclusion
    }

    results_path = OUTPUT_DIR / 'black_background_experiment_results.json'
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResultados guardados: {results_path}")

    print("\n" + "="*70)
    print("EXPERIMENTO COMPLETADO")
    print("="*70)

    return not is_shortcut  # True si el fondo NO es shortcut


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
