#!/usr/bin/env python3
"""
Script para probar el LandmarkDataset y verificar transformaciones.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from src_v2.data.utils import (
    load_coordinates_csv,
    get_image_path,
    get_landmarks_array,
    LANDMARK_NAMES,
    SYMMETRIC_PAIRS
)
from src_v2.data.dataset import LandmarkDataset, create_dataloaders
from src_v2.data.transforms import TrainTransform, ValTransform


def test_dataset_loading():
    """Prueba que el dataset carga correctamente."""
    print("=" * 60)
    print("TEST 1: Carga del Dataset")
    print("=" * 60)

    data_root = project_root / 'data'
    csv_path = data_root / 'coordenadas' / 'coordenadas_maestro.csv'

    df = load_coordinates_csv(csv_path)
    val_transform = ValTransform(output_size=224)

    dataset = LandmarkDataset(df.head(10), data_root, val_transform)

    print(f"Dataset creado con {len(dataset)} muestras")

    # Probar carga de una muestra
    image, landmarks, meta = dataset[0]

    print(f"\nMuestra 0:")
    print(f"  Image shape: {image.shape}")
    print(f"  Landmarks shape: {landmarks.shape}")
    print(f"  Image dtype: {image.dtype}")
    print(f"  Landmarks dtype: {landmarks.dtype}")
    print(f"  Landmarks range: [{landmarks.min():.3f}, {landmarks.max():.3f}]")
    print(f"  Meta: {meta}")

    assert image.shape == (3, 224, 224), f"Expected (3, 224, 224), got {image.shape}"
    assert landmarks.shape == (30,), f"Expected (30,), got {landmarks.shape}"
    assert 0 <= landmarks.min() <= landmarks.max() <= 1, "Landmarks should be in [0, 1]"

    print("\n[OK] Dataset carga correctamente")
    return True


def test_flip_horizontal():
    """Prueba que el flip horizontal intercambia pares correctamente."""
    print("\n" + "=" * 60)
    print("TEST 2: Flip Horizontal")
    print("=" * 60)

    data_root = project_root / 'data'
    csv_path = data_root / 'coordenadas' / 'coordenadas_maestro.csv'

    df = load_coordinates_csv(csv_path)
    row = df.iloc[0]

    # Cargar imagen y landmarks originales
    img_path = get_image_path(row['image_name'], row['category'], data_root)
    image = Image.open(img_path).convert('RGB')
    landmarks = get_landmarks_array(row)

    print(f"Imagen: {row['image_name']}")
    print(f"Original size: {image.size}")

    # Crear transform
    train_transform = TrainTransform(
        output_size=224,
        flip_prob=1.0,  # Forzar flip
        rotation_degrees=0.0  # Sin rotacion
    )

    # Normalizar coords manualmente para comparar
    original_norm = landmarks.copy() / 299.0

    # Aplicar transform (que incluye flip forzado)
    image_t, landmarks_t = train_transform(image, landmarks, (299, 299))
    landmarks_flipped = landmarks_t.reshape(15, 2).numpy()

    print(f"\nVerificando intercambio de pares simetricos:")
    all_correct = True

    for left, right in SYMMETRIC_PAIRS:
        # Original
        orig_left_x = original_norm[left, 0]
        orig_right_x = original_norm[right, 0]

        # Despues del flip, las coordenadas X se reflejan Y los indices se intercambian
        # La posicion "left" ahora contiene el valor de "right" reflejado
        flip_left_x = landmarks_flipped[left, 0]
        flip_right_x = landmarks_flipped[right, 0]

        # El valor en posicion left despues del flip debe ser 1 - original_right_x
        expected_left_x = 1.0 - orig_right_x
        expected_right_x = 1.0 - orig_left_x

        diff_left = abs(flip_left_x - expected_left_x)
        diff_right = abs(flip_right_x - expected_right_x)

        is_correct = diff_left < 0.001 and diff_right < 0.001
        status = "[OK]" if is_correct else "[FAIL]"
        all_correct = all_correct and is_correct

        print(f"  {status} {LANDMARK_NAMES[left]}-{LANDMARK_NAMES[right]}: "
              f"diff_left={diff_left:.4f}, diff_right={diff_right:.4f}")

    # Verificar que landmarks centrales (L1, L2, L9, L10, L11) solo se reflejaron en X
    print("\nVerificando landmarks centrales (solo reflexion X):")
    central_indices = [0, 1, 8, 9, 10]

    for idx in central_indices:
        orig_x = original_norm[idx, 0]
        flip_x = landmarks_flipped[idx, 0]
        expected_x = 1.0 - orig_x

        diff = abs(flip_x - expected_x)
        is_correct = diff < 0.001
        status = "[OK]" if is_correct else "[FAIL]"
        all_correct = all_correct and is_correct

        print(f"  {status} {LANDMARK_NAMES[idx]}: orig_x={orig_x:.3f}, flip_x={flip_x:.3f}, "
              f"expected={expected_x:.3f}")

    if all_correct:
        print("\n[OK] Flip horizontal funciona correctamente")
    else:
        print("\n[FAIL] Flip horizontal tiene problemas")

    return all_correct


def test_dataloader():
    """Prueba que el dataloader funciona."""
    print("\n" + "=" * 60)
    print("TEST 3: DataLoader")
    print("=" * 60)

    data_root = project_root / 'data'
    csv_path = data_root / 'coordenadas' / 'coordenadas_maestro.csv'

    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path=str(csv_path),
        data_root=str(data_root),
        batch_size=8,
        num_workers=0,  # Para evitar problemas en testing
        flip_prob=0.5,
        rotation_degrees=10.0
    )

    # Probar un batch
    images, landmarks, metas = next(iter(train_loader))

    print(f"\nBatch de training:")
    print(f"  Images shape: {images.shape}")
    print(f"  Landmarks shape: {landmarks.shape}")
    print(f"  Num metas: {len(metas)}")

    assert images.shape == (8, 3, 224, 224), f"Expected (8, 3, 224, 224), got {images.shape}"
    assert landmarks.shape == (8, 30), f"Expected (8, 30), got {landmarks.shape}"

    print("\n[OK] DataLoader funciona correctamente")
    return True


def visualize_samples(num_samples=4, save_path=None):
    """Visualiza muestras con landmarks superpuestos."""
    print("\n" + "=" * 60)
    print("TEST 4: Visualizacion")
    print("=" * 60)

    data_root = project_root / 'data'
    csv_path = data_root / 'coordenadas' / 'coordenadas_maestro.csv'

    df = load_coordinates_csv(csv_path)

    # Crear figura
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))

    # Seleccionar muestras aleatorias
    indices = np.random.choice(len(df), num_samples, replace=False)

    for i, idx in enumerate(indices):
        row = df.iloc[idx]
        img_path = get_image_path(row['image_name'], row['category'], data_root)
        image = Image.open(img_path).convert('RGB')
        landmarks = get_landmarks_array(row)

        # Fila 1: Original
        ax1 = axes[0, i]
        ax1.imshow(image)
        ax1.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=30, marker='x')
        ax1.set_title(f"{row['category']}\n{row['image_name'][:15]}...")
        ax1.axis('off')

        # Fila 2: Con transformacion de validacion (224x224)
        val_transform = ValTransform(output_size=224)
        image_t, landmarks_t = val_transform(image, landmarks, (299, 299))

        # Desnormalizar para visualizacion
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_vis = image_t * std + mean
        image_vis = image_vis.permute(1, 2, 0).numpy()
        image_vis = np.clip(image_vis, 0, 1)

        landmarks_vis = landmarks_t.reshape(15, 2).numpy() * 224

        ax2 = axes[1, i]
        ax2.imshow(image_vis)
        ax2.scatter(landmarks_vis[:, 0], landmarks_vis[:, 1], c='lime', s=30, marker='x')

        # Conectar L1-L2
        ax2.plot([landmarks_vis[0, 0], landmarks_vis[1, 0]],
                 [landmarks_vis[0, 1], landmarks_vis[1, 1]], 'c--', linewidth=1)
        ax2.set_title(f"224x224 normalized")
        ax2.axis('off')

    plt.suptitle("Arriba: Original | Abajo: Transformado 224x224", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")
    else:
        plt.savefig(project_root / 'outputs' / 'sample_visualization.png', dpi=150, bbox_inches='tight')
        print(f"Figura guardada en: outputs/sample_visualization.png")

    plt.close()
    print("[OK] Visualizacion completada")
    return True


def visualize_flip_comparison():
    """Visualiza antes/despues del flip horizontal."""
    print("\n" + "=" * 60)
    print("TEST 5: Comparacion Flip")
    print("=" * 60)

    data_root = project_root / 'data'
    csv_path = data_root / 'coordenadas' / 'coordenadas_maestro.csv'

    df = load_coordinates_csv(csv_path)
    row = df.iloc[10]

    img_path = get_image_path(row['image_name'], row['category'], data_root)
    image = Image.open(img_path).convert('RGB')
    landmarks = get_landmarks_array(row)

    # Transform sin flip
    val_transform = ValTransform(output_size=224)
    image_orig, landmarks_orig = val_transform(image, landmarks, (299, 299))

    # Transform con flip
    train_transform = TrainTransform(output_size=224, flip_prob=1.0, rotation_degrees=0.0)
    image_flip, landmarks_flip = train_transform(image, landmarks.copy(), (299, 299))

    # Desnormalizar
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img_orig_vis = (image_orig * std + mean).permute(1, 2, 0).numpy()
    img_flip_vis = (image_flip * std + mean).permute(1, 2, 0).numpy()

    lm_orig = landmarks_orig.reshape(15, 2).numpy() * 224
    lm_flip = landmarks_flip.reshape(15, 2).numpy() * 224

    # Crear figura
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Colores para pares simetricos
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    pair_labels = ['L3-L4', 'L5-L6', 'L7-L8', 'L12-L13', 'L14-L15']

    # Original
    axes[0].imshow(np.clip(img_orig_vis, 0, 1))
    for idx, (left, right) in enumerate(SYMMETRIC_PAIRS):
        axes[0].scatter(lm_orig[left, 0], lm_orig[left, 1], c=colors[idx], s=100, marker='o', label=f'{LANDMARK_NAMES[left]}')
        axes[0].scatter(lm_orig[right, 0], lm_orig[right, 1], c=colors[idx], s=100, marker='s')
        axes[0].annotate(LANDMARK_NAMES[left], (lm_orig[left, 0]+3, lm_orig[left, 1]-3), fontsize=8, color=colors[idx])
        axes[0].annotate(LANDMARK_NAMES[right], (lm_orig[right, 0]+3, lm_orig[right, 1]-3), fontsize=8, color=colors[idx])

    axes[0].plot([lm_orig[0, 0], lm_orig[1, 0]], [lm_orig[0, 1], lm_orig[1, 1]], 'c--', linewidth=2)
    axes[0].scatter([lm_orig[0, 0], lm_orig[1, 0]], [lm_orig[0, 1], lm_orig[1, 1]], c='cyan', s=80, marker='D')
    axes[0].set_title(f"Original\n{row['image_name']}")
    axes[0].axis('off')

    # Flip
    axes[1].imshow(np.clip(img_flip_vis, 0, 1))
    for idx, (left, right) in enumerate(SYMMETRIC_PAIRS):
        axes[1].scatter(lm_flip[left, 0], lm_flip[left, 1], c=colors[idx], s=100, marker='o')
        axes[1].scatter(lm_flip[right, 0], lm_flip[right, 1], c=colors[idx], s=100, marker='s')
        axes[1].annotate(LANDMARK_NAMES[left], (lm_flip[left, 0]+3, lm_flip[left, 1]-3), fontsize=8, color=colors[idx])
        axes[1].annotate(LANDMARK_NAMES[right], (lm_flip[right, 0]+3, lm_flip[right, 1]-3), fontsize=8, color=colors[idx])

    axes[1].plot([lm_flip[0, 0], lm_flip[1, 0]], [lm_flip[0, 1], lm_flip[1, 1]], 'c--', linewidth=2)
    axes[1].scatter([lm_flip[0, 0], lm_flip[1, 0]], [lm_flip[0, 1], lm_flip[1, 1]], c='cyan', s=80, marker='D')
    axes[1].set_title("Flip Horizontal\n(pares intercambiados)")
    axes[1].axis('off')

    plt.suptitle("Circulo (o) = Izquierda | Cuadrado (s) = Derecha\nMismo color = mismo par simetrico", fontsize=10)
    plt.tight_layout()

    output_path = project_root / 'outputs' / 'flip_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Figura guardada en: {output_path}")
    print("[OK] Comparacion de flip completada")
    return True


if __name__ == '__main__':
    # Crear directorio de outputs
    outputs_dir = project_root / 'outputs'
    outputs_dir.mkdir(exist_ok=True)

    # Ejecutar tests
    results = []

    results.append(("Dataset Loading", test_dataset_loading()))
    results.append(("Flip Horizontal", test_flip_horizontal()))
    results.append(("DataLoader", test_dataloader()))
    results.append(("Visualization", visualize_samples()))
    results.append(("Flip Comparison", visualize_flip_comparison()))

    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DE TESTS")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        all_passed = all_passed and passed

    if all_passed:
        print("\nTodos los tests pasaron!")
    else:
        print("\nAlgunos tests fallaron. Revisar errores arriba.")
