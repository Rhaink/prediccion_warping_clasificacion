"""
Fixtures compartidos para tests del proyecto COVID-19 Landmark Detection.
"""

import numpy as np
import pytest
import torch
from pathlib import Path


# =============================================================================
# FIXTURES DE DATOS SINTETICOS
# =============================================================================

@pytest.fixture
def sample_landmarks():
    """Landmarks de ejemplo normalizados en [0, 1]."""
    # 15 landmarks con coordenadas x, y en [0.2, 0.8] para estar dentro de la imagen
    landmarks = np.array([
        [0.5, 0.1],   # L1 - centro superior
        [0.5, 0.9],   # L2 - centro inferior
        [0.3, 0.3],   # L3 - izq superior
        [0.7, 0.3],   # L4 - der superior
        [0.25, 0.5],  # L5 - izq medio
        [0.75, 0.5],  # L6 - der medio
        [0.3, 0.7],   # L7 - izq inferior
        [0.7, 0.7],   # L8 - der inferior
        [0.5, 0.325], # L9 - centro t=0.25
        [0.5, 0.5],   # L10 - centro t=0.50
        [0.5, 0.675], # L11 - centro t=0.75
        [0.35, 0.15], # L12 - borde sup izq
        [0.65, 0.15], # L13 - borde sup der
        [0.35, 0.85], # L14 - esquina inf izq
        [0.65, 0.85], # L15 - esquina inf der
    ], dtype=np.float32)
    return landmarks


@pytest.fixture
def sample_landmarks_tensor(sample_landmarks):
    """Landmarks como tensor de PyTorch (1, 30)."""
    return torch.from_numpy(sample_landmarks.flatten()).unsqueeze(0)


@pytest.fixture
def batch_landmarks_tensor(sample_landmarks):
    """Batch de landmarks como tensor (4, 30)."""
    landmarks_flat = sample_landmarks.flatten()
    batch = np.stack([landmarks_flat] * 4)
    # Bug fix Session 31: Seed fijo para reproducibilidad en tests
    rng = np.random.RandomState(42)
    batch[1] += rng.randn(30) * 0.01
    batch[2] += rng.randn(30) * 0.01
    batch[3] += rng.randn(30) * 0.01
    return torch.from_numpy(batch.astype(np.float32))


@pytest.fixture
def sample_image():
    """Imagen de ejemplo como numpy array (224, 224, 3)."""
    return np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_gray():
    """Imagen grayscale de ejemplo (224, 224)."""
    return np.random.randint(0, 256, (224, 224), dtype=np.uint8)


@pytest.fixture
def sample_image_tensor():
    """Imagen como tensor normalizado (1, 3, 224, 224)."""
    img = torch.randn(1, 3, 224, 224)
    return img


@pytest.fixture
def batch_images_tensor():
    """Batch de imagenes como tensor (4, 3, 224, 224)."""
    return torch.randn(4, 3, 224, 224)


# =============================================================================
# FIXTURES DE MODELO
# =============================================================================

@pytest.fixture
def model_device():
    """Dispositivo para tests (CPU para CI)."""
    return torch.device('cpu')


@pytest.fixture
def untrained_model(model_device):
    """Modelo sin entrenar para tests."""
    from src_v2.models import create_model
    model = create_model(pretrained=False)
    model = model.to(model_device)
    model.eval()
    return model


@pytest.fixture
def pretrained_model(model_device):
    """Modelo con pesos preentrenados de ImageNet."""
    from src_v2.models import create_model
    model = create_model(pretrained=True)
    model = model.to(model_device)
    model.eval()
    return model


# =============================================================================
# FIXTURES DE PATHS
# =============================================================================

@pytest.fixture
def project_root():
    """Raiz del proyecto."""
    return Path(__file__).parent.parent


@pytest.fixture
def src_v2_path(project_root):
    """Path a src_v2."""
    return project_root / 'src_v2'


@pytest.fixture
def conf_path(src_v2_path):
    """Path a configuracion Hydra."""
    return src_v2_path / 'conf'


@pytest.fixture
def temp_dir(tmp_path):
    """Directorio temporal para tests."""
    return tmp_path


# =============================================================================
# FIXTURES DE CONFIGURACION
# =============================================================================

@pytest.fixture
def default_config():
    """Configuracion por defecto como diccionario."""
    return {
        'model': {
            'name': 'resnet18_landmarks',
            'pretrained': True,
        },
        'training': {
            'phase1_epochs': 2,
            'phase2_epochs': 2,
            'batch_size': 4,
        },
        'data': {
            'image_size': 224,
        }
    }


# =============================================================================
# FIXTURES PARA TESTS DE INTEGRACION CLI
# =============================================================================

@pytest.fixture
def test_image_file(tmp_path):
    """
    Imagen de prueba guardada en archivo.
    Retorna path a imagen PNG de 224x224 en escala de grises.

    Session 33: Bug M1 fix - Verificar que la imagen se guardo correctamente.
    """
    from PIL import Image

    # Crear imagen grayscale simulando radiografia
    img_array = np.random.randint(50, 200, (224, 224), dtype=np.uint8)
    # Agregar patron central para simular torax
    center = 112
    for i in range(224):
        for j in range(224):
            dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            if dist < 80:
                img_array[i, j] = min(255, img_array[i, j] + 30)

    img = Image.fromarray(img_array, mode='L')
    img_path = tmp_path / "test_xray.png"
    img.save(img_path)

    # Session 33: Bug M1 fix - Verificaciones de integridad
    assert img_path.exists(), f"Image file {img_path} was not created"
    assert img_path.stat().st_size > 0, f"Image file {img_path} is empty"

    # Verificar que la imagen se puede abrir y tiene dimensiones correctas
    opened_img = Image.open(img_path)
    assert opened_img.size == (224, 224), f"Image size {opened_img.size} != (224, 224)"
    assert opened_img.mode == 'L', f"Image mode {opened_img.mode} != 'L' (grayscale)"

    return img_path


@pytest.fixture
def mock_landmark_checkpoint(tmp_path, model_device):
    """
    Checkpoint mock de modelo de landmarks.
    Crea un modelo real pero sin entrenar, guardado como checkpoint.
    """
    from src_v2.models import create_model

    # Crear modelo sin pesos pretrained (mas rapido)
    model = create_model(
        pretrained=False,
        use_coord_attention=False,  # Simple para tests
        deep_head=False,
        hidden_dim=256
    )
    model = model.to(model_device)

    # Guardar checkpoint en formato esperado
    checkpoint_path = tmp_path / "landmark_model.pt"
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': 1,
        'val_loss': 10.0,
        'config': {
            'use_coord_attention': False,
            'deep_head': False,
            'hidden_dim': 256,
        }
    }
    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


@pytest.fixture
def mock_classifier_checkpoint(tmp_path, model_device):
    """
    Checkpoint mock de clasificador.
    Crea clasificador real pero sin entrenar.

    Session 33: Bug M4 fix - Verificar guardado y limpieza de memoria.
    """
    import gc
    from src_v2.models import ImageClassifier

    # Crear clasificador simple
    model = ImageClassifier(
        backbone='resnet18',
        num_classes=3,
        pretrained=False
    )
    model = model.to(model_device)

    # Guardar checkpoint con estructura completa esperada por el CLI
    checkpoint_path = tmp_path / "classifier_model.pt"
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': 1,
        'val_accuracy': 0.33,
        'class_names': ['COVID', 'Normal', 'Viral_Pneumonia'],  # Bug #1 fix
        'model_name': 'resnet18',  # Bug #1 fix
        'best_val_f1': 0.33,  # Bug #1 fix
        'config': {
            'backbone': 'resnet18',
            'num_classes': 3,
        }
    }
    torch.save(checkpoint, checkpoint_path)

    # Session 33: Bug M4 fix - Verificaciones de integridad
    assert checkpoint_path.exists(), f"Checkpoint file {checkpoint_path} was not created"
    assert checkpoint_path.stat().st_size > 0, f"Checkpoint file {checkpoint_path} is empty"

    # Round-trip validation: verificar que el checkpoint se puede cargar
    loaded_checkpoint = torch.load(checkpoint_path, map_location=model_device, weights_only=False)
    required_fields = ['model_state_dict', 'class_names', 'model_name', 'config']
    for field in required_fields:
        assert field in loaded_checkpoint, f"Missing required field '{field}' in checkpoint"
    assert len(loaded_checkpoint['class_names']) == 3, "class_names should have 3 elements"

    # Session 33: Bug M4 fix - Limpieza explicita de memoria
    del model
    del loaded_checkpoint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return checkpoint_path


@pytest.fixture
def minimal_landmark_dataset(tmp_path):
    """
    Dataset minimo para entrenar modelo de landmarks.

    Estructura:
        tmp_path/data/
            COVID/images/COVID-001.png, COVID-002.png
            Normal/images/Normal-001.png, Normal-002.png
        tmp_path/coords.csv

    Retorna: (data_dir, csv_path, image_names)
    """
    from PIL import Image

    data_dir = tmp_path / "data"
    classes = ["COVID", "Normal"]
    images_per_class = 3  # Minimo para train/val/test

    image_names = []

    for cls in classes:
        cls_dir = data_dir / cls / "images"
        cls_dir.mkdir(parents=True)

        for i in range(images_per_class):
            # Crear imagen con patron distintivo
            img_array = np.zeros((224, 224, 3), dtype=np.uint8)

            if cls == "COVID":
                img_array[:, :, 0] = 128  # Canal rojo
                img_array[50:174, 50:174] = [200, 100, 100]
            else:
                img_array[:, :, 1] = 128  # Canal verde
                img_array[50:174, 50:174] = [100, 200, 100]

            # Agregar ruido
            noise = np.random.randint(-20, 20, img_array.shape, dtype=np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            img = Image.fromarray(img_array, mode='RGB')
            img_name = f"{cls}-{i+1:03d}.png"
            img.save(cls_dir / img_name)
            image_names.append((img_name, cls))

    # Crear CSV de coordenadas
    csv_path = tmp_path / "coords.csv"

    # Base landmarks (15 puntos) en pixels
    base_landmarks = np.array([
        [112, 25],    # L1
        [112, 200],   # L2
        [50, 60],     # L3
        [174, 60],    # L4
        [40, 112],    # L5
        [184, 112],   # L6
        [50, 170],    # L7
        [174, 170],   # L8
        [112, 70],    # L9
        [112, 112],   # L10
        [112, 160],   # L11
        [60, 35],     # L12
        [164, 35],    # L13
        [35, 195],    # L14
        [189, 195],   # L15
    ], dtype=np.float32)

    # Escribir CSV
    with open(csv_path, 'w') as f:
        # Header: nombre_imagen, clase, L1_x, L1_y, ..., L15_x, L15_y
        header = ["image_name", "category"]
        for i in range(1, 16):
            header.extend([f"L{i}_x", f"L{i}_y"])
        f.write(','.join(header) + '\n')

        # Datos
        np.random.seed(42)
        for img_name, cls in image_names:
            noise = np.random.randn(15, 2) * 3
            landmarks = base_landmarks + noise
            landmarks = np.clip(landmarks, 5, 219)

            row = [img_name.replace('.png', ''), cls]
            for lm in landmarks:
                row.extend([f"{lm[0]:.1f}", f"{lm[1]:.1f}"])
            f.write(','.join(row) + '\n')

    return data_dir, csv_path, image_names


@pytest.fixture
def canonical_shape_json(tmp_path):
    """
    JSON con forma canonica normalizada para warping.
    """
    import json

    json_path = tmp_path / "canonical_shape.json"

    # Forma canonica normalizada (15 landmarks)
    canonical_shape = [
        [0.0, -0.245],     # L1 - Superior
        [0.0, 0.245],      # L2 - Inferior
        [-0.28, -0.16],    # L3 - Apex izq
        [0.28, -0.16],     # L4 - Apex der
        [-0.32, 0.0],      # L5 - Hilio izq
        [0.32, 0.0],       # L6 - Hilio der
        [-0.28, 0.16],     # L7 - Base izq
        [0.28, 0.16],      # L8 - Base der
        [0.0, -0.12],      # L9 - Centro sup
        [0.0, 0.0],        # L10 - Centro med
        [0.0, 0.12],       # L11 - Centro inf
        [-0.2, -0.22],     # L12 - Borde sup izq
        [0.2, -0.22],      # L13 - Borde sup der
        [-0.32, 0.24],     # L14 - Costofrénico izq
        [0.32, 0.24],      # L15 - Costofrénico der
    ]

    data = {
        "canonical_shape_normalized": canonical_shape,
        "canonical_shape_pixels": [
            [112 + x * 224, 112 + y * 224] for x, y in canonical_shape
        ],
        "image_size": 224,
        "n_landmarks": 15,
        "method": "Test GPA",
    }

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    return json_path


@pytest.fixture
def triangles_json(tmp_path):
    """
    JSON con triangulacion Delaunay para warping.
    """
    import json

    json_path = tmp_path / "triangles.json"

    # Triangulacion Delaunay sobre 15 puntos
    triangles = [
        [0, 2, 11], [0, 11, 12], [0, 12, 3],
        [2, 4, 11], [3, 12, 5], [4, 6, 10],
        [4, 10, 8], [4, 8, 2], [5, 8, 10],
        [5, 10, 9], [5, 9, 3], [6, 13, 10],
        [7, 9, 10], [7, 10, 14], [1, 6, 13],
        [1, 13, 14], [1, 14, 7], [0, 8, 9],
    ]

    data = {
        "triangles": triangles,
        "num_triangles": len(triangles),
        "method": "Delaunay",
    }

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    return json_path


@pytest.fixture
def warp_input_dataset(tmp_path):
    """
    Dataset de entrada para tests de warping.

    Estructura:
        tmp_path/input/
            COVID/images/*.png
            Normal/images/*.png
    """
    from PIL import Image

    input_dir = tmp_path / "input"
    classes = ["COVID", "Normal"]
    images_per_class = 2

    for cls in classes:
        cls_dir = input_dir / cls / "images"
        cls_dir.mkdir(parents=True)

        for i in range(images_per_class):
            # Imagen grayscale
            img_array = np.random.randint(50, 200, (224, 224), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='L')
            img.save(cls_dir / f"{cls}-{i+1:03d}.png")

    return input_dir


@pytest.fixture
def integration_test_setup(
    tmp_path,
    minimal_landmark_dataset,
    mock_landmark_checkpoint,
    canonical_shape_json,
    triangles_json
):
    """
    Setup completo para tests de integracion CLI.
    Combina todas las fixtures necesarias.
    """
    data_dir, csv_path, image_names = minimal_landmark_dataset
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    return {
        "data_dir": data_dir,
        "csv_path": csv_path,
        "checkpoint": mock_landmark_checkpoint,
        "canonical_json": canonical_shape_json,
        "triangles_json": triangles_json,
        "output_dir": output_dir,
        "image_names": image_names,
        "tmp_path": tmp_path,
    }
