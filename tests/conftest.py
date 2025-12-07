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
    # Agregar ruido pequeno para variacion
    batch[1] += np.random.randn(30) * 0.01
    batch[2] += np.random.randn(30) * 0.01
    batch[3] += np.random.randn(30) * 0.01
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
