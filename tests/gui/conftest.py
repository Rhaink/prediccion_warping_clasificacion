"""Fixtures compartidos para tests de GUI."""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile


@pytest.fixture
def dummy_image_224x224():
    """Imagen numpy 224x224 grayscale."""
    return np.random.randint(0, 256, (224, 224), dtype=np.uint8)


@pytest.fixture
def dummy_landmarks():
    """Landmarks dummy (15, 2) en rango [0, 224]."""
    return np.random.uniform(50, 174, (15, 2))


@pytest.fixture
def temp_image_file():
    """Archivo temporal de imagen PNG válida."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img = Image.new('L', (224, 224))
        img.save(f.name)
        yield f.name
        Path(f.name).unlink()


@pytest.fixture(scope="session")
def initialized_model_manager():
    """ModelManager inicializado (compartido entre tests)."""
    from src_v2.gui.model_manager import get_model_manager

    manager = get_model_manager()
    manager.initialize(verbose=False)

    return manager


@pytest.fixture
def mock_classification_result():
    """Resultado dummy de clasificación."""
    return {
        'COVID-19': 0.7,
        'Normal': 0.2,
        'Viral Pneumonia': 0.1
    }
