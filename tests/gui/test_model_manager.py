"""Tests para ModelManager singleton."""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock


def test_model_manager_singleton():
    """Verifica que ModelManager es singleton."""
    from src_v2.gui.model_manager import get_model_manager

    manager1 = get_model_manager()
    manager2 = get_model_manager()

    assert manager1 is manager2  # Misma instancia


def test_model_manager_initialize_once():
    """Verifica que initialize() solo carga modelos una vez."""
    from src_v2.gui.model_manager import get_model_manager

    manager = get_model_manager()

    # Primera inicialización
    manager.initialize(verbose=False)
    assert manager._initialized is True

    # Contar modelos cargados
    initial_count = len(manager.landmark_models)

    # Segunda llamada no debe duplicar
    manager.initialize(verbose=False)
    assert len(manager.landmark_models) == initial_count


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requiere GPU")
def test_model_manager_gpu_detection():
    """Verifica que detecta GPU correctamente."""
    from src_v2.gui.model_manager import get_model_manager

    manager = get_model_manager()
    manager.initialize(verbose=False)

    assert manager.device.type == 'cuda'


def test_model_manager_cpu_fallback():
    """Verifica fallback a CPU si no hay GPU."""
    from src_v2.gui.model_manager import get_model_manager

    with patch('torch.cuda.is_available', return_value=False):
        manager = get_model_manager()
        manager._initialized = False  # Reset
        manager.initialize(verbose=False)

        assert manager.device.type == 'cpu'


def test_predict_landmarks_shape():
    """Verifica que predict_landmarks retorna shape (15, 2)."""
    from src_v2.gui.model_manager import get_model_manager
    import numpy as np

    manager = get_model_manager()
    manager.initialize(verbose=False)

    # Imagen dummy 224x224
    image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)

    landmarks = manager.predict_landmarks(image, use_tta=True, use_clahe=True)

    assert landmarks.shape == (15, 2)
    assert landmarks.dtype == np.float32 or landmarks.dtype == np.float64

    # Verificar que coordenadas están en rango [0, 224]
    assert np.all(landmarks >= 0)
    assert np.all(landmarks <= 224)


def test_warp_image_output_size():
    """Verifica que warp_image retorna imagen 224x224."""
    from src_v2.gui.model_manager import get_model_manager
    import numpy as np

    manager = get_model_manager()
    manager.initialize(verbose=False)

    image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    landmarks = np.random.uniform(50, 174, (15, 2))  # Dentro de bounds

    warped = manager.warp_image(image, landmarks)

    assert warped.shape == (224, 224)
    assert warped.dtype == np.uint8


def test_classify_with_gradcam_returns_dict():
    """Verifica que classify_with_gradcam retorna estructura correcta."""
    from src_v2.gui.model_manager import get_model_manager
    import numpy as np

    manager = get_model_manager()
    manager.initialize(verbose=False)

    warped = np.random.randint(0, 256, (224, 224), dtype=np.uint8)

    probs, heatmap, pred_idx = manager.classify_with_gradcam(warped)

    # Verificar probabilidades
    assert isinstance(probs, dict)
    assert len(probs) == 3
    assert sum(probs.values()) == pytest.approx(1.0, abs=0.01)

    # Verificar heatmap
    assert heatmap.shape == (224, 224)
    assert heatmap.dtype == np.float32 or heatmap.dtype == np.float64

    # Verificar índice predicho
    assert 0 <= pred_idx < 3
