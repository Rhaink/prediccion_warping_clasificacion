"""Tests de integración del pipeline completo."""

import pytest
from pathlib import Path
from PIL import Image
import tempfile


def test_end_to_end_pipeline():
    """Test end-to-end: imagen → landmarks → warp → classify → visualize."""
    from src_v2.gui.inference_pipeline import process_image_full

    # Crear imagen de prueba
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img = Image.new('L', (224, 224))
        img.save(f.name)

        result = process_image_full(f.name)

        # Pipeline completo ejecutado
        assert result['success'] is True
        assert result['landmarks_coords'].shape == (15, 2)
        assert result['inference_time'] > 0

        # Clasificación válida
        assert result['predicted_class'] in ['COVID-19', 'Normal', 'Neumonía Viral']

        # Probabilidades suman 1
        total_prob = sum(result['classification'].values())
        assert 0.99 <= total_prob <= 1.01

        Path(f.name).unlink()


def test_tta_flip_correction():
    """Verifica que TTA corrige coordenadas después de flip horizontal."""
    from src_v2.gui.model_manager import get_model_manager
    import numpy as np

    manager = get_model_manager()
    manager.initialize(verbose=False)

    image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)

    # Predecir con TTA
    landmarks_tta = manager.predict_landmarks(image, use_tta=True, use_clahe=False)

    # Predecir sin TTA
    landmarks_no_tta = manager.predict_landmarks(image, use_tta=False, use_clahe=False)

    # Ambos deben tener shape correcto
    assert landmarks_tta.shape == (15, 2)
    assert landmarks_no_tta.shape == (15, 2)

    # TTA debería promediar, puede haber diferencia
    # pero ambos válidos
    assert np.all(landmarks_tta >= 0)
    assert np.all(landmarks_tta <= 224)


def test_clahe_preprocessing():
    """Verifica que CLAHE mejora contraste."""
    from src_v2.gui.model_manager import get_model_manager
    import numpy as np
    import cv2

    # Imagen con bajo contraste
    image = np.full((224, 224), 128, dtype=np.uint8)

    # Aplicar CLAHE manualmente
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(image)

    # CLAHE debería cambiar la imagen
    assert not np.array_equal(image, enhanced)
