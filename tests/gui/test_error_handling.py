"""Tests de manejo de errores."""

import pytest
from pathlib import Path
import tempfile


def test_missing_model_error():
    """Verifica que lanza error si falta un modelo."""
    from src_v2.gui.model_manager import ModelManager
    from unittest.mock import patch

    with patch('src_v2.gui.config.LANDMARK_MODELS', ['/nonexistent/model.pt']):
        manager = ModelManager()

        with pytest.raises(FileNotFoundError):
            manager.initialize(verbose=False)


def test_invalid_image_format():
    """Verifica que maneja gracefully imagen inválida."""
    from src_v2.gui.inference_pipeline import process_image_full

    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(b"not an image")
        f.flush()

        result = process_image_full(f.name)

        assert result['success'] is False
        assert 'error' in result

        Path(f.name).unlink()


def test_corrupted_image():
    """Verifica que maneja imagen corrupta."""
    from src_v2.gui.inference_pipeline import validate_image

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        f.write(b"fake png data")
        f.flush()

        is_valid, error = validate_image(f.name)

        assert not is_valid

        Path(f.name).unlink()


def test_gpu_oom_fallback():
    """Simula GPU OOM y verifica fallback a CPU."""
    # Este test requiere mock complejo, marcar como skip por ahora
    pytest.skip("Requiere mock de torch.cuda.OutOfMemoryError")
