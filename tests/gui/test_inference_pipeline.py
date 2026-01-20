"""Tests para inference_pipeline.py."""

import pytest
from pathlib import Path
import numpy as np
from PIL import Image
import tempfile


def test_validate_image_valid_formats():
    """Verifica que acepta formatos válidos."""
    from src_v2.gui.inference_pipeline import validate_image

    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

    for ext in valid_extensions:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            # Crear imagen válida
            img = Image.new('L', (224, 224))
            img.save(f.name)

            is_valid, error = validate_image(f.name)
            assert is_valid, f"Formato {ext} debería ser válido"

            Path(f.name).unlink()


def test_validate_image_invalid_extension():
    """Verifica que rechaza extensiones inválidas."""
    from src_v2.gui.inference_pipeline import validate_image

    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(b"not an image")
        f.flush()

        is_valid, error = validate_image(f.name)
        assert not is_valid
        assert "formato" in error.lower() or "extensión" in error.lower()

        Path(f.name).unlink()


def test_validate_image_too_small():
    """Verifica que rechaza imágenes < 100x100."""
    from src_v2.gui.inference_pipeline import validate_image

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img = Image.new('L', (50, 50))  # Muy pequeña
        img.save(f.name)

        is_valid, error = validate_image(f.name)
        assert not is_valid
        assert "pequeña" in error.lower() or "tamaño" in error.lower()

        Path(f.name).unlink()


def test_validate_image_nonexistent():
    """Verifica que maneja archivos inexistentes."""
    from src_v2.gui.inference_pipeline import validate_image

    is_valid, error = validate_image("/path/inexistente/imagen.png")
    assert not is_valid


def test_load_and_preprocess():
    """Verifica que load_and_preprocess retorna imagen 224x224 grayscale."""
    from src_v2.gui.inference_pipeline import load_and_preprocess

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img = Image.new('L', (512, 512))
        img.save(f.name)

        preprocessed = load_and_preprocess(f.name)

        assert preprocessed.shape == (224, 224)
        assert preprocessed.dtype == np.uint8

        Path(f.name).unlink()


def test_process_image_full_success():
    """Verifica que process_image_full retorna resultado completo."""
    from src_v2.gui.inference_pipeline import process_image_full

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img = Image.new('L', (224, 224))
        img.save(f.name)

        result = process_image_full(f.name)

        # Verificar estructura del resultado
        assert result['success'] is True
        assert 'original' in result
        assert 'landmarks' in result
        assert 'warped' in result
        assert 'gradcam' in result
        assert 'classification' in result
        assert 'predicted_class' in result
        assert 'metrics' in result
        assert 'inference_time' in result

        # Verificar que son PIL Images
        from PIL import Image
        assert isinstance(result['original'], Image.Image)
        assert isinstance(result['landmarks'], Image.Image)
        assert isinstance(result['warped'], Image.Image)
        assert isinstance(result['gradcam'], Image.Image)

        Path(f.name).unlink()


def test_process_image_quick_success():
    """Verifica que process_image_quick retorna clasificación rápida."""
    from src_v2.gui.inference_pipeline import process_image_quick

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img = Image.new('L', (224, 224))
        img.save(f.name)

        result = process_image_quick(f.name)

        assert result['success'] is True
        assert 'classification' in result
        assert 'predicted_class' in result
        assert 'inference_time' in result

        # No debe tener GradCAM
        assert 'gradcam' not in result

        Path(f.name).unlink()


def test_create_metrics_table():
    """Verifica que create_metrics_table retorna DataFrame válido."""
    from src_v2.gui.inference_pipeline import create_metrics_table
    import numpy as np
    import pandas as pd

    landmarks = np.random.uniform(0, 224, (15, 2))

    df = create_metrics_table(landmarks)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 15
    assert 'Landmark' in df.columns
    assert 'Grupo' in df.columns
    assert 'X' in df.columns
    assert 'Y' in df.columns
    assert 'Error Ref (px)' in df.columns
