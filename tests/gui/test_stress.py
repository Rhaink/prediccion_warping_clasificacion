"""Tests de stress y performance."""

import pytest
from pathlib import Path
from PIL import Image
import tempfile
import time


@pytest.mark.slow
def test_sequential_processing():
    """Procesa 10 imágenes secuencialmente sin memory leak."""
    from src_v2.gui.inference_pipeline import process_image_full
    import psutil
    import os

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img = Image.new('L', (224, 224))
        img.save(f.name)

        for i in range(10):
            result = process_image_full(f.name)
            assert result['success'] is True

        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Memory leak < 100 MB después de 10 imágenes
        memory_increase = final_memory - initial_memory
        print(f"\nMemory increase: {memory_increase:.1f} MB")
        assert memory_increase < 100

        Path(f.name).unlink()


@pytest.mark.slow
def test_inference_time_reasonable():
    """Verifica que tiempo de inferencia es razonable."""
    from src_v2.gui.inference_pipeline import process_image_quick

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img = Image.new('L', (224, 224))
        img.save(f.name)

        start = time.time()
        result = process_image_quick(f.name)
        elapsed = time.time() - start

        # Primera vez puede tardar más (carga modelos)
        # Pero no debería exceder 30 segundos
        assert elapsed < 30

        # Segunda vez debería ser más rápido
        start = time.time()
        result = process_image_quick(f.name)
        elapsed = time.time() - start

        # < 5 segundos en CPU, < 2s en GPU
        assert elapsed < 5

        Path(f.name).unlink()
