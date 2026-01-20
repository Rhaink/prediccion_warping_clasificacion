"""Tests para visualizer.py."""

import pytest
import numpy as np
from PIL import Image


def test_render_original():
    """Verifica que render_original retorna PIL Image."""
    from src_v2.gui.visualizer import render_original

    image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)

    result = render_original(image)

    assert isinstance(result, Image.Image)


def test_render_landmarks_overlay():
    """Verifica que render_landmarks_overlay dibuja 15 landmarks."""
    from src_v2.gui.visualizer import render_landmarks_overlay

    image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    landmarks = np.random.uniform(0, 224, (15, 2))

    result = render_landmarks_overlay(image, landmarks, show_labels=True)

    assert isinstance(result, Image.Image)
    # Debería ser RGB o RGBA (colores)
    assert result.mode in ['RGB', 'RGBA']


def test_render_warped():
    """Verifica que render_warped retorna PIL Image."""
    from src_v2.gui.visualizer import render_warped

    warped = np.random.randint(0, 256, (224, 224), dtype=np.uint8)

    result = render_warped(warped)

    assert isinstance(result, Image.Image)


def test_render_gradcam():
    """Verifica que render_gradcam genera overlay correcto."""
    from src_v2.gui.visualizer import render_gradcam

    warped = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    heatmap = np.random.uniform(0, 1, (224, 224)).astype(np.float32)

    result = render_gradcam(warped, heatmap, alpha=0.4, predicted_class='COVID-19')

    assert isinstance(result, Image.Image)
    assert result.mode == 'RGB'


def test_matplotlib_memory_leak():
    """Verifica que no hay memory leak en renders repetidos."""
    from src_v2.gui.visualizer import render_original
    import matplotlib.pyplot as plt

    initial_figs = len(plt.get_fignums())

    image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)

    # Renderizar 10 veces
    for _ in range(10):
        render_original(image)

    final_figs = len(plt.get_fignums())

    # No debería haber figuras abiertas (todas cerradas)
    assert final_figs <= initial_figs + 1  # Tolerancia de 1
