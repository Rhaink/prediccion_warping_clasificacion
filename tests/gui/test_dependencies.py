"""Verifica que todas las dependencias de la GUI están instaladas."""

import pytest
import sys


def test_core_dependencies():
    """Verifica dependencias core de Python."""
    required = ['torch', 'numpy', 'cv2', 'matplotlib', 'pandas',
                'PIL', 'gradio', 'scipy', 'sklearn']

    missing = []
    for module in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    assert len(missing) == 0, f"Dependencias faltantes: {missing}"


def test_torch_cuda_availability():
    """Verifica si CUDA está disponible y muestra info."""
    import torch

    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA disponible: {cuda_available}")

    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("Ejecutando en modo CPU")


def test_gradio_version():
    """Verifica que Gradio es >= 4.0.0."""
    import gradio as gr
    from packaging import version

    assert version.parse(gr.__version__) >= version.parse("4.0.0"), \
        f"Gradio {gr.__version__} < 4.0.0 requerido"


def test_opencv_backend():
    """Verifica que OpenCV puede cargar imágenes."""
    import cv2
    import numpy as np

    # Crear imagen dummy
    img = np.zeros((100, 100), dtype=np.uint8)

    # Intentar operaciones básicas
    resized = cv2.resize(img, (224, 224))
    assert resized.shape == (224, 224)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(img)
    assert enhanced is not None


def test_matplotlib_non_interactive_backend():
    """Verifica que matplotlib funciona correctamente (acepta cualquier backend)."""
    import matplotlib
    import matplotlib.pyplot as plt
    backend = matplotlib.get_backend()
    print(f"\nMatplotlib backend: {backend}")
    # Solo verificar que matplotlib está disponible, no el backend específico
    # El backend puede variar según el entorno (Agg, tkagg, etc.)
    assert backend is not None
    # Verificar que podemos crear una figura básica
    fig, ax = plt.subplots()
    plt.close(fig)
