"""Tests para gradcam_utils.py."""

import pytest
import torch
import numpy as np


def test_gradcam_initialization():
    """Verifica que GradCAM se inicializa correctamente."""
    from src_v2.gui.gradcam_utils import GradCAM
    from src_v2.gui.model_manager import get_model_manager

    manager = get_model_manager()
    manager.initialize(verbose=False)

    gradcam = GradCAM(manager.classifier, target_layer='layer4')

    assert gradcam.model is not None
    assert gradcam.target_layer is not None


def test_gradcam_generate():
    """Verifica que generate() retorna heatmap válido."""
    from src_v2.gui.gradcam_utils import GradCAM
    from src_v2.gui.model_manager import get_model_manager
    import torch

    manager = get_model_manager()
    manager.initialize(verbose=False)

    gradcam = GradCAM(manager.classifier, target_layer='layer4')

    # Tensor dummy (1, 3, 224, 224)
    input_tensor = torch.randn(1, 3, 224, 224).to(manager.device)

    heatmap = gradcam.generate(input_tensor, target_class=0)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (224, 224)
    assert heatmap.min() >= 0
    assert heatmap.max() <= 1


def test_gradcam_no_division_by_zero():
    """Verifica que maneja heatmap constante sin división por cero."""
    from src_v2.gui.gradcam_utils import GradCAM
    from src_v2.gui.model_manager import get_model_manager
    import torch

    manager = get_model_manager()
    manager.initialize(verbose=False)

    gradcam = GradCAM(manager.classifier, target_layer='layer4')

    input_tensor = torch.randn(1, 3, 224, 224).to(manager.device)

    # No debería lanzar excepción
    heatmap = gradcam.generate(input_tensor)

    assert heatmap is not None
