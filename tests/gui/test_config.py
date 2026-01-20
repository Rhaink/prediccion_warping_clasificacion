"""Verifica que config.py contiene valores válidos."""

import pytest
from src_v2.gui.config import (
    VALIDATED_METRICS,
    PER_LANDMARK_ERRORS,
    LANDMARK_COLORS,
    LANDMARK_GROUPS,
    CLASS_NAMES,
    CLASS_NAMES_ES,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_SIZE,
    MARGIN_SCALE,
    GRADCAM_ALPHA
)


def test_validated_metrics_structure():
    """Verifica que VALIDATED_METRICS tiene todas las claves esperadas."""
    required_keys = [
        'landmark_error_px', 'landmark_std_px', 'landmark_median_px',
        'classification_accuracy', 'classification_f1_macro',
        'classification_f1_weighted', 'clahe_clip', 'clahe_tile',
        'margin_scale', 'model_input_size', 'fill_rate'
    ]

    for key in required_keys:
        assert key in VALIDATED_METRICS, f"Falta clave en VALIDATED_METRICS: {key}"


def test_per_landmark_errors():
    """Verifica que hay error para cada uno de los 15 landmarks."""
    assert len(PER_LANDMARK_ERRORS) == 15

    # Verificar claves L1-L15
    for i in range(1, 16):
        key = f'L{i}'
        assert key in PER_LANDMARK_ERRORS, f"Falta error para {key}"
        assert isinstance(PER_LANDMARK_ERRORS[key], (int, float))
        assert 0 < PER_LANDMARK_ERRORS[key] < 10  # Error razonable en píxeles


def test_landmark_colors_valid():
    """Verifica que todos los colores son hex válidos."""
    import re
    hex_pattern = re.compile(r'^#[0-9A-Fa-f]{6}$')

    for group, color in LANDMARK_COLORS.items():
        assert hex_pattern.match(color), \
            f"Color inválido para grupo {group}: {color}"


def test_landmark_groups_complete():
    """Verifica que todos los 15 landmarks tienen grupo asignado."""
    assert len(LANDMARK_GROUPS) == 15

    for i in range(15):
        assert i in LANDMARK_GROUPS
        assert LANDMARK_GROUPS[i] in LANDMARK_COLORS


def test_class_names_consistency():
    """Verifica que hay misma cantidad de nombres en inglés y español."""
    assert len(CLASS_NAMES) == len(CLASS_NAMES_ES) == 3


def test_clahe_parameters():
    """Verifica que CLAHE tiene parámetros validados."""
    assert CLAHE_CLIP_LIMIT == 2.0  # Valor validado en GROUND_TRUTH.json
    assert CLAHE_TILE_SIZE == (4, 4)  # Mejor que (8, 8)


def test_margin_scale():
    """Verifica que margin_scale es el óptimo (1.05)."""
    assert MARGIN_SCALE == 1.05  # Validado en Session 25


def test_gradcam_alpha():
    """Verifica que alpha está en rango [0, 1]."""
    assert 0 <= GRADCAM_ALPHA <= 1
