"""Verifica que todos los modelos requeridos existen."""

import pytest
from pathlib import Path
import os


def test_development_mode_paths():
    """Verifica rutas en modo development (sin COVID_DEMO_MODELS_DIR)."""
    # Asegurar que no estamos en modo deployment
    if 'COVID_DEMO_MODELS_DIR' in os.environ:
        pytest.skip("Ejecutar sin COVID_DEMO_MODELS_DIR para test development")

    from src_v2.gui.config import (
        LANDMARK_MODELS,
        CANONICAL_SHAPE,
        DELAUNAY_TRIANGLES,
        CLASSIFIER_CHECKPOINT
    )

    # Verificar landmarks (4 modelos)
    for i, model_path in enumerate(LANDMARK_MODELS):
        assert Path(model_path).exists(), \
            f"Landmark model {i+1} no encontrado: {model_path}"

    # Verificar canonical shape
    assert Path(CANONICAL_SHAPE).exists(), \
        f"Canonical shape no encontrado: {CANONICAL_SHAPE}"

    # Verificar triangulación
    assert Path(DELAUNAY_TRIANGLES).exists(), \
        f"Triangulación no encontrada: {DELAUNAY_TRIANGLES}"

    # Verificar clasificador
    assert Path(CLASSIFIER_CHECKPOINT).exists(), \
        f"Clasificador no encontrado: {CLASSIFIER_CHECKPOINT}"


def test_deployment_mode_paths(tmp_path, monkeypatch):
    """Verifica rutas en modo deployment (con COVID_DEMO_MODELS_DIR)."""
    # Crear estructura de deployment dummy
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    (models_dir / "landmarks").mkdir()
    (models_dir / "classifier").mkdir()
    (models_dir / "shape_analysis").mkdir()

    # Crear archivos dummy
    for seed in ['seed123', 'seed321', 'seed111', 'seed666']:
        (models_dir / f"landmarks/{seed}_final.pt").touch()

    (models_dir / "classifier/best_classifier.pt").touch()
    (models_dir / "shape_analysis/canonical_shape_gpa.json").touch()
    (models_dir / "shape_analysis/canonical_delaunay_triangles.json").touch()

    # Configurar env var
    monkeypatch.setenv('COVID_DEMO_MODELS_DIR', str(models_dir))

    # Reload config para aplicar cambio
    import importlib
    import src_v2.gui.config as cfg
    importlib.reload(cfg)

    # Verificar que detectó modo deployment
    assert cfg.MODELS_DIR is not None

    # Verificar rutas
    for model_path in cfg.LANDMARK_MODELS:
        assert Path(model_path).exists()

    assert Path(cfg.CANONICAL_SHAPE).exists()
    assert Path(cfg.DELAUNAY_TRIANGLES).exists()
    assert Path(cfg.CLASSIFIER_CHECKPOINT).exists()


def test_model_file_sizes():
    """Verifica que los modelos tienen tamaño razonable (no vacíos o corruptos)."""
    from src_v2.gui.config import LANDMARK_MODELS, CLASSIFIER_CHECKPOINT

    # Landmarks deben ser ~46 MB cada uno
    for model_path in LANDMARK_MODELS:
        size_mb = Path(model_path).stat().st_size / (1024 * 1024)
        assert 40 < size_mb < 60, \
            f"Landmark model tamaño sospechoso: {size_mb:.1f} MB (esperado ~46 MB)"

    # Clasificador debe ser ~43 MB
    classifier_size_mb = Path(CLASSIFIER_CHECKPOINT).stat().st_size / (1024 * 1024)
    assert 40 < classifier_size_mb < 50, \
        f"Clasificador tamaño sospechoso: {classifier_size_mb:.1f} MB"


def test_examples_exist():
    """Verifica que existen imágenes de ejemplo para la demo."""
    from src_v2.gui.config import EXAMPLES_DIR

    expected_examples = [
        'covid_example.png',
        'normal_example.png',
        'viral_example.png'
    ]

    if not EXAMPLES_DIR.exists():
        pytest.skip("Directorio de ejemplos no existe (opcional)")

    existing = []
    for example in expected_examples:
        if (EXAMPLES_DIR / example).exists():
            existing.append(example)

    print(f"\nEjemplos encontrados: {existing}")
    # Al menos un ejemplo debe existir
    assert len(existing) > 0, "No se encontraron imágenes de ejemplo"
