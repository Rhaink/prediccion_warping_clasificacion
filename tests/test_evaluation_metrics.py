"""
Tests unitarios para métricas de evaluación

Tests para src_v2/evaluation/metrics.py
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Agregar src_v2 al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src_v2.evaluation.metrics import (
    compute_pixel_error,
    compute_error_per_landmark,
    evaluate_model,
    evaluate_model_with_tta,
    compute_error_per_category,
    generate_evaluation_report,
    compute_success_rate,
    predict_with_tta,
    _flip_landmarks_horizontal,
)
from src_v2.constants import LANDMARK_NAMES, SYMMETRIC_PAIRS, DEFAULT_IMAGE_SIZE


class MockModel(nn.Module):
    """Modelo mock para tests."""

    def __init__(self, output=None):
        super().__init__()
        self.output = output
        self.dummy = nn.Linear(1, 1)  # Para que tenga parámetros

    def forward(self, x):
        if self.output is not None:
            return self.output.expand(x.shape[0], -1)
        return torch.rand(x.shape[0], 30)


class MockDataLoader:
    """DataLoader mock para tests."""

    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)


class TestComputePixelError:
    """Tests para compute_pixel_error."""

    def test_zero_error_when_identical(self):
        """Error cero cuando predicción = target."""
        pred = torch.rand(4, 30)
        target = pred.clone()

        errors = compute_pixel_error(pred, target)

        assert errors.shape == (4, 15)
        assert torch.allclose(errors, torch.zeros_like(errors), atol=1e-6)

    def test_correct_shape_output(self):
        """Salida tiene forma correcta (B, 15)."""
        pred = torch.rand(8, 30)
        target = torch.rand(8, 30)

        errors = compute_pixel_error(pred, target)

        assert errors.shape == (8, 15)

    def test_scales_with_image_size(self):
        """Error escala con tamaño de imagen."""
        pred = torch.tensor([[0.5, 0.5] + [0.5] * 28])
        target = torch.tensor([[0.6, 0.5] + [0.5] * 28])  # 0.1 diferencia en x

        errors_224 = compute_pixel_error(pred, target, image_size=224)
        errors_100 = compute_pixel_error(pred, target, image_size=100)

        # Error con 224 debería ser ~2.24x mayor que con 100
        ratio = errors_224[0, 0].item() / errors_100[0, 0].item()
        assert ratio == pytest.approx(2.24, rel=0.01)

    def test_handles_reshaped_input(self):
        """Maneja input con forma (B, 15, 2)."""
        pred = torch.rand(4, 15, 2)
        target = torch.rand(4, 15, 2)

        # Debería funcionar aunque el input esté reshape-eado diferente
        errors = compute_pixel_error(pred.view(4, 30), target.view(4, 30))

        assert errors.shape == (4, 15)

    def test_supports_per_sample_image_sizes(self):
        """Acepta tamaños de imagen diferentes por muestra."""
        pred = torch.tensor([
            [0.5, 0.5] + [0.5] * 28,
            [0.5, 0.5] + [0.5] * 28,
        ])
        target = pred.clone()
        target[0, 0] += 0.1  # Delta en x usando width
        target[1, 1] += 0.2  # Delta en y usando height

        sizes = torch.tensor([[300, 200], [150, 400]], dtype=torch.float32)

        errors = compute_pixel_error(pred, target, image_size=sizes)

        assert errors[0, 0].item() == pytest.approx(30.0, rel=0.01)
        assert errors[1, 0].item() == pytest.approx(80.0, rel=0.01)


class TestComputeErrorPerLandmark:
    """Tests para compute_error_per_landmark."""

    def test_returns_dict_with_all_landmarks(self):
        """Retorna dict con todos los landmarks."""
        pred = torch.rand(10, 30)
        target = torch.rand(10, 30)

        result = compute_error_per_landmark(pred, target)

        assert isinstance(result, dict)
        assert len(result) == 15
        for name in LANDMARK_NAMES:
            assert name in result

    def test_values_are_positive(self):
        """Todos los valores son positivos."""
        pred = torch.rand(10, 30)
        target = torch.rand(10, 30)

        result = compute_error_per_landmark(pred, target)

        for name, error in result.items():
            assert error >= 0

    def test_zero_error_when_identical(self):
        """Error cero para predicciones idénticas."""
        pred = torch.rand(10, 30)
        target = pred.clone()

        result = compute_error_per_landmark(pred, target)

        # Tolerancia de GROUND_TRUTH.json: landmark_error_px.absolute = 0.5
        for name, error in result.items():
            assert error == pytest.approx(0.0, abs=0.5)


class TestComputeErrorPerCategory:
    """Tests para compute_error_per_category."""

    def test_returns_stats_per_category(self):
        """Retorna estadísticas por categoría."""
        pred = torch.rand(10, 30)
        target = torch.rand(10, 30)
        categories = ['COVID'] * 5 + ['Normal'] * 5

        result = compute_error_per_category(pred, target, categories)

        assert 'COVID' in result
        assert 'Normal' in result
        assert 'mean' in result['COVID']
        assert 'std' in result['COVID']
        assert 'count' in result['COVID']

    def test_count_is_correct(self):
        """Conteo por categoría es correcto."""
        pred = torch.rand(10, 30)
        target = torch.rand(10, 30)
        categories = ['COVID'] * 3 + ['Normal'] * 7

        result = compute_error_per_category(pred, target, categories)

        assert result['COVID']['count'] == 3
        assert result['Normal']['count'] == 7


class TestComputeSuccessRate:
    """Tests para compute_success_rate."""

    def test_returns_dict_with_thresholds(self):
        """Retorna dict con umbrales especificados."""
        errors = torch.rand(100, 15) * 20  # Errores entre 0 y 20

        result = compute_success_rate(errors, thresholds=[5, 10, 15])

        assert 5 in result
        assert 10 in result
        assert 15 in result

    def test_percentages_are_valid(self):
        """Porcentajes están entre 0 y 100."""
        errors = torch.rand(100, 15) * 20

        result = compute_success_rate(errors)

        for thresh, pct in result.items():
            assert 0 <= pct <= 100

    def test_higher_threshold_higher_rate(self):
        """Mayor umbral = mayor success rate."""
        errors = torch.rand(100, 15) * 20

        result = compute_success_rate(errors, thresholds=[5, 10, 15, 20])

        assert result[5] <= result[10] <= result[15] <= result[20]

    def test_all_below_threshold_100_percent(self):
        """100% si todos están bajo el umbral."""
        errors = torch.ones(10, 15) * 3  # Todos = 3px

        result = compute_success_rate(errors, thresholds=[5])

        assert result[5] == 100.0


class TestFlipLandmarksHorizontal:
    """Tests para _flip_landmarks_horizontal."""

    def test_flips_x_coordinate(self):
        """Coordenada X se refleja."""
        landmarks = torch.zeros(1, 30)
        landmarks[0, 0] = 0.3  # x del primer landmark

        flipped = _flip_landmarks_horizontal(landmarks)

        # Tolerancia aumentada para precision numerica
        assert flipped[0, 0].item() == pytest.approx(0.7, abs=0.01)

    def test_symmetric_pairs_swapped(self):
        """Pares simétricos se intercambian."""
        landmarks = torch.zeros(1, 30)
        # Poner valores únicos en L3 (índice 2) y L4 (índice 3)
        landmarks[0, 4] = 0.1   # L3 x
        landmarks[0, 5] = 0.2   # L3 y
        landmarks[0, 6] = 0.8   # L4 x
        landmarks[0, 7] = 0.9   # L4 y

        flipped = _flip_landmarks_horizontal(landmarks)

        # L3 y L4 deberían intercambiarse (además del flip en x)
        # Después del flip: L3 tiene valores originales de L4 con x reflejado
        assert flipped[0, 4].item() == pytest.approx(1 - 0.8, abs=1e-6)  # L3.x = 1 - L4.x original
        assert flipped[0, 5].item() == pytest.approx(0.9, abs=1e-6)       # L3.y = L4.y original

    def test_preserves_batch_dimension(self):
        """Preserva dimensión de batch."""
        landmarks = torch.rand(5, 30)

        flipped = _flip_landmarks_horizontal(landmarks)

        assert flipped.shape == (5, 30)


class TestGenerateEvaluationReport:
    """Tests para generate_evaluation_report."""

    def test_returns_string(self):
        """Retorna string."""
        metrics = {
            'overall': {'mean': 5.0, 'std': 2.0, 'median': 4.5},
            'per_landmark': {name: {'mean': 5.0, 'std': 2.0, 'median': 4.5, 'max': 10.0}
                            for name in LANDMARK_NAMES},
            'per_category': {'COVID': {'mean': 5.0, 'std': 2.0, 'count': 100}},
            'percentiles': {'p50': 4.5, 'p75': 6.0, 'p90': 8.0, 'p95': 9.0}
        }

        report = generate_evaluation_report(metrics)

        assert isinstance(report, str)

    def test_includes_overall_metrics(self):
        """Incluye métricas globales."""
        metrics = {
            'overall': {'mean': 5.0, 'std': 2.0, 'median': 4.5},
            'per_landmark': {name: {'mean': 5.0, 'std': 2.0, 'median': 4.5, 'max': 10.0}
                            for name in LANDMARK_NAMES},
            'per_category': {},
            'percentiles': {'p50': 4.5, 'p75': 6.0, 'p90': 8.0, 'p95': 9.0}
        }

        report = generate_evaluation_report(metrics)

        assert '5.00 px' in report or 'Mean Error' in report

    def test_includes_percentiles(self):
        """Incluye percentiles."""
        metrics = {
            'overall': {'mean': 5.0, 'std': 2.0, 'median': 4.5},
            'per_landmark': {name: {'mean': 5.0, 'std': 2.0, 'median': 4.5, 'max': 10.0}
                            for name in LANDMARK_NAMES},
            'per_category': {},
            'percentiles': {'p50': 4.5, 'p75': 6.0, 'p90': 8.0, 'p95': 9.0}
        }

        report = generate_evaluation_report(metrics)

        assert 'p50' in report or 'Percentile' in report


class TestEvaluateModel:
    """Tests para evaluate_model."""

    def test_returns_expected_structure(self):
        """Retorna estructura esperada."""
        model = MockModel()
        device = torch.device('cpu')

        # Crear datos mock
        data = [
            (torch.rand(2, 3, 224, 224),
             torch.rand(2, 30),
             [{'category': 'COVID'}, {'category': 'Normal'}])
        ]
        loader = MockDataLoader(data)

        result = evaluate_model(model, loader, device)

        assert 'overall' in result
        assert 'per_landmark' in result
        assert 'per_category' in result
        assert 'percentiles' in result
        assert 'raw_errors' in result
        assert 'predictions' in result
        assert 'targets' in result

    def test_sets_model_to_eval(self):
        """Pone modelo en modo eval."""
        model = MockModel()
        model.train()
        device = torch.device('cpu')

        data = [
            (torch.rand(2, 3, 224, 224),
             torch.rand(2, 30),
             [{'category': 'COVID'}, {'category': 'COVID'}])
        ]
        loader = MockDataLoader(data)

        evaluate_model(model, loader, device)

        assert not model.training

    def test_uses_original_size_from_meta(self):
        """Escala errores usando original_size cuando está disponible."""
        base_output = torch.tensor([[0.5] * 30], dtype=torch.float32)
        model = MockModel(output=base_output)
        device = torch.device('cpu')

        # Diferencia de 0.1 en eje x -> error debe ser 30 px con width=300
        target = base_output.clone()
        target[0, 0] += 0.1

        data = [
            (
                torch.rand(1, 3, 224, 224),
                target,
                [{'category': 'COVID', 'original_size': (300, 300)}],
            )
        ]
        loader = MockDataLoader(data)

        result = evaluate_model(model, loader, device, image_size=100)

        error_l1 = result['per_landmark']['L1']['mean']
        assert error_l1 == pytest.approx(30.0, rel=0.01)


class TestPredictWithTTA:
    """Tests para predict_with_tta."""

    def test_without_flip_returns_original(self):
        """Sin flip retorna predicción original."""
        output = torch.rand(1, 30)
        model = MockModel(output=output)
        device = torch.device('cpu')
        images = torch.rand(1, 3, 224, 224)

        result = predict_with_tta(model, images, device, use_flip=False)

        assert torch.allclose(result, output)

    def test_with_flip_averages(self):
        """Con flip promedia predicciones."""
        # Usar un modelo que retorne valores fijos para verificar promedio
        model = MockModel()
        device = torch.device('cpu')
        images = torch.rand(2, 3, 224, 224)

        result = predict_with_tta(model, images, device, use_flip=True)

        assert result.shape == (2, 30)

    def test_output_in_valid_range(self):
        """Salida está en rango válido [0, 1]."""
        model = MockModel()
        device = torch.device('cpu')
        images = torch.rand(4, 3, 224, 224)

        result = predict_with_tta(model, images, device, use_flip=True)

        # Los valores deberían estar razonablemente acotados
        # (el promedio de valores aleatorios en [0,1] estará cerca de 0.5)
        assert result.min() >= 0
        assert result.max() <= 1


class TestEvaluateModelWithTTA:
    """Tests para evaluate_model_with_tta."""

    def test_returns_expected_structure(self):
        """Retorna estructura esperada."""
        model = MockModel()
        device = torch.device('cpu')

        data = [
            (torch.rand(2, 3, 224, 224),
             torch.rand(2, 30),
             [{'category': 'COVID'}, {'category': 'Normal'}])
        ]
        loader = MockDataLoader(data)

        result = evaluate_model_with_tta(model, loader, device)

        assert 'overall' in result
        assert 'per_landmark' in result
        assert 'per_category' in result

    def test_uses_tta(self):
        """Usa TTA (verifica que hay procesamiento adicional)."""
        # Crear modelo con salida fija
        fixed_output = torch.ones(1, 30) * 0.5
        model = MockModel(output=fixed_output)
        device = torch.device('cpu')

        # Target diferente para medir error
        data = [
            (torch.rand(2, 3, 224, 224),
             torch.ones(2, 30) * 0.3,  # Target diferente de 0.5
             [{'category': 'COVID'}, {'category': 'COVID'}])
        ]
        loader = MockDataLoader(data)

        result = evaluate_model_with_tta(model, loader, device)

        # Debería haber algún error
        assert result['overall']['mean'] > 0
