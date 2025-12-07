"""
Tests unitarios para funciones de pérdida (losses)
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Agregar src_v2 al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src_v2.models.losses import (
    WingLoss,
    WeightedWingLoss,
    CentralAlignmentLoss,
    SoftSymmetryLoss,
    CombinedLandmarkLoss,
    get_landmark_weights,
)
from src_v2.constants import SYMMETRIC_PAIRS, CENTRAL_LANDMARKS


class TestWingLoss:
    """Tests para WingLoss básico."""

    def test_init_default_params(self):
        """Verifica inicialización con parámetros por defecto (normalizados)."""
        loss = WingLoss()
        # Con normalized=True (default), omega y epsilon se dividen por image_size=224
        assert loss.omega == pytest.approx(10.0 / 224, rel=1e-4)
        assert loss.epsilon == pytest.approx(2.0 / 224, rel=1e-4)
        assert hasattr(loss, 'C')

    def test_init_custom_params(self):
        """Verifica inicialización con parámetros personalizados (no normalizados)."""
        loss = WingLoss(omega=5.0, epsilon=1.0, normalized=False)
        assert loss.omega == 5.0
        assert loss.epsilon == 1.0

    def test_zero_error_zero_loss(self):
        """Error cero produce loss cero."""
        loss = WingLoss()
        pred = torch.rand(4, 30)
        target = pred.clone()

        result = loss(pred, target)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_small_error_log_regime(self):
        """Errores pequeños (<omega) usan régimen logarítmico."""
        # Usar normalized=False para verificar el comportamiento con valores absolutos
        loss = WingLoss(omega=10.0, epsilon=2.0, normalized=False)

        # Error pequeño = 0.1
        pred = torch.tensor([[0.5]])
        target = torch.tensor([[0.6]])

        result = loss(pred, target)

        # Para error=0.1 < omega=10: loss = omega * log(1 + |diff|/epsilon)
        expected = 10.0 * np.log(1 + 0.1 / 2.0)
        assert result.item() == pytest.approx(expected, rel=1e-4)

    def test_large_error_linear_regime(self):
        """Errores grandes (>=omega) usan régimen lineal."""
        # Usar normalized=False para verificar el comportamiento con valores absolutos
        loss = WingLoss(omega=10.0, epsilon=2.0, normalized=False)

        # Error grande = 15
        pred = torch.tensor([[0.0]])
        target = torch.tensor([[15.0]])

        result = loss(pred, target)

        # Para error=15 >= omega=10: loss = |diff| - C
        C = 10.0 - 10.0 * np.log(1 + 10.0 / 2.0)
        expected = 15.0 - C
        assert result.item() == pytest.approx(expected, rel=1e-4)

    def test_continuity_at_omega(self):
        """Wing Loss es continuo en omega."""
        # Usar normalized=False para verificar continuidad con valores absolutos
        loss = WingLoss(omega=10.0, epsilon=2.0, normalized=False)

        # Justo debajo de omega
        pred1 = torch.tensor([[0.0]])
        target1 = torch.tensor([[9.999]])
        result1 = loss(pred1, target1)

        # Justo arriba de omega
        pred2 = torch.tensor([[0.0]])
        target2 = torch.tensor([[10.001]])
        result2 = loss(pred2, target2)

        # Deben ser muy cercanos (continuidad)
        assert abs(result1.item() - result2.item()) < 0.01

    def test_shape_batch(self):
        """Funciona con diferentes tamaños de batch."""
        loss = WingLoss()

        for batch_size in [1, 8, 32]:
            pred = torch.rand(batch_size, 30)
            target = torch.rand(batch_size, 30)
            result = loss(pred, target)

            assert result.shape == ()  # Escalar
            assert not torch.isnan(result)

    def test_gradients(self):
        """Los gradientes se propagan correctamente."""
        loss = WingLoss()

        pred = torch.rand(4, 30, requires_grad=True)
        target = torch.rand(4, 30)

        result = loss(pred, target)
        result.backward()

        assert pred.grad is not None
        assert pred.grad.shape == pred.shape
        assert not torch.isnan(pred.grad).any()


class TestWeightedWingLoss:
    """Tests para WeightedWingLoss."""

    def test_uniform_weights_equals_wing_loss(self):
        """Con pesos uniformes, debe ser similar a WingLoss estándar."""
        wing = WingLoss()
        weighted = WeightedWingLoss(weights=torch.ones(15))

        pred = torch.rand(4, 30)
        target = torch.rand(4, 30)

        result_wing = wing(pred, target)
        result_weighted = weighted(pred, target)

        # Pueden diferir por el reshape, pero deben ser cercanos
        assert abs(result_wing.item() - result_weighted.item()) < 0.5

    def test_weights_affect_loss(self):
        """Landmarks con mayor peso contribuyen más al loss."""
        # Crear pesos: solo L0 tiene peso (peso=10), resto peso=0
        weights = torch.zeros(15)
        weights[0] = 10.0

        loss = WeightedWingLoss(weights=weights)

        # Error solo en L0
        pred1 = torch.zeros(1, 30)
        target1 = torch.zeros(1, 30)
        target1[0, 0] = 0.1  # Error en L0_x

        # Error solo en L1 (peso 0)
        pred2 = torch.zeros(1, 30)
        target2 = torch.zeros(1, 30)
        target2[0, 2] = 0.1  # Error en L1_x

        result1 = loss(pred1, target1)
        result2 = loss(pred2, target2)

        # Error en L0 (peso alto) debe dar mayor loss
        assert result1.item() > result2.item()

    def test_default_weights_are_uniform(self):
        """Sin pesos especificados, usa uniformes."""
        loss = WeightedWingLoss()

        assert torch.allclose(loss.weights, torch.ones(15))

    def test_shape_validation(self):
        """Funciona con entrada (B, 30)."""
        loss = WeightedWingLoss()

        pred = torch.rand(8, 30)
        target = torch.rand(8, 30)

        result = loss(pred, target)
        assert result.shape == ()


class TestCentralAlignmentLoss:
    """Tests para CentralAlignmentLoss."""

    def test_perfectly_aligned_zero_loss(self):
        """Puntos perfectamente alineados dan loss cercano a cero."""
        loss = CentralAlignmentLoss(image_size=224)

        # Crear predicciones donde L9, L10, L11 están exactamente sobre eje L1-L2
        pred = torch.zeros(1, 30)

        # L1 en (0.5, 0.2), L2 en (0.5, 0.8) - eje vertical
        pred[0, 0:2] = torch.tensor([0.5, 0.2])  # L1
        pred[0, 2:4] = torch.tensor([0.5, 0.8])  # L2

        # L9, L10, L11 sobre el eje (x=0.5)
        pred[0, 16:18] = torch.tensor([0.5, 0.35])  # L9 (idx 8)
        pred[0, 18:20] = torch.tensor([0.5, 0.50])  # L10 (idx 9)
        pred[0, 20:22] = torch.tensor([0.5, 0.65])  # L11 (idx 10)

        result = loss(pred)
        assert result.item() < 1.0  # Casi cero

    def test_misaligned_positive_loss(self):
        """Puntos desalineados dan loss positivo."""
        loss = CentralAlignmentLoss(image_size=224)

        pred = torch.zeros(1, 30)

        # L1 en (0.5, 0.2), L2 en (0.5, 0.8) - eje vertical
        pred[0, 0:2] = torch.tensor([0.5, 0.2])
        pred[0, 2:4] = torch.tensor([0.5, 0.8])

        # L9, L10, L11 FUERA del eje (x=0.6, no 0.5)
        pred[0, 16:18] = torch.tensor([0.6, 0.35])  # L9 desplazado
        pred[0, 18:20] = torch.tensor([0.6, 0.50])  # L10 desplazado
        pred[0, 20:22] = torch.tensor([0.6, 0.65])  # L11 desplazado

        result = loss(pred)

        # Desplazamiento de 0.1 normalizado (ya no se multiplica por image_size)
        # Loss en escala normalizada: ~0.1
        assert result.item() > 0.05  # Loss significativo en escala normalizada

    def test_only_affects_central_landmarks(self):
        """Solo evalúa L9, L10, L11 (índices 8, 9, 10)."""
        loss = CentralAlignmentLoss(image_size=224)

        # Predicción base con centrales alineados
        pred1 = torch.zeros(1, 30)
        pred1[0, 0:2] = torch.tensor([0.5, 0.2])
        pred1[0, 2:4] = torch.tensor([0.5, 0.8])
        pred1[0, 16:22] = torch.tensor([0.5, 0.35, 0.5, 0.50, 0.5, 0.65])

        # Misma predicción pero con otros landmarks desplazados
        pred2 = pred1.clone()
        pred2[0, 4:16] = torch.rand(12)  # L3-L8 aleatorios
        pred2[0, 22:30] = torch.rand(8)  # L12-L15 aleatorios

        result1 = loss(pred1)
        result2 = loss(pred2)

        # El loss debe ser igual porque solo evalúa centrales
        assert result1.item() == pytest.approx(result2.item(), rel=1e-4)

    def test_gradients(self):
        """Los gradientes se propagan."""
        loss = CentralAlignmentLoss()

        pred = torch.rand(4, 30, requires_grad=True)
        result = loss(pred)
        result.backward()

        assert pred.grad is not None


class TestSoftSymmetryLoss:
    """Tests para SoftSymmetryLoss."""

    def test_symmetric_within_margin_zero_loss(self):
        """Asimetría dentro del margen da loss cero."""
        loss = SoftSymmetryLoss(margin=6.0, image_size=224)

        pred = torch.zeros(1, 30)

        # Eje vertical en x=0.5
        pred[0, 0:2] = torch.tensor([0.5, 0.2])  # L1
        pred[0, 2:4] = torch.tensor([0.5, 0.8])  # L2

        # Par L3-L4 (índices 2,3) simétrico
        pred[0, 4:6] = torch.tensor([0.4, 0.3])  # L3 izquierda
        pred[0, 6:8] = torch.tensor([0.6, 0.3])  # L4 derecha (simétrico)

        # Llenar resto de pares de forma simétrica
        for left_idx, right_idx in SYMMETRIC_PAIRS:
            # Posición base según eje
            y_pos = 0.3 + left_idx * 0.05
            x_offset = 0.1
            pred[0, left_idx*2:left_idx*2+2] = torch.tensor([0.5 - x_offset, y_pos])
            pred[0, right_idx*2:right_idx*2+2] = torch.tensor([0.5 + x_offset, y_pos])

        result = loss(pred)
        assert result.item() < 1.0  # Cercano a cero

    def test_asymmetry_beyond_margin_positive_loss(self):
        """Asimetría mayor al margen produce loss positivo."""
        loss = SoftSymmetryLoss(margin=6.0, image_size=224)

        pred = torch.zeros(1, 30)

        # Eje vertical
        pred[0, 0:2] = torch.tensor([0.5, 0.2])
        pred[0, 2:4] = torch.tensor([0.5, 0.8])

        # Par L3-L4 MUY asimétrico (30 px de diferencia >> 6 px margen)
        pred[0, 4:6] = torch.tensor([0.4, 0.3])   # L3: dist = 0.1*224 = 22.4 px
        pred[0, 6:8] = torch.tensor([0.7, 0.3])   # L4: dist = 0.2*224 = 44.8 px
        # Asimetría = 22.4 px > 6 px margen

        result = loss(pred)
        assert result.item() > 0  # Loss positivo

    def test_margin_respected(self):
        """Asimetrías pequeñas (< margen) no penalizan."""
        margin_px = 10.0
        loss = SoftSymmetryLoss(margin=margin_px, image_size=224)

        pred = torch.zeros(1, 30)
        pred[0, 0:2] = torch.tensor([0.5, 0.2])
        pred[0, 2:4] = torch.tensor([0.5, 0.8])

        # Asimetría de ~5 px (menor que margen de 10)
        offset_left = 0.1
        offset_right = 0.1 + (5.0 / 224)  # 5 px diferencia

        for left_idx, right_idx in SYMMETRIC_PAIRS:
            y_pos = 0.3
            pred[0, left_idx*2:left_idx*2+2] = torch.tensor([0.5 - offset_left, y_pos])
            pred[0, right_idx*2:right_idx*2+2] = torch.tensor([0.5 + offset_right, y_pos])

        result = loss(pred)
        # Con margen de 10 y asimetría de 5, no debería penalizar mucho
        assert result.item() < 10.0  # Loss bajo

    def test_only_symmetric_pairs(self):
        """Solo evalúa los pares simétricos definidos."""
        assert len(SYMMETRIC_PAIRS) == 5
        expected_pairs = [(2, 3), (4, 5), (6, 7), (11, 12), (13, 14)]
        assert SYMMETRIC_PAIRS == expected_pairs


class TestCombinedLandmarkLoss:
    """Tests para CombinedLandmarkLoss."""

    def test_returns_dict(self):
        """Retorna diccionario con todas las pérdidas."""
        loss = CombinedLandmarkLoss()

        pred = torch.rand(4, 30)
        target = torch.rand(4, 30)

        result = loss(pred, target)

        assert isinstance(result, dict)
        assert 'total' in result
        assert 'wing' in result
        assert 'central' in result
        assert 'symmetry' in result

    def test_total_is_weighted_sum(self):
        """Total es suma ponderada de componentes."""
        central_w = 0.3
        sym_w = 0.1

        loss = CombinedLandmarkLoss(
            central_weight=central_w,
            symmetry_weight=sym_w
        )

        pred = torch.rand(4, 30)
        target = torch.rand(4, 30)

        result = loss(pred, target)

        expected_total = (
            result['wing'] +
            central_w * result['central'] +
            sym_w * result['symmetry']
        )

        assert result['total'].item() == pytest.approx(expected_total.item(), rel=1e-4)

    def test_custom_weights(self):
        """Acepta pesos de landmarks personalizados."""
        custom_weights = torch.ones(15) * 2.0

        loss = CombinedLandmarkLoss(landmark_weights=custom_weights)

        pred = torch.rand(4, 30)
        target = torch.rand(4, 30)

        result = loss(pred, target)
        assert result['total'].item() > 0


class TestGetLandmarkWeights:
    """Tests para función get_landmark_weights."""

    def test_uniform_strategy(self):
        """Estrategia uniforme retorna unos."""
        weights = get_landmark_weights('uniform')

        assert weights.shape == (15,)
        assert torch.allclose(weights, torch.ones(15))

    def test_inverse_variance_strategy(self):
        """Estrategia inverse_variance retorna pesos específicos."""
        weights = get_landmark_weights('inverse_variance')

        assert weights.shape == (15,)
        # L9 (central, fácil) debe tener mayor peso
        assert weights[8] > weights[13]  # L9 > L14

    def test_custom_strategy(self):
        """Estrategia custom retorna pesos personalizados."""
        weights = get_landmark_weights('custom')

        assert weights.shape == (15,)
        # L14, L15 (costofrénicos) tienen mayor peso
        assert weights[13] == 2.0
        assert weights[14] == 2.0

    def test_unknown_strategy_returns_uniform(self):
        """Estrategia desconocida retorna uniformes."""
        weights = get_landmark_weights('unknown_strategy')

        assert torch.allclose(weights, torch.ones(15))


class TestNumericalStability:
    """Tests de estabilidad numérica."""

    def test_wing_loss_no_nan(self):
        """WingLoss no produce NaN."""
        loss = WingLoss()

        # Valores extremos
        pred = torch.tensor([[0.0, 1.0, 0.5]])
        target = torch.tensor([[1.0, 0.0, 0.5]])

        result = loss(pred, target)
        assert not torch.isnan(result)
        assert not torch.isinf(result)

    def test_central_loss_degenerate_axis(self):
        """CentralAlignmentLoss maneja eje degenerado (L1=L2)."""
        loss = CentralAlignmentLoss()

        pred = torch.zeros(1, 30)
        # L1 = L2 (eje de longitud cero)
        pred[0, 0:4] = torch.tensor([0.5, 0.5, 0.5, 0.5])

        result = loss(pred)

        # Debe manejar división por cero (eps=1e-8)
        assert not torch.isnan(result)
        assert not torch.isinf(result)

    def test_symmetry_loss_vertical_axis(self):
        """SoftSymmetryLoss funciona con ejes de diferentes orientaciones."""
        loss = SoftSymmetryLoss()

        # Eje horizontal
        pred = torch.zeros(1, 30)
        pred[0, 0:2] = torch.tensor([0.2, 0.5])  # L1
        pred[0, 2:4] = torch.tensor([0.8, 0.5])  # L2

        result = loss(pred)

        assert not torch.isnan(result)
        assert not torch.isinf(result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
