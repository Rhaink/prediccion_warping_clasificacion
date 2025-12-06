"""
Tests unitarios para el modulo transforms.
"""

import sys
from pathlib import Path

# Agregar src_v2 al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import torch
from PIL import Image

from src_v2.data.transforms import (
    LandmarkTransform,
    TrainTransform,
    ValTransform,
    SYMMETRIC_PAIRS,
    get_train_transforms,
    get_val_transforms,
)


class TestLandmarkTransform:
    """Tests para la clase base LandmarkTransform."""

    def test_normalize_coords(self):
        """Verifica que las coordenadas se normalizan correctamente a [0, 1]."""
        transform = LandmarkTransform()
        landmarks = np.array([
            [0, 0],       # Esquina superior izquierda
            [299, 299],   # Esquina inferior derecha
            [149.5, 149.5],  # Centro
        ], dtype=np.float32)

        normalized = transform.normalize_coords(landmarks, (299, 299))

        np.testing.assert_almost_equal(normalized[0], [0, 0], decimal=3)
        np.testing.assert_almost_equal(normalized[1], [1, 1], decimal=3)
        np.testing.assert_almost_equal(normalized[2], [0.5, 0.5], decimal=3)

    def test_normalize_coords_clips_to_range(self):
        """Verifica que valores fuera de rango se clampean a [0, 1]."""
        transform = LandmarkTransform()
        landmarks = np.array([
            [-10, -10],    # Fuera de rango
            [310, 310],    # Fuera de rango
        ], dtype=np.float32)

        normalized = transform.normalize_coords(landmarks, (299, 299))

        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_resize_image(self):
        """Verifica que el resize funciona correctamente."""
        transform = LandmarkTransform(output_size=224)
        image = Image.new('RGB', (299, 299), color='red')

        resized = transform.resize_image(image)

        assert resized.size == (224, 224)

    def test_image_to_tensor_shape(self):
        """Verifica forma del tensor de imagen."""
        transform = LandmarkTransform()
        image = Image.new('RGB', (224, 224))

        tensor = transform.image_to_tensor(image)

        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_landmarks_to_tensor(self):
        """Verifica conversion de landmarks a tensor flat."""
        transform = LandmarkTransform()
        landmarks = np.random.rand(15, 2).astype(np.float32)

        tensor = transform.landmarks_to_tensor(landmarks)

        assert tensor.shape == (30,)
        assert tensor.dtype == torch.float32


class TestHorizontalFlip:
    """Tests para el flip horizontal."""

    @pytest.fixture
    def sample_data(self):
        """Genera datos de prueba."""
        image = Image.new('RGB', (224, 224), color='blue')
        # Landmarks simétricos de prueba
        landmarks = np.array([
            [0.5, 0.1],   # L1 - centro superior
            [0.5, 0.9],   # L2 - centro inferior
            [0.2, 0.2],   # L3 - izquierda
            [0.8, 0.2],   # L4 - derecha
            [0.3, 0.4],   # L5 - izquierda
            [0.7, 0.4],   # L6 - derecha
            [0.25, 0.6],  # L7 - izquierda
            [0.75, 0.6],  # L8 - derecha
            [0.5, 0.25],  # L9 - central
            [0.5, 0.5],   # L10 - central
            [0.5, 0.75],  # L11 - central
            [0.15, 0.15], # L12 - izquierda
            [0.85, 0.15], # L13 - derecha
            [0.1, 0.85],  # L14 - izquierda
            [0.9, 0.85],  # L15 - derecha
        ], dtype=np.float32)
        return image, landmarks

    def test_flip_reflects_x_coordinates(self, sample_data):
        """Verifica que las coordenadas X se reflejan (1 - x)."""
        image, landmarks = sample_data
        transform = TrainTransform(flip_prob=1.0, rotation_degrees=0)

        _, flipped = transform.horizontal_flip(image, landmarks.copy())

        # L1 y L2 (centrales) deben tener x = 1 - x
        np.testing.assert_almost_equal(flipped[0, 0], 1 - landmarks[0, 0], decimal=5)
        np.testing.assert_almost_equal(flipped[1, 0], 1 - landmarks[1, 0], decimal=5)

    def test_flip_swaps_symmetric_pairs(self, sample_data):
        """Verifica que los pares simétricos se intercambian."""
        image, landmarks = sample_data
        transform = TrainTransform(flip_prob=1.0, rotation_degrees=0)

        _, flipped = transform.horizontal_flip(image, landmarks.copy())

        for left, right in SYMMETRIC_PAIRS:
            # Después del flip, la posición 'left' debe tener el valor reflejado de 'right'
            expected_left = landmarks[right].copy()
            expected_left[0] = 1 - expected_left[0]

            expected_right = landmarks[left].copy()
            expected_right[0] = 1 - expected_right[0]

            np.testing.assert_almost_equal(
                flipped[left], expected_left, decimal=5,
                err_msg=f"Par {left}-{right}: left incorrecto"
            )
            np.testing.assert_almost_equal(
                flipped[right], expected_right, decimal=5,
                err_msg=f"Par {left}-{right}: right incorrecto"
            )

    def test_flip_preserves_y_coordinates(self, sample_data):
        """Verifica que las coordenadas Y no cambian (excepto por el swap)."""
        image, landmarks = sample_data
        transform = TrainTransform(flip_prob=1.0, rotation_degrees=0)

        _, flipped = transform.horizontal_flip(image, landmarks.copy())

        # Centrales no cambian su Y
        central_indices = [0, 1, 8, 9, 10]
        for idx in central_indices:
            np.testing.assert_almost_equal(
                flipped[idx, 1], landmarks[idx, 1], decimal=5,
                err_msg=f"Central {idx}: Y debería mantenerse"
            )

    def test_double_flip_returns_original(self, sample_data):
        """Verifica que dos flips devuelven el original."""
        image, landmarks = sample_data
        transform = TrainTransform(flip_prob=1.0, rotation_degrees=0)

        _, flipped_once = transform.horizontal_flip(image, landmarks.copy())
        _, flipped_twice = transform.horizontal_flip(image, flipped_once.copy())

        np.testing.assert_almost_equal(
            flipped_twice, landmarks, decimal=5,
            err_msg="Doble flip debería devolver original"
        )


class TestRotation:
    """Tests para rotación."""

    def test_zero_rotation_preserves_landmarks(self):
        """Verifica que rotación de 0° no cambia landmarks."""
        transform = TrainTransform(flip_prob=0, rotation_degrees=0)
        image = Image.new('RGB', (224, 224))
        landmarks = np.random.rand(15, 2).astype(np.float32) * 0.8 + 0.1

        _, rotated = transform.rotate(image, landmarks.copy(), 0)

        np.testing.assert_almost_equal(rotated, landmarks, decimal=5)

    def test_90_degree_rotation(self):
        """Verifica rotación de 90°."""
        transform = TrainTransform()
        image = Image.new('RGB', (224, 224))
        landmarks = np.array([[0.5, 0.25]], dtype=np.float32)  # Punto arriba del centro

        _, rotated = transform.rotate(image, landmarks.copy(), 90)

        # Después de rotar 90° antihorario alrededor del centro (0.5, 0.5):
        # El punto debería moverse a la izquierda del centro
        assert rotated[0, 0] < 0.5  # Ahora está a la izquierda
        np.testing.assert_almost_equal(rotated[0, 1], 0.5, decimal=2)

    def test_rotation_keeps_landmarks_in_range(self):
        """Verifica que los landmarks se clampean a [0, 1] después de rotar."""
        transform = TrainTransform()
        image = Image.new('RGB', (224, 224))
        # Punto en esquina que al rotar puede salirse
        landmarks = np.array([[0.9, 0.9]], dtype=np.float32)

        _, rotated = transform.rotate(image, landmarks.copy(), 45)

        assert rotated.min() >= 0.0
        assert rotated.max() <= 1.0


class TestValTransform:
    """Tests para transformación de validación."""

    def test_val_transform_no_augmentation(self):
        """Verifica que val_transform no aplica augmentation."""
        transform = ValTransform(output_size=224)
        image = Image.new('RGB', (299, 299))
        landmarks = np.array([[100, 100]] * 15, dtype=np.float32)

        # Ejecutar múltiples veces - debe dar el mismo resultado
        results = []
        for _ in range(5):
            _, lm = transform(image, landmarks.copy(), (299, 299))
            results.append(lm.numpy())

        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])

    def test_val_transform_output_shape(self):
        """Verifica shapes de salida."""
        transform = ValTransform(output_size=224)
        image = Image.new('RGB', (299, 299))
        landmarks = np.random.rand(15, 2).astype(np.float32) * 299

        img_tensor, lm_tensor = transform(image, landmarks, (299, 299))

        assert img_tensor.shape == (3, 224, 224)
        assert lm_tensor.shape == (30,)


class TestTrainTransform:
    """Tests para transformación de entrenamiento."""

    def test_train_transform_output_shape(self):
        """Verifica shapes de salida."""
        transform = TrainTransform(output_size=224, flip_prob=0.5, rotation_degrees=10)
        image = Image.new('RGB', (299, 299))
        landmarks = np.random.rand(15, 2).astype(np.float32) * 299

        img_tensor, lm_tensor = transform(image, landmarks, (299, 299))

        assert img_tensor.shape == (3, 224, 224)
        assert lm_tensor.shape == (30,)

    def test_train_transform_landmarks_in_range(self):
        """Verifica que landmarks están en [0, 1] después de augmentation."""
        transform = TrainTransform(output_size=224, flip_prob=0.5, rotation_degrees=15)
        image = Image.new('RGB', (299, 299))
        landmarks = np.random.rand(15, 2).astype(np.float32) * 299

        for _ in range(10):  # Probar múltiples veces con augmentation aleatorio
            _, lm_tensor = transform(image, landmarks.copy(), (299, 299))
            assert lm_tensor.min() >= 0.0
            assert lm_tensor.max() <= 1.0

    def test_flip_prob_zero_no_flip(self):
        """Verifica que flip_prob=0 nunca hace flip."""
        transform = TrainTransform(flip_prob=0.0, rotation_degrees=0)
        image = Image.new('RGB', (224, 224))
        # Landmarks en píxeles (serán normalizados por el transform)
        landmarks = np.array([[22.4, 112]] * 15, dtype=np.float32)  # 0.1*224, 0.5*224

        for _ in range(10):
            _, lm = transform(image, landmarks.copy(), (224, 224))
            # Si no hay flip, x normalizado debería ser ~0.1
            np.testing.assert_almost_equal(lm[0], 0.1, decimal=2)

    def test_flip_prob_one_always_flips(self):
        """Verifica que flip_prob=1 siempre hace flip."""
        transform = TrainTransform(flip_prob=1.0, rotation_degrees=0)
        image = Image.new('RGB', (224, 224))
        # Landmarks en píxeles (serán normalizados por el transform)
        landmarks = np.array([[22.4, 112]] * 15, dtype=np.float32)  # 0.1*224, 0.5*224

        for _ in range(10):
            _, lm = transform(image, landmarks.copy(), (224, 224))
            # Si siempre hay flip, x normalizado 0.1 debería ser 0.9
            np.testing.assert_almost_equal(lm[0], 0.9, decimal=2)


class TestFactoryFunctions:
    """Tests para funciones factory."""

    def test_get_train_transforms(self):
        """Verifica factory de train transforms."""
        transform = get_train_transforms(output_size=256, flip_prob=0.3, rotation_degrees=5)

        assert isinstance(transform, TrainTransform)
        assert transform.output_size == 256
        assert transform.flip_prob == 0.3
        assert transform.rotation_degrees == 5

    def test_get_val_transforms(self):
        """Verifica factory de val transforms."""
        transform = get_val_transforms(output_size=256)

        assert isinstance(transform, ValTransform)
        assert transform.output_size == 256


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
