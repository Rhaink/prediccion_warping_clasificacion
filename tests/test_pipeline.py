"""
Tests de integracion para el pipeline completo.

Estos tests verifican que los componentes del sistema funcionan
correctamente cuando se integran entre si.
"""

import numpy as np
import pytest
import torch

from src_v2.constants import (
    NUM_LANDMARKS,
    NUM_COORDINATES,
    DEFAULT_IMAGE_SIZE,
    SYMMETRIC_PAIRS,
)


class TestModelCreation:
    """Tests para creacion de modelos."""

    def test_create_model_pretrained(self):
        """Debe poder crear modelo con pesos preentrenados."""
        from src_v2.models import create_model

        model = create_model(pretrained=True)
        assert model is not None

    def test_create_model_random(self):
        """Debe poder crear modelo con pesos aleatorios."""
        from src_v2.models import create_model

        model = create_model(pretrained=False)
        assert model is not None

    def test_model_output_shape(self, untrained_model, sample_image_tensor):
        """Modelo debe producir salida (B, 30)."""
        output = untrained_model(sample_image_tensor)
        assert output.shape == (1, NUM_COORDINATES)

    def test_model_output_range(self, untrained_model, sample_image_tensor):
        """Salida del modelo debe estar en [0, 1] (sigmoid)."""
        output = untrained_model(sample_image_tensor)
        # Puede haber valores ligeramente fuera por precision numerica
        assert output.min() >= -0.01
        assert output.max() <= 1.01

    def test_model_batch_processing(self, untrained_model, batch_images_tensor):
        """Modelo debe procesar batches correctamente."""
        batch_size = batch_images_tensor.shape[0]
        output = untrained_model(batch_images_tensor)
        assert output.shape == (batch_size, NUM_COORDINATES)


class TestLossComputation:
    """Tests para calculo de loss."""

    def test_wing_loss_forward(self, batch_landmarks_tensor):
        """WingLoss debe calcular loss sin errores."""
        from src_v2.models import WingLoss

        loss_fn = WingLoss()
        pred = batch_landmarks_tensor
        target = batch_landmarks_tensor + torch.randn_like(batch_landmarks_tensor) * 0.01

        loss = loss_fn(pred, target)
        assert loss.shape == ()  # Escalar
        assert loss >= 0

    def test_combined_loss_forward(self, batch_landmarks_tensor):
        """CombinedLandmarkLoss debe retornar dict con componentes."""
        from src_v2.models import CombinedLandmarkLoss

        loss_fn = CombinedLandmarkLoss()
        pred = batch_landmarks_tensor
        target = batch_landmarks_tensor + torch.randn_like(batch_landmarks_tensor) * 0.01

        loss_dict = loss_fn(pred, target)
        assert isinstance(loss_dict, dict)
        assert 'total' in loss_dict
        assert 'wing' in loss_dict
        assert loss_dict['total'] >= 0


class TestTransformsPipeline:
    """Tests para pipeline de transforms."""

    def test_train_transform_output(self, sample_image, sample_landmarks):
        """Train transform debe producir tensor de imagen y landmarks."""
        from PIL import Image
        from src_v2.data import get_train_transforms

        # Convertir a PIL Image
        img_pil = Image.fromarray(sample_image)
        original_size = img_pil.size  # (width, height)

        transform = get_train_transforms()
        img_tensor, landmarks_tensor = transform(img_pil, sample_landmarks, original_size)

        assert img_tensor.shape == (3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
        assert landmarks_tensor.shape == (NUM_COORDINATES,)

    def test_val_transform_output(self, sample_image, sample_landmarks):
        """Val transform debe producir tensor de imagen y landmarks."""
        from PIL import Image
        from src_v2.data import get_val_transforms

        # Convertir a PIL Image
        img_pil = Image.fromarray(sample_image)
        original_size = img_pil.size

        transform = get_val_transforms()
        img_tensor, landmarks_tensor = transform(img_pil, sample_landmarks, original_size)

        assert img_tensor.shape == (3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
        assert landmarks_tensor.shape == (NUM_COORDINATES,)

    def test_transform_landmarks_in_range(self, sample_image, sample_landmarks):
        """Landmarks transformados deben estar en [0, 1]."""
        from PIL import Image
        from src_v2.data import get_train_transforms

        # Convertir a PIL Image
        img_pil = Image.fromarray(sample_image)
        original_size = img_pil.size

        transform = get_train_transforms()

        # Ejecutar varias veces por la augmentacion estocastica
        for _ in range(5):
            _, landmarks_tensor = transform(img_pil, sample_landmarks, original_size)
            assert landmarks_tensor.min() >= 0
            assert landmarks_tensor.max() <= 1


class TestModelForwardBackward:
    """Tests para forward y backward pass."""

    def test_forward_pass(self, untrained_model, sample_image_tensor):
        """Forward pass debe funcionar sin errores."""
        with torch.no_grad():
            output = untrained_model(sample_image_tensor)
        assert output is not None

    def test_backward_pass(self, untrained_model, sample_image_tensor, sample_landmarks_tensor):
        """Backward pass debe calcular gradientes."""
        from src_v2.models import WingLoss

        untrained_model.train()
        loss_fn = WingLoss()

        output = untrained_model(sample_image_tensor)
        loss = loss_fn(output, sample_landmarks_tensor)
        loss.backward()

        # Verificar que hay gradientes
        has_grads = False
        for param in untrained_model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grads = True
                break
        assert has_grads


class TestModelFreezing:
    """Tests para congelacion de capas del modelo."""

    def test_freeze_backbone(self, untrained_model):
        """Debe poder congelar backbone."""
        untrained_model.freeze_backbone()

        # Verificar que backbone esta congelado
        for param in untrained_model.backbone.parameters():
            assert not param.requires_grad

        # Head debe seguir entrenable
        for param in untrained_model.head.parameters():
            assert param.requires_grad

    def test_unfreeze_backbone(self, untrained_model):
        """Debe poder descongelar backbone."""
        untrained_model.freeze_backbone()
        untrained_model.unfreeze_backbone()

        # Verificar que backbone esta descongelado
        for param in untrained_model.backbone.parameters():
            assert param.requires_grad

    def test_get_trainable_params(self, untrained_model):
        """Debe retornar grupos de parametros entrenables."""
        param_groups = untrained_model.get_trainable_params()

        assert isinstance(param_groups, list)
        assert len(param_groups) >= 1
        for group in param_groups:
            assert 'params' in group


class TestEvaluationMetrics:
    """Tests para metricas de evaluacion."""

    def test_compute_pixel_error(self, batch_landmarks_tensor):
        """Debe calcular error en pixeles."""
        from src_v2.evaluation.metrics import compute_pixel_error

        pred = batch_landmarks_tensor
        target = batch_landmarks_tensor + 0.01  # Pequeno error

        errors = compute_pixel_error(pred, target)
        assert errors.shape == (batch_landmarks_tensor.shape[0], NUM_LANDMARKS)
        assert (errors >= 0).all()

    def test_compute_error_per_landmark(self, batch_landmarks_tensor):
        """Debe calcular error por landmark."""
        from src_v2.evaluation.metrics import compute_error_per_landmark

        pred = batch_landmarks_tensor
        target = batch_landmarks_tensor + 0.01

        errors = compute_error_per_landmark(pred, target)
        assert isinstance(errors, dict)
        assert len(errors) == NUM_LANDMARKS
        for name, error in errors.items():
            assert error >= 0


class TestDataIntegration:
    """Tests de integracion de datos."""

    def test_landmark_coordinates_structure(self, sample_landmarks):
        """Landmarks deben tener estructura correcta."""
        assert sample_landmarks.shape == (NUM_LANDMARKS, 2)
        assert sample_landmarks.dtype == np.float32

    def test_flatten_unflatten_landmarks(self, sample_landmarks):
        """Aplanar y desaplanar landmarks debe ser reversible."""
        flat = sample_landmarks.flatten()
        assert flat.shape == (NUM_COORDINATES,)

        unflat = flat.reshape(NUM_LANDMARKS, 2)
        np.testing.assert_array_equal(unflat, sample_landmarks)

    def test_symmetric_pairs_geometry(self, sample_landmarks):
        """Pares simetricos deben tener x reflejado."""
        # En una imagen simetrica, pares deben estar equidistantes del centro
        center_x = 0.5

        for left_idx, right_idx in SYMMETRIC_PAIRS:
            left_x = sample_landmarks[left_idx, 0]
            right_x = sample_landmarks[right_idx, 0]

            # Distancia al centro debe ser similar
            left_dist = abs(left_x - center_x)
            right_dist = abs(right_x - center_x)

            # Tolerancia amplia porque son landmarks sinteticos
            assert abs(left_dist - right_dist) < 0.3


class TestEndToEndInference:
    """Tests end-to-end de inferencia."""

    def test_full_inference_pipeline(self, pretrained_model, sample_image_tensor, model_device):
        """Pipeline completo de inferencia debe funcionar."""
        pretrained_model.eval()

        with torch.no_grad():
            # Forward
            output = pretrained_model(sample_image_tensor.to(model_device))

            # Convertir a coordenadas
            landmarks = output.squeeze().cpu().numpy()
            landmarks = landmarks.reshape(NUM_LANDMARKS, 2)

            # Escalar a pixeles
            landmarks_px = landmarks * DEFAULT_IMAGE_SIZE

        # Verificaciones
        assert landmarks_px.shape == (NUM_LANDMARKS, 2)
        # Landmarks deben estar dentro de la imagen (con margen por precision)
        assert landmarks_px.min() >= -10
        assert landmarks_px.max() <= DEFAULT_IMAGE_SIZE + 10

    def test_batch_inference(self, pretrained_model, batch_images_tensor, model_device):
        """Inferencia en batch debe funcionar."""
        pretrained_model.eval()

        with torch.no_grad():
            output = pretrained_model(batch_images_tensor.to(model_device))

        assert output.shape[0] == batch_images_tensor.shape[0]
        assert output.shape[1] == NUM_COORDINATES


class TestDataSplitReproducibility:
    """Tests para verificar que el split de datos es reproducible."""

    @pytest.fixture
    def csv_path(self):
        """Ruta al CSV de coordenadas."""
        from pathlib import Path
        return str(Path(__file__).parent.parent / "data/coordenadas/coordenadas_maestro.csv")

    @pytest.fixture
    def data_root(self):
        """Directorio raiz de datos."""
        from pathlib import Path
        return str(Path(__file__).parent.parent / "data/")

    def test_split_is_deterministic(self, csv_path, data_root):
        """El split con random_state=42 debe ser siempre igual."""
        from sklearn.model_selection import train_test_split
        from src_v2.data.dataset import load_coordinates_csv

        df = load_coordinates_csv(csv_path)

        # Primera ejecucion
        train1, temp1 = train_test_split(
            df, test_size=0.25, random_state=42, stratify=df['category']
        )
        val1, test1 = train_test_split(
            temp1, test_size=0.4, random_state=42, stratify=temp1['category']
        )

        # Segunda ejecucion
        train2, temp2 = train_test_split(
            df, test_size=0.25, random_state=42, stratify=df['category']
        )
        val2, test2 = train_test_split(
            temp2, test_size=0.4, random_state=42, stratify=temp2['category']
        )

        # Deben ser identicos
        assert list(train1.index) == list(train2.index), "Train split debe ser determinístico"
        assert list(val1.index) == list(val2.index), "Val split debe ser determinístico"
        assert list(test1.index) == list(test2.index), "Test split debe ser determinístico"

    def test_split_sizes_match_expected(self, csv_path, data_root):
        """El split debe producir ~75% train, 15% val, 10% test."""
        from sklearn.model_selection import train_test_split
        from src_v2.data.dataset import load_coordinates_csv

        df = load_coordinates_csv(csv_path)
        total = len(df)

        train_df, temp_df = train_test_split(
            df, test_size=0.25, random_state=42, stratify=df['category']
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.4, random_state=42, stratify=temp_df['category']
        )

        # Verificar proporciones con tolerancia
        train_ratio = len(train_df) / total
        val_ratio = len(val_df) / total
        test_ratio = len(test_df) / total

        assert 0.73 <= train_ratio <= 0.77, f"Train ratio {train_ratio:.2f} fuera de rango esperado"
        assert 0.13 <= val_ratio <= 0.17, f"Val ratio {val_ratio:.2f} fuera de rango esperado"
        assert 0.08 <= test_ratio <= 0.12, f"Test ratio {test_ratio:.2f} fuera de rango esperado"

    def test_test_split_has_expected_samples(self, csv_path, data_root):
        """El test split debe tener ~96 muestras (10% de 957)."""
        from sklearn.model_selection import train_test_split
        from src_v2.data.dataset import load_coordinates_csv

        df = load_coordinates_csv(csv_path)

        train_df, temp_df = train_test_split(
            df, test_size=0.25, random_state=42, stratify=df['category']
        )
        _, test_df = train_test_split(
            temp_df, test_size=0.4, random_state=42, stratify=temp_df['category']
        )

        # 10% de 957 = 95.7, redondeado ~96
        assert 90 <= len(test_df) <= 100, f"Test size {len(test_df)} fuera de rango esperado"

    def test_stratification_preserves_categories(self, csv_path, data_root):
        """El split estratificado debe preservar la proporcion de categorias."""
        from sklearn.model_selection import train_test_split
        from src_v2.data.dataset import load_coordinates_csv

        df = load_coordinates_csv(csv_path)

        # Proporciones originales
        original_ratios = df['category'].value_counts(normalize=True)

        train_df, temp_df = train_test_split(
            df, test_size=0.25, random_state=42, stratify=df['category']
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.4, random_state=42, stratify=temp_df['category']
        )

        # Proporciones en cada split
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            split_ratios = split_df['category'].value_counts(normalize=True)
            for category in original_ratios.index:
                orig = original_ratios.get(category, 0)
                split = split_ratios.get(category, 0)
                # Tolerancia del 5% absoluto
                assert abs(orig - split) < 0.05, \
                    f"{split_name}: {category} ratio {split:.2f} difiere de original {orig:.2f}"
