"""
Tests para src_v2/constants.py

Valida que todas las constantes del proyecto esten correctamente definidas
y sean consistentes entre si.
"""

import pytest

from src_v2.constants import (
    # Landmarks
    NUM_LANDMARKS,
    NUM_COORDINATES,
    LANDMARK_NAMES,
    SYMMETRIC_PAIRS,
    CENTRAL_LANDMARKS,
    AXIS_LANDMARKS,
    # Dimensiones
    DEFAULT_IMAGE_SIZE,
    ORIGINAL_IMAGE_SIZE,
    # Normalizacion
    IMAGENET_MEAN,
    IMAGENET_STD,
    # Categorias
    CATEGORIES,
    NUM_CLASSES,
    DEFAULT_CATEGORY_WEIGHTS,
    # Modelo
    BACKBONE_FEATURE_DIM,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_DROPOUT_RATE,
    # Entrenamiento
    DEFAULT_BATCH_SIZE,
    DEFAULT_PHASE1_LR,
    DEFAULT_PHASE2_BACKBONE_LR,
    DEFAULT_PHASE2_HEAD_LR,
    DEFAULT_PHASE1_EPOCHS,
    DEFAULT_PHASE2_EPOCHS,
    # Loss
    DEFAULT_WING_OMEGA,
    DEFAULT_WING_EPSILON,
    DEFAULT_SYMMETRY_MARGIN,
    # Augmentation
    DEFAULT_FLIP_PROB,
    DEFAULT_ROTATION_DEGREES,
    DEFAULT_CLAHE_CLIP_LIMIT,
    DEFAULT_CLAHE_TILE_SIZE,
)


class TestLandmarkConstants:
    """Tests para constantes de landmarks."""

    def test_num_landmarks_is_15(self):
        """Debe haber exactamente 15 landmarks."""
        assert NUM_LANDMARKS == 15

    def test_num_coordinates_is_30(self):
        """30 coordenadas = 15 landmarks * 2 (x, y)."""
        assert NUM_COORDINATES == 30
        assert NUM_COORDINATES == NUM_LANDMARKS * 2

    def test_landmark_names_count(self):
        """Debe haber un nombre por cada landmark."""
        assert len(LANDMARK_NAMES) == NUM_LANDMARKS

    def test_landmark_names_format(self):
        """Nombres deben ser L1 a L15."""
        expected = [f'L{i}' for i in range(1, 16)]
        assert LANDMARK_NAMES == expected

    def test_symmetric_pairs_count(self):
        """Debe haber 5 pares simetricos."""
        assert len(SYMMETRIC_PAIRS) == 5

    def test_symmetric_pairs_are_tuples(self):
        """Cada par debe ser una tupla de dos indices."""
        for pair in SYMMETRIC_PAIRS:
            assert isinstance(pair, tuple)
            assert len(pair) == 2

    def test_symmetric_pairs_valid_indices(self):
        """Indices de pares deben estar en rango [0, 14]."""
        for left, right in SYMMETRIC_PAIRS:
            assert 0 <= left < NUM_LANDMARKS
            assert 0 <= right < NUM_LANDMARKS
            assert left != right  # No pueden ser el mismo

    def test_symmetric_pairs_left_right_convention(self):
        """En cada par, el indice izquierdo debe ser menor que el derecho."""
        for left, right in SYMMETRIC_PAIRS:
            assert left < right, f"Par ({left}, {right}) no sigue convencion izq < der"

    def test_central_landmarks_count(self):
        """Debe haber 3 landmarks centrales (L9, L10, L11)."""
        assert len(CENTRAL_LANDMARKS) == 3

    def test_central_landmarks_valid_indices(self):
        """Indices centrales deben estar en rango [0, 14]."""
        for idx in CENTRAL_LANDMARKS:
            assert 0 <= idx < NUM_LANDMARKS

    def test_central_landmarks_are_l9_l10_l11(self):
        """Landmarks centrales deben ser indices de L9, L10, L11."""
        # L9 = indice 8, L10 = indice 9, L11 = indice 10
        assert set(CENTRAL_LANDMARKS) == {8, 9, 10}

    def test_axis_landmarks_count(self):
        """Debe haber 2 landmarks de eje (L1, L2)."""
        assert len(AXIS_LANDMARKS) == 2

    def test_axis_landmarks_are_l1_l2(self):
        """Landmarks de eje deben ser L1 (indice 0) y L2 (indice 1)."""
        assert set(AXIS_LANDMARKS) == {0, 1}

    def test_no_overlap_central_symmetric(self):
        """Landmarks centrales no deben estar en pares simetricos."""
        symmetric_indices = set()
        for left, right in SYMMETRIC_PAIRS:
            symmetric_indices.add(left)
            symmetric_indices.add(right)

        central_set = set(CENTRAL_LANDMARKS)
        assert symmetric_indices.isdisjoint(central_set)


class TestDimensionConstants:
    """Tests para constantes de dimensiones."""

    def test_default_image_size_is_224(self):
        """Tamano de imagen por defecto debe ser 224."""
        assert DEFAULT_IMAGE_SIZE == 224

    def test_original_image_size_is_299(self):
        """Tamano original del dataset debe ser 299."""
        assert ORIGINAL_IMAGE_SIZE == 299

    def test_image_sizes_are_positive(self):
        """Tamanos de imagen deben ser positivos."""
        assert DEFAULT_IMAGE_SIZE > 0
        assert ORIGINAL_IMAGE_SIZE > 0


class TestNormalizationConstants:
    """Tests para constantes de normalizacion."""

    def test_imagenet_mean_is_tuple(self):
        """ImageNet mean debe ser tupla de 3 elementos."""
        assert isinstance(IMAGENET_MEAN, tuple)
        assert len(IMAGENET_MEAN) == 3

    def test_imagenet_std_is_tuple(self):
        """ImageNet std debe ser tupla de 3 elementos."""
        assert isinstance(IMAGENET_STD, tuple)
        assert len(IMAGENET_STD) == 3

    def test_imagenet_mean_values(self):
        """Valores de ImageNet mean deben ser los estandar."""
        assert IMAGENET_MEAN == (0.485, 0.456, 0.406)

    def test_imagenet_std_values(self):
        """Valores de ImageNet std deben ser los estandar."""
        assert IMAGENET_STD == (0.229, 0.224, 0.225)

    def test_imagenet_values_in_range(self):
        """Valores deben estar en rango [0, 1]."""
        for v in IMAGENET_MEAN:
            assert 0 <= v <= 1
        for v in IMAGENET_STD:
            assert 0 < v <= 1


class TestCategoryConstants:
    """Tests para constantes de categorias."""

    def test_categories_count(self):
        """Debe haber 3 categorias."""
        assert len(CATEGORIES) == 3

    def test_num_classes_matches_categories(self):
        """NUM_CLASSES debe coincidir con len(CATEGORIES)."""
        assert NUM_CLASSES == len(CATEGORIES)

    def test_categories_content(self):
        """Categorias deben ser COVID, Normal, Viral_Pneumonia."""
        assert set(CATEGORIES) == {'COVID', 'Normal', 'Viral_Pneumonia'}

    def test_category_weights_keys(self):
        """Pesos deben existir para todas las categorias."""
        assert set(DEFAULT_CATEGORY_WEIGHTS.keys()) == set(CATEGORIES)

    def test_category_weights_positive(self):
        """Todos los pesos deben ser positivos."""
        for cat, weight in DEFAULT_CATEGORY_WEIGHTS.items():
            assert weight > 0, f"Peso de {cat} debe ser positivo"


class TestModelConstants:
    """Tests para constantes del modelo."""

    def test_backbone_feature_dim(self):
        """ResNet-18 tiene 512 features en la ultima capa."""
        assert BACKBONE_FEATURE_DIM == 512

    def test_hidden_dim_positive(self):
        """Dimension oculta debe ser positiva."""
        assert DEFAULT_HIDDEN_DIM > 0

    def test_dropout_rate_valid(self):
        """Dropout rate debe estar en [0, 1)."""
        assert 0 <= DEFAULT_DROPOUT_RATE < 1


class TestTrainingConstants:
    """Tests para constantes de entrenamiento."""

    def test_batch_size_positive(self):
        """Batch size debe ser positivo."""
        assert DEFAULT_BATCH_SIZE > 0

    def test_learning_rates_positive(self):
        """Learning rates deben ser positivos."""
        assert DEFAULT_PHASE1_LR > 0
        assert DEFAULT_PHASE2_BACKBONE_LR > 0
        assert DEFAULT_PHASE2_HEAD_LR > 0

    def test_phase2_backbone_lr_smaller(self):
        """LR del backbone en phase2 debe ser menor que cabeza."""
        assert DEFAULT_PHASE2_BACKBONE_LR < DEFAULT_PHASE2_HEAD_LR

    def test_epochs_positive(self):
        """Epocas deben ser positivas."""
        assert DEFAULT_PHASE1_EPOCHS > 0
        assert DEFAULT_PHASE2_EPOCHS > 0


class TestLossConstants:
    """Tests para constantes de loss."""

    def test_wing_omega_positive(self):
        """Wing omega debe ser positivo."""
        assert DEFAULT_WING_OMEGA > 0

    def test_wing_epsilon_positive(self):
        """Wing epsilon debe ser positivo."""
        assert DEFAULT_WING_EPSILON > 0

    def test_symmetry_margin_non_negative(self):
        """Margen de simetria debe ser no negativo."""
        assert DEFAULT_SYMMETRY_MARGIN >= 0


class TestAugmentationConstants:
    """Tests para constantes de augmentation."""

    def test_flip_prob_valid(self):
        """Probabilidad de flip debe estar en [0, 1]."""
        assert 0 <= DEFAULT_FLIP_PROB <= 1

    def test_rotation_degrees_reasonable(self):
        """Rotacion debe ser razonable (<= 45 grados)."""
        assert 0 <= DEFAULT_ROTATION_DEGREES <= 45

    def test_clahe_clip_limit_positive(self):
        """CLAHE clip limit debe ser positivo."""
        assert DEFAULT_CLAHE_CLIP_LIMIT > 0

    def test_clahe_tile_size_positive(self):
        """CLAHE tile size debe ser positivo."""
        assert DEFAULT_CLAHE_TILE_SIZE > 0


class TestConstantsIntegrity:
    """Tests de integridad entre constantes."""

    def test_all_landmarks_accounted_for(self):
        """Cada landmark debe estar en exactamente una categoria."""
        central = set(CENTRAL_LANDMARKS)
        axis = set(AXIS_LANDMARKS)
        symmetric = set()
        for left, right in SYMMETRIC_PAIRS:
            symmetric.add(left)
            symmetric.add(right)

        all_categorized = central | axis | symmetric
        all_indices = set(range(NUM_LANDMARKS))

        # Verificar que todos los landmarks estan categorizados
        assert all_categorized == all_indices, \
            f"Landmarks no categorizados: {all_indices - all_categorized}"

    def test_imports_work(self):
        """Todas las constantes deben poder importarse."""
        # Este test pasa si el modulo se importa sin errores
        from src_v2.constants import (
            NUM_LANDMARKS, SYMMETRIC_PAIRS, CATEGORIES
        )
        assert True
