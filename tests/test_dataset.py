"""
Tests unitarios para el modulo src_v2/data/dataset.py

Sesion 2 Auditoria: V01 - Crear tests basicos para funciones publicas.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from src_v2.constants import DEFAULT_CATEGORY_WEIGHTS


# =============================================================================
# FIXTURES ESPECIFICAS PARA DATASET
# =============================================================================

@pytest.fixture
def sample_dataframe():
    """DataFrame de ejemplo con estructura de landmarks."""
    # Crear suficientes muestras para split estratificado doble
    # Con 10 muestras por categoria y splits 15%/10%, tenemos suficiente margen
    images_per_category = 10
    categories = ['COVID', 'Normal', 'Viral_Pneumonia']

    image_names = []
    category_list = []
    for cat in categories:
        for i in range(1, images_per_category + 1):
            if cat == 'Viral_Pneumonia':
                image_names.append(f'Viral Pneumonia-{i:03d}')
            else:
                image_names.append(f'{cat}-{i:03d}')
            category_list.append(cat)

    data = {
        'image_name': image_names,
        'category': category_list,
    }
    n_samples = len(data['image_name'])
    # Agregar columnas de landmarks
    for i in range(1, 16):
        data[f'L{i}_x'] = [100.0 + i * 2] * n_samples
        data[f'L{i}_y'] = [100.0 + i * 3] * n_samples
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_path(tmp_path, sample_dataframe):
    """
    Crea CSV temporal en el formato esperado por load_coordinates_csv.

    Formato: idx, L1_x, L1_y, ..., L15_x, L15_y, image_name
    """
    csv_path = tmp_path / "coords.csv"

    # Crear datos en formato correcto (sin header, columna idx al inicio)
    rows = []
    for idx, row in sample_dataframe.iterrows():
        row_data = [idx]
        for i in range(1, 16):
            row_data.extend([row[f'L{i}_x'], row[f'L{i}_y']])
        row_data.append(row['image_name'])
        rows.append(row_data)

    # Escribir CSV sin header
    with open(csv_path, 'w') as f:
        for row in rows:
            f.write(','.join(str(x) for x in row) + '\n')

    return csv_path


@pytest.fixture
def sample_data_root(tmp_path, sample_dataframe):
    """
    Crea estructura de directorios con imagenes de prueba.
    """
    from PIL import Image

    data_root = tmp_path / "data"
    dataset_dir = data_root / "dataset"

    for _, row in sample_dataframe.iterrows():
        category = row['category']
        image_name = row['image_name']

        # Crear directorio de categoria
        cat_dir = dataset_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        # Crear imagen de prueba
        img = Image.new('RGB', (299, 299), color='gray')
        img.save(cat_dir / f"{image_name}.png")

    return data_root


# =============================================================================
# TESTS PARA compute_sample_weights
# =============================================================================

class TestComputeSampleWeights:
    """Tests para compute_sample_weights()."""

    def test_returns_tensor(self, sample_dataframe):
        """Verifica que retorna un tensor de PyTorch."""
        from src_v2.data.dataset import compute_sample_weights

        weights = compute_sample_weights(sample_dataframe)

        assert isinstance(weights, torch.Tensor)
        assert weights.dtype == torch.float32

    def test_correct_length(self, sample_dataframe):
        """Verifica que el tensor tiene el mismo tamano que el DataFrame."""
        from src_v2.data.dataset import compute_sample_weights

        weights = compute_sample_weights(sample_dataframe)

        assert len(weights) == len(sample_dataframe)

    def test_uses_default_weights_when_none(self, sample_dataframe):
        """Verifica que usa DEFAULT_CATEGORY_WEIGHTS cuando no se pasa category_weights."""
        from src_v2.data.dataset import compute_sample_weights

        weights = compute_sample_weights(sample_dataframe, category_weights=None)

        # Verificar que COVID tiene peso mayor (segun DEFAULT_CATEGORY_WEIGHTS)
        covid_weight = DEFAULT_CATEGORY_WEIGHTS.get('COVID', 1.0)
        normal_weight = DEFAULT_CATEGORY_WEIGHTS.get('Normal', 1.0)

        # Indices: COVID en 0-9, Normal en 10-19
        assert weights[0].item() == pytest.approx(covid_weight)
        assert weights[10].item() == pytest.approx(normal_weight)

    def test_custom_weights(self, sample_dataframe):
        """Verifica que usa pesos personalizados correctamente."""
        from src_v2.data.dataset import compute_sample_weights

        custom_weights = {'COVID': 3.0, 'Normal': 1.0, 'Viral_Pneumonia': 2.0}
        weights = compute_sample_weights(sample_dataframe, category_weights=custom_weights)

        # Verificar pesos asignados (COVID en 0-9, Normal en 10-19, Viral en 20-29)
        assert weights[0].item() == 3.0  # COVID
        assert weights[10].item() == 1.0  # Normal
        assert weights[20].item() == 2.0  # Viral_Pneumonia

    def test_unknown_category_defaults_to_one(self, sample_dataframe):
        """Verifica que categorias desconocidas usan peso 1.0."""
        from src_v2.data.dataset import compute_sample_weights

        # Agregar categoria desconocida
        df = sample_dataframe.copy()
        df.loc[0, 'category'] = 'Unknown_Category'

        custom_weights = {'COVID': 2.0}  # No incluye Unknown_Category
        weights = compute_sample_weights(df, category_weights=custom_weights)

        assert weights[0].item() == 1.0  # Default para desconocido


# =============================================================================
# TESTS PARA LandmarkDataset
# =============================================================================

class TestLandmarkDataset:
    """Tests para LandmarkDataset."""

    def test_len_returns_dataframe_length(self, sample_dataframe, sample_data_root):
        """Verifica que __len__ retorna la longitud correcta."""
        from src_v2.data.dataset import LandmarkDataset

        dataset = LandmarkDataset(
            df=sample_dataframe,
            data_root=str(sample_data_root)
        )

        assert len(dataset) == len(sample_dataframe)

    def test_getitem_returns_tuple(self, sample_dataframe, sample_data_root):
        """Verifica que __getitem__ retorna tupla (tensor, tensor, dict)."""
        from src_v2.data.dataset import LandmarkDataset

        dataset = LandmarkDataset(
            df=sample_dataframe,
            data_root=str(sample_data_root)
        )

        image, landmarks, meta = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert isinstance(landmarks, torch.Tensor)
        assert isinstance(meta, dict)

    def test_image_tensor_shape(self, sample_dataframe, sample_data_root):
        """Verifica que el tensor de imagen tiene shape correcto."""
        from src_v2.data.dataset import LandmarkDataset

        dataset = LandmarkDataset(
            df=sample_dataframe,
            data_root=str(sample_data_root)
        )

        image, _, _ = dataset[0]

        assert image.shape == (3, 224, 224)  # (C, H, W)

    def test_landmarks_tensor_shape(self, sample_dataframe, sample_data_root):
        """Verifica que el tensor de landmarks tiene shape correcto."""
        from src_v2.data.dataset import LandmarkDataset

        dataset = LandmarkDataset(
            df=sample_dataframe,
            data_root=str(sample_data_root)
        )

        _, landmarks, _ = dataset[0]

        assert landmarks.shape == (30,)  # 15 landmarks * 2 coords

    def test_meta_contains_required_keys(self, sample_dataframe, sample_data_root):
        """Verifica que meta contiene las claves requeridas."""
        from src_v2.data.dataset import LandmarkDataset

        dataset = LandmarkDataset(
            df=sample_dataframe,
            data_root=str(sample_data_root)
        )

        _, _, meta = dataset[0]

        assert 'image_name' in meta
        assert 'category' in meta
        assert 'idx' in meta
        assert 'original_size' in meta

    def test_meta_original_size_matches_image(self, sample_dataframe, sample_data_root):
        """original_size en meta debe reflejar tamaño real de la imagen."""
        from src_v2.data.dataset import LandmarkDataset

        dataset = LandmarkDataset(
            df=sample_dataframe,
            data_root=str(sample_data_root)
        )

        _, _, meta = dataset[0]

        assert meta['original_size'] == (299, 299)


# =============================================================================
# TESTS PARA get_dataframe_splits
# =============================================================================

class TestGetDataframeSplits:
    """Tests para get_dataframe_splits()."""

    def test_returns_three_dataframes(self, sample_csv_path):
        """Verifica que retorna 3 DataFrames."""
        from src_v2.data.dataset import get_dataframe_splits

        train_df, val_df, test_df = get_dataframe_splits(str(sample_csv_path))

        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)

    def test_no_overlap_between_splits(self, sample_csv_path):
        """Verifica que no hay muestras repetidas entre splits."""
        from src_v2.data.dataset import get_dataframe_splits

        train_df, val_df, test_df = get_dataframe_splits(str(sample_csv_path))

        train_names = set(train_df['image_name'])
        val_names = set(val_df['image_name'])
        test_names = set(test_df['image_name'])

        assert train_names.isdisjoint(val_names), "Train y Val tienen muestras en comun"
        assert train_names.isdisjoint(test_names), "Train y Test tienen muestras en comun"
        assert val_names.isdisjoint(test_names), "Val y Test tienen muestras en comun"

    def test_deterministic_with_same_seed(self, sample_csv_path):
        """Verifica que el split es determinista con la misma semilla."""
        from src_v2.data.dataset import get_dataframe_splits

        train1, val1, test1 = get_dataframe_splits(str(sample_csv_path), random_state=42)
        train2, val2, test2 = get_dataframe_splits(str(sample_csv_path), random_state=42)

        pd.testing.assert_frame_equal(train1.reset_index(drop=True), train2.reset_index(drop=True))
        pd.testing.assert_frame_equal(val1.reset_index(drop=True), val2.reset_index(drop=True))
        pd.testing.assert_frame_equal(test1.reset_index(drop=True), test2.reset_index(drop=True))

    def test_total_samples_preserved(self, sample_csv_path):
        """Verifica que el total de muestras se preserva."""
        from src_v2.data.dataset import get_dataframe_splits
        from src_v2.data.utils import load_coordinates_csv

        original_df = load_coordinates_csv(str(sample_csv_path))
        train_df, val_df, test_df = get_dataframe_splits(str(sample_csv_path))

        total_split = len(train_df) + len(val_df) + len(test_df)
        assert total_split == len(original_df)
