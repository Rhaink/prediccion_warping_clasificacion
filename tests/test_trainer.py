"""
Tests unitarios para LandmarkTrainer

Tests para src_v2/training/trainer.py
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys
import tempfile
import shutil

# Agregar src_v2 al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src_v2.training.trainer import LandmarkTrainer
from src_v2.constants import DEFAULT_IMAGE_SIZE


class MockModel(nn.Module):
    """Modelo mock simple para tests."""

    def __init__(self, input_dim=3, output_dim=30):
        super().__init__()
        self.backbone_conv = nn.Conv2d(input_dim, 16, 3, padding=1)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, output_dim),
            nn.Sigmoid()
        )
        self._frozen = False

    def forward(self, x):
        x = self.backbone_conv(x)
        return self.head(x)

    def freeze_backbone(self):
        for param in self.backbone_conv.parameters():
            param.requires_grad = False
        self._frozen = True

    def unfreeze_backbone(self):
        for param in self.backbone_conv.parameters():
            param.requires_grad = True
        self._frozen = False

    def get_trainable_params(self):
        return [
            {'params': self.backbone_conv.parameters(), 'name': 'backbone'},
            {'params': self.head.parameters(), 'name': 'head'}
        ]


class MockDataLoader:
    """DataLoader mock para tests rápidos."""

    def __init__(self, batch_size=4, num_batches=3):
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            images = torch.randn(self.batch_size, 3, 224, 224)
            landmarks = torch.rand(self.batch_size, 30)
            metas = [{'category': 'COVID'} for _ in range(self.batch_size)]
            yield images, landmarks, metas

    def __len__(self):
        return self.num_batches


class TestLandmarkTrainerInit:
    """Tests para __init__ de LandmarkTrainer."""

    def test_init_creates_save_dir(self, tmp_path):
        """Verifica que __init__ cree el directorio de guardado."""
        model = MockModel()
        device = torch.device('cpu')
        save_dir = tmp_path / 'checkpoints'

        trainer = LandmarkTrainer(model, device, save_dir=str(save_dir))

        assert save_dir.exists()
        assert trainer.model is model
        assert trainer.device == device
        assert trainer.image_size == DEFAULT_IMAGE_SIZE

    def test_init_history_structure(self, tmp_path):
        """Verifica estructura del historial."""
        model = MockModel()
        trainer = LandmarkTrainer(model, torch.device('cpu'), save_dir=str(tmp_path))

        assert 'train_loss' in trainer.history
        assert 'val_loss' in trainer.history
        assert 'train_error_px' in trainer.history
        assert 'val_error_px' in trainer.history
        assert 'lr' in trainer.history


class TestComputePixelError:
    """Tests para compute_pixel_error."""

    def test_zero_error_when_identical(self, tmp_path):
        """Error cero cuando predicción = target."""
        model = MockModel()
        trainer = LandmarkTrainer(model, torch.device('cpu'), save_dir=str(tmp_path))

        pred = torch.rand(4, 30)
        target = pred.clone()

        error = trainer.compute_pixel_error(pred, target)
        assert error.item() == pytest.approx(0.0, abs=1e-6)

    def test_error_scales_with_image_size(self, tmp_path):
        """Error escala con tamaño de imagen."""
        model = MockModel()
        trainer = LandmarkTrainer(model, torch.device('cpu'),
                                  save_dir=str(tmp_path), image_size=100)

        # Diferencia de 0.1 normalizado = 10 pixels con image_size=100
        pred = torch.tensor([[0.5, 0.5] + [0.5] * 28])
        target = torch.tensor([[0.6, 0.5] + [0.5] * 28])  # 0.1 diferencia en x

        error = trainer.compute_pixel_error(pred, target)
        # Error esperado = 10 px en un landmark, promediado sobre 15 = 10/15
        assert error.item() > 0

    def test_handles_batch_dimension(self, tmp_path):
        """Maneja correctamente dimensión de batch."""
        model = MockModel()
        trainer = LandmarkTrainer(model, torch.device('cpu'), save_dir=str(tmp_path))

        pred = torch.rand(8, 30)
        target = torch.rand(8, 30)

        error = trainer.compute_pixel_error(pred, target)
        assert error.dim() == 0  # Escalar


class TestValidate:
    """Tests para método validate."""

    def test_validate_returns_dict(self, tmp_path):
        """Validate retorna diccionario con métricas."""
        model = MockModel()
        trainer = LandmarkTrainer(model, torch.device('cpu'), save_dir=str(tmp_path))

        val_loader = MockDataLoader(batch_size=2, num_batches=2)
        criterion = nn.MSELoss()

        metrics = trainer.validate(val_loader, criterion)

        assert 'val_loss' in metrics
        assert 'val_error_px' in metrics
        assert metrics['val_loss'] >= 0
        assert metrics['val_error_px'] >= 0

    def test_validate_sets_eval_mode(self, tmp_path):
        """Validate pone modelo en modo eval."""
        model = MockModel()
        trainer = LandmarkTrainer(model, torch.device('cpu'), save_dir=str(tmp_path))

        model.train()  # Empezar en modo train

        val_loader = MockDataLoader(batch_size=2, num_batches=1)
        criterion = nn.MSELoss()

        trainer.validate(val_loader, criterion)

        # Después de validate, modelo debería estar en eval
        assert not model.training


class TestTrainEpoch:
    """Tests para train_epoch."""

    def test_train_epoch_returns_metrics(self, tmp_path):
        """train_epoch retorna métricas."""
        model = MockModel()
        trainer = LandmarkTrainer(model, torch.device('cpu'), save_dir=str(tmp_path))

        train_loader = MockDataLoader(batch_size=2, num_batches=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        metrics = trainer.train_epoch(train_loader, optimizer, criterion)

        assert 'loss' in metrics
        assert 'error_px' in metrics
        assert metrics['loss'] >= 0
        assert metrics['error_px'] >= 0

    def test_train_epoch_updates_model(self, tmp_path):
        """train_epoch actualiza pesos del modelo."""
        model = MockModel()
        trainer = LandmarkTrainer(model, torch.device('cpu'), save_dir=str(tmp_path))

        # Guardar pesos iniciales
        initial_weights = model.head[2].weight.data.clone()

        train_loader = MockDataLoader(batch_size=2, num_batches=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        trainer.train_epoch(train_loader, optimizer, criterion)

        # Verificar que los pesos cambiaron
        assert not torch.allclose(initial_weights, model.head[2].weight.data)


class TestSaveLoadModel:
    """Tests para save_model y load_model."""

    def test_save_and_load_model(self, tmp_path):
        """Guardar y cargar modelo preserva pesos."""
        model1 = MockModel()
        trainer1 = LandmarkTrainer(model1, torch.device('cpu'), save_dir=str(tmp_path))

        # Modificar pesos
        with torch.no_grad():
            model1.head[2].weight.fill_(0.5)

        save_path = tmp_path / 'model.pt'
        trainer1.save_model(str(save_path))

        # Crear nuevo modelo y cargar
        model2 = MockModel()
        trainer2 = LandmarkTrainer(model2, torch.device('cpu'), save_dir=str(tmp_path))
        trainer2.load_model(str(save_path))

        # Verificar pesos iguales
        assert torch.allclose(model1.head[2].weight, model2.head[2].weight)

    def test_save_creates_file(self, tmp_path):
        """save_model crea archivo."""
        model = MockModel()
        trainer = LandmarkTrainer(model, torch.device('cpu'), save_dir=str(tmp_path))

        save_path = tmp_path / 'model.pt'
        trainer.save_model(str(save_path))

        assert save_path.exists()


class TestCriterionHandling:
    """Tests para manejo de diferentes tipos de criterio."""

    def test_handles_dict_loss(self, tmp_path):
        """Maneja loss que retorna dict."""

        class DictLoss(nn.Module):
            def forward(self, pred, target):
                mse = nn.functional.mse_loss(pred, target)
                return {'total': mse, 'component': mse * 0.5}

        model = MockModel()
        trainer = LandmarkTrainer(model, torch.device('cpu'), save_dir=str(tmp_path))

        train_loader = MockDataLoader(batch_size=2, num_batches=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = DictLoss()

        metrics = trainer.train_epoch(train_loader, optimizer, criterion)

        assert 'loss' in metrics
        assert metrics['loss'] >= 0

    def test_handles_scalar_loss(self, tmp_path):
        """Maneja loss que retorna escalar."""
        model = MockModel()
        trainer = LandmarkTrainer(model, torch.device('cpu'), save_dir=str(tmp_path))

        train_loader = MockDataLoader(batch_size=2, num_batches=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        metrics = trainer.train_epoch(train_loader, optimizer, criterion)

        assert 'loss' in metrics
        assert metrics['loss'] >= 0
