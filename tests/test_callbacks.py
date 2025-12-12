"""
Tests unitarios para callbacks de entrenamiento

Tests para src_v2/training/callbacks.py
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys
import tempfile

# Agregar src_v2 al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src_v2.training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LRSchedulerCallback,
)


class TestEarlyStopping:
    """Tests para EarlyStopping."""

    def test_init_default_params(self):
        """Verifica inicialización con parámetros por defecto."""
        es = EarlyStopping()
        assert es.patience == 10
        assert es.min_delta == 0.0
        assert es.mode == 'min'
        assert es.best_score is None
        assert es.counter == 0
        assert not es.should_stop

    def test_init_custom_params(self):
        """Verifica inicialización con parámetros personalizados."""
        es = EarlyStopping(patience=5, min_delta=0.01, mode='max')
        assert es.patience == 5
        assert es.min_delta == 0.01
        assert es.mode == 'max'

    def test_first_call_sets_best(self):
        """Primera llamada establece best_score."""
        es = EarlyStopping()
        es(10.0, epoch=0)
        assert es.best_score == 10.0
        assert es.best_epoch == 0

    def test_improvement_resets_counter_min_mode(self):
        """Mejora en modo min resetea contador."""
        es = EarlyStopping(patience=3, mode='min')
        es(10.0, epoch=0)
        es(11.0, epoch=1)  # Peor
        assert es.counter == 1
        es(9.0, epoch=2)   # Mejor
        assert es.counter == 0
        assert es.best_score == 9.0

    def test_improvement_resets_counter_max_mode(self):
        """Mejora en modo max resetea contador."""
        es = EarlyStopping(patience=3, mode='max')
        es(10.0, epoch=0)
        es(9.0, epoch=1)   # Peor
        assert es.counter == 1
        es(11.0, epoch=2)  # Mejor
        assert es.counter == 0
        assert es.best_score == 11.0

    def test_triggers_after_patience(self):
        """Dispara después de patience épocas sin mejora."""
        es = EarlyStopping(patience=3, mode='min', verbose=False)
        es(10.0, epoch=0)
        assert not es(11.0, epoch=1)  # counter=1
        assert not es(11.0, epoch=2)  # counter=2
        assert es(11.0, epoch=3)      # counter=3, dispara

    def test_min_delta_considered(self):
        """min_delta se considera en mejora."""
        es = EarlyStopping(patience=3, min_delta=0.5, mode='min', verbose=False)
        es(10.0, epoch=0)
        # 9.6 no es suficiente mejora (necesita < 9.5)
        es(9.6, epoch=1)
        assert es.counter == 1
        # 9.4 sí es suficiente mejora
        es(9.4, epoch=2)
        assert es.counter == 0

    def test_reset(self):
        """reset() reinicia estado."""
        es = EarlyStopping(patience=3)
        es(10.0, epoch=0)
        es(11.0, epoch=1)
        es(11.0, epoch=2)

        es.reset()

        assert es.best_score is None
        assert es.counter == 0
        assert es.best_epoch == 0
        assert not es.should_stop


class TestModelCheckpoint:
    """Tests para ModelCheckpoint."""

    def test_init_creates_directory(self, tmp_path):
        """__init__ crea directorio de guardado."""
        save_dir = tmp_path / 'checkpoints'
        mc = ModelCheckpoint(save_dir=str(save_dir))
        assert save_dir.exists()

    def test_saves_checkpoint_on_improvement(self, tmp_path):
        """Guarda checkpoint cuando hay mejora."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        mc = ModelCheckpoint(save_dir=str(tmp_path), monitor='val_loss', mode='min')

        path = mc(model, optimizer, epoch=0, metrics={'val_loss': 0.5})
        assert path is not None
        assert Path(path).exists()

    def test_does_not_save_without_improvement(self, tmp_path):
        """No guarda cuando no hay mejora con save_best_only=True."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        mc = ModelCheckpoint(save_dir=str(tmp_path), monitor='val_loss',
                            mode='min', save_best_only=True)

        mc(model, optimizer, epoch=0, metrics={'val_loss': 0.5})
        path = mc(model, optimizer, epoch=1, metrics={'val_loss': 0.6})  # Peor

        assert path is None

    def test_updates_best_score(self, tmp_path):
        """Actualiza best_score cuando mejora."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        mc = ModelCheckpoint(save_dir=str(tmp_path), monitor='val_loss', mode='min')

        mc(model, optimizer, epoch=0, metrics={'val_loss': 0.5})
        assert mc.best_score == 0.5

        mc(model, optimizer, epoch=1, metrics={'val_loss': 0.3})
        assert mc.best_score == 0.3

    def test_max_mode_works(self, tmp_path):
        """mode='max' funciona correctamente."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        mc = ModelCheckpoint(save_dir=str(tmp_path), monitor='accuracy',
                            mode='max', save_best_only=True)

        mc(model, optimizer, epoch=0, metrics={'accuracy': 0.8})
        path = mc(model, optimizer, epoch=1, metrics={'accuracy': 0.9})

        assert path is not None
        assert mc.best_score == 0.9

    def test_load_best_restores_weights(self, tmp_path):
        """load_best restaura pesos correctamente."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        mc = ModelCheckpoint(save_dir=str(tmp_path), monitor='val_loss', mode='min')

        # Guardar estado inicial
        with torch.no_grad():
            model.weight.fill_(1.0)

        mc(model, optimizer, epoch=0, metrics={'val_loss': 0.5})

        # Cambiar pesos
        with torch.no_grad():
            model.weight.fill_(2.0)

        # Restaurar
        mc.load_best(model)

        # Verificar restauración
        assert torch.allclose(model.weight, torch.ones_like(model.weight))

    def test_returns_none_if_monitor_missing(self, tmp_path):
        """Retorna None si métrica monitoreada no está."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        mc = ModelCheckpoint(save_dir=str(tmp_path), monitor='val_loss')

        path = mc(model, optimizer, epoch=0, metrics={'train_loss': 0.5})
        assert path is None


class TestLRSchedulerCallback:
    """Tests para LRSchedulerCallback."""

    def test_init(self):
        """Verifica inicialización."""
        optimizer = torch.optim.SGD([torch.zeros(10)], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        callback = LRSchedulerCallback(scheduler, step_on='epoch')

        assert callback.scheduler is scheduler
        assert callback.step_on == 'epoch'

    def test_step_epoch_updates_lr(self):
        """step_epoch actualiza learning rate."""
        optimizer = torch.optim.SGD([torch.zeros(10)], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        callback = LRSchedulerCallback(scheduler, step_on='epoch')

        initial_lr = optimizer.param_groups[0]['lr']
        callback.step_epoch()
        new_lr = optimizer.param_groups[0]['lr']

        assert new_lr < initial_lr

    def test_step_batch_when_configured(self):
        """step_batch actualiza cuando step_on='batch'."""
        optimizer = torch.optim.SGD([torch.zeros(10)], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        callback = LRSchedulerCallback(scheduler, step_on='batch')

        initial_lr = optimizer.param_groups[0]['lr']
        callback.step_batch()
        new_lr = optimizer.param_groups[0]['lr']

        assert new_lr < initial_lr

    def test_step_epoch_ignores_when_batch_mode(self):
        """step_epoch no actualiza cuando step_on='batch'."""
        optimizer = torch.optim.SGD([torch.zeros(10)], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        callback = LRSchedulerCallback(scheduler, step_on='batch')

        initial_lr = optimizer.param_groups[0]['lr']
        callback.step_epoch()
        new_lr = optimizer.param_groups[0]['lr']

        assert new_lr == initial_lr

    def test_get_last_lr(self):
        """get_last_lr retorna learning rate actual."""
        optimizer = torch.optim.SGD([torch.zeros(10)], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        callback = LRSchedulerCallback(scheduler)

        lr = callback.get_last_lr()
        assert lr == [0.1]

    def test_reduce_lr_on_plateau_with_metrics(self):
        """ReduceLROnPlateau funciona con métricas."""
        optimizer = torch.optim.SGD([torch.zeros(10)], lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=0
        )

        callback = LRSchedulerCallback(scheduler, step_on='epoch')

        # Primera llamada establece baseline
        callback.step_epoch(metrics={'val_loss': 0.5})
        # Segunda llamada con peor loss debería reducir LR
        callback.step_epoch(metrics={'val_loss': 0.6})

        new_lr = optimizer.param_groups[0]['lr']
        assert new_lr < 0.1
