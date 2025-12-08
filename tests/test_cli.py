"""
Smoke tests para el CLI del proyecto.

Estos tests verifican que el CLI responde correctamente sin ejecutar
operaciones completas (que serian muy lentas para tests).
"""

import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from src_v2.cli import app


runner = CliRunner()


class TestCLIHelp:
    """Tests para --help de cada comando."""

    def test_main_help(self):
        """CLI principal debe mostrar ayuda."""
        result = runner.invoke(app, ['--help'])
        assert result.exit_code == 0
        assert 'COVID-19 Detection' in result.stdout
        assert 'train' in result.stdout
        assert 'evaluate' in result.stdout
        assert 'predict' in result.stdout
        assert 'warp' in result.stdout

    def test_train_help(self):
        """Comando train debe mostrar ayuda."""
        result = runner.invoke(app, ['train', '--help'])
        assert result.exit_code == 0
        assert 'Entrenar modelo' in result.stdout
        assert '--config-path' in result.stdout
        assert '--batch-size' in result.stdout

    def test_evaluate_help(self):
        """Comando evaluate debe mostrar ayuda."""
        result = runner.invoke(app, ['evaluate', '--help'])
        assert result.exit_code == 0
        assert 'Evaluar modelo' in result.stdout
        assert 'CHECKPOINT' in result.stdout
        assert '--tta' in result.stdout

    def test_predict_help(self):
        """Comando predict debe mostrar ayuda."""
        result = runner.invoke(app, ['predict', '--help'])
        assert result.exit_code == 0
        assert 'Predecir landmarks' in result.stdout
        assert 'IMAGE' in result.stdout
        assert '--checkpoint' in result.stdout

    def test_warp_help(self):
        """Comando warp debe mostrar ayuda."""
        result = runner.invoke(app, ['warp', '--help'])
        assert result.exit_code == 0
        assert 'warping' in result.stdout.lower()
        assert 'INPUT_DIR' in result.stdout
        assert 'OUTPUT_DIR' in result.stdout

    def test_version_command(self):
        """Comando version debe mostrar version."""
        result = runner.invoke(app, ['version'])
        assert result.exit_code == 0
        assert 'COVID-19 Landmark Detection' in result.stdout
        assert 'v' in result.stdout


class TestCLIErrorHandling:
    """Tests para manejo de errores del CLI."""

    def test_train_missing_data_shows_error(self):
        """Train con datos inexistentes debe mostrar error."""
        result = runner.invoke(app, [
            'train',
            '--data-root', '/nonexistent/path',
            '--csv-path', '/nonexistent/coords.csv'
        ])
        # Puede ser exit code 1 o mostrar error
        assert result.exit_code != 0 or 'error' in result.stdout.lower()

    def test_evaluate_missing_checkpoint(self):
        """Evaluate sin checkpoint debe fallar."""
        result = runner.invoke(app, ['evaluate', '/nonexistent/model.pt'])
        assert result.exit_code != 0

    def test_predict_missing_image(self):
        """Predict con imagen inexistente debe fallar."""
        result = runner.invoke(app, [
            'predict',
            '/nonexistent/image.png',
            '--checkpoint', '/nonexistent/model.pt'
        ])
        assert result.exit_code != 0

    def test_warp_missing_input_dir(self):
        """Warp con directorio inexistente debe fallar."""
        result = runner.invoke(app, [
            'warp',
            '/nonexistent/input',
            '/tmp/output',
            '--checkpoint', '/nonexistent/model.pt'
        ])
        assert result.exit_code != 0


class TestCLIModuleExecution:
    """Tests para ejecutar como modulo Python."""

    def test_module_help(self):
        """python -m src_v2 --help debe funcionar."""
        result = subprocess.run(
            [sys.executable, '-m', 'src_v2', '--help'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert 'COVID-19 Detection' in result.stdout

    def test_module_version(self):
        """python -m src_v2 version debe funcionar."""
        result = subprocess.run(
            [sys.executable, '-m', 'src_v2', 'version'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert 'v' in result.stdout

    def test_module_train_help(self):
        """python -m src_v2 train --help debe funcionar."""
        result = subprocess.run(
            [sys.executable, '-m', 'src_v2', 'train', '--help'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert 'Entrenar' in result.stdout


class TestCLIImports:
    """Tests para verificar que imports del CLI funcionan."""

    def test_cli_module_imports(self):
        """Modulo cli debe poder importarse."""
        from src_v2 import cli
        assert hasattr(cli, 'app')
        assert hasattr(cli, 'main')

    def test_cli_app_is_typer(self):
        """app debe ser instancia de Typer."""
        import typer
        from src_v2.cli import app
        assert isinstance(app, typer.Typer)

    def test_cli_commands_registered(self):
        """Comandos deben estar registrados."""
        from src_v2.cli import app
        # Typer registra comandos en registered_commands
        # Los nombres pueden estar en el callback o ser None si no se especifico
        # Verificamos que hay 5 comandos registrados
        assert len(app.registered_commands) == 5

        # Verificar a traves de --help que los comandos existen
        result = runner.invoke(app, ['--help'])
        assert 'train' in result.stdout
        assert 'evaluate' in result.stdout
        assert 'predict' in result.stdout
        assert 'warp' in result.stdout
        assert 'version' in result.stdout


class TestCLIDeviceDetection:
    """Tests para deteccion de dispositivo."""

    def test_get_device_cpu(self):
        """get_device('cpu') debe retornar CPU."""
        import torch
        from src_v2.cli import get_device

        device = get_device('cpu')
        assert device == torch.device('cpu')

    def test_get_device_auto(self):
        """get_device('auto') debe retornar dispositivo valido."""
        import torch
        from src_v2.cli import get_device

        device = get_device('auto')
        # Debe ser cpu, cuda, o mps
        assert device.type in ['cpu', 'cuda', 'mps']


class TestArchitectureDetection:
    """Tests para deteccion automatica de arquitectura desde checkpoint."""

    def test_detect_simple_architecture(self):
        """Detectar arquitectura sin coord_attention ni deep_head."""
        import torch
        from src_v2.cli import detect_architecture_from_checkpoint

        # Simular state_dict de modelo simple (sin coord_attention, sin deep_head)
        # head simple tiene: Flatten, Dropout, Linear(512, 256), ReLU, Dropout, Linear(256, 30), Sigmoid
        # indices: 0=Flatten, 1=Dropout, 2=Linear, 3=ReLU, 4=Dropout, 5=Linear, 6=Sigmoid
        state_dict = {
            'backbone_conv.0.weight': torch.randn(64, 3, 7, 7),
            'head.2.weight': torch.randn(256, 512),  # Linear(512, 256)
            'head.2.bias': torch.randn(256),
            'head.5.weight': torch.randn(30, 256),   # Linear(256, 30)
            'head.5.bias': torch.randn(30),
        }

        result = detect_architecture_from_checkpoint(state_dict)

        assert result['use_coord_attention'] is False
        assert result['deep_head'] is False
        assert result['hidden_dim'] == 256

    def test_detect_coord_attention(self):
        """Detectar arquitectura con coord_attention."""
        import torch
        from src_v2.cli import detect_architecture_from_checkpoint

        state_dict = {
            'backbone_conv.0.weight': torch.randn(64, 3, 7, 7),
            'coord_attention.conv1.weight': torch.randn(16, 512, 1, 1),
            'coord_attention.bn1.weight': torch.randn(16),
            'head.2.weight': torch.randn(256, 512),
            'head.5.weight': torch.randn(30, 256),
        }

        result = detect_architecture_from_checkpoint(state_dict)

        assert result['use_coord_attention'] is True
        assert result['deep_head'] is False

    def test_detect_deep_head(self):
        """Detectar arquitectura con deep_head."""
        import torch
        from src_v2.cli import detect_architecture_from_checkpoint

        # deep_head tiene indices hasta 9 (Linear final)
        state_dict = {
            'backbone_conv.0.weight': torch.randn(64, 3, 7, 7),
            'head.1.weight': torch.randn(512, 512),    # Linear(512, 512)
            'head.2.weight': torch.randn(512),         # GroupNorm
            'head.5.weight': torch.randn(768, 512),    # Linear(512, 768)
            'head.6.weight': torch.randn(768),         # GroupNorm
            'head.9.weight': torch.randn(30, 768),     # Linear(768, 30)
            'head.9.bias': torch.randn(30),
        }

        result = detect_architecture_from_checkpoint(state_dict)

        assert result['use_coord_attention'] is False
        assert result['deep_head'] is True
        assert result['hidden_dim'] == 768

    def test_detect_full_architecture(self):
        """Detectar arquitectura con coord_attention y deep_head."""
        import torch
        from src_v2.cli import detect_architecture_from_checkpoint

        state_dict = {
            'backbone_conv.0.weight': torch.randn(64, 3, 7, 7),
            'coord_attention.conv1.weight': torch.randn(16, 512, 1, 1),
            'coord_attention.bn1.weight': torch.randn(16),
            'head.1.weight': torch.randn(512, 512),
            'head.5.weight': torch.randn(512, 512),    # hidden_dim=512
            'head.9.weight': torch.randn(30, 512),
        }

        result = detect_architecture_from_checkpoint(state_dict)

        assert result['use_coord_attention'] is True
        assert result['deep_head'] is True
        assert result['hidden_dim'] == 512
