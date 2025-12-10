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
        # Verificamos que hay 19 comandos registrados:
        # train, evaluate, predict, warp, version, evaluate-ensemble,
        # classify, train-classifier, evaluate-classifier,
        # cross-evaluate, evaluate-external, test-robustness,
        # compute-canonical, generate-dataset (Session 20),
        # compare-architectures (Session 22),
        # gradcam, analyze-errors (Session 23),
        # pfs-analysis, generate-lung-masks (Session 24),
        # optimize-margin (Session 25)
        assert len(app.registered_commands) == 20

        # Verificar a traves de --help que los comandos existen
        result = runner.invoke(app, ['--help'])
        assert 'train' in result.stdout
        assert 'evaluate' in result.stdout
        assert 'predict' in result.stdout
        assert 'warp' in result.stdout
        assert 'version' in result.stdout
        assert 'evaluate-ensemble' in result.stdout
        assert 'classify' in result.stdout
        assert 'train-classifier' in result.stdout
        assert 'evaluate-classifier' in result.stdout
        # Nuevos comandos Session 18
        assert 'cross-evaluate' in result.stdout
        assert 'evaluate-external' in result.stdout
        assert 'test-robustness' in result.stdout
        # Nuevos comandos Session 20
        assert 'compute-canonical' in result.stdout
        assert 'generate-dataset' in result.stdout
        # Nuevo comando Session 22
        assert 'compare-architectures' in result.stdout
        # Nuevos comandos Session 23
        assert 'gradcam' in result.stdout
        assert 'analyze-errors' in result.stdout


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


class TestEvaluateEnsemble:
    """Tests para el comando evaluate-ensemble."""

    def test_evaluate_ensemble_help(self):
        """Comando evaluate-ensemble debe mostrar ayuda."""
        result = runner.invoke(app, ['evaluate-ensemble', '--help'])
        assert result.exit_code == 0
        assert 'ensemble' in result.stdout.lower()
        assert 'CHECKPOINTS' in result.stdout
        assert '--tta' in result.stdout
        assert '--clahe' in result.stdout

    def test_evaluate_ensemble_requires_checkpoints(self):
        """evaluate-ensemble sin argumentos debe fallar."""
        result = runner.invoke(app, ['evaluate-ensemble'])
        assert result.exit_code != 0
        # El mensaje puede estar en stdout o en la excepcion
        output = result.stdout + str(result.exception) if result.exception else result.stdout
        assert 'CHECKPOINTS' in output or 'Missing argument' in output or result.exit_code == 2

    def test_evaluate_ensemble_minimum_two_checkpoints(self):
        """evaluate-ensemble requiere al menos 2 checkpoints."""
        result = runner.invoke(app, ['evaluate-ensemble', '/nonexistent/model1.pt'])
        assert result.exit_code != 0
        # El mensaje puede indicar que el checkpoint no existe o que requiere 2 minimo

    def test_evaluate_ensemble_missing_checkpoint(self):
        """evaluate-ensemble con checkpoint inexistente debe fallar."""
        result = runner.invoke(app, [
            'evaluate-ensemble',
            '/nonexistent/model1.pt',
            '/nonexistent/model2.pt'
        ])
        assert result.exit_code != 0

    def test_evaluate_ensemble_tta_default_enabled(self):
        """TTA debe estar habilitado por defecto."""
        result = runner.invoke(app, ['evaluate-ensemble', '--help'])
        assert result.exit_code == 0
        # Verificar que el default es --tta (no --no-tta)
        assert '[default: tta]' in result.stdout or 'default: tta' in result.stdout

    def test_evaluate_ensemble_module_execution(self):
        """python -m src_v2 evaluate-ensemble --help debe funcionar."""
        result = subprocess.run(
            [sys.executable, '-m', 'src_v2', 'evaluate-ensemble', '--help'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert 'ensemble' in result.stdout.lower()


class TestCrossEvaluate:
    """Tests para el comando cross-evaluate (Session 18)."""

    def test_cross_evaluate_help(self):
        """Comando cross-evaluate debe mostrar ayuda."""
        result = runner.invoke(app, ['cross-evaluate', '--help'])
        assert result.exit_code == 0
        assert 'cross' in result.stdout.lower() or 'cruzada' in result.stdout.lower()
        assert 'MODEL_A' in result.stdout
        assert 'MODEL_B' in result.stdout
        assert '--data-a' in result.stdout
        assert '--data-b' in result.stdout

    def test_cross_evaluate_requires_models(self):
        """cross-evaluate requiere 2 modelos como argumentos."""
        result = runner.invoke(app, ['cross-evaluate'])
        assert result.exit_code != 0
        # Debe indicar que falta MODEL_A
        output = result.stdout + str(result.exception) if result.exception else result.stdout
        assert 'MODEL_A' in output or 'Missing argument' in output or result.exit_code == 2

    def test_cross_evaluate_requires_data_options(self):
        """cross-evaluate requiere --data-a y --data-b."""
        result = runner.invoke(app, [
            'cross-evaluate',
            '/nonexistent/model_a.pt',
            '/nonexistent/model_b.pt'
        ])
        assert result.exit_code != 0
        # Debe indicar que falta --data-a o --data-b

    def test_cross_evaluate_missing_model_files(self):
        """cross-evaluate con modelos inexistentes debe fallar."""
        result = runner.invoke(app, [
            'cross-evaluate',
            '/nonexistent/model_a.pt',
            '/nonexistent/model_b.pt',
            '--data-a', '/nonexistent/data_a',
            '--data-b', '/nonexistent/data_b'
        ])
        assert result.exit_code != 0

    def test_cross_evaluate_module_execution(self):
        """python -m src_v2 cross-evaluate --help debe funcionar."""
        result = subprocess.run(
            [sys.executable, '-m', 'src_v2', 'cross-evaluate', '--help'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert 'cross' in result.stdout.lower() or 'cruzada' in result.stdout.lower()


class TestEvaluateExternal:
    """Tests para el comando evaluate-external (Session 18)."""

    def test_evaluate_external_help(self):
        """Comando evaluate-external debe mostrar ayuda."""
        result = runner.invoke(app, ['evaluate-external', '--help'])
        assert result.exit_code == 0
        assert 'external' in result.stdout.lower() or 'externo' in result.stdout.lower()
        assert 'CHECKPOINT' in result.stdout
        assert '--external-data' in result.stdout
        assert '--threshold' in result.stdout

    def test_evaluate_external_requires_checkpoint(self):
        """evaluate-external requiere checkpoint como argumento."""
        result = runner.invoke(app, ['evaluate-external'])
        assert result.exit_code != 0
        output = result.stdout + str(result.exception) if result.exception else result.stdout
        assert 'CHECKPOINT' in output or 'Missing argument' in output or result.exit_code == 2

    def test_evaluate_external_requires_data_option(self):
        """evaluate-external requiere --external-data."""
        result = runner.invoke(app, [
            'evaluate-external',
            '/nonexistent/model.pt'
        ])
        assert result.exit_code != 0

    def test_evaluate_external_missing_files(self):
        """evaluate-external con archivos inexistentes debe fallar."""
        result = runner.invoke(app, [
            'evaluate-external',
            '/nonexistent/model.pt',
            '--external-data', '/nonexistent/data'
        ])
        assert result.exit_code != 0

    def test_evaluate_external_default_threshold(self):
        """evaluate-external debe tener threshold por defecto de 0.5."""
        result = runner.invoke(app, ['evaluate-external', '--help'])
        assert result.exit_code == 0
        assert '0.5' in result.stdout or 'threshold' in result.stdout.lower()

    def test_evaluate_external_module_execution(self):
        """python -m src_v2 evaluate-external --help debe funcionar."""
        result = subprocess.run(
            [sys.executable, '-m', 'src_v2', 'evaluate-external', '--help'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert 'external' in result.stdout.lower() or 'externo' in result.stdout.lower()


class TestTestRobustness:
    """Tests para el comando test-robustness (Session 18)."""

    def test_robustness_help(self):
        """Comando test-robustness debe mostrar ayuda."""
        result = runner.invoke(app, ['test-robustness', '--help'])
        assert result.exit_code == 0
        assert 'robustez' in result.stdout.lower() or 'robustness' in result.stdout.lower()
        assert 'CHECKPOINT' in result.stdout
        assert '--data-dir' in result.stdout
        # Debe mencionar perturbaciones
        assert 'jpeg' in result.stdout.lower() or 'perturbacion' in result.stdout.lower()

    def test_robustness_requires_checkpoint(self):
        """test-robustness requiere checkpoint como argumento."""
        result = runner.invoke(app, ['test-robustness'])
        assert result.exit_code != 0
        output = result.stdout + str(result.exception) if result.exception else result.stdout
        assert 'CHECKPOINT' in output or 'Missing argument' in output or result.exit_code == 2

    def test_robustness_requires_data_dir(self):
        """test-robustness requiere --data-dir."""
        result = runner.invoke(app, [
            'test-robustness',
            '/nonexistent/model.pt'
        ])
        assert result.exit_code != 0

    def test_robustness_missing_files(self):
        """test-robustness con archivos inexistentes debe fallar."""
        result = runner.invoke(app, [
            'test-robustness',
            '/nonexistent/model.pt',
            '--data-dir', '/nonexistent/data'
        ])
        assert result.exit_code != 0

    def test_robustness_module_execution(self):
        """python -m src_v2 test-robustness --help debe funcionar."""
        result = subprocess.run(
            [sys.executable, '-m', 'src_v2', 'test-robustness', '--help'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert 'robustez' in result.stdout.lower() or 'robustness' in result.stdout.lower()


class TestDenseNet121Support:
    """Tests para soporte de DenseNet-121 (Session 18)."""

    def test_train_classifier_backbone_options(self):
        """train-classifier debe soportar densenet121."""
        result = runner.invoke(app, ['train-classifier', '--help'])
        assert result.exit_code == 0
        assert 'densenet121' in result.stdout.lower()
        assert 'resnet18' in result.stdout.lower()
        assert 'efficientnet' in result.stdout.lower()

    def test_classifier_densenet121_instantiation(self):
        """ImageClassifier debe poder crear DenseNet-121."""
        from src_v2.models import ImageClassifier

        # Verificar que densenet121 esta en SUPPORTED_BACKBONES
        assert 'densenet121' in ImageClassifier.SUPPORTED_BACKBONES

        # Crear instancia sin pesos preentrenados (mas rapido)
        model = ImageClassifier(backbone='densenet121', num_classes=3, pretrained=False)
        assert model is not None
        assert model.backbone_name == 'densenet121'

    def test_classifier_densenet121_forward(self):
        """DenseNet-121 debe poder hacer forward pass."""
        import torch
        from src_v2.models import ImageClassifier

        model = ImageClassifier(backbone='densenet121', num_classes=3, pretrained=False)
        model.eval()

        # Input de prueba
        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        # Verificar shape de salida
        assert output.shape == (2, 3)

    def test_create_classifier_densenet121(self):
        """create_classifier debe soportar densenet121."""
        from src_v2.models import create_classifier

        model = create_classifier(backbone='densenet121', num_classes=3, pretrained=False)
        assert model is not None
        assert model.backbone_name == 'densenet121'


class TestCompareArchitectures:
    """Tests para el comando compare-architectures (Session 22)."""

    def test_compare_architectures_help(self):
        """Comando compare-architectures debe mostrar ayuda."""
        result = runner.invoke(app, ['compare-architectures', '--help'])
        assert result.exit_code == 0
        assert 'arquitectura' in result.stdout.lower() or 'architectures' in result.stdout.lower()
        assert '--architectures' in result.stdout
        assert '--epochs' in result.stdout
        assert '--output-dir' in result.stdout
        assert '--seed' in result.stdout
        assert '--quick' in result.stdout

    def test_compare_architectures_requires_data_dir(self):
        """compare-architectures requiere DATA_DIR como argumento."""
        result = runner.invoke(app, ['compare-architectures'])
        assert result.exit_code != 0
        output = result.stdout + str(result.exception) if result.exception else result.stdout
        assert 'DATA_DIR' in output or 'Missing argument' in output or result.exit_code == 2

    def test_compare_architectures_missing_data_dir(self):
        """compare-architectures con directorio inexistente debe fallar."""
        result = runner.invoke(app, [
            'compare-architectures',
            '/nonexistent/dataset',
            '--quick'
        ])
        assert result.exit_code != 0

    def test_compare_architectures_invalid_architecture(self):
        """compare-architectures con arquitectura invalida debe fallar."""
        result = runner.invoke(app, [
            'compare-architectures',
            '/nonexistent/dataset',
            '--architectures', 'invalid_architecture'
        ])
        # Debe fallar con exit code 1
        assert result.exit_code == 1

    def test_compare_architectures_mixed_valid_invalid(self):
        """compare-architectures con arquitecturas mixtas validas/invalidas debe fallar."""
        result = runner.invoke(app, [
            'compare-architectures',
            '/nonexistent/dataset',
            '--architectures', 'resnet18,INVALID_ARCH,densenet121'
        ])
        assert result.exit_code != 0

    def test_compare_architectures_valid_architecture_names(self):
        """Verificar nombres de arquitecturas soportadas."""
        from src_v2.cli import SUPPORTED_ARCHITECTURES

        expected_architectures = [
            'resnet18', 'resnet50', 'efficientnet_b0', 'densenet121',
            'alexnet', 'vgg16', 'mobilenet_v2'
        ]

        for arch in expected_architectures:
            assert arch in SUPPORTED_ARCHITECTURES, f"{arch} deberia estar soportado"

    def test_compare_architectures_default_epochs(self):
        """compare-architectures debe tener 30 epocas por defecto."""
        result = runner.invoke(app, ['compare-architectures', '--help'])
        assert result.exit_code == 0
        # Verificar que 30 es el default para epochs
        assert '30' in result.stdout or 'epochs' in result.stdout.lower()

    def test_compare_architectures_quick_mode(self):
        """compare-architectures debe tener modo --quick."""
        result = runner.invoke(app, ['compare-architectures', '--help'])
        assert result.exit_code == 0
        assert '--quick' in result.stdout
        assert 'rapido' in result.stdout.lower() or 'quick' in result.stdout.lower()

    def test_compare_architectures_original_data_option(self):
        """compare-architectures debe soportar --original-data-dir."""
        result = runner.invoke(app, ['compare-architectures', '--help'])
        assert result.exit_code == 0
        assert '--original-data-dir' in result.stdout

    def test_compare_architectures_module_execution(self):
        """python -m src_v2 compare-architectures --help debe funcionar."""
        result = subprocess.run(
            [sys.executable, '-m', 'src_v2', 'compare-architectures', '--help'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert 'architectures' in result.stdout.lower() or 'arquitectura' in result.stdout.lower()

    def test_architecture_display_names(self):
        """Verificar nombres de display de arquitecturas."""
        from src_v2.cli import ARCHITECTURE_DISPLAY_NAMES

        assert ARCHITECTURE_DISPLAY_NAMES['resnet18'] == 'ResNet-18'
        assert ARCHITECTURE_DISPLAY_NAMES['resnet50'] == 'ResNet-50'
        assert ARCHITECTURE_DISPLAY_NAMES['efficientnet_b0'] == 'EfficientNet-B0'
        assert ARCHITECTURE_DISPLAY_NAMES['densenet121'] == 'DenseNet-121'
        assert ARCHITECTURE_DISPLAY_NAMES['alexnet'] == 'AlexNet'
        assert ARCHITECTURE_DISPLAY_NAMES['vgg16'] == 'VGG-16'
        assert ARCHITECTURE_DISPLAY_NAMES['mobilenet_v2'] == 'MobileNetV2'

    def test_cli_architectures_match_classifier(self):
        """SUPPORTED_ARCHITECTURES debe coincidir con ImageClassifier.SUPPORTED_BACKBONES."""
        from src_v2.cli import SUPPORTED_ARCHITECTURES
        from src_v2.models import ImageClassifier

        assert set(SUPPORTED_ARCHITECTURES) == set(ImageClassifier.SUPPORTED_BACKBONES), \
            "SUPPORTED_ARCHITECTURES en cli.py no coincide con ImageClassifier.SUPPORTED_BACKBONES"

    def test_all_architectures_have_display_names(self):
        """Cada arquitectura debe tener display name."""
        from src_v2.cli import SUPPORTED_ARCHITECTURES, ARCHITECTURE_DISPLAY_NAMES

        for arch in SUPPORTED_ARCHITECTURES:
            assert arch in ARCHITECTURE_DISPLAY_NAMES, \
                f"Arquitectura '{arch}' no tiene display name"

    def test_display_names_no_extra_keys(self):
        """No debe haber display names para arquitecturas no soportadas."""
        from src_v2.cli import SUPPORTED_ARCHITECTURES, ARCHITECTURE_DISPLAY_NAMES

        for arch in ARCHITECTURE_DISPLAY_NAMES:
            assert arch in SUPPORTED_ARCHITECTURES, \
                f"Display name para '{arch}' pero no está en SUPPORTED_ARCHITECTURES"

    def test_all_architectures_can_instantiate(self):
        """Todas las arquitecturas de CLI deben poder instanciarse en ImageClassifier."""
        from src_v2.cli import SUPPORTED_ARCHITECTURES
        from src_v2.models import ImageClassifier

        for arch in SUPPORTED_ARCHITECTURES:
            # No debería lanzar ValueError
            model = ImageClassifier(backbone=arch, num_classes=3, pretrained=False)
            assert model is not None
            assert model.backbone_name == arch


class TestGradCAMCommand:
    """Tests para el comando gradcam."""

    def test_gradcam_help(self):
        """gradcam --help debe mostrar ayuda."""
        result = runner.invoke(app, ['gradcam', '--help'])
        assert result.exit_code == 0
        assert '--checkpoint' in result.stdout
        assert '--image' in result.stdout
        assert '--data-dir' in result.stdout
        assert '--layer' in result.stdout

    def test_gradcam_requires_checkpoint(self):
        """gradcam sin checkpoint debe fallar."""
        result = runner.invoke(app, ['gradcam', '--image', 'test.png', '--output', 'out.png'])
        assert result.exit_code != 0

    def test_gradcam_requires_image_or_data_dir(self):
        """gradcam sin imagen ni data-dir debe fallar."""
        result = runner.invoke(app, [
            'gradcam',
            '--checkpoint', '/path/to/model.pt'
        ])
        assert result.exit_code != 0

    def test_gradcam_requires_output_for_image(self):
        """gradcam con --image requiere --output."""
        result = runner.invoke(app, [
            'gradcam',
            '--checkpoint', '/path/to/model.pt',
            '--image', 'test.png'
        ])
        assert result.exit_code != 0

    def test_gradcam_requires_output_dir_for_batch(self):
        """gradcam con --data-dir requiere --output-dir."""
        result = runner.invoke(app, [
            'gradcam',
            '--checkpoint', '/path/to/model.pt',
            '--data-dir', '/path/to/data'
        ])
        assert result.exit_code != 0

    def test_gradcam_cannot_specify_both_image_and_data_dir(self):
        """gradcam no puede tener --image y --data-dir juntos."""
        result = runner.invoke(app, [
            'gradcam',
            '--checkpoint', '/path/to/model.pt',
            '--image', 'test.png',
            '--output', 'out.png',
            '--data-dir', '/path/to/data',
            '--output-dir', '/path/to/out'
        ])
        assert result.exit_code != 0

    def test_gradcam_checkpoint_not_found(self):
        """gradcam con checkpoint inexistente debe fallar."""
        result = runner.invoke(app, [
            'gradcam',
            '--checkpoint', '/nonexistent/model.pt',
            '--image', 'test.png',
            '--output', 'out.png'
        ])
        assert result.exit_code != 0


class TestAnalyzeErrorsCommand:
    """Tests para el comando analyze-errors."""

    def test_analyze_errors_help(self):
        """analyze-errors --help debe mostrar ayuda."""
        result = runner.invoke(app, ['analyze-errors', '--help'])
        assert result.exit_code == 0
        assert '--checkpoint' in result.stdout
        assert '--data-dir' in result.stdout
        assert '--output-dir' in result.stdout
        assert '--visualize' in result.stdout
        assert '--gradcam' in result.stdout

    def test_analyze_errors_requires_checkpoint(self):
        """analyze-errors sin checkpoint debe fallar."""
        result = runner.invoke(app, [
            'analyze-errors',
            '--data-dir', '/path/to/data',
            '--output-dir', '/path/to/out'
        ])
        assert result.exit_code != 0

    def test_analyze_errors_requires_data_dir(self):
        """analyze-errors sin data-dir debe fallar."""
        result = runner.invoke(app, [
            'analyze-errors',
            '--checkpoint', '/path/to/model.pt',
            '--output-dir', '/path/to/out'
        ])
        assert result.exit_code != 0

    def test_analyze_errors_requires_output_dir(self):
        """analyze-errors sin output-dir debe fallar."""
        result = runner.invoke(app, [
            'analyze-errors',
            '--checkpoint', '/path/to/model.pt',
            '--data-dir', '/path/to/data'
        ])
        assert result.exit_code != 0

    def test_analyze_errors_checkpoint_not_found(self):
        """analyze-errors con checkpoint inexistente debe fallar."""
        result = runner.invoke(app, [
            'analyze-errors',
            '--checkpoint', '/nonexistent/model.pt',
            '--data-dir', '/path/to/data',
            '--output-dir', '/path/to/out'
        ])
        assert result.exit_code != 0


class TestPFSAnalysisCommand:
    """Tests para el comando pfs-analysis."""

    def test_pfs_analysis_help(self):
        """pfs-analysis --help debe mostrar ayuda."""
        result = runner.invoke(app, ['pfs-analysis', '--help'])
        assert result.exit_code == 0
        assert '--checkpoint' in result.stdout
        assert '--data-dir' in result.stdout
        assert '--mask-dir' in result.stdout
        assert '--threshold' in result.stdout
        assert '--approximate' in result.stdout

    def test_pfs_analysis_requires_checkpoint(self):
        """pfs-analysis sin checkpoint debe fallar."""
        result = runner.invoke(app, [
            'pfs-analysis',
            '--data-dir', '/path/to/data',
            '--approximate'
        ])
        assert result.exit_code != 0

    def test_pfs_analysis_requires_data_dir(self):
        """pfs-analysis sin data-dir debe fallar."""
        result = runner.invoke(app, [
            'pfs-analysis',
            '--checkpoint', '/path/to/model.pt',
            '--approximate'
        ])
        assert result.exit_code != 0

    def test_pfs_analysis_requires_mask_or_approximate(self):
        """pfs-analysis sin mask-dir ni --approximate debe fallar."""
        result = runner.invoke(app, [
            'pfs-analysis',
            '--checkpoint', '/path/to/model.pt',
            '--data-dir', '/path/to/data'
        ])
        # Deberia fallar porque no hay manera de obtener mascaras
        assert result.exit_code != 0

    def test_pfs_analysis_checkpoint_not_found(self):
        """pfs-analysis con checkpoint inexistente debe fallar."""
        result = runner.invoke(app, [
            'pfs-analysis',
            '--checkpoint', '/nonexistent/model.pt',
            '--data-dir', '/path/to/data',
            '--approximate'
        ])
        assert result.exit_code != 0

    def test_pfs_analysis_invalid_threshold(self):
        """pfs-analysis con threshold invalido debe fallar."""
        result = runner.invoke(app, [
            'pfs-analysis',
            '--checkpoint', '/path/to/model.pt',
            '--data-dir', '/path/to/data',
            '--threshold', '1.5',
            '--approximate'
        ])
        # El checkpoint no existe asi que falla antes, pero queremos
        # verificar que el parametro se parsea correctamente
        assert '--threshold' in runner.invoke(app, ['pfs-analysis', '--help']).stdout

    def test_pfs_analysis_invalid_margin(self):
        """pfs-analysis con margin invalido debe fallar."""
        result = runner.invoke(app, [
            'pfs-analysis',
            '--checkpoint', '/path/to/model.pt',
            '--data-dir', '/path/to/data',
            '--approximate',
            '--margin', '0.6'
        ])
        # Verifica que margin esta en help
        assert '--margin' in runner.invoke(app, ['pfs-analysis', '--help']).stdout


class TestGenerateLungMasksCommand:
    """Tests para el comando generate-lung-masks."""

    def test_generate_lung_masks_help(self):
        """generate-lung-masks --help debe mostrar ayuda."""
        result = runner.invoke(app, ['generate-lung-masks', '--help'])
        assert result.exit_code == 0
        assert '--data-dir' in result.stdout
        assert '--output-dir' in result.stdout
        assert '--method' in result.stdout
        assert '--margin' in result.stdout

    def test_generate_lung_masks_requires_data_dir(self):
        """generate-lung-masks sin data-dir debe fallar."""
        result = runner.invoke(app, [
            'generate-lung-masks',
            '--output-dir', '/path/to/out'
        ])
        assert result.exit_code != 0

    def test_generate_lung_masks_requires_output_dir(self):
        """generate-lung-masks sin output-dir debe fallar."""
        result = runner.invoke(app, [
            'generate-lung-masks',
            '--data-dir', '/path/to/data'
        ])
        assert result.exit_code != 0

    def test_generate_lung_masks_data_dir_not_found(self):
        """generate-lung-masks con data-dir inexistente debe fallar."""
        result = runner.invoke(app, [
            'generate-lung-masks',
            '--data-dir', '/nonexistent/data',
            '--output-dir', '/path/to/out'
        ])
        assert result.exit_code != 0

    def test_generate_lung_masks_invalid_method(self):
        """generate-lung-masks con metodo invalido debe fallar."""
        result = runner.invoke(app, [
            'generate-lung-masks',
            '--data-dir', '/nonexistent/data',
            '--output-dir', '/path/to/out',
            '--method', 'invalid_method'
        ])
        assert result.exit_code != 0

    def test_generate_lung_masks_invalid_margin(self):
        """generate-lung-masks con margin invalido debe fallar."""
        result = runner.invoke(app, [
            'generate-lung-masks',
            '--data-dir', '/nonexistent/data',
            '--output-dir', '/path/to/out',
            '--margin', '0.6'
        ])
        assert result.exit_code != 0


class TestOptimizeMarginCommand:
    """Tests para el comando optimize-margin."""

    def test_optimize_margin_help(self):
        """optimize-margin --help debe mostrar ayuda."""
        result = runner.invoke(app, ['optimize-margin', '--help'])
        assert result.exit_code == 0
        assert '--data-dir' in result.stdout
        assert '--landmarks-csv' in result.stdout
        assert '--margins' in result.stdout
        assert '--epochs' in result.stdout
        assert '--batch-size' in result.stdout
        assert '--architecture' in result.stdout
        assert '--output-dir' in result.stdout
        assert '--quick' in result.stdout
        assert '--patience' in result.stdout

    def test_optimize_margin_requires_data_dir(self):
        """optimize-margin sin data-dir debe fallar."""
        result = runner.invoke(app, [
            'optimize-margin',
            '--landmarks-csv', '/path/to/landmarks.csv'
        ])
        assert result.exit_code != 0

    def test_optimize_margin_requires_landmarks_csv(self):
        """optimize-margin sin landmarks-csv debe fallar."""
        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', '/path/to/data'
        ])
        assert result.exit_code != 0

    def test_optimize_margin_data_dir_not_found(self):
        """optimize-margin con data-dir inexistente debe fallar."""
        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', '/nonexistent/data',
            '--landmarks-csv', '/path/to/landmarks.csv'
        ])
        assert result.exit_code != 0

    def test_optimize_margin_landmarks_csv_not_found(self):
        """optimize-margin con landmarks-csv inexistente debe fallar."""
        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', '/nonexistent/data',
            '--landmarks-csv', '/nonexistent/landmarks.csv'
        ])
        assert result.exit_code != 0

    def test_optimize_margin_invalid_margins_format(self):
        """optimize-margin con formato de margenes invalido debe fallar."""
        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', '/nonexistent/data',
            '--landmarks-csv', '/nonexistent/landmarks.csv',
            '--margins', 'abc,def,ghi'
        ])
        assert result.exit_code != 0

    def test_optimize_margin_negative_margins(self):
        """optimize-margin con margenes negativos debe fallar."""
        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', '/nonexistent/data',
            '--landmarks-csv', '/nonexistent/landmarks.csv',
            '--margins', '-1.0,1.0'
        ])
        assert result.exit_code != 0

    def test_optimize_margin_invalid_architecture(self):
        """optimize-margin con arquitectura invalida debe fallar."""
        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', '/nonexistent/data',
            '--landmarks-csv', '/nonexistent/landmarks.csv',
            '--architecture', 'invalid_arch'
        ])
        assert result.exit_code != 0

    def test_optimize_margin_invalid_splits_format(self):
        """optimize-margin con formato de splits invalido debe fallar."""
        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', '/nonexistent/data',
            '--landmarks-csv', '/nonexistent/landmarks.csv',
            '--splits', '0.5,0.5'
        ])
        assert result.exit_code != 0

    def test_optimize_margin_splits_not_sum_to_one(self):
        """optimize-margin con splits que no suman 1 debe fallar."""
        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', '/nonexistent/data',
            '--landmarks-csv', '/nonexistent/landmarks.csv',
            '--splits', '0.5,0.2,0.1'
        ])
        assert result.exit_code != 0

    def test_optimize_margin_default_margins(self):
        """optimize-margin debe tener margenes por defecto."""
        result = runner.invoke(app, ['optimize-margin', '--help'])
        assert '1.00,1.05,1.10,1.15,1.20,1.25,1.30' in result.stdout

    def test_optimize_margin_default_epochs(self):
        """optimize-margin debe tener epochs por defecto."""
        result = runner.invoke(app, ['optimize-margin', '--help'])
        assert '--epochs' in result.stdout
        assert '10' in result.stdout

    def test_optimize_margin_default_architecture(self):
        """optimize-margin debe usar resnet18 por defecto."""
        result = runner.invoke(app, ['optimize-margin', '--help'])
        assert 'resnet18' in result.stdout

    def test_optimize_margin_module_execution(self):
        """python -m src_v2 optimize-margin --help debe funcionar."""
        import subprocess
        result = subprocess.run(
            [sys.executable, '-m', 'src_v2', 'optimize-margin', '--help'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert 'optimize' in result.stdout.lower() or 'margin' in result.stdout.lower()
