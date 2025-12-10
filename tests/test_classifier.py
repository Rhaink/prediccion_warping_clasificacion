"""
Tests para el modulo clasificador COVID-19.

Incluye tests para:
- ImageClassifier model
- Funciones auxiliares (transforms, weights)
- Comandos CLI (classify, train-classifier, evaluate-classifier)
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from typer.testing import CliRunner

from src_v2.cli import app
from src_v2.models.classifier import (
    ImageClassifier,
    create_classifier,
    get_classifier_transforms,
    get_class_weights,
    GrayscaleToRGB,
)
from src_v2.constants import CLASSIFIER_CLASSES, NUM_CLASSES


runner = CliRunner()


class TestImageClassifier:
    """Tests para la clase ImageClassifier."""

    def test_create_resnet18_classifier(self):
        """Crear clasificador con ResNet-18."""
        model = ImageClassifier(backbone="resnet18", num_classes=3)
        assert model is not None
        assert model.backbone_name == "resnet18"
        assert model.num_classes == 3

    def test_create_efficientnet_classifier(self):
        """Crear clasificador con EfficientNet-B0."""
        model = ImageClassifier(backbone="efficientnet_b0", num_classes=3)
        assert model is not None
        assert model.backbone_name == "efficientnet_b0"
        assert model.num_classes == 3

    def test_invalid_backbone_raises_error(self):
        """Backbone invalido debe lanzar ValueError."""
        with pytest.raises(ValueError, match="no soportado"):
            ImageClassifier(backbone="invalid_backbone", num_classes=3)

    def test_forward_pass_resnet18(self):
        """Forward pass con ResNet-18 produce shape correcto."""
        model = ImageClassifier(backbone="resnet18", num_classes=3, pretrained=False)
        model.eval()

        # Input batch de 2 imagenes 224x224
        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 3)  # batch_size x num_classes

    def test_forward_pass_efficientnet(self):
        """Forward pass con EfficientNet-B0 produce shape correcto."""
        model = ImageClassifier(backbone="efficientnet_b0", num_classes=3, pretrained=False)
        model.eval()

        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 3)

    def test_predict_proba(self):
        """predict_proba debe retornar probabilidades (sum=1)."""
        model = ImageClassifier(backbone="resnet18", num_classes=3, pretrained=False)
        model.eval()

        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            probs = model.predict_proba(x)

        assert probs.shape == (2, 3)
        # Verificar que suman 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
        # Verificar que son no-negativos
        assert (probs >= 0).all()

    def test_predict_class(self):
        """predict debe retornar indices de clase."""
        model = ImageClassifier(backbone="resnet18", num_classes=3, pretrained=False)
        model.eval()

        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            classes = model.predict(x)

        assert classes.shape == (2,)
        assert classes.dtype == torch.int64
        # Verificar que son indices validos
        assert (classes >= 0).all()
        assert (classes < 3).all()

    def test_dropout_rate(self):
        """Verificar que dropout se configura correctamente."""
        model = ImageClassifier(backbone="resnet18", num_classes=3, dropout=0.5)
        assert model.dropout_rate == 0.5


class TestCreateClassifier:
    """Tests para la factory function create_classifier."""

    def test_create_without_checkpoint(self):
        """Crear clasificador sin checkpoint."""
        model = create_classifier(backbone="resnet18", num_classes=3)
        assert model is not None
        assert isinstance(model, ImageClassifier)

    def test_create_with_device(self):
        """Crear clasificador en dispositivo especifico."""
        device = torch.device("cpu")
        model = create_classifier(backbone="resnet18", device=device)

        # Verificar que el modelo esta en el dispositivo correcto
        param = next(model.parameters())
        assert param.device == device


class TestCheckpointCompatibility:
    """Tests para compatibilidad de checkpoints antiguos y nuevos."""

    def test_load_checkpoint_new_format(self, tmp_path):
        """Cargar checkpoint con formato nuevo (con prefijo backbone.)."""
        # Crear y guardar modelo con formato nuevo
        model = ImageClassifier(backbone="resnet18", num_classes=3, pretrained=False)
        checkpoint_path = tmp_path / "new_format.pt"

        torch.save({
            "model_state_dict": model.state_dict(),
            "model_name": "resnet18",
            "class_names": ["COVID", "Normal", "Viral_Pneumonia"],
        }, checkpoint_path)

        # Cargar modelo
        loaded_model = create_classifier(checkpoint=str(checkpoint_path))

        assert loaded_model is not None
        assert isinstance(loaded_model, ImageClassifier)

        # Verificar que funciona
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = loaded_model(x)
        assert output.shape == (1, 3)

    def test_load_checkpoint_old_format(self, tmp_path):
        """Cargar checkpoint con formato antiguo (sin prefijo backbone.)."""
        from torchvision import models

        # Crear modelo con formato antiguo (como lo hacia scripts/train_classifier.py)
        old_model = models.resnet18(weights=None)
        import torch.nn as nn
        old_model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 3)
        )

        checkpoint_path = tmp_path / "old_format.pt"

        # Guardar con formato antiguo (sin prefijo backbone.)
        torch.save({
            "model_state_dict": old_model.state_dict(),
            "model_name": "resnet18",
            "class_names": ["COVID", "Normal", "Viral_Pneumonia"],
        }, checkpoint_path)

        # Verificar que el formato es antiguo (sin backbone.)
        ckpt = torch.load(checkpoint_path, weights_only=False)
        sample_key = next(iter(ckpt["model_state_dict"].keys()))
        assert not sample_key.startswith("backbone."), "Checkpoint deberia tener formato antiguo"

        # Cargar modelo - debe convertir automaticamente
        loaded_model = create_classifier(checkpoint=str(checkpoint_path))

        assert loaded_model is not None
        assert isinstance(loaded_model, ImageClassifier)

        # Verificar que funciona
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = loaded_model(x)
        assert output.shape == (1, 3)

    def test_checkpoint_preserves_weights(self, tmp_path):
        """Verificar que los pesos se preservan al guardar/cargar."""
        model = ImageClassifier(backbone="resnet18", num_classes=3, pretrained=False)

        # Obtener prediccion antes de guardar
        x = torch.randn(1, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            output_before = model(x)

        # Guardar
        checkpoint_path = tmp_path / "test.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_name": "resnet18",
            "class_names": ["COVID", "Normal", "Viral_Pneumonia"],
        }, checkpoint_path)

        # Cargar
        loaded_model = create_classifier(checkpoint=str(checkpoint_path))
        loaded_model.eval()

        with torch.no_grad():
            output_after = loaded_model(x)

        # Verificar que las predicciones son identicas
        assert torch.allclose(output_before, output_after, atol=1e-6)

    def test_load_checkpoint_detects_backbone(self, tmp_path):
        """Verificar que detecta correctamente el backbone del checkpoint."""
        # ResNet-18
        model_resnet = ImageClassifier(backbone="resnet18", num_classes=3, pretrained=False)
        path_resnet = tmp_path / "resnet.pt"
        torch.save({
            "model_state_dict": model_resnet.state_dict(),
            "model_name": "resnet18",
            "class_names": ["A", "B", "C"],
        }, path_resnet)

        loaded = create_classifier(checkpoint=str(path_resnet))
        assert loaded.backbone_name == "resnet18"

    def test_load_checkpoint_detects_num_classes(self, tmp_path):
        """Verificar que detecta correctamente el numero de clases."""
        # Modelo con 5 clases
        model = ImageClassifier(backbone="resnet18", num_classes=5, pretrained=False)
        checkpoint_path = tmp_path / "5classes.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_name": "resnet18",
            "class_names": ["A", "B", "C", "D", "E"],
        }, checkpoint_path)

        loaded = create_classifier(checkpoint=str(checkpoint_path))
        assert loaded.num_classes == 5

        # Verificar output shape
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = loaded(x)
        assert output.shape == (1, 5)


class TestClassifierTransforms:
    """Tests para transforms del clasificador."""

    def test_train_transforms_include_augmentation(self):
        """Transforms de train deben incluir augmentacion."""
        transform = get_classifier_transforms(train=True, img_size=224)
        assert transform is not None

        # Verificar que tiene multiples transformaciones
        assert len(transform.transforms) > 3

    def test_eval_transforms_no_augmentation(self):
        """Transforms de eval no deben incluir augmentacion."""
        transform = get_classifier_transforms(train=False, img_size=224)
        assert transform is not None

        # Transforms de eval son mas simples
        train_transform = get_classifier_transforms(train=True)
        assert len(transform.transforms) < len(train_transform.transforms)

    def test_transforms_output_correct_size(self):
        """Transforms deben producir tensores del tamano correcto."""
        from PIL import Image

        transform = get_classifier_transforms(train=False, img_size=224)

        # Crear imagen de prueba
        img = Image.new('L', (299, 299), color=128)  # Grayscale

        tensor = transform(img)

        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32


class TestGrayscaleToRGB:
    """Tests para GrayscaleToRGB transform."""

    def test_grayscale_to_rgb(self):
        """Convertir imagen grayscale a RGB."""
        from PIL import Image

        transform = GrayscaleToRGB()
        img = Image.new('L', (100, 100), color=128)

        result = transform(img)

        assert result.mode == 'RGB'
        assert result.size == (100, 100)

    def test_rgb_unchanged(self):
        """Imagen RGB debe permanecer sin cambios."""
        from PIL import Image

        transform = GrayscaleToRGB()
        img = Image.new('RGB', (100, 100), color=(128, 128, 128))

        result = transform(img)

        assert result.mode == 'RGB'


class TestClassWeights:
    """Tests para calculo de pesos de clase."""

    def test_balanced_classes(self):
        """Clases balanceadas deben tener pesos iguales."""
        labels = [0, 0, 1, 1, 2, 2]  # 2 de cada clase

        weights = get_class_weights(labels, num_classes=3)

        assert weights.shape == (3,)
        # Todas las clases balanceadas = peso 1.0
        assert torch.allclose(weights, torch.ones(3), atol=0.01)

    def test_imbalanced_classes(self):
        """Clases desbalanceadas deben tener pesos diferentes."""
        labels = [0, 0, 0, 0, 1, 1, 2]  # 4, 2, 1

        weights = get_class_weights(labels, num_classes=3)

        assert weights.shape == (3,)
        # Clase minoritaria debe tener mayor peso
        assert weights[2] > weights[1] > weights[0]

    def test_returns_float_tensor(self):
        """Debe retornar FloatTensor."""
        labels = [0, 1, 2]
        weights = get_class_weights(labels, num_classes=3)

        assert isinstance(weights, torch.Tensor)
        assert weights.dtype == torch.float32


class TestClassifyCommand:
    """Tests para comando classify del CLI."""

    def test_classify_help(self):
        """Comando classify debe mostrar ayuda."""
        result = runner.invoke(app, ['classify', '--help'])
        assert result.exit_code == 0
        assert 'Clasificar' in result.stdout or 'classify' in result.stdout.lower()
        assert '--classifier' in result.stdout
        assert '--warp' in result.stdout

    def test_classify_requires_classifier(self):
        """classify sin --classifier debe fallar."""
        result = runner.invoke(app, ['classify', 'image.png'])
        assert result.exit_code != 0

    def test_classify_missing_image(self):
        """classify con imagen inexistente debe fallar."""
        result = runner.invoke(app, [
            'classify',
            '/nonexistent/image.png',
            '--classifier', '/nonexistent/classifier.pt'
        ])
        assert result.exit_code != 0

    def test_classify_warp_requires_landmark_model(self):
        """classify --warp sin modelo de landmarks debe fallar."""
        result = runner.invoke(app, [
            'classify',
            'image.png',
            '--classifier', 'clf.pt',
            '--warp'
        ])
        assert result.exit_code != 0


class TestTrainClassifierCommand:
    """Tests para comando train-classifier del CLI."""

    def test_train_classifier_help(self):
        """Comando train-classifier debe mostrar ayuda."""
        result = runner.invoke(app, ['train-classifier', '--help'])
        assert result.exit_code == 0
        assert 'Entrenar' in result.stdout or 'train' in result.stdout.lower()
        assert '--backbone' in result.stdout
        assert '--epochs' in result.stdout
        assert '--batch-size' in result.stdout

    def test_train_classifier_requires_data_dir(self):
        """train-classifier sin data_dir debe fallar."""
        result = runner.invoke(app, ['train-classifier'])
        assert result.exit_code != 0

    def test_train_classifier_missing_data(self):
        """train-classifier con datos inexistentes debe fallar."""
        result = runner.invoke(app, [
            'train-classifier',
            '/nonexistent/dataset'
        ])
        assert result.exit_code != 0


class TestEvaluateClassifierCommand:
    """Tests para comando evaluate-classifier del CLI."""

    def test_evaluate_classifier_help(self):
        """Comando evaluate-classifier debe mostrar ayuda."""
        result = runner.invoke(app, ['evaluate-classifier', '--help'])
        assert result.exit_code == 0
        assert 'Evaluar' in result.stdout or 'evaluate' in result.stdout.lower()
        assert '--data-dir' in result.stdout
        assert '--split' in result.stdout

    def test_evaluate_classifier_missing_checkpoint(self):
        """evaluate-classifier con checkpoint inexistente debe fallar."""
        result = runner.invoke(app, [
            'evaluate-classifier',
            '/nonexistent/model.pt',
            '--data-dir', '/some/path'
        ])
        assert result.exit_code != 0


class TestConstants:
    """Tests para constantes del clasificador."""

    def test_classifier_classes_defined(self):
        """CLASSIFIER_CLASSES debe estar definido correctamente."""
        assert CLASSIFIER_CLASSES is not None
        assert len(CLASSIFIER_CLASSES) == 3
        assert 'COVID' in CLASSIFIER_CLASSES
        assert 'Normal' in CLASSIFIER_CLASSES
        assert 'Viral_Pneumonia' in CLASSIFIER_CLASSES

    def test_num_classes(self):
        """NUM_CLASSES debe ser 3."""
        assert NUM_CLASSES == 3


class TestModelIntegration:
    """Tests de integracion para el modelo completo."""

    @pytest.mark.parametrize("backbone", ["resnet18", "efficientnet_b0"])
    def test_end_to_end_inference(self, backbone):
        """Test de inferencia end-to-end con diferentes backbones."""
        from PIL import Image

        # Crear modelo
        model = create_classifier(backbone=backbone, num_classes=3, pretrained=False)
        model.eval()

        # Crear imagen de prueba
        img = Image.new('L', (224, 224), color=128)

        # Aplicar transforms
        transform = get_classifier_transforms(train=False)
        tensor = transform(img).unsqueeze(0)

        # Inferencia
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)

        # Verificaciones
        assert logits.shape == (1, 3)
        assert probs.shape == (1, 3)
        assert pred.shape == (1,)
        assert 0 <= pred.item() < 3
