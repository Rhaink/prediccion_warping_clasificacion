"""
Tests para el modulo de visualizacion.

Pruebas unitarias para GradCAM y ErrorAnalyzer.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn


class TestGradCAMModule:
    """Tests para el modulo gradcam."""

    def test_target_layer_map_has_all_architectures(self):
        """TARGET_LAYER_MAP debe tener todas las arquitecturas soportadas."""
        from src_v2.visualization.gradcam import TARGET_LAYER_MAP
        from src_v2.models import ImageClassifier

        for backbone in ImageClassifier.SUPPORTED_BACKBONES:
            assert backbone in TARGET_LAYER_MAP, \
                f"Backbone '{backbone}' no tiene target layer definido"

    def test_get_target_layer_resnet18(self):
        """get_target_layer debe funcionar para ResNet-18."""
        from src_v2.models import ImageClassifier
        from src_v2.visualization.gradcam import get_target_layer

        model = ImageClassifier(backbone="resnet18", num_classes=3, pretrained=False)
        layer = get_target_layer(model, "resnet18")
        assert layer is not None
        # Verificar que es un modulo PyTorch
        assert isinstance(layer, nn.Module)

    def test_get_target_layer_all_architectures(self):
        """get_target_layer debe funcionar para todas las arquitecturas."""
        from src_v2.models import ImageClassifier
        from src_v2.visualization.gradcam import get_target_layer

        for backbone in ImageClassifier.SUPPORTED_BACKBONES:
            model = ImageClassifier(backbone=backbone, num_classes=3, pretrained=False)
            layer = get_target_layer(model, backbone)
            assert layer is not None, f"get_target_layer fallo para {backbone}"
            assert isinstance(layer, nn.Module)

    def test_get_target_layer_invalid_backbone(self):
        """get_target_layer debe fallar para backbones invalidos."""
        from src_v2.models import ImageClassifier
        from src_v2.visualization.gradcam import get_target_layer

        model = ImageClassifier(backbone="resnet18", num_classes=3, pretrained=False)
        with pytest.raises(ValueError):
            get_target_layer(model, "invalid_backbone")

    def test_gradcam_generates_heatmap(self):
        """GradCAM debe generar heatmap con dimensiones correctas."""
        from src_v2.models import ImageClassifier
        from src_v2.visualization.gradcam import GradCAM, get_target_layer

        model = ImageClassifier(backbone="resnet18", num_classes=3, pretrained=False)
        model.eval()
        layer = get_target_layer(model, "resnet18")

        gradcam = GradCAM(model, layer)
        input_tensor = torch.randn(1, 3, 224, 224)

        try:
            heatmap, pred_class, confidence = gradcam(input_tensor)

            # Verificar heatmap
            assert isinstance(heatmap, np.ndarray)
            assert heatmap.shape == (224, 224)
            assert heatmap.min() >= 0
            assert heatmap.max() <= 1

            # Verificar prediccion
            assert isinstance(pred_class, int)
            assert 0 <= pred_class < 3

            # Verificar confianza
            assert isinstance(confidence, float)
            assert 0 <= confidence <= 1
        finally:
            gradcam.remove_hooks()

    def test_gradcam_context_manager(self):
        """GradCAM debe funcionar como context manager."""
        from src_v2.models import ImageClassifier
        from src_v2.visualization.gradcam import GradCAM, get_target_layer

        model = ImageClassifier(backbone="resnet18", num_classes=3, pretrained=False)
        model.eval()
        layer = get_target_layer(model, "resnet18")

        with GradCAM(model, layer) as gradcam:
            input_tensor = torch.randn(1, 3, 224, 224)
            heatmap, pred_class, confidence = gradcam(input_tensor)
            assert heatmap is not None

    def test_calculate_pfs_valid_input(self):
        """calculate_pfs debe calcular correctamente."""
        from src_v2.visualization.gradcam import calculate_pfs

        # Heatmap y mask del mismo tamano
        heatmap = np.ones((224, 224)) * 0.5
        mask = np.zeros((224, 224))
        mask[50:150, 50:150] = 1.0  # Region pulmonar

        pfs = calculate_pfs(heatmap, mask)

        # PFS debe ser la fraccion de overlap
        assert 0 <= pfs <= 1

    def test_calculate_pfs_full_overlap(self):
        """calculate_pfs debe ser 1.0 si heatmap esta completamente en mask."""
        from src_v2.visualization.gradcam import calculate_pfs

        # Heatmap solo donde hay mask
        heatmap = np.zeros((100, 100))
        mask = np.zeros((100, 100))
        mask[25:75, 25:75] = 1.0
        heatmap[25:75, 25:75] = 1.0

        pfs = calculate_pfs(heatmap, mask)
        assert pfs == pytest.approx(1.0, rel=0.01)

    def test_calculate_pfs_no_overlap(self):
        """calculate_pfs debe ser 0.0 si no hay overlap."""
        from src_v2.visualization.gradcam import calculate_pfs

        heatmap = np.zeros((100, 100))
        heatmap[0:25, 0:25] = 1.0  # Esquina superior izquierda

        mask = np.zeros((100, 100))
        mask[75:100, 75:100] = 1.0  # Esquina inferior derecha

        pfs = calculate_pfs(heatmap, mask)
        assert pfs == pytest.approx(0.0, abs=0.01)

    def test_overlay_heatmap_output_shape(self):
        """overlay_heatmap debe retornar imagen RGB."""
        from src_v2.visualization.gradcam import overlay_heatmap

        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        heatmap = np.random.rand(224, 224)

        overlay = overlay_heatmap(image, heatmap)

        assert overlay.shape == (224, 224, 3)
        assert overlay.dtype == np.uint8

    def test_overlay_heatmap_grayscale_input(self):
        """overlay_heatmap debe aceptar imagenes grayscale."""
        from src_v2.visualization.gradcam import overlay_heatmap

        image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        heatmap = np.random.rand(224, 224)

        overlay = overlay_heatmap(image, heatmap)

        assert overlay.shape == (224, 224, 3)
        assert overlay.dtype == np.uint8


class TestErrorAnalysisModule:
    """Tests para el modulo error_analysis."""

    def test_error_analyzer_initialization(self):
        """ErrorAnalyzer debe inicializarse correctamente."""
        from src_v2.visualization.error_analysis import ErrorAnalyzer

        class_names = ['COVID', 'Normal', 'Viral_Pneumonia']
        analyzer = ErrorAnalyzer(class_names)

        assert analyzer.class_names == class_names
        assert analyzer.num_classes == 3
        assert len(analyzer.errors) == 0
        assert len(analyzer.correct) == 0

    def test_add_prediction_correct(self):
        """add_prediction debe registrar predicciones correctas."""
        from src_v2.visualization.error_analysis import ErrorAnalyzer

        analyzer = ErrorAnalyzer(['COVID', 'Normal', 'Viral_Pneumonia'])
        output = torch.tensor([10.0, 1.0, 1.0])  # Prediccion fuerte para clase 0
        label = 0  # COVID

        is_correct = analyzer.add_prediction(output, label, '/path/to/image.png')

        assert is_correct is True
        assert len(analyzer.errors) == 0
        assert len(analyzer.correct) == 1

    def test_add_prediction_error(self):
        """add_prediction debe registrar predicciones incorrectas."""
        from src_v2.visualization.error_analysis import ErrorAnalyzer

        analyzer = ErrorAnalyzer(['COVID', 'Normal', 'Viral_Pneumonia'])
        output = torch.tensor([1.0, 10.0, 1.0])  # Prediccion fuerte para clase 1
        label = 0  # COVID pero predijo Normal

        is_correct = analyzer.add_prediction(output, label, '/path/to/image.png')

        assert is_correct is False
        assert len(analyzer.errors) == 1
        assert len(analyzer.correct) == 0
        assert analyzer.errors[0].true_class == 'COVID'
        assert analyzer.errors[0].predicted_class == 'Normal'

    def test_add_batch(self):
        """add_batch debe procesar multiples predicciones."""
        from src_v2.visualization.error_analysis import ErrorAnalyzer

        analyzer = ErrorAnalyzer(['COVID', 'Normal', 'Viral_Pneumonia'])

        # Batch de 4 predicciones: 2 correctas, 2 incorrectas
        outputs = torch.tensor([
            [10.0, 1.0, 1.0],  # Pred 0 (COVID)
            [1.0, 10.0, 1.0],  # Pred 1 (Normal)
            [10.0, 1.0, 1.0],  # Pred 0 (COVID) - incorrecto
            [1.0, 1.0, 10.0],  # Pred 2 (VP) - incorrecto
        ])
        labels = torch.tensor([0, 1, 1, 0])  # COVID, Normal, Normal, COVID
        paths = ['/a.png', '/b.png', '/c.png', '/d.png']

        errors_count = analyzer.add_batch(outputs, labels, paths)

        assert errors_count == 2
        assert len(analyzer.errors) == 2
        assert len(analyzer.correct) == 2

    def test_get_summary(self):
        """get_summary debe retornar estadisticas correctas."""
        from src_v2.visualization.error_analysis import ErrorAnalyzer

        analyzer = ErrorAnalyzer(['COVID', 'Normal', 'Viral_Pneumonia'])

        # Agregar predicciones
        outputs = torch.tensor([
            [10.0, 1.0, 1.0],  # Pred 0 - correcto
            [1.0, 10.0, 1.0],  # Pred 1 - correcto
            [10.0, 1.0, 1.0],  # Pred 0 - error (label=1)
            [1.0, 1.0, 10.0],  # Pred 2 - error (label=0)
        ])
        labels = torch.tensor([0, 1, 1, 0])
        paths = ['/a.png', '/b.png', '/c.png', '/d.png']
        analyzer.add_batch(outputs, labels, paths)

        summary = analyzer.get_summary()

        assert summary.total_samples == 4
        assert summary.total_errors == 2
        assert summary.error_rate == pytest.approx(0.5)

    def test_get_top_errors(self):
        """get_top_errors debe retornar errores ordenados por confianza."""
        from src_v2.visualization.error_analysis import ErrorAnalyzer

        analyzer = ErrorAnalyzer(['COVID', 'Normal', 'Viral_Pneumonia'])

        # Agregar errores con diferentes confianzas
        analyzer.add_prediction(torch.tensor([1.0, 5.0, 1.0]), 0, '/low_conf.png')
        analyzer.add_prediction(torch.tensor([1.0, 20.0, 1.0]), 0, '/high_conf.png')
        analyzer.add_prediction(torch.tensor([1.0, 10.0, 1.0]), 0, '/med_conf.png')

        top_errors = analyzer.get_top_errors(k=2, descending=True)

        assert len(top_errors) == 2
        # Mayor confianza primero
        assert top_errors[0].confidence > top_errors[1].confidence

    def test_save_reports_creates_files(self, tmp_path):
        """save_reports debe crear archivos JSON y CSV."""
        from src_v2.visualization.error_analysis import ErrorAnalyzer

        analyzer = ErrorAnalyzer(['COVID', 'Normal', 'Viral_Pneumonia'])
        analyzer.add_prediction(torch.tensor([1.0, 10.0, 1.0]), 0, '/error.png')

        saved_files = analyzer.save_reports(tmp_path)

        assert 'json_summary' in saved_files
        assert 'csv_details' in saved_files
        assert 'confusion_analysis' in saved_files
        assert (tmp_path / 'error_summary.json').exists()
        assert (tmp_path / 'error_details.csv').exists()
        assert (tmp_path / 'confusion_analysis.json').exists()

    def test_confusion_matrix_correct(self):
        """confusion_matrix debe actualizarse correctamente."""
        from src_v2.visualization.error_analysis import ErrorAnalyzer

        analyzer = ErrorAnalyzer(['A', 'B', 'C'])

        # 2 predicciones A->A (correctas)
        analyzer.add_prediction(torch.tensor([10.0, 1.0, 1.0]), 0, '/a1.png')
        analyzer.add_prediction(torch.tensor([10.0, 1.0, 1.0]), 0, '/a2.png')
        # 1 prediccion B->B (correcta)
        analyzer.add_prediction(torch.tensor([1.0, 10.0, 1.0]), 1, '/b1.png')
        # 1 prediccion A->B (error)
        analyzer.add_prediction(torch.tensor([1.0, 10.0, 1.0]), 0, '/a3.png')

        summary = analyzer.get_summary()
        cm = np.array(summary.confusion_matrix)

        assert cm[0, 0] == 2  # A->A
        assert cm[1, 1] == 1  # B->B
        assert cm[0, 1] == 1  # A->B (error)


class TestPFSAnalysisModule:
    """Tests para el modulo pfs_analysis."""

    def test_pfs_result_creation(self):
        """PFSResult debe crearse correctamente."""
        from src_v2.visualization.pfs_analysis import PFSResult

        result = PFSResult(
            image_path="/path/to/image.png",
            true_class="COVID",
            predicted_class="COVID",
            confidence=0.95,
            pfs=0.78,
            correct=True,
        )

        assert result.image_path == "/path/to/image.png"
        assert result.true_class == "COVID"
        assert result.pfs == 0.78
        assert result.correct is True

    def test_pfs_result_to_dict(self):
        """PFSResult.to_dict debe retornar diccionario correcto."""
        from src_v2.visualization.pfs_analysis import PFSResult

        result = PFSResult(
            image_path="/test.png",
            true_class="Normal",
            predicted_class="COVID",
            confidence=0.8,
            pfs=0.45,
            correct=False,
        )

        d = result.to_dict()

        assert isinstance(d, dict)
        assert d["image_path"] == "/test.png"
        assert d["pfs"] == 0.45
        assert d["correct"] is False

    def test_pfs_analyzer_initialization(self):
        """PFSAnalyzer debe inicializarse correctamente."""
        from src_v2.visualization.pfs_analysis import PFSAnalyzer

        class_names = ['COVID', 'Normal', 'Viral_Pneumonia']
        analyzer = PFSAnalyzer(class_names, threshold=0.5)

        assert analyzer.class_names == class_names
        assert analyzer.threshold == 0.5
        assert len(analyzer.results) == 0

    def test_pfs_analyzer_empty_class_names_raises(self):
        """PFSAnalyzer debe fallar con class_names vacio."""
        from src_v2.visualization.pfs_analysis import PFSAnalyzer

        with pytest.raises(ValueError):
            PFSAnalyzer([])

    def test_pfs_analyzer_invalid_threshold_raises(self):
        """PFSAnalyzer debe fallar con threshold invalido."""
        from src_v2.visualization.pfs_analysis import PFSAnalyzer

        with pytest.raises(ValueError):
            PFSAnalyzer(['A', 'B'], threshold=1.5)

        with pytest.raises(ValueError):
            PFSAnalyzer(['A', 'B'], threshold=-0.1)

    def test_pfs_analyzer_add_result(self):
        """PFSAnalyzer.add_result debe agregar resultados."""
        from src_v2.visualization.pfs_analysis import PFSAnalyzer, PFSResult

        analyzer = PFSAnalyzer(['COVID', 'Normal', 'Viral_Pneumonia'])

        result = PFSResult(
            image_path="/test.png",
            true_class="COVID",
            predicted_class="COVID",
            confidence=0.9,
            pfs=0.75,
            correct=True,
        )
        analyzer.add_result(result)

        assert len(analyzer.results) == 1
        assert analyzer.results[0].pfs == 0.75

    def test_pfs_analyzer_get_summary(self):
        """PFSAnalyzer.get_summary debe calcular estadisticas correctas."""
        from src_v2.visualization.pfs_analysis import PFSAnalyzer, PFSResult

        analyzer = PFSAnalyzer(['A', 'B'], threshold=0.5)

        # Agregar resultados
        analyzer.add_result(PFSResult("/a1.png", "A", "A", 0.9, 0.7, True))
        analyzer.add_result(PFSResult("/a2.png", "A", "A", 0.85, 0.6, True))
        analyzer.add_result(PFSResult("/b1.png", "B", "B", 0.95, 0.8, True))
        analyzer.add_result(PFSResult("/b2.png", "B", "A", 0.6, 0.4, False))  # Low PFS

        summary = analyzer.get_summary()

        assert summary.total_samples == 4
        assert summary.mean_pfs == pytest.approx(0.625, rel=0.01)
        assert summary.low_pfs_count == 1  # Solo el de 0.4
        assert summary.low_pfs_rate == pytest.approx(0.25, rel=0.01)

    def test_pfs_analyzer_get_summary_no_results_raises(self):
        """PFSAnalyzer.get_summary debe fallar sin resultados."""
        from src_v2.visualization.pfs_analysis import PFSAnalyzer

        analyzer = PFSAnalyzer(['A', 'B'])

        with pytest.raises(ValueError):
            analyzer.get_summary()

    def test_pfs_analyzer_get_low_pfs_results(self):
        """PFSAnalyzer.get_low_pfs_results debe filtrar correctamente."""
        from src_v2.visualization.pfs_analysis import PFSAnalyzer, PFSResult

        analyzer = PFSAnalyzer(['A', 'B'], threshold=0.5)

        analyzer.add_result(PFSResult("/a1.png", "A", "A", 0.9, 0.7, True))
        analyzer.add_result(PFSResult("/a2.png", "A", "A", 0.85, 0.3, True))
        analyzer.add_result(PFSResult("/b1.png", "B", "B", 0.95, 0.45, True))

        low_pfs = analyzer.get_low_pfs_results()

        assert len(low_pfs) == 2  # 0.3 y 0.45 < 0.5
        assert all(r.pfs < 0.5 for r in low_pfs)

    def test_pfs_analyzer_save_reports(self, tmp_path):
        """PFSAnalyzer.save_reports debe crear archivos."""
        from src_v2.visualization.pfs_analysis import PFSAnalyzer, PFSResult

        analyzer = PFSAnalyzer(['A', 'B'], threshold=0.5)
        analyzer.add_result(PFSResult("/test.png", "A", "A", 0.9, 0.7, True))

        saved_files = analyzer.save_reports(tmp_path)

        assert 'summary' in saved_files
        assert 'details' in saved_files
        assert 'by_class' in saved_files
        assert (tmp_path / 'pfs_summary.json').exists()
        assert (tmp_path / 'pfs_details.csv').exists()
        assert (tmp_path / 'pfs_by_class.csv').exists()

    def test_generate_approximate_mask(self):
        """generate_approximate_mask debe generar mascara rectangular."""
        from src_v2.visualization.pfs_analysis import generate_approximate_mask

        mask = generate_approximate_mask((100, 100), margin=0.2)

        assert mask.shape == (100, 100)
        assert mask.dtype == np.float32
        assert mask.min() == 0.0
        assert mask.max() == 1.0

        # Verificar que bordes son 0
        assert mask[0, 0] == 0  # Esquina superior izquierda
        assert mask[99, 99] == 0  # Esquina inferior derecha

        # Verificar que centro es 1
        assert mask[50, 50] == 1

    def test_generate_approximate_mask_invalid_margin_raises(self):
        """generate_approximate_mask debe fallar con margin invalido."""
        from src_v2.visualization.pfs_analysis import generate_approximate_mask

        with pytest.raises(ValueError):
            generate_approximate_mask((100, 100), margin=0.6)

        with pytest.raises(ValueError):
            generate_approximate_mask((100, 100), margin=-0.1)

    def test_load_lung_mask(self, tmp_path):
        """load_lung_mask debe cargar y normalizar mascaras."""
        from src_v2.visualization.pfs_analysis import load_lung_mask
        from PIL import Image

        # Crear mascara de prueba
        mask_array = np.zeros((100, 100), dtype=np.uint8)
        mask_array[25:75, 25:75] = 255  # Region pulmonar

        mask_path = tmp_path / "test_mask.png"
        Image.fromarray(mask_array).save(mask_path)

        # Cargar
        loaded_mask = load_lung_mask(mask_path)

        assert loaded_mask.shape == (100, 100)
        assert loaded_mask.dtype == np.float32
        assert loaded_mask.min() == pytest.approx(0.0, abs=0.01)
        assert loaded_mask.max() == pytest.approx(1.0, abs=0.01)

    def test_find_mask_for_image_not_found(self, tmp_path):
        """find_mask_for_image debe retornar None si no encuentra."""
        from src_v2.visualization.pfs_analysis import find_mask_for_image

        result = find_mask_for_image(
            tmp_path / "image.png",
            tmp_path / "masks",
            "COVID"
        )

        assert result is None

    def test_pfs_summary_by_class(self):
        """PFSSummary debe calcular PFS por clase correctamente."""
        from src_v2.visualization.pfs_analysis import PFSAnalyzer, PFSResult

        analyzer = PFSAnalyzer(['COVID', 'Normal'], threshold=0.5)

        # COVID: PFS promedio 0.7
        analyzer.add_result(PFSResult("/c1.png", "COVID", "COVID", 0.9, 0.8, True))
        analyzer.add_result(PFSResult("/c2.png", "COVID", "COVID", 0.9, 0.6, True))

        # Normal: PFS promedio 0.5
        analyzer.add_result(PFSResult("/n1.png", "Normal", "Normal", 0.9, 0.4, True))
        analyzer.add_result(PFSResult("/n2.png", "Normal", "Normal", 0.9, 0.6, True))

        summary = analyzer.get_summary()

        assert summary.pfs_by_class["COVID"]["mean"] == pytest.approx(0.7, rel=0.01)
        assert summary.pfs_by_class["Normal"]["mean"] == pytest.approx(0.5, rel=0.01)

    def test_pfs_summary_correct_vs_incorrect(self):
        """PFSSummary debe calcular PFS para correctos vs incorrectos."""
        from src_v2.visualization.pfs_analysis import PFSAnalyzer, PFSResult

        analyzer = PFSAnalyzer(['A', 'B'], threshold=0.5)

        # Correctas: PFS alto
        analyzer.add_result(PFSResult("/c1.png", "A", "A", 0.9, 0.8, True))
        analyzer.add_result(PFSResult("/c2.png", "B", "B", 0.9, 0.7, True))

        # Incorrectas: PFS bajo
        analyzer.add_result(PFSResult("/e1.png", "A", "B", 0.6, 0.3, False))
        analyzer.add_result(PFSResult("/e2.png", "B", "A", 0.5, 0.4, False))

        summary = analyzer.get_summary()

        assert summary.pfs_correct_vs_incorrect["correct"]["mean"] == pytest.approx(0.75, rel=0.01)
        assert summary.pfs_correct_vs_incorrect["incorrect"]["mean"] == pytest.approx(0.35, rel=0.01)
