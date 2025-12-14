"""
Tests específicos para el clasificador warped_96 (RECOMENDADO).

Este clasificador es el punto óptimo identificado en la Sesión 53:
- Accuracy: 99.10% (mejor que warped_99)
- Robustez JPEG Q50: 3.06% degradación (2.4x mejor que warped_99)
- Fill rate: 96% (medición honesta con Grayscale+CLAHE)

Los tests verifican que:
1. El clasificador y dataset existen
2. Los resultados coinciden con GROUND_TRUTH.json v2.1.0
3. Las métricas cumplen los umbrales mínimos requeridos
"""

import json
import pytest
from pathlib import Path


# Paths del clasificador warped_96
CLASSIFIER_PATH = Path("outputs/classifier_replication_v2/best_classifier.pt")
DATASET_PATH = Path("outputs/warped_replication_v2")
RESULTS_PATH = Path("outputs/classifier_replication_v2/results.json")
ROBUSTNESS_PATH = Path("outputs/classifier_replication_v2/robustness_results.json")
GROUND_TRUTH_PATH = Path("GROUND_TRUTH.json")


class TestWarped96Existence:
    """Verificar que los artefactos del clasificador warped_96 existen."""

    def test_classifier_checkpoint_exists(self):
        """El checkpoint del clasificador recomendado debe existir."""
        assert CLASSIFIER_PATH.exists(), (
            f"Clasificador warped_96 no encontrado en {CLASSIFIER_PATH}. "
            "Ejecutar: python -m src_v2 train-classifier outputs/warped_replication_v2"
        )

    def test_dataset_exists(self):
        """El dataset warped_96 debe existir."""
        assert DATASET_PATH.exists(), (
            f"Dataset warped_96 no encontrado en {DATASET_PATH}. "
            "Ejecutar: python -m src_v2 generate-dataset"
        )

    def test_dataset_has_splits(self):
        """El dataset debe tener train/val/test splits."""
        if not DATASET_PATH.exists():
            pytest.skip("Dataset no existe")

        for split in ["train", "val", "test"]:
            split_path = DATASET_PATH / split
            assert split_path.exists(), f"Split {split} no encontrado en {DATASET_PATH}"

    def test_results_json_exists(self):
        """Los resultados de evaluación deben existir."""
        assert RESULTS_PATH.exists(), (
            f"Resultados no encontrados en {RESULTS_PATH}. "
            "Ejecutar: python -m src_v2 evaluate-classifier"
        )

    def test_robustness_results_exist(self):
        """Los resultados de robustez deben existir."""
        assert ROBUSTNESS_PATH.exists(), (
            f"Resultados de robustez no encontrados en {ROBUSTNESS_PATH}. "
            "Ejecutar: python -m src_v2 test-robustness"
        )


class TestWarped96Accuracy:
    """Verificar accuracy del clasificador warped_96."""

    @pytest.fixture
    def results(self):
        """Cargar resultados de evaluación."""
        if not RESULTS_PATH.exists():
            pytest.skip("Resultados no disponibles")
        with open(RESULTS_PATH) as f:
            return json.load(f)

    @pytest.fixture
    def ground_truth(self):
        """Cargar GROUND_TRUTH.json."""
        if not GROUND_TRUTH_PATH.exists():
            pytest.skip("GROUND_TRUTH.json no encontrado")
        with open(GROUND_TRUTH_PATH) as f:
            return json.load(f)

    def test_accuracy_meets_minimum(self, results):
        """Accuracy debe ser >= 99.0%."""
        accuracy = results["test_metrics"]["accuracy"]
        assert accuracy >= 0.99, (
            f"Accuracy {accuracy:.2%} es menor que el mínimo requerido (99.0%)"
        )

    def test_accuracy_matches_ground_truth(self, results, ground_truth):
        """Accuracy debe coincidir con GROUND_TRUTH.json (tolerancia 0.5%)."""
        actual = results["test_metrics"]["accuracy"] * 100
        expected = ground_truth["classification"]["datasets"]["warped_96"]["accuracy"]
        tolerance = ground_truth["tolerances"]["classification_accuracy"]["absolute_percent"]

        assert abs(actual - expected) <= tolerance, (
            f"Accuracy {actual:.2f}% difiere de GROUND_TRUTH {expected:.2f}% "
            f"más que la tolerancia permitida ({tolerance}%)"
        )

    def test_f1_macro_meets_minimum(self, results):
        """F1-macro debe ser >= 98.0%."""
        f1 = results["test_metrics"]["f1_macro"]
        assert f1 >= 0.98, (
            f"F1-macro {f1:.2%} es menor que el mínimo requerido (98.0%)"
        )

    def test_per_class_f1_above_threshold(self, results):
        """F1 por clase debe ser >= 96% para todas las clases."""
        for class_name, metrics in results["per_class_metrics"].items():
            if isinstance(metrics, dict) and "f1-score" in metrics:
                f1 = metrics["f1-score"]
                assert f1 >= 0.96, (
                    f"F1-score para {class_name} ({f1:.2%}) es menor que 96%"
                )


class TestWarped96Robustness:
    """Verificar robustez del clasificador warped_96."""

    @pytest.fixture
    def robustness(self):
        """Cargar resultados de robustez."""
        if not ROBUSTNESS_PATH.exists():
            pytest.skip("Resultados de robustez no disponibles")
        with open(ROBUSTNESS_PATH) as f:
            return json.load(f)

    @pytest.fixture
    def ground_truth(self):
        """Cargar GROUND_TRUTH.json."""
        if not GROUND_TRUTH_PATH.exists():
            pytest.skip("GROUND_TRUTH.json no encontrado")
        with open(GROUND_TRUTH_PATH) as f:
            return json.load(f)

    def test_jpeg_q50_degradation_below_maximum(self, robustness):
        """Degradación JPEG Q50 debe ser <= 4.0%."""
        degradation = robustness["perturbations"]["jpeg_q50"]["degradation"]
        assert degradation <= 4.0, (
            f"Degradación JPEG Q50 {degradation:.2f}% excede el máximo permitido (4.0%)"
        )

    def test_jpeg_q50_matches_ground_truth(self, robustness, ground_truth):
        """Degradación JPEG Q50 debe coincidir con GROUND_TRUTH.json."""
        actual = robustness["perturbations"]["jpeg_q50"]["degradation"]
        expected = ground_truth["robustness"]["jpeg_q50"]["warped_96"]
        tolerance = ground_truth["tolerances"]["robustness_degradation"]["absolute_percent"]

        assert abs(actual - expected) <= tolerance, (
            f"JPEG Q50 degradation {actual:.2f}% difiere de GROUND_TRUTH {expected:.2f}% "
            f"más que la tolerancia permitida ({tolerance}%)"
        )

    def test_jpeg_q30_degradation_below_maximum(self, robustness):
        """Degradación JPEG Q30 debe ser <= 6.0%."""
        degradation = robustness["perturbations"]["jpeg_q30"]["degradation"]
        assert degradation <= 6.0, (
            f"Degradación JPEG Q30 {degradation:.2f}% excede el máximo permitido (6.0%)"
        )

    def test_blur_sigma1_degradation_below_maximum(self, robustness):
        """Degradación blur sigma=1 debe ser <= 3.0%."""
        degradation = robustness["perturbations"]["blur_sigma1"]["degradation"]
        assert degradation <= 3.0, (
            f"Degradación blur sigma=1 {degradation:.2f}% excede el máximo permitido (3.0%)"
        )

    def test_better_than_warped_99_jpeg_q50(self, robustness, ground_truth):
        """warped_96 debe ser más robusto que warped_99 en JPEG Q50."""
        warped_96_deg = robustness["perturbations"]["jpeg_q50"]["degradation"]
        warped_99_deg = ground_truth["robustness"]["jpeg_q50"]["warped_99"]

        assert warped_96_deg < warped_99_deg, (
            f"warped_96 ({warped_96_deg:.2f}%) debe ser más robusto que "
            f"warped_99 ({warped_99_deg:.2f}%) en JPEG Q50"
        )


class TestWarped96GroundTruthConsistency:
    """Verificar consistencia con GROUND_TRUTH.json v2.1.0."""

    @pytest.fixture
    def ground_truth(self):
        """Cargar GROUND_TRUTH.json."""
        if not GROUND_TRUTH_PATH.exists():
            pytest.skip("GROUND_TRUTH.json no encontrado")
        with open(GROUND_TRUTH_PATH) as f:
            return json.load(f)

    def test_ground_truth_version(self, ground_truth):
        """GROUND_TRUTH.json debe ser v2.1.0 o superior."""
        version = ground_truth["_metadata"]["version"]
        major, minor, patch = map(int, version.split("."))

        assert (major, minor) >= (2, 1), (
            f"GROUND_TRUTH.json version {version} es anterior a v2.1.0 requerida"
        )

    def test_warped_96_is_recommended(self, ground_truth):
        """warped_96 debe estar marcado como RECOMMENDED."""
        warped_96 = ground_truth["classification"]["datasets"]["warped_96"]
        assert "RECOMMENDED" in warped_96.get("note", ""), (
            "warped_96 debe estar marcado como RECOMMENDED en GROUND_TRUTH.json"
        )

    def test_fill_rate_tradeoff_section_exists(self, ground_truth):
        """La sección fill_rate_tradeoff debe existir."""
        assert "fill_rate_tradeoff" in ground_truth, (
            "Sección fill_rate_tradeoff no encontrada en GROUND_TRUTH.json"
        )

    def test_warped_96_is_optimal_point(self, ground_truth):
        """warped_96 debe ser el punto óptimo en fill_rate_tradeoff."""
        optimal = ground_truth["fill_rate_tradeoff"]["optimal_point"]
        assert optimal == "warped_96", (
            f"Punto óptimo es {optimal}, se esperaba warped_96"
        )

    def test_session_53_in_validated_sessions(self, ground_truth):
        """Sesión 53 debe estar en las sesiones validadas."""
        validated = ground_truth["_metadata"]["validated_sessions"]
        assert 53 in validated, (
            "Sesión 53 (fill rate trade-off) no está en validated_sessions"
        )


class TestWarped96ModelLoadable:
    """Verificar que el modelo se puede cargar correctamente."""

    @pytest.mark.skipif(
        not CLASSIFIER_PATH.exists(),
        reason="Checkpoint no disponible"
    )
    def test_checkpoint_loadable(self):
        """El checkpoint debe ser cargable con torch.load."""
        import torch

        checkpoint = torch.load(CLASSIFIER_PATH, map_location="cpu", weights_only=False)

        assert "model_state_dict" in checkpoint, "Checkpoint debe tener model_state_dict"
        assert "class_names" in checkpoint, "Checkpoint debe tener class_names"

    @pytest.mark.skipif(
        not CLASSIFIER_PATH.exists(),
        reason="Checkpoint no disponible"
    )
    def test_model_has_correct_classes(self):
        """El modelo debe tener las 3 clases COVID-19."""
        import torch

        checkpoint = torch.load(CLASSIFIER_PATH, map_location="cpu", weights_only=False)
        class_names = checkpoint["class_names"]

        expected = ["COVID", "Normal", "Viral_Pneumonia"]
        assert class_names == expected, (
            f"Clases {class_names} no coinciden con esperadas {expected}"
        )

    @pytest.mark.skipif(
        not CLASSIFIER_PATH.exists(),
        reason="Checkpoint no disponible"
    )
    def test_model_inference_works(self):
        """El modelo debe poder hacer inferencia."""
        import torch
        from src_v2.models.classifier import create_classifier

        model = create_classifier(checkpoint=str(CLASSIFIER_PATH))
        model.eval()

        # Input de prueba
        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 3), f"Output shape {output.shape} incorrecto"
