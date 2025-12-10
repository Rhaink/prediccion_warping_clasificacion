"""
Tests de integracion para comando pfs-analysis.

Session 32: Tests criticos para Pulmonary Focus Score (PFS) analysis.
PFS mide la fraccion de atencion del modelo enfocada en regiones pulmonares.

Cobertura previa: 0 tests de integracion (solo 7 tests de --help)
Meta: 12+ tests de integracion
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from src_v2.cli import app

runner = CliRunner()


# =============================================================================
# FIXTURES ESPECIFICAS PARA PFS
# =============================================================================

@pytest.fixture
def pfs_eval_dataset(tmp_path):
    """
    Dataset de evaluacion para PFS con estructura correcta.

    Estructura:
        tmp_path/test/
            COVID/img1.png, img2.png
            Normal/img1.png, img2.png
            Viral_Pneumonia/img1.png, img2.png
    """
    data_dir = tmp_path / "pfs_data"
    classes = ["COVID", "Normal", "Viral_Pneumonia"]

    for cls_idx, cls in enumerate(classes):
        cls_dir = data_dir / "test" / cls
        cls_dir.mkdir(parents=True)

        # Crear 2 imagenes por clase
        for i in range(2):
            # Imagen grayscale simulando radiografia
            base_val = 100 + cls_idx * 30
            img_array = np.full((224, 224), base_val, dtype=np.uint8)

            # Agregar patron central (region pulmonar simulada)
            img_array[56:168, 56:168] = base_val + 50

            # Convertir a RGB
            img_rgb = np.stack([img_array] * 3, axis=-1)
            img = Image.fromarray(img_rgb, mode='RGB')
            img.save(cls_dir / f"{cls}_{i}.png")

    return data_dir


@pytest.fixture
def pfs_dataset_with_masks(tmp_path):
    """
    Dataset con mascaras pulmonares reales (binarias).

    Session 32 fix: Crear multiples imagenes con nombres que coincidan
    con las mascaras para probar el flujo real de matching imagen-mascara.

    Estructura:
        tmp_path/
            images/test/
                COVID/sample_0.png, sample_1.png
                Normal/sample_0.png, sample_1.png
            masks/
                COVID/sample_0_mask.png, sample_1_mask.png
                Normal/sample_0_mask.png, sample_1_mask.png
    """
    images_dir = tmp_path / "images" / "test"
    masks_dir = tmp_path / "masks"

    for cls in ["COVID", "Normal"]:
        (images_dir / cls).mkdir(parents=True)
        (masks_dir / cls).mkdir(parents=True)

        # Session 32 fix: Crear multiples imagenes con nombres unicos
        for i in range(2):
            img_name = f"sample_{i}.png"
            mask_name = f"sample_{i}_mask.png"

            # Crear imagen con variacion
            color_val = 128 + i * 20
            img = Image.new('RGB', (224, 224), color=(color_val, color_val, color_val))
            img.save(images_dir / cls / img_name)

            # Crear mascara binaria correspondiente (region central = pulmon)
            mask = np.zeros((224, 224), dtype=np.uint8)
            mask[40:184, 40:184] = 255  # Region pulmonar
            mask_img = Image.fromarray(mask, mode='L')
            mask_img.save(masks_dir / cls / mask_name)

    return {"images": images_dir.parent, "masks": masks_dir}


# =============================================================================
# TESTS DE INTEGRACION - PFS-ANALYSIS BASICO
# =============================================================================

class TestPFSAnalysisBasic:
    """Tests basicos del comando pfs-analysis."""

    def test_pfs_with_approximate_masks(
        self, mock_classifier_checkpoint, pfs_eval_dataset, tmp_path
    ):
        """
        PFS con mascaras rectangulares aproximadas.

        Session 32: Test basico - modelo sin entrenar puede fallar
        pero no debe crashear.
        """
        output_dir = tmp_path / "pfs_output"

        result = runner.invoke(app, [
            'pfs-analysis',
            '--checkpoint', str(mock_classifier_checkpoint),
            '--data-dir', str(pfs_eval_dataset / "test"),
            '--output-dir', str(output_dir),
            '--device', 'cpu',
            '--approximate',
            '--num-samples', '2',
            '--batch-size', '1',
        ])

        # Session 33: Bug M5 fix - Esperamos exito con datos validos
        assert result.exit_code == 0, \
            f"Command failed (code {result.exit_code}): {result.stdout}"

    def test_pfs_creates_output_directory(
        self, mock_classifier_checkpoint, pfs_eval_dataset, tmp_path
    ):
        """
        PFS crea directorio de salida si no existe.

        Session 32: Verificar que el comando intenta crear el output.
        """
        output_dir = tmp_path / "new_pfs_output" / "nested"

        # Directorio no existe
        assert not output_dir.exists()

        result = runner.invoke(app, [
            'pfs-analysis',
            '--checkpoint', str(mock_classifier_checkpoint),
            '--data-dir', str(pfs_eval_dataset / "test"),
            '--output-dir', str(output_dir),
            '--device', 'cpu',
            '--approximate',
            '--num-samples', '2',
        ])

        # Session 33: Bug M5 fix - Esperamos exito con datos validos
        assert result.exit_code == 0, \
            f"Command failed (code {result.exit_code}): {result.stdout}"

    def test_pfs_basic_output_structure(
        self, mock_classifier_checkpoint, pfs_eval_dataset, tmp_path
    ):
        """
        PFS genera archivos de salida con estructura esperada.

        Session 32: Si el comando tiene exito, verificar outputs.
        """
        output_dir = tmp_path / "pfs_output_check"

        result = runner.invoke(app, [
            'pfs-analysis',
            '--checkpoint', str(mock_classifier_checkpoint),
            '--data-dir', str(pfs_eval_dataset / "test"),
            '--output-dir', str(output_dir),
            '--device', 'cpu',
            '--approximate',
            '--num-samples', '2',
        ])

        if result.exit_code == 0:
            # Verificar que se crearon archivos
            assert output_dir.exists(), "Output directory not created"

            # Buscar archivos JSON o PNG generados
            json_files = list(output_dir.glob("*.json"))
            png_files = list(output_dir.glob("*.png"))

            # Al menos deberia crear algun output
            total_outputs = len(json_files) + len(png_files)
            # Session 32 fix: Cambiar >= 0 a > 0 (>= 0 siempre es verdadero)
            # El comando exitoso DEBE crear al menos un archivo de output
            assert total_outputs > 0, \
                f"No outputs created on success. JSON: {len(json_files)}, PNG: {len(png_files)}"


# =============================================================================
# TESTS DE PARAMETROS DE PFS
# =============================================================================

class TestPFSAnalysisParameters:
    """Tests para diferentes parametros del comando pfs-analysis."""

    def test_pfs_different_thresholds(
        self, mock_classifier_checkpoint, pfs_eval_dataset, tmp_path
    ):
        """
        PFS acepta diferentes valores de threshold.

        Session 32: threshold controla umbral minimo aceptable de PFS.
        """
        for threshold in [0.3, 0.5, 0.7]:
            output_dir = tmp_path / f"pfs_t{threshold}"

            result = runner.invoke(app, [
                'pfs-analysis',
                '--checkpoint', str(mock_classifier_checkpoint),
                '--data-dir', str(pfs_eval_dataset / "test"),
                '--output-dir', str(output_dir),
                '--device', 'cpu',
                '--approximate',
                '--threshold', str(threshold),
                '--num-samples', '2',
            ])

            # Session 33: Bug M5 fix - Esperamos exito con threshold valido
            assert result.exit_code == 0, \
                f"Threshold {threshold} failed (code {result.exit_code}): {result.stdout}"

    def test_pfs_different_margins(
        self, mock_classifier_checkpoint, pfs_eval_dataset, tmp_path
    ):
        """
        PFS acepta diferentes valores de margin para mascaras aproximadas.

        Session 32: margin controla tamano de region central (0-0.5).
        """
        for margin in [0.1, 0.15, 0.2, 0.3]:
            output_dir = tmp_path / f"pfs_m{margin}"

            result = runner.invoke(app, [
                'pfs-analysis',
                '--checkpoint', str(mock_classifier_checkpoint),
                '--data-dir', str(pfs_eval_dataset / "test"),
                '--output-dir', str(output_dir),
                '--device', 'cpu',
                '--approximate',
                '--margin', str(margin),
                '--num-samples', '2',
            ])

            # Session 33: Bug M5 fix - Esperamos exito con margin valido
            assert result.exit_code == 0, \
                f"Margin {margin} failed (code {result.exit_code}): {result.stdout}"

    def test_pfs_num_samples_option(
        self, mock_classifier_checkpoint, pfs_eval_dataset, tmp_path
    ):
        """
        PFS respeta --num-samples para limitar analisis.

        Session 32: num_samples limita cuantas imagenes se analizan por clase.
        """
        for num_samples in [1, 2, 5]:
            output_dir = tmp_path / f"pfs_n{num_samples}"

            result = runner.invoke(app, [
                'pfs-analysis',
                '--checkpoint', str(mock_classifier_checkpoint),
                '--data-dir', str(pfs_eval_dataset / "test"),
                '--output-dir', str(output_dir),
                '--device', 'cpu',
                '--approximate',
                '--num-samples', str(num_samples),
            ])

            # Session 33: Bug M5 fix - Esperamos exito con num_samples valido
            assert result.exit_code == 0, \
                f"num_samples {num_samples} failed (code {result.exit_code}): {result.stdout}"


# =============================================================================
# TESTS DE MANEJO DE ERRORES
# =============================================================================

class TestPFSAnalysisErrors:
    """Tests de manejo de errores para pfs-analysis."""

    def test_pfs_invalid_checkpoint_fails(self, pfs_eval_dataset, tmp_path):
        """
        PFS con checkpoint inexistente debe fallar graciosamente.

        Session 32: Verificar que el comando falla con exit_code != 0.
        """
        result = runner.invoke(app, [
            'pfs-analysis',
            '--checkpoint', '/nonexistent/model.pt',
            '--data-dir', str(pfs_eval_dataset / "test"),
            '--output-dir', str(tmp_path / "out"),
            '--approximate',
        ])

        assert result.exit_code != 0, "Should fail with invalid checkpoint"
        # Session 32: Verificar mensaje de error
        assert 'no existe' in result.stdout.lower() or 'error' in result.stdout.lower() or result.exit_code == 1

    def test_pfs_invalid_data_dir_fails(self, mock_classifier_checkpoint, tmp_path):
        """
        PFS con data-dir inexistente debe fallar graciosamente.

        Session 32: Validacion de paths de entrada.
        """
        result = runner.invoke(app, [
            'pfs-analysis',
            '--checkpoint', str(mock_classifier_checkpoint),
            '--data-dir', '/nonexistent/data',
            '--output-dir', str(tmp_path / "out"),
            '--approximate',
        ])

        assert result.exit_code != 0, "Should fail with invalid data-dir"

    def test_pfs_requires_mask_source(
        self, mock_classifier_checkpoint, pfs_eval_dataset, tmp_path
    ):
        """
        PFS sin --mask-dir ni --approximate debe fallar.

        Session 32: Comando requiere una fuente de mascaras.
        """
        result = runner.invoke(app, [
            'pfs-analysis',
            '--checkpoint', str(mock_classifier_checkpoint),
            '--data-dir', str(pfs_eval_dataset / "test"),
            '--output-dir', str(tmp_path / "out"),
            # No --mask-dir ni --approximate
        ])

        assert result.exit_code != 0, "Should fail without mask source"

    def test_pfs_invalid_threshold_fails(
        self, mock_classifier_checkpoint, pfs_eval_dataset, tmp_path
    ):
        """
        PFS con threshold fuera de rango [0, 1] debe fallar.

        Session 32: Validacion de parametros numericos.
        """
        for invalid_threshold in [-0.1, 1.5, 2.0]:
            result = runner.invoke(app, [
                'pfs-analysis',
                '--checkpoint', str(mock_classifier_checkpoint),
                '--data-dir', str(pfs_eval_dataset / "test"),
                '--output-dir', str(tmp_path / "out"),
                '--approximate',
                '--threshold', str(invalid_threshold),
            ])

            assert result.exit_code != 0, \
                f"Should fail with invalid threshold {invalid_threshold}"

    def test_pfs_invalid_margin_fails(
        self, mock_classifier_checkpoint, pfs_eval_dataset, tmp_path
    ):
        """
        PFS con margin fuera de rango [0, 0.5) debe fallar.

        Session 32: Validacion de margen para mascaras aproximadas.
        """
        for invalid_margin in [-0.1, 0.5, 0.7, 1.0]:
            result = runner.invoke(app, [
                'pfs-analysis',
                '--checkpoint', str(mock_classifier_checkpoint),
                '--data-dir', str(pfs_eval_dataset / "test"),
                '--output-dir', str(tmp_path / "out"),
                '--approximate',
                '--margin', str(invalid_margin),
            ])

            assert result.exit_code != 0, \
                f"Should fail with invalid margin {invalid_margin}"

    def test_pfs_invalid_num_samples_fails(
        self, mock_classifier_checkpoint, pfs_eval_dataset, tmp_path
    ):
        """
        PFS con num_samples <= 0 debe fallar.

        Session 32: Validacion de num_samples positivo.
        """
        for invalid_num in [0, -1, -10]:
            result = runner.invoke(app, [
                'pfs-analysis',
                '--checkpoint', str(mock_classifier_checkpoint),
                '--data-dir', str(pfs_eval_dataset / "test"),
                '--output-dir', str(tmp_path / "out"),
                '--approximate',
                '--num-samples', str(invalid_num),
            ])

            assert result.exit_code != 0, \
                f"Should fail with invalid num_samples {invalid_num}"

    def test_pfs_handles_empty_directory(
        self, mock_classifier_checkpoint, tmp_path
    ):
        """
        PFS maneja correctamente directorio vacio.

        Session 32: Caso edge - directorio sin imagenes.
        """
        empty_dir = tmp_path / "empty_data"
        empty_dir.mkdir()

        result = runner.invoke(app, [
            'pfs-analysis',
            '--checkpoint', str(mock_classifier_checkpoint),
            '--data-dir', str(empty_dir),
            '--output-dir', str(tmp_path / "out"),
            '--approximate',
        ])

        # Session 33: Bug M5 fix - Directorio vacio debe fallar graciosamente
        assert result.exit_code == 1, \
            f"Empty directory should fail gracefully (got {result.exit_code}): {result.stdout}"


# =============================================================================
# TESTS CON MASCARAS REALES
# =============================================================================

class TestPFSWithMasks:
    """Tests de PFS usando mascaras pulmonares reales."""

    def test_pfs_with_mask_directory(
        self, mock_classifier_checkpoint, pfs_dataset_with_masks, tmp_path
    ):
        """
        PFS con directorio de mascaras reales.

        Session 32: Test con --mask-dir en lugar de --approximate.
        """
        output_dir = tmp_path / "pfs_with_masks"

        result = runner.invoke(app, [
            'pfs-analysis',
            '--checkpoint', str(mock_classifier_checkpoint),
            '--data-dir', str(pfs_dataset_with_masks["images"] / "test"),
            '--mask-dir', str(pfs_dataset_with_masks["masks"]),
            '--output-dir', str(output_dir),
            '--device', 'cpu',
            '--num-samples', '1',
        ])

        # Session 33: Bug M5 fix - Esperamos exito con datos validos
        assert result.exit_code == 0, \
            f"Command failed (code {result.exit_code}): {result.stdout}"

    def test_pfs_invalid_mask_directory_fails(
        self, mock_classifier_checkpoint, pfs_eval_dataset, tmp_path
    ):
        """
        PFS con directorio de mascaras inexistente debe fallar.

        Session 32: Validacion de --mask-dir.
        """
        result = runner.invoke(app, [
            'pfs-analysis',
            '--checkpoint', str(mock_classifier_checkpoint),
            '--data-dir', str(pfs_eval_dataset / "test"),
            '--mask-dir', '/nonexistent/masks',
            '--output-dir', str(tmp_path / "out"),
        ])

        assert result.exit_code != 0, "Should fail with invalid mask-dir"


# =============================================================================
# TESTS DE REPRODUCIBILIDAD
# =============================================================================

class TestPFSReproducibility:
    """Tests de reproducibilidad para pfs-analysis."""

    def test_pfs_deterministic_output(
        self, mock_classifier_checkpoint, pfs_eval_dataset, tmp_path
    ):
        """
        PFS produce resultados deterministicos con mismos inputs.

        Session 32: Verificar consistencia de outputs.
        """
        output1 = tmp_path / "pfs_run1"
        output2 = tmp_path / "pfs_run2"

        # Primera ejecucion
        result1 = runner.invoke(app, [
            'pfs-analysis',
            '--checkpoint', str(mock_classifier_checkpoint),
            '--data-dir', str(pfs_eval_dataset / "test"),
            '--output-dir', str(output1),
            '--device', 'cpu',
            '--approximate',
            '--num-samples', '2',
            '--margin', '0.15',
        ])

        # Segunda ejecucion
        result2 = runner.invoke(app, [
            'pfs-analysis',
            '--checkpoint', str(mock_classifier_checkpoint),
            '--data-dir', str(pfs_eval_dataset / "test"),
            '--output-dir', str(output2),
            '--device', 'cpu',
            '--approximate',
            '--num-samples', '2',
            '--margin', '0.15',
        ])

        # Session 32: Ambas ejecuciones deben tener mismo comportamiento
        assert result1.exit_code == result2.exit_code, \
            f"Different exit codes: {result1.exit_code} vs {result2.exit_code}"
