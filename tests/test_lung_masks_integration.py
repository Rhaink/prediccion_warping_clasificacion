"""
Tests de integracion para comando generate-lung-masks.

Session 32: Tests criticos para generacion de mascaras pulmonares aproximadas.
Las mascaras son usadas para calcular el Pulmonary Focus Score (PFS).

Cobertura previa: 0 tests de integracion (solo 5 tests de --help)
Meta: 10+ tests de integracion
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
# FIXTURES ESPECIFICAS PARA LUNG MASKS
# =============================================================================

@pytest.fixture
def lung_mask_dataset(tmp_path):
    """
    Dataset para generar mascaras pulmonares.

    Estructura:
        tmp_path/input/
            COVID/img1.png, img2.png
            Normal/img1.png, img2.png
    """
    data_dir = tmp_path / "input"

    for cls in ["COVID", "Normal"]:
        cls_dir = data_dir / cls
        cls_dir.mkdir(parents=True)

        for i in range(3):
            # Crear imagen grayscale simulando radiografia
            img_array = np.random.randint(50, 200, (224, 224), dtype=np.uint8)

            # Agregar patron central (simulando torax)
            center = 112
            for row in range(224):
                for col in range(224):
                    dist = np.sqrt((row - center) ** 2 + (col - center) ** 2)
                    if dist < 80:
                        img_array[row, col] = min(255, img_array[row, col] + 30)

            img = Image.fromarray(img_array, mode='L')
            img.save(cls_dir / f"{cls}_{i}.png")

    return data_dir


@pytest.fixture
def mixed_format_dataset(tmp_path):
    """
    Dataset con imagenes en diferentes formatos y tamanios.

    Session 32: Probar manejo de diferentes tipos de imagen.
    """
    data_dir = tmp_path / "mixed_input"
    data_dir.mkdir()

    # PNG grayscale
    img_gray = Image.new('L', (224, 224), color=128)
    img_gray.save(data_dir / "gray.png")

    # PNG RGB
    img_rgb = Image.new('RGB', (224, 224), color=(128, 128, 128))
    img_rgb.save(data_dir / "rgb.png")

    # JPEG
    img_jpg = Image.new('RGB', (224, 224), color=(100, 100, 100))
    img_jpg.save(data_dir / "image.jpg", quality=95)

    # Tamano diferente
    img_large = Image.new('L', (512, 512), color=150)
    img_large.save(data_dir / "large.png")

    return data_dir


# =============================================================================
# TESTS DE INTEGRACION - GENERATE-LUNG-MASKS BASICO
# =============================================================================

class TestGenerateLungMasksBasic:
    """Tests basicos del comando generate-lung-masks."""

    def test_generate_masks_rectangular_method(
        self, lung_mask_dataset, tmp_path
    ):
        """
        Generar mascaras con metodo rectangular.

        Session 32: Test basico del metodo por defecto.
        """
        output_dir = tmp_path / "masks_output"

        result = runner.invoke(app, [
            'generate-lung-masks',
            '--data-dir', str(lung_mask_dataset),
            '--output-dir', str(output_dir),
            '--method', 'rectangular',
            '--margin', '0.15',
        ])

        assert result.exit_code == 0, \
            f"Command failed (code {result.exit_code}): {result.stdout}"

        # Verificar que se crearon mascaras
        assert output_dir.exists(), "Output directory not created"

    def test_generate_masks_creates_correct_structure(
        self, lung_mask_dataset, tmp_path
    ):
        """
        Mascaras mantienen estructura de subdirectorios.

        Session 32: Verificar que la estructura de clases se preserva.
        """
        output_dir = tmp_path / "masks_structured"

        result = runner.invoke(app, [
            'generate-lung-masks',
            '--data-dir', str(lung_mask_dataset),
            '--output-dir', str(output_dir),
            '--method', 'rectangular',
        ])

        assert result.exit_code == 0

        # Verificar estructura
        for cls in ["COVID", "Normal"]:
            cls_dir = output_dir / cls
            assert cls_dir.exists(), f"Class directory {cls} not created"

            # Verificar que hay archivos de mascara
            mask_files = list(cls_dir.glob("*_mask.png"))
            assert len(mask_files) > 0, f"No masks created for {cls}"

    def test_generate_masks_output_is_binary(
        self, lung_mask_dataset, tmp_path
    ):
        """
        Mascaras generadas son imagenes binarias (0 o 255).

        Session 32: Verificar formato correcto de mascaras.
        """
        output_dir = tmp_path / "masks_binary"

        result = runner.invoke(app, [
            'generate-lung-masks',
            '--data-dir', str(lung_mask_dataset),
            '--output-dir', str(output_dir),
            '--method', 'rectangular',
            '--margin', '0.15',
        ])

        assert result.exit_code == 0

        # Verificar que las mascaras son binarias
        mask_files = list(output_dir.rglob("*_mask.png"))
        assert len(mask_files) > 0, "No mask files found"

        for mask_path in mask_files[:2]:  # Verificar primeras 2
            mask = np.array(Image.open(mask_path))
            unique_values = np.unique(mask)

            # Debe ser binaria (solo 0 y 255)
            assert all(v in [0, 255] for v in unique_values), \
                f"Mask {mask_path} not binary: {unique_values}"


# =============================================================================
# TESTS DE PARAMETROS
# =============================================================================

class TestGenerateLungMasksParameters:
    """Tests para diferentes parametros del comando."""

    def test_generate_masks_different_margins(
        self, lung_mask_dataset, tmp_path
    ):
        """
        Generar mascaras con diferentes valores de margin.

        Session 32: margin controla tamano de region central.
        """
        for margin in [0.05, 0.15, 0.25, 0.40]:
            output_dir = tmp_path / f"masks_m{margin}"

            result = runner.invoke(app, [
                'generate-lung-masks',
                '--data-dir', str(lung_mask_dataset),
                '--output-dir', str(output_dir),
                '--method', 'rectangular',
                '--margin', str(margin),
            ])

            assert result.exit_code == 0, \
                f"Margin {margin} failed: {result.stdout}"

    def test_generate_masks_margin_affects_size(
        self, lung_mask_dataset, tmp_path
    ):
        """
        Diferentes margins producen mascaras de diferente tamano.

        Session 32: Verificar que margin tiene efecto en el resultado.
        """
        # Mascara con margin pequeno (mas area blanca)
        output_small = tmp_path / "masks_small_margin"
        result1 = runner.invoke(app, [
            'generate-lung-masks',
            '--data-dir', str(lung_mask_dataset),
            '--output-dir', str(output_small),
            '--margin', '0.05',
        ])

        # Mascara con margin grande (menos area blanca)
        output_large = tmp_path / "masks_large_margin"
        result2 = runner.invoke(app, [
            'generate-lung-masks',
            '--data-dir', str(lung_mask_dataset),
            '--output-dir', str(output_large),
            '--margin', '0.40',
        ])

        assert result1.exit_code == 0
        assert result2.exit_code == 0

        # Comparar una mascara de cada
        mask_small = list(output_small.rglob("*_mask.png"))[0]
        mask_large = list(output_large.rglob("*_mask.png"))[0]

        area_small = np.sum(np.array(Image.open(mask_small)) > 0)
        area_large = np.sum(np.array(Image.open(mask_large)) > 0)

        # Margin pequeno debe dar mas area blanca
        assert area_small > area_large, \
            f"Small margin ({area_small}) should have more area than large ({area_large})"


# =============================================================================
# TESTS DE MANEJO DE ERRORES
# =============================================================================

class TestGenerateLungMasksErrors:
    """Tests de manejo de errores."""

    def test_generate_masks_invalid_data_dir_fails(self, tmp_path):
        """
        Comando falla con directorio inexistente.

        Session 32: Validacion de paths.
        """
        result = runner.invoke(app, [
            'generate-lung-masks',
            '--data-dir', '/nonexistent/data',
            '--output-dir', str(tmp_path / "out"),
        ])

        assert result.exit_code != 0

    def test_generate_masks_invalid_method_fails(
        self, lung_mask_dataset, tmp_path
    ):
        """
        Comando falla con metodo no soportado.

        Session 32: Solo 'rectangular' esta implementado.
        """
        result = runner.invoke(app, [
            'generate-lung-masks',
            '--data-dir', str(lung_mask_dataset),
            '--output-dir', str(tmp_path / "out"),
            '--method', 'invalid_method',
        ])

        assert result.exit_code != 0

    def test_generate_masks_invalid_margin_fails(
        self, lung_mask_dataset, tmp_path
    ):
        """
        Comando falla con margin fuera de rango [0, 0.5).

        Session 32: Validacion de parametros numericos.
        """
        for invalid_margin in [-0.1, 0.5, 0.6, 1.0]:
            result = runner.invoke(app, [
                'generate-lung-masks',
                '--data-dir', str(lung_mask_dataset),
                '--output-dir', str(tmp_path / "out"),
                '--margin', str(invalid_margin),
            ])

            assert result.exit_code != 0, \
                f"Should fail with invalid margin {invalid_margin}"

    def test_generate_masks_empty_directory(self, tmp_path):
        """
        Comando maneja correctamente directorio vacio.

        Session 32: Caso edge - sin imagenes.
        """
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = runner.invoke(app, [
            'generate-lung-masks',
            '--data-dir', str(empty_dir),
            '--output-dir', str(tmp_path / "out"),
        ])

        # Session 32: Directorio vacio debe fallar con error descriptivo
        assert result.exit_code != 0


# =============================================================================
# TESTS DE FORMATOS DE IMAGEN
# =============================================================================

class TestGenerateLungMasksFormats:
    """Tests para diferentes formatos de imagen."""

    def test_generate_masks_handles_grayscale(
        self, tmp_path
    ):
        """
        Comando procesa imagenes grayscale.

        Session 32: Radiografias suelen ser grayscale.
        """
        # Crear dataset grayscale
        input_dir = tmp_path / "grayscale_input"
        input_dir.mkdir()

        img = Image.new('L', (224, 224), color=128)
        img.save(input_dir / "gray_image.png")

        output_dir = tmp_path / "grayscale_output"

        result = runner.invoke(app, [
            'generate-lung-masks',
            '--data-dir', str(input_dir),
            '--output-dir', str(output_dir),
        ])

        assert result.exit_code == 0

    def test_generate_masks_handles_rgb(
        self, tmp_path
    ):
        """
        Comando procesa imagenes RGB.

        Session 32: Algunas radiografias vienen en RGB.
        """
        input_dir = tmp_path / "rgb_input"
        input_dir.mkdir()

        img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        img.save(input_dir / "rgb_image.png")

        output_dir = tmp_path / "rgb_output"

        result = runner.invoke(app, [
            'generate-lung-masks',
            '--data-dir', str(input_dir),
            '--output-dir', str(output_dir),
        ])

        assert result.exit_code == 0

    def test_generate_masks_handles_jpeg(
        self, tmp_path
    ):
        """
        Comando procesa imagenes JPEG.

        Session 32: Verificar soporte de formato JPEG.
        """
        input_dir = tmp_path / "jpeg_input"
        input_dir.mkdir()

        img = Image.new('RGB', (224, 224), color=(100, 100, 100))
        img.save(input_dir / "image.jpg", quality=95)

        output_dir = tmp_path / "jpeg_output"

        result = runner.invoke(app, [
            'generate-lung-masks',
            '--data-dir', str(input_dir),
            '--output-dir', str(output_dir),
        ])

        assert result.exit_code == 0

    def test_generate_masks_handles_different_sizes(
        self, tmp_path
    ):
        """
        Comando procesa imagenes de diferentes tamanos.

        Session 32: No todas las imagenes son 224x224.
        """
        input_dir = tmp_path / "multi_size_input"
        input_dir.mkdir()

        # Diferentes tamanos
        sizes = [(224, 224), (256, 256), (512, 512), (320, 240)]

        for i, (w, h) in enumerate(sizes):
            img = Image.new('L', (w, h), color=128)
            img.save(input_dir / f"size_{w}x{h}.png")

        output_dir = tmp_path / "multi_size_output"

        result = runner.invoke(app, [
            'generate-lung-masks',
            '--data-dir', str(input_dir),
            '--output-dir', str(output_dir),
        ])

        assert result.exit_code == 0

        # Verificar que se crearon mascaras para todas
        mask_files = list(output_dir.glob("*_mask.png"))
        assert len(mask_files) == len(sizes), \
            f"Expected {len(sizes)} masks, got {len(mask_files)}"


# =============================================================================
# TESTS DE INTEGRIDAD
# =============================================================================

class TestGenerateLungMasksIntegrity:
    """Tests de integridad y consistencia."""

    def test_generate_masks_preserves_dimensions(
        self, lung_mask_dataset, tmp_path
    ):
        """
        Mascaras tienen mismas dimensiones que imagenes originales.

        Session 32: Las mascaras deben coincidir en tamano.
        """
        output_dir = tmp_path / "masks_dim"

        result = runner.invoke(app, [
            'generate-lung-masks',
            '--data-dir', str(lung_mask_dataset),
            '--output-dir', str(output_dir),
        ])

        assert result.exit_code == 0

        # Verificar dimensiones
        for cls in ["COVID", "Normal"]:
            images = list((lung_mask_dataset / cls).glob("*.png"))
            masks = list((output_dir / cls).glob("*_mask.png"))

            for img_path in images[:1]:
                img = Image.open(img_path)
                img_size = img.size

                # Buscar mascara correspondiente
                mask_name = f"{img_path.stem}_mask.png"
                mask_path = output_dir / cls / mask_name

                if mask_path.exists():
                    mask = Image.open(mask_path)
                    assert mask.size == img_size, \
                        f"Mask size {mask.size} != image size {img_size}"

    def test_generate_masks_deterministic(
        self, lung_mask_dataset, tmp_path
    ):
        """
        Comando produce resultados identicos con mismos inputs.

        Session 32: Verificar reproducibilidad.
        """
        output1 = tmp_path / "masks_run1"
        output2 = tmp_path / "masks_run2"

        # Primera ejecucion
        result1 = runner.invoke(app, [
            'generate-lung-masks',
            '--data-dir', str(lung_mask_dataset),
            '--output-dir', str(output1),
            '--margin', '0.15',
        ])

        # Segunda ejecucion
        result2 = runner.invoke(app, [
            'generate-lung-masks',
            '--data-dir', str(lung_mask_dataset),
            '--output-dir', str(output2),
            '--margin', '0.15',
        ])

        assert result1.exit_code == 0
        assert result2.exit_code == 0

        # Comparar mascaras
        masks1 = sorted(output1.rglob("*_mask.png"))
        masks2 = sorted(output2.rglob("*_mask.png"))

        assert len(masks1) == len(masks2), "Different number of masks"

        for m1, m2 in zip(masks1, masks2):
            arr1 = np.array(Image.open(m1))
            arr2 = np.array(Image.open(m2))
            assert np.array_equal(arr1, arr2), \
                f"Masks differ: {m1.name}"
