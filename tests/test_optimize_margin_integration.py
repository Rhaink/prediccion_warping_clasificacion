"""
Tests de integración para el comando optimize-margin.

Estos tests validan el flujo completo del comando, incluyendo:
- Generación de archivos de salida
- Formato correcto de resultados
- Manejo de edge cases
- Modo quick

Sesión 26: Tests de integración prioritarios.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

# Importar CliRunner para invocar comandos
from typer.testing import CliRunner

from src_v2.cli import app

runner = CliRunner()


# =============================================================================
# FIXTURES PARA DATOS DE PRUEBA
# =============================================================================

@pytest.fixture
def minimal_dataset(tmp_path):
    """
    Dataset mínimo con 3 clases y 6 imágenes cada una.

    Estructura:
        tmp_path/data/
            COVID/
                images/
                    COVID-001.png, COVID-002.png, ...
            Normal/
                images/
                    Normal-001.png, Normal-002.png, ...
            Viral_Pneumonia/
                images/
                    Viral-001.png, Viral-002.png, ...
    """
    data_dir = tmp_path / "data"
    classes = ["COVID", "Normal", "Viral_Pneumonia"]
    images_per_class = 6

    image_names = []

    for cls in classes:
        cls_dir = data_dir / cls / "images"
        cls_dir.mkdir(parents=True)

        for i in range(images_per_class):
            # Crear imagen gris con patrón distintivo por clase
            img_array = np.zeros((224, 224, 3), dtype=np.uint8)

            # Patrón diferente por clase para que el modelo pueda distinguir
            if cls == "COVID":
                img_array[:, :, 0] = 128  # Canal rojo
                img_array[50:174, 50:174] = [200, 100, 100]
            elif cls == "Normal":
                img_array[:, :, 1] = 128  # Canal verde
                img_array[50:174, 50:174] = [100, 200, 100]
            else:  # Viral_Pneumonia
                img_array[:, :, 2] = 128  # Canal azul
                img_array[50:174, 50:174] = [100, 100, 200]

            # Agregar ruido para variación
            noise = np.random.randint(-20, 20, img_array.shape, dtype=np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            img = Image.fromarray(img_array, mode='RGB')

            # Nombre de imagen
            prefix = "Viral" if cls == "Viral_Pneumonia" else cls
            img_name = f"{prefix}-{i+1:03d}.png"
            img.save(cls_dir / img_name)
            image_names.append((img_name, cls))

    return data_dir, image_names


@pytest.fixture
def landmarks_csv(tmp_path, minimal_dataset):
    """
    CSV de landmarks compatible con el formato esperado por optimize-margin.

    Formato con headers: image_name, category, L1_x, L1_y, ..., L15_x, L15_y
    """
    data_dir, image_names = minimal_dataset
    csv_path = tmp_path / "landmarks.csv"

    # Generar landmarks sintéticos pero consistentes
    np.random.seed(42)

    # Crear header
    header = ["image_name", "category"]
    for i in range(1, 16):
        header.extend([f"L{i}_x", f"L{i}_y"])

    rows = [header]
    for idx, (img_name, category) in enumerate(image_names):
        # Landmarks base (15 puntos) en coordenadas de píxel
        base_landmarks = np.array([
            [112, 25],    # L1 - Superior
            [112, 200],   # L2 - Inferior
            [50, 60],     # L3 - Apex izq
            [174, 60],    # L4 - Apex der
            [40, 112],    # L5 - Hilio izq
            [184, 112],   # L6 - Hilio der
            [50, 170],    # L7 - Base izq
            [174, 170],   # L8 - Base der
            [112, 70],    # L9 - Centro sup
            [112, 112],   # L10 - Centro med
            [112, 160],   # L11 - Centro inf
            [60, 35],     # L12 - Borde sup izq
            [164, 35],    # L13 - Borde sup der
            [35, 195],    # L14 - Costofrénico izq
            [189, 195],   # L15 - Costofrénico der
        ], dtype=np.float32)

        # Agregar variación pequeña
        noise = np.random.randn(15, 2) * 5
        landmarks = base_landmarks + noise
        landmarks = np.clip(landmarks, 5, 219)

        # Crear fila: nombre, categoría, coordenadas intercaladas
        row = [img_name.replace('.png', ''), category]
        for lm in landmarks:
            row.extend([int(lm[0]), int(lm[1])])
        rows.append(row)

    # Escribir CSV con header
    with open(csv_path, 'w') as f:
        for row in rows:
            f.write(','.join(map(str, row)) + '\n')

    return csv_path


@pytest.fixture
def canonical_shape_json(tmp_path):
    """
    JSON con forma canónica normalizada.
    Compatible con formato de canonical_shape_gpa.json
    """
    json_path = tmp_path / "canonical_shape.json"

    # Forma canónica normalizada (15 landmarks)
    canonical_shape = [
        [0.0, -0.245],     # L1 - Superior
        [0.0, 0.245],      # L2 - Inferior
        [-0.28, -0.16],    # L3 - Apex izq
        [0.28, -0.16],     # L4 - Apex der
        [-0.32, 0.0],      # L5 - Hilio izq
        [0.32, 0.0],       # L6 - Hilio der
        [-0.28, 0.16],     # L7 - Base izq
        [0.28, 0.16],      # L8 - Base der
        [0.0, -0.12],      # L9 - Centro sup
        [0.0, 0.0],        # L10 - Centro med
        [0.0, 0.12],       # L11 - Centro inf
        [-0.2, -0.22],     # L12 - Borde sup izq
        [0.2, -0.22],      # L13 - Borde sup der
        [-0.32, 0.24],     # L14 - Costofrénico izq
        [0.32, 0.24],      # L15 - Costofrénico der
    ]

    data = {
        "canonical_shape_normalized": canonical_shape,
        "canonical_shape_pixels": [
            [112 + x * 224, 112 + y * 224] for x, y in canonical_shape
        ],
        "image_size": 224,
        "n_landmarks": 15,
        "method": "Test GPA",
        "convergence": {
            "n_iterations": 1,
            "converged": True,
            "final_change": 0.0001,
            "n_shapes_used": 18
        }
    }

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    return json_path


@pytest.fixture
def triangles_json(tmp_path):
    """
    JSON con triangulación Delaunay.
    Compatible con formato de canonical_delaunay_triangles.json
    """
    json_path = tmp_path / "triangles.json"

    # Triangulación Delaunay sobre 15 puntos (18 triángulos típicos)
    triangles = [
        [0, 2, 11],
        [0, 11, 12],
        [0, 12, 3],
        [2, 4, 11],
        [3, 12, 5],
        [4, 6, 10],
        [4, 10, 8],
        [4, 8, 2],
        [5, 8, 10],
        [5, 10, 9],
        [5, 9, 3],
        [6, 13, 10],
        [7, 9, 10],
        [7, 10, 14],
        [1, 6, 13],
        [1, 13, 14],
        [1, 14, 7],
        [0, 8, 9],
    ]

    data = {
        "triangles": triangles,
        "num_triangles": len(triangles),
        "method": "Delaunay",
        "description": "Triangulación de prueba"
    }

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    return json_path


@pytest.fixture
def complete_test_setup(tmp_path, minimal_dataset, landmarks_csv, canonical_shape_json, triangles_json):
    """
    Setup completo con todos los archivos necesarios.
    Retorna diccionario con todas las rutas.
    """
    data_dir, image_names = minimal_dataset
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    return {
        "data_dir": data_dir,
        "landmarks_csv": landmarks_csv,
        "canonical_json": canonical_shape_json,
        "triangles_json": triangles_json,
        "output_dir": output_dir,
        "image_names": image_names,
        "tmp_path": tmp_path
    }


# =============================================================================
# TESTS DE INTEGRACIÓN - FLUJO COMPLETO
# =============================================================================

class TestOptimizeMarginIntegration:
    """Tests de integración del flujo completo de optimize-margin."""

    def test_optimize_margin_quick_mode_single_margin(self, complete_test_setup):
        """Test de flujo completo con modo quick y un solo margen."""
        setup = complete_test_setup

        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', str(setup["data_dir"]),
            '--landmarks-csv', str(setup["landmarks_csv"]),
            '--canonical', str(setup["canonical_json"]),
            '--triangles', str(setup["triangles_json"]),
            '--output-dir', str(setup["output_dir"]),
            '--margins', '1.10',
            '--epochs', '1',
            '--quick',
            '--architecture', 'resnet18',
            '--device', 'cpu',
            '--seed', '42'
        ])

        # Verificar que el comando completó
        assert result.exit_code == 0, f"Command failed with: {result.stdout}"

        # Verificar archivos de salida generados
        output_dir = setup["output_dir"]
        assert (output_dir / "margin_optimization_results.json").exists()
        assert (output_dir / "summary.csv").exists()

    def test_optimize_margin_quick_mode_multiple_margins(self, complete_test_setup):
        """Test de flujo completo con modo quick y múltiples márgenes."""
        setup = complete_test_setup

        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', str(setup["data_dir"]),
            '--landmarks-csv', str(setup["landmarks_csv"]),
            '--canonical', str(setup["canonical_json"]),
            '--triangles', str(setup["triangles_json"]),
            '--output-dir', str(setup["output_dir"]),
            '--margins', '1.00,1.10,1.20',
            '--epochs', '1',
            '--quick',
            '--architecture', 'resnet18',
            '--device', 'cpu'
        ])

        assert result.exit_code == 0, f"Command failed with: {result.stdout}"

        # Verificar que se probaron los 3 márgenes
        results_file = setup["output_dir"] / "margin_optimization_results.json"
        with open(results_file) as f:
            results = json.load(f)

        assert len(results["results"]) == 3
        margins_tested = [r["margin"] for r in results["results"]]
        assert 1.0 in margins_tested
        assert 1.1 in margins_tested
        assert 1.2 in margins_tested


class TestOptimizeMarginOutputs:
    """Tests de generación y formato de archivos de salida."""

    def test_json_results_structure(self, complete_test_setup):
        """Verificar estructura del JSON de resultados."""
        setup = complete_test_setup

        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', str(setup["data_dir"]),
            '--landmarks-csv', str(setup["landmarks_csv"]),
            '--canonical', str(setup["canonical_json"]),
            '--triangles', str(setup["triangles_json"]),
            '--output-dir', str(setup["output_dir"]),
            '--margins', '1.05,1.15',
            '--epochs', '1',
            '--quick',
            '--device', 'cpu'
        ])

        assert result.exit_code == 0

        results_file = setup["output_dir"] / "margin_optimization_results.json"
        with open(results_file) as f:
            results = json.load(f)

        # Verificar campos requeridos
        assert "timestamp" in results
        assert "configuration" in results
        assert "results" in results
        assert "best_margin" in results
        assert "best_accuracy" in results

        # Verificar estructura de configuration
        config = results["configuration"]
        assert "margins_tested" in config
        assert "epochs" in config
        assert "architecture" in config

        # Verificar estructura de cada resultado
        for r in results["results"]:
            assert "margin" in r
            assert "val_accuracy" in r
            assert "test_accuracy" in r
            assert "test_f1" in r

    def test_best_margin_matches_max_accuracy(self, complete_test_setup):
        """Verificar que best_margin corresponde al mayor accuracy."""
        setup = complete_test_setup

        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', str(setup["data_dir"]),
            '--landmarks-csv', str(setup["landmarks_csv"]),
            '--canonical', str(setup["canonical_json"]),
            '--triangles', str(setup["triangles_json"]),
            '--output-dir', str(setup["output_dir"]),
            '--margins', '1.00,1.10,1.20',
            '--epochs', '1',
            '--quick',
            '--device', 'cpu'
        ])

        assert result.exit_code == 0

        results_file = setup["output_dir"] / "margin_optimization_results.json"
        with open(results_file) as f:
            results = json.load(f)

        # Encontrar el mejor test_accuracy manualmente
        best_result = max(results["results"], key=lambda x: x["test_accuracy"])

        # Verificar que coincide
        assert results["best_margin"] == best_result["margin"]
        assert abs(results["best_accuracy"] - best_result["test_accuracy"]) < 0.01

    def test_summary_csv_format(self, complete_test_setup):
        """Verificar formato del CSV de resumen."""
        setup = complete_test_setup

        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', str(setup["data_dir"]),
            '--landmarks-csv', str(setup["landmarks_csv"]),
            '--canonical', str(setup["canonical_json"]),
            '--triangles', str(setup["triangles_json"]),
            '--output-dir', str(setup["output_dir"]),
            '--margins', '1.10',
            '--epochs', '1',
            '--quick',
            '--device', 'cpu'
        ])

        assert result.exit_code == 0

        summary_file = setup["output_dir"] / "summary.csv"
        df = pd.read_csv(summary_file)

        # Verificar columnas requeridas
        required_columns = ['margin', 'val_accuracy', 'test_accuracy', 'test_f1']
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

        # Verificar que hay datos
        assert len(df) >= 1

    def test_per_margin_checkpoints_created(self, complete_test_setup):
        """Verificar que se crean checkpoints por margen."""
        setup = complete_test_setup

        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', str(setup["data_dir"]),
            '--landmarks-csv', str(setup["landmarks_csv"]),
            '--canonical', str(setup["canonical_json"]),
            '--triangles', str(setup["triangles_json"]),
            '--output-dir', str(setup["output_dir"]),
            '--margins', '1.05,1.15',
            '--epochs', '1',
            '--quick',
            '--device', 'cpu'
        ])

        assert result.exit_code == 0

        # Verificar checkpoints
        per_margin_dir = setup["output_dir"] / "per_margin"
        assert per_margin_dir.exists()

        # Verificar subdirectorios por margen
        assert (per_margin_dir / "margin_1.05").exists()
        assert (per_margin_dir / "margin_1.15").exists()

        # Verificar checkpoints
        for margin in ["1.05", "1.15"]:
            checkpoint_path = per_margin_dir / f"margin_{margin}" / "checkpoint.pt"
            assert checkpoint_path.exists(), f"Missing checkpoint for margin {margin}"


class TestOptimizeMarginEdgeCases:
    """Tests de casos límite y edge cases."""

    def test_single_margin_execution(self, complete_test_setup):
        """Test con un solo margen."""
        setup = complete_test_setup

        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', str(setup["data_dir"]),
            '--landmarks-csv', str(setup["landmarks_csv"]),
            '--canonical', str(setup["canonical_json"]),
            '--triangles', str(setup["triangles_json"]),
            '--output-dir', str(setup["output_dir"]),
            '--margins', '1.00',
            '--epochs', '1',
            '--quick',
            '--device', 'cpu'
        ])

        assert result.exit_code == 0

        results_file = setup["output_dir"] / "margin_optimization_results.json"
        with open(results_file) as f:
            results = json.load(f)

        # Con un solo margen, ese debe ser el mejor
        assert len(results["results"]) == 1
        assert results["best_margin"] == 1.0

    def test_margin_1_0_no_scaling(self, complete_test_setup):
        """Test con margen 1.0 (sin escalado)."""
        setup = complete_test_setup

        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', str(setup["data_dir"]),
            '--landmarks-csv', str(setup["landmarks_csv"]),
            '--canonical', str(setup["canonical_json"]),
            '--triangles', str(setup["triangles_json"]),
            '--output-dir', str(setup["output_dir"]),
            '--margins', '1.0',
            '--epochs', '1',
            '--quick',
            '--device', 'cpu'
        ])

        assert result.exit_code == 0

    def test_large_margin(self, complete_test_setup):
        """Test con margen grande (1.5)."""
        setup = complete_test_setup

        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', str(setup["data_dir"]),
            '--landmarks-csv', str(setup["landmarks_csv"]),
            '--canonical', str(setup["canonical_json"]),
            '--triangles', str(setup["triangles_json"]),
            '--output-dir', str(setup["output_dir"]),
            '--margins', '1.50',
            '--epochs', '1',
            '--quick',
            '--device', 'cpu'
        ])

        assert result.exit_code == 0

    def test_quick_mode_limits_epochs(self, complete_test_setup):
        """Verificar que quick mode limita epochs a 3."""
        setup = complete_test_setup

        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', str(setup["data_dir"]),
            '--landmarks-csv', str(setup["landmarks_csv"]),
            '--canonical', str(setup["canonical_json"]),
            '--triangles', str(setup["triangles_json"]),
            '--output-dir', str(setup["output_dir"]),
            '--margins', '1.10',
            '--epochs', '100',  # Valor alto
            '--quick',  # Debe limitarse a 3
            '--device', 'cpu'
        ])

        assert result.exit_code == 0

        # Verificar en config que epochs efectivos <= 3
        results_file = setup["output_dir"] / "margin_optimization_results.json"
        with open(results_file) as f:
            results = json.load(f)

        # En quick mode, epochs debe ser <= 3
        assert results["configuration"]["epochs"] <= 3


class TestOptimizeMarginValidation:
    """Tests de validación de parámetros."""

    def test_invalid_margins_format_rejected(self, complete_test_setup):
        """Márgenes con formato inválido deben ser rechazados."""
        setup = complete_test_setup

        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', str(setup["data_dir"]),
            '--landmarks-csv', str(setup["landmarks_csv"]),
            '--margins', 'abc,def'  # Inválido
        ])

        assert result.exit_code != 0

    def test_negative_margin_rejected(self, complete_test_setup):
        """Márgenes negativos deben ser rechazados."""
        setup = complete_test_setup

        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', str(setup["data_dir"]),
            '--landmarks-csv', str(setup["landmarks_csv"]),
            '--margins', '-1.0,1.0'  # Negativo
        ])

        assert result.exit_code != 0

    def test_zero_margin_rejected(self, complete_test_setup):
        """Margen cero debe ser rechazado."""
        setup = complete_test_setup

        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', str(setup["data_dir"]),
            '--landmarks-csv', str(setup["landmarks_csv"]),
            '--margins', '0.0'  # Cero
        ])

        assert result.exit_code != 0

    def test_missing_data_dir_fails(self):
        """Directorio de datos inexistente debe fallar."""
        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', '/nonexistent/path',
            '--landmarks-csv', '/some/file.csv'
        ])

        assert result.exit_code != 0

    def test_missing_landmarks_csv_fails(self, complete_test_setup):
        """CSV de landmarks inexistente debe fallar."""
        setup = complete_test_setup

        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', str(setup["data_dir"]),
            '--landmarks-csv', '/nonexistent/landmarks.csv'
        ])

        assert result.exit_code != 0


class TestOptimizeMarginConfiguration:
    """Tests de opciones de configuración."""

    def test_custom_architecture(self, complete_test_setup):
        """Test con arquitectura personalizada."""
        setup = complete_test_setup

        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', str(setup["data_dir"]),
            '--landmarks-csv', str(setup["landmarks_csv"]),
            '--canonical', str(setup["canonical_json"]),
            '--triangles', str(setup["triangles_json"]),
            '--output-dir', str(setup["output_dir"]),
            '--margins', '1.10',
            '--epochs', '1',
            '--quick',
            '--architecture', 'mobilenet_v2',  # Diferente a default
            '--device', 'cpu'
        ])

        assert result.exit_code == 0

        # Verificar que se usó la arquitectura correcta
        results_file = setup["output_dir"] / "margin_optimization_results.json"
        with open(results_file) as f:
            results = json.load(f)

        assert results["configuration"]["architecture"] == "mobilenet_v2"

    def test_custom_batch_size(self, complete_test_setup):
        """Test con batch size personalizado."""
        setup = complete_test_setup

        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', str(setup["data_dir"]),
            '--landmarks-csv', str(setup["landmarks_csv"]),
            '--canonical', str(setup["canonical_json"]),
            '--triangles', str(setup["triangles_json"]),
            '--output-dir', str(setup["output_dir"]),
            '--margins', '1.10',
            '--epochs', '1',
            '--quick',
            '--batch-size', '8',
            '--device', 'cpu'
        ])

        assert result.exit_code == 0

    def test_seed_reproducibility(self, complete_test_setup):
        """Verificar que seed produce resultados consistentes."""
        setup = complete_test_setup

        # Primera ejecución
        output1 = setup["tmp_path"] / "output1"
        output1.mkdir()

        result1 = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', str(setup["data_dir"]),
            '--landmarks-csv', str(setup["landmarks_csv"]),
            '--canonical', str(setup["canonical_json"]),
            '--triangles', str(setup["triangles_json"]),
            '--output-dir', str(output1),
            '--margins', '1.10',
            '--epochs', '1',
            '--quick',
            '--seed', '12345',
            '--device', 'cpu'
        ])

        # Segunda ejecución con misma seed
        output2 = setup["tmp_path"] / "output2"
        output2.mkdir()

        result2 = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', str(setup["data_dir"]),
            '--landmarks-csv', str(setup["landmarks_csv"]),
            '--canonical', str(setup["canonical_json"]),
            '--triangles', str(setup["triangles_json"]),
            '--output-dir', str(output2),
            '--margins', '1.10',
            '--epochs', '1',
            '--quick',
            '--seed', '12345',  # Misma seed
            '--device', 'cpu'
        ])

        assert result1.exit_code == 0
        assert result2.exit_code == 0

        # Los resultados deberían ser idénticos con la misma seed
        with open(output1 / "margin_optimization_results.json") as f:
            results1 = json.load(f)
        with open(output2 / "margin_optimization_results.json") as f:
            results2 = json.load(f)

        # Comparar accuracies (pueden tener pequeñas diferencias por floating point)
        acc1 = results1["results"][0]["test_accuracy"]
        acc2 = results2["results"][0]["test_accuracy"]
        # Tolerancia: diferencia absoluta < 0.1 puntos porcentuales
        # o diferencia relativa < 0.5% del valor
        assert abs(acc1 - acc2) < max(0.1, acc1 * 0.005), \
            f"Reproducibilidad fallida: {acc1:.2f}% vs {acc2:.2f}%"


class TestOptimizeMarginModuleExecution:
    """Tests de ejecución como módulo."""

    def test_module_execution(self, complete_test_setup):
        """Test de ejecución vía python -m src_v2 optimize-margin."""
        setup = complete_test_setup

        result = subprocess.run(
            [
                sys.executable, '-m', 'src_v2', 'optimize-margin',
                '--data-dir', str(setup["data_dir"]),
                '--landmarks-csv', str(setup["landmarks_csv"]),
                '--canonical', str(setup["canonical_json"]),
                '--triangles', str(setup["triangles_json"]),
                '--output-dir', str(setup["output_dir"]),
                '--margins', '1.10',
                '--epochs', '1',
                '--quick',
                '--device', 'cpu'
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)  # Raíz del proyecto
        )

        assert result.returncode == 0, f"Failed with stderr: {result.stderr}"


# =============================================================================
# TESTS ADICIONALES - ROBUSTEZ
# =============================================================================

class TestOptimizeMarginRobustness:
    """Tests de robustez y manejo de errores."""

    def test_handles_missing_canonical_gracefully(self, complete_test_setup):
        """Verificar manejo de archivo canónico faltante."""
        setup = complete_test_setup

        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', str(setup["data_dir"]),
            '--landmarks-csv', str(setup["landmarks_csv"]),
            '--canonical', '/nonexistent/canonical.json',
            '--triangles', str(setup["triangles_json"]),
            '--output-dir', str(setup["output_dir"]),
            '--margins', '1.10'
        ])

        # Debe fallar con mensaje claro
        assert result.exit_code != 0

    def test_handles_missing_triangles_gracefully(self, complete_test_setup):
        """Verificar manejo de archivo de triángulos faltante."""
        setup = complete_test_setup

        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', str(setup["data_dir"]),
            '--landmarks-csv', str(setup["landmarks_csv"]),
            '--canonical', str(setup["canonical_json"]),
            '--triangles', '/nonexistent/triangles.json',
            '--output-dir', str(setup["output_dir"]),
            '--margins', '1.10'
        ])

        assert result.exit_code != 0

    def test_output_dir_created_if_not_exists(self, complete_test_setup):
        """Verificar que el directorio de salida se crea si no existe."""
        setup = complete_test_setup

        new_output = setup["tmp_path"] / "new_nested" / "output"

        result = runner.invoke(app, [
            'optimize-margin',
            '--data-dir', str(setup["data_dir"]),
            '--landmarks-csv', str(setup["landmarks_csv"]),
            '--canonical', str(setup["canonical_json"]),
            '--triangles', str(setup["triangles_json"]),
            '--output-dir', str(new_output),
            '--margins', '1.10',
            '--epochs', '1',
            '--quick',
            '--device', 'cpu'
        ])

        assert result.exit_code == 0
        assert new_output.exists()
