"""
Tests para los modulos de procesamiento (GPA y Warp).

Incluye tests para:
- Funciones GPA (Generalized Procrustes Analysis)
- Funciones de warping piecewise affine
- Nuevos comandos CLI (compute-canonical, generate-dataset)
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from typer.testing import CliRunner

from src_v2.cli import app
from src_v2.processing.gpa import (
    center_shape,
    scale_shape,
    optimal_rotation_matrix,
    align_shape,
    procrustes_distance,
    gpa_iterative,
    scale_canonical_to_image,
    compute_delaunay_triangulation,
)
from src_v2.processing.warp import (
    scale_landmarks_from_centroid,
    clip_landmarks_to_image,
    add_boundary_points,
    piecewise_affine_warp,
    compute_fill_rate,
    get_affine_transform_matrix,
    create_triangle_mask,
    get_bounding_box,
    warp_triangle,
    warp_mask,
)


runner = CliRunner()


class TestCenterShape:
    """Tests para center_shape."""

    def test_center_shape_basic(self):
        """Shape debe quedar centrada en origen."""
        shape = np.array([[10, 10], [20, 10], [15, 20]])
        centered, centroid = center_shape(shape)

        # El centroide de la forma centrada debe ser (0, 0)
        assert np.allclose(centered.mean(axis=0), [0, 0], atol=1e-10)

        # El centroide original debe ser correcto
        assert np.allclose(centroid, [15, 40/3], atol=1e-10)

    def test_center_shape_preserves_relative_positions(self):
        """Posiciones relativas se preservan."""
        shape = np.array([[0, 0], [10, 0], [5, 10]])
        centered, _ = center_shape(shape)

        # La diferencia entre puntos debe ser la misma
        original_diff = shape[1] - shape[0]
        centered_diff = centered[1] - centered[0]
        assert np.allclose(original_diff, centered_diff)

    def test_center_shape_already_centered(self):
        """Shape ya centrada debe permanecer igual."""
        shape = np.array([[-1, 0], [1, 0], [0, 1]])  # Centroide en (0, 1/3)
        centered, centroid = center_shape(shape)

        assert centered.shape == shape.shape


class TestScaleShape:
    """Tests para scale_shape."""

    def test_scale_shape_unit_norm(self):
        """Shape escalada debe tener norma unitaria."""
        shape = np.array([[0, 0], [10, 0], [5, 10]])
        scaled, original_scale = scale_shape(shape)

        # Norma Frobenius debe ser 1
        assert np.isclose(np.linalg.norm(scaled, 'fro'), 1.0, atol=1e-10)

    def test_scale_shape_preserves_ratios(self):
        """Proporciones se preservan."""
        shape = np.array([[0, 0], [4, 0], [4, 3]])  # 3-4-5 triangle
        scaled, _ = scale_shape(shape)

        # Las proporciones entre puntos deben preservarse
        original_dist_01 = np.linalg.norm(shape[1] - shape[0])
        original_dist_12 = np.linalg.norm(shape[2] - shape[1])
        scaled_dist_01 = np.linalg.norm(scaled[1] - scaled[0])
        scaled_dist_12 = np.linalg.norm(scaled[2] - scaled[1])

        assert np.isclose(original_dist_01 / original_dist_12,
                         scaled_dist_01 / scaled_dist_12, atol=1e-10)


class TestOptimalRotation:
    """Tests para optimal_rotation_matrix."""

    def test_rotation_identity(self):
        """Shapes identicas deben dar matriz identidad."""
        shape = np.array([[1, 0], [0, 1], [-1, 0]])
        R = optimal_rotation_matrix(shape, shape)

        assert np.allclose(R, np.eye(2), atol=1e-10)

    def test_rotation_returns_valid_matrix(self):
        """Rotacion optima debe retornar matriz ortogonal valida."""
        source = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float64)
        target = np.array([[0.7, 0.7], [-0.7, 0.7], [-0.7, -0.7]], dtype=np.float64)

        R = optimal_rotation_matrix(source, target)

        # R debe ser una matriz 2x2
        assert R.shape == (2, 2)

        # R debe ser ortogonal (R @ R.T = I)
        identity = R @ R.T
        assert np.allclose(identity, np.eye(2), atol=1e-10)

    def test_rotation_is_proper(self):
        """Matriz de rotacion debe tener det = 1 (no reflexion)."""
        source = np.array([[1, 0], [0, 1], [-1, 0]])
        target = np.array([[0, 1], [-1, 0], [0, -1]])

        R = optimal_rotation_matrix(source, target)

        # Determinante debe ser +1 (rotacion propia)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)


class TestAlignShape:
    """Tests para align_shape."""

    def test_align_preserves_shape(self):
        """Alineacion debe preservar la forma (norma)."""
        # Usar formas ya centradas y escaladas
        reference = np.array([[1, 0], [0, 1], [-1, 0], [0.5, -0.5]], dtype=np.float64)
        reference, _ = center_shape(reference)
        reference, _ = scale_shape(reference)

        # Rotar referencia para crear una forma diferente
        angle = 0.3  # radianes
        R_rot = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
        rotated = reference @ R_rot

        aligned = align_shape(rotated, reference)

        # La norma debe preservarse (alineacion solo rota)
        norm_before = np.linalg.norm(rotated, 'fro')
        norm_after = np.linalg.norm(aligned, 'fro')

        assert np.isclose(norm_before, norm_after, atol=1e-10)


class TestProcrustesDistance:
    """Tests para procrustes_distance."""

    def test_distance_zero_identical_shapes(self):
        """Shapes identicas deben tener distancia 0."""
        shape = np.array([[0, 0], [10, 0], [5, 10]])
        dist = procrustes_distance(shape, shape)
        assert np.isclose(dist, 0, atol=1e-10)

    def test_distance_invariant_to_translation(self):
        """Distancia debe ser invariante a traslacion."""
        shape1 = np.array([[0, 0], [10, 0], [5, 10]])
        shape2 = shape1 + [100, 200]  # Trasladar

        dist = procrustes_distance(shape1, shape2)
        assert np.isclose(dist, 0, atol=1e-10)

    def test_distance_invariant_to_scale(self):
        """Distancia debe ser invariante a escala."""
        shape1 = np.array([[0, 0], [10, 0], [5, 10]])
        shape2 = shape1 * 5  # Escalar

        dist = procrustes_distance(shape1, shape2)
        assert np.isclose(dist, 0, atol=1e-10)

    def test_distance_small_for_similar_shapes(self):
        """Formas similares deben tener distancia Procrustes pequena."""
        # Crear dos formas similares con pequenas perturbaciones
        shape1 = np.array([[0, 0], [10, 0], [5, 10], [7, 3]], dtype=np.float64)
        # Agregar ruido pequeno
        np.random.seed(42)
        noise = np.random.randn(4, 2) * 0.5
        shape2 = shape1 + noise

        dist = procrustes_distance(shape1, shape2)
        # La distancia debe ser pequena (menor a 0.5)
        assert dist < 0.5


class TestGPAIterative:
    """Tests para gpa_iterative."""

    def test_gpa_with_identical_shapes(self):
        """GPA con shapes identicas debe converger inmediatamente."""
        shape = np.array([[0, 0], [10, 0], [5, 10]])
        shapes = np.stack([shape, shape, shape])

        canonical, aligned, info = gpa_iterative(shapes, max_iterations=100)

        assert info['converged']
        assert info['n_iterations'] <= 3

    def test_gpa_output_shapes(self):
        """GPA debe retornar shapes correctas."""
        n_shapes = 5
        n_landmarks = 15
        shapes = np.random.randn(n_shapes, n_landmarks, 2)

        canonical, aligned, info = gpa_iterative(shapes)

        assert canonical.shape == (n_landmarks, 2)
        assert aligned.shape == (n_shapes, n_landmarks, 2)
        assert 'n_iterations' in info
        assert 'converged' in info
        assert 'final_change' in info

    def test_gpa_canonical_is_normalized(self):
        """Forma canonica debe tener norma ~1."""
        shapes = np.random.randn(10, 15, 2) * 100

        canonical, _, _ = gpa_iterative(shapes)

        norm = np.linalg.norm(canonical, 'fro')
        assert np.isclose(norm, 1.0, atol=0.01)


class TestScaleCanonicalToImage:
    """Tests para scale_canonical_to_image."""

    def test_output_within_image_bounds(self):
        """Forma escalada debe estar dentro de la imagen."""
        # Forma canonica normalizada
        canonical = np.array([[-0.2, -0.3], [0.2, 0.3], [0, 0.1]])

        scaled = scale_canonical_to_image(canonical, image_size=224, padding=0.1)

        # Todos los puntos deben estar dentro de [0, 224]
        assert scaled.min() >= 0
        assert scaled.max() <= 224

    def test_output_centered(self):
        """Forma escalada debe estar centrada en la imagen."""
        canonical = np.array([[-0.2, -0.3], [0.2, 0.3], [0, 0.1]])

        scaled = scale_canonical_to_image(canonical, image_size=224)

        # El centroide debe estar cerca del centro de la imagen
        centroid = scaled.mean(axis=0)
        assert np.allclose(centroid, [112, 112], atol=20)


class TestComputeDelaunay:
    """Tests para compute_delaunay_triangulation."""

    def test_delaunay_basic(self):
        """Triangulacion Delaunay basica."""
        landmarks = np.array([[0, 0], [10, 0], [5, 10], [15, 10]])

        triangles = compute_delaunay_triangulation(landmarks)

        assert triangles.shape[1] == 3  # Cada triangulo tiene 3 vertices
        assert triangles.max() < len(landmarks)  # Indices validos

    def test_delaunay_15_landmarks(self):
        """Triangulacion con 15 landmarks."""
        # Generar 15 landmarks aleatorios
        np.random.seed(42)
        landmarks = np.random.rand(15, 2) * 224

        triangles = compute_delaunay_triangulation(landmarks)

        assert triangles.shape[1] == 3
        assert len(triangles) > 0  # Debe haber al menos 1 triangulo


class TestScaleLandmarksFromCentroid:
    """Tests para scale_landmarks_from_centroid."""

    def test_scale_expands_from_centroid(self):
        """Escalar > 1 debe expandir desde centroide."""
        landmarks = np.array([[0, 0], [10, 0], [5, 10]])
        centroid = landmarks.mean(axis=0)

        scaled = scale_landmarks_from_centroid(landmarks, scale=2.0)

        # La distancia al centroide debe duplicarse
        original_dist = np.linalg.norm(landmarks[0] - centroid)
        scaled_dist = np.linalg.norm(scaled[0] - centroid)
        assert np.isclose(scaled_dist, original_dist * 2.0, atol=1e-10)

    def test_scale_contracts_from_centroid(self):
        """Escalar < 1 debe contraer hacia centroide."""
        landmarks = np.array([[0, 0], [10, 0], [5, 10]])
        centroid = landmarks.mean(axis=0)

        scaled = scale_landmarks_from_centroid(landmarks, scale=0.5)

        original_dist = np.linalg.norm(landmarks[0] - centroid)
        scaled_dist = np.linalg.norm(scaled[0] - centroid)
        assert np.isclose(scaled_dist, original_dist * 0.5, atol=1e-10)

    def test_scale_one_unchanged(self):
        """Escalar = 1 no debe cambiar nada."""
        landmarks = np.array([[0, 0], [10, 0], [5, 10]])

        scaled = scale_landmarks_from_centroid(landmarks, scale=1.0)

        assert np.allclose(scaled, landmarks)


class TestClipLandmarks:
    """Tests para clip_landmarks_to_image."""

    def test_clip_within_bounds(self):
        """Landmarks fuera de limites deben recortarse."""
        landmarks = np.array([[-10, -10], [300, 300], [112, 112]])

        clipped = clip_landmarks_to_image(landmarks, image_size=224, margin=2)

        assert clipped.min() >= 2
        assert clipped.max() <= 224 - 2 - 1

    def test_clip_preserves_valid_landmarks(self):
        """Landmarks validos deben permanecer sin cambios."""
        landmarks = np.array([[50, 50], [100, 100], [150, 150]])

        clipped = clip_landmarks_to_image(landmarks, image_size=224, margin=2)

        assert np.allclose(clipped, landmarks)


class TestAddBoundaryPoints:
    """Tests para add_boundary_points."""

    def test_adds_8_points(self):
        """Debe agregar 8 puntos (4 esquinas + 4 medios)."""
        landmarks = np.random.rand(15, 2) * 224

        extended = add_boundary_points(landmarks, image_size=224)

        assert extended.shape == (23, 2)  # 15 + 8

    def test_boundary_corners(self):
        """Las esquinas deben estar en las posiciones correctas."""
        landmarks = np.random.rand(15, 2) * 224

        extended = add_boundary_points(landmarks, image_size=224)

        corners = extended[15:19]  # Indices 15-18 son esquinas
        expected_corners = np.array([
            [0, 0], [223, 0], [0, 223], [223, 223]
        ])
        assert np.allclose(corners, expected_corners)


class TestGetAffineTransformMatrix:
    """Tests para get_affine_transform_matrix."""

    def test_identity_transform(self):
        """Triangulos identicos deben dar transformacion identidad."""
        tri = np.array([[0, 0], [100, 0], [50, 100]], dtype=np.float32)
        M = get_affine_transform_matrix(tri, tri)

        # Matriz 2x3: parte 2x2 debe ser identidad, traslacion 0
        assert M.shape == (2, 3)
        assert np.allclose(M[:, :2], np.eye(2), atol=1e-5)
        assert np.allclose(M[:, 2], [0, 0], atol=1e-5)

    def test_translation_transform(self):
        """Trasladar triangulo debe dar matriz de traslacion."""
        src = np.array([[0, 0], [100, 0], [50, 100]], dtype=np.float32)
        dst = src + np.array([10, 20], dtype=np.float32)

        M = get_affine_transform_matrix(src, dst)

        # La parte 2x2 debe ser identidad, traslacion [10, 20]
        assert np.allclose(M[:, :2], np.eye(2), atol=1e-5)
        assert np.allclose(M[:, 2], [10, 20], atol=1e-5)

    def test_scale_transform(self):
        """Escalar triangulo debe dar matriz de escalado."""
        src = np.array([[0, 0], [100, 0], [0, 100]], dtype=np.float32)
        dst = src * 2

        M = get_affine_transform_matrix(src, dst)

        # La matriz debe escalar por 2
        assert np.allclose(M[:, :2], np.eye(2) * 2, atol=1e-5)


class TestCreateTriangleMask:
    """Tests para create_triangle_mask."""

    def test_mask_shape(self):
        """Mascara debe tener forma correcta."""
        triangle = np.array([[10, 10], [50, 10], [30, 50]])
        mask = create_triangle_mask((100, 100), triangle)

        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8

    def test_mask_values(self):
        """Mascara debe tener solo 0 y 255."""
        triangle = np.array([[10, 10], [50, 10], [30, 50]])
        mask = create_triangle_mask((100, 100), triangle)

        unique_values = np.unique(mask)
        assert set(unique_values).issubset({0, 255})

    def test_mask_contains_centroid(self):
        """El centroide del triangulo debe estar dentro de la mascara."""
        triangle = np.array([[20, 20], [80, 20], [50, 80]])
        mask = create_triangle_mask((100, 100), triangle)

        centroid = triangle.mean(axis=0).astype(int)
        assert mask[centroid[1], centroid[0]] == 255


class TestGetBoundingBox:
    """Tests para get_bounding_box."""

    def test_basic_bounding_box(self):
        """Bounding box basico."""
        triangle = np.array([[10, 20], [50, 30], [30, 80]])
        x, y, w, h = get_bounding_box(triangle)

        assert x == 10
        assert y == 20
        assert w == 40  # 50 - 10
        assert h == 60  # 80 - 20

    def test_bounding_box_with_float(self):
        """Bounding box con coordenadas float."""
        triangle = np.array([[10.5, 20.7], [50.3, 30.2], [30.8, 80.9]])
        x, y, w, h = get_bounding_box(triangle)

        assert x == 10  # floor(10.5)
        assert y == 20  # floor(20.7)
        # ceil(50.3) - floor(10.5) = 51 - 10 = 41
        assert w == 41
        # ceil(80.9) - floor(20.7) = 81 - 20 = 61
        assert h == 61

    def test_bounding_box_negative_clipped(self):
        """Bounding box con valores negativos se recorta a 0."""
        triangle = np.array([[-10, -5], [50, 30], [30, 80]])
        x, y, w, h = get_bounding_box(triangle)

        assert x == 0  # max(0, -10)
        assert y == 0  # max(0, -5)


class TestWarpTriangle:
    """Tests para warp_triangle."""

    def test_warp_triangle_in_place(self):
        """Warp debe modificar la imagen destino in-place."""
        src_img = np.ones((100, 100), dtype=np.uint8) * 128
        dst_img = np.zeros((100, 100), dtype=np.uint8)

        src_tri = np.array([[20, 20], [80, 20], [50, 80]], dtype=np.float64)
        dst_tri = np.array([[25, 25], [75, 25], [50, 75]], dtype=np.float64)

        warp_triangle(src_img, dst_img, src_tri, dst_tri)

        # La imagen destino debe haber sido modificada
        assert dst_img.max() > 0

    def test_warp_triangle_preserves_outside(self):
        """Warp no debe modificar pixeles fuera del triangulo destino."""
        src_img = np.ones((100, 100), dtype=np.uint8) * 255
        dst_img = np.zeros((100, 100), dtype=np.uint8)

        # Triangulo pequeno en una esquina
        src_tri = np.array([[10, 10], [30, 10], [20, 30]], dtype=np.float64)
        dst_tri = np.array([[10, 10], [30, 10], [20, 30]], dtype=np.float64)

        warp_triangle(src_img, dst_img, src_tri, dst_tri)

        # La esquina opuesta debe seguir siendo 0
        assert dst_img[90, 90] == 0

    def test_warp_triangle_handles_color(self):
        """Warp debe manejar imagenes a color."""
        src_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        dst_img = np.zeros((100, 100, 3), dtype=np.uint8)

        src_tri = np.array([[20, 20], [80, 20], [50, 80]], dtype=np.float64)
        dst_tri = np.array([[25, 25], [75, 25], [50, 75]], dtype=np.float64)

        warp_triangle(src_img, dst_img, src_tri, dst_tri)

        # La imagen destino debe haber sido modificada en todos los canales
        assert dst_img.max() > 0


class TestGPAEdgeCases:
    """Tests de edge cases para GPA."""

    def test_gpa_max_iterations_zero(self):
        """GPA con max_iterations=0 no debe fallar."""
        shapes = np.random.randn(5, 15, 2)

        canonical, aligned, info = gpa_iterative(shapes, max_iterations=0)

        assert canonical.shape == (15, 2)
        assert info['n_iterations'] == 0

    def test_gpa_single_shape(self):
        """GPA con una sola forma no debe fallar."""
        shape = np.array([[0, 0], [10, 0], [5, 10]])
        shapes = np.stack([shape])

        # Con una sola forma, GPA puede comportarse de forma especial
        # Lo importante es que no falle
        canonical, aligned, info = gpa_iterative(shapes)

        # La forma canonica debe tener la forma correcta
        assert canonical.shape == (3, 2)
        # Aligned debe tener la forma correcta
        assert aligned.shape == (1, 3, 2)

    def test_scale_canonical_zero_range(self):
        """scale_canonical_to_image con rango cero debe manejar division por cero."""
        # Forma con todos los puntos en el mismo lugar
        canonical = np.array([[0, 0], [0, 0], [0, 0]])

        # No debe lanzar excepcion
        scaled = scale_canonical_to_image(canonical, image_size=224)

        assert scaled.shape == (3, 2)


class TestPiecewiseAffineWarp:
    """Tests para piecewise_affine_warp."""

    def test_warp_output_shape(self):
        """Warp debe producir imagen del tamano correcto."""
        image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        source = np.random.rand(15, 2) * 200 + 12
        target = np.random.rand(15, 2) * 200 + 12

        warped = piecewise_affine_warp(image, source, target, output_size=224)

        assert warped.shape == (224, 224)

    def test_warp_identical_landmarks_preserves_image(self):
        """Warp con landmarks identicos debe preservar imagen (aprox)."""
        image = np.random.randint(50, 200, (224, 224), dtype=np.uint8)
        landmarks = np.array([
            [30, 30], [194, 30], [30, 194], [194, 194],
            [60, 60], [164, 60], [60, 164], [164, 164],
            [112, 30], [112, 112], [112, 194], [60, 30],
            [164, 30], [60, 194], [164, 194]
        ], dtype=np.float64)

        warped = piecewise_affine_warp(
            image, landmarks, landmarks,
            use_full_coverage=True
        )

        # Con landmarks identicos, la imagen debe ser similar
        # Permitimos diferencia por interpolacion
        diff = np.abs(warped.astype(float) - image.astype(float))
        assert diff.mean() < 50  # Diferencia promedio baja


class TestComputeFillRate:
    """Tests para compute_fill_rate."""

    def test_full_image(self):
        """Imagen sin pixeles negros debe tener fill_rate = 1."""
        image = np.ones((224, 224), dtype=np.uint8) * 128

        fill_rate = compute_fill_rate(image)

        assert np.isclose(fill_rate, 1.0, atol=1e-10)

    def test_half_black(self):
        """Imagen mitad negra debe tener fill_rate ~ 0.5."""
        image = np.zeros((224, 224), dtype=np.uint8)
        image[:, 112:] = 128  # Mitad derecha con valor

        fill_rate = compute_fill_rate(image)

        assert np.isclose(fill_rate, 0.5, atol=0.01)


class TestComputeCanonicalCommand:
    """Tests para comando compute-canonical del CLI."""

    def test_compute_canonical_help(self):
        """Comando compute-canonical debe mostrar ayuda."""
        result = runner.invoke(app, ['compute-canonical', '--help'])
        assert result.exit_code == 0
        assert 'GPA' in result.stdout or 'canonical' in result.stdout.lower()
        assert 'LANDMARKS_CSV' in result.stdout

    def test_compute_canonical_missing_csv(self):
        """compute-canonical con CSV inexistente debe fallar."""
        result = runner.invoke(app, [
            'compute-canonical',
            '/nonexistent/landmarks.csv'
        ])
        assert result.exit_code != 0


class TestGenerateDatasetCommand:
    """Tests para comando generate-dataset del CLI."""

    def test_generate_dataset_help(self):
        """Comando generate-dataset debe mostrar ayuda."""
        result = runner.invoke(app, ['generate-dataset', '--help'])
        assert result.exit_code == 0
        assert 'warped' in result.stdout.lower() or 'dataset' in result.stdout.lower()
        assert '--checkpoint' in result.stdout
        assert '--margin' in result.stdout
        assert '--splits' in result.stdout

    def test_generate_dataset_missing_input(self):
        """generate-dataset con directorio inexistente debe fallar."""
        result = runner.invoke(app, [
            'generate-dataset',
            '/nonexistent/input',
            '/nonexistent/output',
            '--checkpoint', '/nonexistent/model.pt'
        ])
        assert result.exit_code != 0

    def test_generate_dataset_invalid_splits(self):
        """generate-dataset con splits invalidos debe fallar."""
        result = runner.invoke(app, [
            'generate-dataset',
            '/some/input',
            '/some/output',
            '--checkpoint', '/some/model.pt',
            '--splits', '0.5,0.5'  # Solo 2 valores
        ])
        assert result.exit_code != 0

    def test_generate_dataset_splits_not_sum_to_one(self):
        """generate-dataset con splits que no suman 1 debe fallar."""
        result = runner.invoke(app, [
            'generate-dataset',
            '/some/input',
            '/some/output',
            '--checkpoint', '/some/model.pt',
            '--splits', '0.3,0.3,0.3'  # Suma 0.9
        ])
        assert result.exit_code != 0

    def test_generate_dataset_negative_splits(self):
        """generate-dataset con splits negativos debe fallar."""
        result = runner.invoke(app, [
            'generate-dataset',
            '/some/input',
            '/some/output',
            '--checkpoint', '/some/model.pt',
            '--splits', '-0.1,0.5,0.6'
        ])
        assert result.exit_code != 0


class TestCLIEdgeCases:
    """Tests de edge cases adicionales para CLI."""

    def test_compute_canonical_empty_csv(self, tmp_path):
        """compute-canonical con CSV vacio debe manejar el error."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("x0,y0,x1,y1\n")  # Solo headers

        result = runner.invoke(app, [
            'compute-canonical',
            str(empty_csv)
        ])

        # Debe fallar pero no con crash
        assert result.exit_code != 0

    def test_compute_canonical_invalid_columns(self, tmp_path):
        """compute-canonical con columnas invalidas debe fallar."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("col1,col2\na,b\n")

        result = runner.invoke(app, [
            'compute-canonical',
            str(bad_csv)
        ])

        assert result.exit_code != 0


class TestNewClassifierArchitectures:
    """Tests para las nuevas arquitecturas del clasificador."""

    def test_resnet50_classifier(self):
        """Crear clasificador con ResNet-50."""
        from src_v2.models.classifier import ImageClassifier

        model = ImageClassifier(backbone="resnet50", num_classes=3, pretrained=False)
        assert model is not None
        assert model.backbone_name == "resnet50"

        # Verificar forward pass
        x = torch.randn(2, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output.shape == (2, 3)

    def test_alexnet_classifier(self):
        """Crear clasificador con AlexNet."""
        from src_v2.models.classifier import ImageClassifier

        model = ImageClassifier(backbone="alexnet", num_classes=3, pretrained=False)
        assert model is not None
        assert model.backbone_name == "alexnet"

        x = torch.randn(2, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output.shape == (2, 3)

    def test_vgg16_classifier(self):
        """Crear clasificador con VGG-16."""
        from src_v2.models.classifier import ImageClassifier

        model = ImageClassifier(backbone="vgg16", num_classes=3, pretrained=False)
        assert model is not None
        assert model.backbone_name == "vgg16"

        x = torch.randn(2, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output.shape == (2, 3)

    def test_mobilenet_v2_classifier(self):
        """Crear clasificador con MobileNetV2."""
        from src_v2.models.classifier import ImageClassifier

        model = ImageClassifier(backbone="mobilenet_v2", num_classes=3, pretrained=False)
        assert model is not None
        assert model.backbone_name == "mobilenet_v2"

        x = torch.randn(2, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output.shape == (2, 3)

    def test_densenet121_classifier(self):
        """Verificar que DenseNet-121 sigue funcionando."""
        from src_v2.models.classifier import ImageClassifier

        model = ImageClassifier(backbone="densenet121", num_classes=3, pretrained=False)
        assert model is not None
        assert model.backbone_name == "densenet121"

        x = torch.randn(2, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output.shape == (2, 3)

    @pytest.mark.parametrize("backbone", [
        "resnet18", "resnet50", "efficientnet_b0", "densenet121",
        "alexnet", "vgg16", "mobilenet_v2"
    ])
    def test_all_backbones_forward_pass(self, backbone):
        """Todos los backbones deben funcionar correctamente."""
        from src_v2.models.classifier import ImageClassifier

        model = ImageClassifier(backbone=backbone, num_classes=3, pretrained=False)
        model.eval()

        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 3)

    def test_supported_backbones_list(self):
        """Verificar lista de backbones soportados."""
        from src_v2.models.classifier import ImageClassifier

        expected = {
            "resnet18", "resnet50", "efficientnet_b0", "densenet121",
            "alexnet", "vgg16", "mobilenet_v2"
        }
        actual = set(ImageClassifier.SUPPORTED_BACKBONES)

        assert actual == expected


class TestWarpMask:
    """Tests para warp_mask (transformar máscaras con misma geometría que imágenes)."""

    def test_warp_mask_output_shape(self):
        """warp_mask debe producir máscara del tamaño correcto."""
        mask = np.zeros((224, 224), dtype=np.uint8)
        mask[50:150, 50:150] = 255  # Cuadrado central

        source = np.random.rand(15, 2) * 180 + 22
        target = np.random.rand(15, 2) * 180 + 22

        warped = warp_mask(mask, source, target, output_size=224)

        assert warped.shape == (224, 224)
        assert warped.dtype == np.uint8

    def test_warp_mask_binary_values(self):
        """warp_mask debe producir solo valores 0 o 255."""
        mask = np.zeros((224, 224), dtype=np.uint8)
        mask[80:144, 80:144] = 255

        # Landmarks que no distorsionan mucho
        landmarks = np.array([
            [30, 30], [194, 30], [30, 194], [194, 194],
            [60, 60], [164, 60], [60, 164], [164, 164],
            [112, 30], [112, 112], [112, 194], [60, 30],
            [164, 30], [60, 194], [164, 194]
        ], dtype=np.float64)

        warped = warp_mask(mask, landmarks, landmarks)

        unique_values = np.unique(warped)
        assert set(unique_values).issubset({0, 255})

    def test_warp_mask_identical_landmarks_preserves_shape(self):
        """warp_mask con landmarks idénticos debe preservar la máscara aproximadamente."""
        mask = np.zeros((224, 224), dtype=np.uint8)
        mask[60:160, 60:160] = 255

        landmarks = np.array([
            [30, 30], [194, 30], [30, 194], [194, 194],
            [60, 60], [164, 60], [60, 164], [164, 164],
            [112, 30], [112, 112], [112, 194], [60, 30],
            [164, 30], [60, 194], [164, 194]
        ], dtype=np.float64)

        warped = warp_mask(mask, landmarks, landmarks, use_full_coverage=True)

        # El área blanca debe ser aproximadamente igual
        original_area = np.sum(mask > 0)
        warped_area = np.sum(warped > 0)
        ratio = warped_area / original_area

        assert 0.8 < ratio < 1.2  # Permitir 20% de diferencia por interpolación

    def test_warp_mask_handles_rgb_input(self):
        """warp_mask debe manejar entrada RGB (convertir a grayscale)."""
        mask_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        mask_rgb[80:144, 80:144, :] = 255

        landmarks = np.random.rand(15, 2) * 180 + 22

        warped = warp_mask(mask_rgb, landmarks, landmarks)

        assert warped.shape == (224, 224)  # Debe ser 2D
        assert warped.dtype == np.uint8

    def test_warp_mask_handles_normalized_input(self):
        """warp_mask debe manejar máscaras normalizadas (0-1)."""
        mask = np.zeros((224, 224), dtype=np.float32)
        mask[80:144, 80:144] = 1.0

        landmarks = np.random.rand(15, 2) * 180 + 22

        warped = warp_mask(mask, landmarks, landmarks)

        assert warped.dtype == np.uint8
        assert warped.max() == 255 or warped.max() == 0

    def test_warp_mask_full_coverage_vs_partial(self):
        """warp_mask con full_coverage debe cubrir más área."""
        mask = np.ones((224, 224), dtype=np.uint8) * 255

        source = np.random.rand(15, 2) * 140 + 42
        target = np.random.rand(15, 2) * 140 + 42

        warped_partial = warp_mask(mask, source, target, use_full_coverage=False)
        warped_full = warp_mask(mask, source, target, use_full_coverage=True)

        # Full coverage debe tener más píxeles cubiertos
        partial_coverage = np.sum(warped_partial > 0) / warped_partial.size
        full_coverage = np.sum(warped_full > 0) / warped_full.size

        assert full_coverage >= partial_coverage
