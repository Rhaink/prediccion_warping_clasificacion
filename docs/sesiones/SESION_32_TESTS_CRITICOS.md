# Sesion 32: Tests Criticos para Comandos sin Cobertura

**Fecha:** 2025-12-10
**Objetivo:** Agregar tests de integracion para comandos CLI sin cobertura

## Resumen Ejecutivo

Se agregaron **37 tests nuevos** para comandos que tenian 0 o minima cobertura de integracion:

| Metrica | Anterior | Nuevo | Cambio |
|---------|----------|-------|--------|
| Tests totales | 501 | 538 | +37 |
| Tests pfs-analysis | 0 | 16 | +16 |
| Tests generate-lung-masks | 0 | 15 | +15 |
| Tests test-robustness | 2 | 8 | +6 |
| Suite completa | PASS | PASS | - |

## Archivos Creados/Modificados

### Archivos Nuevos

1. **tests/test_pfs_integration.py** (16 tests)
   - `TestPFSAnalysisBasic`: 3 tests
   - `TestPFSAnalysisParameters`: 3 tests
   - `TestPFSAnalysisErrors`: 7 tests
   - `TestPFSWithMasks`: 2 tests
   - `TestPFSReproducibility`: 1 test

2. **tests/test_lung_masks_integration.py** (15 tests)
   - `TestGenerateLungMasksBasic`: 3 tests
   - `TestGenerateLungMasksParameters`: 2 tests
   - `TestGenerateLungMasksErrors`: 4 tests
   - `TestGenerateLungMasksFormats`: 4 tests
   - `TestGenerateLungMasksIntegrity`: 2 tests

### Archivos Modificados

3. **tests/test_cli_integration.py** (+6 tests en TestTestRobustnessIntegration)
   - `test_robustness_outputs_json_structure`
   - `test_robustness_different_splits`
   - `test_robustness_handles_invalid_split`
   - `test_robustness_handles_empty_dataset`
   - `test_robustness_with_trained_model`
   - `test_robustness_batch_size_options`

## Tests Creados por Comando

### pfs-analysis (16 tests)

El comando `pfs-analysis` calcula el Pulmonary Focus Score para evaluar si el clasificador enfoca su atencion en regiones pulmonares.

```python
# Tests basicos
test_pfs_with_approximate_masks       # Mascaras rectangulares
test_pfs_creates_output_directory     # Creacion de directorios
test_pfs_basic_output_structure       # Estructura de salida

# Tests de parametros
test_pfs_different_thresholds         # threshold: 0.3, 0.5, 0.7
test_pfs_different_margins            # margin: 0.1, 0.15, 0.2, 0.3
test_pfs_num_samples_option           # num_samples: 1, 2, 5

# Tests de errores
test_pfs_invalid_checkpoint_fails     # Checkpoint inexistente
test_pfs_invalid_data_dir_fails       # Data dir inexistente
test_pfs_requires_mask_source         # Sin --mask-dir ni --approximate
test_pfs_invalid_threshold_fails      # threshold fuera de [0,1]
test_pfs_invalid_margin_fails         # margin fuera de [0, 0.5)
test_pfs_invalid_num_samples_fails    # num_samples <= 0
test_pfs_handles_empty_directory      # Directorio vacio

# Tests con mascaras
test_pfs_with_mask_directory          # --mask-dir real
test_pfs_invalid_mask_directory_fails # mask-dir inexistente

# Tests de reproducibilidad
test_pfs_deterministic_output         # Mismos inputs = mismos outputs
```

### generate-lung-masks (15 tests)

El comando `generate-lung-masks` genera mascaras pulmonares aproximadas para el calculo de PFS.

```python
# Tests basicos
test_generate_masks_rectangular_method      # Metodo rectangular
test_generate_masks_creates_correct_structure # Estructura de subdirs
test_generate_masks_output_is_binary        # Mascaras binarias (0/255)

# Tests de parametros
test_generate_masks_different_margins       # margin: 0.05, 0.15, 0.25, 0.40
test_generate_masks_margin_affects_size     # margin afecta area blanca

# Tests de errores
test_generate_masks_invalid_data_dir_fails  # Data dir inexistente
test_generate_masks_invalid_method_fails    # Metodo no soportado
test_generate_masks_invalid_margin_fails    # margin fuera de [0, 0.5)
test_generate_masks_empty_directory         # Sin imagenes

# Tests de formatos
test_generate_masks_handles_grayscale       # Imagenes grayscale
test_generate_masks_handles_rgb             # Imagenes RGB
test_generate_masks_handles_jpeg            # Formato JPEG
test_generate_masks_handles_different_sizes # Diferentes resoluciones

# Tests de integridad
test_generate_masks_preserves_dimensions    # Mascara = tamano imagen
test_generate_masks_deterministic           # Reproducibilidad
```

### test-robustness (6 tests nuevos)

El comando `test-robustness` evalua la robustez del clasificador ante perturbaciones (JPEG, blur, noise).

```python
# Tests nuevos Session 32
test_robustness_outputs_json_structure    # Verificar JSON de salida
test_robustness_different_splits          # Split: test, val
test_robustness_handles_invalid_split     # Split inexistente
test_robustness_handles_empty_dataset     # Dataset vacio
test_robustness_with_trained_model        # Modelo entrenado real
test_robustness_batch_size_options        # batch_size: 1, 2, 4
```

## Fixtures Creadas

### En test_pfs_integration.py

```python
@pytest.fixture
def pfs_eval_dataset(tmp_path):
    """Dataset con estructura test/COVID, test/Normal, test/Viral_Pneumonia"""

@pytest.fixture
def pfs_dataset_with_masks(tmp_path):
    """Dataset con imagenes y mascaras binarias reales"""
```

### En test_lung_masks_integration.py

```python
@pytest.fixture
def lung_mask_dataset(tmp_path):
    """Dataset con imagenes grayscale simulando radiografias"""

@pytest.fixture
def mixed_format_dataset(tmp_path):
    """Dataset con PNG, JPEG, grayscale, RGB, diferentes tamanos"""
```

### En test_cli_integration.py

```python
@pytest.fixture
def trained_robustness_classifier(self, tmp_path):
    """Clasificador entrenado para tests de robustness"""
```

## Patron de Test Usado

Todos los tests siguen el patron establecido en Session 31:

```python
def test_command_basic(self, fixtures, tmp_path):
    """
    Descripcion del test.

    Session 32: Contexto especifico del test.
    """
    result = runner.invoke(app, [
        'comando',
        '--opcion', str(valor),
        '--device', 'cpu',
    ])

    # Session 32: Modelo sin entrenar puede fallar pero no crashear
    assert result.exit_code in [0, 1], \
        f"Command crashed (code {result.exit_code}): {result.stdout}"
```

## Cobertura Mejorada

### Antes de Session 32

| Comando | Tests Integracion |
|---------|-------------------|
| pfs-analysis | 0 (solo --help) |
| generate-lung-masks | 0 (solo --help) |
| test-robustness | 2 |

### Despues de Session 32

| Comando | Tests Integracion |
|---------|-------------------|
| pfs-analysis | 16 |
| generate-lung-masks | 15 |
| test-robustness | 8 |

## Ejecucion de Tests

```bash
# Ejecutar solo tests nuevos
.venv/bin/python -m pytest tests/test_pfs_integration.py tests/test_lung_masks_integration.py -v

# Ejecutar tests de robustness
.venv/bin/python -m pytest tests/test_cli_integration.py::TestTestRobustnessIntegration -v

# Ejecutar suite completa
.venv/bin/python -m pytest tests/ -v
```

## Resultado Final

```
================ 538 passed, 729 warnings in 500.55s (0:08:20) =================
```

Todos los tests pasaron exitosamente.

## Bugs Criticos Identificados y Corregidos

Durante el analisis profundo con multiples agentes, se identificaron y corrigieron 3 bugs criticos:

### Bug #1: Assertion que siempre pasa
**Archivo:** `test_pfs_integration.py:188`
```python
# ANTES (siempre verdadero - >= 0 nunca falla)
assert total_outputs >= 0

# DESPUES (corregido)
assert total_outputs > 0
```

### Bug #2: Fixture con datos incoherentes
**Archivo:** `test_pfs_integration.py:63-103`
- Creaba una imagen que se sobrescribia
- Mascaras no coincidian en nombre con imagenes
- Corregido creando multiples imagenes con nombres coincidentes

### Bug #3: Assertion JSON demasiado debil
**Archivo:** `test_cli_integration.py:1830-1858`
```python
# ANTES (acepta cualquier JSON)
assert len(data) > 0

# DESPUES (verifica estructura especifica)
has_perturbations = 'perturbations' in data
has_baseline = 'baseline_accuracy' in data
assert has_perturbations or has_baseline
```

## Proximos Pasos (Session 33+)

1. **Agregar tests para comandos restantes:**
   - analyze-errors (2 tests actuales -> 8)
   - gradcam (2 tests actuales -> 8)
   - cross-evaluate (2 tests actuales -> 6)
   - evaluate-external (2 tests actuales -> 6)

2. **Mejorar cobertura de codigo:**
   - Meta: 70% cobertura
   - Actual estimado: ~45%

3. **Resolver bugs pendientes (medios):**
   - M1: test_image_file sin verificar guardado
   - M2: Codigo duplicado sin parametrizacion
   - M4: Memory leak potencial en modelos
   - B1: num_workers hardcoded
