# Prompt para Sesion 33: Tests de Visualizacion y Correccion de Bugs Medios

> **Instrucciones:** Copia todo el contenido de este archivo y pegalo como tu primer mensaje en una nueva conversacion con Claude.

---

Comienza la Sesion 33. Usa ultrathink para planificar y ejecutar
las tareas de manera sistematica. Ejecuta los tests frecuentemente
para verificar que todo funciona.

## Contexto del Proyecto

Estoy desarrollando un sistema CLI para clasificacion de COVID-19 en rayos X usando normalizacion geometrica (warping). El proyecto demuestra que las imagenes warpeadas mejoran la generalizacion y robustez de los clasificadores.

**Hipotesis de tesis (CONFIRMADA con datos reales):**
> "Las imagenes warpeadas mejoran la generalizacion 11x (gap 25.36% -> 2.24%) y la robustez 30x (degradacion JPEG 16.14% -> 0.53%) porque eliminan variabilidad geometrica que causa overfitting."

**Datos verificados en archivos JSON reales:**
- `outputs/session30_analysis/consolidated_results.json` - Generalizacion
- `outputs/session29_robustness/artifact_robustness_results.json` - Robustez

## Estado Actual Post-Sesion 32

### Metricas Verificadas:
- **538 tests** pasando (100%)
- **20 comandos CLI** funcionando
- **0 bugs criticos** (3 corregidos en Sesion 32)
- **5 bugs medios** pendientes
- **~45% cobertura** estimada (meta: 70%)
- **85-90% proyecto completo**

### Logros Sesion 32:
1. +37 tests nuevos (pfs-analysis: 16, generate-lung-masks: 15, robustness: 6)
2. Bug corregido: assertion `>= 0` que siempre pasaba
3. Bug corregido: fixture `pfs_dataset_with_masks` incoherente
4. Bug corregido: assertion JSON demasiado debil
5. Verificacion completa de datos de hipotesis (NO inventados)

### Bugs Medios Pendientes:
| ID | Bug | Ubicacion | Severidad |
|----|-----|-----------|-----------|
| M1 | test_image_file sin verificar guardado | conftest.py:168-189 | Media |
| M2 | Codigo duplicado sin parametrizacion | test_cli_integration.py | Media |
| M4 | Memory leak potencial en modelos | conftest.py:93-99 | Media |
| B1 | num_workers hardcoded (problemas Windows) | cli.py:5189, 5501 | Media |
| M5 | Assertions debiles con `exit_code in [0,1]` | Multiples archivos | Media |

## Objetivo Principal - Sesion 33

**Completar cobertura de comandos de visualizacion y corregir bugs medios.**

### Prioridad 1: Tests para `gradcam` (8-10 tests)
El comando genera mapas de atencion GradCAM para interpretar decisiones del clasificador.

```python
# Tests a crear en test_cli_integration.py o test_visualization.py
- test_gradcam_single_image_creates_output
- test_gradcam_batch_directory
- test_gradcam_different_layers
- test_gradcam_overlay_options
- test_gradcam_saves_heatmap_correctly
- test_gradcam_invalid_checkpoint_fails
- test_gradcam_invalid_image_fails
- test_gradcam_alpha_parameter
```

### Prioridad 1: Tests para `analyze-errors` (8-10 tests)
El comando analiza errores de clasificacion y genera reportes.

```python
# Tests a crear
- test_analyze_errors_creates_report
- test_analyze_errors_top_k_parameter
- test_analyze_errors_saves_confusion_matrix
- test_analyze_errors_identifies_worst_samples
- test_analyze_errors_per_class_analysis
- test_analyze_errors_invalid_checkpoint_fails
- test_analyze_errors_with_gradcam_output
- test_analyze_errors_json_output_structure
```

### Prioridad 2: Corregir Bugs Medios (5 bugs)

#### M1: test_image_file sin verificar guardado
```python
# conftest.py:168-189 - Agregar verificacion
@pytest.fixture
def test_image_file(tmp_path):
    # ... crear imagen ...
    img.save(img_path)
    # Session 33: Verificar que la imagen se guardo correctamente
    assert img_path.exists(), f"Image not saved: {img_path}"
    assert img_path.stat().st_size > 0, "Image file is empty"
    return img_path
```

#### M2: Codigo duplicado sin parametrizacion
```python
# Usar @pytest.mark.parametrize en lugar de loops
@pytest.mark.parametrize("margin", [0.1, 0.15, 0.2, 0.3])
def test_pfs_different_margins(self, mock_checkpoint, dataset, tmp_path, margin):
    # Test parametrizado
```

#### M4: Memory leak potencial
```python
# conftest.py:93-99 - Agregar limpieza explicita
@pytest.fixture
def mock_classifier_checkpoint(tmp_path):
    # ... crear checkpoint ...
    yield checkpoint_path
    # Session 33: Limpieza explicita
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

#### B1: num_workers hardcoded
```python
# cli.py - Detectar OS y ajustar
import platform
num_workers = 0 if platform.system() == 'Windows' else 4
```

#### M5: Assertions debiles
```python
# Reemplazar exit_code in [0,1] con verificaciones especificas
# Si esperamos exito:
assert result.exit_code == 0, f"Expected success: {result.stdout}"
# Si esperamos fallo gracioso:
assert result.exit_code == 1, f"Expected graceful failure: {result.stdout}"
```

## Estructura de Archivos Relevantes

```
tests/
  conftest.py                 # Fixtures compartidas (bugs M1, M4)
  test_cli.py                 # Tests unitarios CLI
  test_cli_integration.py     # Tests integracion (bug M2)
  test_visualization.py       # Tests GradCAM, ErrorAnalyzer
  test_pfs_integration.py     # Tests PFS (Session 32)
  test_lung_masks_integration.py  # Tests mascaras (Session 32)

src_v2/
  cli.py                      # CLI principal (bug B1)
  visualization/
    gradcam.py                # Implementacion GradCAM
    error_analyzer.py         # Analisis de errores
    pfs_analysis.py           # Pulmonary Focus Score
```

## Comandos CLI Relevantes

```bash
# GradCAM - generar mapa de atencion
python -m src_v2 gradcam \
    --checkpoint outputs/classifier/best.pt \
    --image data/test/COVID/sample.png \
    --output gradcam_output.png \
    --alpha 0.5

# Analyze-errors - analizar errores
python -m src_v2 analyze-errors \
    --checkpoint outputs/classifier/best.pt \
    --data-dir data/test \
    --output-dir outputs/error_analysis \
    --top-k 10
```

## Fixtures Disponibles

```python
# Ya existentes en conftest.py
- mock_classifier_checkpoint   # Checkpoint de clasificador
- test_image_file             # Imagen PNG temporal
- classification_dataset      # Dataset 3 clases

# Ya existentes en test_cli_integration.py
- trained_classifier          # Clasificador entrenado (lento)
- robustness_dataset          # Dataset para robustez
```

## Meta de la Sesion

| Metrica | Actual | Meta Sesion 33 |
|---------|--------|----------------|
| Tests totales | 538 | 560 (+22) |
| Tests gradcam | 2 | 10 |
| Tests analyze-errors | 2 | 10 |
| Bugs medios | 5 | 0 |
| Cobertura estimada | 45% | 50% |

## Comandos de Verificacion

```bash
# Ejecutar tests de visualizacion
.venv/bin/python -m pytest tests/test_visualization.py -v

# Ejecutar tests de CLI integration
.venv/bin/python -m pytest tests/test_cli_integration.py -v -k "gradcam or error"

# Ver cobertura
.venv/bin/python -m pytest tests/ --cov=src_v2 --cov-report=term-missing

# Ejecutar suite completa
.venv/bin/python -m pytest tests/ -v
```

## Patron de Test Recomendado

```python
class TestGradCAMIntegration:
    """Tests de integracion para comando gradcam."""

    def test_gradcam_creates_output_file(
        self, mock_classifier_checkpoint, test_image_file, tmp_path
    ):
        """
        GradCAM crea archivo de salida con mapa de atencion.

        Session 33: Verificar que el comando genera output visual.
        """
        output_file = tmp_path / "gradcam_output.png"

        result = runner.invoke(app, [
            'gradcam',
            '--checkpoint', str(mock_classifier_checkpoint),
            '--image', str(test_image_file),
            '--output', str(output_file),
            '--device', 'cpu',
        ])

        # Session 33: Verificar exito y creacion de archivo
        assert result.exit_code == 0, f"Command failed: {result.stdout}"
        assert output_file.exists(), "Output file not created"
        assert output_file.stat().st_size > 1000, "Output file too small (possibly corrupt)"
```

## Notas Importantes

1. **Usar `exit_code == 0`** para tests que esperan exito (no `in [0, 1]`)
2. **Agregar comentarios "Session 33:"** para nuevas assertions y fixes
3. **Parametrizar tests** con `@pytest.mark.parametrize` en lugar de loops
4. **Limpiar memoria** despues de usar modelos en fixtures
5. **Ejecutar suite completa** al final para verificar no regresiones

## Orden de Trabajo Recomendado

1. Leer este prompt completo
2. Revisar `tests/test_visualization.py` para entender tests existentes
3. Agregar 8-10 tests para `gradcam`
4. Agregar 8-10 tests para `analyze-errors`
5. Corregir bugs M1, M4 en `conftest.py`
6. Corregir bug M2 parametrizando tests duplicados
7. Corregir bug B1 en `cli.py`
8. Ejecutar suite completa
9. Documentar en `docs/sesiones/SESION_33_VISUALIZATION_TESTS.md`

## Contexto Adicional

El proyecto esta al 85-90% de completitud. Despues de esta sesion, quedan:
- Sesion 34: Galeria visual original vs warped
- Sesion 35: Tests estadisticos (p-values)
- Sesion 36: PFS con mascaras warpeadas
- Sesion 37-40: Documentacion final y defensa

**El objetivo final es demostrar cientificamente que el warping mejora clasificadores medicos.**
