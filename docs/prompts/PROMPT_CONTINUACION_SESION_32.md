# Prompt de Continuacion - Sesion 32

## Instrucciones de Uso
Copia todo el contenido desde "---INICIO DEL PROMPT---" hasta "---FIN DEL PROMPT---" y pegalo como tu primer mensaje en una nueva conversacion con Claude.

---INICIO DEL PROMPT---

## Contexto del Proyecto

Estoy desarrollando un sistema CLI para clasificacion de COVID-19 en rayos X usando normalizacion geometrica (warping). El proyecto tiene como objetivo demostrar que las imagenes warpeadas mejoran la generalizacion y robustez de los clasificadores.

**Hipotesis de tesis (confirmada parcialmente):**
> "Las imagenes warpeadas mejoran la generalizacion 11x (gap 25.36% -> 2.24%) y la robustez 30x (degradacion JPEG 16.14% -> 0.53%) porque eliminan variabilidad geometrica que causa overfitting."

## Estado Actual Post-Sesion 31

### Metricas Verificadas (datos reales, no inventados):
- **501 tests** pasando
- **20 comandos CLI** funcionando
- **0 bugs criticos** (corregidos en Sesion 31)
- **5 bugs medios** pendientes
- **~40% cobertura** estimada (meta: 70%)
- **15,153 imagenes** en dataset

### Bugs Corregidos en Sesion 31:
1. Race conditions en fixtures (conftest.py:51-53) - CORREGIDO
2. 25 assertions debiles documentadas - CORREGIDO
3. JSON loading sin verificacion - CORREGIDO
4. Tolerancia reproducibilidad 5% -> 1% - CORREGIDO
5. GradCAM no inicializado en pfs-analysis - CORREGIDO
6. console.print sin importar - CORREGIDO

### Bugs Pendientes (Media Severidad):
| ID | Bug | Ubicacion |
|----|-----|-----------|
| M1 | test_image_file sin verificar guardado | conftest.py:168-189 |
| M2 | Codigo duplicado sin parametrizacion | test_cli_integration.py:217-231 |
| M4 | Memory leak potencial en modelos | conftest.py:93-99 |
| B1 | num_workers hardcoded (problemas Windows) | cli.py:5189, 5501 |

## Objetivo Principal - Sesion 32

**Agregar tests criticos para comandos sin cobertura de integracion.**

### Comandos con 0 Tests de Integracion:
1. **pfs-analysis** - Solo tiene 7 tests de --help
2. **generate-lung-masks** - Solo tiene 5 tests de --help

### Comandos con Cobertura Minima (2 tests):
3. **test-robustness** - Falta probar diferentes perturbaciones
4. **analyze-errors** - Falta probar outputs y gradcam
5. **gradcam** - Falta probar batch processing y diferentes capas
6. **cross-evaluate** - Falta probar comparacion real
7. **evaluate-external** - Falta probar datasets binarios

## Tests Especificos a Crear

### Prioridad 1: pfs-analysis (12-15 tests)
```python
# tests/test_pfs_integration.py
- test_pfs_analysis_with_approximate_masks
- test_pfs_analysis_outputs_json
- test_pfs_analysis_different_thresholds
- test_pfs_analysis_handles_empty_dir
- test_pfs_analysis_error_invalid_checkpoint
- test_pfs_analysis_reproducibility
```

### Prioridad 1: generate-lung-masks (10-12 tests)
```python
# tests/test_lung_masks_integration.py
- test_generate_lung_masks_rectangular_method
- test_generate_lung_masks_different_margins
- test_generate_lung_masks_creates_output_dir
- test_generate_lung_masks_handles_grayscale
- test_generate_lung_masks_error_invalid_method
```

### Prioridad 2: test-robustness (8-10 tests)
```python
# Agregar a test_cli_integration.py
- test_robustness_jpeg_compression
- test_robustness_gaussian_blur
- test_robustness_gaussian_noise
- test_robustness_outputs_json_structure
- test_robustness_compares_clean_vs_perturbed
```

## Estructura de Archivos Relevantes

```
tests/
  conftest.py                 # Fixtures compartidas
  test_cli.py                 # Tests unitarios CLI (102 tests)
  test_cli_integration.py     # Tests integracion (principales)
  test_classifier.py          # Tests clasificador
  test_visualization.py       # Tests GradCAM, ErrorAnalyzer
  test_processing.py          # Tests GPA, Warp

src_v2/
  cli.py                      # CLI principal (6000+ lineas)
  visualization/
    gradcam.py                # Implementacion GradCAM
    error_analyzer.py         # Analisis de errores
    pfs.py                    # Pulmonary Focus Score
```

## Fixtures Disponibles en conftest.py

```python
# Para tests de clasificador
- mock_classifier_checkpoint   # Checkpoint sin entrenar
- classification_dataset       # Dataset sintetico 3 clases
- classification_eval_dataset  # Dataset solo test

# Para tests de landmarks
- mock_landmark_checkpoint     # Checkpoint landmarks
- minimal_landmark_dataset     # Dataset CSV + imagenes
- sample_landmarks             # 15 puntos numpy

# Para tests de warp
- warp_input_dataset          # Dataset para warping
- canonical_shape_json        # Forma canonica
- triangles_json              # Triangulacion Delaunay

# Utilidades
- test_image_file             # Imagen PNG temporal
- model_device                # 'cpu' o 'cuda'
```

## Patron de Test Recomendado

```python
class TestPFSAnalysisIntegration:
    """Tests de integracion para comando pfs-analysis."""

    def test_pfs_basic(self, mock_classifier_checkpoint, classification_eval_dataset, tmp_path):
        """PFS con configuracion basica."""
        output_dir = tmp_path / "pfs_output"

        result = runner.invoke(app, [
            'pfs-analysis',
            '--checkpoint', str(mock_classifier_checkpoint),
            '--data-dir', str(classification_eval_dataset),
            '--output-dir', str(output_dir),
            '--device', 'cpu',
            '--approximate',  # Usar mascaras aproximadas
            '--num-samples', '4',
        ])

        # Session 32: Modelo sin entrenar puede fallar pero no crashear
        assert result.exit_code in [0, 1], \
            f"Command crashed (code {result.exit_code}): {result.stdout}"
```

## Meta de la Sesion

| Metrica | Actual | Meta Sesion 32 |
|---------|--------|----------------|
| Tests totales | 501 | 556 (+55) |
| Tests pfs-analysis | 0 | 12 |
| Tests generate-lung-masks | 0 | 10 |
| Tests robustness mejorados | 2 | 10 |
| Cobertura estimada | 40% | 45% |

## Comandos Utiles

```bash
# Ejecutar tests especificos
.venv/bin/python -m pytest tests/test_cli_integration.py -v -k "pfs"

# Ver cobertura
.venv/bin/python -m pytest tests/ --cov=src_v2 --cov-report=term-missing

# Probar comando manualmente
.venv/bin/python -m src_v2 pfs-analysis --help
```

## Notas Importantes

1. **Usar `exit_code in [0, 1]`** para tests con modelos sin entrenar (documentar por que)
2. **Agregar comentarios "Session 32:"** para nuevas assertions
3. **Crear fixtures nuevas** si se necesitan datasets especificos
4. **NO modificar tests existentes** a menos que sea necesario
5. **Ejecutar suite completa** al final para verificar no regresiones

## Inicio Recomendado

1. Leer este prompt completo
2. Revisar `tests/conftest.py` para entender fixtures disponibles
3. Crear `tests/test_pfs_integration.py` con 12 tests
4. Crear `tests/test_lung_masks_integration.py` con 10 tests
5. Agregar tests de robustness a `test_cli_integration.py`
6. Ejecutar suite completa
7. Documentar en `docs/sesiones/SESION_32_TESTS_CRITICOS.md`

---FIN DEL PROMPT---

## Notas para el Usuario

### Como iniciar la proxima sesion:

1. **Abre una nueva conversacion** con Claude
2. **Copia y pega** todo el contenido entre los marcadores
3. **Agrega al final:**
   ```
   Comienza la Sesion 32. Usa ultrathink para planificar los tests
   criticos y crealos sistematicamente. Ejecuta los tests al final
   para verificar que funcionan.
   ```

### Alternativa mas corta:

Si prefieres un inicio mas conciso, usa:

```
Lee el archivo docs/PROMPT_CONTINUACION_SESION_32.md y comienza a trabajar
en la Sesion 32. El objetivo es agregar 55 tests de integracion para
los comandos pfs-analysis, generate-lung-masks y test-robustness.

Estado actual: 501 tests, 20 comandos CLI, 0 bugs criticos.
Meta: 556 tests, 45% cobertura.

Usa ultrathink para planificar sistematicamente.
```

### Que esperar de la Sesion 32:

- **Duracion estimada:** 30-45 minutos
- **Archivos nuevos:** 2 (test_pfs_integration.py, test_lung_masks_integration.py)
- **Archivos modificados:** 1 (test_cli_integration.py)
- **Tests agregados:** ~55
- **Verificacion:** Suite completa debe pasar (556 tests)
