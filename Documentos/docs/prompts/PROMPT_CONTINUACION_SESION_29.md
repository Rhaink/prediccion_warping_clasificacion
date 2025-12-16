# Prompt para Sesión 29: Tests de Integración para Clasificación

## Contexto del Proyecto

Este es un proyecto de tesis sobre clasificación de COVID-19 en radiografías de tórax usando **normalización geométrica (warping)**. El CLI en `src_v2/` permite reproducir todos los experimentos.

**HIPÓTESIS DEMOSTRADA (Verificada con archivos JSON reales):**
- Warping mejora generalización **11x** (gap 25.36% → 2.24%)
- Warping mejora robustez **30x** (degradación JPEG: 16.14% → 0.53%)
- Margen óptimo: **1.25** con 96.51% accuracy

## Estado Actual (Fin Sesión 28)

| Métrica | Valor |
|---------|-------|
| Comandos CLI | 21 funcionando |
| Tests totales | **482 pasando** |
| Cobertura CLI | **61%** |
| Cobertura total | **60%** |
| Rama | feature/restructure-production |

### Logros de Sesión 28

1. **Hydra corregido** - Warning → Info con mensaje claro
2. **--quick estandarizado** - Ambos comandos usan `min(epochs, CONSTANT)`
3. **Progress bars** - 3 loops con tqdm (compare-architectures x2, optimize-margin)
4. **7+ hints** - Mensajes de error informativos
5. **Datos verificados** - 4 agentes paralelos confirmaron que TODOS los datos son reales
6. **1 bug adicional corregido** - Inconsistencia en asignación de epochs

### Archivos de Resultados Verificados

```
outputs/session30_analysis/consolidated_results.json     # 11x generalización
outputs/session29_robustness/artifact_robustness_results.json  # 30x robustez
outputs/session28_margin_experiment/margin_experiment_results.json  # Margen 1.25
```

### Gaps de Tests Identificados (Sesión 28)

El análisis con agentes paralelos identificó los siguientes gaps:

| Prioridad | Comando | Estado Actual | Tests Faltantes |
|-----------|---------|---------------|-----------------|
| **CRÍTICO** | `train-classifier` | 3 tests básicos | Execution completo, reproducibilidad, early stopping |
| **CRÍTICO** | `evaluate-classifier` | 3 tests básicos | Métricas reales, confusion matrix, JSON output |
| **ALTO** | `test-robustness` | 2 tests validación | Perturbaciones individuales (JPEG, blur, noise) |
| **MEDIO** | `cross-evaluate` | 2 tests validación | Comparación simétrica, output JSON |
| **MEDIO** | `generate-dataset` | 2 tests validación | Estructura de salida, metadata |
| **BAJO** | `gradcam` | 2 tests validación | Generación de heatmaps |

### Comandos con Buena Cobertura (No tocar)

- `train` (6 tests), `evaluate` (7 tests), `predict` (7 tests), `warp` (8 tests)
- `optimize-margin` (22 tests - excelente)
- `compare-architectures` (15 tests smoke)

## Objetivo de Esta Sesión (29)

**Agregar tests de integración para comandos de clasificación** que actualmente solo tienen smoke tests.

### Tarea Principal: Tests para train-classifier

**Ubicación:** `tests/test_cli_integration.py`

**Tests a agregar:**

```python
class TestTrainClassifierIntegration:
    def test_train_classifier_creates_checkpoint(self, classifier_dataset):
        """Verificar que train-classifier crea checkpoint."""

    def test_train_classifier_saves_metrics_json(self, classifier_dataset):
        """Verificar que guarda métricas en JSON."""

    def test_train_classifier_early_stopping_works(self, classifier_dataset):
        """Verificar que early stopping funciona con patience."""

    def test_train_classifier_different_backbones(self, classifier_dataset):
        """Verificar que soporta diferentes arquitecturas."""

    def test_train_classifier_reproducibility_with_seed(self, classifier_dataset):
        """Verificar reproducibilidad con --seed."""
```

### Tarea Secundaria: Tests para evaluate-classifier

```python
class TestEvaluateClassifierIntegration:
    def test_evaluate_classifier_computes_accuracy(self, trained_classifier):
        """Verificar que calcula accuracy correctamente."""

    def test_evaluate_classifier_outputs_confusion_matrix(self, trained_classifier):
        """Verificar que genera confusion matrix."""

    def test_evaluate_classifier_saves_json_report(self, trained_classifier):
        """Verificar que guarda reporte JSON."""
```

### Tarea Opcional: Tests para test-robustness

```python
class TestRobustnessIntegration:
    def test_robustness_jpeg_perturbation(self, trained_classifier):
        """Verificar perturbación JPEG."""

    def test_robustness_blur_perturbation(self, trained_classifier):
        """Verificar perturbación blur."""

    def test_robustness_outputs_comparison(self, trained_classifier):
        """Verificar que genera comparación."""
```

## Fixtures Necesarias

Las fixtures ya existen en `tests/conftest.py`:

```python
# Ya disponibles:
@pytest.fixture
def sample_image()  # Imagen de prueba

@pytest.fixture
def landmark_checkpoint()  # Checkpoint de landmarks

@pytest.fixture
def classifier_dataset()  # Dataset con estructura train/val/test

# Posiblemente necesaria:
@pytest.fixture
def trained_classifier(classifier_dataset):
    """Entrenar un clasificador mínimo para tests."""
    # Entrenar con 1-2 epochs
    # Retornar path al checkpoint
```

## Archivos Clave

```
src_v2/cli.py                              # CLI principal (~6800 líneas)
tests/test_cli_integration.py              # Tests de integración (agregar aquí)
tests/conftest.py                          # Fixtures compartidos
tests/test_optimize_margin_integration.py  # Ejemplo de buenos tests
docs/sesiones/SESION_28_UX_IMPROVEMENTS.md # Documentación sesión anterior
```

## Comandos Útiles

```bash
# Ejecutar tests específicos
.venv/bin/python -m pytest tests/test_cli_integration.py -v -x -k "classifier"

# Verificar cobertura
.venv/bin/python -m pytest tests/ --cov=src_v2 --cov-report=term

# Probar comando específico
.venv/bin/python -m src_v2 train-classifier --help

# Suite completa
.venv/bin/python -m pytest tests/ -v --tb=short
```

## Criterios de Éxito

1. **5+ tests nuevos** para `train-classifier`
2. **3+ tests nuevos** para `evaluate-classifier`
3. **Tests pasando** - 490+ tests sin regresiones
4. **Cobertura mejorada** - CLI > 65%
5. **Documentación** - `docs/sesiones/SESION_29_CLASSIFIER_TESTS.md`

## Patrón de Test Recomendado

Usar el patrón de `test_optimize_margin_integration.py` como referencia:

```python
def test_train_classifier_creates_checkpoint(self, classifier_dataset, tmp_path):
    """Test que train-classifier crea checkpoint correctamente."""
    runner = CliRunner()
    output_dir = tmp_path / "classifier_output"

    result = runner.invoke(app, [
        "train-classifier",
        str(classifier_dataset),
        "--output-dir", str(output_dir),
        "--epochs", "2",  # Mínimo para test rápido
        "--backbone", "resnet18",
        "--seed", "42",
    ])

    # Verificar ejecución
    assert result.exit_code == 0 or "error" not in result.stdout.lower()

    # Verificar outputs
    checkpoint = output_dir / "best_classifier.pt"
    assert checkpoint.exists(), f"Checkpoint no creado: {result.stdout}"

    # Verificar estructura del checkpoint
    import torch
    ckpt = torch.load(checkpoint, map_location="cpu")
    assert "model_state_dict" in ckpt
    assert "class_names" in ckpt
```

## Notas Importantes

1. **NO modificar lógica de entrenamiento** - Solo agregar tests
2. **Mantener compatibilidad** - Los comandos existentes deben seguir funcionando
3. **Tests rápidos** - Usar epochs mínimos (1-2) y datasets pequeños
4. **La hipótesis ya está demostrada** - El enfoque es mejorar cobertura de tests

## Resultados Experimentales de Referencia

```
Margen óptimo: 1.25 (96.51% accuracy)
Mejora generalización: 11x (95.78% vs 73.45%)
Robustez JPEG Q50: 30x (0.53% vs 16.14% degradación)
Dataset: 957 imágenes (COVID: 306, Normal: 468, Viral: 183)
```

## Historial de Sesiones Relevantes

- **Sesión 25:** Implementó `optimize-margin`, corrigió 7 bugs
- **Sesión 26:** 22 tests integración para optimize-margin
- **Sesión 27:** +59 tests, análisis exhaustivo, verificó hipótesis
- **Sesión 28:** UX improvements, 4 agentes paralelos, gaps identificados

---

## Cómo Iniciar la Conversación

**Opción Recomendada:**

> "Lee este prompt y comienza a trabajar en la Sesión 29. El objetivo principal es agregar tests de integración para `train-classifier` y `evaluate-classifier`. Usa ultrathink para planificar y verificar que los tests cubren los casos importantes."

**Opción Alternativa (si hay límite de contexto):**

> "Continúa el trabajo de la Sesión 28. Estado: 482 tests, 61% cobertura CLI. Objetivo: Agregar 5+ tests para train-classifier y 3+ tests para evaluate-classifier. Los gaps fueron identificados con análisis de 4 agentes paralelos."

---

## Alternativa: Sesión de Parametrización

Si prefieres enfocarte en UX en lugar de tests:

> "Lee este prompt. En lugar de agregar tests, quiero que:
> 1. Parametrices los pesos de loss (`--central-weight`, `--symmetry-weight`)
> 2. Agregues flag `--verbose` global
> 3. Mejores los mensajes de progreso durante entrenamiento"

---

**Última actualización:** 2025-12-09 (Sesión 28)
