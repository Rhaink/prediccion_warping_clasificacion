# Prompt para Sesión 27: Tests de Integración y UX del CLI

## Instrucciones de Inicio

Copia y pega este prompt completo al iniciar una nueva conversación con Claude.

---

## Contexto del Proyecto

Este es un proyecto de tesis sobre clasificación de COVID-19 en radiografías de tórax usando normalización geométrica (warping). El CLI en `src_v2/` permite reproducir los experimentos.

**HIPÓTESIS DEMOSTRADA (Sesión 25-26):**
- Warping mejora generalización 11x (cross-evaluation)
- Warping mejora robustez 30x (JPEG) y 3x (blur)
- Margen óptimo: 1.25 con 96.51% accuracy

## Estado Actual (Fin Sesión 26)

| Métrica | Valor |
|---------|-------|
| Comandos CLI | 21 funcionando |
| Tests totales | 423 pasando |
| Cobertura promedio | 47% |
| Rama | feature/restructure-production |

### Cobertura por Comando

| Comando | Cobertura | Prioridad |
|---------|-----------|-----------|
| `optimize-margin` | 85% | - (completo) |
| `compare-architectures` | 60% | Media |
| `train` | 30% | **ALTA** |
| `evaluate` | 30% | **ALTA** |
| `predict` | 30% | **ALTA** |
| `warp` | 30% | **ALTA** |

### Bugs Corregidos en Sesión 26

1. Tolerancia en test reproducibilidad (1.0% → 0.1%)
2. Fallback silencioso de imagen negra (agregados warnings)
3. Validación de landmarks (error si 0, warning si ≠ 15)

### Issues Pendientes Identificados

1. **Pesos hardcodeados** en `losses.py:395-411` sin documentación
2. **Integración Hydra** no funcional (código muerto)
3. **Magic numbers** 500/100/100 en quick mode
4. **Discrepancias menores** entre README.md y configs

## Objetivo de Esta Sesión (27)

Aumentar cobertura de tests y mejorar UX del CLI.

### Tareas Principales (Prioridad Alta)

#### 1. Tests de Integración para Comandos Core

Crear `tests/test_cli_integration.py` con tests para:

```python
# Comandos prioritarios (actualmente 30% cobertura)
- train: Test con dataset sintético mínimo
- evaluate: Test con checkpoint mock
- predict: Test con imagen y checkpoint mock
- warp: Test de warping real con fixtures
```

**Modelo a seguir:** `tests/test_optimize_margin_integration.py`

#### 2. Fixtures Reutilizables

Agregar a `tests/conftest.py`:

```python
@pytest.fixture
def mock_landmark_checkpoint(tmp_path):
    """Checkpoint mock de modelo de landmarks."""

@pytest.fixture
def mock_classifier_checkpoint(tmp_path):
    """Checkpoint mock de clasificador."""

@pytest.fixture
def minimal_landmark_dataset(tmp_path):
    """Dataset mínimo para entrenar landmarks."""
```

#### 3. Mejorar UX del CLI (Opcional)

- Agregar progress bars con tqdm a comandos largos
- Mejorar mensajes de error para ser más descriptivos
- Agregar `--verbose` flag donde falte

### Tareas Secundarias (Prioridad Media)

4. **Limpiar código Hydra** - Eliminar o documentar integración no funcional
5. **Extraer magic numbers** a constantes en `constants.py`
6. **Documentar pesos de landmarks** en `losses.py`

## Archivos Clave

```
src_v2/cli.py                              # 21 comandos CLI (~7000 líneas)
tests/test_cli.py                          # Tests smoke existentes
tests/test_optimize_margin_integration.py  # Modelo de tests integración
tests/conftest.py                          # Fixtures compartidos
src_v2/models/losses.py                    # Pesos hardcodeados (línea 395)
```

## Estructura de Tests Sugerida

```python
# tests/test_cli_integration.py

class TestTrainIntegration:
    """Tests de integración para comando train."""

    def test_train_minimal_dataset(self, minimal_landmark_dataset):
        """Train completa con dataset sintético."""

    def test_train_saves_checkpoint(self, minimal_landmark_dataset, tmp_path):
        """Train genera checkpoint válido."""

    def test_train_early_stopping(self, minimal_landmark_dataset):
        """Early stopping funciona correctamente."""


class TestEvaluateIntegration:
    """Tests de integración para comando evaluate."""

    def test_evaluate_with_checkpoint(self, mock_landmark_checkpoint):
        """Evaluate con checkpoint mock."""

    def test_evaluate_outputs_metrics(self, mock_landmark_checkpoint):
        """Evaluate reporta métricas correctamente."""


class TestPredictIntegration:
    """Tests de integración para comando predict."""

    def test_predict_single_image(self, test_image, mock_landmark_checkpoint):
        """Predict retorna landmarks para imagen."""

    def test_predict_saves_visualization(self, test_image, mock_landmark_checkpoint, tmp_path):
        """Predict genera imagen con landmarks."""


class TestWarpIntegration:
    """Tests de integración para comando warp."""

    def test_warp_single_image(self, test_image, canonical_shape_json, triangles_json):
        """Warp procesa imagen correctamente."""

    def test_warp_directory(self, minimal_dataset, canonical_shape_json, triangles_json):
        """Warp procesa directorio completo."""
```

## Criterios de Éxito

1. **Al menos 15 tests de integración nuevos** para comandos core
2. **Cobertura promedio ≥ 55%** (actualmente 47%)
3. **Todos los tests pasando** (435+ tests)
4. **Fixtures reutilizables** en conftest.py
5. **Documentación** en `docs/sesiones/SESION_27_CLI_INTEGRATION.md`

## Comandos Útiles

```bash
# Ejecutar tests específicos
.venv/bin/python -m pytest tests/test_cli_integration.py -v

# Ejecutar con cobertura
.venv/bin/python -m pytest tests/ --cov=src_v2 --cov-report=term-missing

# Verificar comando individual
.venv/bin/python -m src_v2 train --help
.venv/bin/python -m src_v2 evaluate --help

# Suite completa
.venv/bin/python -m pytest tests/ -v --tb=short
```

## Notas Importantes

1. **NO entrenar modelos reales** - usar `--epochs 1` o mocks
2. **Device CPU** para tests - evitar dependencia de GPU
3. **tmp_path** de pytest para archivos temporales
4. **Verificar exit_code == 0** para comandos exitosos

## Resultados Experimentales de Referencia

```
Margen óptimo: 1.25 (96.51% accuracy)
Mejora generalización: 11x (95.78% vs 73.45%)
Robustez JPEG Q50: 30x (0.53% vs 16.14% degradación)
Dataset: 957 imágenes (COVID: 306, Normal: 468, Viral: 183)
```

## Historial de Sesiones Relevantes

- **Sesión 25:** Implementó `optimize-margin`, corrigió 7 bugs, demostró hipótesis
- **Sesión 26:** 22 tests integración, 3 bugs corregidos, análisis exhaustivo

---

## Cómo Iniciar la Conversación

**Opción 1 (Recomendada):** Pega este prompt completo y di:
> "Lee este prompt y comienza a trabajar en la Sesión 27. Utiliza ultrathink y múltiples agentes cuando sea necesario."

**Opción 2 (Si hay límite de contexto):** Pega solo las secciones "Contexto", "Estado Actual" y "Objetivo" y di:
> "Continúa el trabajo de la Sesión 26. El objetivo es aumentar la cobertura de tests de los comandos core (train, evaluate, predict, warp) que actualmente tienen solo 30%."

---

**Última actualización:** 2025-12-09 (Sesión 26)
