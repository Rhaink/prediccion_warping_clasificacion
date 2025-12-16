# Prompt para Sesión 28: Mejoras de UX y Limpieza de Código

## Instrucciones de Inicio

Copia y pega este prompt completo al iniciar una nueva conversación con Claude.

---

## Contexto del Proyecto

Este es un proyecto de tesis sobre clasificación de COVID-19 en radiografías de tórax usando normalización geométrica (warping). El CLI en `src_v2/` permite reproducir los experimentos.

**HIPÓTESIS DEMOSTRADA (Verificada en Sesión 27):**
- Warping mejora generalización **11x** (95.78% vs 73.45% en cross-evaluation)
- Warping mejora robustez **30x** (0.53% vs 16.14% degradación JPEG Q50)
- Margen óptimo: **1.25** con 96.51% accuracy
- Datos verificados en archivos JSON reales (no inventados)

## Estado Actual (Fin Sesión 27)

| Métrica | Valor |
|---------|-------|
| Comandos CLI | 21 funcionando |
| Tests totales | **482 pasando** |
| Cobertura CLI | **61%** |
| Cobertura total | **60%** |
| Rama | feature/restructure-production |

### Logros de Sesión 27

1. **+59 tests de integración** para todos los comandos CLI
2. **8 fixtures reutilizables** en `conftest.py`
3. **Análisis exhaustivo** con 3 agentes paralelos
4. **Magic numbers extraídos** a `constants.py`
5. **Hipótesis verificada** con archivos fuente identificados

### Archivos de Resultados Verificados

```
outputs/session30_analysis/consolidated_results.json     # 11x generalización
outputs/session29_robustness/artifact_robustness_results.json  # 30x robustez
outputs/session28_margin_experiment/margin_experiment_results.json  # Margen 1.25
```

### Bugs Documentados (Pendientes)

| Bug | Severidad | Ubicación | Descripción |
|-----|-----------|-----------|-------------|
| Hydra falla silenciosa | **ALTA** | `cli.py:296-307` | Config se ignora sin error explícito |
| `--quick` inconsistente | MEDIA | Varios comandos | 3 epochs vs 5 epochs según comando |
| Pesos loss no parametrizables | BAJA | `cli.py:416-417` | `central_weight`, `symmetry_weight` hardcodeados |

### Constantes Ya Agregadas (Sesión 27)

```python
# En src_v2/constants.py
QUICK_MODE_MAX_TRAIN = 500
QUICK_MODE_MAX_VAL = 100
QUICK_MODE_MAX_TEST = 100
QUICK_MODE_EPOCHS_OPTIMIZE = 3
QUICK_MODE_EPOCHS_COMPARE = 5
OPTIMAL_MARGIN_SCALE = 1.25
DEFAULT_MARGIN_SCALE = 1.05
DEFAULT_CENTRAL_WEIGHT = 1.0
DEFAULT_SYMMETRY_WEIGHT = 0.5
```

## Objetivo de Esta Sesión (28)

Mejorar la experiencia de usuario (UX) del CLI y limpiar código pendiente.

### Tareas Principales (Prioridad Alta)

#### 1. Corregir Hydra Fail Silencioso

**Problema:** `cli.py:296-307` - Si Hydra falla, solo se loguea warning y continúa silenciosamente.

**Solución propuesta:**
```python
# Actual (malo)
except Exception as e:
    logger.warning("No se pudo cargar config Hydra: %s", e)

# Propuesto (mejor)
except Exception as e:
    logger.info("Usando valores por defecto (Hydra config no disponible)")
    # O hacer explícito que Hydra es opcional
```

#### 2. Estandarizar `--quick` en Todos los Comandos

**Problema:** Comportamiento inconsistente:
- `optimize-margin --quick`: 3 epochs + 500/100/100 datos
- `compare-architectures --quick`: 5 epochs (sin reducir datos)

**Solución:** Usar constantes de `constants.py` en ambos comandos.

#### 3. Agregar Progress Bars con tqdm

Comandos que se beneficiarían:
- `warp` (procesando imágenes)
- `generate-dataset` (generando splits)
- `train-classifier` (épocas)
- `optimize-margin` (probando márgenes)

#### 4. Mejorar Mensajes de Error

Ejemplos de mejoras:
```python
# Actual
raise typer.Exit(1)

# Propuesto
console.print("[red]Error:[/red] Checkpoint no encontrado: {path}")
console.print("[dim]Hint: Usa 'python -m src_v2 train' para crear uno[/dim]")
raise typer.Exit(1)
```

### Tareas Secundarias (Prioridad Media)

#### 5. Parametrizar Pesos de Loss

Agregar opciones al comando `train`:
```python
--central-weight FLOAT    # Peso para landmarks centrales (default: 1.0)
--symmetry-weight FLOAT   # Peso para penalización de simetría (default: 0.5)
```

#### 6. Agregar `--verbose` Flag Global

Para mostrar información detallada de debug cuando se necesite.

### Tareas Opcionales (Si hay tiempo)

7. **Mejorar fixtures de tests** - Usar imágenes grayscale con estructura
8. **Validar estructura JSON** - No solo existencia, sino contenido
9. **Documentar todos los comandos** - Actualizar README con ejemplos

## Archivos Clave

```
src_v2/cli.py                              # CLI principal (~6800 líneas)
src_v2/constants.py                        # Constantes centralizadas
tests/test_cli_integration.py              # 59 tests de integración
tests/conftest.py                          # Fixtures compartidos
docs/sesiones/SESION_27_CLI_INTEGRATION.md # Documentación sesión anterior
```

## Comandos Útiles

```bash
# Ejecutar tests rápido
.venv/bin/python -m pytest tests/test_cli_integration.py -v -x

# Verificar cobertura
.venv/bin/python -m pytest tests/ --cov=src_v2 --cov-report=term

# Probar comando específico
.venv/bin/python -m src_v2 <comando> --help

# Suite completa
.venv/bin/python -m pytest tests/ -v --tb=short
```

## Criterios de Éxito

1. **Hydra corregido** - Mensaje claro cuando no se usa config
2. **`--quick` estandarizado** - Mismo comportamiento en todos los comandos
3. **Progress bars** - Al menos en 2 comandos largos
4. **Mensajes de error** - Al menos 5 errores con hints útiles
5. **Tests pasando** - 482+ tests sin regresiones
6. **Documentación** - `docs/sesiones/SESION_28_UX_IMPROVEMENTS.md`

## Notas Importantes

1. **NO modificar lógica de entrenamiento** - Solo UX y limpieza
2. **Mantener compatibilidad** - Los comandos existentes deben seguir funcionando
3. **Tests primero** - Verificar que no hay regresiones después de cada cambio
4. **La hipótesis ya está demostrada** - El enfoque es mejorar usabilidad

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

---

## Cómo Iniciar la Conversación

**Opción Recomendada:**

> "Lee este prompt y comienza a trabajar en la Sesión 28. El objetivo principal es mejorar la UX del CLI corrigiendo el bug de Hydra, estandarizando --quick, y agregando progress bars. Usa ultrathink para planificar las tareas."

**Opción Alternativa (si hay límite de contexto):**

> "Continúa el trabajo de la Sesión 27. Estado: 482 tests, 61% cobertura CLI. Objetivo: Mejorar UX - corregir Hydra fail silencioso (cli.py:296-307), estandarizar --quick, agregar progress bars con tqdm."

---

## Alternativa: Sesión de Documentación

Si prefieres una sesión enfocada en documentación en lugar de código:

> "Lee este prompt. En lugar de modificar código, quiero que:
> 1. Actualices el README.md con ejemplos de todos los comandos
> 2. Crees un archivo REPRODUCIBILITY.md con instrucciones paso a paso para reproducir los experimentos
> 3. Documentes la arquitectura del proyecto"

---

**Última actualización:** 2025-12-09 (Sesión 27)
**Autor:** Claude Code + Usuario
