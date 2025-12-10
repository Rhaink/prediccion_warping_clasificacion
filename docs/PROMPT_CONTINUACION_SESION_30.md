# Prompt de Continuacion - Sesion 30

## Instruccion de Inicio

Copia y pega esto al iniciar la conversacion:

```
Lee el archivo docs/PROMPT_CONTINUACION_SESION_30.md y comienza a trabajar
en la Sesion 30. El objetivo principal es corregir los 3 bugs criticos
identificados en la sesion anterior y agregar tests para evaluate-ensemble.

Estado actual: 494 tests, 61% cobertura CLI, 21 comandos funcionando.
Bugs criticos: 3 identificados con soluciones listas.

Usa ultrathink para planificar las correcciones y verificar que no
introducimos regresiones.
```

---

## Contexto del Proyecto

Este es un proyecto de tesis sobre clasificacion de COVID-19 en radiografias de torax usando **normalizacion geometrica (warping)**. El CLI en `src_v2/` permite reproducir todos los experimentos.

**HIPOTESIS DEMOSTRADA (Verificada en Sesion 29 con 4 agentes paralelos):**
- Warping mejora generalizacion **11x** (gap 25.36% → 2.24%)
- Warping mejora robustez **30x** (degradacion JPEG: 16.14% → 0.53%)
- Margen optimo: **1.25** con 96.51% accuracy
- **Datos verificados como AUTENTICOS** (99% confianza)

---

## Estado Actual (Fin Sesion 29)

| Metrica | Valor |
|---------|-------|
| Comandos CLI | 21 funcionando |
| Tests totales | **494 pasando** |
| Cobertura CLI | **61%** |
| Cobertura total | **60%** |
| Rama | feature/restructure-production |

### Logros de Sesion 29

1. **+12 tests nuevos** para train-classifier y evaluate-classifier
2. **Verificacion de autenticidad** de datos con agente especializado
3. **17 bugs identificados** en tests (con soluciones detalladas)
4. **110 tests faltantes** mapeados por prioridad
5. **Introspeccion profunda** con roadmap de proximas sesiones

### Documentacion Generada en Sesion 29

```
docs/sesiones/SESION_29_CLASSIFIER_TESTS.md  # Documentacion tecnica
docs/INTROSPECCION_SESION_29.md              # Analisis estrategico
```

---

## Objetivo de Esta Sesion (30)

### Tarea Principal: Corregir 3 Bugs Criticos

#### Bug #1: mock_classifier_checkpoint falta class_names
**Ubicacion:** `tests/conftest.py` linea ~252

**Problema:** La fixture `mock_classifier_checkpoint` NO incluye `class_names` en el checkpoint, pero el CLI lo espera.

**Solucion:**
```python
# ANTES (incorrecto)
checkpoint = {
    'model_state_dict': model.state_dict(),
    'epoch': 1,
    'val_accuracy': 0.33,
    'config': {
        'backbone': 'resnet18',
        'num_classes': 3,
    }
}

# DESPUES (correcto)
checkpoint = {
    'model_state_dict': model.state_dict(),
    'epoch': 1,
    'val_accuracy': 0.33,
    'class_names': ['COVID', 'Normal', 'Viral_Pneumonia'],  # AGREGAR
    'model_name': 'resnet18',  # AGREGAR
    'best_val_f1': 0.33,  # AGREGAR
    'config': {
        'backbone': 'resnet18',
        'num_classes': 3,
    }
}
```

#### Bug #6: Fixture trained_classifier_for_eval falla silenciosamente
**Ubicacion:** `tests/test_cli_integration.py` linea ~1187-1198

**Problema:** Si el entrenamiento falla, la fixture retorna path a checkpoint inexistente.

**Solucion:**
```python
# ANTES (incorrecto)
result = runner.invoke(app, [...])
checkpoint_path = output_dir / "best_classifier.pt"
return {"checkpoint": checkpoint_path, ...}

# DESPUES (correcto)
result = runner.invoke(app, [...])

# VERIFICAR que el entrenamiento fue exitoso
assert result.exit_code == 0, f"Training failed: {result.stdout}"

checkpoint_path = output_dir / "best_classifier.pt"
assert checkpoint_path.exists(), f"Checkpoint not created: {result.stdout}"

return {"checkpoint": checkpoint_path, ...}
```

#### Bug #7: Tests usan pytest.skip en vez de assert
**Ubicacion:** `tests/test_cli_integration.py` lineas 1213, 1237, 1273, 1311, 1335

**Problema:** Los tests skipean si el checkpoint no existe, ocultando fallos reales.

**Solucion:**
```python
# ANTES (incorrecto)
if not setup["checkpoint"].exists():
    pytest.skip("Checkpoint no fue creado")

# DESPUES (correcto)
assert setup["checkpoint"].exists(), \
    "Checkpoint was not created by fixture. This is a test setup failure."
```

---

### Tarea Secundaria: Agregar Tests para evaluate-ensemble

El comando `evaluate-ensemble` tiene solo 30% de cobertura. Agregar 5-8 tests:

```python
class TestEvaluateEnsembleIntegration:
    """Tests de integracion para evaluate-ensemble."""

    def test_ensemble_two_checkpoints(self, landmark_checkpoints):
        """Ensemble con 2 checkpoints."""

    def test_ensemble_with_tta(self, landmark_checkpoints):
        """Ensemble con TTA habilitado."""

    def test_ensemble_saves_json(self, landmark_checkpoints, tmp_path):
        """Ensemble guarda resultados en JSON."""

    def test_ensemble_outperforms_single(self, landmark_checkpoints):
        """Ensemble mejora sobre modelo individual."""

    def test_ensemble_invalid_checkpoint_fails(self, tmp_path):
        """Ensemble con checkpoint invalido debe fallar."""
```

---

## Archivos Clave

```
tests/conftest.py                          # Fixtures (Bug #1)
tests/test_cli_integration.py              # Tests de integracion (Bugs #6, #7)
src_v2/cli.py                              # CLI principal (~6800 lineas)
docs/sesiones/SESION_29_CLASSIFIER_TESTS.md # Documentacion bugs
docs/INTROSPECCION_SESION_29.md            # Roadmap
```

---

## Comandos Utiles

```bash
# Ejecutar tests especificos
.venv/bin/python -m pytest tests/test_cli_integration.py -v -x -k "classifier"

# Verificar cobertura
.venv/bin/python -m pytest tests/ --cov=src_v2 --cov-report=term

# Probar evaluate-ensemble
.venv/bin/python -m src_v2 evaluate-ensemble --help

# Suite completa
.venv/bin/python -m pytest tests/ -v --tb=short
```

---

## Criterios de Exito

1. **3 bugs corregidos** - Sin regresiones
2. **5+ tests nuevos** para evaluate-ensemble
3. **Tests pasando** - 500+ tests sin regresiones
4. **Cobertura mejorada** - CLI > 63%
5. **Documentacion** - `docs/sesiones/SESION_30_BUG_FIXES.md`

---

## Otros Bugs Identificados (Para Sesiones Futuras)

### Prioridad Media (8 bugs)

| Bug | Problema | Ubicacion |
|-----|----------|-----------|
| #2 | Tolerancia reproducibilidad muy laxa (5%) | test_cli_integration.py:1052 |
| #3 | Early stopping test asume siempre detencion | test_cli_integration.py:1008 |
| #5 | Falta verificar per_class_metrics | test_cli_integration.py:979 |
| #8 | Verificacion JSON structure incompleta | test_cli_integration.py:1307 |
| #10 | per_class no verifica estructura interna | test_cli_integration.py:1373 |
| #11 | Pipeline test no especifica --split | test_cli_integration.py:1438 |
| #12 | Comparacion ignora variabilidad | test_cli_integration.py:1464 |
| #17 | Assertions insuficientes confusion matrix | Multiples lineas |

### Prioridad Baja (6 bugs)

| Bug | Problema |
|-----|----------|
| #4 | Checkpoint test no verifica best_val_f1 |
| #9 | Test de splits no verifica que existen |
| #13 | Test de backbones no verifica coherencia |
| #14 | Codigo duplicado en fixtures |
| #15 | Falta verificacion de race conditions |
| #16 | Tests no limpian recursos |

---

## Gaps de Tests (110 Faltantes)

### Prioridad Critica (60 tests)

| Comando | Tests Faltantes | Cobertura |
|---------|-----------------|-----------|
| `evaluate-ensemble` | 8 | 30% |
| `compare-architectures` | 7 | 20% |
| `optimize-margin` | 8 | 20% |
| `generate-lung-masks` | 6 | 20% |
| `pfs-analysis` | 7 | 40% |

### Prioridad Alta (30 tests)

| Comando | Tests Faltantes | Cobertura |
|---------|-----------------|-----------|
| `cross-evaluate` | 5 | 35% |
| `test-robustness` | 5 | 35% |
| `generate-dataset` | 5 | 45% |
| `gradcam` | 5 | 60% |

---

## Resultados Experimentales de Referencia

```
Margen optimo: 1.25 (96.51% accuracy)
Mejora generalizacion: 11x (95.78% vs 73.45%)
Robustez JPEG Q50: 30x (0.53% vs 16.14% degradacion)
Dataset: 15,153 imagenes (COVID, Normal, Viral_Pneumonia)
Datos verificados: 99% confianza de autenticidad
```

---

## Historial de Sesiones Recientes

- **Sesion 26:** 22 tests integracion para optimize-margin
- **Sesion 27:** +59 tests, analisis exhaustivo
- **Sesion 28:** UX improvements, 4 agentes paralelos
- **Sesion 29:** +12 tests clasificador, 17 bugs identificados, introspeccion

---

## Alternativas de Inicio

### Opcion 1: Solo Bugs (Recomendada)
```
Lee docs/PROMPT_CONTINUACION_SESION_30.md. Enfocate SOLO en corregir
los 3 bugs criticos (#1, #6, #7). Verifica que no hay regresiones
ejecutando la suite completa de tests.
```

### Opcion 2: Bugs + Tests Ensemble
```
Lee docs/PROMPT_CONTINUACION_SESION_30.md. Corrige los 3 bugs criticos
y luego agrega tests para evaluate-ensemble. Meta: 500+ tests.
```

### Opcion 3: Sesion Completa con Analisis
```
Lee docs/PROMPT_CONTINUACION_SESION_30.md. Usa ultrathink para:
1. Corregir bugs criticos
2. Agregar tests evaluate-ensemble
3. Analizar si hay mas bugs en el codigo corregido
```

---

**Ultima actualizacion:** 2025-12-09 (Sesion 29)
