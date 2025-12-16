# Prompt de Continuacion - Sesion 31

## Instruccion de Inicio

Copia y pega esto al iniciar la conversacion:

```
Lee el archivo docs/PROMPT_CONTINUACION_SESION_31.md y comienza a trabajar
en la Sesion 31. El objetivo principal es corregir los bugs de assertions
debiles y mejorar la robustez de los tests.

Estado actual: 501 tests, 62% cobertura CLI, 21 comandos funcionando.
Bugs a corregir: 7 de alta severidad identificados en Sesion 30.
Hipotesis: CONFIRMADA (warped 11x mejor generalizacion, 30x mas robusto).

Usa ultrathink para planificar las correcciones sistematicamente.
```

---

## Contexto del Proyecto

Este es un proyecto de tesis sobre clasificacion de COVID-19 en radiografias de torax usando **normalizacion geometrica (warping)**. El CLI en `src_v2/` permite reproducir todos los experimentos.

**HIPOTESIS DEMOSTRADA (Verificada en Sesion 30 con datos autenticos):**
- Warping mejora generalizacion **11x** (gap 25.36% → 2.24%)
- Warping mejora robustez **30x** (degradacion JPEG: 16.14% → 0.53%)
- Margen optimo: **1.25** con 96.51% accuracy
- **Datos verificados como AUTENTICOS** (99% confianza, 3 agentes paralelos)

---

## Estado Actual (Fin Sesion 30)

| Metrica | Valor |
|---------|-------|
| Comandos CLI | 21 funcionando |
| Tests totales | **501 pasando** |
| Cobertura CLI | **62%** |
| Cobertura total | **61%** |
| Rama | feature/restructure-production |

### Logros de Sesion 30

1. **3 bugs criticos corregidos** (#1, #6, #7)
2. **+7 tests nuevos** para evaluate-ensemble
3. **Verificacion de autenticidad** de datos (3 agentes paralelos)
4. **15 bugs adicionales identificados** con severidades
5. **75-90 tests faltantes** mapeados por prioridad
6. **Introspeccion profunda** con roadmap estrategico

### Documentacion Generada en Sesion 30

```
docs/sesiones/SESION_30_BUG_FIXES.md      # Documentacion tecnica
docs/INTROSPECCION_SESION_30.md          # Analisis profundo y roadmap
```

---

## Objetivo de Esta Sesion (31)

### Tarea Principal: Corregir Bugs de Assertions Debiles

#### Bug Critico #1: Assertions `exit_code in [0, 1]`

**Problema:** Muchos tests aceptan tanto exit_code=0 como exit_code=1 como "exito", permitiendo que tests pasen cuando comandos fallan.

**Ubicaciones (buscar con grep):**
```bash
grep -n "exit_code in \[0, 1\]" tests/test_cli_integration.py
```

**Lineas identificadas:** ~58, 109, 158, 215, 337, 372, 430, 521, 736, 808, 874, 1125, 1674, 1720, 1765, 1800, 1858, 1937, 1960, 2017

**Solucion:**
```python
# ANTES (permite fallos silenciosos)
assert result.exit_code in [0, 1], f"Unexpected: {result.stdout}"

# DESPUES (explicito sobre expectativa)
# Opcion A: Si el comando DEBE funcionar
assert result.exit_code == 0, f"Command failed: {result.stdout}"

# Opcion B: Si es aceptable que falle por datos sinteticos
# Agregar comentario explicando por que
# Tests con datos sinteticos pueden fallar por dataset muy pequeno
assert result.exit_code in [0, 1], \
    "Command crashed unexpectedly (exit codes 0-1 acceptable for synthetic data)"
```

**Estrategia:**
1. Identificar tests donde el comando DEBE funcionar → cambiar a `== 0`
2. Identificar tests con datos sinteticos donde fallo es aceptable → mantener pero documentar
3. Eliminar assertions completamente permisivas

---

#### Bug Critico #2: JSON Loading Sin Verificacion

**Problema:** Tests hacen `json.load()` sin verificar que el archivo existe o que el comando fue exitoso.

**Ubicaciones:**
```python
# Lineas 178-183, 299, 1238-1246, 1275-1283
if result.exit_code == 0:
    assert output_json.exists()  # OK
    with open(output_json) as f:
        data = json.load(f)
# PERO si exit_code != 0, no hay assertion de fallo!
```

**Solucion:**
```python
# ANTES
if result.exit_code == 0:
    assert output_json.exists()
    with open(output_json) as f:
        data = json.load(f)
    assert 'metrics' in data

# DESPUES
assert result.exit_code == 0, f"Command failed: {result.stdout}"
assert output_json.exists(), f"JSON not created: {result.stdout}"
with open(output_json) as f:
    data = json.load(f)
assert 'metrics' in data, f"Invalid JSON structure: {data.keys()}"
```

---

#### Bug Alto #3: Race Conditions en Fixtures

**Ubicacion:** `tests/conftest.py` lineas 51-53

**Problema:** `batch_landmarks_tensor` usa `np.random.randn()` sin seed fijo.

**Solucion:**
```python
# ANTES
batch[1] += np.random.randn(30) * 0.01
batch[2] += np.random.randn(30) * 0.01
batch[3] += np.random.randn(30) * 0.01

# DESPUES
np.random.seed(42)  # Seed fijo para reproducibilidad
batch[1] += np.random.randn(30) * 0.01
batch[2] += np.random.randn(30) * 0.01
batch[3] += np.random.randn(30) * 0.01
```

---

### Tarea Secundaria: Actualizar Conteo de Comandos

**Ubicacion:** `tests/test_cli.py` linea 178

**Problema:** Test hardcodeado con `len(app.registered_commands) == 20`

**Solucion:**
```python
# ANTES
assert len(app.registered_commands) == 20

# DESPUES
# Verificar que hay comandos registrados sin hardcodear numero
assert len(app.registered_commands) >= 20, \
    f"Expected at least 20 commands, got {len(app.registered_commands)}"
```

---

### Tarea Terciaria: Mejorar Tolerancia de Reproducibilidad

**Ubicacion:** `tests/test_cli_integration.py` linea 1049-1052

**Problema:** Tolerancia de 5% muy alta para reproducibilidad con seed.

**Solucion:**
```python
# ANTES
assert abs(acc1 - acc2) < 0.05  # 5% tolerancia

# DESPUES
assert abs(acc1 - acc2) < 0.01, \  # 1% tolerancia (mas estricto)
    f"Reproducibility failed with same seed: {acc1:.4f} vs {acc2:.4f}"
```

---

## Archivos Clave

```
tests/conftest.py                          # Fixtures (Bug #3)
tests/test_cli_integration.py              # Tests de integracion (Bugs #1, #2)
tests/test_cli.py                          # Tests de unidad (Bug conteo)
docs/INTROSPECCION_SESION_30.md            # Roadmap y bugs completos
docs/sesiones/SESION_30_BUG_FIXES.md       # Documentacion sesion anterior
```

---

## Comandos Utiles

```bash
# Buscar assertions debiles
grep -n "exit_code in \[0, 1\]" tests/test_cli_integration.py | wc -l

# Ejecutar tests especificos
.venv/bin/python -m pytest tests/test_cli_integration.py -v -x --tb=short

# Verificar cobertura
.venv/bin/python -m pytest tests/ --cov=src_v2 --cov-report=term

# Suite completa
.venv/bin/python -m pytest tests/ -v --tb=short -q
```

---

## Criterios de Exito

1. **Assertions corregidas** - Eliminar `exit_code in [0, 1]` innecesarios
2. **JSON loading seguro** - Verificar existencia antes de cargar
3. **Race conditions eliminadas** - Seeds fijos en fixtures
4. **Tests pasando** - 500+ tests sin regresiones
5. **Documentacion** - `docs/sesiones/SESION_31_ASSERTIONS.md`

---

## Bugs Completos Identificados en Sesion 30

### Severidad CRITICA (2)

| ID | Descripcion | Ubicacion |
|----|-------------|-----------|
| C1 | JSON loading sin verificar existencia | test_cli_integration.py:178-183 |
| C2 | Encadenamiento fixtures sin validacion | test_cli_integration.py:1240-1246 |

### Severidad ALTA (5)

| ID | Descripcion | Ubicacion |
|----|-------------|-----------|
| A1 | Race conditions batch_landmarks_tensor | conftest.py:51-53 |
| A2 | Assertions debiles `exit_code in [0,1]` | test_cli_integration.py:58+ |
| A3 | Fixtures checkpoint sin validar estructura | test_cli_integration.py:1187-1202 |
| A4 | Output format validacion insuficiente | test_cli_integration.py:301 |
| A5 | Fixtures duplicadas en multiples lugares | test_cli_integration.py:838+ |

### Severidad MEDIA (6)

| ID | Descripcion | Ubicacion |
|----|-------------|-----------|
| M1 | test_image_file sin verificar guardado | conftest.py:168-189 |
| M2 | Codigo duplicado sin parametrizacion | test_cli_integration.py:217-231 |
| M3 | Hardcoded `len(commands) == 20` | test_cli.py:178 |
| M4 | Memory leak potencial en modelos | conftest.py:93-99 |
| M5 | File I/O fragil sin manejo errores | conftest.py:329-346 |
| M6 | Tolerancia reproducibilidad 5% muy alta | test_cli_integration.py:1049-1052 |

---

## Gaps de Tests (Para Sesiones Futuras)

### Prioridad CRITICA (45 tests faltantes)

| Comando | Tests Faltantes | Cobertura Actual |
|---------|-----------------|------------------|
| `train` | 8 | 60% |
| `warp` | 7 | 50% |
| `evaluate-ensemble` | 5 | 70% |
| `predict` | 5 | 65% |
| `classify` | 4 | 70% |
| `train-classifier` | 6 | 75% |
| `evaluate-classifier` | 4 | 80% |
| `evaluate` | 6 | 65% |

---

## Resultados Experimentales de Referencia

```
Hipotesis: CONFIRMADA
Margen optimo: 1.25 (96.51% accuracy)
Mejora generalizacion: 11x (95.78% vs 73.45%)
Robustez JPEG Q50: 30x (0.53% vs 16.14% degradacion)
Dataset: 15,153 imagenes (COVID, Normal, Viral_Pneumonia)
Datos verificados: 99% confianza de autenticidad
```

---

## Historial de Sesiones Recientes

- **Sesion 27:** +59 tests, analisis exhaustivo
- **Sesion 28:** UX improvements, 4 agentes paralelos
- **Sesion 29:** +12 tests clasificador, 17 bugs identificados
- **Sesion 30:** 3 bugs corregidos, 7 tests ensemble, verificacion datos

---

## Alternativas de Inicio

### Opcion 1: Solo Assertions (Recomendada)
```
Lee docs/PROMPT_CONTINUACION_SESION_31.md. Enfocate SOLO en corregir
las assertions debiles (exit_code in [0,1]). Cuenta cuantas hay,
categoriza cuales deben ser == 0 vs cuales son aceptables, y corrige.
```

### Opcion 2: Assertions + Race Conditions
```
Lee docs/PROMPT_CONTINUACION_SESION_31.md. Corrige las assertions
debiles y las race conditions en fixtures. Asegurate de agregar
seeds fijos donde falten.
```

### Opcion 3: Sesion Completa
```
Lee docs/PROMPT_CONTINUACION_SESION_31.md. Usa ultrathink para:
1. Corregir assertions debiles sistematicamente
2. Arreglar race conditions en fixtures
3. Actualizar hardcoded values
4. Verificar que no hay regresiones
```

---

## Notas Importantes

1. **NO inventar datos** - Los datos experimentales ya estan verificados
2. **Mantener 500+ tests** - No eliminar tests, solo mejorarlos
3. **Documentar cambios** - Agregar comentarios explicando decisiones
4. **Ejecutar suite completa** - Verificar no regresiones despues de cada cambio

---

## Meta de la Sesion 31

| Metrica | Actual | Objetivo |
|---------|--------|----------|
| Tests totales | 501 | 505+ |
| Assertions debiles | ~20 | 0-5 (documentadas) |
| Race conditions | 1+ | 0 |
| Cobertura CLI | 62% | 63%+ |

---

**Ultima actualizacion:** 2025-12-09 (Sesion 30)
