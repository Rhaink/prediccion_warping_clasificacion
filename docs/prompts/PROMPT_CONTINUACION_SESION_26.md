# Prompt de Continuación - Sesión 26

**Fecha:** 2025-12-09
**Sesión anterior:** 25 (optimize-margin + revisión de calidad)
**Objetivo:** Tests de integración y validación con datos reales

---

## PROMPT PARA COPIAR Y PEGAR

```
Lee este prompt y comienza a trabajar en la Sesión 26. Utiliza ultrathink y múltiples agentes cuando sea necesario.

## Contexto del Proyecto

Este es un proyecto de tesis sobre clasificación de COVID-19 en radiografías de tórax usando normalización geométrica (warping). El CLI en `src_v2/` permite reproducir los experimentos.

## Estado Actual (Fin Sesión 25)

- **Comandos CLI:** 20 implementados (incluyendo optimize-margin)
- **Tests:** 401 pasando
- **Cobertura CLI:** ~95%
- **Rama:** feature/restructure-production
- **Bugs corregidos en Sesión 25:** 7 (críticos y altos)

## Hallazgos Clave de Sesión 25

### Hipótesis del Proyecto DEMOSTRADA:
- Warping mejora generalización 11x (cross-evaluation)
- Warping mejora robustez 30x (JPEG) y 3x (blur)
- Trade-off: accuracy puro baja ~4% pero es aceptable clínicamente

### Gaps Identificados:
1. **Tests de integración faltantes** para optimize-margin (cobertura actual ~30%)
2. **41 tests adicionales** identificados como necesarios
3. **Comando optimize-margin** no probado con datos reales del proyecto

## Objetivo de Esta Sesión

Implementar tests de integración prioritarios y validar el comando optimize-margin con datos reales.

### Tareas Principales

1. **Crear tests de integración para optimize-margin** (Prioridad 1):
   - Test de flujo completo con datos mínimos
   - Test de estructura de archivos de salida
   - Test de formato JSON de resultados
   - Tests de edge cases (un solo margen, márgenes duplicados)

2. **Probar optimize-margin con datos reales**:
   - Usar `data/coordenadas/coordenadas_maestro.csv` como landmarks
   - Usar `outputs/shape_analysis/canonical_shape_gpa.json`
   - Ejecutar con modo --quick para validación rápida

3. **Opcional - Mejoras de UX**:
   - Agregar progress bars con tqdm
   - Mejorar mensajes de logging

### Archivos Clave

- CLI principal: `src_v2/cli.py` (optimize-margin en líneas 5838-6620)
- Tests CLI: `tests/test_cli.py`
- Datos de prueba: `data/coordenadas/coordenadas_maestro.csv`
- Canonical shape: `outputs/shape_analysis/canonical_shape_gpa.json`
- Triangulación: `outputs/shape_analysis/canonical_delaunay_triangles.json`

### Estructura de Tests Sugerida

Crear archivo: `tests/test_optimize_margin_integration.py`

```python
# Fixtures para datos mínimos
@pytest.fixture
def minimal_dataset(tmp_path):
    """Dataset mínimo: 3 clases, 5 imágenes cada una"""

@pytest.fixture
def valid_landmarks_csv(tmp_path):
    """CSV con landmarks válidos"""

@pytest.fixture
def canonical_shape_json(tmp_path):
    """JSON con forma canónica válida"""

class TestOptimizeMarginIntegration:
    """Tests de integración del flujo completo"""

class TestOptimizeMarginOutputs:
    """Tests de generación de archivos"""

class TestOptimizeMarginEdgeCases:
    """Tests de edge cases"""
```

### Criterios de Éxito

1. Al menos 10 tests de integración nuevos
2. Todos los tests pasando (411+ tests)
3. Comando validate con datos reales sin errores
4. Documentación en `docs/sesiones/SESION_26_INTEGRATION_TESTS.md`

### Notas Importantes

- El CSV `coordenadas_maestro.csv` NO tiene headers (formato: idx, coords..., image_name)
- Los archivos JSON usan claves como `canonical_shape_normalized` (ya soportado)
- Usar `--quick` para pruebas rápidas (epochs=3, subconjunto de datos)
- Ver `docs/sesiones/SESION_25_OPTIMIZE_MARGIN.md` para bugs ya corregidos
```

---

## Contexto Adicional para el Agente

### Tests de Integración Prioritarios (de análisis Sesión 25)

**P1 - Críticos (implementar primero):**

1. `test_optimize_margin_complete_flow_with_real_data`
   - Crear dataset mínimo temporal
   - Ejecutar con 2-3 márgenes
   - Verificar JSON, CSV, checkpoints generados

2. `test_optimize_margin_output_files_structure`
   - Verificar `margin_optimization_results.json`
   - Verificar `summary.csv`
   - Verificar `per_margin/{margin}/checkpoint.pt`

3. `test_optimize_margin_json_results_format`
   - Verificar campos: timestamp, configuration, results, best_margin
   - Verificar que best_accuracy coincide con max(results)

4. `test_optimize_margin_single_margin`
   - Ejecutar con un solo margen
   - Verificar comportamiento correcto

5. `test_optimize_margin_quick_mode`
   - Verificar epochs reducidos a 3
   - Verificar subconjunto de datos usado

### Formato del CSV de Landmarks (coordenadas_maestro.csv)

```
# Sin headers, formato:
# idx, L1_x, L1_y, L2_x, L2_y, ..., L15_x, L15_y, image_name
0,145,56,150,229,65,98,238,93,51,145,252,140,51,187,257,182,145,98,150,140,150,187,107,56,182,51,42,234,276,224,COVID-269
```

El comando optimize-margin espera columnas como `L1_x, L1_y, ...` y columna `category` o `class`.
Puede ser necesario preprocesar o adaptar el código para este formato.

### Estructura del Proyecto

```
prediccion_warping_clasificacion/
├── src_v2/
│   ├── cli.py              # 20 comandos CLI
│   ├── models/             # Clasificadores y landmarks
│   └── processing/         # Warping y GPA
├── tests/
│   ├── test_cli.py         # 102 tests CLI
│   └── test_*.py           # Otros tests
├── data/
│   ├── coordenadas/        # coordenadas_maestro.csv
│   └── dataset/            # COVID-19_Radiography_Dataset
├── outputs/
│   ├── shape_analysis/     # canonical_shape_gpa.json
│   └── session28_margin_experiment/  # Resultados históricos
└── docs/
    └── sesiones/           # Documentación por sesión
```

### Resultados Esperados del Experimento de Margen

Basado en `outputs/session28_margin_experiment/margin_experiment_results.json`:

| Margen | Test Accuracy |
|--------|---------------|
| 1.05   | 94.40% |
| 1.10   | 96.25% |
| 1.15   | 95.06% |
| 1.20   | 96.11% |
| **1.25** | **96.51%** |
| 1.30   | 96.31% |

El margen óptimo histórico es **1.25**.

---

## Historial de Sesiones Relevantes

- **Sesión 20:** Implementó `generate-dataset` y `compute-canonical`
- **Sesión 22:** Implementó `compare-architectures`
- **Sesión 23:** Implementó `gradcam` y `analyze-errors`
- **Sesión 24:** Implementó `pfs-analysis`
- **Sesión 25:** Implementó `optimize-margin` + corrección de 7 bugs

---

## Recordatorios

1. El proyecto usa Python 3.12 con virtualenv en `.venv/`
2. Ejecutar tests con: `.venv/bin/python -m pytest tests/ -v`
3. Los comandos CLI se ejecutan con: `.venv/bin/python -m src_v2 <comando>`
4. La rama actual es `feature/restructure-production`

---

**Última actualización:** 2025-12-09 (Sesión 25)
