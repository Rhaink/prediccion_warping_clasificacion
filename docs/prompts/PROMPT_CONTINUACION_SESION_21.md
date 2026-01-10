# Prompt de Continuacion - Sesion 21: Validacion Funcional y Verificacion de Integridad

**Fecha:** 2025-12-08
**Sesion anterior:** 20 (Implementacion de Comandos de Procesamiento)
**Estado previo:** 14 comandos CLI, 293 tests, 7 arquitecturas de clasificador

## Objetivo de esta Sesion

Validar funcionalmente los nuevos comandos implementados en Sesion 20 y verificar la integridad del proyecto para asegurar que:
1. No hay bugs ni errores ocultos
2. No hay datos inventados o hardcodeados
3. El CLI esta listo para nuevos usuarios
4. Los experimentos son completamente reproducibles

## Contexto de Sesion 20

### Comandos Nuevos Implementados
1. **`compute-canonical`** - Calcula forma canonica de landmarks usando GPA
2. **`generate-dataset`** - Genera dataset warped con splits train/val/test

### Modulos Nuevos
- `src_v2/processing/gpa.py` - Generalized Procrustes Analysis
- `src_v2/processing/warp.py` - Piecewise Affine Warping

### Arquitecturas Nuevas del Clasificador
- ResNet-50, AlexNet, VGG-16, MobileNetV2 (7 total)

## Tareas para Sesion 21

### FASE 1: Validacion Funcional de Comandos Nuevos

#### 1.1 Validar `compute-canonical`
```bash
# Ejecutar con datos reales
python -m src_v2 compute-canonical \
    data/coordenadas/coordenadas_maestro.csv \
    --output-dir outputs/shape_analysis_test \
    --visualize

# Verificaciones:
# 1. ¿Se genera canonical_shape_gpa.json correctamente?
# 2. ¿La forma canonica es similar a outputs/shape_analysis/canonical_shape_gpa.json existente?
# 3. ¿Los triangulos Delaunay son correctos?
# 4. ¿Las visualizaciones se generan sin errores?
```

#### 1.2 Validar `generate-dataset`
```bash
# Ejecutar con subset pequeno primero
python -m src_v2 generate-dataset \
    data/COVID-19_Radiography_Dataset \
    outputs/warped_dataset_test \
    --checkpoint checkpoints_v2/final_model.pt \
    --margin 1.05 \
    --splits 0.75,0.125,0.125 \
    --seed 42

# Verificaciones:
# 1. ¿Se crean los directorios train/val/test?
# 2. ¿Las proporciones de splits son correctas?
# 3. ¿Las imagenes warped se ven correctas visualmente?
# 4. ¿Se genera dataset_summary.json con estadisticas?
# 5. ¿No hay data leakage entre splits?
```

### FASE 2: Verificacion de Integridad del Proyecto

#### 2.1 Verificar No Hay Datos Inventados
- Revisar que todos los outputs se generan de datos reales
- Verificar que metricas reportadas corresponden a evaluaciones reales
- Confirmar que no hay valores hardcodeados en funciones de evaluacion

#### 2.2 Verificar Reproducibilidad
```bash
# Ejecutar tests completos
python -m pytest tests/ -v

# Verificar que seed produce resultados deterministicos
python -m src_v2 generate-dataset ... --seed 42
python -m src_v2 generate-dataset ... --seed 42
# Los resultados deben ser identicos
```

#### 2.3 Verificar Consistencia con Experimentos Originales
- Comparar forma canonica generada vs `outputs/shape_analysis/canonical_shape_gpa.json`
- Verificar que warping produce imagenes similares a `scripts/piecewise_affine_warp.py`

### FASE 3: Revision de Gaps Restantes

Consultar `docs/ANALISIS_GAPS_CLI.md` actualizado:

| Funcionalidad | Estado | Prioridad |
|---------------|--------|-----------|
| `compare-architectures` | Pendiente | Alta |
| `gradcam` | Pendiente | Media |
| `analyze-errors` | Pendiente | Media |
| `optimize-margin` | Pendiente | Baja |

### FASE 4: Preparacion para Nuevos Usuarios

#### 4.1 Verificar Documentacion
- ¿README.md esta actualizado con los 14 comandos?
- ¿Hay ejemplos de uso para cada comando?
- ¿Las dependencias estan documentadas?

#### 4.2 Verificar Facilidad de Uso
```bash
# Un nuevo usuario deberia poder ejecutar:
python -m src_v2 --help
python -m src_v2 compute-canonical --help
python -m src_v2 generate-dataset --help

# Y obtener documentacion clara
```

## Archivos Clave a Consultar

```
docs/sesiones/SESION_20_PROCESSING_COMMANDS.md  # Documentacion sesion anterior
docs/ANALISIS_GAPS_CLI.md                       # Gaps actualizados
src_v2/cli.py                                   # Comandos CLI
src_v2/processing/gpa.py                        # Modulo GPA
src_v2/processing/warp.py                       # Modulo Warp
tests/test_processing.py                        # Tests de procesamiento
outputs/shape_analysis/                         # Datos de referencia
```

## Criterios de Exito para Sesion 21

1. [ ] `compute-canonical` ejecuta sin errores con datos reales
2. [ ] `generate-dataset` genera dataset estructurado correctamente
3. [ ] No se detectan bugs criticos ni datos inventados
4. [ ] Tests siguen pasando (293+)
5. [ ] Documentacion revisada y actualizada si es necesario
6. [ ] Lista clara de gaps restantes para Sesion 22

## Notas Adicionales

- Si se encuentran bugs, documentarlos y corregirlos inmediatamente
- Si hay discrepancias con datos de referencia, investigar la causa
- Priorizar la robustez sobre agregar nuevas funcionalidades
- Documentar cualquier hallazgo importante en sesion markdown

---

**Para comenzar la sesion:** Lee este prompt completo, luego ejecuta los comandos de validacion en orden.
