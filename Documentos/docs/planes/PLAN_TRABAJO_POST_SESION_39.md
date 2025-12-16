# PLAN DE TRABAJO POST-SESION 39

**Fecha:** 10 Diciembre 2025
**Deadline Tesis:** < 1 mes
**Estado Proyecto:** ~92% completo

---

## RESUMEN DE HALLAZGOS SESION 39

El experimento de control reveló que la robustez tiene **DOS componentes**:
- **~75%: Reducción de información** (47% fill rate = regularización implícita)
- **~25%: Normalización geométrica** (alineación anatómica adicional)

---

## TAREAS PRIORITARIAS (ANTES DE DEFENSA)

### PRIORIDAD 1: Cross-Evaluation Válido [~3 horas]

**Problema:** Cross-evaluation actual compara 4 clases vs 3 clases = INVÁLIDO

**Solución:**
```bash
# Paso 1: Crear script para filtrar dataset
# scripts/filter_dataset_3_classes.py

# Paso 2: Entrenar modelo original con 3 clases
.venv/bin/python -m src_v2 train-classifier \
    outputs/original_3_classes \
    --output-dir outputs/classifier_original_3classes \
    --epochs 50

# Paso 3: Cross-evaluation justo
.venv/bin/python -m src_v2 cross-evaluate \
    outputs/classifier_original_3classes/best_classifier.pt \
    outputs/classifier_warped_full_coverage/best_classifier.pt \
    --data-a outputs/original_3_classes \
    --data-b outputs/full_coverage_warped_dataset \
    --output-dir outputs/cross_evaluation_valid_3classes
```

**Resultado esperado:** Gap de generalización medido correctamente

---

### PRIORIDAD 2: Actualizar Documentación [~2 horas]

**Archivo principal:** `docs/RESULTADOS_EXPERIMENTALES_v2.md`

**Cambios requeridos:**

1. **Agregar Sección: Experimento de Control**
   - Tabla de robustez con 4 datasets
   - Interpretación de mecanismos (75% info + 25% geo)

2. **Corregir Claims:**
   - ❌ "11x mejor generalización" → Pendiente validación
   - ✓ "30x robustez JPEG" → Agregar análisis causal
   - ❌ "Elimina marcas" → Reformular a "excluye/recorta"
   - ❌ "Fuerza atención pulmonar" → PFS invalidado

3. **Nueva Tabla Principal:**
```
| Modelo | Fill Rate | JPEG Q50 deg | JPEG Q30 deg | Blur σ1 deg |
|--------|-----------|--------------|--------------|-------------|
| Original 100% | 100% | 16.14% | 29.97% | 14.43% |
| Original Cropped 47% | 47% | 2.11% | 7.65% | 7.65% |
| Warped 47% | 47% | 0.53% | 1.32% | 6.06% |
| Warped 99% | 99% | 7.34% | 16.73% | 11.35% |
```

---

### PRIORIDAD 3: Recalcular PFS con Máscaras Warped [~2 horas]

**Problema:** PFS actual calculado con máscaras no transformadas = INVÁLIDO

**Solución:**
```python
# Ya existe warp_mask() en src_v2/processing/warp.py
from src_v2.processing.warp import warp_mask

# Crear script: scripts/calculate_pfs_warped.py
warped_mask = warp_mask(mask, source_lm, target_lm, use_full_coverage=True)
pfs = calculate_pfs(warped_img, warped_mask)
```

**Resultado esperado:** PFS válido o confirmación de que no hay diferencia significativa

---

### PRIORIDAD 4: Tests Críticos Faltantes [~4 horas]

**Tests que implementar:**

1. **test_warp_mask_consistency.py**
   - Verificar que imagen y máscara se transforman igual
   - CRÍTICO para validez científica

2. **test_fill_rate_full_coverage.py**
   - Verificar fill rate >= 96% en dataset full_coverage
   - Regresión para boundary points

3. **test_cross_evaluation_classes.py**
   - Validar que cross-eval usa mismas clases
   - Prevenir error metodológico futuro

4. **test_robustness_comparative.py**
   - Automatizar comparación warped vs original
   - CI/CD para reproducibilidad

---

## TAREAS SECUNDARIAS (POST-DEFENSA)

### Mejoras CLI

1. **Agregar comando `filter-dataset-classes`**
```bash
.venv/bin/python -m src_v2 filter-dataset-classes \
    --data-dir data/COVID-19_Radiography_Dataset \
    --output-dir outputs/original_3_classes \
    --exclude Lung_Opacity
```

2. **Agregar flag `--verbose` global**
   - Debugging sin modificar código

3. **Agregar validación de clases en cross-evaluate**
   - Prevenir comparaciones inválidas

### Limpieza de Código

1. Eliminar código duplicado `triangle_area_2x`
2. Remover dependencias no usadas (hydra-core, omegaconf)
3. Agregar seaborn a requirements.txt

---

## CRONOGRAMA SUGERIDO

| Día | Tarea | Horas |
|-----|-------|-------|
| 1 | Cross-evaluation válido (filtrar + entrenar + evaluar) | 3h |
| 2 | Actualizar documentación principal | 2h |
| 2 | Recalcular PFS con máscaras warped | 2h |
| 3 | Implementar tests críticos | 4h |
| 4 | Revisión final y commit | 2h |

**Total estimado:** 13 horas (~2-3 días de trabajo)

---

## ESTADO DE CLAIMS

### Claims VÁLIDOS (pueden usarse)
- ✅ "Robustez 30x superior a JPEG" (con explicación causal)
- ✅ "Robustez 3x superior a blur"
- ✅ "Trade-off cuantificable info vs robustez"
- ✅ "Pipeline reproducible y bien testeado"

### Claims que REQUIEREN reformulación
- ⚠️ "Normalización mejora robustez" → "contribuye ~25% adicional"
- ⚠️ "Elimina marcas hospitalarias" → "excluye/recorta marcas"

### Claims INVÁLIDOS (no usar hasta corregir)
- ❌ "Generaliza 11x mejor" (cross-eval inválido)
- ❌ "Fuerza atención pulmonar" (PFS sin máscaras warped)

---

## ARCHIVOS CLAVE

### Para Modificar
```
docs/RESULTADOS_EXPERIMENTALES_v2.md  # Actualizar con Sesión 39
src_v2/cli.py                          # Agregar validaciones
tests/                                  # Nuevos tests críticos
```

### Resultados Nuevos
```
outputs/robustness_original_cropped_47.json  # Experimento control
outputs/original_cropped_47/                  # Dataset control
outputs/classifier_original_cropped_47/       # Modelo control
```

### Scripts a Crear
```
scripts/filter_dataset_3_classes.py    # Filtrar a 3 clases
scripts/calculate_pfs_warped.py        # PFS con máscaras warped
```

---

## CHECKLIST FINAL ANTES DE DEFENSA

- [x] Cross-evaluation con 3 clases consistentes ejecutado **COMPLETADO - 2.4x mejor generalizacion**
- [x] Documentación actualizada con resultados Sesión 39 **COMPLETADO**
- [x] Claims reformulados según evidencia **COMPLETADO**
- [x] PFS recalculado con mascaras warped **COMPLETADO - PFS ~0.49 (NO fuerza atencion pulmonar)**
- [x] 4 tests críticos implementados y pasando **COMPLETADO - 23 tests pasando**
- [ ] README actualizado con instrucciones de reproducción
- [ ] Commit final con tag de versión

---

**Progreso:**
- Cross-evaluation válido: **2.4x mejor generalización** (no 11x)
- PFS con máscaras warped: **~0.49** (claim de "fuerza atención pulmonar" INVALIDADO)

**Próximo paso:** Implementar tests críticos faltantes.
