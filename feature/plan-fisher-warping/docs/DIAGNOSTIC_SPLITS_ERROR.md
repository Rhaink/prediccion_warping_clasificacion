# DIAGNÓSTICO: Error Fundamental en Metodología de Splits

**Fecha:** 2026-01-07
**Severidad:** CRÍTICA - Invalida comparación científica 2-class vs 3-class
**Estado:** PENDIENTE DE CORRECCIÓN

---

## RESUMEN EJECUTIVO

Los CSVs `01_full_balanced_3class_*.csv` y `02_full_balanced_2class_*.csv` fueron creados como **datasets DIFERENTES** con imágenes diferentes. Esto hace **científicamente INVÁLIDA** cualquier comparación entre experimentos de 2 clases vs 3 clases.

**Evidencia:**
- CSV 02_* (2-class): 12,402 imágenes
- CSV 01_* (3-class): 6,725 imágenes
- Solapamiento: Solo 5,946 imágenes comunes (47.9%)
- Test sets: Solo 604 de 1,245 imágenes son comunes (48.5%)

**Impacto:**
❌ No podemos decir "2-class funciona mejor que 3-class"
❌ La comparación es entre DATASETS DIFERENTES, no entre esquemas de clasificación
❌ Cualquier diferencia puede ser por los datos, no por el método

---

## ANÁLISIS DETALLADO

### 1. ¿Qué se hizo (INCORRECTO)?

Se crearon DOS CSVs separados con balanceo diferente:

**CSV 01_* (3-class):**
```
Total: 6,725 imágenes
Test: 680 imágenes
Balance: COVID=272, Normal=272, Viral=136 (ratio 2:2:1)
```

**CSV 02_* (2-class):**
```
Total: 12,402 imágenes
Test: 1,245 imágenes
Balance: Enfermo=498, Normal=747 (ratio 1.5:1)
```

**Problema:** Son datasets diferentes. Solo 48.5% de las imágenes de test son comunes.

---

### 2. ¿Qué DEBIÓ haberse hecho?

Se debió crear **UN SOLO CSV** con TODAS las imágenes y AMBOS esquemas de etiquetado:

```csv
image_id,split,path,class_original,label_2class,label_3class
COVID-1,train,/path/to/image,COVID,Enfermo,COVID
COVID-2,val,/path/to/image,COVID,Enfermo,COVID
Normal-1,train,/path/to/image,Normal,Normal,Normal
Viral-1,test,/path/to/image,Viral_Pneumonia,Enfermo,Viral_Pneumonia
...
```

**Ventajas:**
- ✅ MISMAS imágenes exactas en ambos experimentos
- ✅ MISMO split (train/val/test)
- ✅ Comparación JUSTA: solo cambia el esquema de etiquetado
- ✅ Científicamente VÁLIDO

---

### 3. Comparación: Actual vs Correcto

| Aspecto | Actual (INCORRECTO) | Correcto |
|---------|---------------------|----------|
| **Datasets** | 2 CSVs diferentes | 1 CSV único |
| **Imágenes** | Diferentes (47.9% comunes) | 100% las mismas |
| **Split** | Diferentes splits | Mismo split |
| **Test size** | 680 vs 1,245 | Mismo tamaño |
| **Balance** | Diferente en cada CSV | Puede ser diferente pero sobre MISMAS imágenes |
| **Comparación** | ❌ INVÁLIDA | ✅ VÁLIDA |

---

### 4. ¿Por qué pasó esto?

**Hipótesis:**
1. Se crearon los CSVs de forma independiente (probablemente fuera de feature/)
2. Cada CSV se balanceó según su propio criterio
3. No se verificó que fueran las MISMAS imágenes
4. Se asumió que "balanceo correcto" era más importante que "mismas imágenes"

**Error conceptual:**
- Se priorizó el BALANCEO ÓPTIMO de cada esquema
- Se ignoró el requisito de COMPARACIÓN JUSTA
- En ciencia, la comparación requiere **mismas condiciones experimentales**

---

### 5. Objetivo Científico Original

Según `docs/00_OBJETIVOS.md`:

> **Escenarios de Clases**
> 1. **Principal (2 clases)**: Enfermo (COVID + Viral_Pneumonia) vs Normal
> 2. **Secundario (3 clases)**: COVID vs Viral_Pneumonia vs Normal

**Pregunta implícita:** ¿Qué esquema de clasificación funciona mejor?

**Para responder esto válidamente, DEBEN ser las MISMAS imágenes.**

---

## IMPACTO EN RESULTADOS ACTUALES

### Resultados Reportados (INVÁLIDOS como comparación)

**Phase 7 - Comparación 2-class vs 3-class:**
```
2-class (CSV 02_*): 81.69% accuracy (1,245 test)
3-class (CSV 02_*): 80.64% accuracy (1,245 test)
```

**Problema:**
- La 3-class usa CSV 02_* que fue **balanceado para 2 clases**
- Al extraer 3 clases del image_id, el balance queda desigual
- No es una comparación justa

**Alternativa que se intentó:**
```
2-class (CSV 02_*): 81.69% accuracy
3-class (CSV 01_*): ???% accuracy
```

**Problema:**
- Son datasets DIFERENTES
- No podemos comparar 81.69% en un dataset vs X% en otro dataset

---

## SOLUCIÓN REQUERIDA

### Opción 1: Crear CSV Unificado Nuevo (RECOMENDADO)

**Archivo:** `results/metrics/00_unified_split.csv`

**Estructura:**
```csv
image_id,split,path_warped,path_original,class_original,label_2class,label_3class
COVID-1,train,/path/warped.png,/path/original.png,COVID,Enfermo,COVID
Normal-1,train,/path/warped.png,/path/original.png,Normal,Normal,Normal
Viral-1,val,/path/warped.png,/path/original.png,Viral_Pneumonia,Enfermo,Viral_Pneumonia
...
```

**Criterios de creación:**
1. Incluir TODAS las imágenes disponibles (probablemente ~15k)
2. Split 80/10/10 aleatorio con seed fijo
3. NO balancear artificialmente (o balancear pero sobre TODAS las imágenes)
4. Incluir AMBOS esquemas de etiquetado en el mismo CSV

**Ventajas:**
- ✅ Un solo archivo = una sola fuente de verdad
- ✅ Comparación 2-class vs 3-class será VÁLIDA
- ✅ Transparencia total: se ve que son las mismas imágenes

---

### Opción 2: Usar Solo Imágenes Comunes (WORKAROUND)

Si no queremos regenerar todo:

1. Extraer las 604 imágenes comunes en test de ambos CSVs
2. Re-ejecutar experimentos SOLO en ese subset
3. Reportar: "En 604 imágenes comunes: 2-class=X%, 3-class=Y%"

**Ventajas:**
- ✅ Rápido de implementar
- ✅ Comparación justa en ese subset

**Desventajas:**
- ❌ Sample pequeño (604 test en lugar de 1,245)
- ❌ No aprovecha todos los datos disponibles
- ❌ Sigue siendo un workaround, no la solución correcta

---

## PLAN DE CORRECCIÓN PARA PRÓXIMA SESIÓN

### Fase 1: Investigación (30 min)
1. Encontrar el script que generó los CSVs 01_* y 02_*
   - Probablemente está fuera de feature/
   - Buscar en `src/`, `scripts/`, o historial git
2. Entender el protocolo de balanceo usado
3. Identificar TODAS las imágenes disponibles

### Fase 2: Decisión (15 min)
Decidir entre:
- **Opción A:** Crear CSV unificado nuevo (más trabajo, científicamente correcto)
- **Opción B:** Usar subset común (rápido, limitado)

### Fase 3: Implementación (variable)

**Si Opción A (CSV unificado):**
1. Crear script `src/create_unified_split.py`
2. Generar `00_unified_split.csv` con TODAS las imágenes
3. Modificar `data_loader.py` para usar el nuevo CSV
4. Re-ejecutar Phases 4-7 con el CSV unificado
5. Actualizar documentación

**Si Opción B (subset común):**
1. Crear script `src/extract_common_subset.py`
2. Ejecutar experimentos solo en las 604 imágenes comunes
3. Reportar resultados con disclaimer: "subset de 604 imágenes"

### Fase 4: Documentación (30 min)
1. Actualizar `SPLIT_PROTOCOL.md`
2. Explicar el error y la corrección
3. Actualizar `DOCUMENTO_FINAL.md` con metodología correcta

---

## PREGUNTAS PARA EL USUARIO

1. **¿Dónde se crearon originalmente los CSVs 01_* y 02_*?**
   - ¿Hay un script en src/ o fuera de feature/?
   - ¿Hay documentación del proceso?

2. **¿Cuántas imágenes hay disponibles en total?**
   - ¿~15k según 00_OBJETIVOS.md?
   - ¿Todas tienen warped + original?

3. **¿Qué preferimos: Opción A (correcto) u Opción B (rápido)?**
   - Opción A: más trabajo pero científicamente robusto
   - Opción B: workaround rápido pero limitado

4. **¿Es crítico el balanceo de clases?**
   - ¿O podemos trabajar con desbalanceo natural?
   - Muchos papers usan desbalanceo + métricas balanceadas (F1 macro)

---

## LECCIONES APRENDIDAS

1. **Splits ANTES que balanceo**
   - Primero: definir train/val/test FIJO
   - Después: balancear si es necesario

2. **Un CSV es mejor que dos**
   - Si tienes múltiples experimentos, un CSV con múltiples columnas de labels
   - Evita inconsistencias

3. **Verificar assumptions**
   - No asumir que "dos CSVs para dos experimentos" es correcto
   - Verificar que son las MISMAS imágenes

4. **Documentar decisiones**
   - ¿Por qué dos CSVs?
   - ¿Cuál es el criterio de balanceo?
   - ¿Son las mismas imágenes?

---

**Próximos pasos:** Responder las preguntas arriba y decidir Opción A o B.

**Autor:** Claude
**Fecha:** 2026-01-07
