# Sesion 35: Analisis Critico y Hallazgos

**Fecha:** 2025-12-10
**Objetivo:** Tests de integracion CLI + Analisis profundo de validez experimental

---

## Resumen Ejecutivo

### Trabajo Completado
1. **5 tests de integracion para classify** agregados (5→10 tests)
2. **Codigo Hydra muerto eliminado** (-314 lineas, -36 tests obsoletos)
3. **Analisis profundo con 5 agentes** ejecutado

### Hallazgos Criticos

| Area | Estado | Severidad |
|------|--------|-----------|
| CLI | OK | - |
| Datos experimentales | VERIFICADOS | - |
| Codigo warping | BUGS | ALTA |
| GradCAM/PFS | CUESTIONABLE | MEDIA |
| **Metodologia cross-eval** | **SESGO CRITICO** | **CRITICA** |
| Robustez JPEG/Blur | VALIDO | - |

---

## 1. Commit de la Sesion

```
4e3d604 refactor: eliminar codigo Hydra muerto y agregar tests classify
```

**Cambios:**
- +5 tests classify (directory_batch, clahe, batch_size, output_fields, margin_scale)
- Eliminado `setup_hydra_config()` de cli.py
- Eliminado directorio `src_v2/conf/`
- Eliminado `tests/test_config.py` (36 tests obsoletos)
- Actualizado test_train_help para reflejar cambios

**Metricas:**
- Tests totales: 556 → 520 (-36 obsoletos)
- Tests integracion CLI: 97 → 102 (+5)
- Cobertura comandos CLI: 75% (15/20)

---

## 2. Hallazgos del Analisis Profundo

### 2.1 Bugs en Codigo de Warping

**Ubicacion:** `src_v2/cli.py` lineas 1025 y 1684

**Problema:** Los comandos `warp` y `classify --warp` usan la version ANTIGUA
del warping (`scripts/piecewise_affine_warp.py`) que tiene:
- Metodo de deteccion de triangulos degenerados **incorrecto**
- Calculo de determinante en matriz 3x3 malformada

**Solucion requerida:**
```python
# ANTES (cli.py linea 1025)
from scripts.piecewise_affine_warp import piecewise_affine_warp

# DESPUES
from src_v2.processing.warp import piecewise_affine_warp
```

**Nota:** Los comandos `generate-dataset` y `optimize-margin` SI usan la version
correcta (`src_v2/processing/warp.py`).

### 2.2 GradCAM y Pulmonary Focus Score (PFS)

**Hallazgo critico:**

| Dataset | PFS Promedio | Interpretacion |
|---------|--------------|----------------|
| Original | 35.5% | Modelo mira mayormente FUERA de pulmones |
| Warped | 35.6% | Similar al original |
| Diferencia | +0.1% (p=0.856) | NO significativa |

**Implicaciones:**
- Ambos modelos tienen PFS bajo (~35%)
- Los modelos NO enfocan principalmente en tejido pulmonar
- Las mascaras pulmonares NO estan warped → comparacion PFS invalida para warped
- **La hipotesis "warped fuerza atencion en pulmones" NO esta validada**

**Documentacion:** `/docs/TRABAJO_FUTURO_PFS.md`

### 2.3 SESGO METODOLOGICO CRITICO en Cross-Evaluation

**Este es el hallazgo mas importante de la sesion:**

#### Problema

Las imagenes warped tienen **informacion asimetrica** respecto a las originales:

| Caracteristica | Original | Warped |
|----------------|----------|--------|
| Fill rate | ~100% | ~47% |
| Fondo negro | 0% | ~53% |
| Marcas hospitalarias | SI | NO |
| Bordes/esquinas | SI | NO |

#### Consecuencia

El gap de generalizacion 25.36% (original→warped) NO indica:
> "El modelo original sobreajusta a caracteristicas geometricas irrelevantes"

En realidad indica:
> "El modelo original aprendio features que NO EXISTEN en imagenes warped"

Es comparable a evaluar un modelo RGB sobre imagenes grayscale.

#### Validez de Experimentos

| Experimento | Validez | Razon |
|-------------|---------|-------|
| Cross-evaluation | INVALIDO | Informacion asimetrica |
| Robustez JPEG | VALIDO | Mismo dataset, perturbacion simetrica |
| Robustez Blur | VALIDO | Mismo dataset, perturbacion simetrica |
| Evaluacion externa | VALIDO | Mide domain shift real |

### 2.4 Datos Experimentales: VERIFICADOS

Los 5 agentes confirmaron que los datos son **REALES y VERIFICABLES**:

- 15+ verificaciones matematicas perfectas
- 4 matrices de confusion coherentes
- Dataset fisico verificado (15,153 imagenes)
- Checkpoints reales (600+ MB totales)
- Scripts reproducibles documentados
- Timestamps cronologicamente coherentes

**Gap calculado vs reportado:**
- Original: 98.81% - 73.45% = 25.36% (diferencia: 0.000000%)
- Warped: 98.02% - 95.78% = 2.24% (diferencia: 0.000000%)

**Robustez JPEG verificada:**
- Original: 98.81% - 82.67% = 16.14% degradacion
- Warped: 98.02% - 97.50% = 0.53% degradacion
- Ratio: 30.62x (correcto)

---

## 3. Recomendaciones

### 3.1 Bugs a Corregir (ALTA PRIORIDAD)

1. **Actualizar imports en cli.py:**
   - Linea 1025: Cambiar a `from src_v2.processing.warp import ...`
   - Linea 1684: Mismo cambio

2. **Mejorar fill_rate:**
   ```python
   # Actual (solo == 0)
   black_pixels = np.sum(warped_image == 0)

   # Recomendado (umbral robusto)
   black_pixels = np.sum(warped_image < 5)
   ```

### 3.2 Metodologia (CRITICA)

**Para publicacion/tesis, reformular claims:**

ANTES:
> "Warped generaliza 11x mejor (gap 25.36% → 2.24%)"

DESPUES:
> "Modelos con normalizacion geometrica son 30x mas robustos a JPEG
> y 3x mas robustos a blur, con trade-off de -0.8% accuracy interno.
> La normalizacion NO resuelve domain shift externo (~55% en ambos)."

### 3.3 Experimentos Adicionales Sugeridos

1. **Comparacion justa con misma informacion:**
   - Generar dataset original croppeado al bounding box pulmonar
   - Comparar con warped (misma cantidad de informacion)

2. **Usar full_coverage=True en warping:**
   - Generaria fill rate ~96% vs ~47%
   - Mantendria mas informacion

3. **Warpear mascaras pulmonares:**
   - Permitiria calculo PFS valido para warped
   - Validaria/invalidaria hipotesis de atencion pulmonar

---

## 4. Estado del Proyecto Post-Sesion 35

### Metricas Actualizadas

| Metrica | Valor |
|---------|-------|
| Tests totales | 520 |
| Tests integracion CLI | 102 |
| Cobertura comandos | 75% (15/20) |
| Commits en branch | 10 |

### Tareas Pendientes Criticas

1. [ ] Corregir bugs de import en warping
2. [ ] Implementar warp de mascaras pulmonares
3. [ ] Ejecutar experimento con full_coverage=True
4. [ ] Reformular claims de generalizacion en documentacion

### Lo que SI esta validado

- Robustez JPEG 30x mejor
- Robustez Blur 3x mejor
- Datos experimentales son reales
- Pipeline CLI funciona correctamente
- Tests tienen buena cobertura

### Lo que NO esta completamente validado

- "Generalizacion 11x mejor" (sesgo metodologico)
- "Modelo warped enfoca en pulmones" (PFS bajo en ambos)
- Comparacion cross-evaluation (informacion asimetrica)

---

## 5. Proximos Pasos Sugeridos

### Sesion 36 (Inmediato)

1. Corregir bugs de import en cli.py
2. Agregar tests para verificar version correcta de warping
3. Crear script para generar dataset con full_coverage=True

### Sesion 37+ (Medio plazo)

1. Implementar warp_mask() para mascaras pulmonares
2. Recalcular PFS con mascaras correctamente transformadas
3. Ejecutar cross-evaluation con informacion simetrica
4. Actualizar documentacion de tesis con claims corregidos

---

## Conclusion

La Sesion 35 logro sus objetivos de testing pero revelo **problemas metodologicos
criticos** que requieren atencion antes de publicar resultados:

1. **Los datos son reales** - No hay fabricacion de resultados
2. **El codigo tiene bugs** - Comandos usan version antigua de warping
3. **La metodologia tiene sesgos** - Cross-evaluation invalida
4. **Las conclusiones deben reformularse** - Enfocarse en robustez, no generalizacion

El proyecto esta ~85% completo pero necesita ajustes en la narrativa cientifica.
