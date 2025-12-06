# RESUMEN EJECUTIVO - Revisión 07_ensemble_tta.tex

## Estado General: ⚠️ INCONSISTENCIAS CRÍTICAS

El documento presenta **errores de nomenclatura y confusión entre configuraciones**, aunque la implementación técnica es correcta.

---

## PROBLEMAS CRÍTICOS (DEBEN CORREGIRSE)

### 1. ❌ Confusión "2 vs 3 vs 4 modelos"

**Problema:**
- **Abstract:** "ensemble de 2 modelos = 3.71 px"
- **Tabla Sesión 12:** "ensemble de 2 modelos = 3.79 px"
- **SESSION_LOG:** "4 modelos = 3.71 px"

**Realidad:**
- Sesión 12: 2 modelos (seed123 + seed456) = **3.79 px**
- Sesión 13: 4 modelos (seed123 + seed456 + seed321 + seed789) = **3.71 px**

**Corrección:** El documento cubre sesiones 10-12 pero incluye resultados de sesión 13 sin aclararlo.

---

### 2. ❌ Tabla "Sin TTA vs Con TTA" Engañosa

**Tabla documentada (líneas 121-127):**

| Seed | Val Error | Test Error (sin TTA) | Test Error (con TTA) |
|------|-----------|----------------------|----------------------|
| 42   | 7.22 px   | 6.75 px             | **6.75 px** ← Idénticos |
| 123  | 7.84 px   | 7.16 px             | **7.16 px** ← Idénticos |
| 456  | 7.87 px   | 7.20 px             | **7.20 px** ← Idénticos |

**Problema:** Los valores "sin TTA" y "con TTA" son **idénticos**.

**Evidencia del código:**
```python
# evaluate_ensemble.py línea 116
pred = predict_with_tta(model, images, device)
```
El script **SIEMPRE usa TTA**, no tiene opción para desactivarlo.

**Conclusión:** No hay evidencia de evaluación "sin TTA". La columna es engañosa.

---

### 3. ❌ Valores de Test Error Incorrectos

**Documento dice:**
- seed123: test_error = 7.16 px
- seed456: test_error = 7.20 px

**Código dice (`evaluate_ensemble.py` líneas 196-202):**
```python
'seed123': {
    'val_error': 5.05,
    'test_error': 4.05,  # ← No 7.16
},
'seed456': {
    'val_error': 5.21,
    'test_error': 4.04,  # ← No 7.20
},
```

**Explicación:** Los valores 7.16 y 7.20 px son **errores de validación**, no de test.

---

### 4. ❌ Subsección "3 Modelos" con Tabla "4 Modelos"

**Línea 136:**
```latex
\subsection{Ensemble de 3 Modelos}
```

**Línea 140 (caption):**
```latex
\caption{Resultados del ensemble de 4 modelos con TTA}
```

**Contradicción directa** entre título de subsección y tabla.

---

### 5. ❌ Comparación Individual vs Ensemble Incorrecta

**Documentado (tabla líneas 144-149):**
- Mejor individual (seed=456): 7.20 px
- Ensemble 4 modelos: 3.71 px
- **Mejora: 48%**

**Realidad (según código):**
- Mejor individual (seed=456): **4.04 px** (no 7.20)
- Ensemble 4 modelos: 3.71 px
- **Mejora real: 8.2%** (no 48%)

---

## VERIFICACIONES CORRECTAS ✅

### 1. Implementación TTA

**✅ Flip horizontal correcto:**
```python
# src_v2/evaluation/metrics.py línea 331
images_flipped = torch.flip(images, dims=[3])
```

**✅ Inversión de X correcta:**
```python
# Línea 289
landmarks[:, :, 0] = 1.0 - landmarks[:, :, 0]
```

**✅ Intercambio de pares simétricos:**
```python
# Línea 272
SYMMETRIC_PAIRS = [(2, 3), (4, 5), (6, 7), (11, 12), (13, 14)]
# L3↔L4, L5↔L6, L7↔L8, L12↔L13, L14↔L15
```

**✅ Promedio correcto:**
```python
# Línea 338
return (pred_original + pred_flipped) / 2
```

---

### 2. Implementación Ensemble

**✅ Combinación de predicciones:**
```python
# scripts/evaluate_ensemble.py líneas 113-122
preds = []
for model in models:
    pred = predict_with_tta(model, images, device)
    preds.append(pred)

preds_stack = torch.stack(preds)
weights_tensor = torch.tensor(weights, device=device).view(-1, 1, 1)
ensemble_pred = (preds_stack * weights_tensor).sum(dim=0)
```

**Proceso:**
1. Cada modelo → 2 predicciones (original + flip)
2. Promedio dentro de modelo
3. Promedio entre modelos (con pesos opcionales)

**Total:** Con M modelos → **2×M predicciones** por imagen

---

### 3. Valores Numéricos de Modelos Individuales

**✅ Verificados contra outputs:**

**seed42:**
- Documentado: 6.75 px
- Real (`outputs/session10/exp4_epochs100/evaluation_report.txt`): 6.75 px ✅

**seed123:**
- Documentado: 7.16 px
- Real (`outputs/session10/ensemble/seed123/evaluation_report.txt`): 7.16 px ✅

**seed456:**
- Documentado: 7.20 px
- Real (`outputs/session10/ensemble/seed456/evaluation_report.txt`): 7.20 px ✅

---

### 4. Cálculos de Porcentajes

**✅ Todos correctos:**

- **48%:** (7.20 - 3.71) / 7.20 = 48.5% ✓
- **16%:** (4.50 - 3.79) / 4.50 = 15.8% ✓
- **58%:** (9.08 - 3.79) / 9.08 = 58.3% ✓

---

### 5. Resultados por Categoría

**Documentado (líneas 317-321):**

| Categoría | Error | N | vs Normal |
|-----------|-------|---|-----------|
| Normal    | 3.42  | 47 | baseline |
| COVID     | 3.77  | 31 | +10.2%  |
| Viral     | 4.40  | 18 | +28.7%  |
| Overall   | 3.71  | 96 | --      |

**Verificación matemática:**
```python
overall = (3.42*47 + 3.77*31 + 4.40*18) / 96 = 3.72 px
```
Redondeado: **3.71 px** ✅ CORRECTO

---

## RECOMENDACIONES

### Correcciones Urgentes

1. **Aclarar configuración final en abstract:**
   ```latex
   La configuración óptima de las SESIONES 10-12 fue el ensemble de 2 modelos
   (seed=123 y seed=456) con TTA, que logró 3.79 píxeles. En SESIÓN 13
   (fuera del alcance de este documento), el ensemble de 4 modelos alcanzó 3.71 píxeles.
   ```

2. **Eliminar columna "sin TTA" de tabla:**
   ```latex
   \textbf{Seed} & \textbf{Val Error} & \textbf{Test Error (con TTA)} \\
   42 & 7.22 px & 6.75 px \\
   123 & 5.05 px & 4.05 px \\
   456 & 5.21 px & 4.04 px \\
   ```

3. **Corregir subsección "Ensemble de 3 Modelos":**
   - Cambiar título a: "Ensemble Inicial (3 Modelos, Sesión 10)"
   - Cambiar caption tabla a: "Resultados preliminares - evolución hacia 4 modelos"

4. **Actualizar comparación:**
   ```latex
   Mejor individual (seed=456): 4.04 px
   Ensemble 2 modelos (S12): 3.79 px (mejora 6.2%)
   Ensemble 4 modelos (S13): 3.71 px (mejora 8.2%)
   ```

---

## RESUMEN DE RESPUESTAS

### 1. Test-Time Augmentation (TTA)
- ✅ **Flip horizontal:** Implementado correctamente
- ✅ **Corrección X:** `X' = 1.0 - X` verificado
- ❌ **Mejora documentada:** No hay evidencia de evaluación "sin TTA"
- ✅ **Verificado en:** `src_v2/evaluation/metrics.py` líneas 275-338

### 2. Ensemble de modelos
- ✅ **Combinación:** Promedio ponderado implementado correctamente
- ⚠️ **Número de modelos:** CONFUSO - mezcla 2, 3 y 4 modelos
- ✅ **Pesos:** Inversamente proporcionales a val_error
- ✅ **Verificado en:** `scripts/evaluate_ensemble.py` líneas 77-178

### 3. Resultados documentados
- ❌ **Sin TTA vs Con TTA:** Tabla engañosa, valores idénticos
- ❌ **Individual vs Ensemble:** Usa valores incorrectos (val en vez de test)
- ⚠️ **Verificación outputs:** Solo individuales, no ensemble completo

### 4. Métricas de mejora
- ✅ **Porcentajes:** 48%, 16%, 58% verificados y correctos
- ⚠️ **Baseline:** Varía entre 7.20, 9.08 sin aclaración

---

## CONCLUSIÓN FINAL

**Código:** ✅ Implementación técnica CORRECTA (TTA y Ensemble)

**Documentación:** ❌ Presentación CONFUSA con inconsistencias graves

**Datos numéricos:** ✅ Valores individuales VERIFICADOS, ensemble parcialmente verificado

**Recomendación:** **REVISAR Y CORREGIR** el documento antes de publicación, especialmente:
1. Aclarar 2 vs 3 vs 4 modelos
2. Eliminar/corregir tabla "sin TTA vs con TTA"
3. Usar valores de test correctos (4.04-4.05 px, no 7.16-7.20 px)
4. Definir claramente el alcance (sesiones 10-12 vs 10-13)

---

**Fecha:** 2025-12-06
**Archivos verificados:** 14 archivos (código, outputs, checkpoints, documentación)
