# REPORTE DE VERIFICACIÓN: Documentos 16 y 17
## Validación Externa y Resultados Consolidados

**Fecha de verificación:** 2025-12-06
**Documentos revisados:**
- `/home/donrobot/Projects/Tesis/documentación/16_validacion_externa.tex`
- `/home/donrobot/Projects/Tesis/documentación/17_resultados_consolidados.tex`

**Fuentes de datos:**
- `/home/donrobot/Projects/Tesis/outputs/external_validation/baseline_results.json`
- `/home/donrobot/Projects/Tesis/outputs/external_validation/mapping_analysis_results.json`
- `/home/donrobot/Projects/Tesis/outputs/session30_analysis/consolidated_results.json`
- `/home/donrobot/Projects/Tesis/RESULTADOS_SESION36.md`

---

## DOCUMENTO 16: VALIDACIÓN EXTERNA (16_validacion_externa.tex)

### 1. Dataset FedCOVIDx (8,482 imágenes)

**DOCUMENTADO:**
- Línea 21-23: "el dataset FedCOVIDx, una colección de 8,482 radiografías de tórax"
- Tabla 1 (línea 81): "Imágenes totales & 8,482"
- Tabla 2 (línea 100): "Número de imágenes & 957 & 8,482"

**VERIFICADO EN DATOS:**
```json
{
  "external_dataset": "Dataset3_FedCOVIDx",
  "n_samples": 8482
}
```

**ESTADO:** ✅ **VERIFICADO**

---

### 2. Resultados de Validación Externa

#### 2.1 Accuracy en Dataset Externo

**DOCUMENTADO (Tabla 2, línea 175):**
```latex
Accuracy & \textbf{57.5\%} & 53.5\% \\
```

**VERIFICADO EN DATOS:**
```
Mejor modelo Original: resnet18_original
  - Accuracy: 57.50% ✅

Mejor modelo Warped: vgg16_warped
  - Accuracy: 56.44% ⚠️ (documentado como 53.5%)
```

**HALLAZGO CRÍTICO:**

El documento menciona **53.5% para warped**, pero los datos muestran:
- VGG16 warped: **56.44%** (mejor warped)
- MobileNetV2 warped: 55.48%
- ResNet50 warped: 55.18%

**INVESTIGACIÓN ADICIONAL:**

Busqué si existe un modelo específico con 53.5%:
- No hay ningún modelo exacto con 53.5% de accuracy
- El promedio de modelos warped es 53.88%
- Posible confusión con datos de un experimento específico

**ESTADO:** ⚠️ **DISCREPANCIA** - El valor 53.5% no corresponde a ningún modelo individual warped en los datos. Debería ser 56.44% (VGG16) o especificar que es un promedio.

---

#### 2.2 Gap de Generalización Cross-Dataset

**DOCUMENTADO (Tabla 3, línea 202-203):**
```latex
Original & 98.81\% & -- & 57.5\% & -41.3\% \\
Warped & 98.02\% & -- & 53.5\% & -44.5\% \\
```

**CÁLCULO DEL GAP:**
- Gap Original: 98.81% - 57.50% = **41.31%** ✅
- Gap Warped (si usamos 56.44%): 98.02% - 56.44% = **41.58%**
- Gap Warped (si usamos 53.5%): 98.02% - 53.5% = **44.52%** ✅

**VERIFICADO EN DATOS:**
```
Accuracy Promedio Original: 53.25%
Accuracy Promedio Warped:   53.88%
Gap Promedio Original:      40.95%
Gap Promedio Warped:        36.15%
```

**ESTADO:** ⚠️ **DISCREPANCIA** - Los gaps documentados parecen calculados sobre modelos específicos, no promedios. Si se usa el mejor modelo warped real (56.44%), el gap sería menor.

---

#### 2.3 Comparación de F1, AUC-ROC (Tabla 2, líneas 176-179)

**DOCUMENTADO:**
```latex
F1-Score (macro) & 56.8\% & 52.9\% \\
AUC-ROC & 0.59 & 0.55 \\
Precision (Positive) & 54.2\% & 51.1\% \\
Recall (Positive) & 63.8\% & 58.2\% \\
```

**VERIFICADO EN DATOS (ResNet18 Original - mejor original):**
```json
{
  "accuracy": 0.5749,
  "f1_score": 0.6400,  ⚠️ (documentado: 56.8%)
  "auc_roc": 0.6075,   ⚠️ (documentado: 0.59)
  "precision": 0.5551, ✅ (documentado: 54.2%)
  "sensitivity": 0.7557 ⚠️ (documentado: 63.8%)
}
```

**VERIFICADO EN DATOS (VGG16 Warped - mejor warped):**
```json
{
  "accuracy": 0.5644,
  "f1_score": 0.5024,  ⚠️ (documentado: 52.9%)
  "auc_roc": 0.5692,   ⚠️ (documentado: 0.55)
  "precision": 0.5857, ⚠️ (documentado: 51.1%)
  "sensitivity": 0.4398 ⚠️ (documentado: 58.2%)
}
```

**ESTADO:** ⚠️ **DISCREPANCIAS MÚLTIPLES** - Los valores documentados no coinciden exactamente con ningún modelo específico en los datos.

---

### 3. Calidad de Landmarks en Datos Externos

**DOCUMENTADO (Tabla 5, línea 244-247):**
```latex
Landmarks bien ubicados & 62\% \\
Landmarks con error moderado (5-15 px) & 28\% \\
Landmarks muy errados (>15 px) & 10\% \\
```

**ESTADO:** ❌ **NO VERIFICABLE** - No se encontraron archivos con análisis de calidad de landmarks en datos externos. Los directorios `outputs/external_validation/` no contienen `landmark_quality_analysis.json` como menciona la Tabla 10 (línea 394).

---

## DOCUMENTO 17: RESULTADOS CONSOLIDADOS (17_resultados_consolidados.tex)

### 4. Gap de Generalización Interna (Cross-Domain)

**DOCUMENTADO (Tabla 7, línea 322):**
```latex
Original & 98.81\% & -- & 57.5\% & -41.3\% \\
Warped & 98.02\% & -- & 53.5\% & -44.5\% \\
\textbf{Gap} & -3.03\% & +24.57\% & \textbf{11.3×} \\
```

**VERIFICADO EN DATOS:**
```
=== VALIDACIÓN INTERNA (Cross-domain) ===
Original en Original: 98.81% ✅
Original en Warped:   73.45% ✅
Warped en Warped:     98.02% ✅
Warped en Original:   95.78% ✅

=== GAP DE GENERALIZACIÓN ===
Gap Original: 25.36% ✅
Gap Warped:   2.24%  ⚠️ (documentado: 24.57%?)

=== RATIOS DE TRANSFERENCIA ===
Ratio Original: 0.7433
Ratio Warped:   0.9772
Gap improvement: 11.32x ✅
```

**ANÁLISIS DEL RATIO 11.3×:**

El cálculo documentado en línea 186:
```latex
\frac{25.36\%}{2.24\%} = \textbf{11.32}
```

**VERIFICADO:** ✅ 25.36 / 2.24 = **11.32** ✅

**ESTADO:** ✅ **VERIFICADO** - El ratio de mejora de 11.3× es correcto.

**NOTA:** La línea 322 parece tener un error tipográfico. Dice "+24.57%" pero debería ser "+2.24%" para el gap de Warped.

---

### 5. Tablas de Resultados Consolidados

#### 5.1 Tabla de Validación Externa (Tabla 12, líneas 370-377)

**DOCUMENTADO:**
```latex
Accuracy & 57.5\% & 53.5\% & -4.0\% \\
F1-Score & 56.8\% & 52.9\% & -3.9\% \\
AUC-ROC & 0.59 & 0.55 & -0.04 \\
Gap vs. interno & -41.3\% & -44.5\% & -- \\
```

**VERIFICACIÓN:**
Ver sección 2.2 y 2.3 arriba.

**ESTADO:** ⚠️ **DISCREPANCIA** - Mismos problemas que en Documento 16.

---

#### 5.2 Tabla de Síntesis de Mejoras (Tabla 13, líneas 399-406)

**DOCUMENTADO:**
```latex
Error landmarks (MAE) & 9.08 px & 3.71 px & 59\% reducción \\
Accuracy clasificación & -- & 98.96\% & Estado del arte \\
Generalización interna & 0.74 ratio & 1.02 ratio & 11.3× mejor \\
Robustez JPEG (Q=50) & -16.52\% & -0.52\% & 31.8× más robusto \\
Robustez blur (σ=3) & -17.56\% & -6.35\% & 2.8× más robusto \\
Fill rate warping & 47.1\% & 96.1\% & 2× cobertura \\
```

**VERIFICACIÓN - Generalización interna:**
```
Ratio Original: 0.7433 ✅ (redondeado a 0.74)
Ratio Warped:   0.9772 ⚠️ (documentado como 1.02)
```

**CÁLCULO DEL RATIO WARPED:**
- Warped en Original: 95.78%
- Warped en Warped: 98.02%
- Ratio: 95.78 / 98.02 = **0.9772** ⚠️

**¿De dónde sale 1.02?**
Posiblemente es el ratio inverso o un cálculo diferente. Revisando datos consolidados:
```json
"generalization_gaps": {
  "original_gap": 25.362318840579704,
  "warped_gap": 2.2397891963109373
}
```

**ESTADO:** ⚠️ **DISCREPANCIA** - El ratio documentado de 1.02 no coincide con el cálculo directo de 0.9772.

---

### 6. Resultados de Clasificación (MobileNetV2)

**DOCUMENTADO (línea 278, abstract línea 25):**
```latex
\textbf{Mejor modelo original}: MobileNetV2 con 98.96\% accuracy
```

**VERIFICADO EN DATOS CONSOLIDADOS:**
```json
"ideal_conditions": {
  "original": {
    "test_accuracy": 98.81,  ⚠️
    "val_accuracy": 99.34
  }
}
```

**ESTADO:** ⚠️ **DISCREPANCIA MENOR** - Los datos consolidados muestran 98.81% para el test, no 98.96%. Posiblemente 98.96% es de validación en otro experimento.

---

## RESUMEN EJECUTIVO

### ✅ VERIFICADO (5 items)

1. **Dataset FedCOVIDx tiene 8,482 imágenes** ✅
2. **Accuracy Original en validación externa: 57.5%** ✅ (ResNet18)
3. **Gap de generalización interna: 25.36% vs 2.24%** ✅
4. **Ratio de mejora: 11.3×** ✅
5. **Valores de validación interna (98.81%, 73.45%, 98.02%, 95.78%)** ✅

---

### ⚠️ DISCREPANCIAS IDENTIFICADAS (7 items)

1. **Accuracy Warped en validación externa: 53.5%**
   - Documentado: 53.5%
   - Datos reales: 56.44% (VGG16 - mejor warped)
   - Posible causa: Confusión con promedio (53.88%) o experimento específico

2. **F1-Score Original en validación externa: 56.8%**
   - Documentado: 56.8%
   - Datos reales (ResNet18): 64.0%

3. **AUC-ROC en validación externa**
   - Original documentado: 0.59 | Real: 0.61
   - Warped documentado: 0.55 | Real: 0.57

4. **Precision/Recall en validación externa**
   - Múltiples discrepancias entre valores documentados y datos

5. **Ratio de generalización Warped: 1.02**
   - Documentado: 1.02
   - Cálculo directo: 0.9772
   - Diferencia: ~4%

6. **Gap Warped en Tabla 7 línea 322: "+24.57%"**
   - Documentado: +24.57%
   - Debería ser: +2.24% (error tipográfico probable)

7. **Accuracy MobileNetV2: 98.96%**
   - Documentado: 98.96%
   - Datos consolidados: 98.81%

---

### ❌ NO VERIFICABLE (2 items)

1. **Calidad de landmarks en datos externos (Tabla 5)**
   - No se encontró archivo `landmark_quality_analysis.json`
   - Porcentajes documentados (62%, 28%, 10%) no verificables

2. **Visualización t-SNE del domain shift**
   - No se encontró archivo `tsne_visualization.png` mencionado en línea 394

---

## RECOMENDACIONES

### CRÍTICAS (Requieren corrección inmediata)

1. **Documento 16, Tabla 2 (línea 175):** Verificar y corregir accuracy Warped en validación externa
   - Actual: 53.5%
   - Sugerido: 56.4% (VGG16) o especificar si es promedio

2. **Documento 17, Tabla 7 (línea 322):** Corregir gap Warped
   - Actual: +24.57%
   - Debería ser: +2.24%

### IMPORTANTES (Requieren verificación)

3. **Documento 17, Tabla 13:** Verificar ratio de generalización Warped
   - Actual: 1.02
   - Calculado: 0.9772
   - Aclarar metodología de cálculo

4. **Documentos 16 y 17:** Verificar métricas de validación externa
   - F1, AUC, Precision, Recall muestran discrepancias
   - Especificar qué modelo exacto se está reportando

### MENORES (Para completitud)

5. **Documento 16:** Generar o documentar fuente de datos de calidad de landmarks
6. **Documento 16:** Generar visualización t-SNE si se menciona en el texto

---

## ARCHIVOS REVISADOS

### Outputs verificados:
- ✅ `/home/donrobot/Projects/Tesis/outputs/external_validation/baseline_results.json`
- ✅ `/home/donrobot/Projects/Tesis/outputs/external_validation/mapping_analysis_results.json`
- ✅ `/home/donrobot/Projects/Tesis/outputs/session30_analysis/consolidated_results.json`
- ❌ `/home/donrobot/Projects/Tesis/outputs/external_validation/landmark_quality_analysis.json` (NO EXISTE)
- ❌ `/home/donrobot/Projects/Tesis/outputs/external_validation/tsne_visualization.png` (NO VERIFICADO)

### Documentos fuente:
- ✅ `/home/donrobot/Projects/Tesis/RESULTADOS_SESION36.md`
- ✅ `/home/donrobot/Projects/Tesis/SESSION_LOG.md` (consultado parcialmente)

---

## CONCLUSIÓN

Los documentos 16 y 17 contienen **información mayormente correcta** pero con **discrepancias específicas** en:
- Métricas de validación externa (accuracy warped, F1, AUC)
- Ratio de generalización warped (1.02 vs 0.9772)
- Error tipográfico en gap de Tabla 7

El ratio de mejora de **11.3× en generalización interna** está **correctamente calculado y verificado**.

El dataset FedCOVIDx con **8,482 imágenes** está **correctamente documentado**.

**Prioridad:** Corregir los 7 ítems marcados con ⚠️ para garantizar consistencia entre documentación y datos.
