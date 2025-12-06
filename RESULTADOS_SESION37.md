# Resultados Sesion 37: Analisis de Mapeo de Clases y Evaluacion Externa

**Fecha:** 03-Dic-2024
**Objetivo:** Validar estrategia de mapeo de clases y preparar evaluacion justa

---

## 1. Analisis de Confusion entre Clases (Modelos Entrenados)

### 1.1 Pregunta clave

El mapeo actual es:
- P(positive) = P(COVID)
- P(negative) = P(Normal) + P(Viral_Pneumonia)

**Preocupacion:** COVID-19 es un tipo de neumonia viral. Si los modelos confunden
Viral_Pneumonia con COVID, el mapeo podria ser problematico.

### 1.2 Analisis de matrices de confusion (14 modelos, 3 clases)

Se analizaron todas las matrices de confusion de los modelos entrenados:

| Tipo de Confusion | Total | Porcentaje |
|-------------------|-------|------------|
| COVID -> Normal | 38/434 | **8.8%** |
| COVID -> Viral_Pneumonia | 1/434 | **0.2%** |
| Normal -> COVID | 42/658 | **6.4%** |
| Viral_Pneumonia -> COVID | 0/252 | **0.0%** |

### 1.3 Hallazgo clave

**Viral_Pneumonia NO se confunde con COVID:**
- Viral_Pneumonia -> COVID: **0.0%** (0 de 252 muestras)
- Normal -> COVID: **6.4%** (42 de 658 muestras)

Los modelos son MEJORES distinguiendo COVID de Viral_Pneumonia que de Normal.
Esto es contraintuitivo pero favorable para nuestro mapeo.

---

## 2. Evaluacion de Estrategias de Mapeo

Se evaluaron 8 estrategias en Dataset3 (8,482 imagenes):

### 2.1 Estrategias evaluadas

| Codigo | Estrategia | Descripcion |
|--------|------------|-------------|
| A | P(neg) = P(Normal) + P(Viral) | Actual (baseline) |
| B | P(neg) = P(Normal) only | Solo Normal como negativo |
| C_0.2 | Excluir P(Viral) > 0.2 | Filtrar muestras ambiguas |
| C_0.3 | Excluir P(Viral) > 0.3 | Filtrar muestras ambiguas |
| C_0.4 | Excluir P(Viral) > 0.4 | Filtrar muestras ambiguas |
| D_0.3 | P(neg) = P(Normal) + 0.3*P(Viral) | Ponderacion 30% |
| D_0.5 | P(neg) = P(Normal) + 0.5*P(Viral) | Ponderacion 50% |
| D_0.7 | P(neg) = P(Normal) + 0.7*P(Viral) | Ponderacion 70% |

### 2.2 Resultados - MobileNetV2 Original

| Estrategia | Acc% | Sens% | Spec% | AUC% | Excluidos |
|------------|------|-------|-------|------|-----------|
| A (baseline) | 55.7 | 77.1 | 34.2 | 58.0 | 0 |
| **B (solo Normal)** | **55.8** | **78.8** | 32.8 | **59.6** | 0 |
| C_0.2 | 55.5 | 78.3 | 32.5 | 57.8 | 290 |
| C_0.3 | 55.5 | 77.9 | 32.8 | 57.7 | 199 |
| C_0.4 | 55.5 | 77.7 | 32.9 | 57.7 | 142 |
| D_0.3 | 55.8 | 78.2 | 33.5 | 58.7 | 0 |
| D_0.5 | 55.8 | 77.9 | 33.7 | 58.4 | 0 |
| D_0.7 | 55.8 | 77.6 | 34.0 | 58.3 | 0 |

**Mejor:** Estrategia B con AUC=59.6% (+1.6% vs baseline)

### 2.3 Resultados - MobileNetV2 Warped

| Estrategia | Acc% | Sens% | Spec% | AUC% | Excluidos |
|------------|------|-------|-------|------|-----------|
| **A (baseline)** | **55.5** | 49.6 | **61.4** | **57.9** | 0 |
| B (solo Normal) | 53.2 | 52.0 | 54.4 | 55.4 | 0 |
| C_0.2 | 53.4 | 51.5 | 55.6 | 56.5 | 1132 |
| C_0.3 | 53.6 | 51.1 | 56.3 | 56.7 | 809 |
| C_0.4 | 53.9 | 51.0 | 57.1 | 56.8 | 593 |
| D_0.3 | 54.3 | 50.9 | 57.7 | 56.9 | 0 |
| D_0.5 | 54.7 | 50.6 | 58.9 | 57.3 | 0 |
| D_0.7 | 55.0 | 50.1 | 59.9 | 57.6 | 0 |

**Mejor:** Estrategia A (baseline) - sin mejora significativa

---

## 3. Conclusion sobre el Mapeo

### 3.1 El mapeo actual es VALIDO

Basado en el analisis:

1. **Viral_Pneumonia no se confunde con COVID (0%)**
   - Los modelos distinguen perfectamente COVID de Viral_Pneumonia
   - La preocupacion teorica no se materializa en la practica

2. **Las estrategias alternativas no mejoran significativamente**
   - Estrategia B mejora AUC en +1.6% para modelos original
   - Pero para modelos warped, el baseline A es mejor
   - Las diferencias son marginales

3. **El mapeo es conceptualmente correcto**
   - "negative" en FedCOVIDx = NO tiene COVID-19
   - Normal + Viral_Pneumonia = "no COVID" = correcto

### 3.2 Recomendacion final

**CONTINUAR con mapeo actual (Opcion A):**
```
P(positive) = P(COVID)
P(negative) = P(Normal) + P(Viral_Pneumonia)
```

Razones:
- Analisis empirico confirma que no hay confusion COVID↔Viral_Pneumonia
- Mantiene consistencia con evaluacion de Sesion 36
- Cualquier alternativa solo ofrece mejoras marginales e inconsistentes

---

## 4. Warping de Dataset3

### 4.1 Proceso de warping

Se aplicó warping geométrico a todas las 8,482 imágenes de Dataset3:

| Métrica | Valor |
|---------|-------|
| Imágenes procesadas | 8,482 |
| Imágenes fallidas | 0 |
| Fill rate promedio | **47.1%** |
| Fill rate std | 0.4% |
| Fill rate min | 28.1% |
| Fill rate max | 47.1% |
| Tiempo de procesamiento | 3.3 minutos |

### 4.2 Consistencia del warping

El fill rate de ~47% es **consistente** con los datos de entrenamiento warped
que tienen ~53% de fondo negro, validando que el proceso de warping se aplicó
correctamente.

---

## 5. Comparación Justa: Original vs Warped

### 5.1 Evaluación de modelos WARPED en Dataset3 WARPEADO

| Modelo | Acc% | Sens% | Spec% | AUC% | F1% | Gap% |
|--------|------|-------|-------|------|-----|------|
| resnet50_warped | **54.8** | 79.4 | 30.1 | **58.5** | **63.7** | 34.8 |
| resnet18_warped | 53.5 | 70.2 | 36.7 | 56.6 | 60.1 | 31.9 |
| mobilenet_v2_warped | 52.5 | 72.1 | 33.0 | 56.0 | 60.3 | 40.2 |
| vgg16_warped | 51.0 | 65.8 | 36.1 | 52.8 | 57.3 | 39.7 |
| densenet121_warped | 50.7 | 77.6 | 23.8 | 50.6 | 61.1 | 38.9 |
| efficientnet_b0_warped | 49.8 | 73.6 | 26.0 | 50.9 | 59.4 | 41.9 |
| alexnet_warped | 41.9 | 54.0 | 29.9 | 39.8 | 48.2 | 48.7 |

**Promedios modelos WARPED en D3_warped:**
- Accuracy: **50.6%**
- AUC-ROC: **52.2%**
- Gap: **39.4%**

### 5.2 Comparación: Warped en D3_original vs Warped en D3_warped

| Modelo | D3_orig% | D3_warp% | Mejora | Gap_orig% | Gap_warp% |
|--------|----------|----------|--------|-----------|-----------|
| resnet50_warped | 55.2 | 54.8 | -0.4 | 34.4 | 34.8 |
| resnet18_warped | 50.2 | 53.5 | **+3.3** | 35.2 | 31.9 |
| mobilenet_v2_warped | 55.5 | 52.5 | -3.0 | 37.2 | 40.2 |
| vgg16_warped | 56.4 | 51.0 | -5.4 | 34.2 | 39.7 |
| densenet121_warped | 56.3 | 50.7 | -5.6 | 33.3 | 38.9 |
| efficientnet_b0_warped | 55.2 | 49.8 | -5.4 | 36.5 | 41.9 |
| alexnet_warped | 48.4 | 41.9 | -6.5 | 42.2 | 48.7 |

### 5.3 Comparación: Modelos ORIGINAL vs WARPED (evaluación justa)

| Arquitectura | Original→D3_orig | Warped→D3_warp | Diferencia |
|--------------|------------------|----------------|------------|
| resnet18 | **57.5%** | 53.5% | -4.0% |
| resnet50 | **56.5%** | 54.8% | -1.7% |
| mobilenet_v2 | **55.7%** | 52.5% | -3.2% |
| vgg16 | 52.5% | **51.0%** | -1.5% |
| efficientnet_b0 | **54.1%** | 49.8% | -4.3% |
| densenet121 | **51.8%** | 50.7% | -1.1% |
| alexnet | 44.7% | 41.9% | -2.8% |

**Promedio Original→D3_orig:** 53.3%
**Promedio Warped→D3_warp:** 50.6%
**Diferencia:** -2.7%

---

## 6. Conclusiones Finales

### 6.1 Hallazgos principales

1. **El warping NO mejora la generalización externa**
   - Modelos ORIGINAL en D3_original: 53.3% promedio
   - Modelos WARPED en D3_warped: 50.6% promedio
   - El warping reduce el rendimiento en ~2.7%

2. **ResNet50_warped es el mejor modelo warped**
   - Accuracy: 54.8%
   - AUC-ROC: 58.5%
   - Mejor sensibilidad (79.4%)

3. **El domain shift persiste**
   - Gap de generalización promedio: ~37-40%
   - La normalización geométrica no elimina completamente
     la diferencia de dominio

4. **ResNet18_original sigue siendo el mejor**
   - Accuracy en D3_original: 57.5%
   - AUC-ROC: 60.7%
   - Mejor balance sensibilidad/especificidad

### 6.2 Implicaciones para la tesis

- La hipótesis de que el warping mejoraría la generalización
  NO se confirma en evaluación externa
- Los modelos entrenados en datos originales generalizan
  ligeramente mejor que los warped
- El warping puede ser útil para otras aplicaciones
  (visualización, interpretabilidad) pero no para clasificación

---

## 7. Archivos Generados

```
outputs/external_validation/
├── mapping_analysis_results.json        <- Análisis de mapeo
├── baseline_results.json                <- Sesión 36 (referencia)
├── warped_on_warped_results.json        <- Evaluación justa
├── dataset3_warped/
│   ├── test/positive/                   <- 4,241 imágenes warped
│   ├── test/negative/                   <- 4,241 imágenes warped
│   ├── test_warping_summary.json        <- Estadísticas warping
│   └── test_landmarks.json              <- Landmarks predichos

scripts/
├── analyze_class_mapping.py             <- Análisis de mapeo
├── warp_dataset3.py                     <- Warping de Dataset3
├── evaluate_external_warped.py          <- Evaluación warped→warped

docs/
├── RESULTADOS_SESION37.md               <- Este documento
```

---

*Sesión 37 completada: 03-Dic-2024*
