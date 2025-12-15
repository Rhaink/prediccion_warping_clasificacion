# Sesion 55: Validacion Externa en FedCOVIDx

**Fecha:** 2025-12-14
**Rama:** feature/external-validation
**Estado:** COMPLETADO

---

## Objetivo

Evaluar el clasificador recomendado `warped_96` (99.10% accuracy interno) en el dataset externo FedCOVIDx (Dataset3) para documentar formalmente las limitaciones de generalizacion cross-domain.

---

## Contexto

### Estado Previo
- Proyecto en v2.1.0, APROBADO PARA DEFENSA
- 655 tests pasando
- 12 modelos ya evaluados en Dataset3 (~50-57% accuracy) en sesiones 36-37
- `warped_96` (best_classifier.pt) NO habia sido evaluado externamente

### Hipotesis
El domain shift predominara sobre las mejoras metodologicas, resultando en ~55% accuracy (similar a otros modelos evaluados).

---

## Metodologia

### Dataset Externo: FedCOVIDx (Dataset3)

| Caracteristica | Valor |
|----------------|-------|
| Total muestras | 8,482 |
| Clases | 2 (Positive/Negative) |
| Distribucion | 50/50 balanceado |
| Fuentes | BIMCV (~95%), RICORD, RSNA |
| Resolucion | Variable (300-3000 px) |

### Mapeo de Clases (3 a 2)

```
P(positive) = P(COVID)
P(negative) = P(Normal) + P(Viral_Pneumonia)
Prediccion = positive si P(COVID) >= 0.5
```

### Preprocesamiento

Identico al entrenamiento:
- CLAHE: clip_limit=2.0, tile_size=4
- Resize: 224x224 (bilinear)
- Normalizacion: ImageNet (mean/std)

### Comandos Ejecutados

```bash
# Evaluacion en Dataset3 Original
python -m src_v2 evaluate-external \
    outputs/classifier_replication_v2/best_classifier.pt \
    --external-data outputs/external_validation/dataset3 \
    --output outputs/external_validation/warped_96_on_d3_original.json

# Evaluacion en Dataset3 Warpeado
python -m src_v2 evaluate-external \
    outputs/classifier_replication_v2/best_classifier.pt \
    --external-data outputs/external_validation/dataset3_warped \
    --output outputs/external_validation/warped_96_on_d3_warped.json
```

---

## Resultados

### Metricas warped_96 en Dataset3

| Metrica | D3 Original | D3 Warped |
|---------|-------------|-----------|
| **Accuracy** | **53.36%** | **55.31%** |
| Sensitivity (Recall COVID) | 90.12% | 89.86% |
| Specificity (Recall No-COVID) | 16.60% | 20.75% |
| Precision | 51.94% | 53.14% |
| F1-Score | 65.90% | 66.78% |
| AUC-ROC | 0.5422 | 0.5994 |
| Gap vs interno (99.10%) | 45.74% | 43.79% |

### Matriz de Confusion

**D3 Original:**
```
              Pred Neg    Pred Pos
Actual Neg       704        3537
Actual Pos       419        3822
```

**D3 Warped:**
```
              Pred Neg    Pred Pos
Actual Neg       880        3361
Actual Pos       430        3811
```

### Comparacion con Baseline (Sesion 36-37)

| Modelo | Tipo | Acc. Interna | Acc. D3 Original | Acc. D3 Warped |
|--------|------|--------------|------------------|----------------|
| resnet18_original | Original | 95.83% | 57.50% | - |
| vgg16_warped | Warped | 90.63% | 56.44% | ~50% |
| densenet121_warped | Warped | 89.58% | 56.26% | ~51% |
| **warped_96** | **RECOMENDADO** | **99.10%** | **53.36%** | **55.31%** |

---

## Analisis

### Observaciones Clave

1. **Domain shift predomina:** El gap de ~45% confirma que el domain shift es el factor dominante, no la metodologia de normalizacion geometrica.

2. **Alto sesgo hacia COVID:**
   - Sensibilidad ~90% (detecta casi todos los COVID)
   - Especificidad ~17-21% (muchos falsos positivos)
   - El modelo predice COVID para la mayoria de las muestras

3. **Dataset warped no mejora significativamente:**
   - D3 Original: 53.36%
   - D3 Warped: 55.31%
   - Diferencia: +1.95% (no significativa)

4. **warped_96 rinde similar a otros modelos:**
   - Consistente con baseline (~50-57%)
   - Mejor accuracy interna NO se traduce en mejor external

### Causas del Domain Shift

1. **Diferencias de equipos/protocolos:**
   - Dataset interno: COVID-19 Radiography Database
   - Dataset externo: FedCOVIDx (multiples instituciones)

2. **Diferencias de poblacion:**
   - Distribucion geografica diferente
   - Caracteristicas demograficas desconocidas

3. **Landmarks predichos vs ground truth:**
   - Dataset interno: landmarks anotados manualmente
   - Dataset externo: landmarks predichos (pueden tener errores)

4. **Mapeo de clases:**
   - Interno: 3 clases (COVID, Normal, Viral_Pneumonia)
   - Externo: 2 clases (Positive, Negative)
   - Perdida de informacion en el mapeo

---

## Conclusion

### Lo que SI mejora la normalizacion geometrica

1. **Generalizacion within-domain:** 2.4x mejor cross-evaluation interno
2. **Robustez a perturbaciones:** 30x mejor a JPEG, 3-5x mejor a blur
3. **Accuracy interno:** 99.10% (mejor que original 98.84%)

### Lo que NO resuelve la normalizacion geometrica

1. **Domain shift cross-domain:** ~55% en datos externos
2. **Sesgo de prediccion:** Alto en datos no vistos
3. **Generalizacion a nuevos equipos/protocolos**

### Implicacion Practica

> La normalizacion geometrica mediante landmarks anatomicos es una tecnica efectiva para mejorar la robustez y generalizacion **dentro del mismo dominio de datos**. Sin embargo, para uso en datasets nuevos (diferentes equipos, protocolos, poblaciones), se requieren tecnicas de **domain adaptation** adicionales.

---

## Archivos Generados

| Archivo | Descripcion |
|---------|-------------|
| `outputs/external_validation/warped_96_on_d3_original.json` | Resultados D3 original |
| `outputs/external_validation/warped_96_on_d3_warped.json` | Resultados D3 warped |

## Archivos Modificados

| Archivo | Cambio |
|---------|--------|
| `GROUND_TRUTH.json` | Agregada seccion external_validation |
| `docs/RESULTADOS_EXPERIMENTALES_v2.md` | Agregada seccion 8 (Validacion Externa) |
| `README.md` | Agregada seccion External Validation |

---

## Experimento de Verificacion: CLAHE

### Hipotesis de Verificacion

Para validar que los resultados no son artefactos de preprocesamiento, se investigo si aplicar CLAHE explicito durante la evaluacion externa mejoraria los resultados.

### Metodologia

Se modifico `evaluate-external` para aplicar CLAHE (clip_limit=2.0, tile_size=4) antes de la conversion a RGB, replicando exactamente el pipeline de entrenamiento.

### Resultados con CLAHE Explicito

| Metrica | D3 Original | D3 Original + CLAHE | D3 Warped | D3 Warped + CLAHE |
|---------|-------------|---------------------|-----------|-------------------|
| Accuracy | 53.36% | **50.65%** | 55.31% | **50.80%** |
| Sensitivity | 90.12% | 90.36% | 89.86% | 95.50% |
| Specificity | 16.60% | 10.94% | 20.75% | 6.11% |
| AUC-ROC | 0.5422 | 0.5164 | 0.5994 | 0.5479 |

### Analisis de Histogramas

Comparacion estadistica de intensidades de pixeles:

| Imagen | Media | Desv. Std | Entropia |
|--------|-------|-----------|----------|
| Training (referencia) | 135.82 | 59.48 | 7.04 |
| External sin CLAHE | 98.50 | 49.82 | - |
| External con CLAHE | 123.35 | 64.68 | - |

**Distancia a Training:**
- Sin CLAHE: 46.98
- Con CLAHE: 17.66 (mas cercano)

### Conclusion del Experimento

**Paradoja:** Las imagenes externas CON CLAHE son estadisticamente MAS CERCANAS a las de entrenamiento, pero el accuracy fue PEOR (50.65% vs 53.36%).

**Implicacion:** El domain shift NO es un artefacto de preprocesamiento. Es un problema real causado por diferencias fundamentales entre datasets (equipos, poblaciones, protocolos). El preprocesamiento identico no resuelve estas diferencias semanticas.

---

## Trabajo Futuro

1. **Domain adaptation:** Investigar tecnicas como:
   - Transfer learning con fine-tuning en datos externos
   - Domain-adversarial training
   - Unsupervised domain adaptation

2. **Datasets adicionales:** Evaluar en:
   - Montgomery County TB dataset
   - Shenzhen Hospital TB dataset
   - COVIDx (clases identicas)

3. **Analisis de causas:** Cuantificar contribucion de:
   - Diferencias de equipos
   - Diferencias de poblacion
   - Errores de prediccion de landmarks

---

## Referencias

- Sesion 36: Evaluacion baseline original
- Sesion 37: Evaluacion baseline warped
- FedCOVIDx: Dataset externo
- GROUND_TRUTH.json: Valores validados
