# Resultados Sesion 36: Evaluacion Externa en Dataset3 (FedCOVIDx)

**Fecha:** 03-Dic-2024
**Objetivo:** Establecer baseline de generalizacion en dataset externo

---

## 1. Dataset3 (FedCOVIDx) - Preparacion

### Estadisticas del Dataset
| Split | Total | Positive (COVID) | Negative | Balance |
|-------|-------|------------------|----------|---------|
| Test | 8,482 | 4,241 (50%) | 4,241 (50%) | Perfectamente balanceado |

### Fuentes de datos
| Fuente | Imagenes | Porcentaje | Descripcion |
|--------|----------|------------|-------------|
| BIMCV | 8,082 | 95.3% | Banco de Imagenes Medicas de Valencia |
| RICORD | 200 | 2.4% | RSNA COVID-19 Open Radiology Database |
| RSNA | 200 | 2.4% | Pneumonia Detection Challenge |

### Definicion de clases en Dataset3 (FedCOVIDx)
- **positive**: Pacientes con COVID-19 confirmado
- **negative**: Pacientes SIN COVID-19, que incluye:
  - Pacientes sanos (normales)
  - Pacientes con neumonia viral no-COVID
  - Pacientes con neumonia bacteriana
  - Otras patologias pulmonares (edema, derrame pleural, etc.)

### Preprocesamiento aplicado
- Redimensionamiento a 299x299 (LANCZOS)
- Conversion a escala de grises
- Guardado en formato PNG
- En evaluacion: resize a 224x224, normalizacion ImageNet (igual que entrenamiento)

---

## 1.1 Mapeo de Clases (3 -> 2)

### Justificacion del mapeo

Nuestros modelos tienen 3 clases, Dataset3 tiene 2:

| Modelo Original | Indice | Mapeo a Dataset3 |
|-----------------|--------|------------------|
| COVID | 0 | positive |
| Normal | 1 | negative |
| Viral_Pneumonia | 2 | negative |

**Estrategia de prediccion:**
```
P(positive) = P(COVID) = softmax[0]
P(negative) = P(Normal) + P(Viral_Pneumonia) = softmax[1] + softmax[2]
Prediccion: positive si P(COVID) > 0.5
```

**Este mapeo es conceptualmente correcto porque:**
- "positive" en Dataset3 = tiene COVID-19
- "negative" en Dataset3 = NO tiene COVID-19 (incluye sanos + otras patologias)
- Normal + Viral_Pneumonia = "no tiene COVID" = negative

**Limitacion:** Los "negativos" de Dataset3 pueden incluir patologias que nuestro
modelo no conoce (ej: neumonia bacteriana), lo cual puede introducir ruido.

---

## 1.2 Diferencias de Dominio Criticas

### Caracteristicas de las imagenes

| Dataset | % Pixeles Negros | Media Intensidad | Tamaño Original |
|---------|------------------|------------------|-----------------|
| Original (entrenamiento) | ~6% | ~153 | 299x299 |
| Warped (entrenamiento) | **~53%** | ~51 | 224x224 |
| Dataset3 (externo) | ~2% | ~130 | Variable |

**NOTA IMPORTANTE:**
- Los modelos WARPED fueron entrenados con imagenes que tienen **53% de fondo negro**
- Dataset3 NO tiene este fondo negro (imagenes completas)
- Esto significa que los modelos WARPED se evaluan en un dominio MUY diferente
- Para una comparacion justa, Dataset3 deberia warpearse primero

---

## 2. Resultados de Evaluacion

### Tabla completa de resultados

| Modelo | Tipo | Train% | Ext% | Gap% | Sens% | AUC% |
|--------|------|--------|------|------|-------|------|
| resnet18_original | original | 95.8 | 57.5 | 38.3 | 75.6 | 60.7 |
| resnet50_original | original | 93.8 | 56.5 | 37.3 | 75.2 | 58.9 |
| vgg16_warped | warped | 90.6 | 56.4 | 34.2 | 44.0 | 56.9 |
| densenet121_warped | warped | 89.6 | 56.3 | 33.3 | 35.9 | 59.6 |
| mobilenet_v2_original | original | 99.0 | 55.7 | 43.3 | 77.1 | 58.0 |
| mobilenet_v2_warped | warped | 92.7 | 55.5 | 37.2 | 49.6 | 57.9 |
| efficientnet_b0_warped | warped | 91.7 | 55.2 | 36.5 | 44.6 | 56.7 |
| resnet50_warped | warped | 89.6 | 55.2 | 34.4 | 84.7 | 58.8 |
| efficientnet_b0_original | original | 95.8 | 54.1 | 41.8 | 84.3 | 54.8 |
| vgg16_original | original | 93.8 | 52.5 | 41.2 | 87.2 | 57.4 |
| densenet121_original | original | 94.8 | 51.8 | 43.0 | 74.0 | 52.5 |
| resnet18_warped | warped | 85.4 | 50.2 | 35.2 | 42.1 | 51.1 |
| alexnet_warped | warped | 90.6 | 48.4 | 42.2 | 18.8 | 39.3 |
| alexnet_original | original | 86.5 | 44.7 | 41.8 | 47.6 | 41.1 |

### Resumen comparativo

| Metrica | Modelos ORIGINAL | Modelos WARPED |
|---------|------------------|----------------|
| Accuracy externa promedio | 53.25% | 53.88% |
| Gap de generalizacion promedio | 40.95% | **36.15%** |

---

## 3. Hallazgos Clave

### 3.1 Gap de Generalizacion
Los modelos **WARPED tienen un gap de generalizacion 4.8% menor** que los modelos originales:
- Gap promedio Original: 40.95%
- Gap promedio Warped: **36.15%**

### 3.2 Accuracy Externa Similar
La accuracy externa es practicamente igual (~53-54%), pero los modelos warped:
- Parten de accuracy de entrenamiento menor (menos overfitting)
- Por lo tanto, sufren menos degradacion al generalizar

### 3.3 Trade-off Sensibilidad vs Especificidad
- **Modelos Original**: Alta sensibilidad COVID (74-87%), baja especificidad
- **Modelos Warped**: Sensibilidad mas baja (35-85%), mejor balance

### 3.4 Mejores modelos por tipo

| Tipo | Mejor Modelo | Acc Externa | Gap |
|------|--------------|-------------|-----|
| Original | ResNet-18 | 57.5% | 38.3% |
| Warped | VGG-16 | 56.4% | 34.2% |
| Warped | DenseNet-121 | 56.3% | **33.3%** |

---

## 4. Interpretacion

### 4.1 Analisis del Gap de Generalizacion

| Tipo | Gap Promedio | Interpretacion |
|------|--------------|----------------|
| ORIGINAL | 40.95% | Modelos con mas overfitting, mayor degradacion |
| WARPED | 36.15% | Menos overfitting inicial, menor degradacion |

**Diferencia: 4.8% menor gap para modelos WARPED**

PERO esta comparacion tiene un sesgo importante:
- Los modelos ORIGINAL se evaluan en imagenes similares a su entrenamiento
- Los modelos WARPED se evaluan en imagenes MUY diferentes (sin fondo negro)

### 4.2 Limitaciones metodologicas

1. **Comparacion asimetrica**:
   - Modelos ORIGINAL: entrenados sin warp -> evaluados sin warp (dominio similar)
   - Modelos WARPED: entrenados con warp -> evaluados sin warp (dominio diferente)

2. **Mapeo de clases imperfecto**:
   - Dataset3 "negative" incluye patologias no cubiertas por nuestro modelo
   - Ej: neumonia bacteriana no es ni Normal ni Viral_Pneumonia

3. **Domain shift severo**:
   - Dataset3 proviene de multiples hospitales/equipos
   - Diferentes protocolos de adquisicion de imagenes

### 4.3 Siguiente paso critico (Sesion 37)

Para una comparacion JUSTA, necesitamos:

1. **Warpear Dataset3** usando el ensemble de landmarks
2. Evaluar modelos WARPED en Dataset3 warpeado
3. Entonces comparar:
   - Original en Dataset3 original vs Warped en Dataset3 warpeado
   - Esta seria la comparacion correcta de generalizacion

---

## 5. Archivos Generados

```
outputs/external_validation/
├── dataset3/
│   ├── test/
│   │   ├── positive/     (4,241 imagenes)
│   │   ├── negative/     (4,241 imagenes)
│   │   └── metadata.csv
│   └── preparation_stats.json
├── baseline_results.json
├── baseline_comparison.png
└── gap_analysis.png
```

---

## 6. Conclusiones Sesion 36

### Tareas completadas:
1. Dataset3 preparado exitosamente: 8,482 imagenes de test procesadas
2. Baseline establecido: 14 modelos evaluados en dataset externo
3. Mapeo de clases documentado y justificado (3->2 clases)
4. Diferencias de dominio identificadas y documentadas

### Hallazgos preliminares:
- Gap promedio WARPED (36.15%) < Gap promedio ORIGINAL (40.95%)
- PERO la comparacion actual tiene sesgo metodologico
- Los modelos WARPED se evaluan en dominio diferente al de entrenamiento

### Proximo paso critico (Sesion 37):

**PRIMERO - Analizar validez del mapeo de clases:**

La preocupación es: ¿Es correcto sumar P(Normal) + P(Viral_Pneumonia) = P(negative)?

| Argumento | A favor | En contra |
|-----------|---------|-----------|
| Conceptual | Ambos son "no COVID" | COVID es un tipo de neumonía viral |
| Visual | - | Viral_Pneumonia puede verse similar a COVID |
| Modelo | - | Si confunde COVID↔Viral_Pneumonia, el mapeo falla |

**Tareas Sesión 37:**
1. Analizar matrices de confusión: ¿Cuánto confunde COVID con Viral_Pneumonia?
2. Probar estrategias alternativas de mapeo:
   - Opción A: P(negative) = P(Normal) + P(Viral_Pneumonia) [actual]
   - Opción B: P(negative) = P(Normal) solamente
   - Opción C: Ponderar según confusión observada
3. Solo si el mapeo es válido, continuar con:
   - Warpear Dataset3
   - Comparación justa de generalización

---

## 7. Referencias

- [BIMCV Dataset](https://bimcv.cipf.es/)
- [RICORD - RSNA COVID-19 Database](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230281)
- [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)

---

*Sesion 36 completada: 03-Dic-2024*
