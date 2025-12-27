# ESTRUCTURA APROBADA DE LA TESIS

**Fecha de aprobación:** 16 Diciembre 2025
**Estado:** Estructura aprobada, iniciando Fase 3 (Redacción)

---

## INFORMACIÓN GENERAL

- **Título:** "Normalización y alineación automática de la forma de la región pulmonar integrada con selección de características discriminantes para detección de neumonía y COVID-19"
- **Extensión objetivo:** 80-120 páginas
- **Formato:** LaTeX
- **Referencias:** Mínimo 50, estilo IEEE

---

## ESTRUCTURA DE CAPÍTULOS

### PÁGINAS PRELIMINARES (~10 páginas)
- Portada
- Carta de liberación
- Dedicatoria (opcional)
- Agradecimientos
- Resumen / Abstract
- Índice general
- Índice de figuras
- Índice de tablas
- Lista de abreviaturas

---

### CAPÍTULO 1: INTRODUCCIÓN (10-12 páginas)

| Sección | Páginas | Descripción |
|---------|---------|-------------|
| 1.1 Antecedentes | 3 | Contexto de COVID-19 y diagnóstico por imagen |
| 1.2 Planteamiento del problema | 2 | Variabilidad en radiografías, necesidad de normalización |
| 1.3 Justificación | 2 | Importancia del sistema propuesto |
| 1.4 Objetivos | 1 | General y específicos (6) |
| 1.5 Hipótesis | 1 | Hipótesis de investigación |
| 1.6 Alcances y limitaciones | 1 | Scope del trabajo |
| 1.7 Organización del documento | 1 | Descripción de capítulos |

---

### CAPÍTULO 2: MARCO TEÓRICO (18-22 páginas)

| Sección | Páginas | Descripción |
|---------|---------|-------------|
| 2.1 Imágenes radiográficas de tórax | 3 | Principios de formación, anatomía pulmonar |
| 2.2 Redes neuronales convolucionales | 5 | CNNs, transfer learning, Coordinate Attention |
| 2.3 Predicción de landmarks anatómicos | 4 | Formulación de regresión, Wing Loss |
| 2.4 Normalización geométrica de imágenes | 4 | GPA, Delaunay, transformación afín por partes |
| 2.5 Métricas de evaluación | 3 | Métricas de regresión y clasificación |

---

### CAPÍTULO 3: ESTADO DEL ARTE (12-15 páginas)

| Sección | Páginas | Descripción |
|---------|---------|-------------|
| 3.1 Detección de COVID-19 en radiografías | 5 | Enfoques CNN, datasets públicos |
| 3.2 Predicción de landmarks en imágenes médicas | 4 | Métodos clásicos y deep learning |
| 3.3 Normalización geométrica en análisis médico | 3 | Trabajos relacionados |
| 3.4 Análisis comparativo y posicionamiento | 2 | Comparación con este trabajo |

---

### CAPÍTULO 4: METODOLOGÍA (22-28 páginas) 🔴 ALTO RIESGO

| Sección | Páginas | Descripción |
|---------|---------|-------------|
| 4.1 Descripción general del sistema | 2 | Diagrama de bloques del pipeline |
| 4.2 Dataset y preprocesamiento | 4 | Dataset, anotación, CLAHE, splits |
| 4.3 Modelo de predicción de landmarks | 6 | ResNet-18 + CoordAttn, Wing Loss, entrenamiento |
| 4.4 Normalización geométrica | 6 | GPA, warping, full coverage, fill rate |
| 4.5 Clasificación de enfermedades pulmonares | 4 | Arquitecturas CNN, ensemble, TTA |
| 4.6 Protocolo de evaluación experimental | 4 | Evaluación de landmarks, clasificación, robustez |

---

### CAPÍTULO 5: RESULTADOS Y DISCUSIÓN (18-22 páginas) 🔴 ALTO RIESGO

| Sección | Páginas | Descripción |
|---------|---------|-------------|
| 5.1 Resultados de predicción de landmarks | 4 | Ensemble 3.71 px, análisis por categoría |
| 5.2 Resultados de clasificación | 4 | Comparación arquitecturas, impacto warping |
| 5.3 Análisis de robustez | 4 | JPEG, blur, experimento de control |
| 5.4 Evaluación de generalización | 4 | Cross-evaluation, validación externa |
| 5.5 Discusión general | 4 | Interpretación, limitaciones, comparación |

---

### CAPÍTULO 6: CONCLUSIONES Y TRABAJO FUTURO (6-8 páginas)

| Sección | Páginas | Descripción |
|---------|---------|-------------|
| 6.1 Conclusiones | 3 | Cumplimiento de objetivos, contribuciones |
| 6.2 Trabajo futuro | 2 | Domain adaptation, validación clínica |
| 6.3 Consideraciones éticas | 1 | Disclaimer médico, limitaciones |

---

### REFERENCIAS BIBLIOGRÁFICAS (~5 páginas)
- Mínimo 50 referencias
- Estilo IEEE
- 60% de referencias de últimos 4 años

---

### ANEXOS (10-15 páginas)

| Anexo | Descripción |
|-------|-------------|
| A | Detalles de implementación |
| B | Hiperparámetros y configuraciones |
| C | Resultados adicionales |
| D | Guía de uso del sistema (CLI) |

---

## RESUMEN DE EXTENSIÓN

| Sección | Páginas |
|---------|---------|
| Preliminares | 10 |
| Capítulo 1: Introducción | 10-12 |
| Capítulo 2: Marco Teórico | 18-22 |
| Capítulo 3: Estado del Arte | 12-15 |
| Capítulo 4: Metodología | 22-28 |
| Capítulo 5: Resultados | 18-22 |
| Capítulo 6: Conclusiones | 6-8 |
| Referencias | 5 |
| Anexos | 10-15 |
| **TOTAL** | **111-137** |

---

## ORDEN DE REDACCIÓN APROBADO

```
FASE A: Núcleo Técnico
1. Capítulo 4: Metodología
2. Capítulo 5: Resultados

FASE B: Contexto
3. Capítulo 2: Marco Teórico
4. Capítulo 3: Estado del Arte

FASE C: Encuadre
5. Capítulo 1: Introducción
6. Capítulo 6: Conclusiones

FASE D: Complementos
7. Anexos
8. Preliminares
```

---

## ALINEACIÓN CON OBJETIVOS DEL ASESOR

**NOTA:** Los objetivos oficiales son los propuestos por el asesor en `5-Objetivos.tex`.

| Objetivo del Asesor | Secciones | Estado |
|--------------------|-----------|--------|
| 1. Método de alineación/normalización | 4.3, 4.4, 5.1 | ✅ Cumplido |
| 2. Selección de características | 4.4 (implícito) | ⚠️ Reinterpretado |
| 3. Clasificadores KNN, CNN, MLP | 4.5 (solo CNN) | ⚠️ Parcial |
| 4. Validación (precisión, sensibilidad, etc.) | 4.6, 5.1-5.4 | ✅ Cumplido |
| 5. Contraste con/sin alineación | 5.3 | ✅ Cumplido |
| 6. Publicación de resultados | — | ⏳ Pendiente |

### Brechas a Justificar en la Tesis

| Brecha | Justificación Propuesta |
|--------|------------------------|
| KNN no implementado | Las CNNs han demostrado ser superiores para clasificación de imágenes médicas; KNN requeriría extracción manual de características |
| MLP no implementado | Similar a KNN; las CNNs integran extracción de características y clasificación |
| "Selección de características" | La normalización geométrica actúa como selección implícita eliminando información no discriminante |

---

## CAPÍTULOS DE ALTO RIESGO

| Capítulo | Nivel | Razón |
|----------|-------|-------|
| Cap. 4: Metodología | 🔴 ALTO | Más extenso, precisión técnica requerida |
| Cap. 5: Resultados | 🔴 ALTO | Claims validados, no inflar resultados |
| Cap. 3: Estado del Arte | 🟡 MEDIO | Búsqueda bibliográfica extensa |
| Cap. 2: Marco Teórico | 🟡 MEDIO | Rigor matemático |

---

## PROGRESO DE REDACCIÓN

| Capítulo | Estado | Fecha Inicio | Fecha Fin |
|----------|--------|--------------|-----------|
| Cap. 4: Metodología | ✅ Completado | 16-Dic-2025 | 16-Dic-2025 |

### Detalle Capítulo 4 - Metodología

| Sección | Páginas | Estado | Fecha |
|---------|---------|--------|-------|
| 4.1 Descripción general del sistema | 2 | ✅ Completada | 16-Dic-2025 |
| 4.2 Dataset y preprocesamiento | 4 | ✅ Completada | 16-Dic-2025 |
| 4.3 Modelo de predicción de landmarks | 6 | ✅ Completada | 16-Dic-2025 |
| 4.4 Normalización geométrica | 6 | ✅ Completada | 16-Dic-2025 |
| 4.5 Clasificación de enfermedades | 4 | ✅ Completada | 16-Dic-2025 |
| 4.6 Protocolo de evaluación | 4 | ✅ Completada | 16-Dic-2025 |
| Cap. 5: Resultados | ⏳ Pendiente | - | - |
| Cap. 2: Marco Teórico | ⏳ Pendiente | - | - |
| Cap. 3: Estado del Arte | ⏳ Pendiente | - | - |
| Cap. 1: Introducción | ⏳ Pendiente | - | - |
| Cap. 6: Conclusiones | ⏳ Pendiente | - | - |
| Anexos | ⏳ Pendiente | - | - |
| Preliminares | ⏳ Pendiente | - | - |

---

## HISTORIAL DE SESIONES DE REDACCIÓN

### Sesión 06 - 16 Diciembre 2025

**Objetivo:** Redactar sección 4.6 (Protocolo de Evaluación Experimental) - COMPLETAR CAPÍTULO 4

**Trabajo realizado:**
1. ✅ Revisión de archivos de contexto antes de redactar:
   - `src_v2/evaluation/metrics.py` - implementación de métricas
   - `GROUND_TRUTH.json` - valores validados
   - Sesiones 29, 39, 55 - protocolos de robustez, control, validación externa

2. ✅ Redacción de sección 4.6 (~4 páginas) con:
   - Métricas de landmarks: MED, error por landmark, error por categoría, percentiles
   - Métricas de clasificación: Accuracy, Precision, Recall, F1
   - Justificación expandida de F1-Macro vs F1-Weighted
   - Protocolo de robustez: JPEG Q50/Q30, blur σ=1/2 (procedimiento técnico completo)
   - Protocolo de cross-evaluation: matriz 2×2 de evaluación cruzada
   - Protocolo de validación externa: FedCOVIDx, mapeo de clases 3→2
   - TTA para landmarks: flip horizontal + promediado

3. ✅ Verificación de valores contra código fuente:
   - compute_pixel_error() ✓ (metrics.py:23-44)
   - compute_error_per_landmark() ✓ (metrics.py:47-61)
   - compute_error_per_category() ✓ (metrics.py:162-195)
   - predict_with_tta() ✓ (metrics.py:300-338)
   - SYMMETRIC_PAIRS para TTA ✓ (constants.py)
   - FedCOVIDx 8,482 muestras ✓ (GROUND_TRUTH.json:174)

4. ✅ Elementos incluidos:
   - 17 ecuaciones formales
   - 5 tablas (percentiles, F1 comparación, robustez, FedCOVIDx, resumen)
   - 1 figura pendiente (esquema cross-evaluation)
   - 4 referencias

**Figuras documentadas pendientes:**
- F4.13: Esquema de evaluación cruzada (matriz 2×2)
- F4.14: Ejemplos de perturbaciones (JPEG, blur)

**Archivos creados:**
- `capitulo4/4_6_protocolo_evaluacion.tex`

5. ✅ Revisión exhaustiva y corrección de errores:

**ERRORES DETECTADOS Y CORREGIDOS:**

| Línea | Error | Antes | Después |
|-------|-------|-------|---------|
| 45 | Posición anatómica incorrecta | "L9, L10 en el ápex pulmonar" | "L9, L10, L11 del eje central" |
| 45 | Posición anatómica incorrecta | "L12, L13 en ángulos costofrénicos" | "L12, L13 en bordes superiores; L14, L15 en ángulos costofrénicos" |
| 225 | Kernel size incorrecto | "kernel de tamaño 5×5" | "kernel automático según OpenCV" |
| 286-287 | Parámetros tabla incorrectos | "kernel=5, σ=1.0/2.0" | "σ=1.0/2.0 (kernel automático)" |
| 107 | Ecuación accuracy confusa | Mezcla notación binaria/multiclase | Ecuación multiclase pura |

**Verificaciones realizadas:**
- ✓ SYMMETRIC_PAIRS correcto: (L3,L4), (L5,L6), (L7,L8), (L12,L13), (L14,L15)
- ✓ FedCOVIDx: 8,482 muestras confirmado en GROUND_TRUTH.json
- ✓ CLAHE: clip_limit=2.0, tile_size=4 confirmado en constants.py
- ✓ Referencias cruzadas: todas válidas

**HITO:** ✅ CAPÍTULO 4 COMPLETADO (6/6 secciones, ~26 páginas)

---

### Sesión 05 - 16 Diciembre 2025

**Objetivo:** Redactar sección 4.5 (Clasificación de Enfermedades Pulmonares)

**Trabajo realizado:**
1. ✅ Investigación exhaustiva antes de redactar:
   - Verificado que NO existe ensemble de clasificadores (solo de landmarks)
   - Verificado que TTA solo aplica a landmarks, no al clasificador
   - Encontrados resultados de comparación de 7 arquitecturas en `outputs/classifier_comparison/`
   - Identificada justificación implícita de selección de ResNet-18

2. ✅ Redacción de sección 4.5 (~4 páginas) con:
   - 7 arquitecturas CNN evaluadas (AlexNet, VGG-16, ResNet-18/50, DenseNet-121, MobileNetV2, EfficientNet-B0)
   - Enfoque en ResNet-18 y EfficientNet-B0 como principales candidatos
   - Comparación: ResNet-18 (99.10%) vs EfficientNet-B0 (97.76%)
   - Estrategia de transfer learning desde ImageNet
   - Configuración de entrenamiento (LR=1e-4, batch=32, dropout=0.3, patience=10)
   - Manejo de desbalance con pesos de clase (ecuación documentada)
   - Data augmentation: flip, rotación, transformación afín

3. ✅ Verificación de TODOS los valores contra código fuente:
   - 7 arquitecturas ✓ (classifier.py:55-63)
   - Dropout=0.3 ✓ (classifier.py:71)
   - Learning rate=1e-4 ✓ (Sesión 22:66)
   - Batch size=32 ✓ (Sesión 22:68)
   - Early stopping patience=10 ✓ (Sesión 15:47)
   - Accuracy 99.10% ✓ (GROUND_TRUTH.json:59)

4. ✅ Autoevaluación completada (7/8 criterios aprobados)

**Figuras documentadas pendientes:**
- F4.11: Ejemplos de data augmentation del clasificador

**Archivos creados:**
- `capitulo4/4_5_clasificacion.tex`

---

### Sesión 04 - 16 Diciembre 2025

**Objetivo:** Redactar sección 4.4 (Normalización Geométrica)

**Trabajo realizado:**
1. ✅ Revisión de archivos de contexto antes de redactar:
   - `src_v2/processing/warp.py` - implementación de warping
   - `src_v2/processing/gpa.py` - implementación de GPA
   - `src_v2/constants.py` - valores de parámetros
   - `GROUND_TRUTH.json` - valores validados
   - Sesiones 25, 52, 53 - documentación de warping y fill rate

2. ✅ Redacción de sección 4.4 (~6-7 páginas) con:
   - Ecuaciones completas de GPA (centrado, escalado, SVD, rotación óptima)
   - Algoritmo iterativo de GPA (pseudocódigo)
   - Triangulación de Delaunay
   - Transformación afín por partes (ecuaciones y algoritmo)
   - Estrategia de full coverage (8 puntos de borde)
   - Parámetro margin_scale óptimo (1.05)

3. ✅ Verificación de TODOS los valores contra código fuente:
   - margin_scale = 1.05 ✓ (constants.py:217)
   - tolerancia GPA = 1e-8 ✓ (gpa.py:141)
   - max_iterations = 100 ✓ (gpa.py:140)
   - 8 puntos borde = 4 esquinas + 4 midpoints ✓ (warp.py:80-114)

4. ✅ Autoevaluación completada (7/8 criterios aprobados)

**DECISIÓN:** Trade-off de fill rate (96% vs 99%) reservado para Capítulo 5 (Resultados)

**Figuras documentadas pendientes:**
- F4.6: Proceso de GPA (formas antes/después de alineación)
- F4.7: Triangulación Delaunay sobre landmarks
- F4.8: Comparación imagen original vs warped
- F4.9: Efecto de diferentes margin_scale
- F4.10: Pipeline completo de normalización

**Archivos creados:**
- `capitulo4/4_4_normalizacion_geometrica.tex`

---

### Sesión 03 - 16 Diciembre 2025

**Objetivo:** Redactar sección 4.3 (Modelo de predicción de landmarks)

**Trabajo realizado:**
1. ✅ Redacción inicial de sección 4.3 (~6 páginas)
2. ✅ Investigación para verificar datos contra fuentes reales
3. ✅ Corrección de errores identificados

**ERRORES DETECTADOS Y CORREGIDOS:**

| Archivo | Error | Antes | Después |
|---------|-------|-------|---------|
| 4.2 | Imágenes anotadas | 956 | 957 |
| 4.2 | Viral Pneumonia | 182 (19.0%) | 183 (19.1%) |
| 4.2 | Split validación | 12.5% | 15% |
| 4.2 | Split prueba | 12.5% | 10% |
| 4.2 | Pérdida de simetría | "se aprovecha durante entrenamiento" | Eliminado (no se usa) |
| 4.3 | Cabeza de regresión | 2 capas (512→768→30) | 3 capas (512→512→768→30) |
| 4.3 | Dropout | 0.5 / 0.25 | 0.3 / 0.15 |
| 4.3 | Normalización cabeza | Sin normalización | GroupNorm |
| 4.3 | Función de pérdida | CombinedLandmarkLoss | Solo WingLoss |
| 4.3 | Batch size fase 2 | 16 | 8 |
| 4.3 | Early stopping fase 2 | 10 épocas | 15 épocas |
| 4.3 | Parámetros cabeza | 417,822 | 683,038 |
| final_config.json | Estructura cabeza | 512→768→256→30 | 512→512→768→30 |

**Archivos modificados:**
- `capitulo4/4_2_dataset_preprocesamiento.tex`
- `capitulo4/4_3_modelo_landmarks.tex`
- `configs/final_config.json`

**Lección aprendida:**
- SIEMPRE verificar datos contra checkpoints reales y documentación de sesiones ANTES de redactar
- `final_config.json` puede estar desactualizado respecto al código real

---

### Sesión 02 - 16 Diciembre 2025

**Trabajo realizado:**
- Redacción de secciones 4.1 y 4.2
- Experimento de clasificación binaria completado
- Estructura de tesis aprobada

---

### Sesión 01 - 16 Diciembre 2025

**Trabajo realizado:**
- Fase 1: Análisis exhaustivo del proyecto
- Fase 2: Definición de estructura de tesis
- Ajuste de objetivos específicos

---

*Documento generado como parte del proceso de redacción de tesis.*
*Última actualización: 16 Diciembre 2025 - Sesión 06*
