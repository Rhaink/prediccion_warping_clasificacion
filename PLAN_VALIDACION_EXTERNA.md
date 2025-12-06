# Plan de Validacion Externa - Experimentos de Generalizacion

**Fecha:** 03-Dic-2024
**Objetivo:** Demostrar que el warping mejora la generalizacion usando datasets externos
**Sesion de planificacion:** 35

---

## 1. ANALISIS CRITICO PREVIO

### 1.1 Problema Identificado

Los experimentos anteriores **NO demuestran generalizacion real**, solo demuestran:
- Transferencia entre representaciones del MISMO dataset
- La asimetria (Warped->Original funciona, Original->Warped falla) es estructural

### 1.2 Lo que NECESITAMOS demostrar

Para afirmar que "warping mejora generalizacion":
1. Entrenar en Dataset A
2. Evaluar en Dataset B (diferente fuente/hospital)
3. Mostrar que modelo Warped tiene menor degradacion que Original

---

## 2. DATASETS DISPONIBLES (Orden de prioridad)

### 2.1 Dataset3 - FedCOVIDx (PRIORIDAD 1)

**Razon de prioridad:** Es el mas diferente al original (multiples fuentes federadas).

| Caracteristica | Valor |
|----------------|-------|
| **Total imagenes** | 84,818 |
| **Train** | 67,863 |
| **Val** | 8,473 |
| **Test** | 8,482 (4,241 positive + 4,241 negative) |
| **Clases** | 2 (COVID positive / COVID negative) |
| **Formato** | PNG mayoritariamente, algunos JPG |
| **Tamano** | Variable (~1024xN), requiere redimensionamiento |
| **Modo color** | Escala de grises (L) |
| **Fuentes** | bimcv (95%), ricord, rsna |

**Estructura de etiquetas (archivos .txt):**
```
patient_id filename label source
419639-003251 MIDRC-RICORD-1C-419639-003251-46647-0.png positive ricord
```

**Desafios:**
- Solo 2 clases (clasificacion binaria COVID vs no-COVID)
- Tamanos variables (necesita resize)
- Etiquetas en archivos TXT separados

**Preprocesamiento requerido:**
```python
# 1. Parsear archivos TXT para obtener etiquetas
# 2. Redimensionar a 299x299
# 3. Reorganizar en estructura de directorios o crear DataLoader custom
```

---

### 2.2 Dataset2 - Chest X-ray Images (PRIORIDAD 2)

| Caracteristica | Valor |
|----------------|-------|
| **Total imagenes** | 9,208 |
| **Train** | 7,367 (Normal: 2,616, COVID: 1,025, Pneumonia: 3,726) |
| **Validation** | 1,841 (Normal: 654, COVID: 256, Pneumonia: 931) |
| **Test** | **NO HAY** |
| **Clases** | 3 (0_Normal, 1_Covid19, 2_Pneumonia) |
| **Formato** | JPEG 299x299 |
| **Modo color** | RGB (3 canales) |

**Desafios:**
- **SIN conjunto de test** - usar validation como test
- RGB requiere conversion a escala de grises
- JPEG tiene artefactos de compresion

**Mapeo de clases:**
```
0_Normal -> Normal
1_Covid19 -> COVID
2_Pneumonia -> Viral_Pneumonia
```

---

### 2.3 Dataset1 - COVID-QU-Ex (PRIORIDAD 3)

| Caracteristica | Valor |
|----------------|-------|
| **Total imagenes** | ~67,840 (Lung Segmentation) |
| **Clases** | 3 (COVID-19, Normal, Non-COVID) |
| **Formato** | PNG 256x256 |
| **Modo color** | Escala de grises (L) |
| **Extra** | Mascaras de segmentacion (NO utiles para nuestro warping) |

**Nota:** Similar al dataset original, usar como ultima validacion.

---

## 3. PLAN DE EXPERIMENTOS POR SESION

### SESION 36: Preparacion de Dataset3 y Baseline

**Objetivo:** Preparar Dataset3 para evaluacion y establecer baseline.

#### Tareas:

1. **Crear script de preprocesamiento Dataset3**
   ```python
   # scripts/prepare_dataset3.py
   - Parsear train.txt, val.txt, test.txt
   - Redimensionar imagenes a 299x299
   - Crear estructura compatible con nuestro DataLoader
   - Guardar en outputs/external_datasets/dataset3/
   ```

2. **Adaptar pipeline de inferencia para clasificacion binaria**
   - Nuestros modelos tienen 3 clases (COVID, Normal, Viral_Pneumonia)
   - Dataset3 tiene 2 clases (positive=COVID, negative=no-COVID)
   - **Opcion A:** Mapear Normal+Viral_Pneumonia -> negative
   - **Opcion B:** Entrenar nuevos modelos binarios

3. **Evaluar modelos existentes en Dataset3 (sin warpear)**
   - Modelo Original (3 clases) -> Dataset3 test
   - Modelo Warped (3 clases) -> Dataset3 test
   - Medir: Accuracy, Sensibilidad COVID, Especificidad

---

### SESION 37: Warping de Dataset3 y Comparacion

**Objetivo:** Aplicar warping a Dataset3 y comparar generalizacion.

#### Tareas:

1. **Predecir landmarks en Dataset3**
   ```python
   # Usar ensemble de landmarks para predecir en imagenes de Dataset3
   # Guardar predicciones: outputs/external_datasets/dataset3/landmarks/
   ```

2. **Generar version warpeada de Dataset3**
   ```python
   # Aplicar piecewise_affine_warp con margin 1.25
   # Guardar: outputs/external_datasets/dataset3_warped/
   ```

3. **Experimento principal: Cross-evaluation externa**

   | Entrenado en | Evaluado en | Metrica esperada |
   |--------------|-------------|------------------|
   | Original (nuestro) | Dataset3 Original | Baseline |
   | Original (nuestro) | Dataset3 Warped | Degradacion? |
   | Warped (nuestro) | Dataset3 Original | Mejor? |
   | Warped (nuestro) | Dataset3 Warped | Comparacion |

4. **Analisis de degradacion**
   - Cual modelo se degrada menos en datos externos?
   - La asimetria se mantiene con datos externos?

---

### SESION 38: Experimentos con Dataset2

**Objetivo:** Validacion con dataset de 3 clases.

#### Tareas:

1. **Preparar Dataset2**
   - Convertir RGB -> Escala de grises
   - Usar validation como test (80% train / 20% test split)

2. **Repetir experimentos de Sesion 37 con Dataset2**
   - Cross-evaluation Original vs Warped
   - Comparar con resultados de Dataset3

3. **Ventaja de Dataset2:** 3 clases compatibles con nuestros modelos

---

### SESION 39: Experimentos de Ablacion

**Objetivo:** Aislar el efecto del warping vs otras variables.

#### Experimento de Ablacion:

| Condicion | Descripcion | Proposito |
|-----------|-------------|-----------|
| A | Original completo | Baseline |
| B | Warped margin 1.25 (fondo negro) | Actual |
| C | Crop rectangular (bbox de landmarks) | Es solo el crop? |
| D | Warped con fondo = media imagen | Es el fondo negro? |
| E | Warped con borde replicado | Alternativa literatura |

**Pregunta clave:** El beneficio viene de:
- La normalizacion geometrica (warping)?
- Eliminar informacion de contexto (crop)?
- O el fondo negro?

---

### SESION 40: Alternativas al Fondo Negro

**Objetivo:** Evaluar si el fondo negro es optimo.

#### Estrategias de fondo a probar:

1. **Fondo negro (valor 0)** - Actual
2. **Fondo gris (valor 127)**
3. **Fondo = media de imagen original**
4. **Borde replicado (border padding)**
5. **Contexto original preservado** (warpear solo ROI, mantener resto)

**Metricas:**
- Accuracy en test propio
- Accuracy en validacion externa
- Tiempo de convergencia
- Estabilidad del entrenamiento

---

### SESION 41: Analisis Final y Conclusiones

**Objetivo:** Consolidar resultados y reformular hipotesis.

#### Tareas:

1. **Tabla consolidada de todos los experimentos**
2. **Analisis estadistico de diferencias**
3. **Reformulacion de hipotesis basada en evidencia**
4. **Documentacion para tesis**

---

## 4. ESTRUCTURA DE ARCHIVOS A CREAR

```
prediccion_coordenadas/
├── scripts/
│   ├── prepare_dataset3.py          # Preprocesamiento Dataset3
│   ├── prepare_dataset2.py          # Preprocesamiento Dataset2
│   ├── evaluate_external.py         # Evaluacion en datasets externos
│   ├── warp_external_dataset.py     # Warpear datasets externos
│   ├── ablation_experiment.py       # Experimento de ablacion
│   └── background_experiment.py     # Experimento de fondos
│
├── outputs/
│   └── external_validation/
│       ├── dataset3/
│       │   ├── processed/           # Imagenes preprocesadas
│       │   ├── landmarks/           # Predicciones de landmarks
│       │   └── warped/              # Imagenes warpeadas
│       ├── dataset2/
│       └── results/
│           ├── cross_evaluation.json
│           └── ablation_results.json
│
└── docs/
    ├── PLAN_VALIDACION_EXTERNA.md   # Este documento
    └── RESULTADOS_VALIDACION.md     # Resultados (a crear)
```

---

## 5. CRITERIOS DE EXITO

### Para afirmar "Warping mejora generalizacion":

**Modelo Warped debe tener menor degradacion en datos externos que Modelo Original**

| Escenario | Interpretacion |
|-----------|----------------|
| Gap_Warped < Gap_Original en Dataset3 | Hipotesis soportada |
| Gap_Warped ~= Gap_Original | No hay ventaja de generalizacion |
| Gap_Warped > Gap_Original | Warping perjudica generalizacion |

### Medicion del Gap:
```
Gap = Accuracy_test_propio - Accuracy_dataset_externo
```

Ejemplo:
- Modelo Original: 98.81% propio -> 75% externo = Gap 23.81%
- Modelo Warped: 98.02% propio -> 85% externo = Gap 13.02%
- **Conclusion:** Warped generaliza mejor (gap menor)

---

## 6. CONSIDERACIONES IMPORTANTES

### 6.1 Sobre las Mascaras de Segmentacion

Las mascaras de Dataset1 **NO son utiles** para nuestro proposito porque:
- Nuestro warping incluye la zona entre pulmones (mediastino)
- Las mascaras solo cubren los pulmones aislados
- No hay correspondencia directa

### 6.2 Sobre el Fondo Negro

La literatura cientifica indica que:
- Zero-padding extenso NO es optimo
- Mejor practica: preprocesamiento con cropping previo
- Alternativas: borde replicado, valor medio

**Debemos probar alternativas en Sesion 40.**

### 6.3 Sobre Clasificacion Binaria vs 3 Clases

Dataset3 solo tiene 2 clases (COVID positive/negative).
Opciones:
1. **Mapear salidas:** Normal + Viral_Pneumonia -> negative
2. **Entrenar modelos nuevos:** Clasificacion binaria desde cero
3. **Usar umbral:** Si P(COVID) > 0.5 -> positive

**Recomendacion:** Opcion 1 para evaluacion rapida, Opcion 2 para rigor.

---

## 7. PREGUNTAS A RESPONDER

Al final de todas las sesiones, debemos poder responder:

1. El warping mejora la generalizacion a datos EXTERNOS?
2. O solo mejora transferencia entre representaciones del MISMO dataset?
3. El fondo negro es un problema?
4. Cual es la mejor estrategia de fondo?
5. El beneficio viene del warping o simplemente de eliminar contexto?
6. Que arquitectura generaliza mejor (DenseNet vs otros)?

---

## 8. TIMELINE SUGERIDO

| Sesion | Duracion estimada | Entregable |
|--------|-------------------|------------|
| 36 | 2-3 horas | Dataset3 preparado, baseline |
| 37 | 3-4 horas | Cross-evaluation Dataset3 |
| 38 | 2-3 horas | Cross-evaluation Dataset2 |
| 39 | 3-4 horas | Resultados ablacion |
| 40 | 2-3 horas | Resultados fondos |
| 41 | 2 horas | Documentacion final |

**Total estimado:** 14-19 horas de trabajo

---

*Documento creado: Sesion 35*
*Autor: Claude Code + Usuario*
