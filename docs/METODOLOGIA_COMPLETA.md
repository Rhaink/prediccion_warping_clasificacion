# Metodología Completa: Detección de COVID-19 mediante Radiografías de Tórax con Normalización Geométrica

## Resumen

Este documento describe la metodología completa para la detección de COVID-19 a partir de radiografías de tórax mediante un sistema de tres etapas: (1) detección de puntos anatómicos mediante redes neuronales convolucionales, (2) normalización geométrica por deformación afín por piezas, y (3) clasificación por ensamble con Test-Time Augmentation (TTA). El sistema alcanza una exactitud final de 98.26% en el conjunto de prueba, representando una mejora de +0.58 puntos porcentuales sobre modelos individuales.

## 1. Conjunto de Datos

### 1.1 Fuente y Composición

Se utilizó el dataset público **COVID-19 Radiography Database** (disponible en Kaggle), que contiene radiografías de tórax frontales organizadas en tres clases diagnósticas:

- **COVID**: Casos confirmados de COVID-19 (n=3,616 imágenes)
- **Normal**: Radiografías normales sin patología (n=10,192 imágenes)
- **Viral_Pneumonia**: Neumonía viral no-COVID (n=1,345 imágenes)

**Total del dataset:** 15,153 imágenes en formato PNG

### 1.2 Particiones del Dataset

El conjunto de datos se dividió en tres particiones mutuamente excluyentes utilizando una semilla aleatoria fija (`split_seed=42`) para garantizar reproducibilidad:

| Partición | COVID | Normal | Viral_Pneumonia | Total | Porcentaje |
|-----------|-------|--------|-----------------|-------|------------|
| **Entrenamiento** | 2,530 | 7,134 | 942 | 10,606 | 70.0% |
| **Validación** | 634 | 1,784 | 234 | 2,652 | 17.5% |
| **Prueba** | 452 | 1,274 | 169 | 1,895 | 12.5% |

**Nota importante:** El conjunto de prueba se mantuvo completamente aislado durante todo el proceso de desarrollo, entrenamiento y selección de hiperparámetros. Solo se utilizó para la evaluación final descrita en la Sección 6.

### 1.3 Manejo de Duplicados

Durante la auditoría pre-implementación (Fase 1), se identificaron **9 imágenes duplicadas** mediante hash SHA-256:
- **1 duplicado** en el conjunto de prueba: `test/Normal/Normal-817.png` (idéntico a `train/Normal/Normal-818.png`)
- **8 duplicados** en el conjunto de validación

**Protocolo de manejo:**
1. Los duplicados se **documentaron** pero **no se removieron** de los archivos originales para preservar la proveniencia de los datos
2. Para la evaluación final, se reportan dos conjuntos de resultados:
   - **Original:** 1,895 muestras (con duplicado)
   - **Limpio:** 1,894 muestras (duplicado filtrado en tiempo de ejecución)
3. Ambos conjuntos produjeron **métricas idénticas** (exactitud 98.26%), confirmando que el duplicado no afectó los resultados

Este enfoque de reporte dual maximiza la transparencia metodológica para defensa de tesis.

## 2. Detección de Puntos Anatómicos (Landmarks)

### 2.1 Definición de Landmarks

Se definieron **15 puntos anatómicos** que delimitan los contornos pulmonares en radiografías frontales. Estos landmarks NO corresponden a estructuras anatómicas específicas, sino que definen la envolvente geométrica de los pulmones:

**Eje central vertical (5 puntos):**
- L1: Punto superior central
- L9: 25% de descenso desde L1
- L10: 50% de descenso (punto medio)
- L11: 75% de descenso
- L2: Punto inferior central

**Contorno pulmonar izquierdo (5 puntos):**
- L12, L3, L5, L7, L14 (distribuidos en altura)

**Contorno pulmonar derecho (5 puntos):**
- L13, L4, L6, L8, L15 (distribuidos en altura)

**Pares simétricos:** (L3, L4), (L5, L6), (L7, L8), (L12, L13), (L14, L15)

Esta configuración permite capturar la geometría completa de ambos pulmones con simetría bilateral.

### 2.2 Arquitectura del Modelo

**Modelo base:** ResNet-18 preentrenado en ImageNet

**Modificaciones arquitectónicas:**
1. **Coordinate Attention Module:** Mecanismo de atención espacial para mejorar la localización de landmarks
2. **Cabeza de regresión profunda:** Red completamente conectada con tres capas:
   - Capa oculta: 768 dimensiones con ReLU y Dropout (p=0.3)
   - Salida: 30 valores (15 landmarks × 2 coordenadas [x, y])

**Normalización de coordenadas:** Las coordenadas se normalizan al rango [0, 1] durante el entrenamiento y se desnormalizan a píxeles (224×224) para evaluación.

### 2.3 Entrenamiento en Dos Fases

**Fase 1 - Congelación del backbone (15 épocas):**
- Solo se entrena la cabeza de regresión
- Tasa de aprendizaje: 1e-3
- Optimizador: AdamW
- Permite adaptación inicial sin corromper pesos preentrenados

**Fase 2 - Fine-tuning completo (100 épocas):**
- Se descongelan todas las capas
- Tasa de aprendizaje diferenciada:
  - Backbone: 2e-5 (bajo para preservar características preentrenadas)
  - Cabeza: 2e-4 (alto para refinar regresión)
- Early stopping basado en error de validación

**Función de pérdida:** Wing Loss con penalización por asimetría bilateral (peso λ=0.1)

### 2.4 Preprocesamiento: CLAHE

**CLAHE (Contrast Limited Adaptive Histogram Equalization)** se aplicó a todas las imágenes para mejorar el contraste local:

- **clip_limit:** 2.0
- **tile_size:** 4×4 (determinado experimentalmente como óptimo vs. 8×8)
- **Espacio de color:** Escala de grises

CLAHE mejora la visibilidad de estructuras pulmonares en regiones de bajo contraste sin amplificar ruido.

### 2.5 Ensamble de 4 Modelos

Para reducir la varianza y mejorar la robustez, se entrenaron **4 modelos independientes** con diferentes semillas aleatorias:

| Modelo | Semilla | Checkpoint |
|--------|---------|------------|
| Modelo 1 | 123 | `checkpoints/session10/ensemble/seed123/final_model.pt` |
| Modelo 2 | 321 | `checkpoints/session13/seed321/final_model.pt` |
| Modelo 3 | 111 | `checkpoints/repro_split111/session14/seed111/final_model.pt` |
| Modelo 4 | 666 | `checkpoints/repro_split666/session16/seed666/final_model.pt` |

**Combinación de predicciones:** Promedio aritmético de las coordenadas predichas por los 4 modelos.

### 2.6 Test-Time Augmentation (TTA) para Landmarks

Durante la inferencia, se aplica TTA con **reflexión horizontal** para reducir aún más la varianza:

1. Predicción en imagen original → (x₁, y₁)
2. Predicción en imagen reflejada horizontalmente → (x₂, y₂)
3. Corrección de pares simétricos: Al reflejar, los landmarks izquierdos y derechos se intercambian
4. Promedio: landmark_final = (landmark_original + landmark_reflejado_corregido) / 2

Esta técnica aprovecha la simetría bilateral natural de los pulmones.

### 2.7 Error de Predicción

**Error final del ensamble 4 modelos + TTA:**
- **Error medio:** 3.61 píxeles (en imágenes de 224×224)
- **Desviación estándar:** 2.48 píxeles
- **Mediana:** 3.07 píxeles

**Error por clase diagnóstica:**
- Normal: 3.22 px (el más bajo, imágenes más claras)
- COVID: 3.93 px
- Neumonía Viral: 4.11 px (el más alto, opacidades dificultan detección)

**Landmarks con mayor error:**
- L12, L13 (extremos laterales superiores): ~5.4 px
- L14, L15 (extremos laterales inferiores): ~4.3 px

**Landmarks con menor error:**
- L10 (punto medio central): 2.44 px
- L9 (cuartil superior central): 2.76 px
- L5, L6 (región media lateral): ~2.9 px

El error de 3.61 px en una imagen de 224×224 representa **1.6% del ancho de la imagen**, un nivel de precisión suficiente para normalización geométrica efectiva.

## 3. Normalización Geométrica por Deformación Afín

### 3.1 Análisis de Procrustes Generalizado (GPA)

Para definir una **forma canónica de referencia**, se aplicó GPA (Generalized Procrustes Analysis) a los landmarks del conjunto de entrenamiento:

**Procedimiento iterativo:**
1. Centrar cada conjunto de landmarks en su centroide
2. Escalar a tamaño unitario (norma Frobenius)
3. Rotar para minimizar la distancia a la forma promedio actual
4. Actualizar la forma promedio
5. Repetir hasta convergencia (cambio < 1e-6)

**Resultado:** Una configuración de 15 landmarks que representa la geometría pulmonar promedio del conjunto de entrenamiento, independiente de traslación, rotación y escala.

### 3.2 Triangulación de Delaunay

A partir de la forma canónica, se calculó una **triangulación de Delaunay** que divide la región pulmonar en triángulos no solapados. Esta triangulación define las transformaciones afines locales para la deformación.

**Propiedades de Delaunay:**
- Maximiza el ángulo mínimo de los triángulos (evita triángulos degenerados)
- Provee una partición única y estable del espacio
- Se computa una sola vez y se reutiliza para todas las imágenes

### 3.3 Deformación Afín por Piezas (Piecewise Affine Warping)

**Algoritmo:**
1. Para cada imagen de entrada:
   - Obtener los 15 landmarks predichos (fuente)
   - Usar la forma canónica como landmarks destino
2. Para cada triángulo en la triangulación de Delaunay:
   - Calcular la transformación afín que mapea el triángulo fuente → triángulo destino
   - Aplicar `cv2.warpAffine()` solo a la región del triángulo
3. Componer todos los triángulos transformados en una imagen de salida

**Características clave:**
- **Continuidad:** Las transformaciones son continuas dentro de cada triángulo
- **Discontinuidad controlada:** Puede haber discontinuidad en los bordes de triángulos (pero mínima debido a la optimización de Delaunay)
- **Preservación local:** Cada triángulo sufre solo una transformación afín (traslación + rotación + escala + cizallamiento)

### 3.4 Expansión de Margen

Para evitar recortar estructuras pulmonares importantes cerca de los bordes, se aplica un **margen de expansión del 5%**:

- `margin_scale = 1.05` (determinado por búsqueda en cuadrícula experimental)
- Cálculo: `landmarks_expandidos = centroide + margin_scale × (landmarks - centroide)`
- Efecto: Expande ligeramente la región delimitada por los landmarks antes de la deformación

Este valor se validó como óptimo para maximizar la inclusión de tejido pulmonar sin agregar exceso de fondo negro.

### 3.5 Tasa de Relleno (Fill Rate)

**Definición:** Porcentaje de píxeles no-negros en la imagen deformada.

**Fill rate promedio:** 47%

Este valor indica que aproximadamente la mitad de la imagen deformada contiene información pulmonar, mientras que el resto es fondo negro. Un fill rate moderado (vs. ~100%) se considera deseable porque:
1. Preserva la integridad de los valores de píxel originales (escala de grises pura)
2. Evita artefactos de CLAHE en regiones de fondo
3. Reduce el riesgo de sobreajuste a píxeles artificiales

## 4. Clasificación por Ensamble

### 4.1 Arquitectura del Clasificador

**Modelo base:** ResNet-18 preentrenado en ImageNet

**Cabeza de clasificación:** Fully connected layer con 3 salidas (COVID, Normal, Viral_Pneumonia)

**Entrada:** Imágenes deformadas de 224×224 píxeles, escala de grises, preprocesadas con CLAHE

### 4.2 Validación Cruzada de 5 Foldios

Para evaluar la estabilidad del modelo sin tocar el conjunto de prueba, se realizó **5-fold cross-validation** en la unión de entrenamiento + validación:

**Configuración:**
- **Total de muestras:** 10,606 (train) + 2,652 (val) = 13,258 imágenes
- **Folios:** 5 particiones mutuamente excluyentes
- **Evaluación en test:** Deshabilitada (`eval_test=false`) para preservar aislamiento
- **Semilla:** 42 (reproducibilidad)

**Resultados en validación cruzada:**

| Métrica | Media | Desviación Estándar |
|---------|-------|---------------------|
| Exactitud | 98.60% | 0.26% |
| F1-macro | 98.00% | 0.36% |
| F1-weighted | 98.60% | 0.25% |

La baja desviación estándar (<0.4%) indica alta estabilidad del modelo frente a diferentes particiones de datos.

### 4.3 Ensamble de 5 Modelos

Los **5 modelos de validación cruzada** se combinaron en un ensamble para la evaluación final:

**Checkpoints:**
- `outputs/classifier_cv/fold_01/best_classifier.pt`
- `outputs/classifier_cv/fold_02/best_classifier.pt`
- `outputs/classifier_cv/fold_03/best_classifier.pt`
- `outputs/classifier_cv/fold_04/best_classifier.pt`
- `outputs/classifier_cv/fold_05/best_classifier.pt`

**Estrategia de combinación:** Votación suave (soft voting) con pesos por F1-macro de validación:

1. Cada modelo produce probabilidades para las 3 clases: [p_COVID, p_Normal, p_Viral]
2. Las probabilidades se ponderan por el F1-macro del modelo en validación
3. Se promedian las probabilidades ponderadas
4. La clase con mayor probabilidad promedio es la predicción final

**Justificación de pesos por F1-macro:**
- F1-macro trata las clases por igual (importante en dataset desbalanceado)
- Se calcula en **validación**, no en prueba (evita contaminación del conjunto de test)
- Refleja la capacidad de generalización del modelo

### 4.4 Exactitud Baseline (Modelo Individual)

**Promedio de los 5 folios individuales:** 97.68% ± 0.16%

Este valor se estableció como **baseline** para medir el beneficio del ensamble y TTA.

### 4.5 Exactitud del Ensamble sin TTA

**Ensamble de 5 modelos (sin TTA):** 98.10%

**Mejora sobre baseline:** +0.42 puntos porcentuales

**Reducción de error:** (100 - 98.10) / (100 - 97.68) = 47% de reducción del error residual

Este resultado (Fase 2) confirmó el beneficio del ensamble antes de agregar TTA.

## 5. Test-Time Augmentation (TTA) para Clasificación

### 5.1 Protocolo de TTA Dual-Nivel

TTA se aplica en **dos niveles** para maximizar la reducción de varianza:

**Nivel 1 - TTA por modelo:**
1. Para cada uno de los 5 modelos del ensamble:
   - Predicción en imagen original → prob_original
   - Predicción en imagen reflejada horizontalmente → prob_reflejada
   - Promedio: prob_modelo = (prob_original + prob_reflejada) / 2

**Nivel 2 - Ensamble de TTA:**
2. Combinar los 5 prob_modelo ponderados por F1-macro de validación
3. Clase final = argmax(promedio_ponderado)

**Nota sobre simetría:** A diferencia de los landmarks, las etiquetas de clase (COVID, Normal, Viral) **no requieren corrección de simetría** al reflejar horizontalmente, ya que el diagnóstico es invariante a la orientación izquierda-derecha.

### 5.2 Implementación Técnica

```python
# Pseudocódigo simplificado
for modelo in modelos_ensamble:
    prob_orig = modelo(imagen)
    prob_flip = modelo(flip_horizontal(imagen))
    prob_tta[modelo] = (prob_orig + prob_flip) / 2

# Votación suave ponderada
prob_final = sum(peso[i] * prob_tta[i] for i in range(5))
prediccion = argmax(prob_final)
```

**Configuración del DataLoader:**
- `num_workers=0` (esencial para reproducibilidad determinística)
- `shuffle=False` (orden fijo para comparación de hashes)

### 5.3 Impacto de TTA

**Exactitud con TTA:** 98.26%

**Mejora sobre ensamble sin TTA:** +0.16 puntos porcentuales

**Mejora total sobre baseline:** +0.58 puntos porcentuales

**Análisis a nivel de muestra (1,895 imágenes):**
- **Ayudadas:** 6 muestras (clasificadas incorrectamente sin TTA, correctas con TTA)
- **Perjudicadas:** 3 muestras (clasificadas correctamente sin TTA, incorrectas con TTA)
- **Neutrales:** 1,886 muestras (sin cambio)
- **Mejora neta:** +3 muestras

**Impacto por clase (delta F1-score):**
- COVID: +0.44% (beneficio máximo)
- Normal: +0.12% (beneficio leve)
- Neumonía Viral: -0.28% (ligera degradación)

**Interpretación:** TTA beneficia principalmente a la clase COVID, posiblemente porque las opacidades COVID tienen patrones más sutiles que se estabilizan con promediado.

## 6. Evaluación Final en Conjunto de Prueba

### 6.1 Protocolo de Evaluación

La evaluación final se realizó en **2026-02-16** (Fase 5) siguiendo un protocolo riguroso:

**Pre-requisitos verificados programáticamente:**
1. **Aislamiento del conjunto de prueba:**
   - ✓ Historial de entrenamiento: ninguna métrica contiene "test"
   - ✓ Archivos de configuración: todos los folds tienen `eval_test=false`
   - ✓ Separación temporal: checkpoints fechados 2026-01-16 (anterior a evaluación final)

2. **Verificación de conteos de clases:**
   - COVID: 452 muestras (esperado: 452) ✓
   - Normal: 1,274 muestras (esperado: 1,274) ✓
   - Viral_Pneumonia: 169 muestras (esperado: 169) ✓

3. **Reproducibilidad determinística:**
   - Evaluación ejecutada **dos veces** en ambos conjuntos (original y limpio)
   - Comparación de hash SHA-256 de los JSON de resultados
   - ✓ VERIFICADO: hashes idénticos entre ejecuciones (reproducibilidad bit-a-bit)

### 6.2 Resultados en Conjunto Original (1,895 muestras)

**Métricas globales:**
- **Exactitud:** 98.26%
- **F1-macro:** 97.12%
- **F1-weighted:** 98.25%

**Matriz de confusión:**

```
                      Predicción
                 COVID  Normal  Viral
Verdadero COVID    441      10      1    (452 total)
Verdadero Normal     4   1,264      6  (1,274 total)
Verdadero Viral      0      12    157    (169 total)
```

**Métricas por clase:**

| Clase | Precisión | Recall | F1-Score | Soporte |
|-------|-----------|--------|----------|---------|
| COVID | 99.10% | 97.57% | 98.33% | 452 |
| Normal | 98.29% | 99.22% | 98.75% | 1,274 |
| Viral_Pneumonia | 95.73% | 92.90% | 94.29% | 169 |

**Observaciones:**
- **COVID:** Alta precisión (99.10%), indica pocos falsos positivos
- **Normal:** Recall excepcional (99.22%), casi todas las radiografías normales identificadas
- **Viral:** Clase más difícil (94.29% F1), probablemente por similitud visual con COVID

### 6.3 Resultados en Conjunto Limpio (1,894 muestras)

**Métricas globales:**
- **Exactitud:** 98.26%
- **F1-macro:** 97.12%
- **F1-weighted:** 98.25%

**Matriz de confusión:**

```
                      Predicción
                 COVID  Normal  Viral
Verdadero COVID    441      10      1    (452 total)
Verdadero Normal     4   1,263      6  (1,273 total)  ← -1 muestra
Verdadero Viral      0      12    157    (169 total)
```

**Diferencia con conjunto original:**
- Cambio en soporte de Normal: 1,274 → 1,273 (duplicado removido)
- **Métricas globales idénticas:** 98.26% exactitud en ambos casos
- **Conclusión:** El duplicado no afectó los resultados

### 6.4 Verificación de Rango de Mejora

**Baseline esperado:** 97.68% (promedio de modelos individuales en CV)

**Resultado actual:** 98.26%

**Mejora:** +0.58 puntos porcentuales

**Rango esperado:** +0.5 a +1.0 puntos porcentuales (basado en experimentos previos)

**✓ Verificación:** La mejora está **dentro del rango esperado**, confirmando consistencia con resultados previos.

### 6.5 Errores de Clasificación

**Total de errores:** 33 muestras (1.74% del conjunto de prueba)

**Desglose por tipo de error:**

1. **COVID predicho como Normal:** 10 muestras
   - Posible causa: Opacidades muy tenues o imágenes tempranas de COVID

2. **COVID predicho como Viral:** 1 muestra
   - Patrón radiológico similar entre COVID y neumonía viral

3. **Normal predicho como COVID:** 4 muestras
   - Posibles artefactos o falsos negativos de la anotación original

4. **Normal predicho como Viral:** 6 muestras
   - Posibles variaciones anatómicas interpretadas como patología

5. **Viral predicho como Normal:** 12 muestras (error más frecuente)
   - Posible causa: Neumonías virales con opacidades muy leves

6. **Viral predicho como COVID:** 0 muestras
   - Indica buena separabilidad entre estas dos clases de neumonía

**Error más crítico:** Clasificar Viral como Normal (12 casos), ya que implica no detectar una patología.

**Error menos crítico:** Confundir COVID con Viral (1 caso), ambas requieren atención médica.

## 7. Integridad Metodológica

### 7.1 Verificación de Aislamiento del Conjunto de Prueba

Se implementaron **3 métodos independientes** de verificación para garantizar que el conjunto de prueba no fue utilizado durante el desarrollo:

**Método 1: Análisis de historial de entrenamiento**
- Inspeccionados los archivos `training_history.json` de los 5 folds de CV
- Verificación: Ninguna clave de métrica contiene la palabra "test"
- Claves encontradas: `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`
- ✓ PASADO

**Método 2: Análisis de archivos de configuración**
- Inspeccionados los archivos `config.json` de los 5 folds de CV
- Verificación: Todos tienen `eval_test: false`
- ✓ PASADO

**Método 3: Separación temporal**
- Fecha de los checkpoints de CV: 2026-01-16
- Fecha de evaluación final: 2026-02-16
- Los modelos se entrenaron **un mes antes** de evaluar en test
- ✓ PASADO

**Verificación adicional: git commit history**
- El código de evaluación final (`scripts/evaluate_final_ensemble_tta.py`) se creó el 2026-02-16
- Los checkpoints de CV no fueron modificados después de su creación
- ✓ PASADO

**Conclusión:** El conjunto de prueba permaneció completamente aislado hasta la evaluación final. Los resultados son válidos para inferencia sobre datos nunca vistos.

### 7.2 No Contaminación por Selección de Hiperparámetros

**Principio:** Ningún hiperparámetro se seleccionó basándose en el rendimiento en el conjunto de prueba.

**Validación:**
- **Pesos del ensamble:** Calculados usando **F1-macro de validación** (no de prueba)
- **Margen de expansión (1.05):** Optimizado en conjunto de validación durante Fase 0
- **CLAHE tile_size (4):** Seleccionado en experimentos con conjunto de validación
- **Arquitectura del modelo:** Definida antes de evaluar en prueba

**Resultado:** No hay riesgo de overfitting al conjunto de prueba debido a búsqueda de hiperparámetros.

### 7.3 Reproducibilidad Determinística

**Protocolo de verificación:**
1. Ejecutar evaluación final **dos veces** con el mismo código y checkpoints
2. Calcular hash SHA-256 del JSON de resultados (con `sort_keys=True`)
3. Comparar hashes entre ejecuciones

**Resultados:**

**Conjunto original (1,895 muestras):**
- Hash ejecución 1: `74655a817f1a731a...`
- Hash ejecución 2: `74655a817f1a731a...`
- ✓ IDÉNTICOS

**Conjunto limpio (1,894 muestras):**
- Hash ejecución 1: `1823e039d5a72d07...`
- Hash ejecución 2: `1823e039d5a72d07...`
- ✓ IDÉNTICOS

**Conclusión:** La evaluación es **completamente determinística**. Los mismos datos y modelos producen resultados bit-a-bit idénticos.

**Factores críticos para reproducibilidad:**
- `torch.manual_seed()` fija la semilla antes de cargar modelos
- `num_workers=0` en DataLoader (evita no-determinismo de multiprocesamiento)
- `shuffle=False` en DataLoader (orden fijo de muestras)

### 7.4 Reporte Transparente de Duplicados

En lugar de silenciar o eliminar el duplicado identificado, se adoptó un enfoque de **transparencia total**:

1. **Documentación:** El duplicado (`Normal-817`) se reporta explícitamente en todos los documentos
2. **Evaluación dual:** Se presentan resultados para ambos conjuntos (con y sin duplicado)
3. **Análisis de impacto:** Se demuestra que el duplicado no afectó las métricas (98.26% en ambos casos)

Este enfoque permite a revisores de tesis evaluar el impacto de problemas de calidad de datos.

## 8. Resumen de Resultados

### 8.1 Tabla de Métricas Clave

| Componente | Métrica | Valor |
|------------|---------|-------|
| **Landmarks** | Error medio (px) | 3.61 |
| **Landmarks** | Error Normal (px) | 3.22 |
| **Landmarks** | Error COVID (px) | 3.93 |
| **Landmarks** | Error Viral (px) | 4.11 |
| **Clasificador Individual** | Exactitud media (CV) | 97.68% ± 0.16% |
| **Ensamble sin TTA** | Exactitud | 98.10% |
| **Ensamble con TTA** | Exactitud | 98.26% |
| **Ensamble con TTA** | F1-macro | 97.12% |
| **Ensamble con TTA** | F1-weighted | 98.25% |
| **Mejora Total** | Sobre baseline | +0.58 pp |
| **Reducción de Error** | Relativa | 25% |

### 8.2 Progresión de Mejora

| Etapa | Exactitud | Mejora Incremental | Mejora Acumulada |
|-------|-----------|-------------------|------------------|
| Baseline (individual) | 97.68% | — | — |
| Ensamble (sin TTA) | 98.10% | +0.42 pp | +0.42 pp |
| Ensamble + TTA | 98.26% | +0.16 pp | +0.58 pp |

**Interpretación:**
- El **ensamble** aporta la mayor mejora (+0.42 pp), reduciendo varianza entre folds
- **TTA** aporta una mejora adicional (+0.16 pp), reduciendo varianza intra-modelo
- El efecto combinado es **aditivo**: 0.42 + 0.16 ≈ 0.58

### 8.3 Comparación con Estado del Arte

**Contexto:** El COVID-19 Radiography Dataset es ampliamente utilizado en la literatura, con reportes de exactitudes entre 95-99%.

**Posición de este trabajo:**
- Exactitud de **98.26%** en conjunto de prueba aislado
- Metodología completamente reproducible con protocolo de verificación
- Normalización geométrica como paso de preprocesamiento novedoso
- Ensamble dual-nivel (CV + TTA) para maximizar robustez

**Ventaja diferenciadora:** La combinación de normalización geométrica + ensamble + TTA no es común en la literatura COVID-19, y ofrece una ruta clara de mejora sobre modelos baseline.

### 8.4 Limitaciones Reconocidas

1. **Distribución de datos:**
   - Conjunto muy desbalanceado (67% Normal, 24% COVID, 9% Viral)
   - F1-macro (97.12%) inferior a exactitud (98.26%) debido a desbalance

2. **Landmarks manuales:**
   - Los 15 landmarks fueron anotados manualmente en un subconjunto de entrenamiento
   - Posible sesgo del anotador (aunque mitigado por promediado GPA)

3. **Dataset único:**
   - Resultados en COVID-19 Radiography Dataset pueden no generalizarse a otros hospitales
   - Domain shift conocido (validado en experimentos externos)

4. **Clase Viral difícil:**
   - F1-score de Viral (94.29%) notablemente inferior a COVID (98.33%) y Normal (98.75%)
   - Posible mejora con más datos de neumonía viral

### 8.5 Recomendación Final

**Para uso clínico en el mismo dominio (dataset similar):**
- **Modelo recomendado:** Ensamble de 5 modelos + TTA dual-nivel
- **Configuración:** Normalización geométrica con margin_scale=1.05, CLAHE (clip=2.0, tile=4)
- **Exactitud esperada:** 98.26% (validado)

**Para uso clínico en nuevos hospitales:**
- **Requerimiento:** Fine-tuning con datos locales (protocolo de adquisición diferente)
- **Protocolo:** Transfer learning desde checkpoints actuales, re-entrenar última capa
- **Validación:** Re-evaluar en conjunto de prueba local antes de despliegue

## 9. Configuración de Reproducción

### 9.1 Archivos de Configuración

**Landmarks ensemble:**
- `configs/ensemble_best.json` — 4 modelos (seeds 123, 321, 111, 666)

**Warping:**
- `configs/warping_best.json` — margin_scale=1.05, CLAHE enabled

**Clasificador CV:**
- `configs/ensemble_classifier.json` — 5 folds, checkpoint paths, expected sample counts

### 9.2 Checkpoints Requeridos

**Landmarks (4 modelos):**
- `checkpoints/session10/ensemble/seed123/final_model.pt`
- `checkpoints/session13/seed321/final_model.pt`
- `checkpoints/repro_split111/session14/seed111/final_model.pt`
- `checkpoints/repro_split666/session16/seed666/final_model.pt`

**Clasificadores (5 folds):**
- `outputs/classifier_cv/fold_01/best_classifier.pt`
- `outputs/classifier_cv/fold_02/best_classifier.pt`
- `outputs/classifier_cv/fold_03/best_classifier.pt`
- `outputs/classifier_cv/fold_04/best_classifier.pt`
- `outputs/classifier_cv/fold_05/best_classifier.pt`

### 9.3 Comando de Evaluación Final

```bash
python scripts/evaluate_final_ensemble_tta.py \
  --data-dir outputs/warped_lung_best/session_warping \
  --config configs/ensemble_classifier.json \
  --output-dir outputs/classifier_cv \
  --verify-test-isolation \
  --verify-reproducibility
```

**Salidas generadas:**
- `outputs/classifier_cv/final_evaluation_original.json`
- `outputs/classifier_cv/final_evaluation_cleaned.json`
- `outputs/classifier_cv/final_evaluation_reproducibility.json`

### 9.4 Métricas Canónicas

Todas las métricas validadas se almacenan en:
- **Archivo:** `GROUND_TRUTH.json`
- **Versión:** 2.2.0
- **Fecha:** 2026-02-16
- **Sección relevante:** `classification.final_evaluation`

## 10. Conclusiones

Este trabajo presenta una metodología completa para detección de COVID-19 en radiografías de tórax que combina:

1. **Detección robusta de landmarks** (3.61 px de error) mediante ensamble de 4 modelos ResNet-18 con Coordinate Attention y TTA
2. **Normalización geométrica** por deformación afín por piezas basada en GPA y triangulación de Delaunay
3. **Clasificación por ensamble** de 5 modelos de validación cruzada con votación suave ponderada
4. **Test-Time Augmentation dual-nivel** aplicada tanto a nivel de modelo como de ensamble

**Resultado final:** **98.26% de exactitud** en conjunto de prueba completamente aislado, con mejora de +0.58 puntos porcentuales sobre modelos individuales baseline (97.68%).

**Fortalezas metodológicas:**
- Aislamiento estricto del conjunto de prueba (verificado por 3 métodos independientes)
- Reproducibilidad determinística (verificada por comparación de hashes en doble ejecución)
- Transparencia total (reporte dual de resultados con y sin duplicado)
- Protocolos de verificación automáticos (conteos de clase, rangos esperados)

**Contribución principal:** Demostración de que la normalización geométrica automática basada en landmarks mejora la clasificación de patologías pulmonares, ofreciendo una alternativa sistemática al recorte manual de regiones de interés.

---

**Fecha de elaboración:** 2026-02-16
**Versión del documento:** 1.0
**Fuente de datos:** GROUND_TRUTH.json v2.2.0, outputs/classifier_cv/final_evaluation_*.json
**Reproducibilidad:** Todos los checkpoints y configuraciones disponibles en el repositorio del proyecto
