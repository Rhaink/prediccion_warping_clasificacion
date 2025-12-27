# PROMPT DE CONTINUACIÓN - SESIÓN 11: PRESENTACIÓN Y GENERACIÓN DE IMÁGENES CIENTÍFICAS

## INSTRUCCIONES PARA CLAUDE

**IMPORTANTE**: Antes de comenzar, lee completamente el archivo `Documentos/Tesis/prompts/prompt_tesis.md` ubicado en `/home/donrobot/Projects/prediccion_warping_clasificacion/Documentos/Tesis/prompts/prompt_tesis.md`. Este archivo contiene las reglas fundamentales del proyecto de tesis.

Este prompt introduce una **NUEVA FASE** del proyecto: transición de redacción textual a presentación visual y generación de imágenes científicas rigurosas.

---

## CONTEXTO DE LA SESIÓN ANTERIOR

**Fecha de sesión anterior:** 18 Diciembre 2025
**Actividad completada:** Eliminación completa de anglicismos en todo el Capítulo 4 de Metodología

### Cambios Realizados en Sesión Anterior

Se reemplazaron ~100 términos en inglés por sus equivalentes en español:
- dataset → conjunto de datos
- batch → lote
- learning rate → tasa de aprendizaje
- early stopping → parada temprana
- transfer learning → aprendizaje por transferencia
- fine-tuning → ajuste fino
- data augmentation → aumento de datos
- ground truth → valores de referencia
- baseline → línea base
- fill rate → tasa de llenado
- kernel → núcleo
- pipeline → flujo de procesamiento/secuencia de módulos/sistema

**Resultado:** Capítulo 4 completamente en español, compilación exitosa (16 páginas), sin anglicismos.

### Estado Actual del Proyecto de Tesis

| Fase | Capítulo | Estado | Progreso |
|------|----------|--------|----------|
| 1️⃣ | **Cap. 4 - Metodología** | ✅ **COMPLETADO** | Redacción completa en español, sin anglicismos |
| 2️⃣ | **Cap. 5 - Resultados** | ⏳ PENDIENTE | No iniciado |
| 3️⃣ | **Cap. 2, 3, 1** | ⏳ PENDIENTE | No iniciados |
| 4️⃣ | **Cap. 6 - Conclusiones** | ⏳ PENDIENTE | No iniciado |

---

## ARCHIVOS CLAVE DEL PROYECTO

### Archivos de Referencia Fundamentales

| Archivo | Contenido | Estado |
|---------|-----------|--------|
| `Documentos/Tesis/prompts/prompt_tesis.md` | Reglas fundamentales, rol de Claude, requerimientos de formato | ✅ VIGENTE |
| `DECISIONES_FASE_1.md` | Decisiones de estructura y enfoque aprobadas | ✅ BLOQUEADO |
| `ESTRUCTURA_TESIS.md` | Estructura de 6 capítulos aprobada | ✅ BLOQUEADO |
| `GROUND_TRUTH.json` | Valores experimentales validados (métricas, resultados) | ✅ REFERENCIA |
| `FIGURAS_PENDIENTES.md` | Lista de 24 figuras pendientes con especificaciones | 🔄 EN PROCESO |

### Archivos de LaTeX del Capítulo 4 (Completados)

| Archivo | Sección | Páginas |
|---------|---------|---------|
| `4_1_descripcion_general.tex` | Descripción General del Sistema | ~3 |
| `4_2_dataset_preprocesamiento.tex` | Dataset y Preprocesamiento | ~4 |
| `4_3_modelo_landmarks.tex` | Modelo de Predicción de Landmarks | ~6 |
| `4_4_normalizacion_geometrica.tex` | Normalización Geométrica | ~5 |
| `4_5_clasificacion.tex` | Clasificación | ~4 |
| `4_6_protocolo_evaluacion.tex` | Protocolo de Evaluación | ~4 |

---

## DECISIONES APROBADAS (BLOQUEADAS - NO MODIFICAR)

### Título de la Tesis (FIJO)

**"Normalización Geométrica de Radiografías de Tórax mediante Predicción de Landmarks para Clasificación de COVID-19"**

### 6 Objetivos Específicos Ajustados (BLOQUEADOS)

1. Implementar un modelo de predicción de 15 landmarks anatómicos en radiografías de tórax
2. Desarrollar un método de normalización geométrica mediante warping afín por partes
3. Entrenar clasificadores de COVID-19 sobre imágenes geométricamente normalizadas
4. Evaluar el impacto de la normalización en exactitud de clasificación y robustez ante perturbaciones
5. Comparar el desempeño entre imágenes originales y normalizadas mediante evaluación cruzada
6. Analizar la generalización del sistema mediante validación externa

### Claims Científicos Validados (USAR EN PRESENTACIÓN)

- ✅ Error de landmarks (ensemble + TTA): **3.71 px** (desv. 2.42 px)
- ✅ Accuracy clasificación (warped 96%): **99.10%**, F1-Macro: **98.45%**
- ✅ Mejora de robustez vs JPEG Q50: **5.27×**
- ✅ Mejora de robustez vs JPEG Q30: **5.68×**
- ✅ Mejora de robustez vs Blur σ=1: **5.94×**
- ✅ Factor de mejora de generalización (cross-eval): **2.43×**
- ✅ Warped 96% recomendado como configuración óptima

### Claims INVALIDADOS (NO USAR NUNCA)

- ❌ NO "11× mejor generalización" → Correcto: "2.4×"
- ❌ NO "Fuerza atención en región pulmonar" → PFS ≈ 0.49 (50%, aleatorio)
- ❌ NO "Resuelve domain shift" → External validation ≈ 55% (problema de dominio persistente)

---

## OBJETIVO DE ESTA SESIÓN

### NUEVA FASE: Transformación a Presentación Visual + Generación de Imágenes Científicas Rigurosas

Esta sesión marca la transición de:
- ❌ Redacción textual → ✅ Presentación visual ejecutiva
- ❌ Figuras pendientes con placeholders → ✅ Imágenes científicas generables
- ❌ Capítulo extenso (26 páginas) → ✅ Slides concisos con método assertive-evidence

### Alcance de la Sesión

1. **Diseñar estructura de presentación** usando el método assertive-evidence
2. **Generar especificaciones rigurosas** para las 24 figuras pendientes
3. **Crear código/scripts Python** para generar imágenes reproducibles
4. **Vincular datos de GROUND_TRUTH.json** con visualizaciones
5. **Garantizar rigor científico**: Sin errores, sin superposiciones, con datos reales validados

---

## MÉTODO ASSERTIVE-EVIDENCE

### Definición

El método **Assertive-Evidence** (diseñado por Michael Alley, Penn State University) estructura presentaciones científicas mediante:

1. **Slide Headline = Claim Assertivo**: Cada slide tiene un título que es una afirmación completa (no solo un tema)
2. **Body = Evidencia Visual**: El contenido visual demuestra el claim del título
3. **Simplicidad**: Una idea principal por slide
4. **Enfoque Visual**: Gráficas, diagramas y figuras predominan sobre texto

### Ejemplo de Estructura

#### ❌ Enfoque Tradicional (Topic-Based):
**Título**: "Normalización Geométrica"
**Contenido**: Viñetas de texto explicando el proceso

#### ✅ Enfoque Assertive-Evidence:
**Título**: "El warping afín por partes alinea la región pulmonar a una forma canónica"
**Contenido**: Diagrama mostrando: imagen original → triangulación → imagen warped, con flechas y datos cuantitativos (fill rate 47% → 96%)

### Aplicación al Proyecto

**Presentación sugerida**: 15-20 slides organizados en 5 bloques

1. **Introducción** (2 slides)
   - Claim: "COVID-19 requiere métodos automatizados de diagnóstico por imagen"
   - Claim: "La variabilidad geométrica dificulta la clasificación automática"

2. **Metodología** (8-10 slides)
   - Claim: "El sistema opera en dos fases: preparación offline y operación runtime"
   - Claim: "15 landmarks definen el contorno pulmonar bilateral"
   - Claim: "GPA elimina traslación, escala y rotación mediante alineación iterativa"
   - Claim: "El warping con cobertura completa alcanza 96% de tasa de llenado"
   - Claim: "El ensemble de 4 modelos reduce el error a 3.71 píxeles"

3. **Resultados** (4-5 slides)
   - Claim: "La normalización geométrica alcanza 99.10% de accuracy"
   - Claim: "La normalización geométrica mejora la robustez 5.94× ante blur"
   - Claim: "El clasificador warped generaliza 2.43× mejor entre dominios"

4. **Análisis** (2-3 slides)
   - Claim: "La normalización no resuelve el domain shift externo (55% en FedCOVIDx)"
   - Claim: "PFS ≈ 50% indica que el modelo no fuerza atención pulmonar"

5. **Conclusiones** (1-2 slides)
   - Claim: "El warping mejora robustez intra-dominio pero no generalización inter-institucional"

---

## FIGURAS PENDIENTES A CREAR (24 FIGURAS TOTALES)

### PRIORIDAD CRÍTICA (12 figuras fundamentales)

#### **Bloque 1: Descripción General del Sistema**

**F4.1 - Diagrama de fases del sistema**
- **Ubicación LaTeX**: `4_1_descripcion_general.tex` línea 20
- **Descripción**: Dos bloques diferenciados por color
  - **Fase de Preparación (offline)**: Anotación manual → Entrenamiento de modelos → Cálculo de forma canónica (GPA)
  - **Fase de Operación (runtime)**: Imagen nueva → Secuencia de 4 módulos → Clasificación
- **Especificaciones técnicas**:
  - Formato: Diagrama de bloques con flechas direccionales
  - Colores: Azul (#3498db) para preparación, Verde (#27ae60) para operación
  - Incluir datos: 957 imágenes anotadas, 4 modelos ensemble, forma canónica
  - Dimensiones: 1920×1080 px, 300 DPI
- **Herramientas**: matplotlib + networkx o draw.io/Inkscape

**F4.2 - Diagrama de bloques del flujo de operación**
- **Ubicación LaTeX**: `4_1_descripcion_general.tex` línea 39
- **Descripción**: Flujo secuencial de 4 módulos con dimensiones de datos
- **Especificaciones técnicas**:
  - **Entrada**: 224×224×3 (RGB)
  - **Módulo 1 (CLAHE)**: 224×224×3 → 224×224×3
  - **Módulo 2 (Predicción Landmarks)**: 224×224×3 → 15×2
  - **Módulo 3 (Warping Geométrico)**: (224×224×3, 15×2) → 224×224×3
  - **Módulo 4 (Clasificación)**: 224×224×3 → 3 (COVID, Normal, Viral)
  - Incluir iconos representativos para cada módulo
  - Dimensiones: 2400×800 px, 300 DPI
- **Datos**: Usar ejemplo real del dataset con landmarks superpuestos

#### **Bloque 2: Dataset y Preprocesamiento**

**F4.2a - Diagrama de 15 landmarks sobre radiografía**
- **Ubicación LaTeX**: `4_2_dataset_preprocesamiento.tex` línea 56
- **Descripción**: Radiografía con 15 landmarks numerados y coloreados por grupos
- **Especificaciones técnicas**:
  - **Eje central**: L1 (ápex) → L9, L10, L11 → L2 (base) [Color: Rojo #e74c3c]
  - **Contorno izquierdo**: L12 → L3 → L5 → L7 → L14 [Color: Verde #2ecc71]
  - **Contorno derecho**: L13 → L4 → L6 → L8 → L15 [Color: Azul #3498db]
  - Pares simétricos conectados con líneas punteadas (alpha=0.5)
  - Números de landmark claramente visibles: fuente ≥ 14pt, fondo blanco semitransparente
  - Dimensiones: 800×800 px, 300 DPI
- **Datos**: Usar coordenadas de `data/coordenadas/coordenadas_maestro.csv` (ejemplo representativo)
- **Script**: `src_v2/visualization/` o script dedicado

**F4.2b - Interfaz de herramienta de etiquetado**
- **Ubicación LaTeX**: `4_2_dataset_preprocesamiento.tex` línea 116
- **Descripción**: Captura de pantalla de la herramienta OpenCV de anotación
- **Especificaciones técnicas**:
  - Radiografía con línea central azul vertical
  - Puntos verdes numerados (L1-L15) con radio 5px
  - Líneas rojas conectando contorno pulmonar
  - Menú de teclas visible en esquina (instrucciones de ajuste horizontal)
  - Dimensiones: 1200×900 px, 300 DPI
- **Datos**: Screenshot real de la herramienta o mockup fiel generado con cv2

**F4.3 - Comparación CLAHE (antes/después)**
- **Ubicación LaTeX**: `4_2_dataset_preprocesamiento.tex` línea 218
- **Descripción**: Panel lado a lado mostrando efecto de mejora de contraste
- **Especificaciones técnicas**:
  - **(a) Imagen original** con bajo contraste en región pulmonar
  - **(b) Imagen con CLAHE aplicado**: clip_limit=2.0, tile_size=(4,4)
  - Misma imagen de entrada, misma escala de grises
  - Etiquetas de parámetros visibles: "Original" vs "CLAHE (limit=2.0, tile=4×4)"
  - Layout: 2 columnas, dimensiones: 1600×800 px, 300 DPI
- **Datos**: Seleccionar imagen representativa del dataset con bajo contraste inicial
- **Script**: Aplicar `src_v2/data/transforms.py` (función apply_clahe)

#### **Bloque 3: Normalización Geométrica (CORE del trabajo)**

**F4.6 - Proceso de GPA (panel de 4 subfiguras)**
- **Ubicación LaTeX**: `4_4_normalizacion_geometrica.tex` línea 137
- **Descripción**: Panel de 4 subfiguras mostrando transformación progresiva de 957 configuraciones
- **Especificaciones técnicas**:
  - **(a) Configuraciones originales**: 957 landmarks superpuestos (alta variabilidad en posición/escala/rotación)
  - **(b) Después de centrado y escalado**: Origen común (0,0), norma unitaria
  - **(c) Después de alineación rotacional**: Variabilidad mínima
  - **(d) Forma canónica final**: Consenso de Procrustes (media de configuraciones alineadas)
  - Usar scatter plots con alpha=0.1 para visualizar densidad
  - Ejes: -0.5 a 0.5 (coordenadas normalizadas)
  - Layout: Grid 2×2, dimensiones totales: 1600×1600 px, 300 DPI
- **Datos**: Generar desde `data/coordenadas/coordenadas_maestro.csv` aplicando `src_v2/processing/gpa.py`
- **Script**: Implementar visualización de cada paso del algoritmo GPA

**F4.7 - Triangulación de Delaunay**
- **Ubicación LaTeX**: `4_4_normalizacion_geometrica.tex` línea 173
- **Descripción**: Forma canónica con ~20-25 triángulos de Delaunay
- **Especificaciones técnicas**:
  - 15 landmarks como puntos rojos: radio 5px, borde negro
  - Triángulos con bordes negros: grosor 1px
  - Relleno de triángulos con colores alternados (azul/verde claro, alpha=0.3) para claridad
  - Ejes con dimensiones de imagen: 0-224 (píxeles)
  - Dimensiones: 800×800 px, 300 DPI
- **Datos**: Calcular triangulación con `scipy.spatial.Delaunay` sobre forma canónica de GPA
- **Script**: Usar forma canónica + `scipy.spatial.Delaunay` + matplotlib

**F4.8 - Comparación Original vs Warped (panel de 3)**
- **Ubicación LaTeX**: `4_4_normalizacion_geometrica.tex` línea 287
- **Descripción**: Panel de 3 imágenes mostrando diferencia de cobertura
- **Especificaciones técnicas**:
  - **(a) Imagen original** con variabilidad de pose/escala
  - **(b) Warped SIN cobertura completa**: use_full_coverage=False, tasa de llenado ≈ 47%, esquinas negras visibles
  - **(c) Warped CON cobertura completa**: use_full_coverage=True, tasa de llenado ≈ 96%, sin esquinas negras
  - Misma imagen de entrada para (a), (b), (c)
  - Etiquetas de tasa de llenado visibles en cada subfigura
  - Layout: 3 columnas, dimensiones totales: 2400×800 px, 300 DPI
- **Datos**: Usar `src_v2/processing/warp.py` con/sin parámetro use_full_coverage
- **Script**: Aplicar warping en ambas configuraciones y calcular fill rate

**F4.9 - Efecto de margin_scale (panel de 3)**
- **Ubicación LaTeX**: `4_4_normalizacion_geometrica.tex` (mencionado en texto, sin línea específica)
- **Descripción**: Comparación de imágenes warped con diferentes valores de margin_scale
- **Especificaciones técnicas**:
  - **(a) margin_scale = 1.00**: Sin margen adicional, región puede quedar recortada
  - **(b) margin_scale = 1.05**: Valor óptimo (balanceado)
  - **(c) margin_scale = 1.25**: Margen excesivo, incluye regiones periféricas irrelevantes
  - Misma imagen de entrada
  - Etiquetas de margin_scale visibles
  - Layout: 3 columnas, dimensiones: 2400×800 px, 300 DPI
- **Datos**: Usar `src_v2/processing/warp.py` variando parámetro margin_scale
- **Script**: Generar 3 versiones de misma imagen con diferentes margin_scale

**F4.10 - Diagrama de flujo de normalización geométrica (6 pasos)**
- **Ubicación LaTeX**: `4_4_normalizacion_geometrica.tex` (mencionado en texto)
- **Descripción**: Diagrama de flujo mostrando proceso completo paso a paso
- **Especificaciones técnicas**:
  - **Paso 1**: Predicción de landmarks (224×224×3 → 15×2)
  - **Paso 2**: Escalado con margin_scale (15×2 → 15×2 escalados)
  - **Paso 3**: Adición de puntos de borde (+8 puntos → 23 puntos totales)
  - **Paso 4**: Triangulación Delaunay (23 puntos → ~35-40 triángulos)
  - **Paso 5**: Warping afín por partes (imagen + triángulos → imagen warped)
  - **Paso 6**: Imagen normalizada (224×224×3 alineada a forma canónica)
  - Incluir dimensiones de datos en cada paso
  - Flechas direccionales con etiquetas descriptivas
  - Dimensiones: 2000×1200 px, 300 DPI
- **Herramientas**: matplotlib + networkx o diagrama manual con Inkscape

#### **Bloque 4: Protocolo de Evaluación**

**F4.13 - Esquema de evaluación cruzada (matriz 2×2)**
- **Ubicación LaTeX**: `4_6_protocolo_evaluacion.tex` (mencionado en texto)
- **Descripción**: Matriz mostrando 4 combinaciones de entrenamiento/evaluación
- **Especificaciones técnicas**:
  - **Filas**: Entrenado en [Original, Warped]
  - **Columnas**: Evaluado en [Original, Warped]
  - **Celdas**:
    - $Acc_{O→O}$: 98.84% (in-domain, diagonal)
    - $Acc_{O→W}$: 91.13% (cross-domain, off-diagonal)
    - $Acc_{W→O}$: 95.57% (cross-domain, off-diagonal)
    - $Acc_{W→W}$: 98.73% (in-domain, diagonal)
  - Diferenciar in-domain (fondo verde) vs cross-domain (fondo amarillo)
  - Incluir gaps de generalización: 7.70% vs 3.17%
  - Dimensiones: 1200×1200 px, 300 DPI
- **Datos**: Extraer de `GROUND_TRUTH.json` → sección "cross_evaluation_summary"
- **Script**: Tabla estilizada con matplotlib o seaborn heatmap

**F4.14 - Perturbaciones de robustez (panel de 5)**
- **Ubicación LaTeX**: `4_6_protocolo_evaluacion.tex` línea 298
- **Descripción**: Panel mostrando efecto visual de perturbaciones
- **Especificaciones técnicas**:
  - **(a) Original**: Sin perturbación
  - **(b) JPEG Q=50**: Compresión moderada
  - **(c) JPEG Q=30**: Compresión severa
  - **(d) Blur σ=1**: Desenfoque leve (kernel automático)
  - **(e) Blur σ=2**: Desenfoque moderado (kernel automático)
  - Misma imagen de entrada para todas las variantes
  - Parámetros visibles en cada subfigura
  - Layout: 5 columnas o grid 2×3, dimensiones: 2400×960 px, 300 DPI
- **Datos**: Aplicar `cv2.GaussianBlur` (σ=1, σ=2) y `PIL.Image.save(quality=50/30)` a imagen real
- **Script**: Generar perturbaciones y visualizar lado a lado

---

### PRIORIDAD MEDIA (8 figuras complementarias)

**F4.4 - Arquitectura ResNet-18 + Coordinate Attention**
- **Ubicación**: `4_3_modelo_landmarks.tex` línea 18
- **Descripción**: Diagrama de arquitectura mostrando backbone + módulo de atención + cabeza de regresión
- **Dimensiones sugeridas**: 2400×1000 px, 300 DPI

**F4.5 - Detalle del módulo Coordinate Attention**
- **Ubicación**: `4_3_modelo_landmarks.tex` línea 232
- **Descripción**: Zoom en módulo de atención (4 fases: pooling H/W → transformación → generación de mapas → aplicación)
- **Dimensiones sugeridas**: 1600×800 px, 300 DPI

**F4.11 - Ejemplos de aumento de datos (panel de 4)**
- **Ubicación**: `4_5_clasificacion.tex` línea 258
- **Descripción**: Transformaciones aplicadas durante entrenamiento
  - (a) Original
  - (b) Flip horizontal
  - (c) Rotación ±10°
  - (d) Traslación+escala
- **Dimensiones sugeridas**: 1600×1600 px, 300 DPI

---

### PRIORIDAD BAJA (4 figuras opcionales)

**F4.12 - Arquitectura del clasificador**
- **Ubicación**: `4_5_clasificacion.tex` (no tiene placeholder explícito)
- **Descripción**: Opcional, puede omitirse o simplificar como diagrama de bloques

---

## DATOS DISPONIBLES EN GROUND_TRUTH.json

### Métricas de Landmarks (para gráficas de barras/scatter)

```json
"ensemble_4models_tta": {
  "overall": {
    "mean_error_px": 3.71,
    "std_error_px": 2.42,
    "median_error_px": 3.15
  },
  "per_category": {
    "COVID-19": {"mean_error_px": 3.77, "std_error_px": 2.51},
    "Normal": {"mean_error_px": 3.42, "std_error_px": 2.24},
    "Viral_Pneumonia": {"mean_error_px": 4.40, "std_error_px": 2.76}
  },
  "per_landmark": {
    "L1": 3.73, "L2": 3.46, "L3": 3.28, "L4": 3.22,
    "L5": 2.97, "L6": 3.01, "L7": 3.26, "L8": 3.32,
    "L9": 2.84, "L10": 2.57, "L11": 3.03,
    "L12": 5.50, "L13": 5.21, "L14": 4.63, "L15": 4.45
  }
}
```

### Clasificación - Rendimiento por Configuración

```json
"baseline_original_100": {
  "accuracy": 0.9884,
  "f1_macro": 0.9816,
  "f1_weighted": 0.9884
},
"warped_96_recommended": {
  "accuracy": 0.9910,
  "f1_macro": 0.9845,
  "f1_weighted": 0.9910
},
"warped_99": {
  "accuracy": 0.9873,
  "f1_macro": 0.9795,
  "f1_weighted": 0.9873
}
```

### Robustez bajo Perturbaciones (para gráficas de barras comparativas)

```json
"robustness": {
  "degradation_jpeg_q50": {
    "original_100": 0.1614,
    "warped_96": 0.0306,
    "improvement_factor": 5.27
  },
  "degradation_jpeg_q30": {
    "original_100": 0.2997,
    "warped_96": 0.0528,
    "improvement_factor": 5.68
  },
  "degradation_blur_sigma1": {
    "original_100": 0.1443,
    "warped_96": 0.0243,
    "improvement_factor": 5.94
  },
  "degradation_blur_sigma2": {
    "original_100": 0.3185,
    "warped_96": 0.0671,
    "improvement_factor": 4.75
  }
}
```

### Cross-Evaluation (para matriz F4.13)

```json
"cross_evaluation_summary": {
  "trained_on_original": {
    "eval_on_original": 0.9884,
    "eval_on_warped": 0.9113,
    "generalization_gap": 0.0770
  },
  "trained_on_warped": {
    "eval_on_original": 0.9557,
    "eval_on_warped": 0.9873,
    "generalization_gap": 0.0317
  },
  "improvement_factor": 2.43
}
```

### Validación Externa (FedCOVIDx)

```json
"external_validation_fedcovidx": {
  "warped_96_on_d3_original": {
    "accuracy": 0.5336,
    "internal_gap": -0.4574
  },
  "warped_96_on_d3_warped": {
    "accuracy": 0.5531,
    "internal_gap": -0.4379
  }
}
```

### PFS (Pulmonary Focus Score)

```json
"pfs_analysis": {
  "mean_pfs": 0.487,
  "interpretation": "No hay foco preferencial en región pulmonar (≈50% aleatorio)"
}
```

---

## PROCESO SUGERIDO PARA ESTA SESIÓN

### Paso 1: Diseño de Estructura de Presentación (30-45 min)

1. **Crear outline** de 15-20 slides con claims assertivos (no tópicos)
2. **Mapear** cada claim a figuras específicas (de las 24)
3. **Validar estructura** con usuario antes de generar imágenes

**Entregable**: Documento Markdown con outline completo

### Paso 2: Priorización de Figuras (15 min)

1. **Identificar** las 12 figuras críticas que DEBEN generarse primero
2. **Confirmar** disponibilidad de datos para cada figura
3. **Definir** orden de generación (empezar por F4.1, F4.2, F4.2a, F4.3)

**Entregable**: Lista priorizada de figuras a generar

### Paso 3: Generación de Scripts Python (2-4 horas)

Para cada figura:
1. **Crear script** independiente `generate_figure_FX_Y.py`
2. **Usar bibliotecas**:
   - **Visualización**: matplotlib, seaborn
   - **Procesamiento**: opencv-python (cv2), PIL
   - **Datos**: numpy, pandas, scipy
   - **Diagramas** (opcional): networkx, graphviz
3. **Cargar datos** de:
   - `data/coordenadas/coordenadas_maestro.csv`
   - `GROUND_TRUTH.json`
   - Imágenes reales del dataset
   - Checkpoints de modelos (si necesario)
4. **Exportar** en alta resolución:
   - Formato: PNG (con transparencia) o PDF (vectorial)
   - Resolución: 300 DPI mínimo
   - Dimensiones: Según especificaciones de cada figura
5. **Incluir comentarios** explicando cada paso del código

**Entregable**: Directorio `scripts/figures/` con scripts independientes

### Paso 4: Validación Científica (30-45 min)

Para cada figura generada:
1. **Verificar** que datos numéricos coinciden con GROUND_TRUTH.json
2. **Revisar** etiquetas de ejes, leyendas, títulos (español, sin anglicismos)
3. **Confirmar** que no hay superposiciones ilegibles de texto o elementos
4. **Validar** rigor científico:
   - Unidades correctas (píxeles, porcentajes, accuracy)
   - Escalas apropiadas
   - Precisión decimal consistente (2 decimales para accuracy, 2 decimales para error px)
5. **Verificar** accesibilidad:
   - Colores distinguibles (evitar rojo-verde puro)
   - Contraste suficiente para proyección
   - Fuentes legibles (≥ 10pt)

**Entregable**: Checklist de validación completado para cada figura

### Paso 5: Integración con LaTeX (30-45 min)

1. **Reemplazar** placeholders `\fbox` con `\includegraphics`
2. **Ajustar** captions para coherencia con figuras reales
3. **Compilar** LaTeX y verificar que figuras se integran correctamente
4. **Revisar** referencias cruzadas (`\ref{fig:...}`) en el texto

**Entregable**: Archivos .tex actualizados con figuras reales

---

## ARCHIVOS DE CÓDIGO RELEVANTES PARA GENERACIÓN

### Scripts de Procesamiento Existentes
- `src_v2/processing/warp.py` - Implementación de warping afín por partes
- `src_v2/processing/gpa.py` - Implementación de GPA (Generalized Procrustes Analysis)
- `src_v2/data/dataset.py` - Carga de datos y aplicación de CLAHE
- `src_v2/data/transforms.py` - Transformaciones de aumento de datos
- `src_v2/visualization/gradcam.py` - Visualizaciones Grad-CAM existentes
- `src_v2/visualization/pfs_analysis.py` - Análisis de Pulmonary Focus Score

### Datos
- `data/coordenadas/coordenadas_maestro.csv` - 957 configuraciones de landmarks anotadas
- `GROUND_TRUTH.json` - Métricas experimentales validadas
- `configs/final_config.json` - Hiperparámetros de entrenamiento
- `checkpoints/ensemble/` - Modelos ensemble (seeds 123, 456, 321, 789)

### Estructura de Salida Sugerida

```
Documentos/Tesis/
├── figures/
│   ├── generated/               # Figuras generadas por scripts
│   │   ├── F4_1_diagrama_fases_sistema.png
│   │   ├── F4_2_flujo_operacion.png
│   │   ├── F4_2a_landmarks_anatomicos.png
│   │   ├── F4_3_clahe_comparison.png
│   │   ├── F4_6_gpa_proceso.png
│   │   ├── F4_7_triangulacion_delaunay.png
│   │   ├── F4_8_warping_comparison.png
│   │   ├── F4_9_margin_scale_effect.png
│   │   ├── F4_10_normalizacion_pipeline.png
│   │   ├── F4_13_cross_evaluation_matrix.png
│   │   └── F4_14_perturbaciones_robustez.png
│   └── scripts/                 # Scripts Python para generación
│       ├── generate_F4_1.py
│       ├── generate_F4_2.py
│       ├── generate_F4_2a.py
│       └── ...
└── DOCUMENTACION_FIGURAS.md     # Documentación de cada figura
```

---

## RECORDATORIOS CRÍTICOS

### Rigor Científico (OBLIGATORIO)

1. ✅ Usar **SOLO** datos de `GROUND_TRUTH.json` (validados experimentalmente)
2. ✅ Verificar **unidades** en todas las visualizaciones:
   - Error de landmarks: **píxeles (px)**
   - Accuracy, F1: **porcentaje (%) o decimal con 4 cifras (0.9910)**
   - Fill rate: **porcentaje (%)**
   - Degradación: **porcentaje (%) o decimal**
3. ✅ Incluir **barras de error** cuando aplique:
   - Desviación estándar para error de landmarks
   - Intervalos de confianza para accuracy (si disponibles)
4. ✅ **Etiquetas** siempre en español:
   - Ejes: "Error (px)", "Tasa de Llenado (%)", "Accuracy"
   - Leyendas: "Original", "Warped", "COVID-19", "Normal", "Neumonía Viral"
5. ❌ **NO inventar** datos ni aproximaciones visuales
6. ❌ **NO usar** claims invalidados en ninguna figura

### Calidad Visual (OBLIGATORIO)

1. ✅ **Resolución mínima**: 300 DPI para todas las figuras
2. ✅ **Fuentes legibles**:
   - Texto general: ≥ 10pt
   - Títulos de subfiguras: ≥ 12pt
   - Etiquetas de ejes: ≥ 11pt
3. ✅ **Colores accesibles**:
   - Evitar combinaciones rojo-verde puro (daltonismo)
   - Usar paletas perceptually uniform: viridis, plasma, cividis
   - Asegurar contraste suficiente para proyección
4. ✅ **Sin superposiciones**:
   - Ajustar layout si elementos se sobreponen
   - Usar `bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)` para etiquetas sobre imágenes
5. ✅ **Consistencia visual**:
   - Mismo estilo de fuente en todas las figuras (sans-serif recomendado: Arial, Helvetica)
   - Mismo esquema de colores para elementos equivalentes
   - Mismo grosor de líneas (1-2 pt típicamente)

### Reproducibilidad (OBLIGATORIO)

1. ✅ **Scripts autocontenidos**:
   - Cada script debe ejecutarse independientemente
   - Incluir imports completos al inicio
   - Manejar rutas de archivos con `pathlib` o variables configurables
2. ✅ **Seeds aleatorias fijas**: `seed=42` para todas las operaciones estocásticas
3. ✅ **Documentar dependencias**:
   - Versiones de bibliotecas usadas (matplotlib, numpy, opencv-python, etc.)
   - Incluir `requirements.txt` o sección de dependencias en README
4. ✅ **Instrucciones de ejecución**:
   - Comentarios al inicio de cada script explicando:
     - Qué figura genera
     - Datos de entrada necesarios
     - Comandos de ejecución
     - Salida esperada (archivo PNG/PDF)

### Formato de Código Python (RECOMENDADO)

```python
"""
Script: generate_F4_X_descripcion.py
Genera: Figura F4.X - [Descripción breve]
Datos: [Fuentes de datos]
Salida: figures/generated/F4_X_nombre.png (300 DPI)

Ejecución:
    python generate_F4_X_descripcion.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path

# Configuración global
SEED = 42
DPI = 300
OUTPUT_DIR = Path("figures/generated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)

# Cargar datos
# ...

# Generar figura
fig, ax = plt.subplots(figsize=(8, 6))
# ...

# Guardar
output_path = OUTPUT_DIR / "F4_X_nombre.png"
fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
print(f"Figura guardada en: {output_path}")
```

---

## FORMATO ESPERADO DE ENTREGA

### 1. Estructura de Presentación (Markdown)

Crear archivo: `PRESENTACION_METODOLOGIA.md`

```markdown
# PRESENTACIÓN: Normalización Geométrica para Clasificación de COVID-19
**Método Assertive-Evidence**

## Slide 1: [Claim assertivo completo]
- **Figura**: F4.X
- **Evidencia visual**: [Descripción de qué muestra]
- **Datos cuantitativos**: [Valores específicos de GROUND_TRUTH.json]
- **Mensaje clave**: [Conclusión que debe recordar la audiencia]

## Slide 2: [Claim assertivo completo]
...
```

### 2. Scripts de Generación (Python)

Crear directorio: `figures/scripts/`

```
figures/scripts/
├── generate_F4_1.py
├── generate_F4_2.py
├── generate_F4_2a.py
├── generate_F4_3.py
├── generate_F4_6.py
├── generate_F4_7.py
├── generate_F4_8.py
├── generate_F4_9.py
├── generate_F4_10.py
├── generate_F4_13.py
├── generate_F4_14.py
├── requirements.txt
└── README.md
```

### 3. Figuras Generadas (PNG/PDF)

Crear directorio: `figures/generated/`

```
figures/generated/
├── F4_1_diagrama_fases_sistema.png
├── F4_2_flujo_operacion.png
├── F4_2a_landmarks_anatomicos.png
├── F4_3_clahe_comparison.png
├── F4_6_gpa_proceso.png
├── F4_7_triangulacion_delaunay.png
├── F4_8_warping_comparison.png
├── F4_9_margin_scale_effect.png
├── F4_10_normalizacion_pipeline.png
├── F4_13_cross_evaluation_matrix.png
└── F4_14_perturbaciones_robustez.png
```

### 4. Documentación de Figuras (Markdown)

Crear archivo: `DOCUMENTACION_FIGURAS.md`

```markdown
# DOCUMENTACIÓN DE FIGURAS GENERADAS

## F4.1 - Diagrama de Fases del Sistema
- **Script**: `figures/scripts/generate_F4_1.py`
- **Datos usados**: N/A (diagrama conceptual)
- **Herramientas**: matplotlib + networkx
- **Validación**: ✅ Aprobado por usuario (fecha)
- **Resolución**: 300 DPI, 1920×1080 px
- **Observaciones**: [Notas adicionales]

## F4.2 - Flujo de Operación
...
```

---

## COMANDO INICIAL SUGERIDO

```
Hola Claude, vamos a trabajar en crear una presentación de la metodología usando el método assertive-evidence y generar las figuras científicas pendientes.

Por favor:
1. Lee este prompt completo (PROMPT_CONTINUACION_SESION_11.md)
2. Confirma que entiendes el método assertive-evidence y sus diferencias con presentaciones tradicionales
3. Propón un outline de 15-20 slides con claims assertivos para cada uno (no tópicos, sino afirmaciones completas)
4. Mapea cada slide a las figuras que necesitamos generar (de las 24 documentadas)

Una vez aprobado el outline, comenzaremos a generar los scripts Python para las figuras prioritarias en este orden:
1. F4.1 (Diagrama de fases)
2. F4.2 (Flujo de operación)
3. F4.2a (15 landmarks)
4. F4.3 (CLAHE before/after)
5. F4.6 (Proceso GPA)
6. F4.7 (Triangulación Delaunay)
7. F4.8 (Comparación warping)
8. F4.13 (Matriz cross-evaluation)
9. F4.14 (Perturbaciones)

Prioridad: Rigor científico absoluto. Cada figura debe ser reproducible y verificable contra GROUND_TRUTH.json.
```

---

## NOTAS ADICIONALES

### Método Assertive-Evidence - Referencias

- **Libro**: "The Craft of Scientific Presentations" (Michael Alley, 2013, 2nd Edition)
- **Principio clave**: "El título del slide es la conclusión, el cuerpo es la evidencia que la demuestra"
- **Beneficio**: Audiencia comprende mensaje principal sin leer viñetas de texto
- **Estructura típica**:
  - Headline assertivo (afirmación completa, no fragmento)
  - Visual dominante (ocupa 60-80% del slide)
  - Texto mínimo (solo etiquetas, datos cuantitativos esenciales)

### Figuras Científicas - Best Practices

- **Simplicidad**: Una idea visual por figura
- **Contraste**: Alto contraste para proyección (evitar grises claros sobre blanco)
- **Etiquetas**: Descriptivas pero concisas (≤ 8 palabras típicamente)
- **Colormaps**:
  - Datos continuos: viridis, plasma, cividis (perceptually uniform)
  - Datos categóricos: tab10, Set2, Paired
  - Evitar: jet, rainbow (distorsionan percepción de datos)
- **Legends vs Annotations**:
  - Preferir anotaciones directas sobre elementos
  - Usar leyendas solo si >4 categorías

### Herramientas Recomendadas

#### Diagramas de Flujo/Arquitectura
- **matplotlib + networkx**: Diagramas programáticos (Python)
- **draw.io / diagrams.net**: Diagramas interactivos (GUI, exporta PNG/SVG)
- **Inkscape**: Gráficos vectoriales (SVG, exporta PDF)

#### Visualización de Datos
- **matplotlib**: Biblioteca base (scatter, line, bar, heatmap)
- **seaborn**: Estilización y gráficas estadísticas
- **plotly**: Visualizaciones interactivas (opcional, para versión web)

#### Procesamiento de Imágenes
- **opencv-python (cv2)**: Operaciones de visión por computadora
- **PIL / Pillow**: Manipulación básica de imágenes
- **scikit-image**: Filtros y transformaciones avanzadas

#### Layouts Complejos
- **matplotlib.gridspec**: Grids no uniformes
- **matplotlib subfigures**: Paneles anidados (matplotlib ≥ 3.4)

### Paletas de Colores Sugeridas

Para consistencia visual en todas las figuras:

```python
# Paleta principal (categorías diagnósticas)
COLORS = {
    'COVID-19': '#e74c3c',      # Rojo
    'Normal': '#27ae60',         # Verde
    'Viral_Pneumonia': '#f39c12' # Naranja
}

# Paleta para fases (F4.1)
PHASES = {
    'Preparacion': '#3498db',    # Azul
    'Operacion': '#27ae60'       # Verde
}

# Paleta para landmarks (F4.2a)
LANDMARKS = {
    'Eje': '#e74c3c',            # Rojo
    'Izquierdo': '#2ecc71',      # Verde claro
    'Derecho': '#3498db'         # Azul
}
```

---

**Prompt generado:** 18 Diciembre 2025 - Sesión 11
**Objetivo:** Crear presentación con método assertive-evidence y generar 24 figuras científicas rigurosas
**Prioridad:** 12 figuras críticas (F4.1, F4.2, F4.2a, F4.3, F4.6, F4.7, F4.8, F4.9, F4.10, F4.13, F4.14)
**Enfoque:** Rigor científico absoluto, reproducibilidad, sin errores, sin superposiciones
