# PROMPT DE CONTINUACIÓN - SESIÓN 07 DE REDACCIÓN DE TESIS

## INSTRUCCIONES PARA CLAUDE

Lee el archivo `Documentos/Tesis/prompts/prompt_tesis.md` para entender tu rol como Asesor Senior de Tesis y el proceso de trabajo en fases.

**IMPORTANTE - LECCIONES APRENDIDAS:**
- ANTES de redactar cualquier sección, SIEMPRE verifica los datos contra:
  1. Checkpoints reales del modelo (cargar y analizar)
  2. Documentación de sesiones en `Documentos/docs/sesiones/`
  3. `GROUND_TRUTH.json` para valores validados
  4. Código fuente actual (no confiar solo en `final_config.json`)
- NO asumas valores de hiperparámetros o arquitectura sin verificar
- Verificar posiciones anatómicas de landmarks contra `constants.py`

---

## CONTEXTO DE LA SESIÓN ANTERIOR

### Fecha de sesión anterior: 16 Diciembre 2025 (Sesión 06)

### Estado del Proyecto de Tesis

| Fase | Estado | Descripción |
|------|--------|-------------|
| Fase 1: Análisis del Proyecto | ✅ COMPLETADA | Análisis exhaustivo del código, resultados, documentación |
| Fase 2: Estructura de Tesis | ✅ COMPLETADA | Estructura de 6 capítulos aprobada |
| Fase 3: Redacción | 🔄 EN PROGRESO | **Capítulo 4 COMPLETADO**, iniciando Capítulo 5 |
| Fase 4: Revisión Final | ⏳ PENDIENTE | — |

---

## ARCHIVOS CLAVE

Revisar estos archivos para contexto completo:

| Archivo | Contenido |
|---------|-----------|
| `Documentos/Tesis/DECISIONES_FASE_1.md` | Decisiones tomadas, claims validados/invalidados, limitaciones |
| `Documentos/Tesis/ESTRUCTURA_TESIS.md` | Estructura de 6 capítulos, historial de sesiones |
| `Documentos/Tesis/5-Objetivos-Ajustados.tex` | 6 objetivos específicos ajustados (aprobados) |
| `Documentos/Tesis/FIGURAS_PENDIENTES.md` | Lista de figuras por crear |
| `Documentos/Tesis/EXPERIMENTOS_PENDIENTES.md` | Experimentos pendientes |
| `GROUND_TRUTH.json` | Valores validados experimentalmente |

### Capítulos completados:

| Archivo | Contenido |
|---------|-----------|
| `capitulo4/4_1_descripcion_general.tex` | Sección 4.1 - Pipeline general |
| `capitulo4/4_2_dataset_preprocesamiento.tex` | Sección 4.2 - Dataset y CLAHE |
| `capitulo4/4_3_modelo_landmarks.tex` | Sección 4.3 - Modelo ResNet-18 + CoordAttn |
| `capitulo4/4_4_normalizacion_geometrica.tex` | Sección 4.4 - GPA, Delaunay, Warping |
| `capitulo4/4_5_clasificacion.tex` | Sección 4.5 - Clasificador CNN |
| `capitulo4/4_6_protocolo_evaluacion.tex` | Sección 4.6 - Protocolo de evaluación |

---

## HITO ALCANZADO - CAPÍTULO 4 COMPLETADO

**Capítulo 4: Metodología** ✅ COMPLETADO (6/6 secciones, ~26 páginas)

| Sección | Páginas | Estado |
|---------|---------|--------|
| 4.1 Descripción general del sistema | 2 | ✅ COMPLETADA |
| 4.2 Dataset y preprocesamiento | 4 | ✅ COMPLETADA |
| 4.3 Modelo de predicción de landmarks | 6 | ✅ COMPLETADA |
| 4.4 Normalización geométrica | 6 | ✅ COMPLETADA |
| 4.5 Clasificación de enfermedades | 4 | ✅ COMPLETADA |
| 4.6 Protocolo de evaluación | 4 | ✅ COMPLETADA |

---

## RESUMEN DE SESIÓN 06

### Trabajo Completado:

1. **Sección 4.6 redactada (~4 páginas):**
   - Métricas de landmarks: MED, error por landmark, percentiles
   - Métricas de clasificación: Accuracy, F1-Macro vs F1-Weighted (justificación)
   - Protocolo de robustez: JPEG Q50/Q30, blur σ=1/2
   - Protocolo de cross-evaluation: matriz 2×2
   - Protocolo de validación externa: FedCOVIDx
   - TTA para landmarks
   - 17 ecuaciones, 6 tablas, 2 figuras (placeholders)

2. **Revisión exhaustiva y corrección de errores:**

| Error | Antes | Después |
|-------|-------|---------|
| Posición anatómica L9, L10 | "ápex pulmonar" | "eje central" |
| Posición anatómica L12, L13 | "ángulos costofrénicos" | "bordes superiores" |
| Kernel blur gaussiano | "5×5 fijo" | "automático según σ" |
| Ecuación accuracy | Mezcla binaria/multiclase | Multiclase pura |
| Referencia en 4.5 | `\ref{sec:dataset}` inexistente | `\ref{sec:dataset_preprocesamiento}` |

3. **Verificaciones confirmadas:**
   - SYMMETRIC_PAIRS: (L3,L4), (L5,L6), (L7,L8), (L12,L13), (L14,L15) ✓
   - FedCOVIDx: 8,482 muestras ✓
   - CLAHE: clip_limit=2.0, tile_size=4 ✓

---

## CLAIMS CIENTÍFICOS VALIDADOS (Usar en tesis)

| Claim | Valor | Fuente |
|-------|-------|--------|
| Error de landmarks (ensemble 4 + TTA) | **3.71 px** | GROUND_TRUTH.json |
| Error individual mejor modelo + TTA | 4.04 px | GROUND_TRUTH.json |
| Mediana error | 3.17 px | GROUND_TRUTH.json |
| Accuracy clasificación (warped_96) | **99.10%** | GROUND_TRUTH.json |
| Accuracy clasificación binaria | 99.05% | GROUND_TRUTH.json |
| Mejora robustez JPEG Q50 | **30×** | GROUND_TRUTH.json |
| Mejora generalización cross-dataset | **2.4×** | GROUND_TRUTH.json |
| Mecanismo causal | 75% reducción info + 25% normalización | Sesión 39 |
| Fill rate óptimo | 96% | Sesión 52-53 |
| Margin scale óptimo | 1.05 | constants.py |

### Valores de error por landmark (GROUND_TRUTH.json):
```
L1: 3.20, L2: 4.34, L3: 3.20, L4: 3.49, L5: 2.97, L6: 3.01
L7: 3.39, L8: 3.67, L9: 2.84, L10: 2.57, L11: 3.19
L12: 5.50, L13: 5.21, L14: 4.63, L15: 4.48
```
- **Mejores:** L10 (2.57), L9 (2.84), L5 (2.97)
- **Peores:** L12 (5.50), L13 (5.21), L14 (4.63)

### Valores de error por categoría (GROUND_TRUTH.json):
```
COVID: 3.77 px, Normal: 3.42 px, Viral_Pneumonia: 4.40 px
```

## CLAIMS INVALIDADOS (NO usar)

| Claim Incorrecto | Corrección |
|------------------|------------|
| "11× mejor generalización" | Solo 2.4× |
| "Fuerza atención pulmonar" | PFS ≈ 0.49 = aleatorio |
| "Resuelve domain shift externo" | ~55% en FedCOVIDx |
| "Se usa CombinedLandmarkLoss" | Solo WingLoss |
| "Hay ensemble de clasificadores" | Solo ensemble de landmarks |
| "TTA aplica a clasificación" | Solo aplica a landmarks |

---

## TAREA PARA LA SIGUIENTE SESIÓN

### Iniciar Capítulo 5: Resultados y Discusión

**SECCIÓN 5.1: Resultados de Predicción de Landmarks (~4 páginas)**

**ANTES DE REDACTAR, VERIFICAR:**
1. Revisar `GROUND_TRUTH.json` sección `landmarks` y `per_landmark_errors`
2. Leer documentación de sesiones:
   - Sesión 10: Tests de integración landmarks
   - Sesión 12: Optimización de ensemble
   - Sesión 13: Ensemble de 4 modelos (valores definitivos)
3. Revisar `configs/final_config.json` sección de error por landmark

**Contenido a incluir:**

1. **Rendimiento del modelo individual:**
   - Error medio: ~4.04 px (mejor modelo individual + TTA)
   - Análisis de convergencia del entrenamiento

2. **Rendimiento del ensemble:**
   - Error medio: 3.71 px (ensemble 4 modelos + TTA)
   - Mejora vs modelo individual
   - Justificación del número de modelos (4)

3. **Análisis por landmark:**
   - Tabla con error por cada L1-L15
   - Identificar patrones: centrales vs bordes
   - Visualización sugerida: gráfica de barras

4. **Análisis por categoría diagnóstica:**
   - COVID: 3.77 px, Normal: 3.42 px, Viral_Pneumonia: 4.40 px
   - Discusión de por qué Viral_Pneumonia tiene más error

5. **Análisis de distribución de errores:**
   - Percentiles: P50=3.17, P75, P90, P95
   - Casos extremos y outliers

6. **Impacto del TTA:**
   - Comparar con/sin TTA
   - Justificación del uso de flip horizontal

**Archivos de referencia:**
- `GROUND_TRUTH.json` - Valores definitivos
- `Documentos/docs/sesiones/SESION_13_ENSEMBLE_4_MODELOS.md` - Resultados ensemble
- `configs/final_config.json` - Configuración y valores por landmark

---

## PROGRESO DE REDACCIÓN - CAPÍTULO 5

| Sección | Páginas | Estado |
|---------|---------|--------|
| 5.1 Resultados de predicción de landmarks | 4 | ⏳ **SIGUIENTE** |
| 5.2 Resultados de clasificación | 4 | ⏳ Pendiente |
| 5.3 Análisis de robustez | 4 | ⏳ Pendiente |
| 5.4 Evaluación de generalización | 4 | ⏳ Pendiente |
| 5.5 Discusión general | 4 | ⏳ Pendiente |

**Progreso Capítulo 5:** 0% (0/20 páginas)
**Progreso Total Estimado:** ~35% (Cap. 4 completo de ~6 capítulos)

---

## FIGURAS PENDIENTES PARA SECCIÓN 5.1

Figuras anticipadas (documentar en `FIGURAS_PENDIENTES.md` después de redactar):
- F5.1: Gráfica de error por landmark (barras horizontales)
- F5.2: Distribución de errores (histograma o boxplot)
- F5.3: Comparación modelo individual vs ensemble

---

## DECISIONES YA TOMADAS

| Decisión | Sesión | Resultado |
|----------|--------|-----------|
| Trade-off fill rate | Sesión 04 | Reservado para Cap. 5 |
| Ensemble de clasificadores | Sesión 05 | NO existe, omitir |
| TTA para clasificación | Sesión 05 | NO existe, omitir |
| Arquitectura clasificador | Sesión 05 | ResNet-18 seleccionado |
| Posiciones anatómicas L9-L15 | Sesión 06 | Verificadas contra constants.py |
| Kernel blur gaussiano | Sesión 06 | Automático según σ |

---

## RECORDATORIOS

- **Verificar antes de redactar:** Cargar checkpoints, leer código fuente, no confiar solo en configs
- **Figuras:** Están pendientes, documentadas en `FIGURAS_PENDIENTES.md`
- **Referencias:** Mínimo 50, estilo IEEE, 60% recientes
- **Extensión total:** 80-120 páginas
- **Formato:** LaTeX
- **IMPORTANTE:** El Capítulo 5 es de ALTO RIESGO - solo usar claims validados

---

## COMANDO INICIAL SUGERIDO

```
Por favor, revisa los archivos de contexto mencionados arriba.
ANTES de redactar la sección 5.1, verifica:
1. Revisa GROUND_TRUTH.json secciones landmarks y per_landmark_errors
2. Lee la documentación de sesión 13 (ensemble de 4 modelos)
3. Verifica los valores en configs/final_config.json

Luego hazme las preguntas necesarias para clarificar detalles antes de redactar.
```

---

## ORDEN SUGERIDO PARA CAPÍTULO 5

1. **5.1 Resultados de predicción de landmarks** ← SIGUIENTE
2. 5.2 Resultados de clasificación
3. 5.3 Análisis de robustez (incluir experimento de control)
4. 5.4 Evaluación de generalización (cross-evaluation + validación externa)
5. 5.5 Discusión general (limitaciones, comparación con estado del arte)

---

## ADVERTENCIAS PARA CAPÍTULO 5

⚠️ **ALTO RIESGO:** El Capítulo 5 presenta resultados experimentales. Asegurarse de:

1. **NO inflar resultados:**
   - Usar solo valores de GROUND_TRUTH.json
   - No redondear hacia arriba
   - Reportar limitaciones honestamente

2. **Validación externa:**
   - ~55% en FedCOVIDx es un resultado negativo
   - Discutir como limitación del domain shift, NO como fallo del warping

3. **Mecanismo de robustez:**
   - 75% reducción de información + 25% normalización geométrica
   - NO afirmar que "fuerza atención pulmonar" (PFS ≈ aleatorio)

4. **Comparación justa:**
   - Comparar warped_96 vs original_100, no vs original_cropped_47
   - El beneficio principal es robustez, no accuracy in-domain

---

*Prompt generado: 16 Diciembre 2025 - Sesión 06*
