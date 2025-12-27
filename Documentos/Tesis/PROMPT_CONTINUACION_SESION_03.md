# PROMPT DE CONTINUACIÓN - SESIÓN 04 DE REDACCIÓN DE TESIS

## INSTRUCCIONES PARA CLAUDE

Lee el archivo `prompt_tesis.md` en la raíz del proyecto para entender tu rol como Asesor Senior de Tesis y el proceso de trabajo en fases.

**IMPORTANTE - LECCIÓN DE SESIÓN 03:**
- ANTES de redactar cualquier sección, SIEMPRE verifica los datos contra:
  1. Checkpoints reales del modelo (cargar y analizar)
  2. Documentación de sesiones en `Documentos/docs/sesiones/`
  3. `GROUND_TRUTH.json` para valores validados
  4. Código fuente actual (no confiar solo en `final_config.json`)
- NO asumas valores de hiperparámetros o arquitectura sin verificar
- El archivo `final_config.json` puede estar desactualizado

---

## CONTEXTO DE LA SESIÓN ANTERIOR

### Fecha de sesión anterior: 16 Diciembre 2025 (Sesión 03)

### Estado del Proyecto de Tesis

| Fase | Estado | Descripción |
|------|--------|-------------|
| Fase 1: Análisis del Proyecto | ✅ COMPLETADA | Análisis exhaustivo del código, resultados, documentación |
| Fase 2: Estructura de Tesis | ✅ COMPLETADA | Estructura de 6 capítulos aprobada |
| Fase 3: Redacción | 🔄 EN PROGRESO | Secciones 4.1, 4.2, 4.3 completadas |
| Fase 4: Revisión Final | ⏳ PENDIENTE | — |

---

## ARCHIVOS CLAVE

Revisar estos archivos para contexto completo:

| Archivo | Contenido |
|---------|-----------|
| `Documentos/Tesis/DECISIONES_FASE_1.md` | Decisiones tomadas, claims validados/invalidados, limitaciones |
| `Documentos/Tesis/ESTRUCTURA_TESIS.md` | Estructura de 6 capítulos, historial de sesiones, errores corregidos |
| `Documentos/Tesis/5-Objetivos-Ajustados.tex` | 6 objetivos específicos ajustados (aprobados) |
| `Documentos/Tesis/FIGURAS_PENDIENTES.md` | Lista de figuras por crear |
| `Documentos/Tesis/EXPERIMENTOS_PENDIENTES.md` | Experimentos pendientes |
| `GROUND_TRUTH.json` | Valores validados experimentalmente |

### Secciones ya redactadas:
| Archivo | Contenido |
|---------|-----------|
| `capitulo4/4_1_descripcion_general.tex` | Sección 4.1 - Pipeline general |
| `capitulo4/4_2_dataset_preprocesamiento.tex` | Sección 4.2 - Dataset y CLAHE |
| `capitulo4/4_3_modelo_landmarks.tex` | Sección 4.3 - Modelo ResNet-18 + CoordAttn |

---

## DECISIONES APROBADAS

### Título (FIJO - No modificable)
> "Normalización y alineación automática de la forma de la región pulmonar integrada con selección de características discriminantes para detección de neumonía y COVID-19"

### Objetivos Ajustados (6)
1. Modelo de predicción de landmarks (ResNet-18 + Coordinate Attention)
2. Normalización geométrica (warping afín por partes + GPA)
3. Evaluación de 7 arquitecturas CNN
4. Validación con métricas de clasificación y robustez
5. Cuantificación de contribución (~75% info + ~25% geo)
6. Evaluación de generalización (cross-eval + validación externa)

### Orden de Redacción
1. Capítulo 4: Metodología ← EN PROGRESO (4.1-4.3 completadas)
2. Capítulo 5: Resultados
3. Capítulo 2: Marco Teórico
4. Capítulo 3: Estado del Arte
5. Capítulo 1: Introducción
6. Capítulo 6: Conclusiones

---

## PROGRESO DE REDACCIÓN - CAPÍTULO 4

| Sección | Páginas | Estado |
|---------|---------|--------|
| 4.1 Descripción general del sistema | 2 | ✅ COMPLETADA |
| 4.2 Dataset y preprocesamiento | 4 | ✅ COMPLETADA |
| 4.3 Modelo de predicción de landmarks | 6 | ✅ COMPLETADA |
| 4.4 Normalización geométrica | 6 | ⏳ **SIGUIENTE** |
| 4.5 Clasificación de enfermedades | 4 | ⏳ PENDIENTE |
| 4.6 Protocolo de evaluación | 4 | ⏳ PENDIENTE |

---

## ERRORES CORREGIDOS EN SESIÓN 03

La sesión 03 identificó y corrigió múltiples errores causados por no verificar datos antes de redactar:

| Archivo | Error | Valor incorrecto | Valor correcto |
|---------|-------|------------------|----------------|
| 4.2 | Imágenes anotadas | 956 | **957** |
| 4.2 | Split validación/prueba | 12.5%/12.5% | **15%/10%** |
| 4.3 | Arquitectura cabeza | 2 capas | **3 capas con GroupNorm** |
| 4.3 | Dropout | 0.5/0.25 | **0.3/0.15** |
| 4.3 | Función de pérdida | CombinedLandmarkLoss | **Solo WingLoss** |
| 4.3 | Batch size fase 2 | 16 | **8** |
| final_config.json | Estructura cabeza | Desactualizado | **Corregido** |

---

## CLAIMS CIENTÍFICOS VALIDADOS (Usar en tesis)

| Claim | Valor |
|-------|-------|
| Error de landmarks (ensemble 4 modelos + TTA) | 3.71 px |
| Accuracy clasificación (warped_96) | 99.10% |
| Accuracy clasificación binaria | 99.05% |
| Mejora robustez JPEG Q50 | 30× |
| Mejora generalización cross-dataset | 2.4× |
| Mecanismo causal | 75% reducción info + 25% normalización geo |
| Fill rate óptimo | 96% |
| Margin scale óptimo | 1.05 |

## CLAIMS INVALIDADOS (NO usar)

| Claim Incorrecto | Corrección |
|------------------|------------|
| "11× mejor generalización" | Solo 2.4× |
| "Fuerza atención pulmonar" | PFS ≈ 0.49 = aleatorio |
| "Resuelve domain shift externo" | ~55% en FedCOVIDx |
| "Se usa CombinedLandmarkLoss" | Solo WingLoss |

---

## TAREA PARA LA SIGUIENTE SESIÓN

### Continuar con Sección 4.4: Normalización Geométrica

**ANTES DE REDACTAR, VERIFICAR:**
1. Leer `src_v2/processing/warp.py` para detalles de implementación
2. Leer `src_v2/processing/gpa.py` para GPA
3. Verificar valores de `OPTIMAL_MARGIN_SCALE` en `constants.py`
4. Revisar documentación de sesiones relacionadas con warping

**Contenido a incluir (~6 páginas):**
1. Análisis Procrustes Generalizado (GPA) para forma canónica
2. Triangulación Delaunay de landmarks
3. Transformación afín por partes (piecewise affine warping)
4. Estrategia de full coverage
5. Concepto de fill rate y su impacto
6. Valor óptimo de margin_scale (1.05)

**Archivos de referencia:**
- `src_v2/processing/warp.py` - Implementación de warping
- `src_v2/processing/gpa.py` - GPA para forma canónica
- `src_v2/constants.py` - OPTIMAL_MARGIN_SCALE = 1.05
- `GROUND_TRUTH.json` - fill_rate_tradeoff

---

## FIGURAS PENDIENTES

Ver archivo `Documentos/Tesis/FIGURAS_PENDIENTES.md` para lista completa.

Figuras prioritarias para sección 4.4:
- F4.6: Proceso de GPA (formas antes/después de alineación)
- F4.7: Triangulación Delaunay sobre landmarks
- F4.8: Comparación imagen original vs warped
- F4.9: Efecto de diferentes margin_scale

---

## RECORDATORIOS

- **Verificar antes de redactar:** Cargar checkpoints, leer código fuente, no confiar solo en configs
- **Figuras:** Están pendientes, documentadas en `FIGURAS_PENDIENTES.md`
- **Referencias:** Mínimo 50, estilo IEEE, 60% recientes
- **Extensión total:** 80-120 páginas
- **Formato:** LaTeX

---

## COMANDO INICIAL SUGERIDO

```
Por favor, revisa los archivos de contexto mencionados arriba.
ANTES de redactar la sección 4.4, verifica:
1. Lee src_v2/processing/warp.py y gpa.py
2. Verifica los valores en GROUND_TRUTH.json relacionados con warping
3. Lee la documentación de sesiones relevantes

Luego hazme las preguntas necesarias para clarificar detalles antes de redactar.
```

---

*Prompt generado: 16 Diciembre 2025 - Sesión 03*
