# PROMPT DE CONTINUACIÓN - SESIÓN 03 DE REDACCIÓN DE TESIS

## INSTRUCCIONES PARA CLAUDE

Lee el archivo `Documentos/Tesis/prompts/prompt_tesis.md` para entender tu rol como Asesor Senior de Tesis y el proceso de trabajo en fases.

---

## CONTEXTO DE LA SESIÓN ANTERIOR

### Fecha de sesión anterior: 16 Diciembre 2025 (Sesión 02)

### Estado del Proyecto de Tesis

| Fase | Estado | Descripción |
|------|--------|-------------|
| Fase 1: Análisis del Proyecto | ✅ COMPLETADA | Análisis exhaustivo del código, resultados, documentación |
| Fase 2: Estructura de Tesis | ✅ COMPLETADA | Estructura de 6 capítulos aprobada |
| Fase 3: Redacción | 🔄 EN PROGRESO | Secciones 4.1 y 4.2 completadas |
| Fase 4: Revisión Final | ⏳ PENDIENTE | — |

---

## ARCHIVOS CLAVE

Revisar estos archivos para contexto completo:

| Archivo | Contenido |
|---------|-----------|
| `Documentos/Tesis/DECISIONES_FASE_1.md` | Decisiones tomadas, claims validados/invalidados, limitaciones |
| `Documentos/Tesis/ESTRUCTURA_TESIS.md` | Estructura de 6 capítulos aprobada, orden de redacción |
| `Documentos/Tesis/5-Objetivos-Ajustados.tex` | 6 objetivos específicos ajustados (aprobados) |
| `Documentos/Tesis/FIGURAS_PENDIENTES.md` | Lista de figuras por crear |
| `Documentos/Tesis/EXPERIMENTOS_PENDIENTES.md` | Experimentos (Exp. 1 COMPLETADO, Exp. 2 pendiente) |
| `Documentos/Tesis/capitulo4/4_1_descripcion_general.tex` | Sección 4.1 redactada |
| `Documentos/Tesis/capitulo4/4_2_dataset_preprocesamiento.tex` | Sección 4.2 redactada |

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

### Estructura Aprobada
- Capítulo 1: Introducción (10-12 págs)
- Capítulo 2: Marco Teórico (18-22 págs)
- Capítulo 3: Estado del Arte (12-15 págs)
- Capítulo 4: Metodología (22-28 págs) 🔄 EN PROGRESO
- Capítulo 5: Resultados (18-22 págs)
- Capítulo 6: Conclusiones (6-8 págs)

### Orden de Redacción
1. Capítulo 4: Metodología ← EN PROGRESO (4.1, 4.2 completadas)
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
| 4.3 Modelo de predicción de landmarks | 6 | ⏳ SIGUIENTE |
| 4.4 Normalización geométrica | 6 | ⏳ PENDIENTE |
| 4.5 Clasificación de enfermedades | 4 | ⏳ PENDIENTE |
| 4.6 Protocolo de evaluación | 4 | ⏳ PENDIENTE |

---

## EXPERIMENTO COMPLETADO EN SESIÓN 02

### Clasificación Binaria: Neumonía vs Normal ✅

**Resultados principales:**

| Métrica | 3 clases | 2 clases (binario) |
|---------|----------|-------------------|
| Accuracy | 99.10% | 99.05% |
| F1 Macro | 98.45% | 98.92% |

**Robustez (degradación de accuracy):**

| Perturbación | 3 clases | 2 clases |
|--------------|----------|----------|
| JPEG Q50 | 3.06% | 6.44% |
| Blur σ=1 | 2.43% | 4.12% |

**Conclusiones:**
1. El modelo binario logra rendimiento similar al de 3 clases
2. El modelo de 3 clases es más robusto ante perturbaciones
3. Estos resultados apoyan usar 3 clases como configuración principal

**Archivos generados:**
- `outputs/classifier_binary_neumonia_vs_normal/best_classifier.pt`
- `outputs/classifier_binary_neumonia_vs_normal/results.json`
- `outputs/classifier_binary_neumonia_vs_normal/robustness_results.json`

---

## CLAIMS CIENTÍFICOS VALIDADOS (Usar en tesis)

| Claim | Valor |
|-------|-------|
| Error de landmarks (ensemble) | 3.71 px |
| Accuracy clasificación (warped_96) | 99.10% |
| **Accuracy clasificación binaria** | **99.05%** |
| Mejora robustez JPEG Q50 | 30× |
| Mejora generalización cross-dataset | 2.4× |
| Mecanismo causal | 75% reducción info + 25% normalización geo |

## CLAIMS INVALIDADOS (NO usar)

| Claim Incorrecto | Corrección |
|------------------|------------|
| "11× mejor generalización" | Solo 2.4× |
| "Fuerza atención pulmonar" | PFS ≈ 0.49 = aleatorio |
| "Resuelve domain shift externo" | ~55% en FedCOVIDx |

---

## FIGURAS PENDIENTES

Ver archivo `Documentos/Tesis/FIGURAS_PENDIENTES.md` para lista completa.

Figuras prioritarias para secciones completadas:
- F4.1: Diagrama de bloques del pipeline (Sección 4.1)
- F4.2: Diagrama de 15 landmarks sobre radiografía (Sección 4.2)
- F4.3: Comparación CLAHE antes/después (Sección 4.2)

---

## TAREA PARA LA SIGUIENTE SESIÓN

### Continuar con Sección 4.3: Modelo de Predicción de Landmarks

**Contenido a incluir (~6 páginas):**
1. Arquitectura ResNet-18 como backbone
2. Módulo Coordinate Attention
3. Cabeza de regresión para 30 coordenadas
4. Función de pérdida Wing Loss
5. Estrategia de entrenamiento en dos fases
6. Detalles de hiperparámetros

**Archivos de referencia:**
- `src_v2/models/resnet_landmark.py` - Arquitectura del modelo
- `src_v2/models/losses.py` - Wing Loss y variantes
- `src_v2/constants.py` - Hiperparámetros

---

## RECORDATORIOS

- **Figuras:** Están pendientes, documentadas en `FIGURAS_PENDIENTES.md`
- **Referencias:** Mínimo 50, estilo IEEE, 60% recientes
- **Extensión total:** 80-120 páginas
- **Formato:** LaTeX

---

## COMANDO INICIAL SUGERIDO

```
Por favor, revisa los archivos de contexto mencionados arriba y confirma que entiendes el estado del proyecto. Luego procede con la redacción de la Sección 4.3 (Modelo de predicción de landmarks).
```

---

*Prompt generado: 16 Diciembre 2025 - Sesión 02*
