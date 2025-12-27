# PROMPT DE CONTINUACIÓN - SESIÓN 06 DE REDACCIÓN DE TESIS

## INSTRUCCIONES PARA CLAUDE

Lee el archivo `prompt_tesis.md` en la raíz del proyecto para entender tu rol como Asesor Senior de Tesis y el proceso de trabajo en fases.

**IMPORTANTE - LECCIONES APRENDIDAS:**
- ANTES de redactar cualquier sección, SIEMPRE verifica los datos contra:
  1. Checkpoints reales del modelo (cargar y analizar)
  2. Documentación de sesiones en `Documentos/docs/sesiones/`
  3. `GROUND_TRUTH.json` para valores validados
  4. Código fuente actual (no confiar solo en `final_config.json`)
- NO asumas valores de hiperparámetros o arquitectura sin verificar

---

## CONTEXTO DE LA SESIÓN ANTERIOR

### Fecha de sesión anterior: 16 Diciembre 2025 (Sesión 05)

### Estado del Proyecto de Tesis

| Fase | Estado | Descripción |
|------|--------|-------------|
| Fase 1: Análisis del Proyecto | ✅ COMPLETADA | Análisis exhaustivo del código, resultados, documentación |
| Fase 2: Estructura de Tesis | ✅ COMPLETADA | Estructura de 6 capítulos aprobada |
| Fase 3: Redacción | 🔄 EN PROGRESO | Secciones 4.1-4.5 completadas |
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

### Secciones ya redactadas:
| Archivo | Contenido |
|---------|-----------|
| `capitulo4/4_1_descripcion_general.tex` | Sección 4.1 - Pipeline general |
| `capitulo4/4_2_dataset_preprocesamiento.tex` | Sección 4.2 - Dataset y CLAHE |
| `capitulo4/4_3_modelo_landmarks.tex` | Sección 4.3 - Modelo ResNet-18 + CoordAttn |
| `capitulo4/4_4_normalizacion_geometrica.tex` | Sección 4.4 - GPA, Delaunay, Warping |
| `capitulo4/4_5_clasificacion.tex` | Sección 4.5 - Clasificador CNN |

---

## PROGRESO DE REDACCIÓN - CAPÍTULO 4

| Sección | Páginas | Estado |
|---------|---------|--------|
| 4.1 Descripción general del sistema | 2 | ✅ COMPLETADA |
| 4.2 Dataset y preprocesamiento | 4 | ✅ COMPLETADA |
| 4.3 Modelo de predicción de landmarks | 6 | ✅ COMPLETADA |
| 4.4 Normalización geométrica | 6 | ✅ COMPLETADA |
| 4.5 Clasificación de enfermedades | 4 | ✅ COMPLETADA |
| 4.6 Protocolo de evaluación | 4 | ⏳ **SIGUIENTE** |

**Progreso Capítulo 4:** 83% (~22/26 páginas)

---

## RESUMEN DE SESIÓN 05

### Trabajo Completado:
1. **Investigación previa a redacción:**
   - Verificado que NO existe ensemble de clasificadores (solo de landmarks)
   - Verificado que TTA solo aplica a landmarks, no al clasificador
   - Encontrados resultados de comparación de 7 arquitecturas
   - Identificada justificación de selección de ResNet-18

2. **Sección 4.5 redactada (~4 páginas):**
   - 7 arquitecturas CNN evaluadas
   - Enfoque en ResNet-18 (99.10%) y EfficientNet-B0 (97.76%)
   - Transfer learning desde ImageNet
   - Manejo de desbalance con pesos de clase
   - Data augmentation documentado
   - 7 tablas, 2 ecuaciones, 9 referencias

3. **Archivos actualizados:**
   - `ESTRUCTURA_TESIS.md` - Progreso y historial sesión 05
   - `FIGURAS_PENDIENTES.md` - Agregadas figuras de sección 4.5

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
| "Hay ensemble de clasificadores" | Solo ensemble de landmarks |
| "TTA aplica a clasificación" | Solo aplica a landmarks |

---

## TAREA PARA LA SIGUIENTE SESIÓN

### Continuar con Sección 4.6: Protocolo de Evaluación Experimental

**ANTES DE REDACTAR, VERIFICAR:**
1. Leer `src_v2/evaluation/metrics.py` para métricas implementadas
2. Revisar `GROUND_TRUTH.json` sección completa
3. Leer documentación de sesiones sobre evaluación:
   - Sesión 29: Test de robustez
   - Sesión 30: Cross-evaluation
   - Sesión 39: Experimento de control
   - Sesión 55: Validación externa

**Contenido a incluir (~4 páginas):**
1. **Métricas de evaluación para landmarks:**
   - Error euclidiano medio (en píxeles)
   - Error por categoría de landmark
   - Normalización del error

2. **Métricas de clasificación:**
   - Accuracy, Precision, Recall, F1-Score
   - F1-Macro vs F1-Weighted (justificar uso de Macro)
   - Matriz de confusión

3. **Protocolo de evaluación de robustez:**
   - Perturbaciones evaluadas: JPEG (Q50, Q30), blur gaussiano
   - Cálculo de degradación

4. **Protocolo de cross-evaluation:**
   - Original→Original, Original→Warped
   - Warped→Warped, Warped→Original
   - Cálculo de ratio de generalización

5. **Protocolo de validación externa:**
   - Dataset FedCOVIDx
   - Mapeo de clases (3→2)
   - Limitaciones de domain shift

**Archivos de referencia:**
- `src_v2/evaluation/metrics.py` - Implementación de métricas
- `GROUND_TRUTH.json` - Valores de referencia
- `Documentos/docs/sesiones/SESION_29_*.md` - Robustez
- `Documentos/docs/sesiones/SESION_30_*.md` - Cross-evaluation
- `Documentos/docs/sesiones/SESION_39_*.md` - Experimento control
- `Documentos/docs/sesiones/SESION_55_*.md` - Validación externa

---

## FIGURAS PENDIENTES PARA SECCIÓN 4.6

Figuras anticipadas (documentar en `FIGURAS_PENDIENTES.md` después de redactar):
- Diagrama de protocolo de evaluación
- Ejemplos de perturbaciones (JPEG, blur)
- Esquema de cross-evaluation

---

## DECISIONES YA TOMADAS

| Decisión | Sesión | Resultado |
|----------|--------|-----------|
| Trade-off fill rate | Sesión 04 | Reservado para Cap. 5 |
| Ensemble de clasificadores | Sesión 05 | NO existe, omitir |
| TTA para clasificación | Sesión 05 | NO existe, omitir |
| Arquitectura clasificador | Sesión 05 | ResNet-18 seleccionado |

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
ANTES de redactar la sección 4.6, verifica:
1. Lee src_v2/evaluation/metrics.py
2. Verifica los valores en GROUND_TRUTH.json
3. Lee la documentación de sesiones 29, 30, 39 y 55

Luego hazme las preguntas necesarias para clarificar detalles antes de redactar.
```

---

## AL COMPLETAR CAPÍTULO 4

Después de la sección 4.6, el Capítulo 4 (Metodología) estará completo.
**Siguiente paso:** Iniciar Capítulo 5 (Resultados y Discusión)

Orden sugerido para Cap. 5:
1. 5.1 Resultados de predicción de landmarks
2. 5.2 Resultados de clasificación
3. 5.3 Análisis de robustez
4. 5.4 Evaluación de generalización
5. 5.5 Discusión general

---

*Prompt generado: 16 Diciembre 2025 - Sesión 05*
