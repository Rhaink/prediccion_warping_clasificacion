# PROMPT DE CONTINUACIÓN - SESIÓN 02 DE REDACCIÓN DE TESIS

## INSTRUCCIONES PARA CLAUDE

Lee el archivo `prompt_tesis.md` en la raíz del proyecto para entender tu rol como Asesor Senior de Tesis y el proceso de trabajo en fases.

---

## CONTEXTO DE LA SESIÓN ANTERIOR

### Fecha de sesión anterior: 16 Diciembre 2025

### Estado del Proyecto de Tesis

| Fase | Estado | Descripción |
|------|--------|-------------|
| Fase 1: Análisis del Proyecto | ✅ COMPLETADA | Análisis exhaustivo del código, resultados, documentación |
| Fase 2: Estructura de Tesis | ✅ COMPLETADA | Estructura de 6 capítulos aprobada |
| Fase 3: Redacción | 🔄 EN PROGRESO | Sección 4.1 completada |
| Fase 4: Revisión Final | ⏳ PENDIENTE | — |

---

## ARCHIVOS CLAVE CREADOS

Revisar estos archivos para contexto completo:

| Archivo | Contenido |
|---------|-----------|
| `Documentos/Tesis/DECISIONES_FASE_1.md` | Decisiones tomadas, claims validados/invalidados, limitaciones |
| `Documentos/Tesis/ESTRUCTURA_TESIS.md` | Estructura de 6 capítulos aprobada, orden de redacción |
| `Documentos/Tesis/5-Objetivos-Ajustados.tex` | 6 objetivos específicos ajustados (aprobados) |
| `Documentos/Tesis/FIGURAS_PENDIENTES.md` | Lista de figuras por crear |
| `Documentos/Tesis/EXPERIMENTOS_PENDIENTES.md` | Experimentos a ejecutar antes de terminar |
| `Documentos/Tesis/capitulo4/4_1_descripcion_general.tex` | Sección 4.1 redactada |

---

## DECISIONES APROBADAS

### Título (FIJO - No modificable)
> "Normalización y alineación automática de la forma de la región pulmonar integrada con selección de características discriminantes para detección de neumonía y COVID-19"

### Interpretación de "Selección de características discriminantes"
La normalización geométrica mediante landmarks actúa como un mecanismo de selección de características a nivel de imagen, eliminando información no discriminante (background, artefactos) y reteniendo solo la región pulmonar relevante.

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
- Capítulo 4: Metodología (22-28 págs) 🔴 EN PROGRESO
- Capítulo 5: Resultados (18-22 págs)
- Capítulo 6: Conclusiones (6-8 págs)

### Orden de Redacción
1. Capítulo 4: Metodología ← ACTUAL
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
| 4.2 Dataset y preprocesamiento | 4 | ⏳ PENDIENTE |
| 4.3 Modelo de predicción de landmarks | 6 | ⏳ PENDIENTE |
| 4.4 Normalización geométrica | 6 | ⏳ PENDIENTE |
| 4.5 Clasificación de enfermedades | 4 | ⏳ PENDIENTE |
| 4.6 Protocolo de evaluación | 4 | ⏳ PENDIENTE |

---

## EXPERIMENTO PENDIENTE IMPORTANTE

### Clasificación Binaria: Neumonía vs Normal

**Pregunta surgida:** ¿Qué pasaría si agrupamos COVID + Viral_Pneumonia como "Neumonía" vs "Normal"?

**Estado:** NO existe este experimento. Debe ejecutarse antes de terminar la tesis.

**Configuración:**
- Neumonía: 324 (COVID) + 200 (Viral) = 524 imágenes
- Normal: 475 imágenes

**Decisión pendiente:** El usuario decidirá en esta sesión si ejecutar el experimento ahora o después.

Ver detalles en: `Documentos/Tesis/EXPERIMENTOS_PENDIENTES.md`

---

## CLAIMS CIENTÍFICOS VALIDADOS (Usar en tesis)

| Claim | Valor |
|-------|-------|
| Error de landmarks (ensemble) | 3.71 px |
| Accuracy clasificación (warped_96) | 99.10% |
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

## TAREAS PARA ESTA SESIÓN

### Opción A: Ejecutar experimento primero
1. Ejecutar experimento "Neumonía vs Normal"
2. Documentar resultados
3. Continuar con redacción de Sección 4.2

### Opción B: Continuar redacción
1. Continuar con Sección 4.2 (Dataset y preprocesamiento)
2. Dejar experimento para después

### El usuario debe decidir qué opción prefiere.

---

## RECORDATORIOS

- **Figuras:** Están pendientes, documentadas en `FIGURAS_PENDIENTES.md`
- **Referencias:** Mínimo 50, estilo IEEE, 60% recientes
- **Extensión total:** 80-120 páginas
- **Formato:** LaTeX

---

## COMANDO INICIAL SUGERIDO

```
Por favor, revisa los archivos de contexto mencionados arriba y confirma que entiendes el estado del proyecto. Luego pregúntame cómo deseo proceder:
1. ¿Ejecutar el experimento Neumonía vs Normal?
2. ¿Continuar con la redacción de la Sección 4.2?
```

---

*Prompt generado: 16 Diciembre 2025*
