# PROMPT DE CONTINUACIÓN - SESIÓN 05 DE REDACCIÓN DE TESIS

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

### Fecha de sesión anterior: 16 Diciembre 2025 (Sesión 04)

### Estado del Proyecto de Tesis

| Fase | Estado | Descripción |
|------|--------|-------------|
| Fase 1: Análisis del Proyecto | ✅ COMPLETADA | Análisis exhaustivo del código, resultados, documentación |
| Fase 2: Estructura de Tesis | ✅ COMPLETADA | Estructura de 6 capítulos aprobada |
| Fase 3: Redacción | 🔄 EN PROGRESO | Secciones 4.1-4.4 completadas |
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

---

## PROGRESO DE REDACCIÓN - CAPÍTULO 4

| Sección | Páginas | Estado |
|---------|---------|--------|
| 4.1 Descripción general del sistema | 2 | ✅ COMPLETADA |
| 4.2 Dataset y preprocesamiento | 4 | ✅ COMPLETADA |
| 4.3 Modelo de predicción de landmarks | 6 | ✅ COMPLETADA |
| 4.4 Normalización geométrica | 6 | ✅ COMPLETADA |
| 4.5 Clasificación de enfermedades | 4 | ⏳ **SIGUIENTE** |
| 4.6 Protocolo de evaluación | 4 | ⏳ PENDIENTE |

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

### Continuar con Sección 4.5: Clasificación de Enfermedades Pulmonares

**ANTES DE REDACTAR, VERIFICAR:**
1. Leer `src_v2/models/classifier.py` para arquitecturas soportadas
2. Revisar `GROUND_TRUTH.json` sección `classification`
3. Verificar documentación de sesiones sobre clasificación (Sesiones 14-16, 22)
4. Revisar resultados de comparación de arquitecturas

**Contenido a incluir (~4 páginas):**
1. Arquitecturas CNN evaluadas (ResNet-18, DenseNet-121, EfficientNet-B0, VGG-16, etc.)
2. Estrategia de transfer learning para clasificación
3. Entrenamiento del clasificador (hiperparámetros, early stopping)
4. Ensemble de clasificadores (si aplica)
5. Test-Time Augmentation (TTA) para clasificación

**Archivos de referencia:**
- `src_v2/models/classifier.py` - Implementación del clasificador
- `src_v2/training/trainer.py` - Entrenamiento (si hay trainer para clasificador)
- `GROUND_TRUTH.json` - Resultados de clasificación
- `Documentos/docs/sesiones/SESION_22_COMPARE_ARCHITECTURES.md` - Comparación de arquitecturas

---

## DECISIONES PENDIENTES PARA SECCIÓN 4.5

1. **Trade-off fill rate (96% vs 99%):** Reservado para Capítulo 5 (Resultados), NO incluir en 4.5

2. **7 arquitecturas evaluadas:** Verificar lista exacta en código:
   - ResNet-18, ResNet-50
   - DenseNet-121
   - EfficientNet-B0
   - VGG-16
   - MobileNet-V2
   - AlexNet (?)

---

## FIGURAS PENDIENTES PARA SECCIÓN 4.5

Ver archivo `Documentos/Tesis/FIGURAS_PENDIENTES.md` para lista completa.

Figuras anticipadas para sección 4.5:
- F4.11: Arquitectura del clasificador
- F4.12: Comparación de arquitecturas CNN (tabla o gráfico)

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
ANTES de redactar la sección 4.5, verifica:
1. Lee src_v2/models/classifier.py
2. Verifica los valores en GROUND_TRUTH.json relacionados con clasificación
3. Lee la documentación de sesiones 14-16 y 22

Luego hazme las preguntas necesarias para clarificar detalles antes de redactar.
```

---

*Prompt generado: 16 Diciembre 2025 - Sesión 04*
