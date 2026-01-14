# DECISIONES Y ACUERDOS - FASE 1 DE LA TESIS

**Fecha:** 16 Diciembre 2025
**Estado:** Fase 1 completada, pendiente aprobación para Fase 2

---

## 1. INFORMACIÓN INSTITUCIONAL

- **Institución:** Benemérita Universidad Autónoma de Puebla (BUAP)
- **Facultad:** Ciencias de la Electrónica (FCE)
- **Programa:** Maestría en Ingeniería Electrónica, Opción Instrumentación Electrónica
- **Formato:** LaTeX
- **Fecha objetivo:** Noviembre 2025

---

## 2. TÍTULO DE LA TESIS (FIJO - NO MODIFICABLE)

> "Normalización y alineación automática de la forma de la región pulmonar integrada con selección de características discriminantes para detección de neumonía y COVID-19"

### Interpretación Acordada del Título

| Componente | Implementación |
|------------|----------------|
| Normalización y alineación automática | Warping afín por partes a forma canónica |
| Forma de la región pulmonar | 15 landmarks que definen contorno pulmonar |
| **Selección de características discriminantes** | **Interpretación: La normalización geométrica actúa como selección implícita de características, eliminando información no discriminante (background, artefactos) y reteniendo solo la región pulmonar relevante** |
| Detección de neumonía y COVID-19 | Clasificación 3 clases: COVID/Normal/Viral_Pneumonia |

---

## 3. OBJETIVOS

### 3.1 Objetivos Oficiales (DEL ASESOR - NO MODIFICABLES)

**Archivo:** `5-Objetivos.tex`

**Objetivo General:**
> Desarrollar e implementar algoritmos de visión por computadora para la detección, alineación y normalización de la forma de la región pulmonar en imágenes radiográficas de tórax, utilizando además un método eficaz para la selección de características discriminantes, con el fin de mejorar la precisión en la detección automática de neumonía y COVID-19.

**Objetivos Específicos:**
1. Diseñar, implementar y evaluar un método deformable de alineación y normalización que localice, segmente y ajuste automáticamente la región pulmonar en términos de forma, escala, posición y rotación.
2. Proponer un método de extracción y selección de características que maximicen la discriminación entre las clases.
3. Evaluar el rendimiento de diferentes clasificadores de aprendizaje supervisado para la técnica de alineación propuesta en la tesis: KNN, CNN, MLP.
4. Validar el clasificador desarrollado a través de medir la precisión, sensibilidad, especificidad y además de realizar pruebas de validación cruzada para caracterizar el algoritmo propuesto.
5. Contrastar los resultados de clasificación del objetivo anterior con resultados obtenidos por los mismos clasificadores pero sin realizar el proceso de alineación propuesto.
6. Publicación de resultados.

### 3.2 Objetivos Ajustados (SOLO SUGERENCIA/REFERENCIA)

**Archivo:** `5-Objetivos-Ajustados.tex`

**NOTA:** Estos objetivos ajustados son solo una referencia interna para mapear lo implementado. Los objetivos oficiales de la tesis son los del asesor (sección 3.1).

| Objetivo Ajustado | Mapeo a Objetivo Original |
|-------------------|---------------------------|
| Modelo de predicción de landmarks | Objetivo 1 (método de alineación) |
| Normalización geométrica (warping + GPA) | Objetivo 1 (normalización) |
| Evaluación de 7 arquitecturas CNN | Objetivo 3 (parcial - solo CNN) |
| Validación con métricas | Objetivo 4 |
| Cuantificación de contribución | Objetivo 5 |
| Evaluación de generalización | Objetivo 4 (validación cruzada) |

### 3.3 Brechas entre Objetivos y Trabajo Implementado

| Objetivo Original | Estado | Notas |
|-------------------|--------|-------|
| Objetivo 1 (alineación/normalización) | ✅ CUMPLIDO | Warping afín por partes + GPA |
| Objetivo 2 (selección características) | ⚠️ REINTERPRETADO | La normalización actúa como selección implícita |
| Objetivo 3 - CNN | ✅ CUMPLIDO | 7 arquitecturas evaluadas |
| Objetivo 3 - KNN | ❌ NO IMPLEMENTADO | Pendiente o justificar omisión |
| Objetivo 3 - MLP | ❌ NO IMPLEMENTADO | Pendiente o justificar omisión |
| Objetivo 4 (validación) | ✅ CUMPLIDO | Métricas + validación cruzada |
| Objetivo 5 (contraste con/sin) | ✅ CUMPLIDO | Experimentos de control |
| Objetivo 6 (publicación) | ⏳ PENDIENTE | Fuera del alcance de la tesis |

---

## 4. DECISIONES SOBRE DOCUMENTACIÓN

- **Documentación LaTeX existente (17 archivos .tex):** NO se reutilizará. Se empieza de cero.
- **Documentación técnica (REFERENCIA_SESIONES_FUTURAS.md, GROUND_TRUTH.json):** Se usará como referencia para datos validados.
- **Plantilla LaTeX:** Se usará plantilla estándar de tesis de posgrado.

---

## 5. CLAIMS CIENTÍFICOS VALIDADOS (Usar en Tesis)

| Claim | Valor | Sesión de Validación |
|-------|-------|---------------------|
| Error de landmarks (ensemble) | 3.71 px | Sesión 13 |
| Accuracy clasificación (warped_96) | 99.10% | Sesión 53 |
| Mejora robustez JPEG Q50 | 30× | Sesión 39 |
| Mejora robustez Blur | 2.4× | Sesión 39 |
| Mejora generalización cross-dataset | 2.4× | Sesión 39 |
| Mecanismo causal | 75% reducción info + 25% normalización geo | Sesión 39 |

---

## 6. CLAIMS INVALIDADOS (NO Usar)

| Claim Incorrecto | Corrección |
|------------------|------------|
| "11× mejor generalización" | Solo 2.4× |
| "Fuerza atención pulmonar" | PFS ≈ 0.49 = aleatorio |
| "Resuelve domain shift externo" | ~55% en FedCOVIDx ≈ aleatorio |

---

## 7. LIMITACIONES A DOCUMENTAR

1. **Domain shift:** Modelos no generalizan a otros hospitales (~55% externo)
2. **PFS ≈ 50%:** Atención del modelo no se enfoca específicamente en pulmones
3. **Dataset pequeño:** 999 imágenes
4. **Anotación manual:** Variabilidad inter-anotador no cuantificada

---

## 8. PRÓXIMOS PASOS

Pendiente aprobación del usuario para:
- [ ] Iniciar FASE 2: Propuesta de estructura de capítulos

---

*Documento generado como parte del proceso de redacción de tesis.*
