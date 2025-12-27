# FIGURAS PENDIENTES DE LA TESIS

**Fecha de creación:** 16 Diciembre 2025
**Última actualización:** 17 Diciembre 2025 - Sesión 08
**Estado:** En progreso

Este documento registra las figuras que deben crearse para la tesis.

---

## CAPÍTULO 4: METODOLOGÍA

### Sección 4.1 - Descripción General del Sistema

| ID | Figura | Descripción | Archivo Destino | Estado |
|----|--------|-------------|-----------------|--------|
| F4.1 | Diagrama de fases del sistema | Diagrama que muestre las dos fases: Preparación (offline) y Operación (runtime) | `Figures/fases_sistema.pdf` | ⏳ Pendiente |
| F4.2 | Diagrama de bloques del pipeline | Diagrama del pipeline de operación: Entrada → Preprocesamiento → Predicción Landmarks → Normalización → Clasificación | `Figures/pipeline_general.pdf` | ⏳ Pendiente |

**Especificaciones para F4.1 (Fases del Sistema):**
- Formato: PDF vectorial o PNG ≥300 DPI
- Dos bloques principales:
  - **Fase de Preparación (offline):** Anotación manual de landmarks → Entrenamiento de modelos → Cálculo de forma canónica (GPA)
  - **Fase de Operación (runtime):** Pipeline de 4 módulos
- Indicar que la fase de preparación se ejecuta una sola vez
- Usar colores diferentes para distinguir ambas fases

**Especificaciones para F4.2 (Pipeline de Operación):**
- Formato: PDF vectorial o PNG ≥300 DPI
- Mostrar los 4 módulos: Preprocesamiento, Predicción de Landmarks, Normalización Geométrica, Clasificación
- Incluir dimensiones de datos en cada etapa (224×224×3 → 15×2 → 224×224×3 → 3)
- Usar flechas para indicar flujo
- Colores consistentes con el resto de la tesis

---

## CAPÍTULO 4: METODOLOGÍA - Sección 4.2

| ID | Figura | Sección | Estado |
|----|--------|---------|--------|
| F4.2a | Diagrama de 15 landmarks sobre radiografía | 4.2 | ⏳ Pendiente |
| F4.2b | Interfaz de la herramienta de etiquetado | 4.2 | ⏳ Pendiente |
| F4.3 | Comparación CLAHE (antes/después) | 4.2 | ⏳ Pendiente |

**Especificaciones para F4.2a (Landmarks Anatómicos):**
- Mostrar radiografía de ejemplo con los 15 landmarks numerados
- Indicar eje central (L1-L2), contornos bilaterales, pares simétricos
- Usar colores diferentes para distinguir grupos de landmarks

**Especificaciones para F4.2b (Herramienta de Etiquetado):**
- Captura de pantalla de la herramienta desarrollada en OpenCV
- Mostrar:
  - Radiografía con landmarks superpuestos
  - Línea central azul de referencia
  - Puntos verdes numerados (landmarks)
  - Líneas rojas conectando el contorno
- Opcionalmente: incluir panel con menú de teclas de ajuste

**Especificaciones para F4.3 (CLAHE):**
- Comparación lado a lado: imagen original vs CLAHE
- Mostrar mejora de contraste en región pulmonar
- Parámetros: clip_limit=2.0, tile_size=4

---

## CAPÍTULO 4: METODOLOGÍA - Sección 4.3

| ID | Figura | Sección | Estado |
|----|--------|---------|--------|
| F4.4 | Arquitectura ResNet-18 + Coordinate Attention | 4.3 | ⏳ Pendiente |
| F4.5 | Detalle del módulo Coordinate Attention | 4.3 | ⏳ Pendiente |

---

## CAPÍTULO 4: METODOLOGÍA - Sección 4.4

| ID | Figura | Sección | Estado |
|----|--------|---------|--------|
| F4.6 | Proceso de GPA | 4.4 | ⏳ Pendiente |
| F4.7 | Triangulación de Delaunay sobre landmarks | 4.4 | ⏳ Pendiente |
| F4.8 | Comparación imagen original vs warped | 4.4 | ⏳ Pendiente |
| F4.9 | Efecto de diferentes margin_scale | 4.4 | ⏳ Pendiente |
| F4.10 | Pipeline completo de normalización geométrica | 4.4 | ⏳ Pendiente |

**Especificaciones para F4.6 (Proceso de GPA):**
- Panel de 4 subfiguras:
  - (a) Configuraciones originales de landmarks superpuestas (957 formas)
  - (b) Después de centrado y escalado (norma unitaria)
  - (c) Después de alineación rotacional
  - (d) Forma canónica resultante (consenso de Procrustes)
- Mostrar variabilidad reduciéndose en cada paso

**Especificaciones para F4.7 (Triangulación Delaunay):**
- Mostrar la forma canónica con los 15 landmarks
- Conectar landmarks con triangulación de Delaunay
- Indicar número de triángulos (~20-25)
- Usar colores para distinguir triángulos

**Especificaciones para F4.8 (Original vs Warped):**
- Panel de 3 subfiguras:
  - (a) Imagen original
  - (b) Imagen warped SIN full coverage (fill rate ~47%)
  - (c) Imagen warped CON full coverage (fill rate ~96%)
- Mostrar esquinas negras en (b) vs completas en (c)

**Especificaciones para F4.9 (Efecto margin_scale):**
- Panel de 3 subfiguras comparando:
  - (a) margin_scale = 1.00 (sin margen)
  - (b) margin_scale = 1.05 (óptimo)
  - (c) margin_scale = 1.25 (excesivo)
- Mostrar diferencias en cobertura de región pulmonar

**Especificaciones para F4.10 (Pipeline de normalización):**
- Diagrama de flujo mostrando:
  1. Predicción de landmarks → 2. Escalado margin_scale →
  3. Adición puntos borde → 4. Triangulación → 5. Warping → 6. Salida
- Incluir dimensiones de datos en cada paso

---

## CAPÍTULO 4: METODOLOGÍA - Sección 4.5

| ID | Figura | Sección | Estado |
|----|--------|---------|--------|
| F4.11 | Ejemplos de data augmentation del clasificador | 4.5 | ⏳ Pendiente |
| F4.12 | Arquitectura del clasificador (opcional) | 4.5 | ⏳ Opcional |

**Especificaciones para F4.11 (Data Augmentation):**
- Panel de 4 subfiguras mostrando:
  - (a) Imagen original normalizada
  - (b) Flip horizontal
  - (c) Rotación aleatoria (±10°)
  - (d) Traslación + escala
- Usar imagen de ejemplo representativa del dataset
- Mostrar que las transformaciones preservan contenido diagnóstico

---

## CAPÍTULO 4: METODOLOGÍA - Sección 4.6

| ID | Figura | Sección | Estado |
|----|--------|---------|--------|
| F4.13 | Esquema de evaluación cruzada (cross-evaluation) | 4.6 | ⏳ Pendiente |
| F4.14 | Ejemplos de perturbaciones de robustez | 4.6 | ⏳ Pendiente |

**Especificaciones para F4.13 (Cross-Evaluation):**
- Diagrama de matriz 2×2 mostrando:
  - Eje horizontal: Dataset de evaluación (Original, Warped)
  - Eje vertical: Modelo entrenado (Original, Warped)
  - Cuadrantes diagonales: In-domain (coloreados similar)
  - Cuadrantes off-diagonal: Cross-domain (coloreados diferente)
- Flechas indicando dirección de evaluación
- Etiquetas claras: $Acc_{O \to O}$, $Acc_{O \to W}$, etc.

**Especificaciones para F4.14 (Perturbaciones de Robustez):**
- Panel de 5 subfiguras mostrando:
  - (a) Imagen original sin perturbación
  - (b) JPEG Q50 (compresión moderada)
  - (c) JPEG Q30 (compresión severa, artefactos visibles)
  - (d) Blur σ=1 (desenfoque leve)
  - (e) Blur σ=2 (desenfoque moderado)
- Usar misma imagen de ejemplo para comparar
- Indicar parámetros en cada subfigura

---

## CAPÍTULO 5: RESULTADOS (anticipadas)

| ID | Figura | Sección | Estado |
|----|--------|---------|--------|
| F5.1 | Gráfica de error de landmarks por categoría | 5.1 | ⏳ Pendiente |
| F5.2 | Matriz de confusión del clasificador | 5.2 | ⏳ Pendiente |
| F5.3 | Comparación de robustez (barras) | 5.3 | ⏳ Pendiente |
| F5.4 | Resultados de cross-evaluation | 5.4 | ⏳ Pendiente |

---

## NOTAS

- Las figuras se crearán una vez que se complete la redacción de cada sección
- Prioridad: Crear figuras críticas antes de la revisión final
- Verificar que todas las figuras estén referenciadas en el texto

---

*Última actualización: 17 Diciembre 2025 - Sesión 08*
