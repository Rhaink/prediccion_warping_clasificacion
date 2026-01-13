# Lista de Deseos del Proyecto

**Proyecto:** Detección de COVID-19 mediante Landmarks Anatómicos y Normalización Geométrica
**Autor:** [Nombre del Autor]
**Fecha:** Enero 2026

---

## Propósito de Este Documento

Este documento expresa los deseos y expectativas sobre lo que el proyecto debe lograr y demostrar. Sirve como guía para evaluar si los objetivos de la tesis han sido cumplidos.

---

## 1. Deseos sobre la Hipótesis Principal

### 1.1. Hipótesis Central

> **Deseo demostrar que el alineamiento geométrico (warping) de radiografías de tórax a una forma canónica mejora la clasificación de COVID-19 en comparación con el uso de imágenes sin normalizar.**

### 1.2. Deseos Específicos sobre la Hipótesis

- [ ] **Deseo** que los modelos entrenados con imágenes warpeadas alcancen **mayor accuracy** que los modelos entrenados con imágenes originales.

- [ ] **Deseo** que la diferencia de accuracy sea estadísticamente significativa y no producto del azar.

- [ ] **Deseo** poder rechazar la hipótesis nula (H₀: no hay diferencia) con evidencia experimental sólida.

---

## 2. Deseos sobre Predicción de Landmarks

### 2.1. Precisión

- [ ] **Deseo** que el modelo de predicción de landmarks alcance un error **menor a 5 píxeles** en promedio.

- [ ] **Deseo** que el error sea **menor a 4 píxeles** para demostrar precisión suficiente para el warping.

- [ ] **Deseo** que el ensemble de modelos mejore el rendimiento respecto a un modelo individual.

### 2.2. Consistencia

- [ ] **Deseo** que el error sea consistente entre las tres categorías (COVID, Normal, Neumonía Viral), sin sesgos hacia ninguna clase.

- [ ] **Deseo** que los landmarks centrales tengan menor error que los landmarks de borde (validando que el modelo aprende la estructura anatómica).

---

## 3. Deseos sobre Clasificación

### 3.1. Accuracy

- [ ] **Deseo** que el clasificador entrenado con imágenes warpeadas alcance una accuracy **mayor al 95%**.

- [ ] **Deseo** que la accuracy sea **mayor al 98%** para ser competitivo con el estado del arte.

- [ ] **Deseo** que la accuracy warped sea **igual o superior** a la accuracy con imágenes originales.

### 3.2. Métricas Adicionales

- [ ] **Deseo** que el F1-Score sea **mayor al 95%** para demostrar balance entre precisión y recall.

- [ ] **Deseo** que ninguna clase tenga un recall **menor al 90%** (evitar sesgo de clase).

---

## 4. Deseos sobre Robustez (MUY IMPORTANTE)

### 4.1. Robustez a Compresión JPEG

- [ ] **Deseo** que los modelos warped sean **significativamente más robustos** a compresión JPEG que los modelos originales.

- [ ] **Deseo** que la degradación de accuracy bajo JPEG Q50 sea **menor al 5%** para modelos warped.

- [ ] **Deseo** demostrar una mejora de robustez de **al menos 2x** (idealmente 5x o más).

### 4.2. Robustez a Blur

- [ ] **Deseo** que los modelos warped mantengan su rendimiento bajo blur gaussiano.

- [ ] **Deseo** que la degradación bajo blur σ=1 sea **menor al 5%**.

### 4.3. Robustez a Ruido

- [ ] **Deseo** que los modelos warped sean **al menos tan robustos** como los originales frente a ruido gaussiano.

### 4.4. Justificación Clínica

- [ ] **Deseo** poder argumentar que la robustez es importante para despliegue clínico real, donde las imágenes pueden estar comprimidas o degradadas.

---

## 5. Deseos sobre Generalización

### 5.1. Generalización Within-Domain

- [ ] **Deseo** que los modelos warped generalicen **mejor** que los originales entre variantes del mismo dataset.

- [ ] **Deseo** que el gap de generalización cruzada sea **menor** para modelos warped.

- [ ] **Deseo** demostrar una mejora de generalización de **al menos 1.5x** (idealmente 2x o más).

### 5.2. Generalización Cross-Domain (Limitación Esperada)

- [ ] **Deseo** evaluar honestamente el rendimiento en datos externos (otro hospital).

- [ ] **Deseo** documentar las limitaciones de domain shift de manera transparente.

- [ ] **Deseo** demostrar que el domain shift afecta **tanto a warped como a original** (no es culpa del método).

---

## 6. Deseos sobre Validación Científica

### 6.1. Validación Geométrica

- [ ] **Deseo** validar que el efecto del warping es **genuinamente geométrico**, no un artefacto del deep learning.

- [ ] **Deseo** usar métodos clásicos (Fisher LDA, PCA) para demostrar que el warping mejora la separabilidad lineal.

- [ ] **Deseo** demostrar una mejora de **al menos 3%** en accuracy con métodos clásicos.

### 6.2. Análisis del Mecanismo

- [ ] **Deseo** entender **por qué** el warping mejora la robustez.

- [ ] **Deseo** descomponer el mecanismo en componentes cuantificables.

- [ ] **Deseo** poder explicar el mecanismo de forma clara y defendible.

---

## 7. Deseos sobre Comparaciones

### 7.1. Comparación Justa

- [ ] **Deseo** que la comparación warped vs original sea **justa y bajo las mismas condiciones**:
  - Misma arquitectura de clasificador
  - Mismo preprocesamiento (CLAHE)
  - Mismos splits de datos
  - Mismos hiperparámetros

### 7.2. Múltiples Arquitecturas

- [ ] **Deseo** evaluar múltiples arquitecturas de CNN para demostrar que el efecto no es específico de una arquitectura.

- [ ] **Deseo** incluir al menos 5 arquitecturas diferentes (ResNet, EfficientNet, DenseNet, VGG, MobileNet).

---

## 8. Deseos sobre Reproducibilidad

### 8.1. Código

- [ ] **Deseo** que todo el código sea reproducible con seeds fijos.

- [ ] **Deseo** que los comandos de reproducción estén documentados.

- [ ] **Deseo** que los checkpoints estén disponibles.

### 8.2. Documentación

- [ ] **Deseo** que cada experimento tenga documentación de sesión.

- [ ] **Deseo** que los resultados estén validados en un archivo de referencia (GROUND_TRUTH.json).

---

## 9. Deseos sobre la Tesis

### 9.1. Estructura

- [ ] **Deseo** que la tesis tenga una estructura clara y lógica.

- [ ] **Deseo** que incluya marco teórico completo sobre:
  - COVID-19 y diagnóstico por imagen
  - Deep learning en imágenes médicas
  - Normalización geométrica y warping
  - Landmarks anatómicos

### 9.2. Resultados

- [ ] **Deseo** que todas las tablas de resultados estén completas y sean claras.

- [ ] **Deseo** incluir visualizaciones (Grad-CAM, ejemplos de warping, gráficos de robustez).

### 9.3. Discusión Honesta

- [ ] **Deseo** discutir honestamente las limitaciones.

- [ ] **Deseo** no sobreinterpretar los resultados.

- [ ] **Deseo** incluir trabajo futuro realista.

---

## 10. Deseos sobre el Artículo

### 10.1. Contribuciones Claras

- [ ] **Deseo** que el artículo tenga contribuciones claras y novedosas.

- [ ] **Deseo** que sea publicable en una revista indexada.

### 10.2. Formato

- [ ] **Deseo** que siga el formato de revistas como IEEE, MDPI, o Springer.

- [ ] **Deseo** que tenga entre 8-12 páginas.

### 10.3. Claims Defendibles

- [ ] **Deseo** que todos los claims sean defendibles con evidencia experimental.

---

## 11. Resumen de Métricas Deseadas

| Aspecto | Métrica Deseada | Mínimo Aceptable |
|---------|-----------------|------------------|
| Error de landmarks | < 4 px | < 5 px |
| Accuracy clasificación | > 98% | > 95% |
| F1-Score | > 98% | > 95% |
| Degradación JPEG Q50 | < 3% | < 5% |
| Degradación Blur | < 3% | < 5% |
| Mejora robustez vs original | > 5x | > 2x |
| Mejora generalización | > 2x | > 1.5x |
| Mejora Fisher LDA | > 3% | > 2% |

---

## 12. Checklist Final de Deseos

### Hipótesis
- [ ] Demostrar que warping mejora clasificación
- [ ] Evidencia estadística sólida

### Predicción de Landmarks
- [ ] Error < 5 px (idealmente < 4 px)
- [ ] Consistencia entre categorías

### Clasificación
- [ ] Accuracy > 98%
- [ ] Warped ≥ Original

### Robustez
- [ ] Mejora > 5x en JPEG
- [ ] Mejora > 2x en Blur
- [ ] Degradación < 5% bajo perturbaciones

### Generalización
- [ ] Mejora > 2x within-domain
- [ ] Domain shift documentado honestamente

### Validación Científica
- [ ] Fisher LDA confirma efecto geométrico
- [ ] Mecanismo explicado

### Comparación Justa
- [ ] Mismas condiciones para ambos
- [ ] Múltiples arquitecturas

### Documentación
- [ ] Tesis completa en LaTeX
- [ ] Artículo publicable
- [ ] Código reproducible

---

## 13. Notas para el Equipo

### Para el Programador
Asegúrate de que los experimentos comparativos warped vs original estén bajo las **mismas condiciones exactas**. Esto es crítico para la validez de la hipótesis.

### Para el Redactor
El tono de la tesis debe ser **objetivo y científico**. No exagerar los claims. Documentar las limitaciones (especialmente domain shift) de manera honesta.

### Para el Director
Los resultados muestran que el warping mejora robustez significativamente pero **no resuelve domain shift**. Esto es una limitación fundamental de medical imaging, no del método propuesto.

---

*Este documento representa los deseos y expectativas del proyecto. Sirve como guía para evaluar el cumplimiento de objetivos.*
