Nota de objetivos y criterios (Semana 1)

Objetivo principal
- Demostrar que la alineacion geometrica (warping) mejora la discriminacion
  entre clases en un pipeline simple (PCA + Fisher + clasificador simple).

Pregunta central
- ¿Las imagenes alineadas producen caracteristicas mas separables que las no
  alineadas, manteniendo el mismo split y protocolo?

Criterios de evidencia
- Comparacion directa: original vs warped con el mismo split.
- Metricas robustas al desbalance: macro-F1 y balanced accuracy, ademas de
  accuracy y matriz de confusion.
- Evidencia visual y numerica en cada fase (PCA, Fisher, clasificador).

Alcance y restricciones
- Dataset manual (957 con coords) se usa sin balanceo.
- Dataset completo se usa balanceado por split para 3-clases y 2-clases.

Principios de trabajo
- Explicar matematicas y geometria paso a paso, sin cajas negras.
- Priorizar clasificador simple como evidencia principal.
