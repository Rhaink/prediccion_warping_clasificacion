Plan Detallado (de principio a fin)

  Fase 0 — Definición del objetivo y criterios de evidencia

  - Objetivo principal: demostrar que la alineación geométrica (warping) mejora la discriminación, incluso con clasificadores simples.
  - Pregunta central: ¿las imágenes alineadas producen características más separables que las no alineadas?
  - Evidencia esperada: mejoras consistentes en métricas y visualizaciones interpretables en cada etapa (no solo accuracy).

  Resultados esperados

  - Tabla comparativa: “sin alineación vs con alineación” en 2 clases y 3 clases.
  - Visualizaciones: ejemplos de warping, eigenfaces, distribuciones de características, Fisher por dimensión.
  - Métricas: accuracy, F1 macro, matriz de confusión, separabilidad (Fisher por dimensión).

  ———

  Fase 1 — Dataset y clases (2 vs 3)

  1. Inventario de datos
      - Revisar cómo vienen etiquetadas las clases: normal, neumonía, COVID.
      - Verificar balance de clases, tamaño de imágenes, formato.
  2. Definir escenarios
      - Escenario A (principal, asesor): 2 clases = {neumonía (incluye COVID), sano}
      - Escenario B (secundario): 3 clases = {sano, neumonía, COVID}
  3. Separación de datos
      - Split fijo: train/val/test con semillas (documentar seed).
      - Asegurar que las mismas imágenes estén en ambos escenarios (solo cambia la etiqueta).

  Evidencia

  - Tabla de conteos por clase y split.
  - Histograma de tamaños/formatos si hay variabilidad.

  ———

  Fase 2 — Preprocesamiento básico (antes del warping)

  1. Normalización geométrica básica
      - Unificar tamaño final (por ejemplo 256×256 o el que uses).
      - Recorte/padding consistente.
  2. Contraste
      - Aplicar CLAHE si ya lo usaste, pero guardar comparativos (con y sin CLAHE).
  3. Documentación
      - Guardar imágenes representativas de cada clase pre‑y post‑CLAHE.

  Evidencia

  - 6–9 ejemplos visuales por clase (antes/después).
  - Mini tabla con parámetros de CLAHE.

  ———

Fase 3 — Alineación (Warping)

  1. Definir puntos de referencia
      - Puntos clave (landmarks) o máscara pulmonar si ya existe.
  2. Modelo geométrico
      - Tipo de warping: afin, thin-plate spline, etc. (documentar cuál y por qué).
  3. Aplicar warping
      - A todas las imágenes.
  4. Validación del warping
      - Mostrar pares “original vs warped”.
      - Verificar que estructuras anatómicas estén alineadas.

  Evidencia

  - Panel comparativo con 8–12 imágenes.
  - Métrica simple: distancia promedio de landmarks antes vs después.
  - Entrega explícita de muestras de imágenes warped para revisión del asesor.

  ———

Fase 4 — Construcción del Eigen-space (PCA)

  1. Concepto geométrico
      - La imagen es un vector en un espacio de alta dimensión.
      - PCA encuentra direcciones de máxima varianza → eigenfaces.
  2. Procedimiento
      - Flatten de imágenes alineadas (warped únicamente).
      - Calcular PCA con solo training set.
      - Elegir número de componentes (por varianza explicada).
  3. Selección de K
      - Probar K = [5, 10, 20, 50…] para ver trade‑off.
  4. Reconstrucciones
      - Reconstruir imágenes con K componentes.

  Evidencia

  - Curva de varianza explicada.
  - Galería de eigenfaces.
  - Reconstrucciones (original vs reconstruida) por K.

  ———

  Fase 5 — Características (ponderantes)

  1. Definición
      - Cada imagen = vector de pesos en el espacio PCA (K dimensiones).
  2. Extraer pesos
      - Para cada imagen, guardar su vector de ponderantes.
  3. Interpretación
      - Cada dimensión representa variación dominante de la población.

  Evidencia

  - Tabla con ejemplo de ponderantes (por imagen).
  - Scatter plot de 2 o 3 dimensiones principales.

  ———

  Fase 6 — Estandarización de características

  1. Motivo
      - Evitar sesgos por escala.
  2. Proceso
      - Para cada dimensión: restar media y dividir por desviación estándar (solo training).
  3. Aplicar a val/test
      - Usar media y desviación del training.

  Evidencia

  - Gráficas antes y después (distribuciones).
  - Confirmar media≈0 y std≈1 en training.

  ———

  Fase 7 — Criterio de Fisher por característica

  1. Definición simple
      - Razón de Fisher = separación entre medias / dispersión intra‑clase.
  2. Cálculo
      - Para cada dimensión: calcular media y desviación por clase.
  3. Interpretación
      - Grande = buena separación, pequeño = poca separación.
  4. Ponderación
      - Multiplicar cada dimensión por su razón de Fisher.

  Evidencia

  - Tabla con Fisher por dimensión (ordenado).
  - Gráfica barras con top‑K dimensiones más discriminantes.

  ———

  Fase 8 — Clasificación simple

  1. Modelos
      - KNN y/o MLP pequeño (1–2 capas).
  2. Entrenamiento
      - Usar vectores ponderados por Fisher.
  3. Evaluación
      - Accuracy, F1 macro, matriz de confusión.
  4. Comparativa clave
      - Con y sin alineación, con y sin Fisher.

  Evidencia

  - Tabla comparativa de métricas.
  - Matrices de confusión.
  - Curvas de validación si aplica.

  ———

  Fase 9 — Experimento 2‑clases vs 3‑clases

  1. Repetir el pipeline en ambos escenarios.
  2. Comparar
      - ¿Dónde el warping aporta más?
      - ¿Se conserva la ganancia con 3 clases?
  3. Conclusión
      - El experimento principal sigue 2 clases (tesis), el 3‑clases es control.

  Evidencia

  - Tabla comparativa final con 4 escenarios:
      - 2C no‑warp / 2C warp / 3C no‑warp / 3C warp.

  ———

  Fase 10 — Interpretabilidad y evidencia fuerte

  1. Visualización en 2D/3D
      - Proyectar a 2D/3D para ver separación entre clases.
  2. Casos difíciles
      - Ejemplos mal clasificados y por qué.
  3. Sensibilidad
      - Variar K (componentes) y ver estabilidad.

  Evidencia

  - Proyecciones con colores por clase.
  - Tabla de errores y análisis breve.

  ———

Fase 11 — Documentación final

      - PCA/Eigenspace explicado con ejemplos visuales.
      - Fisher explicado con un ejemplo numérico sencillo y la fórmula explícita.
      - Resultados comparativos con tablas y figuras.
  - Guardar scripts y resultados replicables.

  ———

  Auditoría del plan (checklist final)

  - ¿El objetivo principal (alineación como aporte) está evaluado con evidencia sólida?
  - ¿Se puede reproducir cada etapa sin “caja negra”?
  - ¿Se muestran resultados visuales para warping, PCA y Fisher?
  - ¿Se documentó por qué se eligió K en PCA?
  - ¿Se usó solo el training para calcular medias/STD/PCA?
  - ¿La comparación 2‑clases vs 3‑clases está completa?
  - ¿Se incluyen fallos y análisis de errores?
  - ¿Los resultados muestran ganancia consistente con warping?
  - ¿El clasificador es simple y no domina el resultado?
  - ¿Hay tablas y figuras para cada fase crítica?
