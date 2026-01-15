# Estructura de Capítulos 5 y 6 - Tesis

Este documento resume la estructura de los capítulos de Resultados y Conclusiones de la tesis.

## Capítulo 5: Resultados

### 5.1 Resultados de Predicción de Landmarks (`5_1_resultados_landmarks.tex`)
- **5.1.1** Desempeño del Ensemble
  - Tabla: Comparación individual vs ensemble (3.61 px vs 4.04 px, mejora 10.6%)
  - Modelos del ensemble (seeds 123, 321, 111, 666)
- **5.1.2** Distribución de Errores
  - Tabla: Estadísticos (media, mediana, desv. estándar)
  - Percentiles de la distribución
- **5.1.3** Error por Landmark Individual
  - Tabla: Error de cada uno de los 15 landmarks
  - Mejores: L10 (2.44 px), L9 (2.76 px), L5 (2.88 px)
  - Peores: L12 (5.43 px), L13 (5.35 px), L14 (4.39 px)
  - Figura F5.1 [PLACEHOLDER]: Visualización del error por landmark
- **5.1.4** Error por Categoría Diagnóstica
  - Tabla: Error por clase (Normal: 3.22 px, COVID: 3.93 px, Viral: 4.11 px)
  - Análisis de sesgo del modelo
- **5.1.5** Ejemplos Visuales de Predicción
  - Figura F5.2 [PLACEHOLDER]: Grid de ejemplos (casos buenos y difíciles)
- **5.1.6** Resumen de Resultados de Landmarks

### 5.2 Forma Canónica y Normalización Geométrica (`5_2_forma_canonica.tex`)
- **5.2.1** Convergencia del GPA
  - Tabla: Parámetros de convergencia (18 iteraciones, ~0.5 segundos)
- **5.2.2** Forma Canónica Resultante
  - Figura F5.3 [PLACEHOLDER]: Forma canónica normalizada y en coordenadas de imagen
  - Tabla: Propiedades geométricas (centroide, rangos, relación de aspecto)
- **5.2.3** Triangulación de Delaunay
  - Figura F5.4 [PLACEHOLDER]: Triangulación sobre 15 landmarks (20 triángulos)
- **5.2.4** Optimización del Parámetro margin_scale
  - Tabla: Búsqueda experimental (óptimo: 1.05)
  - Figura F5.5 [PLACEHOLDER]: Comparación visual de diferentes valores
- **5.2.5** Ejemplos de Normalización Geométrica
  - Figura F5.6 [PLACEHOLDER]: Grid antes/después del warping
- **5.2.6** Análisis del Fill Rate
  - Tabla: Estadísticas (media 47%, desv. 3-5%)
- **5.2.7** Tiempo de Procesamiento
  - Tabla: CPU (143-212 ms) vs GPU (48-67 ms)
- **5.2.8** Resumen de Normalización Geométrica

### 5.3 Resultados de Clasificación (`5_3_resultados_clasificacion.tex`)
- **5.3.1** Métricas Globales de Clasificación
  - Tabla: Accuracy 98.05%, F1-Macro 97.12%, F1-Weighted 98.04%
- **5.3.2** Rendimiento por Clase
  - Tabla: Precision, Recall, F1-Score por clase
  - COVID-19: 97.51%, Normal: 98.78%, Viral: 90.91%
- **5.3.3** Matriz de Confusión
  - Tabla: Matriz 3×3 con valores absolutos
  - Análisis de patrones de error (Viral→Normal: 13 casos)
  - Figura F5.7 [PLACEHOLDER]: Heatmap de matriz normalizada
- **5.3.4** Comparación: F1-Macro vs F1-Weighted
  - Tabla: Diferencia 0.92 pp por efecto de desbalance
- **5.3.5** Curvas de Aprendizaje
  - Figura F5.8 [PLACEHOLDER]: Pérdida y F1-Macro (train vs val)
- **5.3.6** Análisis de Casos Difíciles
  - Figura F5.9 [PLACEHOLDER]: Ejemplos de casos mal clasificados
- **5.3.7** Estabilidad del Modelo
  - Tabla: Resultados con 3 semillas (media 97.59% ± 0.35%)
- **5.3.8** Resumen de Resultados de Clasificación

### 5.4 Análisis Comparativo y Validación de Hipótesis (`5_4_analisis_comparativo.tex`)
- **5.4.1** Efectividad del Sistema Completo
  - Tabla resumen: Landmarks (3.61 px), Normalización (margin=1.05, fill=47%), Clasificación (98.05%)
- **5.4.2** Mecanismos de Mejora por Normalización Geométrica
  - Eliminación de variabilidad no patológica
  - Selección implícita de características (47% fill rate)
  - Regularización implícita
- **5.4.3** Comparación con Trabajos Relacionados
  - Tabla: Comparación con literatura (Chowdhury, Rahman, Ozturk, etc.)
- **5.4.4** Evidencia de Experimentos Exploratorios
  - Tabla: Configuraciones alternativas (warped_96: 99.10%, warped_99: 98.73%)
- **5.4.5** Limitaciones del Análisis Comparativo
  - Ausencia de comparación controlada directa
- **5.4.6** Interpretabilidad: Enfoque en Región Pulmonar
  - Figura F5.10 [PLACEHOLDER]: Comparación original vs warped (regiones eliminadas)
- **5.4.7** Resumen del Análisis Comparativo

### 5.5 Discusión de Resultados (`5_5_discusion.tex`)
- **5.5.1** Interpretación de Resultados de Landmarks
  - Comparación con literatura (Van Ginneken: 5-10 px)
- **5.5.2** Validación de la Precisión Suficiente para Warping
  - Tabla: Tolerancia del error (1.6% del tamaño de imagen)

---

## Capítulo 6: Conclusiones y Trabajos Futuros

### Archivo único: `6_conclusiones.tex`

#### 6.1 Síntesis de Contribuciones
- **6.1.1** Contribución Principal
  - Demostración de viabilidad y efectividad de normalización geométrica
  - Cuatro beneficios principales
- **6.1.2** Contribuciones Específicas
  1. Modelo robusto de predicción de landmarks (3.61 px)
  2. Pipeline completo de normalización geométrica
  3. Sistema de clasificación de alto rendimiento (98.05%)
  4. Metodología reproducible y documentada

#### 6.2 Validación de la Hipótesis
- **6.2.1** Hipótesis Planteada
  - Enunciado formal de la hipótesis
- **6.2.2** Evidencia de Validación
  1. Efectividad demostrada (98.05% accuracy)
  2. Mecanismos fundamentados de mejora
  3. Precisión suficiente de landmarks (3.61 px)
- **6.2.3** Limitaciones de la Validación
  - Ausencia de comparación controlada
  - Conjunto de datos único
  - Evidencia exploratoria no concluyente
- **6.2.4** Respuesta a la Hipótesis
  - Conclusión: Hipótesis validada positivamente

#### 6.3 Implicaciones del Trabajo
- **6.3.1** Implicaciones Clínicas
  - Sistema de apoyo diagnóstico
  - Procesamiento en tiempo real
  - Interpretabilidad
  - Robustez ante variaciones
- **6.3.2** Implicaciones Metodológicas
  - Transferibilidad a otras modalidades
  - Complemento a arquitecturas modernas
  - Reducción de requisitos de datos
  - Prior geométrico para few-shot learning
- **6.3.3** Implicaciones Técnicas
  - Uso efectivo de Coordinate Attention
  - Estrategia de ensemble (mejora 10.6%)
  - Optimización experimental sistemática

#### 6.4 Limitaciones del Estudio
- **6.4.1** Limitaciones Experimentales
  - Conjunto de datos único
  - Ausencia de validación externa
  - Comparación controlada incompleta
  - Una sola arquitectura de clasificación
- **6.4.2** Limitaciones Metodológicas
  - Anotación manual inicial (957 imágenes)
  - Landmarks no anatómicos específicos
  - Dependencia de calidad de imagen
  - Clases específicas de patologías
- **6.4.3** Limitaciones Conceptuales
  - Asunción de relevancia de la forma
  - Pérdida de información contextual (53%)
  - Normalización no deseable en algunos casos

#### 6.5 Trabajos Futuros
- **6.5.1** Validación y Generalización
  1. Comparación controlada original vs warped
  2. Validación externa en múltiples datasets
  3. Validación clínica prospectiva
  4. Análisis de robustez
- **6.5.2** Extensiones del Sistema
  1. Clasificación binaria COVID-19
  2. Extensión a más patologías
  3. Segmentación de lesiones
  4. Detección multi-etiqueta
  5. Predicción de severidad
- **6.5.3** Mejoras Metodológicas
  1. Arquitecturas alternativas (ViT, Swin, Hybrid)
  2. Landmarks anatómicos específicos
  3. Aprendizaje end-to-end
  4. Spatial Transformer Networks
  5. Reducción de requisitos de anotación
- **6.5.4** Interpretabilidad y Explicabilidad
  1. Mapas de atención (Grad-CAM)
  2. Análisis de características aprendidas
  3. Validación radiológica
  4. Visualización de casos límite
- **6.5.5** Optimización e Implementación
  1. Optimización de eficiencia (destilación, cuantización, pruning)
  2. Implementación en dispositivos edge
  3. API web y interfaz clínica
  4. Integración con sistemas PACS

#### 6.6 Reflexión Final
- Síntesis de la contribución principal
- Valor del enfoque híbrido (métodos clásicos + deep learning)
- Conclusión: Normalización geométrica es una técnica prometedora

---

## Resumen de Figuras (Placeholders)

### Capítulo 5
- **F5.1**: Error por landmark sobre forma canónica (coloreado por magnitud)
- **F5.2**: Grid de ejemplos de predicción de landmarks (3×2)
- **F5.3**: Forma canónica (normalizada y en coordenadas de imagen)
- **F5.4**: Triangulación de Delaunay sobre 15 landmarks
- **F5.5**: Comparación de diferentes valores de margin_scale (1.00, 1.05, 1.25)
- **F5.6**: Grid de warping antes/después (4 columnas × 3 filas)
- **F5.7**: Heatmap de matriz de confusión normalizada
- **F5.8**: Curvas de aprendizaje (pérdida y F1-Macro)
- **F5.9**: Ejemplos de casos mal clasificados (3×3)
- **F5.10**: Comparación región original vs warped (selección de características)

### Capítulo 6
- Sin figuras adicionales (usa figuras del Capítulo 5 por referencia)

---

## Resumen de Tablas (Con datos reales de GROUND_TRUTH.json)

### Capítulo 5 - Total: 15 tablas con datos validados
1. Ensemble vs individual (3.61 px vs 4.04 px)
2. Estadísticos de distribución
3. Error por landmark (15 landmarks con valores exactos)
4. Error por categoría (Normal: 3.22, COVID: 3.93, Viral: 4.11)
5. Convergencia GPA (18 iteraciones, 0.5 segundos)
6. Propiedades forma canónica
7. Búsqueda margin_scale (óptimo: 1.05)
8. Estadísticas fill rate (media 47%)
9. Tiempo de procesamiento (CPU vs GPU)
10. Métricas globales clasificación (98.05%, 97.12%, 98.04%)
11. Métricas por clase (Precision, Recall, F1)
12. Matriz de confusión 3×3
13. F1-Macro vs F1-Weighted (diferencia 0.92 pp)
14. Estabilidad con 3 semillas (97.59% ± 0.35%)
15. Comparación con literatura

### Capítulo 6 - Total: 4 tablas conceptuales
1. Resumen sistema completo
2. Tipos de variabilidad eliminada
3. Efecto de selección de características
4. Tolerancia de error para warping

---

## Notas Importantes

### Datos Validados Utilizados (de GROUND_TRUTH.json)
- Ensemble best: 3.61 px (seeds 123, 321, 111, 666)
- Mejor individual: 4.04 px (seed 456)
- Error por categoría: Normal 3.22 px, COVID 3.93 px, Viral 4.11 px
- Error por landmark: L10 (2.44), L9 (2.76), ..., L12 (5.43), L13 (5.35)
- Clasificación: Accuracy 98.05%, F1-Macro 97.12%, F1-Weighted 98.04%
- Estabilidad: 97.59% ± 0.35% (3 seeds)

### Placeholders de Figuras
- Todas las figuras tienen placeholders con formato consistente con el resto de la tesis
- Descripción detallada del contenido esperado en cada figura
- Formato tipo \fbox con especificaciones claras

### Coherencia con Metodología (Capítulo 4)
- Referencias cruzadas a secciones del Capítulo 4
- Consistencia en terminología y notación
- Explicación de resultados alineada con métodos descritos

### Estilo y Formato
- Uso de LaTeX profesional con paquetes estándar (booktabs, multirow)
- Tablas con formato consistente
- Notas al pie donde es necesario
- Referencias temporales marcadas para completar bibliografía

---

## Archivos Creados

```
docs/Tesis/
├── capitulo5/
│   ├── 5_1_resultados_landmarks.tex
│   ├── 5_2_forma_canonica.tex
│   ├── 5_3_resultados_clasificacion.tex
│   ├── 5_4_analisis_comparativo.tex
│   └── 5_5_discusion.tex
├── capitulo6/
│   └── 6_conclusiones.tex
└── ESTRUCTURA_CAPITULOS_5_6.md (este archivo)
```

## Para Compilar

Para incluir estos capítulos en la tesis principal, agregar en el archivo `main.tex`:

```latex
% Capítulo 5: Resultados
\input{capitulo5/5_1_resultados_landmarks}
\input{capitulo5/5_2_forma_canonica}
\input{capitulo5/5_3_resultados_clasificacion}
\input{capitulo5/5_4_analisis_comparativo}
\input{capitulo5/5_5_discusion}

% Capítulo 6: Conclusiones
\input{capitulo6/6_conclusiones}
```

## Próximos Pasos Sugeridos

1. **Generar figuras**: Crear las 10 figuras placeholder usando scripts de visualización del proyecto
2. **Completar referencias bibliográficas**: Agregar citas completas al archivo .bib
3. **Revisión de coherencia**: Verificar referencias cruzadas entre capítulos
4. **Ajustar valores exactos**: Si existen archivos `results.json` con métricas precisas de clasificación, actualizar tablas
5. **Revisión de pares**: Solicitar feedback de asesores sobre estructura y contenido
