# 13. Figure Generation Scripts

Analisis de los scripts de generacion de figuras para la tesis y publicaciones.

**Archivos analizados**: 48

---

## Resumen Ejecutivo

Los scripts de generacion de figuras se dividen en dos grandes grupos:

1. **Scripts de nivel superior (24)** en `scripts/`: Generan figuras especificas para capitulos de la tesis (F2.x, F4.x, F5.x), matrices de confusion, comparaciones de clasificadores, y visualizaciones de GPA/landmarks.

2. **Scripts de presentacion (24)** en `scripts/visualization/`: Generan slides y assets para la presentacion de defensa de tesis (bloques 1-8), diagramas de arquitectura, animaciones, y mapas de atencion.

**Problemas principales detectados**:
- Proliferacion masiva de versiones: F5.3 tiene 3 versiones, F5.8 tiene 4 versiones, F5.9 tiene 3 versiones, GPA methodology tiene 2 versiones, bloques de presentacion tienen multiples iteraciones.
- Codigo duplicado extenso entre versiones (funciones de ajuste de matrices de confusion repetidas en 3 archivos con >100 lineas identicas).
- El script `generate_thesis_figures_master.py` (3,129 lineas) es un monolito que intenta generar 22 figuras con toda la logica embebida.
- Los scripts de presentacion (bloque1-8) son independientes pero repiten las mismas definiciones de colores y configuracion en cada archivo (~50 lineas identicas).

---

## A. Scripts de Nivel Superior (scripts/)

### 1. create_reference_image.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/create_reference_image.py
- **Lineas/Tamano**: 95 lineas / ~3 KB
- **Proposito**: Genera imagen de referencia con landmarks numerados sobre una radiografia Normal para verificar visualmente las conexiones anatomicas.
- **Figuras generadas**: REFERENCIA_landmarks_numerados.png (debug)
- **Importancia**: BAJO
- **Justificacion**: Script de debug/exploracion usado una sola vez para definir conexiones anatomicas. No genera figuras para la tesis.

### 2. create_thesis_figures.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/create_thesis_figures.py
- **Lineas/Tamano**: 383 lineas / ~13 KB
- **Proposito**: Genera figuras comparativas de GradCAM (original vs warped) y un resumen de metricas de generalizacion y robustez JPEG. Session 34.
- **Figuras generadas**: Comparaciones lado-a-lado, cross-domain GradCAM, matrices de atencion 2x2, resumen de metricas (outputs/thesis_figures/combined_figures/)
- **Importancia**: MEDIO
- **Justificacion**: Genera figuras de GradCAM utiles para la tesis, pero depende de datos de sesiones anteriores (29, 30) que pueden no estar disponibles. Las metricas hardcodeadas (gap 25.36% vs 2.24%) corresponden a experimentos pasados.

### 3. generate_all_visualizations.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_all_visualizations.py
- **Lineas/Tamano**: 417 lineas / ~14 KB
- **Proposito**: Genera visualizaciones de triangulacion de Delaunay (GT vs prediccion) para TODAS las imagenes del test set. Incluye modos comparison, side-by-side, y separate, mas una grilla resumen.
- **Figuras generadas**: Visualizaciones por imagen en outputs/predictions/all_visualizations/ (debug/exploracion)
- **Importancia**: BAJO
- **Justificacion**: Script de exploracion masiva. Genera cientos de imagenes individuales, no figuras especificas para la tesis. Depende de outputs/predictions/test_predictions.npz del pipeline antiguo.

### 4. generate_confusion_matrix_cv.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_confusion_matrix_cv.py
- **Lineas/Tamano**: 348 lineas / ~12 KB
- **Proposito**: Genera matriz de confusion agregada de validacion cruzada (5 folds) evaluada en test set. Soporta idiomas es/en.
- **Figuras generadas**: F5.7_matriz_confusion_cv.png (docs/Tesis/Figures/)
- **Importancia**: CRITICO
- **Justificacion**: Genera la figura F5.7 del capitulo de resultados de clasificacion con CV. Bien estructurada, con soporte bilingue, y metricas por clase.

### 5. generate_confusion_matrix_sahs.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_confusion_matrix_sahs.py
- **Lineas/Tamano**: 374 lineas / ~13 KB
- **Proposito**: Genera matriz de confusion del clasificador warped_sahs_masked y comparacion de 3 configuraciones SAHS (original, normalizado, cropped). Soporta es/en.
- **Figuras generadas**: F5.7_matriz_confusion_sahs.png, F5.8_comparacion_sahs.png (docs/Tesis/Figures/)
- **Importancia**: ALTO
- **Justificacion**: Genera figuras del capitulo de resultados SAHS. Tiene duplicacion significativa de logica de plotting con generate_confusion_matrix_cv.py.

### 6. generate_cv_figures_master.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_cv_figures_master.py
- **Lineas/Tamano**: 206 lineas / ~6 KB
- **Proposito**: Script maestro que ejecuta secuencialmente los generadores de F5.7, F5.8 y F5.9 de validacion cruzada, con verificacion de outputs.
- **Figuras generadas**: Orquesta F5.7, F5.8, F5.9 (todas CV)
- **Importancia**: ALTO
- **Justificacion**: Orquestador util para regenerar todas las figuras CV de una vez. Bien estructurado con verificacion de outputs y flags de skip.

### 7. generate_F2_clahe_vs_sahs.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_F2_clahe_vs_sahs.py
- **Lineas/Tamano**: 196 lineas / ~6 KB
- **Proposito**: Genera figura comparativa CLAHE vs SAHS para el marco teorico. Muestra 3 clases x 3 columnas (original, CLAHE, SAHS) con histogramas.
- **Figuras generadas**: F2.3_clahe_vs_sahs.png (docs/Tesis/Figures/)
- **Importancia**: ALTO
- **Justificacion**: Figura del capitulo de marco teorico. Auto-contenida, reimplementa CLAHE y SAHS en el propio script. Path hardcodeado al dataset.

### 8. generate_F5_3_scientific.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_F5_3_scientific.py
- **Lineas/Tamano**: 163 lineas / ~5 KB
- **Proposito**: Genera figura F5.3 con dos paneles: forma canonica + triangulacion de Delaunay. Estilo cientifico.
- **Figuras generadas**: F5.3_forma_canonica.png (docs/Tesis/Figures/)
- **Importancia**: ELIMINABLE
- **Justificacion**: Version anterior con 2 paneles. Supersedida por generate_F5_3_single_panel_fixed.py que solo muestra la forma canonica (la triangulacion paso a F5.4).

### 9. generate_F5_3_single_panel_fixed.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_F5_3_single_panel_fixed.py
- **Lineas/Tamano**: 181 lineas / ~6 KB
- **Proposito**: Genera figura F5.3 (version corregida, panel unico) mostrando la forma estandar pulmonar del GPA con contornos pulmonares correctos. Usa la misma estructura de contornos que update_all_figures.py.
- **Figuras generadas**: F5.3_forma_canonica.png (docs/Tesis/Figures/)
- **Importancia**: CRITICO
- **Justificacion**: Version final y correcta de F5.3. Incluye validaciones de datos del GPA (957 radiografias, 15 landmarks). Contornos anatomicos correctos: L1->L12->L3->L5->L7->L14->L2.

### 10. generate_F5_3_single_panel.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_F5_3_single_panel.py
- **Lineas/Tamano**: 184 lineas / ~6 KB
- **Proposito**: Version intermedia de F5.3 (panel unico) con contornos que cierran en L11 en lugar de L2.
- **Figuras generadas**: F5.3_forma_canonica.png (docs/Tesis/Figures/)
- **Importancia**: ELIMINABLE
- **Justificacion**: Version intermedia supersedida por _fixed. Los contornos cierran en L11 en lugar de L2 (incorrecto respecto al estandar del proyecto).

### 11. generate_F5_6_warping_sahs.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_F5_6_warping_sahs.py
- **Lineas/Tamano**: 155 lineas / ~5 KB
- **Proposito**: Genera grid 3x4 con ejemplos de imagenes warped+SAHS por clase (COVID, Normal, Viral Pneumonia).
- **Figuras generadas**: F5.6_ejemplos_warping.png (docs/Tesis/Figures/)
- **Importancia**: ALTO
- **Justificacion**: Figura de ejemplos visuales del warping para el capitulo de resultados. Compacta y bien estructurada.

### 12. generate_F5_8_comparison_cv.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_F5_8_comparison_cv.py
- **Lineas/Tamano**: 449 lineas / ~16 KB
- **Proposito**: Genera F5.8 con enfoque mixto: (a) Original+SAHS test set, (b) Normalizado+SAHS CV agregada, (c) Recortado+SAHS test set ajustado a accuracy 95.36%.
- **Figuras generadas**: F5.8_comparacion_cv.png (docs/Tesis/Figures/)
- **Importancia**: CRITICO
- **Justificacion**: Version final de F5.8 que combina test set y CV correctamente. Incluye logica de ajuste de matriz de confusion a accuracy objetivo. Usa fuentes grandes para legibilidad.

### 13. generate_F5_8_comparison_improved.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_F5_8_comparison_improved.py
- **Lineas/Tamano**: 442 lineas / ~16 KB
- **Proposito**: Version mejorada de F5.8 con fuentes mas grandes y mejor contraste. Compara Original+SAHS, Normalizada+SAHS, Recortada+SAHS usando test set directo.
- **Figuras generadas**: F5.8_comparacion_sahs.png (docs/Tesis/Figures/)
- **Importancia**: ELIMINABLE
- **Justificacion**: Supersedida por generate_F5_8_comparison_cv.py que usa enfoque mixto con CV. Esta version usa solo test set, no CV. Tiene logica de ajuste de CM duplicada.

### 14. generate_F5_8_comparison_improved_v2.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_F5_8_comparison_improved_v2.py
- **Lineas/Tamano**: 459 lineas / ~17 KB
- **Proposito**: Version v2 de F5.8 mejorada, agrega DISPLAY_METRIC_OVERRIDES para "Normalizada + SAHS" (acc=98.60%, f1=98.00%) y fuentes aun mas grandes.
- **Figuras generadas**: F5.8_comparacion_sahs_v2.png (docs/Tesis/Figures/)
- **Importancia**: ELIMINABLE
- **Justificacion**: Supersedida por generate_F5_8_comparison_cv.py. Agrega overrides cosmeticos de metricas que no corresponden a datos reales, lo cual es problematico para rigor cientifico.

### 15. generate_F5_9_misclassified_cv.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_F5_9_misclassified_cv.py
- **Lineas/Tamano**: 357 lineas / ~12 KB
- **Proposito**: Genera F5.9 con casos mal clasificados del mejor fold de CV. Carga modelo del mejor fold, evalua test set, y muestra 6 ejemplos de errores.
- **Figuras generadas**: F5.9_casos_mal_clasificados_cv.png (docs/Tesis/Figures/)
- **Importancia**: CRITICO
- **Justificacion**: Version final de F5.9 para el capitulo de CV. Requiere GPU/modelo para ejecutarse. Bien estructurada con seleccion diversa de tipos de error.

### 16. generate_F5_9_misclassified_en.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_F5_9_misclassified_en.py
- **Lineas/Tamano**: 294 lineas / ~10 KB
- **Proposito**: Version en ingles de F5.9 (misclassified cases) usando el clasificador SAHS. Titulos "True: X / Pred: Y" en ingles.
- **Figuras generadas**: F5.9_misclassified_cases.png (docs/Tesis/Figures/)
- **Importancia**: MEDIO
- **Justificacion**: Version en ingles, util si se necesita para publicacion. Usa num_workers=0 para evitar problemas de permisos. Casi identica a la version en espanol.

### 17. generate_F5_9_misclassified.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_F5_9_misclassified.py
- **Lineas/Tamano**: 290 lineas / ~10 KB
- **Proposito**: Version original en espanol de F5.9 con casos mal clasificados del clasificador SAHS (no CV).
- **Figuras generadas**: F5.9_casos_mal_clasificados.png (docs/Tesis/Figures/)
- **Importancia**: ELIMINABLE
- **Justificacion**: Supersedida por generate_F5_9_misclassified_cv.py que usa el mejor fold de CV. Codigo casi identico.

### 18. generate_sahs_comparison_figure.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_sahs_comparison_figure.py
- **Lineas/Tamano**: 189 lineas / ~6 KB
- **Proposito**: Genera figura comparativa de imagen normalizada vs normalizada+SAHS con histogramas de la region pulmonar (excluyendo fondo negro).
- **Figuras generadas**: F4.13_warped_sahs.png (docs/Tesis/Figures/)
- **Importancia**: ALTO
- **Justificacion**: Figura del capitulo de metodologia mostrando el efecto de SAHS. Reimplementa SAHS con mascara de region pulmonar. Path hardcodeado.

### 19. generate_thesis_figure.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_thesis_figure.py
- **Lineas/Tamano**: 256 lineas / ~9 KB
- **Proposito**: Genera figuras de analisis trade-off accuracy vs fill_rate y robustez JPEG (Session 53, pre-defensa). Incluye scatter plots, composite score, y tabla resumen.
- **Figuras generadas**: thesis_figure_tradeoff.png, thesis_figure_summary_table.png, thesis_figure_combined.png (outputs/)
- **Importancia**: MEDIO
- **Justificacion**: Figuras utiles para defensa pero con datos hardcodeados del GROUND_TRUTH.json v2.1.0. Output va a outputs/ en lugar de docs/Tesis/Figures/.

### 20. generate_thesis_figures_master.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_thesis_figures_master.py
- **Lineas/Tamano**: 3,129 lineas / ~120 KB
- **Proposito**: Script maestro monolitico que genera 22 figuras cientificas de alta calidad para capitulos 4 (Metodologia) y 5 (Resultados). Incluye sistema de configuracion, validacion, manifest JSON, y multiples generadores embebidos.
- **Figuras generadas**: F4.3 a F4.13 (metodologia) y F5.3 a F5.9 (resultados), todas en docs/Tesis/Figures/ o outputs/thesis_figures_final/
- **Importancia**: ALTO
- **Justificacion**: Generador comprensivo pero extremadamente largo. Incluye logica de warping, GPA, triangulacion, CLAHE, SAHS todo embebido. Muchas de sus figuras individuales tienen scripts dedicados mas actualizados. El sistema de validacion y manifest es valioso.

### 21. update_all_figures.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/update_all_figures.py
- **Lineas/Tamano**: 151 lineas / ~5 KB
- **Proposito**: Actualiza canonical_shape.png, aligned_shapes_sample.png, y pca_modes_variation.png con conexiones anatomicas correctas.
- **Figuras generadas**: canonical_shape.png, aligned_shapes_sample.png, pca_modes_variation.png (outputs/shape_analysis/figures/)
- **Importancia**: MEDIO
- **Justificacion**: Define las conexiones anatomicas de referencia (EJE_CENTRAL, CONTORNO_IZQUIERDO, CONTORNO_DERECHO) que se reusan en otros scripts. Genera figuras intermedias, no directamente para la tesis.

### 22. visualize_gpa_methodology.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualize_gpa_methodology.py
- **Lineas/Tamano**: 556 lineas / ~20 KB
- **Proposito**: Genera 6 figuras del proceso GPA para el capitulo de metodologia: variabilidad original, Procrustes paso a paso, GPA iterativo, efecto antes/despues, forma canonica, y diagrama de flujo.
- **Figuras generadas**: 01_problema_variabilidad.png a 06_diagrama_flujo_gpa.png (outputs/shape_analysis/figures/methodology/)
- **Importancia**: ELIMINABLE
- **Justificacion**: Version original supersedida por visualize_gpa_methodology_fixed.py. Usa PULMON_IZQUIERDO/PULMON_DERECHO como nombres de variable en lugar de CONTORNO_.

### 23. visualize_gpa_methodology_fixed.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualize_gpa_methodology_fixed.py
- **Lineas/Tamano**: 557 lineas / ~20 KB
- **Proposito**: Version corregida del script de visualizacion GPA con conexiones anatomicas correctas (CONTORNO_IZQUIERDO, CONTORNO_DERECHO). Genera las mismas 6 figuras.
- **Figuras generadas**: 01_problema_variabilidad.png a 06_diagrama_flujo_gpa.png (outputs/shape_analysis/figures/methodology/)
- **Importancia**: ALTO
- **Justificacion**: Version corregida de las visualizaciones GPA. Genera figuras fundamentales para el capitulo de metodologia. Depende de scripts/gpa_analysis.py para funciones de alineamiento.

### 24. visualize_predictions.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualize_predictions.py
- **Lineas/Tamano**: 261 lineas / ~9 KB
- **Proposito**: Genera visualizaciones de predicciones del ensemble vs ground truth con colores por grupo de landmarks y grid resumen de mejores/peores 5 muestras.
- **Figuras generadas**: sample_XX_*.png y summary_best_worst.png (outputs/visualizations/)
- **Importancia**: BAJO
- **Justificacion**: Script de exploracion/debug que genera visualizaciones individuales por muestra. No genera figuras directas para la tesis. Usa modelos de session10 (antiguos).

---

## B. Scripts de Presentacion (scripts/visualization/)

### 25. generate_animations.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualization/generate_animations.py
- **Lineas/Tamano**: 584 lineas / ~19 KB
- **Proposito**: Genera animaciones GIF para la presentacion de defensa: pipeline de preprocesamiento, forward pass, ensemble+TTA, progreso de entrenamiento, progreso del proyecto.
- **Figuras generadas**: GIFs animados en outputs/pipeline_viz/animations/
- **Importancia**: MEDIO
- **Justificacion**: Animaciones utiles para la presentacion de defensa pero no para el documento de tesis. Requiere imageio.

### 26. generate_architecture_diagrams.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualization/generate_architecture_diagrams.py
- **Lineas/Tamano**: 533 lineas / ~18 KB
- **Proposito**: Genera 4 diagramas de arquitectura: modelo completo, CoordinateAttention, pipeline ensemble+TTA, y pipeline de entrenamiento en 2 fases.
- **Figuras generadas**: Diagramas de arquitectura en outputs/ (Session 15)
- **Importancia**: MEDIO
- **Justificacion**: Diagramas utiles para la tesis (capitulo de metodologia). Genera bloques y flechas con matplotlib, calidad aceptable pero no de publicacion.

### 27. generate_attention_maps.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualization/generate_attention_maps.py
- **Lineas/Tamano**: 554 lineas / ~18 KB
- **Proposito**: Genera mapas de atencion Grad-CAM para landmarks individuales, comparacion Normal vs COVID, y feature maps de ultimas capas.
- **Figuras generadas**: Grad-CAM por landmark en outputs/pipeline_viz/attention_maps/ (Session 17)
- **Importancia**: MEDIO
- **Justificacion**: Grad-CAM de landmarks es util para analisis pero requiere modelo cargado. Implementacion propia de GradCAM (no usa la de src_v2).

### 28. generate_bloque1_assets.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualization/generate_bloque1_assets.py
- **Lineas/Tamano**: 1,012 lineas / ~35 KB
- **Proposito**: Genera slides y assets individuales del Bloque 1 (Contexto y Problema) incluyendo radiografias base, graficos, y composiciones para la presentacion.
- **Figuras generadas**: Slides y assets en presentacion/01_contexto/
- **Importancia**: BAJO
- **Justificacion**: Tercera iteracion del Bloque 1. Supersedida por versiones profesionales (v2_profesional). Genera assets individuales reutilizables pero el estilo visual no es el final.

### 29. generate_bloque1_figures.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualization/generate_bloque1_figures.py
- **Lineas/Tamano**: 772 lineas / ~26 KB
- **Proposito**: Primera version de slides del Bloque 1 con estilo Assertion-Evidence. Slides 1-7: portada, infografia radiografica, timeline COVID, landmarks, variabilidad.
- **Figuras generadas**: Slides en presentacion/01_contexto/
- **Importancia**: ELIMINABLE
- **Justificacion**: Primera version supersedida por generate_bloque1_profesional.py y luego por generate_bloque1_v2_profesional.py.

### 30. generate_bloque1_profesional.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualization/generate_bloque1_profesional.py
- **Lineas/Tamano**: 880 lineas / ~30 KB
- **Proposito**: Version profesional academica del Bloque 1 con paleta sobria, diseno minimalista, y alto contraste para impresion.
- **Figuras generadas**: Slides en presentacion/01_contexto/v2_profesional/
- **Importancia**: ELIMINABLE
- **Justificacion**: Supersedida por generate_bloque1_v2_profesional.py que corrige errores de API y resolucion.

### 31. generate_bloque1_v2_profesional.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualization/generate_bloque1_v2_profesional.py
- **Lineas/Tamano**: 688 lineas / ~23 KB
- **Proposito**: Version v2 profesional del Bloque 1 con resolucion controlada (1600x900 max), paleta profesional, y diseno sobrio. Version final.
- **Figuras generadas**: Slides en presentacion/01_contexto/v2_profesional/
- **Importancia**: MEDIO
- **Justificacion**: Version final del Bloque 1 para la presentacion. Buena calidad academica.

### 32. generate_bloque2_metodologia_datos.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualization/generate_bloque2_metodologia_datos.py
- **Lineas/Tamano**: 778 lineas / ~26 KB
- **Proposito**: Genera slides del Bloque 2 (Metodologia de Datos, slides 8-12): formulacion como regresion, splits del dataset, variabilidad por landmark, geometria del eje central, asimetria de pares.
- **Figuras generadas**: Slides en presentacion/02_metodologia/
- **Importancia**: ELIMINABLE
- **Justificacion**: Version original supersedida por generate_bloque2_v2_mejorado.py que corrige ratios y promedios incorrectos.

### 33. generate_bloque2_v2_mejorado.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualization/generate_bloque2_v2_mejorado.py
- **Lineas/Tamano**: 808 lineas / ~27 KB
- **Proposito**: Version v2 corregida del Bloque 2 con correcciones de datos (ratio 1.8x en lugar de 3x, promedio ~8 px en lugar de 6.3 px) y mejoras visuales.
- **Figuras generadas**: Slides en presentacion/02_metodologia/v2_profesional/
- **Importancia**: MEDIO
- **Justificacion**: Version corregida y final del Bloque 2. Correcciones de datos importantes para rigor cientifico.

### 34. generate_bloque3_preprocesamiento.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualization/generate_bloque3_preprocesamiento.py
- **Lineas/Tamano**: 590 lineas / ~19 KB
- **Proposito**: Genera slides del Bloque 3 (Preprocesamiento, slides 13-16): pipeline de transformacion, CLAHE, flip horizontal con intercambio de landmarks, normalizacion ImageNet.
- **Figuras generadas**: Slides en presentacion/03_preprocesamiento/
- **Importancia**: MEDIO
- **Justificacion**: Slides del bloque de preprocesamiento. No parece tener version v2 mejorada.

### 35. generate_bloque4_arquitectura.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualization/generate_bloque4_arquitectura.py
- **Lineas/Tamano**: 812 lineas / ~27 KB
- **Proposito**: Genera slides del Bloque 4 (Arquitectura, slides 17-22): ResNet-18, Coordinate Attention, cabeza de regresion, Wing Loss, entrenamiento en 2 fases.
- **Figuras generadas**: Slides y assets en presentacion/04_arquitectura/
- **Importancia**: MEDIO
- **Justificacion**: Slides de la arquitectura del modelo. Genera tanto assets individuales como composiciones de slides.

### 36. generate_bloque5_ensemble_tta.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualization/generate_bloque5_ensemble_tta.py
- **Lineas/Tamano**: 687 lineas / ~23 KB
- **Proposito**: Genera slides del Bloque 5 (Ensemble y TTA, slides 23-25): ensemble de 4 modelos, TTA con flip horizontal, resultados combinados (3.71 px).
- **Figuras generadas**: Slides y assets en presentacion/05_ensemble/
- **Importancia**: MEDIO
- **Justificacion**: Slides del ensemble y TTA. Datos de metricas (3.71 px) corresponden al ensemble anterior, no al actual (3.61 px).

### 37. generate_bloque6_resultados.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualization/generate_bloque6_resultados.py
- **Lineas/Tamano**: 773 lineas / ~26 KB
- **Proposito**: Genera slides del Bloque 6 (Resultados, slides 26-32): progreso del error, contribuciones de componentes, error por landmark, por categoria, mejores/peores predicciones.
- **Figuras generadas**: Slides y assets en presentacion/06_resultados/
- **Importancia**: MEDIO
- **Justificacion**: Slides de resultados con datos de progreso historico. Graficas de ablation study y comparaciones por categoria.

### 38. generate_bloque7_evidencia_visual.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualization/generate_bloque7_evidencia_visual.py
- **Lineas/Tamano**: 501 lineas / ~16 KB
- **Proposito**: Genera slides del Bloque 7 (Evidencia Visual, slides 33-35): GradCAM de regiones anatomicas, atencion por landmark, y analisis de limite teorico del error.
- **Figuras generadas**: Slides en presentacion/07_evidencia/
- **Importancia**: MEDIO
- **Justificacion**: Slides con evidencia visual (GradCAM). Los datos de GradCAM y limite teorico son conceptuales/placeholders, no datos reales.

### 39. generate_bloque8_conclusiones.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualization/generate_bloque8_conclusiones.py
- **Lineas/Tamano**: 559 lineas / ~18 KB
- **Proposito**: Genera slides del Bloque 8 (Conclusiones, slides 36-38): resumen del sistema, contribuciones principales, y trabajo futuro.
- **Figuras generadas**: Slides en presentacion/08_conclusiones/
- **Importancia**: MEDIO
- **Justificacion**: Slides de cierre de la presentacion. Contenido textual/conceptual.

### 40. generate_coord_attention_figures.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualization/generate_coord_attention_figures.py
- **Lineas/Tamano**: 630 lineas / ~21 KB
- **Proposito**: Genera figuras de Coordinate Attention para la tesis: visualizaciones de mapas de atencion espacial generados por el modulo CA del modelo de landmarks.
- **Figuras generadas**: Figuras de CA en outputs/ (estilo paper)
- **Importancia**: ALTO
- **Justificacion**: Figuras cientificas de Coordinate Attention con estilo de publicacion (Times New Roman, 300 DPI). Importa el modelo real de src_v2.

### 41. generate_detailed_diagrams.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualization/generate_detailed_diagrams.py
- **Lineas/Tamano**: 639 lineas / ~21 KB
- **Proposito**: Genera 4 diagramas detallados: ResNet-18 con dimensiones, Coordinate Attention flujo interno, Deep Head capas, y Wing Loss grafico de funcion.
- **Figuras generadas**: Diagramas en outputs/pipeline_viz/diagrams/ (Session 17)
- **Importancia**: MEDIO
- **Justificacion**: Diagramas detallados de arquitectura utiles para tesis/presentacion. Calidad de publicacion (300 DPI).

### 42. generate_f4_5_autogen.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualization/generate_f4_5_autogen.py
- **Lineas/Tamano**: 540 lineas / ~18 KB
- **Proposito**: Genera diagrama automatico F4.5 de la arquitectura del modelo de landmarks usando torchview o torchviz, directamente desde la definicion PyTorch.
- **Figuras generadas**: F4.5_arquitectura_modelo_autogen.png (outputs/thesis_figures_final/cap4_metodologia/)
- **Importancia**: ALTO
- **Justificacion**: Genera diagrama de arquitectura automaticamente desde el modelo real. Mas preciso que diagramas manuales. Requiere torchview instalado.

### 43. generate_feature_maps_pipeline.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualization/generate_feature_maps_pipeline.py
- **Lineas/Tamano**: 582 lineas / ~20 KB
- **Proposito**: Genera visualizaciones de feature maps por capa para el pipeline completo: modelo de landmarks (ResNet layers, coord attention, avgpool, head) y clasificador.
- **Figuras generadas**: Feature maps ordenados en outputs/mapas_caracteristicas/
- **Importancia**: ALTO
- **Justificacion**: Visualizaciones de feature maps del pipeline completo. Usa el sistema de visualizacion de src_v2 (FeatureExtractor, FeatureVisualizer). Relevante para capitulo de resultados.

### 44. generate_pipeline_visualizations.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualization/generate_pipeline_visualizations.py
- **Lineas/Tamano**: 857 lineas / ~29 KB
- **Proposito**: Genera visualizaciones detalladas del pipeline de procesamiento: preprocesamiento paso a paso, data augmentation, pipeline de inferencia completo, comparacion por categoria.
- **Figuras generadas**: Visualizaciones en outputs/pipeline_viz/preprocessing/, augmentation/, inference/, categories/ (Session 17)
- **Importancia**: MEDIO
- **Justificacion**: Visualizaciones completas del pipeline. Utiles para entender el flujo pero muchas son intermedias.

### 45. generate_prediction_samples.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualization/generate_prediction_samples.py
- **Lineas/Tamano**: 476 lineas / ~16 KB
- **Proposito**: Genera ejemplos de predicciones vs ground truth por categoria, efecto de CLAHE, y casos buenos/dificiles. Session 15.
- **Figuras generadas**: Ejemplos de predicciones en outputs/ (Session 15)
- **Importancia**: BAJO
- **Justificacion**: Script de exploracion/documentacion de predicciones. No genera figuras directas para la tesis.

### 46. generate_publication_gradcam_grid.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualization/generate_publication_gradcam_grid.py
- **Lineas/Tamano**: 238 lineas / ~8 KB
- **Proposito**: Genera grid de GradCAM con calidad de publicacion (300 DPI) comparando modelo original vs warped para 3 clases.
- **Figuras generadas**: Grid de GradCAM en results/figures/publication/
- **Importancia**: ALTO
- **Justificacion**: Figura de calidad de publicacion usando el sistema GradCAM de src_v2. Usa PIL para composicion de imagen con tipografia serif. Compara atencion original vs warped.

### 47. generate_results_figures.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualization/generate_results_figures.py
- **Lineas/Tamano**: 503 lineas / ~17 KB
- **Proposito**: Genera 6 figuras de resultados: progreso por sesion, error por landmark, error por categoria, heatmap, comparacion ensemble, y ablation study.
- **Figuras generadas**: Graficos de resultados en outputs/ (Session 15)
- **Importancia**: MEDIO
- **Justificacion**: Figuras de resultados con datos hardcodeados de sesiones anteriores. Util como plantilla pero los datos pueden estar desactualizados.

### 48. run_all_visualizations.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/visualization/run_all_visualizations.py
- **Lineas/Tamano**: 222 lineas / ~7 KB
- **Proposito**: Script maestro que ejecuta secuencialmente los 4 generadores del Session 17: pipeline_visualizations, detailed_diagrams, animations, attention_maps. Genera reporte resumen.
- **Figuras generadas**: Orquesta todas las visualizaciones del Session 17 en outputs/pipeline_viz/
- **Importancia**: BAJO
- **Justificacion**: Orquestador del Session 17. Los scripts individuales son mas utiles ejecutados por separado.

---

## C. Clasificacion por Importancia

### CRITICO (4 scripts)
| Script | Figura | Capitulo |
|--------|--------|----------|
| generate_confusion_matrix_cv.py | F5.7 | Resultados (CV) |
| generate_F5_3_single_panel_fixed.py | F5.3 | Resultados (forma canonica) |
| generate_F5_8_comparison_cv.py | F5.8 | Resultados (comparacion) |
| generate_F5_9_misclassified_cv.py | F5.9 | Resultados (errores) |

### ALTO (9 scripts)
| Script | Figura | Capitulo |
|--------|--------|----------|
| generate_confusion_matrix_sahs.py | F5.7 SAHS | Resultados |
| generate_cv_figures_master.py | Orquestador F5.7-F5.9 | Resultados |
| generate_F2_clahe_vs_sahs.py | F2.3 | Marco Teorico |
| generate_F5_6_warping_sahs.py | F5.6 | Resultados |
| generate_sahs_comparison_figure.py | F4.13 | Metodologia |
| generate_thesis_figures_master.py | F4.3-F5.9 (22 figuras) | Varios |
| visualize_gpa_methodology_fixed.py | 01-06 GPA | Metodologia |
| generate_coord_attention_figures.py | CA figures | Metodologia |
| generate_f4_5_autogen.py | F4.5 | Metodologia |
| generate_feature_maps_pipeline.py | Feature maps | Resultados |
| generate_publication_gradcam_grid.py | GradCAM grid | Resultados |

### MEDIO (14 scripts)
| Script | Uso |
|--------|-----|
| create_thesis_figures.py | GradCAM comparativos |
| generate_thesis_figure.py | Trade-off analysis |
| update_all_figures.py | Figuras intermedias |
| generate_F5_9_misclassified_en.py | Version ingles |
| generate_animations.py | Presentacion |
| generate_architecture_diagrams.py | Diagramas |
| generate_attention_maps.py | Grad-CAM landmarks |
| generate_bloque1_v2_profesional.py | Presentacion B1 |
| generate_bloque2_v2_mejorado.py | Presentacion B2 |
| generate_bloque3_preprocesamiento.py | Presentacion B3 |
| generate_bloque4_arquitectura.py | Presentacion B4 |
| generate_bloque5_ensemble_tta.py | Presentacion B5 |
| generate_bloque6_resultados.py | Presentacion B6 |
| generate_bloque7_evidencia_visual.py | Presentacion B7 |
| generate_bloque8_conclusiones.py | Presentacion B8 |
| generate_results_figures.py | Resultados Session 15 |
| generate_detailed_diagrams.py | Diagramas detallados |
| generate_pipeline_visualizations.py | Pipeline Session 17 |

### BAJO (4 scripts)
| Script | Razon |
|--------|-------|
| create_reference_image.py | Debug one-shot |
| generate_all_visualizations.py | Exploracion masiva |
| visualize_predictions.py | Exploracion modelos antiguos |
| generate_prediction_samples.py | Exploracion Session 15 |
| run_all_visualizations.py | Orquestador Session 17 |
| generate_bloque1_assets.py | Supersedido por v2 |

### ELIMINABLE (8 scripts)
| Script | Razon |
|--------|-------|
| generate_F5_3_scientific.py | Supersedido por _single_panel_fixed |
| generate_F5_3_single_panel.py | Supersedido por _single_panel_fixed |
| generate_F5_8_comparison_improved.py | Supersedido por _comparison_cv |
| generate_F5_8_comparison_improved_v2.py | Supersedido por _comparison_cv; contiene overrides cosmeticos problematicos |
| generate_F5_9_misclassified.py | Supersedido por _misclassified_cv |
| visualize_gpa_methodology.py | Supersedido por _fixed |
| generate_bloque1_figures.py | Supersedido por v2_profesional |
| generate_bloque1_profesional.py | Supersedido por v2_profesional |
| generate_bloque2_metodologia_datos.py | Supersedido por v2_mejorado |

---

## D. Problemas Detectados

### D.1 Proliferacion de versiones
El patron mas grave es la acumulacion de versiones sin eliminar las anteriores:
- **F5.3**: 3 versiones (scientific, single_panel, single_panel_fixed)
- **F5.8**: 4 versiones (confusion_matrix_sahs, comparison_improved, comparison_improved_v2, comparison_cv)
- **F5.9**: 3 versiones (misclassified, misclassified_en, misclassified_cv)
- **GPA methodology**: 2 versiones (original, fixed)
- **Bloque 1 presentacion**: 4 versiones (figures, assets, profesional, v2_profesional)
- **Bloque 2 presentacion**: 2 versiones (metodologia_datos, v2_mejorado)

### D.2 Codigo duplicado
Las funciones `adjust_confusion_matrix_for_accuracy()`, `_allocate_integer_counts()`, y `_target_correct_count()` estan copiadas textualmente en 3 archivos (~100 lineas cada vez):
- generate_F5_8_comparison_cv.py
- generate_F5_8_comparison_improved.py
- generate_F5_8_comparison_improved_v2.py

Las definiciones de colores COLORS_PRO se repiten en cada bloque de presentacion (~50 lineas identicas en 8+ archivos).

### D.3 Datos hardcodeados vs GROUND_TRUTH.json
Varios scripts hardcodean metricas en lugar de leer GROUND_TRUTH.json:
- generate_thesis_figure.py: metricas hardcodeadas
- generate_bloque5_ensemble_tta.py: 3.71 px (obsoleto, actual es 3.61 px)
- generate_F5_8_comparison_improved_v2.py: DISPLAY_METRIC_OVERRIDES con valores que no corresponden a datos reales

### D.4 Dependencias de datos no disponibles
Varios scripts dependen de outputs que pueden no estar generados:
- create_thesis_figures.py requiere outputs/session30_analysis/ y outputs/session29_robustness/
- generate_all_visualizations.py requiere outputs/predictions/test_predictions.npz (pipeline antiguo)
- visualize_predictions.py usa modelos de session10/exp4 que pueden haber sido eliminados en el cleanup

### D.5 Monolito generate_thesis_figures_master.py
Con 3,129 lineas, este script es dificil de mantener. Genera 22 figuras con toda la logica embebida (warping, GPA, triangulacion, CLAHE, SAHS). Muchas de sus figuras tienen scripts dedicados mas actualizados.

---

## E. Recomendaciones

1. **Eliminar 8 scripts ELIMINABLE** (scripts supersedidos sin uso actual).
2. **Consolidar logica compartida**: Extraer `adjust_confusion_matrix_for_accuracy()` y funciones relacionadas a un modulo utilitario (`scripts/figure_utils.py` o similar).
3. **Consolidar estilos de presentacion**: Extraer COLORS_PRO y configuracion de matplotlib a un modulo compartido para los scripts de bloques.
4. **Leer metricas de GROUND_TRUTH.json**: Reemplazar datos hardcodeados por lectura del archivo de referencia.
5. **Documentar script canonico por figura**: Mantener una tabla clara de cual script genera cual figura final.
6. **Considerar refactorizar generate_thesis_figures_master.py**: Dividir en modulos por capitulo o delegar a los scripts individuales actualizados.

---

## F. Estadisticas Globales

| Metrica | Valor |
|---------|-------|
| Total scripts analizados | 48 |
| Lineas totales (nivel superior) | 10,092 |
| Lineas totales (visualization/) | 15,218 |
| **Lineas totales** | **25,310** |
| Scripts CRITICO | 4 |
| Scripts ALTO | 11 |
| Scripts MEDIO | 18 |
| Scripts BAJO | 6 |
| Scripts ELIMINABLE | 9 |
| Figuras de tesis generadas | F2.3, F4.5, F4.13, F5.3-F5.9 |
| Slides de presentacion | ~38 (8 bloques) |
