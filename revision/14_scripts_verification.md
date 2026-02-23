# 14. Verification & Dataset Generation Scripts

Analisis de los scripts de verificacion y generacion de datasets auxiliares.

**Archivos analizados**: 19

---

## A. Scripts de Verificacion (11 archivos)

### verify_canonical_delaunay.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/verify_canonical_delaunay.py
- **Lineas/Tamano**: 152 lineas / ~5 KB
- **Proposito**: Verifica que los triangulos de Delaunay sean validos sobre la forma canonica GPA y los compara con la triangulacion original calculada sobre el promedio simple.
- **Contenido clave**:
  - Carga forma canonica GPA y triangulacion Delaunay existente
  - Recalcula Delaunay sobre la forma canonica y compara con la original
  - Funcion `normalize_triangles()` para comparacion de conjuntos de triangulos
  - Calcula distancia entre forma canonica GPA y promedio simple por landmark
  - Genera visualizacion comparativa de 3 paneles (promedio simple, GPA, superposicion)
  - Guarda triangulacion canonica actualizada en JSON
- **Importancia**: MEDIO
- **Justificacion**: Verificacion puntual que valida la coherencia entre GPA y Delaunay. Util durante desarrollo de sesion 19, pero no es necesario re-ejecutarlo una vez validado. Hardcodea paths a `outputs/predictions/delaunay_triangles.json` que puede no existir en configuraciones actuales.

---

### verify_comparison_alignment.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/verify_comparison_alignment.py
- **Lineas/Tamano**: 187 lineas / ~6 KB
- **Proposito**: Verifica que el dataset de comparaciones de landmarks este perfectamente alineado con el dataset warped de clasificacion (mismos splits, subset correcto, misma distribucion por categoria).
- **Contenido clave**:
  - Verifica que las imagenes de comparacion son un subconjunto del dataset warped
  - Chequea distribucion por categoria dentro de cada split (train/val/test)
  - Valida convenciones de nombrado de archivos (`_comparison.png` vs `_warped.png`)
  - Verifica metadatos JSON (schema_version, alignment flags)
  - Retorna exit code 0/1 para integracion en scripts
- **Importancia**: MEDIO
- **Justificacion**: Util para validar la generacion del dataset de visualizacion de landmarks. Solo aplica cuando se usa el dataset de comparaciones alineado con el clasificador.

---

### verify_data_leakage.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/verify_data_leakage.py
- **Lineas/Tamano**: 131 lineas / ~4 KB
- **Proposito**: Verifica que no exista data leakage entre los splits train/val/test, evaluando un modelo en cada split para detectar discrepancias sospechosas.
- **Contenido clave**:
  - Usa `get_dataframe_splits()` con `random_state=42` para verificar splits
  - Verifica overlap entre conjuntos de imagenes (train vs val, train vs test, val vs test)
  - Evalua modelo seed=42 en los tres splits sin TTA
  - Muestra distribucion por categoria por split
  - Calcula estadisticas de posiciones de landmarks por split
  - **Problema**: Hardcodea path a `checkpoints/session10/exp4_epochs100/final_model.pt` que fue eliminado en la limpieza de checkpoints (ver CHECKPOINTS_CLEANUP_REPORT.md)
- **Importancia**: CRITICO
- **Justificacion**: Verificacion fundamental para la validez cientifica del proyecto (defensa de tesis). Sin embargo, el checkpoint referenciado ya no existe, lo que lo hace no-ejecutable en su estado actual. Necesita actualizacion del path del modelo.

---

### verify_dataset_splits.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/verify_dataset_splits.py
- **Lineas/Tamano**: 594 lineas / ~18 KB
- **Proposito**: Verificacion exhaustiva de integridad del dataset extraido para la GUI, comprobando que no hay overlap entre splits, que imagenes originales y warped son binariamente identicas al dataset fuente, y que los CSVs coinciden.
- **Contenido clave**:
  - 4 verificaciones independientes: no-overlap, originales identicas (MD5), warped identicas (MD5), CSVs identicos
  - `compute_md5()` para comparacion binaria de archivos
  - Genera reporte detallado en archivo de texto (`VERIFICATION_REPORT.txt`)
  - Acepta argumentos CLI para directorios (extracted, original, warped)
  - Usa `mapeo_category_to_original_dir()` para manejar nombres de directorio con espacios
  - Mensaje final explicito "SEGURO PARA DEFENSA DE TESIS" o "NO USAR ESTE DATASET"
- **Importancia**: CRITICO
- **Justificacion**: Script de verificacion mas completo del proyecto. Esencial para garantizar integridad del dataset antes de la defensa de tesis. Las verificaciones MD5 son la unica forma de confirmar que no hubo preprocesamiento no documentado.

---

### verify_gpa_correctness.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/verify_gpa_correctness.py
- **Lineas/Tamano**: 425 lineas / ~14 KB
- **Proposito**: Verificacion exhaustiva de la implementacion de GPA (Generalized Procrustes Analysis) con datos reales, validando cada paso (centrado, escalado, alineacion) y generando visualizaciones sobre imagenes reales.
- **Contenido clave**:
  - Verifica que datos en `.npz` son reales (rangos, nombres de imagen, categorias)
  - Compara landmarks del NPZ con CSV original escalando de 299px a 224px
  - Verificacion paso a paso: centrado (centroide -> 0), escalado (norma -> 1), distancias Procrustes
  - Visualiza landmarks sobre imagenes reales de cada categoria
  - Verifica estructura anatomica (eje central vertical, conexiones pulmonares)
  - Genera 4 figuras de verificacion
  - **Dependencia**: Importa de `scripts.gpa_analysis` (modulo externo no verificado)
  - **Problema**: Referencia `outputs/predictions/all_landmarks.npz` que puede estar desactualizado
- **Importancia**: ALTO
- **Justificacion**: Validacion crucial de la pieza central del pipeline (GPA). Las visualizaciones generadas son utiles para la documentacion de tesis. Sirve como evidencia de correctitud del alineamiento.

---

### verify_gui_setup.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/verify_gui_setup.py
- **Lineas/Tamano**: 311 lineas / ~9 KB
- **Proposito**: Verifica que todos los componentes necesarios para la GUI (Gradio) esten disponibles: version de Python, dependencias, modulos GUI, archivos de modelos, imagenes de ejemplo, dispositivo GPU/CPU, funcion CLAHE, y creacion de interfaz.
- **Contenido clave**:
  - 8 verificaciones: Python version, dependencias (torch, gradio, numpy, cv2, matplotlib, pandas, PIL), modulos GUI (config, gradcam_utils, visualizer, model_manager, inference_pipeline, app), archivos de modelos, imagenes de ejemplo, dispositivo, CLAHE, interfaz Gradio
  - Importa `src_v2.gui.config` para verificar paths de modelos
  - Prueba funcional de CLAHE con imagen random
  - Prueba creacion de demo Gradio
  - Salida con recomendaciones especificas si algo falla
- **Importancia**: ALTO
- **Justificacion**: Script esencial para diagnosticar problemas en despliegue de la GUI. Particularmente util en entornos nuevos o distribucion Windows portable.

---

### verify_gui.sh
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/verify_gui.sh
- **Lineas/Tamano**: 91 lineas / ~3 KB
- **Proposito**: Script bash que ejecuta de forma estructurada todos los tests de la GUI organizados en 5 niveles: dependencias, tests unitarios, integracion, sistema, y manejo de errores.
- **Contenido clave**:
  - Nivel 1: Dependencias y rutas (9 tests con pytest)
  - Nivel 2: Tests unitarios (8 tests: ModelManager, predict, warp, validate, render, GradCAM)
  - Nivel 3: Tests de integracion (3 tests: end-to-end, TTA, CLAHE)
  - Nivel 4: Tests de sistema (2 tests: create demo, export PDF)
  - Nivel 5: Error handling (2 tests: invalid format, corrupted image)
  - Contadores PASSED/FAILED con resumen coloreado
  - Ejecuta tests individuales via `pytest tests/gui/test_*.py::test_*`
- **Importancia**: ALTO
- **Justificacion**: Orchestrador de tests de la GUI con cobertura estructurada. Complementa `verify_gui_setup.py` con pruebas funcionales reales via pytest.

---

### verify_individual_models.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/verify_individual_models.py
- **Lineas/Tamano**: 204 lineas / ~7 KB
- **Proposito**: Verifica que los modelos individuales del ensemble producen errores de landmark consistentes con los valores esperados, reportando metricas por categoria.
- **Contenido clave**:
  - Evalua 3 modelos con TTA: seed=42 (4.10 px esperado), seed=123 (4.05 px), seed=456 (4.04 px)
  - Implementa `predict_with_tta()` con flip horizontal y swap de pares simetricos
  - Metricas: mean, std, median, per-category error
  - Umbral de aceptacion: diferencia < 0.5 px = OK
  - Nota en docstring: "Los valores de referencia validados estan en GROUND_TRUTH.json"
  - **Problema**: Referencia `checkpoints/session10/exp4_epochs100/final_model.pt` que fue eliminado en cleanup
- **Importancia**: ALTO
- **Justificacion**: Verificacion de reproducibilidad de los modelos individuales. Importante para confirmar que los checkpoints producen resultados consistentes. Sin embargo, uno de los paths referenciados no existe post-cleanup.

---

### verify_landmark_viz_dataset.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/verify_landmark_viz_dataset.py
- **Lineas/Tamano**: 96 lineas / ~3 KB
- **Proposito**: Verifica que el dataset de visualizacion de landmarks tiene exactamente las mismas imagenes que el dataset warped en cada split y categoria.
- **Contenido clave**:
  - Compara nombres de archivos entre directorios warped y viz (`_warped` vs `_landmarks_viz`)
  - Verifica por split (train/val/test) y por categoria (COVID/Normal/Viral_Pneumonia)
  - Reporta imagenes faltantes o extras en el dataset de visualizacion
  - Acepta paths como argumentos de linea de comandos
  - **Documentado en CLAUDE.md**: Si, bajo "Verify alignment with warped dataset"
- **Importancia**: MEDIO
- **Justificacion**: Verificacion rapida y simple para asegurar consistencia entre datasets. Util despues de generar el dataset de visualizacion.

---

### verify_no_tta.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/verify_no_tta.py
- **Lineas/Tamano**: 96 lineas / ~3 KB
- **Proposito**: Compara errores de modelos sin TTA contra valores reportados previamente para entender la discrepancia entre con/sin TTA.
- **Contenido clave**:
  - Evalua 3 modelos sin TTA: seed=42 (6.75 esperado), seed=123 (7.16), seed=456 (7.20)
  - Comparacion con valores reportados anteriormente
  - Umbral de match: diferencia < 0.5 px
  - **Problema**: Mismo problema de paths eliminados que verify_individual_models.py
- **Importancia**: BAJO
- **Justificacion**: Script de investigacion puntual de sesion 11 para entender efecto de TTA. Resultado ya incorporado en documentacion (TTA mejora ~3 px). No necesita re-ejecucion.

---

### verify_val_vs_test.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/verify_val_vs_test.py
- **Lineas/Tamano**: 87 lineas / ~3 KB
- **Proposito**: Compara error de modelos en validation vs test set para investigar si el test set es sistematicamente mas facil.
- **Contenido clave**:
  - Evalua 3 modelos sin TTA en val y test
  - Calcula diferencia val-test por modelo
  - Conclusion: "Todos los modelos tienen mejor resultado en TEST que en VAL"
  - **Problema**: Mismo problema de paths eliminados
- **Importancia**: BAJO
- **Justificacion**: Script de investigacion puntual. La conclusion ya esta documentada. No necesita re-ejecucion regular.

---

## B. Scripts de Generacion de Datasets (7 archivos)

### generate_warped_dataset.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_warped_dataset.py
- **Lineas/Tamano**: 409 lineas / ~13 KB
- **Proposito**: Genera dataset warped completo (957 imagenes con landmarks GT) usando piecewise affine warping solo sobre area pulmonar (18 triangulos Delaunay).
- **Contenido clave**:
  - **Marcado como LEGACY en CLAUDE.md**: "Old session 21 workflow with GT landmarks"
  - Usa landmarks Ground Truth (no predichos) desde `all_landmarks.npz`
  - `use_full_coverage=False` (solo area pulmonar)
  - Importa de `scripts.piecewise_affine_warp` (modulo legacy)
  - Genera CSVs de split con columnas `image_name,category,warped_filename`
  - Calcula fill rate por imagen y guarda estadisticas JSON
  - Estructura de salida: `warped_dataset/{train,val,test}/{COVID,Normal,Viral_Pneumonia}/`
- **Importancia**: ELIMINABLE
- **Justificacion**: Reemplazado por el pipeline actual `python -m src_v2 generate-dataset --config`. Usa landmarks GT en lugar de predichos, lo cual no refleja el pipeline real. Documentado como legacy en CLAUDE.md.

---

### generate_full_warped_dataset.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_full_warped_dataset.py
- **Lineas/Tamano**: 353 lineas / ~12 KB
- **Proposito**: Genera dataset warped para el dataset completo (~15K imagenes) con prediccion de landmarks inline y margin_scale=1.05.
- **Contenido clave**:
  - **Marcado como LEGACY en CLAUDE.md**: "Session 25 inline warping without cache"
  - Predice landmarks para cada imagen individualmente (no usa cache)
  - Importa `EnsemblePredictor` de `scripts.predict` (modulo legacy)
  - Importa funciones de `scripts.piecewise_affine_warp` (modulo legacy)
  - Crea splits propios (75/15/10) con seed=42, no usa `split_seed` del sistema actual
  - Aplica `margin_scale=1.05` y `clip_landmarks_to_image()`
  - Guarda landmarks predichos por split en JSON
  - Tiempo estimado: 2-4 horas con GPU
- **Importancia**: ELIMINABLE
- **Justificacion**: Reemplazado por `predict_landmarks_dataset.py` + `generate-dataset` CLI. No usa cache de predicciones, lo cual lo hace ineficiente y no reproducible con el pipeline actual. Depende de modulos legacy.

---

### generate_cropped_sahs_dataset.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_cropped_sahs_dataset.py
- **Lineas/Tamano**: 242 lineas / ~8 KB
- **Proposito**: Aplica SAHS (Statistical Asymmetrical Histogram Stretching) a un dataset de imagenes cropped, generando un nuevo dataset con contraste mejorado.
- **Contenido clave**:
  - Implementa `enhance_contrast_sahs()` con factores asimetricos (2.5 superior, 2.0 inferior)
  - Procesamiento paralelo con `ProcessPoolExecutor`
  - CLI con `--input-dir`, `--output-dir`, `--workers`
  - Guarda metadatos JSON (parametros SAHS, estadisticas, distribucion)
  - Mantiene estructura de splits y categorias
  - **No documentado en CLAUDE.md** (experimental)
- **Importancia**: BAJO
- **Justificacion**: Script experimental para evaluar SAHS sobre imagenes cropped. SAHS no fue seleccionado como preprocesamiento final del pipeline (se usa CLAHE). Podria moverse a archive.

---

### generate_warped_sahs_dataset.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_warped_sahs_dataset.py
- **Lineas/Tamano**: 267 lineas / ~9 KB
- **Proposito**: Aplica SAHS solo a la region pulmonar de imagenes warped (pixeles > threshold), manteniendo el fondo negro intacto.
- **Contenido clave**:
  - Implementa `enhance_contrast_sahs_masked()` con mascara basada en threshold
  - Diferencia clave vs cropped: aplica SAHS solo a pixeles > threshold (default: 10)
  - Procesamiento paralelo con `ProcessPoolExecutor`
  - CLI con `--input-dir`, `--output-dir`, `--threshold`, `--workers`
  - Copia `dataset_summary_original.json` del directorio fuente
  - **No documentado en CLAUDE.md** (experimental)
- **Importancia**: BAJO
- **Justificacion**: Script experimental para evaluar SAHS sobre imagenes warped. La version masked es mas correcta metodologicamente que la cropped, pero SAHS no fue seleccionado en el pipeline final. Podria moverse a archive.

---

### generate_all_landmarks_npz.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_all_landmarks_npz.py
- **Lineas/Tamano**: 192 lineas / ~6 KB
- **Proposito**: Genera `all_landmarks.npz` desde `coordenadas_maestro.csv` con splits reproducibles, y opcionalmente compara con un NPZ existente para verificar reproducibilidad.
- **Contenido clave**:
  - Lee CSV de coordenadas GT y aplica factor de escala (224/299)
  - Crea splits estratificados con `sklearn.train_test_split` (seed=42)
  - Genera NPZ con arrays: all_landmarks, train/val/test landmarks, image_names, categories
  - Funcion `compare_npz()` para verificar reproducibilidad bit-a-bit
  - Usa constantes del proyecto: `DEFAULT_IMAGE_SIZE`, `ORIGINAL_IMAGE_SIZE`, `NUM_LANDMARKS`
  - CLI bien estructurado con parametros configurables
  - Soporta tolerancia absoluta ajustable (`--atol`)
- **Importancia**: ALTO
- **Justificacion**: Script esencial para regenerar el archivo NPZ de landmarks GT necesario para el warping con Ground Truth. La funcionalidad de comparacion es clave para verificar reproducibilidad. Bien estructurado y usa el sistema de constantes del proyecto.

---

### generate_icon.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/generate_icon.py
- **Lineas/Tamano**: 105 lineas / ~3 KB
- **Proposito**: Genera un icono simple con tema medico (cruz blanca sobre fondo azul) en formato ICO multi-resolucion para el ejecutable Windows de la GUI.
- **Contenido clave**:
  - Genera icono en 6 resoluciones: 16, 32, 48, 64, 128, 256 px
  - Dibuja circulo azul con cruz medica blanca usando PIL/ImageDraw
  - Guarda como `.ico` (multi-resolucion) y `.png` (256px)
  - Directorio de salida: `assets/covid_icon.ico`
  - Manejo de errores con sugerencia de alternativas profesionales
- **Importancia**: BAJO
- **Justificacion**: Utilidad menor para el empaquetado Windows. El icono es generico y no critico para la funcionalidad. Solo necesita ejecutarse una vez.

---

### calculate_pfs_warped.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/calculate_pfs_warped.py
- **Lineas/Tamano**: 426 lineas / ~15 KB
- **Proposito**: Calcula PFS (Pulmonary Focus Score) con mascaras warped correctamente alineadas, corrigiendo un error metodologico previo donde se usaban mascaras no-warped sobre imagenes warped.
- **Contenido clave**:
  - Corrige error metodologico documentado: PFS previo era INVALIDO
  - Usa `warp_mask()` con mismos landmarks que el warping de imagen
  - Integra GradCAM para generar heatmaps de atencion
  - Calcula PFS por clase y por correccion de prediccion
  - `get_canonical_landmarks()` hardcodea landmarks canonicos (posible inconsistencia con GPA)
  - Carga landmarks desde JSON, soporta dos formatos
  - Genera resumen y detalles en JSON
  - CLI completo con --model, --data, --masks, --landmarks, --output
  - **Problema**: `get_canonical_landmarks()` en lineas 96-118 hardcodea landmarks que NO corresponden a la forma canonica GPA real del proyecto. Esto podria producir mascaras warped incorrectas.
- **Importancia**: ALTO
- **Justificacion**: Script importante para la metrica PFS que es parte de la evaluacion del clasificador. Sin embargo, tiene un problema potencial serio: los landmarks canonicos hardcodeados en `get_canonical_landmarks()` no coinciden con la forma canonica GPA real del proyecto. Deberia cargar la forma canonica desde `canonical_shape_gpa.json` en lugar de usar valores hardcodeados.

---

### benchmark_inference.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/scripts/benchmark_inference.py
- **Lineas/Tamano**: 533 lineas / ~18 KB
- **Proposito**: Mide tiempos de inferencia de cada etapa del pipeline completo (preprocesamiento, landmarks, warping, clasificacion) generando un reporte con estadisticas detalladas.
- **Contenido clave**:
  - Clase `Timer` como context manager para mediciones precisas (`time.perf_counter`)
  - Implementa CLAHE y SAHS localmente (duplicacion de codigo)
  - Mide 6 etapas: CLAHE, SAHS, landmark single, landmark ensemble+TTA, warping, clasificacion
  - Warmup configurable para estabilizar GPU
  - Genera reporte con media, mediana, std, min, max por modulo
  - Calcula throughput (imagenes/segundo)
  - Guarda resultados en JSON
  - Carga modelos desde `ensemble_best.json` y clasificador desde checkpoint
  - `load_canonical_shape()` carga `.npy` (no JSON) -- posible inconsistencia con GPA output actual
  - **Problema**: Reimplementa CLAHE y SAHS localmente en lugar de importar desde `src_v2`
  - **Problema**: `load_canonical_shape()` busca `canonical_shape.npy` en lugar de `canonical_shape_gpa.json`
- **Importancia**: MEDIO
- **Justificacion**: Util para documentar tiempos de inferencia en la tesis y para optimizacion. La duplicacion de CLAHE/SAHS es un problema de mantenimiento. Los paths de carga de forma canonica podrian estar desalineados con el pipeline actual.

---

## C. Resumen Estadistico

| Categoria | Archivos | Lineas totales |
|-----------|----------|---------------|
| Verificacion | 11 | 2,374 |
| Generacion de datasets | 7 | 2,094 |
| Benchmark | 1 | 533 |
| **Total** | **19** | **4,901** |

## D. Clasificacion por Importancia

| Importancia | Archivos | Detalle |
|-------------|----------|---------|
| CRITICO | 2 | verify_data_leakage.py, verify_dataset_splits.py |
| ALTO | 5 | verify_gpa_correctness.py, verify_gui_setup.py, verify_gui.sh, verify_individual_models.py, generate_all_landmarks_npz.py, calculate_pfs_warped.py |
| MEDIO | 4 | verify_canonical_delaunay.py, verify_comparison_alignment.py, verify_landmark_viz_dataset.py, benchmark_inference.py |
| BAJO | 5 | verify_no_tta.py, verify_val_vs_test.py, generate_cropped_sahs_dataset.py, generate_warped_sahs_dataset.py, generate_icon.py |
| ELIMINABLE | 2 | generate_warped_dataset.py, generate_full_warped_dataset.py |

## E. Documentacion en CLAUDE.md

| Script | Documentado en CLAUDE.md |
|--------|------------------------|
| verify_landmark_viz_dataset.py | Si (comando de verificacion) |
| generate_warped_dataset.py | Si (marcado como Legacy) |
| generate_full_warped_dataset.py | Si (marcado como Legacy) |
| Resto (16 archivos) | No mencionados directamente |

## F. Problemas Detectados

### 1. Paths rotos post-cleanup de checkpoints (CRITICO)
Tres scripts de verificacion refieren `checkpoints/session10/exp4_epochs100/final_model.pt` que fue eliminado durante la limpieza de 133 GB documentada en `CHECKPOINTS_CLEANUP_REPORT.md`:
- `verify_data_leakage.py` (linea 94)
- `verify_individual_models.py` (linea 143)
- `verify_no_tta.py` (linea 77)
- `verify_val_vs_test.py` (linea 61)

**Recomendacion**: Actualizar los paths a checkpoints existentes o parametrizarlos via CLI/config.

### 2. Landmarks canonicos hardcodeados en calculate_pfs_warped.py (ALTO)
La funcion `get_canonical_landmarks()` (lineas 96-118) define 15 landmarks canonicos con valores que NO corresponden a la forma canonica GPA real del proyecto (los nombres como "Left shoulder", "Trachea" no coinciden con la estructura de 15 landmarks del contorno pulmonar).

**Recomendacion**: Reemplazar por carga desde `outputs/shape_analysis/canonical_shape_gpa.json`.

### 3. Duplicacion de codigo CLAHE/SAHS en benchmark_inference.py (MEDIO)
`benchmark_inference.py` reimplementa `apply_clahe_numpy()` y `apply_sahs()` localmente (lineas 43-112) en lugar de importarlas desde `src_v2`. Esto crea riesgo de divergencia si se modifica la implementacion principal.

**Recomendacion**: Importar desde `src_v2.data.transforms` o `src_v2.gui.model_manager`.

### 4. Carga inconsistente de forma canonica en benchmark (MEDIO)
`benchmark_inference.py::load_canonical_shape()` (lineas 186-189) busca `canonical_shape.npy` mientras que el pipeline actual genera `canonical_shape_gpa.json`. Esto causaria un error al ejecutar.

**Recomendacion**: Alinear con el formato JSON usado por `src_v2/processing/gpa.py`.

### 5. Dependencia de modulos legacy en scripts eliminables (BAJO)
`generate_warped_dataset.py` y `generate_full_warped_dataset.py` importan de `scripts.piecewise_affine_warp` y `scripts.predict`, que son modulos legacy. Esto solo es un problema si alguien intenta ejecutarlos.

**Recomendacion**: Mover a `scripts/archive/` como ya se sugiere en CLAUDE.md.

### 6. Codigo SAHS duplicado entre generate_cropped_sahs y generate_warped_sahs (BAJO)
Ambos scripts implementan SAHS independientemente. La version de `generate_warped_sahs_dataset.py` es mas correcta (aplica mascara), pero la logica base es identica.

**Recomendacion**: Si se mantienen, extraer la logica SAHS a un modulo compartido.

## G. Acciones Recomendadas

1. **Mover a archive**: `generate_warped_dataset.py`, `generate_full_warped_dataset.py` (ya marcados como legacy en CLAUDE.md)
2. **Actualizar paths**: Corregir los 4 scripts de verificacion con paths rotos a checkpoints
3. **Corregir landmarks canonicos**: En `calculate_pfs_warped.py`, cargar forma canonica desde JSON en lugar de hardcodear valores incorrectos
4. **Refactorizar benchmark**: Eliminar duplicacion de CLAHE/SAHS y corregir carga de forma canonica
5. **Considerar mover a archive**: `generate_cropped_sahs_dataset.py`, `generate_warped_sahs_dataset.py` (SAHS no es parte del pipeline final)
