# 11. src_v2 GUI Module

Analisis de la interfaz grafica Gradio para demostracion interactiva.

**Archivos analizados**: 9
**Lineas totales**: 3,301 (codigo: 2,811 / documentacion: 490)
**Tamano total**: ~107 KB

---

## Resumen Ejecutivo

El modulo `src_v2/gui/` implementa una interfaz web interactiva basada en Gradio para demostrar el pipeline completo de deteccion de COVID-19. Fue disenado especificamente para la defensa de tesis, proporcionando visualizacion paso a paso del proceso: imagen original, landmarks detectados, malla de Delaunay, imagen normalizada (warped), mejora de contraste SAHS, y clasificacion final con probabilidades.

La arquitectura es solida: patron Singleton para gestion de modelos, separacion clara de responsabilidades (configuracion, inferencia, visualizacion, UI), y manejo robusto de errores. El modulo GradCAM existe pero no se usa activamente en la interfaz actual (v1.0.10), aunque el heatmap se genera internamente en `classify_with_gradcam()`.

**Hallazgos principales**:
- Bug en `config.py`: `sys` se referencia antes de importarse (linea 22 vs 24)
- GradCAM se genera pero el heatmap no se muestra al usuario en la UI actual
- Codigo comentado extenso en `visualizer.py` (leyenda de 40 lineas)
- Tab "Acerca del Sistema" comentado en `app.py`
- Metricas en README.md desactualizadas respecto a config.py
- Duplicacion de normalizacion ImageNet en `model_manager.py` (lineas 248-250 y 428-430)

---

## Analisis Archivo por Archivo

### __init__.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/gui/__init__.py
- **Lineas/Tamano**: 8 lineas / 207 bytes
- **Proposito**: Inicializacion del paquete GUI con docstring descriptivo y numero de version.
- **Contenido clave**: `__version__ = "1.0.10"` - version actual del modulo GUI
- **Dependencias**: Ninguna. Es importado implicitamente al importar cualquier submodulo de `src_v2.gui`.
- **Importancia**: BAJO
- **Justificacion**: Archivo estandar de paquete Python. La version aqui (1.0.10) no se usa programaticamente en ningun otro archivo. El texto "Acerca del Sistema" en config.py tiene version hardcodeada como "1.0.13", mostrando inconsistencia.

---

### app.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/gui/app.py
- **Lineas/Tamano**: 488 lineas / 19.1 KB
- **Proposito**: Define la interfaz Gradio con 2 tabs activos (Demo Completa y Vista Rapida) y uno comentado (Acerca del Sistema). Conecta inputs/outputs con el pipeline de inferencia.
- **Contenido clave**:
  - `create_demo()` (linea 129): Funcion principal que construye la interfaz Gradio Blocks
  - `create_probability_html()` (linea 29): Genera barras de progreso HTML con colores por clase
  - `create_prediction_display_html()` (linea 75): HTML para mostrar la prediccion destacada
  - `highlight_winner_in_probabilities()` (linea 99): Resalta clase ganadora (no parece usarse actualmente)
  - `on_process()` (linea 267): Callback del boton "Procesar" en tab 1
  - `on_quick_classify()` (linea 407): Callback del boton "Clasificar" en tab 2
  - `on_export()` (linea 343): Callback del boton "Exportar PDF"
  - Tab 3 "Acerca del Sistema" completamente comentado (lineas 455-480)
- **Dependencias**:
  - Importa de: `gradio`, `pandas`, `.inference_pipeline` (process_image_full, process_image_quick, export_results), `.config` (TITLE, SUBTITLE, ABOUT_TEXT, THEME, etc.), `.visualizer` (create_probability_chart)
  - Importado por: `scripts/run_demo.py`, `scripts/verify_gui_setup.py`
- **Importancia**: CRITICO
- **Justificacion**: Es el punto de entrada de la interfaz grafica. Sin este archivo no hay GUI.

**Observaciones**:
1. `highlight_winner_in_probabilities()` esta definida pero no se invoca en ningun lugar del modulo.
2. `create_probability_chart` se importa de `visualizer.py` pero no se llama en `app.py` (se usa HTML personalizado en su lugar).
3. El export_btn.click tiene `outputs=[export_status, export_status]` (el mismo componente dos veces) -- parece intencional para hacer visible el componente, pero es un patron inusual.
4. Los callbacks `on_process` y `on_quick_classify` estan definidos como funciones locales dentro de `create_demo()`, lo cual los hace dificiles de testear unitariamente.

---

### config.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/gui/config.py
- **Lineas/Tamano**: 429 lineas / 16.3 KB
- **Proposito**: Configuracion centralizada del modulo GUI: rutas de modelos, metricas validadas, esquema de colores de landmarks, texto de interfaz en espanol, parametros de inferencia y exportacion.
- **Contenido clave**:
  - Rutas de modelos con soporte multi-modo (lineas 14-38): development, deployment (env var), PyInstaller frozen, portable
  - `VALIDATED_METRICS` (linea 90): Metricas de GROUND_TRUTH.json (cross-validation 5-fold)
  - `PER_LANDMARK_ERRORS` (linea 120): Error por landmark individual
  - `LANDMARK_COLORS/GROUPS/LABELS_ES` (lineas 131-154): Esquema visual de landmarks
  - `CLASS_NAMES`, `CLASS_NAMES_ES`, `CLASS_COLORS` (lineas 160-167): Clases de clasificacion
  - `get_class_color_es()` (linea 170): Obtener color por nombre de clase en espanol
  - `get_class_name_es()` (linea 403): Traduccion ingles a espanol de clases
  - `get_landmark_color()` (linea 391): Color por indice de landmark
  - `populate_examples()` (linea 413): Busca imagenes de ejemplo en `examples/`
  - `ABOUT_TEXT` (linea 202): Texto extenso (~130 lineas) para tab "Acerca del Sistema" (actualmente comentado en app.py)
  - Parametros de inferencia (lineas 357-377): DEVICE_PREFERENCE, TTA, CLAHE, SAHS, GradCAM, margin
  - Parametros de exportacion (lineas 383-385): DPI, formato, template de nombre
- **Dependencias**:
  - Importa de: `os`, `pathlib.Path`, `typing`
  - Importado por: `.app`, `.inference_pipeline`, `.model_manager`, `.visualizer`, `scripts/run_demo.py`, `scripts/verify_gui_setup.py`, `scripts/visualization/generate_feature_maps_pipeline.py`
- **Importancia**: CRITICO
- **Justificacion**: Es el eje central de configuracion del modulo GUI. Todos los demas archivos del modulo dependen de el.

**BUG DETECTADO** (lineas 22-24):
```python
elif IS_FROZEN and hasattr(sys, '_MEIPASS'):  # sys no importado aun!
    # PyInstaller frozen mode fallback
    import sys  # import tardio, despues de uso
```
`sys` se usa en `hasattr(sys, '_MEIPASS')` en la linea 22, pero `import sys` esta en la linea 24 (dentro del bloque elif). Si `IS_FROZEN` es `True` y `MODELS_DIR` es `None`, se producira un `NameError: name 'sys' is not defined`. Esto no falla en modo desarrollo porque la condicion `IS_FROZEN` normalmente es `False`, pero fallaria en un empaquetado PyInstaller real.

**Otras observaciones**:
1. `ABOUT_TEXT` tiene version hardcodeada "1.0.13" (linea 328) mientras `__init__.py` tiene "1.0.10".
2. `IS_PORTABLE` solo se define en dos ramas del if/elif (lineas 31 y 38), no en las ramas de `MODELS_DIR` o `IS_FROZEN`.
3. Las metricas de clasificacion cambiaron de single-split (99.10% en CLAUDE.md) a cross-validation (98.60% en config.py), lo cual es correcto pero podria causar confusion.

---

### gradcam_utils.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/gui/gradcam_utils.py
- **Lineas/Tamano**: 261 lineas / 8.1 KB
- **Proposito**: Implementacion de Gradient-weighted Class Activation Mapping (GradCAM) para generar mapas de calor que muestran las regiones de atencion del clasificador.
- **Contenido clave**:
  - `class GradCAM` (linea 15): Implementacion con forward/backward hooks en capa target
    - `__init__()`: Registra hooks en `target_layer` (default: 'layer4')
    - `_get_target_layer()`: Busca la capa en model.backbone, model directo, o named_modules
    - `_register_hooks()`: Forward hook captura activaciones, backward hook captura gradientes
    - `generate()`: Forward + backward pass, weighted combination, ReLU, normalizacion
    - `_normalize_heatmap()`: Normaliza a [0,1] con proteccion contra division por cero
  - `generate_gradcam()` (linea 137): Funcion convenience que crea instancia GradCAM y genera heatmap
  - `resize_heatmap()` (linea 169): Redimensiona con interpolacion bilineal (cv2)
  - `apply_colormap()` (linea 189): Aplica COLORMAP_JET y convierte BGR a RGB
  - `overlay_heatmap_on_image()` (linea 217): Mezcla heatmap coloreado sobre imagen con alpha blending
- **Dependencias**:
  - Importa de: `cv2`, `numpy`, `torch`, `torch.nn`, `torch.nn.functional`
  - Importado por: `.model_manager` (generate_gradcam), `.visualizer` (overlay_heatmap_on_image)
- **Importancia**: MEDIO
- **Justificacion**: GradCAM se genera internamente en `model_manager.classify_with_gradcam()`, pero el heatmap resultante se descarta en `inference_pipeline.py` (linea 169: `probabilities, _, predicted_class_idx`). La funcion `render_gradcam()` existe en `visualizer.py` pero no se invoca desde `app.py`. El modulo esta funcional pero sub-utilizado en la version actual.

**Observaciones**:
1. `cv2` se importa 3 veces: una vez al nivel de modulo (linea 7) y dos veces dentro de funciones (lineas 181 y 203). Las importaciones locales son redundantes.
2. `register_forward_hook` y `register_full_backward_hook` (linea 70-71) no se almacenan como handles, lo cual impide removerlos despues. Esto podria causar memory leaks si se crean multiples instancias GradCAM.
3. La implementacion es estandar y correcta para ResNet-18.

---

### inference_pipeline.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/gui/inference_pipeline.py
- **Lineas/Tamano**: 390 lineas / 11.0 KB
- **Proposito**: Orquestador del pipeline de inferencia para la GUI. Coordina validacion, preprocesamiento, prediccion de landmarks, warping, clasificacion, y generacion de visualizaciones.
- **Contenido clave**:
  - `validate_image()` (linea 46): Valida existencia, formato (.png/.jpg/.jpeg/.bmp/.tiff), carga, y tamano minimo (100x100)
  - `load_and_preprocess()` (linea 82): Carga como grayscale y redimensiona a 224x224
  - `process_image_full()` (linea 102): Pipeline completo - retorna dict con imagenes PIL (original, landmarks, delaunay, warped, sahs), clasificacion, metricas, tiempo. Manejo robusto de errores en cada paso.
  - `process_image_quick()` (linea 234): Pipeline rapido - solo clasificacion sin visualizaciones intermedias. Accede a metodo privado `_prepare_image_for_classifier()` directamente.
  - `export_results()` (linea 312): Exporta resultados a PDF con timestamp en nombre
  - `create_comparison_visualization()` (linea 376): Wrapper simple sobre `render_comparison_side_by_side()`
- **Dependencias**:
  - Importa de: `time`, `datetime`, `pathlib`, `typing`, `cv2`, `numpy`, `torch`, `PIL.Image`, `.model_manager` (get_model_manager), `.visualizer` (render_*, create_*, export_to_pdf), `.config` (get_class_name_es, ERROR_*, SUCCESS_EXPORT, EXPORT_*)
  - Importado por: `.app` (process_image_full, process_image_quick, export_results)
- **Importancia**: CRITICO
- **Justificacion**: Es la capa de orquestacion entre la UI y los modelos. Toda la logica de procesamiento pasa por aqui.

**Observaciones**:
1. En `process_image_full()` linea 169, el heatmap GradCAM se descarta: `probabilities, _, predicted_class_idx = manager.classify_with_gradcam(...)`. Se ejecuta backpropagation innecesariamente (GradCAM requiere gradientes), lo cual es mas lento que una simple forward pass con `torch.no_grad()`.
2. `process_image_quick()` linea 280 accede a metodo privado `manager._prepare_image_for_classifier()`, lo que rompe la encapsulacion.
3. `create_comparison_visualization()` (linea 376) no se usa en ningun lugar.
4. El type hint `tuple[bool, Optional[str]]` (linea 46) usa sintaxis Python 3.10+, pero el proyecto dice soportar Python 3.8+. Deberia ser `Tuple[bool, Optional[str]]` para compatibilidad.

---

### model_manager.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/gui/model_manager.py
- **Lineas/Tamano**: 461 lineas / 14.4 KB
- **Proposito**: Singleton para gestion lazy de todos los modelos: ensemble de 4 landmarks, forma canonica, triangulacion, clasificador. Provee metodos de alto nivel para prediccion, warping y clasificacion.
- **Contenido clave**:
  - `_apply_clahe_numpy()` (linea 35): Funcion standalone para aplicar CLAHE a arrays numpy (creada porque `src_v2/data/transforms.py::apply_clahe()` espera PIL Image)
  - `class ModelManager` (linea 62): Singleton con `__new__()` y flag `_initialized`
    - `initialize()` (linea 83): Carga todos los modelos, detecta dispositivo (GPU/CPU)
    - `_load_landmark_ensemble()` (linea 123): Carga 4 modelos ResNet-18 con Coordinate Attention
    - `_load_canonical_data()` (linea 154): Carga canonical shape (15,2) y triangles (18,3) desde JSON
    - `_load_classifier()` (linea 182): Carga clasificador ResNet-18 via `load_classifier_checkpoint()`
    - `predict_landmarks()` (linea 203): Prediccion con ensemble + TTA + CLAHE
    - `_predict_with_tta()` (linea 275): TTA horizontal flip con correccion de pares simetricos
    - `warp_image()` (linea 309): Warping piecewise affine con margin scale
    - `classify_with_gradcam()` (linea 357): Clasificacion con generacion de GradCAM
    - `_prepare_image_for_classifier()` (linea 401): Prepara tensor normalizado (ImageNet stats)
    - `get_status()` (linea 437): Retorna estado de inicializacion
  - `get_model_manager()` (linea 459): Funcion de acceso al singleton
- **Dependencias**:
  - Importa de: `json`, `pathlib`, `typing`, `cv2`, `numpy`, `torch`, `torch.nn`, `src_v2.models.resnet_landmark` (create_model), `src_v2.models.classifier` (load_classifier_checkpoint), `src_v2.processing.warp` (piecewise_affine_warp, scale_landmarks_from_centroid), `src_v2.constants` (SYMMETRIC_PAIRS), `.config` (rutas, parametros), `.gradcam_utils` (generate_gradcam)
  - Importado por: `.inference_pipeline` (get_model_manager), `scripts/verify_gui_setup.py` (_apply_clahe_numpy)
- **Importancia**: CRITICO
- **Justificacion**: Gestiona la carga y ejecucion de todos los modelos de deep learning. Es el nucleo computacional del modulo GUI.

**Observaciones**:
1. La normalizacion ImageNet (mean/std) se repite en dos metodos: `predict_landmarks()` (lineas 248-250) y `_prepare_image_for_classifier()` (lineas 428-430). Deberia extraerse a una funcion helper.
2. El singleton no es thread-safe. Si dos requests Gradio llegan simultaneamente durante la inicializacion, podrian crear race conditions. Para un demo de tesis esto es aceptable.
3. `_apply_clahe_numpy()` duplica funcionalidad de `src_v2/data/transforms.py` -- fue necesario porque la version en transforms espera PIL Image.
4. `_load_landmark_ensemble()` hardcodea parametros de arquitectura (dropout_rate=0.3, hidden_dim=768, deep_head=True). Estos deberian idealmente leerse del checkpoint o de la configuracion.

---

### visualizer.py
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/gui/visualizer.py
- **Lineas/Tamano**: 774 lineas / 23.4 KB
- **Proposito**: Funciones de renderizado para todas las visualizaciones del pipeline: imagenes originales, overlay de landmarks con colores por grupo, malla de Delaunay, imagen warped, SAHS, GradCAM, graficos de probabilidad, tablas de metricas, y exportacion a PDF multipagina.
- **Contenido clave**:
  - `render_original()` (linea 33): Renderiza imagen original como PIL via matplotlib
  - `render_landmarks_overlay()` (linea 61): Overlay de 15 landmarks con colores por grupo, labels L1-L15, lineas de conexion (simetria + eje central). Leyenda comentada en v1.0.9 (~40 lineas).
  - `_draw_connection_lines()` (linea 195): Dibuja lineas de simetria (pares izq-der, blancas punteadas) y eje central vertical (L1-L9-L10-L11-L2, cyan solida)
  - `render_delaunay_mesh()` (linea 241): Renderiza triangulacion de Delaunay sobre imagen original. Calcula triangulacion via `scipy.spatial.Delaunay` (no usa la triangulacion pre-calculada del canonical).
  - `render_warped()` (linea 350): Renderiza imagen normalizada simple
  - `enhance_contrast_sahs_masked()` (linea 374): Algoritmo SAHS (Statistical Asymmetrical Histogram Stretching) que solo opera sobre region pulmonar (pixeles > threshold). Factores: 2.5 superior, 2.0 inferior.
  - `render_warped_sahs()` (linea 444): Renderiza imagen warped con SAHS aplicado
  - `render_gradcam()` (linea 474): Renderiza overlay de GradCAM con colorbar (no usado actualmente en la UI)
  - `render_comparison_side_by_side()` (linea 522): Comparacion lado a lado original/warped
  - `create_probability_chart()` (linea 556): Grafico de barras horizontales de probabilidades (importado pero no usado en app.py)
  - `create_metrics_table()` (linea 634): Tabla DataFrame con coordenadas de landmarks
  - `export_to_pdf()` (linea 668): PDF multipagina con 5 paginas: landmarks, Delaunay, warped, SAHS, metricas
  - `_fig_to_pil()` (linea 767): Conversion matplotlib figure a PIL Image via buffer BytesIO
- **Dependencias**:
  - Importa de: `io`, `typing`, `cv2`, `matplotlib`, `numpy`, `pandas`, `matplotlib.patches.Polygon`, `PIL.Image`, `scipy.spatial.Delaunay`, `.config` (colores, nombres), `.gradcam_utils` (overlay_heatmap_on_image), `..constants` (SYMMETRIC_PAIRS, CENTRAL_LANDMARKS)
  - Importado por: `.inference_pipeline` (render_*, create_*, export_to_pdf), `.app` (create_probability_chart), `scripts/visualization/generate_feature_maps_pipeline.py` (render_landmarks_overlay)
- **Importancia**: ALTO
- **Justificacion**: Archivo mas grande del modulo (774 lineas). Contiene toda la logica de visualizacion. Critico para la presentacion visual pero no para la logica de inferencia.

**Observaciones**:
1. `render_delaunay_mesh()` usa `scipy.spatial.Delaunay(landmarks_px)` en lugar de la triangulacion pre-calculada que ya esta cargada en `ModelManager`. Esto puede dar triangulaciones diferentes de la usada en el warping real.
2. `create_probability_chart()` se importa en `app.py` pero no se usa (la UI usa HTML personalizado con `create_probability_html()` en su lugar).
3. `render_gradcam()` existe y funciona pero no se invoca desde la UI actual.
4. `render_comparison_side_by_side()` no se usa en la UI.
5. El codigo comentado de la leyenda en `render_landmarks_overlay()` ocupa ~40 lineas (141-181).
6. `matplotlib.use('Agg')` (linea 20) es correcto para thread safety con Gradio.
7. `enhance_contrast_sahs_masked()` es un algoritmo interesante y bien documentado para mejorar contraste solo en la region pulmonar.

---

### README.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/gui/README.md
- **Lineas/Tamano**: 241 lineas / 7.4 KB
- **Proposito**: Documentacion completa del modulo GUI: requisitos, uso, estructura del codigo, arquitectura, metricas, troubleshooting, y notas para defensa de tesis.
- **Contenido clave**:
  - Instrucciones de lanzamiento (3 opciones)
  - Descripcion de cada tab de la interfaz
  - Diagrama del pipeline de inferencia (ASCII art)
  - Tabla de metricas validadas
  - Tabla de colores de landmarks por grupo
  - Seccion de troubleshooting con 4 escenarios comunes
  - Seccion "Notas para Defensa de Tesis" con puntos clave y backup plan
  - Lista de extensiones futuras
- **Dependencias**: Ninguna (documentacion)
- **Importancia**: BAJO
- **Justificacion**: Documentacion util pero no ejecutable. Las metricas estan desactualizadas respecto a config.py (README dice 98.05% accuracy y 97.12% F1-macro; config.py dice 98.60% y 98.00%).

**Observaciones**:
1. Menciona "GradCAM" como Tab 4 del pipeline, pero la UI actual muestra "SAHS" en su lugar.
2. La version "1.0.0" (linea 239) difiere de `__init__.py` (1.0.10).
3. Tab 3 "Acerca del Sistema" esta listado como activo pero esta comentado en el codigo.

---

### CHANGELOG.md
- **Ruta**: /home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/gui/CHANGELOG.md
- **Lineas/Tamano**: 249 lineas / 7.1 KB
- **Proposito**: Historial de cambios detallado del modulo GUI. Documenta la implementacion inicial (v1.0.0, 2026-01-18) con todas las caracteristicas, bugs corregidos, estadisticas de codigo, y metricas de rendimiento.
- **Contenido clave**:
  - Implementacion inicial con 11 archivos nuevos y ~2,600 lineas
  - 4 fixes documentados: CLAHE TypeError, scale_landmarks ArgumentError, torch.Tensor.numpy RuntimeError, torch.no_grad llamada incorrecta
  - Tiempos de inferencia: ~1 segundo total (800ms landmarks + 50ms warping + 100ms clasificacion)
  - Uso de memoria: ~2 GB GPU, ~1.5 GB RAM
  - Solo documenta v1.0.0 (no hay entradas para v1.0.1-v1.0.10)
- **Dependencias**: Ninguna (documentacion)
- **Importancia**: BAJO
- **Justificacion**: Solo tiene una entrada (v1.0.0) a pesar de que la version actual es 1.0.10. Las 10 versiones intermedias no estan documentadas aqui.

---

## Matriz de Dependencias Internas del Modulo

```
config.py -----> (base de todo, sin dependencias internas)
     |
     v
gradcam_utils.py (solo depende de torch, cv2, numpy)
     |
     v
model_manager.py --> config.py, gradcam_utils.py, src_v2.models.*, src_v2.processing.*, src_v2.constants
     |
     v
visualizer.py ----> config.py, gradcam_utils.py, src_v2.constants
     |
     v
inference_pipeline.py --> model_manager.py, visualizer.py, config.py
     |
     v
app.py -----------> inference_pipeline.py, config.py, visualizer.py
```

## Consumidores Externos

| Script externo | Que importa |
|---|---|
| `scripts/run_demo.py` | `config.LANDMARK_MODELS`, `config.CANONICAL_SHAPE`, etc.; `app.create_demo()` |
| `scripts/verify_gui_setup.py` | `config.*`, `model_manager._apply_clahe_numpy`, `app.create_demo` |
| `scripts/visualization/generate_feature_maps_pipeline.py` | `gui.config`, `visualizer.render_landmarks_overlay` |

---

## Resumen de Importancia

| Archivo | Lineas | Importancia | Justificacion |
|---------|--------|-------------|---------------|
| `__init__.py` | 8 | BAJO | Solo version y docstring |
| `app.py` | 488 | CRITICO | Punto de entrada de la interfaz Gradio |
| `config.py` | 429 | CRITICO | Configuracion central de todo el modulo |
| `gradcam_utils.py` | 261 | MEDIO | Funcional pero sub-utilizado (heatmap se genera y descarta) |
| `inference_pipeline.py` | 390 | CRITICO | Orquesta todo el pipeline de inferencia |
| `model_manager.py` | 461 | CRITICO | Nucleo computacional: carga y ejecuta todos los modelos |
| `visualizer.py` | 774 | ALTO | Toda la logica de visualizacion, archivo mas grande |
| `README.md` | 241 | BAJO | Documentacion util pero desactualizada |
| `CHANGELOG.md` | 249 | BAJO | Solo documenta v1.0.0, falta v1.0.1-v1.0.10 |

---

## Hallazgos Consolidados

### Bugs

1. **config.py linea 22**: `sys` se referencia en `hasattr(sys, '_MEIPASS')` antes de `import sys` en linea 24. Causaria `NameError` en modo PyInstaller si `IS_FROZEN=True` y `MODELS_DIR` no esta seteado.

2. **config.py lineas 27-38**: `IS_PORTABLE` solo se define en las ramas de portable mode (linea 31) y development mode (linea 38), no en las ramas de deployment o frozen mode. Podria causar `NameError` si se accede.

3. **inference_pipeline.py linea 46**: `tuple[bool, Optional[str]]` usa sintaxis Python 3.10+. El proyecto declara soporte para Python 3.8+. Deberia ser `Tuple[bool, Optional[str]]` (del modulo `typing`).

### Codigo Muerto o Sub-utilizado

1. **app.py**: `highlight_winner_in_probabilities()` (lineas 99-126) -- definida pero nunca invocada
2. **app.py**: `create_probability_chart` importada de visualizer pero nunca llamada
3. **visualizer.py**: `render_gradcam()` (lineas 474-519) -- nunca invocada desde la UI
4. **visualizer.py**: `create_probability_chart()` (lineas 556-631) -- importada pero no usada en app.py
5. **visualizer.py**: `render_comparison_side_by_side()` (lineas 522-553) -- solo usada en `create_comparison_visualization()` de inference_pipeline.py, que tampoco se invoca
6. **inference_pipeline.py**: `create_comparison_visualization()` (lineas 376-390) -- nunca invocada
7. **config.py**: `ABOUT_TEXT` (~130 lineas) -- tab "Acerca del Sistema" esta comentado

### Ineficiencias

1. **inference_pipeline.py linea 169**: `classify_with_gradcam()` ejecuta backward pass (para GradCAM) pero descarta el heatmap. `process_image_full()` deberia usar forward pass simple como `process_image_quick()` si no va a mostrar el heatmap.

2. **model_manager.py**: Normalizacion ImageNet duplicada en `predict_landmarks()` (lineas 248-250) y `_prepare_image_for_classifier()` (lineas 428-430).

3. **visualizer.py linea 287**: `render_delaunay_mesh()` recalcula la triangulacion de Delaunay via `scipy.spatial.Delaunay()` cuando el `ModelManager` ya tiene la triangulacion pre-calculada cargada desde JSON. Esto puede producir una triangulacion diferente a la realmente usada en el warping.

### Inconsistencias de Version/Metricas

| Fuente | Version | Accuracy | F1-Macro |
|--------|---------|----------|----------|
| `__init__.py` | 1.0.10 | -- | -- |
| `config.py ABOUT_TEXT` | 1.0.13 | 98.60% | 98.00% |
| `README.md` | 1.0.0 | 98.05% | 97.12% |
| `CHANGELOG.md` | 1.0.0 | 98.05% | 97.12% |
| `CLAUDE.md` | -- | 99.10% (single split) | -- |

### Deuda Tecnica

1. Codigo comentado extenso: ~40 lineas de leyenda en visualizer.py, ~25 lineas de Tab 3 en app.py
2. Funciones definidas como closures dentro de `create_demo()` dificultan testing unitario
3. El patron Singleton de ModelManager no es thread-safe
4. Hardcoded model architecture params en `_load_landmark_ensemble()` en lugar de leerlos del checkpoint
