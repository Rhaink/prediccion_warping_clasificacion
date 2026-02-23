# 10. src_v2 Visualization Module

Analisis de las herramientas de visualizacion: GradCAM, PFS, ROC, diagramas y mas.

**Archivos analizados**: 12
**Lineas totales**: 6,267
**Tamano total**: ~207 KB

---

## Resumen Ejecutivo

El modulo `visualization` es el mas grande del paquete `src_v2` y contiene toda la infraestructura para generar figuras de calidad publicable, mapas de activacion (GradCAM), analisis de atencion pulmonar (PFS), curvas ROC, paneles de casos fallidos, visualizacion de features internos del modelo ("glass box"), y visualizaciones de landmarks con y sin ground truth. Es un modulo fundamentalmente orientado a producir artefactos visuales para tesis/paper, no a la logica de entrenamiento o inferencia.

El modulo se organiza conceptualmente en cuatro capas:
1. **Interpretabilidad del clasificador**: `gradcam.py`, `pfs_analysis.py`, `error_analysis.py`
2. **Figuras de publicacion**: `plot_roc_curves.py`, `plot_failure_cases.py`
3. **Visualizacion de landmarks**: `scientific_viz.py`, `comparison_viz.py`
4. **Glass Box (features internos)**: `feature_extractor.py`, `feature_visualizer.py`, `utils.py`
5. **Diagramas de pipeline**: `diagramming.py`

---

## Analisis Archivo por Archivo

### 1. __init__.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/visualization/__init__.py`
- **Lineas/Tamano**: 90 lineas / 2.5 KB
- **Proposito**: Punto de entrada del modulo de visualizacion. Re-exporta las clases y funciones principales de todos los submodulos para facilitar imports limpios desde el exterior.
- **Contenido clave**:
  - Importa y re-exporta `GradCAM`, `get_target_layer`, `calculate_pfs`, `overlay_heatmap` desde `gradcam`
  - Importa `ErrorAnalyzer`, `analyze_classification_errors` desde `error_analysis`
  - Importa todas las clases PFS (`PFSAnalyzer`, `PFSResult`, `PFSSummary`, `run_pfs_analysis`, etc.) desde `pfs_analysis`
  - Importa funciones de `scientific_viz` y `comparison_viz`
  - Los imports de Glass Box (`FeatureExtractor`, `FeatureVisualizer`, utilidades) estan envueltos en `try/except ImportError` con flag `_glass_box_available`, lo cual es una buena practica defensiva
  - Define `__all__` con ~30 simbolos exportados, extendido condicionalmente con los de glass box
- **Dependencias**: Todos los demas archivos del modulo. No importa `diagramming`, `plot_failure_cases` ni `plot_roc_curves` (estos son scripts standalone)
- **Importancia**: MEDIO
- **Justificacion**: Archivo organizativo que facilita el uso del modulo. No contiene logica propia pero es necesario para una API limpia. Bien estructurado con imports condicionales.

---

### 2. comparison_viz.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/visualization/comparison_viz.py`
- **Lineas/Tamano**: 953 lineas / 33.2 KB
- **Proposito**: Genera visualizaciones comparativas mostrando landmarks predichos (cruces rojas) y ground truth (cruces verdes) sobre la misma imagen. Soporta dos modos: (1) splits propios via `train_test_split` y (2) alineamiento con el dataset warped existente.
- **Contenido clave**:
  - `load_ground_truth_mapping(csv_path)`: Carga landmarks GT desde CSV maestro, retorna dict {image_name: landmarks_299}
  - `match_predictions_with_gt(predictions_dict, gt_dict)`: Empareja predicciones con GT. Busqueda lineal O(N*M) en el peor caso (ineficiente para datasets grandes pero aceptable para ~957 imagenes GT)
  - `create_comparison_visualization(image, pred, gt, ...)`: Funcion core que dibuja cruces duales + calcula metricas de error por landmark
  - `generate_comparison_dataset(...)`: Pipeline completo con splits propios (75/15/10), procesamiento batch y generacion de estadisticas detalladas (overall, per-split, per-category, per-landmark, worst cases)
  - `generate_comparison_dataset_aligned(...)`: Variante mas sofisticada que hereda los splits del dataset warped existente (via `images.csv`), garantizando alineamiento 1:1 con el clasificador. Schema version 2.0.0 vs 1.0.0 del anterior
  - `find_original_image(base_dir, name, category)`: Helper para buscar imagen en multiples estructuras de directorio
  - `load_predictions_mapping(npz_path)`: Carga NPZ con manejo de formatos multiples (`landmarks`/`image_names` vs `predictions`/`image_paths`)
- **Dependencias**:
  - Importa: `src_v2.constants`, `src_v2.data.utils`, `src_v2.visualization.utils.draw_scientific_crosses_on_image`, sklearn, cv2, numpy, pandas, tqdm
  - Importado por: `src_v2/cli.py` (comandos `generate-landmark-comparison-dataset` y variante aligned), `__init__.py`
- **Importancia**: ALTO
- **Justificacion**: Herramienta esencial para validar visualmente la calidad del ensemble de landmarks (3.61 px). La variante aligned es especialmente importante para debugging del clasificador. Buen manejo de metadatos y estadisticas. Code duplication moderada entre `generate_comparison_dataset` y `generate_comparison_dataset_aligned` (la logica de estadisticas y procesamiento se repite casi identica).

**Observaciones**:
- Las dos funciones `generate_comparison_dataset` y `generate_comparison_dataset_aligned` comparten ~200 lineas de logica duplicada (calculo de estadisticas, guardado de JSON/CSV). Podrian refactorizarse extrayendo la logica comun.
- `match_predictions_with_gt` usa busqueda lineal anidada en lugar de lookup por diccionario, lo cual es O(N*M). Con 957 GT y 15153 predicciones, se recorren ~14.5M iteraciones innecesariamente. Deberia usar un dict lookup directo.
- Hay un `from sklearn.model_selection import train_test_split` importado dentro de la funcion (linea 322), lo cual es correcto para evitar dependencias al import-time.

---

### 3. diagramming.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/visualization/diagramming.py`
- **Lineas/Tamano**: 307 lineas / 9.1 KB
- **Proposito**: Genera diagramas de arquitectura del pipeline para la tesis/paper. Soporta NetworkX, Graphviz (pydot), Keras `plot_model`, `ann_visualizer` y BertViz. Todas las dependencias son opcionales.
- **Contenido clave**:
  - `PIPELINE_NODES`, `PIPELINE_EDGES`, `PIPELINE_COLORS`: Definicion declarativa del pipeline como grafo dirigido con 8 nodos y 7 aristas
  - `build_pipeline_graph()`: Construye un `nx.DiGraph` con nodos etiquetados por tipo (data, model, process, output)
  - `save_pipeline_networkx_diagram(output_path, layout, title)`: Renderiza el pipeline con NetworkX + matplotlib. Fallback de graphviz_layout a spring_layout
  - `save_pipeline_graphviz_diagram(output_path, rankdir, title)`: Alternativa usando pydot directamente
  - `save_keras_model_diagram(model, output_path, prefer)`: Para modelos Keras (no PyTorch). Soporta `plot_model` y `ann_visualizer`
  - `save_bertviz_attention_html(attentions, tokens, output_path)`: Para visualizar atencion de transformers. Manejo robusto de diferentes versiones de BertViz API
  - `_coerce_output_path(output_path, default_suffix)`: Helper para normalizar paths de salida
- **Dependencias**:
  - Importa: Solo stdlib (logging, pathlib). Las dependencias externas (networkx, matplotlib, pydot, tensorflow, ann_visualizer, bertviz) son lazy imports dentro de cada funcion con mensajes de error claros
  - Importado por: Nadie (no esta en `__init__.py` ni es importado por ningun otro modulo)
- **Importancia**: BAJO
- **Justificacion**: Modulo utilitario para generar diagramas de tesis. Las funciones de Keras y BertViz son irrelevantes para este proyecto (que usa PyTorch, no TensorFlow/transformers). Solo las funciones de pipeline NetworkX/Graphviz podrian ser utiles. No es importado por ningun modulo ni registrado en `__init__.py`. Bien implementado con lazy imports, pero con funcionalidad mayormente no utilizada.

**Observaciones**:
- `save_keras_model_diagram` y `save_bertviz_attention_html` no son relevantes para este proyecto y podrian ser eliminadas o movidas a un modulo de utilidades generico.
- No esta registrado en `__init__.py` ni `__all__`, lo cual indica que es un modulo auxiliar no integrado al API publico.

---

### 4. error_analysis.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/visualization/error_analysis.py`
- **Lineas/Tamano**: 478 lineas / 16.2 KB
- **Proposito**: Herramientas para analizar errores de clasificacion: recopilar muestras mal clasificadas, generar estadisticas de confusion, visualizar distribucion de errores y confianza, y producir reportes en JSON/CSV.
- **Contenido clave**:
  - `ErrorDetail`: Dataclass con detalles de cada error (path, clase real/predicha, confianza, probabilidades)
  - `ErrorSummary`: Dataclass con estadisticas resumidas (total, error_rate, pares de confusion, confianza promedio, matriz de confusion)
  - `ErrorAnalyzer`: Clase principal que acumula predicciones correctas e incorrectas batch por batch
    - `add_prediction()`: Procesa una prediccion individual, actualiza confusion matrix, almacena error si corresponde
    - `add_batch()`: Wrapper para procesar batches completos
    - `get_summary()`: Genera `ErrorSummary` con todas las estadisticas
    - `get_top_errors(k, by, descending)`: Top-K errores ordenados por confianza
    - `get_errors_by_pair(true_class, predicted_class)`: Filtrado por par de confusion
    - `save_reports()`: Genera JSON summary, CSV details, y confusion analysis JSON
  - `analyze_classification_errors(model, dataloader, class_names, device)`: Funcion de conveniencia que ejecuta el analisis completo
  - `create_error_visualizations(analyzer, output_dir, copy_images)`: Genera figuras matplotlib:
    - Distribucion de errores por clase real
    - Histograma de confianza (errores vs correctas)
    - Heatmap de confusion matrix
    - Opcionalmente copia imagenes mal clasificadas organizadas por clase
- **Dependencias**:
  - Importa: torch, numpy, PIL, matplotlib (lazy), dataclasses, csv, json, collections
  - Importado por: `__init__.py`, `src_v2/cli.py` (comando evaluate-classifier)
- **Importancia**: ALTO
- **Justificacion**: Modulo critico para entender fallos del clasificador (99.10% accuracy implica ~136 errores en 15153 imagenes). Bien disenado con dataclasses limpias, acumulacion incremental, y reportes multiformato. Integrado con el CLI via el comando evaluate-classifier.

**Observaciones**:
- El parametro `by` en `get_top_errors` no se usa realmente (siempre ordena por `e.confidence`), deberia soportar diferentes criterios o eliminarse el parametro.
- La funcion `analyze_classification_errors` asume que el batch dict tiene claves `"image"`, `"label"`, `"path"`, lo cual es especifico al formato de `LandmarkDataset`. Deberia documentarse mejor esta dependencia.

---

### 5. feature_extractor.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/visualization/feature_extractor.py`
- **Lineas/Tamano**: 276 lineas / 8.9 KB
- **Proposito**: Extraccion de feature maps intermedios de modelos PyTorch mediante forward hooks. Pieza fundamental del subsistema "glass box" para visualizar que aprenden las capas internas de los modelos.
- **Contenido clave**:
  - `FeatureExtractor`: Clase base generica
    - `_register_hooks()`: Registra forward hooks en capas target (o en todas las hojas si no se especifican)
    - `_make_hook(name)`: Genera closures que capturan outputs detached en CPU
    - `_get_module_by_name(name)`: Navegacion por nombre jerarquico con notacion punto
    - `get_features()`, `clear_features()`, `remove_hooks()`: API limpia para el ciclo de vida
    - `__del__()`: Cleanup automatico de hooks al ser garbage collected
  - `LandmarkFeatureExtractor(FeatureExtractor)`: Especializado para `ResNet18Landmarks`. Hooks por defecto en `backbone_conv.4-7`, `coord_attention`, `avgpool`, `head`. Metodos `get_backbone_features()` y `get_attention_maps()`
  - `ClassifierFeatureExtractor(FeatureExtractor)`: Especializado para `ImageClassifier`. Hooks en `backbone.layer1-4`, `backbone.avgpool`, `fc`
  - `extract_features_from_batch(model, images, target_layers)`: Funcion de conveniencia one-shot
  - `get_available_layers(model, max_depth)`: Utilidad para descubrir capas disponibles
- **Dependencias**:
  - Importa: torch, numpy (solo tipos)
  - Importado por: `__init__.py` (condicional), `scripts/glass_box_visualizations/block_b_landmarks.py`, `scripts/visualization/generate_feature_maps_pipeline.py`
- **Importancia**: MEDIO
- **Justificacion**: Bien implementado con herencia limpia y API clara. Manejo correcto de hooks (detach + CPU para evitar memory leaks). Sin embargo, es usado solo por scripts de visualizacion y no por el pipeline principal. Es la base del subsistema glass box.

**Observaciones**:
- `_make_hook` usa `print()` (linea 67) en lugar de `logger.warning()` para avisar de capas no encontradas. Inconsistente con el patron del resto del proyecto.
- `LandmarkFeatureExtractor.DEFAULT_LAYERS` usa indices numericos (`backbone_conv.4`, `backbone_conv.5`) que son fragiles y dependen de la estructura interna de `nn.Sequential`. Si el modelo cambia, estos indices se rompen silenciosamente.

---

### 6. feature_visualizer.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/visualization/feature_visualizer.py`
- **Lineas/Tamano**: 453 lineas / 15.8 KB
- **Proposito**: Renderizado de feature maps extraidos como figuras de calidad publicable. Complementa `feature_extractor.py` en el subsistema glass box.
- **Contenido clave**:
  - `FeatureVisualizer`: Clase principal con configuracion de estilo (DPI, font sizes, etc.)
    - `__init__()`: Configura `plt.rcParams` globalmente (efecto secundario que afecta todo matplotlib). Usa estilo `seaborn-v0_8-paper` para modo cientifico
    - `plot_feature_grid(feature_map, n_cols, ...)`: Grid de canales individuales con normalizacion per-channel, colorbar opcional. Limita a 64 canales por defecto
    - `plot_feature_hierarchy(layer_features, ...)`: Visualizacion multi-capa (filas = capas, columnas = canales top por varianza). Usa `GridSpec` con columna de labels. Soporta `layer_names` y `layer_details` para anotaciones personalizadas
    - `plot_single_feature_overlay(image, feature_map, channel_idx, alpha)`: Triptico imagen-feature-overlay
    - `save(filepath)` y `close()`: Gestion del ciclo de vida de figuras
  - `quick_visualize_layer(feature_map, ...)`: Funcion de conveniencia para prototipado rapido (DPI 150)
- **Dependencias**:
  - Importa: numpy, matplotlib, torch. Imports lazy de `src_v2.visualization.utils` dentro de metodos
  - Importado por: `__init__.py` (condicional), scripts de glass box
- **Importancia**: MEDIO
- **Justificacion**: Complemento necesario de feature_extractor para producir figuras de calidad publicable de los features intermedios. Bien estructurado con multiples tipos de visualizacion. El init modifica globals de matplotlib, lo cual podria causar efectos secundarios si se usa en combinacion con otros visualizadores.

**Observaciones**:
- `plot_feature_hierarchy` tiene un parametro `annotations` marcado como deprecated a favor de `layer_names`, pero no emite warning. Deberia usar `warnings.warn`.
- `plot_single_feature_overlay` importa `cv2` dentro del metodo (linea 371) lo cual es correcto pero inconsistente con el import de numpy/matplotlib que es a nivel modulo.
- Los imports de `select_top_channels_by_variance` y `normalize_feature_map` son inline dentro de metodos, lo cual evita dependencias circulares pero dificulta el analisis estatico.

---

### 7. gradcam.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/visualization/gradcam.py`
- **Lineas/Tamano**: 376 lineas / 11.9 KB
- **Proposito**: Implementacion de Grad-CAM (Gradient-weighted Class Activation Mapping) para explicabilidad del clasificador. Modulo central de interpretabilidad.
- **Contenido clave**:
  - `TARGET_LAYER_MAP`: Mapeo de backbone -> capa target para 7 arquitecturas (resnet18/50, densenet121, efficientnet_b0, vgg16, alexnet, mobilenet_v2)
  - `get_target_layer(model, backbone_name, layer_name)`: Auto-deteccion o seleccion manual de capa target
  - `_get_layer_by_name(model, layer_path)`: Navegacion por path con soporte de indices numericos para Sequential
  - `GradCAM`: Clase principal
    - Registra forward hook (captura activaciones) y full backward hook (captura gradientes)
    - `__call__(input_tensor, target_class)`: Genera heatmap via: forward -> backward -> weighted combination -> ReLU -> interpolacion a tamanio input -> normalizacion
    - Soporte de context manager (`with GradCAM(...) as gradcam:`) para cleanup automatico de hooks
    - Validacion robusta de input (4D tensor, batch size 1, rango de target_class)
  - `calculate_pfs(heatmap, mask)`: Calcula Pulmonary Focus Score = sum(heatmap * mask) / sum(heatmap). Manejo de mascaras RGB, normalizacion, resize
  - `overlay_heatmap(image, heatmap, alpha, colormap)`: Mezcla imagen original con heatmap coloreado
  - `create_gradcam_visualization(image, heatmap, prediction, confidence, true_label)`: Crea overlay con anotaciones de texto (prediccion, GT, confianza)
- **Dependencias**:
  - Importa: torch, numpy, PIL, matplotlib.cm, cv2 (lazy)
  - Importado por: `__init__.py`, `pfs_analysis.py`, `src_v2/cli.py` (comandos generate-gradcam y pfs-analysis), scripts de visualizacion
- **Importancia**: CRITICO
- **Justificacion**: Componente central de interpretabilidad del clasificador. Usado extensivamente en el CLI y por scripts. Implementacion correcta del algoritmo Grad-CAM con buenas practicas (context manager, validaciones, epsilon en normalizacion). La funcion `calculate_pfs` es usada por el modulo PFS para la metrica cuantitativa de atencion pulmonar.

**Observaciones**:
- Usa `register_full_backward_hook` (linea 120) en lugar del deprecated `register_backward_hook`, lo cual es correcto para PyTorch >= 1.8.
- `overlay_heatmap` usa `getattr(cm, colormap, cm.jet)` (linea 323) lo cual es una forma fragil de acceder al colormap. Deberia usar `plt.get_cmap(colormap)` que es la API oficial.
- `create_gradcam_visualization` importa `cv2` dentro de la funcion, lo cual es correcto dado que solo se usa cuando se llama explicitamente.

---

### 8. pfs_analysis.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/visualization/pfs_analysis.py`
- **Lineas/Tamano**: 630 lineas / 21.7 KB
- **Proposito**: Analisis completo de Pulmonary Focus Score (PFS): recoleccion de resultados, estadisticas per-class y correct/incorrect, visualizaciones, y guardado de muestras de bajo PFS con GradCAM overlay.
- **Contenido clave**:
  - `PFSResult`: Dataclass para resultado individual (image_path, clases, confianza, PFS, correcto)
  - `PFSSummary`: Dataclass para estadisticas globales (mean/std/median PFS, por clase, correct vs incorrect, low PFS count)
  - `PFSAnalyzer`: Clase principal que acumula resultados
    - `add_result()`, `add_results()`: Acumulacion incremental
    - `get_summary()`: Calcula todas las estadisticas
    - `get_low_pfs_results()`: Filtrado por threshold
    - `save_reports()`: Genera JSON summary, CSV details, CSV por clase, CSV de low PFS samples
  - `load_lung_mask(mask_path)`: Carga mascara desde PNG, convierte RGB a grayscale, normaliza a [0,1]
  - `find_mask_for_image(image_path, mask_dir, class_name)`: Busca mascara en 8 posibles paths (incluyendo sufijo `_mask.png`)
  - `generate_approximate_mask(image_shape, margin)`: Genera mascara rectangular aproximada (util cuando no hay mascaras reales)
  - `run_pfs_analysis(model, dataloader, ...)`: Pipeline completo que itera sobre batches, genera GradCAM, busca mascaras, calcula PFS. Soporta limitacion de muestras por clase y mascaras aproximadas
  - `create_pfs_visualizations(detailed_results, output_dir, summary)`: 4 figuras matplotlib (distribucion PFS, PFS por clase, PFS vs confianza scatter, correct vs incorrect comparison)
  - `save_low_pfs_gradcam_samples(detailed_results, output_dir, threshold, max_samples)`: Guarda las peores muestras con overlay GradCAM
- **Dependencias**:
  - Importa: `src_v2.visualization.gradcam` (GradCAM, get_target_layer, calculate_pfs, overlay_heatmap), torch, numpy, PIL, matplotlib (lazy), csv, json
  - Importado por: `__init__.py`, `src_v2/cli.py` (comando pfs-analysis)
- **Importancia**: ALTO
- **Justificacion**: Modulo critico para validar que el clasificador se enfoca en regiones pulmonares (relevancia clinica). Bien estructurado con dataclasses, acumulacion incremental, y multiples formatos de salida. Integrado con el CLI y con GradCAM.

**Observaciones**:
- `run_pfs_analysis` procesa una imagen a la vez (requisito de GradCAM batch=1), lo cual es lento pero necesario.
- El calculo de `max_per_class` (linea 369) usa division entera que podria ser imprecisa: `num_samples // len(class_names)` no distribuye correctamente si hay residuo. Pero es una limitacion menor.
- `run_pfs_analysis` accede a `dataset.samples[start_idx + i][0]` (linea 387) para obtener paths, lo cual depende de que el DataLoader no haga shuffle y de que `dataset.samples` exista (especifico a `ImageFolder`).

---

### 9. plot_failure_cases.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/visualization/plot_failure_cases.py`
- **Lineas/Tamano**: 775 lineas / 24.3 KB
- **Proposito**: Script standalone (con `main()` y `__main__` guard) que genera paneles de publicacion con casos de fallo del clasificador: imagen + barras de probabilidad + landmarks + flechas de region de interes.
- **Contenido clave**:
  - `FailureCase`: Dataclass frozen con probabilities, confidence, margin, y propiedad `pair` para identificar par de confusion
  - `parse_args()`: CLI robusto con 22+ argumentos (classifier, data-dir, split, output, pairs, selection-mode, seed, sahs, landmarks, arrows, intensity-bar, scale-bar, etc.)
  - `resolve_device()`, `resolve_checkpoint()`: Helpers para localizacion de dispositivo y checkpoint
  - `load_landmarks()`, `load_annotations()`: Carga de datos auxiliares desde JSON
  - `apply_sahs_masked(image, threshold)`: Implementacion de SAHS (Statistical Adaptive Histogram Stretching) para mejorar contraste en region pulmonar. Usa 2.5*std_above y 2.0*std_below como limites
  - `remap_labels()`: Alinea indices de labels entre dataset y checkpoint (misma logica que en plot_roc_curves.py)
  - `collect_failure_cases(model, dataloader, dataset, class_names, device)`: Recolecta todos los casos mal clasificados con probabilidades completas
  - `select_cases_by_pair(failure_cases, pairs, cases_per_pair, mode, seed)`: Selecciona representativos por par de confusion (confidence, uncertainty, random)
  - `create_failure_case_figure(cases, ...)`: Genera la figura completa:
    - Layout adaptativo con gridspec (filas x 3 columnas)
    - Sub-gridspec por celda: imagen arriba, barras de probabilidad abajo
    - Panel labels tipo publicacion `(a)`, `(b)`, `(c)`, ...
    - Soporte de SAHS, landmarks, flechas (auto via gradiente o manuales), intensity bar, scale bar
  - `draw_probability_bars()`: Barras horizontales coloreadas (verde=true, rojo=predicted, gris=other)
  - `compute_auto_arrow()`: Calcula punta de flecha basada en gradiente de intensidad
  - `apply_publication_style()`: Estilo Times New Roman para figuras de publicacion
- **Dependencias**:
  - Importa: `src_v2.constants`, `src_v2.models` (create_classifier, get_classifier_transforms), matplotlib, torch, PIL, numpy, torchvision.datasets
  - Importado por: Nadie (script standalone, ejecutado via `python -m src_v2.visualization.plot_failure_cases`)
- **Importancia**: ALTO
- **Justificacion**: Herramienta sofisticada y altamente configurable para generar figuras de publicacion de los casos fallidos. La integracion de landmarks, SAHS, y flechas automaticas la hace particularmente util para analisis cualitativo en tesis. Calidad de publicacion con estilo serif.

**Observaciones**:
- `remap_labels()` esta duplicada casi identicamente en `plot_roc_curves.py`. Deberia extraerse a un modulo compartido.
- `resolve_device()` y `resolve_checkpoint()` tambien estan duplicadas con `plot_roc_curves.py`.
- La funcion `apply_sahs_masked` es una implementacion standalone de mejora de contraste que podria ser util en otros contextos pero esta acoplada a este script.
- No esta integrado con el CLI de `src_v2` (no hay comando correspondiente en `cli.py`). Se ejecuta como script independiente.

---

### 10. plot_roc_curves.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/visualization/plot_roc_curves.py`
- **Lineas/Tamano**: 1,027 lineas / 29.6 KB
- **Proposito**: Script standalone para generar curvas ROC multiclase (one-vs-rest) con tres backends de renderizado (manual matplotlib, scikit-learn RocCurveDisplay, Plotly), multiples layouts (row 1x3, grid 2x2), zoom inset, escala logaritmica, y panel de resumen micro/macro AUC.
- **Contenido clave**:
  - `RocPlotConfig`: Dataclass con toda la configuracion de estilo (colores, fontsize, line_width, grid, baseline)
  - `parse_args()`: CLI con 18+ argumentos (classifier, data-dir, split, output, backend, layout, tta, width-cm, height-cm, inset-zoom, log-fpr, etc.)
  - `resolve_device()`, `resolve_checkpoint()`: Duplicadas de plot_failure_cases.py
  - `load_dataset()`: Soporta splits individuales o `"all"` via ConcatDataset
  - `remap_labels()`: Alineacion de indices (duplicada)
  - `collect_predictions(model, dataloader, device, tta)`: Inferencia con soporte TTA (flip horizontal) y softmax
  - `compute_roc_curves(labels, probs, class_names)`: Calcula FPR/TPR/AUC per-class + micro/macro averages usando scikit-learn
  - `plot_roc_curves(...)`: Backend manual con matplotlib. Paneles con labels `(a)`, `(b)`, `(c)`, AUC annotation, aspect equal, estilo publicacion
  - `plot_roc_curves_sklearn(...)`: Usa `RocCurveDisplay` de scikit-learn. Funcionalmente casi identica a la manual
  - `plot_roc_curves_plotly(...)`: Backend interactivo con Plotly. Soporta HTML y imagen estatica. Usa `make_subplots`
  - `add_zoom_inset()`: Inset con zoom en region de alto TPR / bajo FPR usando `mpl_toolkits.axes_grid1`
  - `apply_log_fpr_axis()`: Escala logaritmica en FPR con ticks personalizados
  - `cm_to_inch()`, `get_log_fpr_values()`: Helpers
- **Dependencias**:
  - Importa: `src_v2.constants`, `src_v2.models`, sklearn.metrics, sklearn.preprocessing, matplotlib, torch, numpy, plotly (lazy)
  - Importado por: Nadie (script standalone)
- **Importancia**: ALTO
- **Justificacion**: Herramienta completa y altamente configurable para generar curvas ROC de calidad publicable. Los tres backends cubren diferentes necesidades (estatico para paper, interactivo para exploracion). El zoom inset y la escala logaritmica son features avanzados para analisis detallado.

**Observaciones**:
- **Duplicacion significativa**: `plot_roc_curves` y `plot_roc_curves_sklearn` comparten ~80% de codigo (styling, layout, annotations, axis configuration). Solo difieren en la linea de plotting (`ax.plot` vs `RocCurveDisplay.plot`). Deberian refactorizarse con una funcion comun que prepare los ejes y una estrategia para el plot en si.
- Las funciones `resolve_device`, `resolve_checkpoint`, `remap_labels`, `apply_publication_style` estan triplicadas entre este archivo, plot_failure_cases.py, y parcialmente en el CLI. Candidato claro a extraccion en un modulo `visualization.common`.
- El DPI por defecto es 600, que es muy alto para previsualizacion pero adecuado para publicacion.
- No esta integrado con el CLI de `src_v2`. Se ejecuta como script independiente.

---

### 11. scientific_viz.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/visualization/scientific_viz.py`
- **Lineas/Tamano**: 448 lineas / 16.0 KB
- **Proposito**: Generacion de visualizaciones cientificas de landmarks: radiografia original con triangulacion de Delaunay y puntos de landmark numerados. Incluye pipeline completo para generar un dataset de visualizacion con splits estratificados.
- **Contenido clave**:
  - `load_prediction_cache(npz_path)`: Carga cache NPZ de predicciones. Requiere claves `landmarks` y `image_paths`. Diferente formato que `comparison_viz.py` (no maneja `predictions`/`image_paths` alternativo)
  - `scale_landmarks_to_viz(landmarks_224, viz_size, model_size)`: Escalado lineal de 224x224 a 299x299
  - `create_scientific_visualization(image, landmarks, triangles, canonical_shape, ...)`: Crea figura matplotlib con:
    - Imagen en escala de grises
    - Malla de triangulacion cyan (alpha=0.6)
    - Circulos rojos numerados (1-15) con borde blanco
    - Leyenda profesional
    - `canonical_shape` se ignora (mantenido por compatibilidad)
  - `generate_viz_dataset(input_dir, output_dir, ...)`: Pipeline completo:
    - Carga predicciones, canonical shape, triangulacion
    - Crea splits usando `get_dataframe_splits()` (misma funcion que warping)
    - Para cada imagen: carga original 299x299, escala landmarks, genera viz
    - Guarda metadata.json detallado
    - Soporte de `max_per_split` para testing
  - Incluye bloque `__main__` con configuracion de ejemplo
- **Dependencias**:
  - Importa: `src_v2.data.dataset.get_dataframe_splits`, cv2, matplotlib, numpy, pandas, json, tqdm
  - Importado por: `__init__.py`, `src_v2/cli.py` (comando generate-landmark-visualization-dataset)
- **Importancia**: ALTO
- **Justificacion**: Modulo esencial para generar el dataset de visualizacion de landmarks que permite verificar visualmente la calidad de las predicciones del ensemble. Los splits son identicos a los del clasificador, permitiendo analisis alineado. Integrado con el CLI.

**Observaciones**:
- Docstrings y comentarios estan en espanol, a diferencia del resto del modulo que esta mayormente en ingles. Consistencia menor.
- `create_scientific_visualization` recibe `canonical_shape` como parametro pero lo ignora (linea 98: "IGNORADO - mantener por compatibilidad"). Deberia marcarse como deprecated o eliminarse.
- `load_prediction_cache` solo maneja formato `landmarks`/`image_paths`, mientras que `comparison_viz.load_predictions_mapping` maneja ambos formatos. Duplicacion parcial con manejo inconsistente de formatos.
- El bloque `__main__` en un modulo de libreria no es ideal. Deberia estar en un script separado.

---

### 12. utils.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/visualization/utils.py`
- **Lineas/Tamano**: 454 lineas / 13.4 KB
- **Proposito**: Funciones utilitarias compartidas por el subsistema de visualizacion: seleccion de canales, normalizacion de feature maps, grids, overlays, paletas de color, y la funcion critica `draw_scientific_crosses_on_image`.
- **Contenido clave**:
  - `select_top_channels_by_variance(feature_map, n_channels)`: Seleccion de canales mas informativos por varianza espacial
  - `select_top_channels_by_gradient(feature_map, gradients, n_channels)`: Seleccion por importancia de gradiente (para integracion con GradCAM)
  - `normalize_feature_map(feature_map, method, percentile)`: 3 metodos de normalizacion: min-max, percentile, z-score (con sigmoid)
  - `create_figure_grid(n_plots, n_cols, ...)`: Helper para crear grids de subplots
  - `resize_feature_map(feature_map, target_size)`: Resize per-channel usando cv2
  - `overlay_heatmap_on_image(image, heatmap, alpha, colormap)`: Overlay usando cv2.applyColorMap (diferente implementacion que gradcam.overlay_heatmap que usa matplotlib.cm)
  - `create_annotation_box(ax, text, position)`: Anotacion con bounding box en matplotlib
  - `get_color_palette(name)`: 3 paletas predefinidas ('scientific', 'vibrant', 'pastel')
  - `compute_activation_statistics(feature_map)`: Estadisticas detalladas incluyendo sparsity, canales muertos, per-channel means/stds/variances
  - `save_figure_with_metadata(fig, filepath, metadata)`: Guarda figura + JSON sidecar
  - `draw_scientific_crosses_on_image(image, landmarks, cross_size, thickness, color)`: **Funcion mas usada del modulo**. Dibuja cruces anti-aliased en posiciones de landmarks usando cv2.line. Soporta 8 colores predefinidos via mapa BGR. Usada por comparison_viz, CLI, y potencialmente otros scripts
- **Dependencias**:
  - Importa: numpy, torch, matplotlib, cv2, json (lazy en save_figure_with_metadata)
  - Importado por: `__init__.py` (condicional), `comparison_viz.py`, `feature_visualizer.py` (inline), `src_v2/cli.py`, scripts de glass box
- **Importancia**: ALTO
- **Justificacion**: Modulo de utilidades que contiene la funcion `draw_scientific_crosses_on_image` usada extensivamente en el pipeline de visualizacion de landmarks. Tambien proporciona toda la infraestructura de procesamiento de feature maps para el subsistema glass box.

**Observaciones**:
- Hay dos implementaciones de overlay de heatmap en el modulo de visualizacion: `gradcam.overlay_heatmap` (usa `matplotlib.cm`) y `utils.overlay_heatmap_on_image` (usa `cv2.applyColorMap`). Deberia consolidarse en una sola implementacion.
- `save_figure_with_metadata` (linea 349) asume que `filepath` es un string (usa `.replace('.png', ...)`), pero deberia funcionar con Path tambien. Bug potencial si se pasa un Path object.
- `draw_scientific_crosses_on_image` usa formato BGR para OpenCV pero el docstring dice "RGB tuple (B,G,R for OpenCV)" lo cual puede confundir. Los colores son correctos para OpenCV pero la imagen retornada esta en BGR, no RGB como sugiere el parametro `return_rgb`.

---

## Tabla Resumen

| # | Archivo | Lineas | KB | Importancia | Dependencias entrantes |
|---|---------|--------|----|-------------|----------------------|
| 1 | `__init__.py` | 90 | 2.5 | MEDIO | Punto de entrada del modulo |
| 2 | `comparison_viz.py` | 953 | 33.2 | ALTO | CLI (2 comandos), __init__ |
| 3 | `diagramming.py` | 307 | 9.1 | BAJO | Nadie |
| 4 | `error_analysis.py` | 478 | 16.2 | ALTO | CLI, __init__ |
| 5 | `feature_extractor.py` | 276 | 8.9 | MEDIO | __init__ (cond.), scripts |
| 6 | `feature_visualizer.py` | 453 | 15.8 | MEDIO | __init__ (cond.), scripts |
| 7 | `gradcam.py` | 376 | 11.9 | CRITICO | CLI (3+ usos), pfs_analysis, __init__, scripts |
| 8 | `pfs_analysis.py` | 630 | 21.7 | ALTO | CLI (1 comando), __init__ |
| 9 | `plot_failure_cases.py` | 775 | 24.3 | ALTO | Script standalone |
| 10 | `plot_roc_curves.py` | 1027 | 29.6 | ALTO | Script standalone |
| 11 | `scientific_viz.py` | 448 | 16.0 | ALTO | CLI (1 comando), __init__ |
| 12 | `utils.py` | 454 | 13.4 | ALTO | CLI, comparison_viz, feature_visualizer, __init__, scripts |

---

## Problemas Transversales

### 1. Duplicacion de codigo significativa
- **`resolve_device()`**: Implementada en `plot_failure_cases.py` (linea 195) y `plot_roc_curves.py` (linea 183). Codigo identico.
- **`resolve_checkpoint()`**: Implementada en ambos scripts de plotting. Codigo identico.
- **`remap_labels()`**: Implementada en `plot_failure_cases.py` (linea 310) y `plot_roc_curves.py` (linea 237). Casi identica.
- **`apply_publication_style()`**: Implementada en ambos scripts con configuraciones ligeramente diferentes.
- **`plot_roc_curves` vs `plot_roc_curves_sklearn`**: ~80% de codigo compartido dentro del mismo archivo.
- **`generate_comparison_dataset` vs `generate_comparison_dataset_aligned`**: ~200 lineas de logica de estadisticas duplicada.
- **`load_prediction_cache` (scientific_viz) vs `load_predictions_mapping` (comparison_viz)**: Funcionalidad similar, formatos diferentes.
- **`overlay_heatmap` (gradcam) vs `overlay_heatmap_on_image` (utils)**: Dos implementaciones del mismo concepto.

**Recomendacion**: Extraer las funciones duplicadas a un modulo `visualization.common` o directamente a `visualization.utils`.

### 2. Inconsistencia de idioma
- `scientific_viz.py` tiene docstrings y logs en espanol
- `comparison_viz.py` tiene docstrings en ingles
- `gradcam.py`, `error_analysis.py` tienen todo en ingles
- Los nombres de funciones estan consistentemente en ingles

### 3. Scripts standalone no integrados con CLI
- `plot_failure_cases.py` y `plot_roc_curves.py` son scripts standalone con `main()` + `if __name__ == "__main__"` pero no estan registrados como comandos en `cli.py`. Podrian integrarse como subcomandos del CLI principal para mantener una interfaz unificada.

### 4. Imports inline vs top-level
- Patron inconsistente: algunos archivos importan matplotlib/cv2 a nivel modulo, otros hacen imports lazy dentro de funciones. La practica de lazy imports es buena para dependencias opcionales (plotly, pydot, bertviz) pero inconsistente para dependencias core como matplotlib.

### 5. diagramming.py: funcionalidad irrelevante
- Las funciones de Keras (`save_keras_model_diagram`) y BertViz (`save_bertviz_attention_html`) no son relevantes para este proyecto PyTorch. Podrian eliminarse o mantenerse como utilidades genericas en un paquete separado.

---

## Metricas del Modulo

| Metrica | Valor |
|---------|-------|
| Total lineas | 6,267 |
| Total tamano | ~207 KB |
| Archivos CRITICO | 1 (gradcam.py) |
| Archivos ALTO | 7 (comparison_viz, error_analysis, pfs_analysis, plot_failure_cases, plot_roc_curves, scientific_viz, utils) |
| Archivos MEDIO | 3 (__init__, feature_extractor, feature_visualizer) |
| Archivos BAJO | 1 (diagramming) |
| Archivos ELIMINABLE | 0 |
| Funciones duplicadas | ~6 funciones con duplicacion parcial o total |
| Tests existentes | No se detectaron tests especificos para el modulo de visualizacion |

---

## Recomendaciones Priorizadas

1. **Extraer utilidades duplicadas** (`resolve_device`, `resolve_checkpoint`, `remap_labels`, `apply_publication_style`) a `visualization/common.py` o directamente a `utils.py`.

2. **Consolidar implementaciones de overlay** (`gradcam.overlay_heatmap` vs `utils.overlay_heatmap_on_image`) en una unica funcion en `utils.py`.

3. **Consolidar funciones de carga de predicciones** (`scientific_viz.load_prediction_cache` vs `comparison_viz.load_predictions_mapping`) en una funcion robusta que maneje ambos formatos NPZ.

4. **Integrar scripts standalone con el CLI**: Registrar `plot_failure_cases` y `plot_roc_curves` como subcomandos en `cli.py` para una interfaz unificada.

5. **Agregar tests unitarios**: El modulo carece de tests. Priorizar tests para `gradcam.py` (calculate_pfs, overlay_heatmap), `error_analysis.py` (ErrorAnalyzer), y `utils.py` (draw_scientific_crosses_on_image, normalize_feature_map).

6. **Limpiar `diagramming.py`**: Eliminar las funciones de Keras y BertViz que no se usan, o mover a un paquete de utilidades genericas fuera de `src_v2`.

7. **Corregir bug potencial en `utils.save_figure_with_metadata`**: Cambiar `filepath.replace('.png', ...)` para soportar Path objects.
