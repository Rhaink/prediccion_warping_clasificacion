# 12. Pipeline Scripts

Analisis de los scripts del pipeline principal y soporte directo.

**Archivos analizados**: 23

---

## Resumen Ejecutivo

Los scripts del pipeline se dividen en varias capas:
1. **Scripts criticos del pipeline actual** (documentados en CLAUDE.md): `predict_landmarks_dataset.py`, `evaluate_ensemble_from_config.py`, `quickstart_warping.sh`, `quickstart_landmarks.sh`
2. **Scripts de entrenamiento**: `train.py`, `train_classifier.py`, `train_hierarchical.py`
3. **Scripts legacy/session-specific**: `gpa_analysis.py`, `piecewise_affine_warp.py`, `extract_predictions.py`, `predict.py` -- Estos han sido reemplazados por `src_v2` pero se mantienen
4. **Scripts de soporte/analisis**: `analyze_data.py`, `analyze_hospital_marks.py`, `landmark_connections.py`, etc.
5. **Shell scripts de automatizacion**: `run_seed_sweep.sh`, `run_classifier_sweep_accuracy.sh`, etc.

**Hallazgo principal**: Existe duplicacion significativa entre scripts standalone (Session 19/20) y los modulos equivalentes en `src_v2/processing/`. El pipeline actual documentado en CLAUDE.md usa `src_v2` como fuente canonica, haciendo que `gpa_analysis.py` y `piecewise_affine_warp.py` sean legacy. Sin embargo, `landmark_connections.py` todavia es importado como dependencia por `piecewise_affine_warp.py`.

---

## Analisis Detallado

### 1. predict_landmarks_dataset.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/predict_landmarks_dataset.py`
- **Lineas/Tamano**: 442 lineas / 16 KB
- **Proposito**: Genera predicciones de landmarks para el dataset completo usando el ensemble de modelos, con soporte de TTA y CLAHE. Guarda cache en formato JSON o NPZ.
- **Contenido clave**:
  - `parse_args()`: CLI extenso con overrides para TTA, CLAHE, batch-size, seed, limit
  - `detect_architecture_from_checkpoint()`: Auto-detecta arquitectura (coord_attention, deep_head, hidden_dim) desde state_dict
  - `preprocess_image()`: Preprocessing completo (resize, CLAHE, RGB, ImageNet normalization)
  - `predict_ensemble()`: Funcion inner con TTA (flip horizontal + symmetric pairs swap)
  - `collect_images()`: Escanea dataset por clase (COVID/Normal/Viral Pneumonia)
  - `main()`: Loop de batches con inferencia + guardado en JSON o NPZ con metadata completa
  - Soporta `--ensemble-config` para leer modelos desde configs/ensemble_best.json
- **Dependencias**: cv2, torch, numpy, tqdm, src_v2.constants (DEFAULT_IMAGE_SIZE, CLAHE params, IMAGENET_MEAN/STD, NUM_LANDMARKS, SYMMETRIC_PAIRS), src_v2.models.create_model
- **Referenciado en CLAUDE.md**: Si (paso 2 del pipeline, documentado como comando principal)
- **Importancia**: CRITICO
- **Justificacion**: Es el paso de cache de predicciones que alimenta el warping. Sin este script no se puede generar el dataset warped sin re-ejecutar inferencia. Documentado explicitamente en CLAUDE.md como parte del pipeline principal.

**Observaciones**:
- Buena gestion de metadata (schema_version, timestamp, configuracion completa)
- Duplicacion de logica con predict.py (EnsemblePredictor) pero esta version es mas robusta (batching, auto-detect arch, config-based)
- El procesamiento de imagenes difiere ligeramente del de `src_v2/data/transforms.py` (aqui se hace manualmente con cv2, no usa las transforms del pipeline de training). Esto podria causar discrepancias sutiles si las transforms cambian.
- `datetime.utcnow()` esta deprecado en Python 3.12+; deberia usar `datetime.now(timezone.utc)`

---

### 2. evaluate_ensemble_from_config.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/evaluate_ensemble_from_config.py`
- **Lineas/Tamano**: 48 lineas / 4 KB
- **Proposito**: Wrapper que lee un JSON de configuracion de ensemble y delega la evaluacion al CLI `python -m src_v2 evaluate-ensemble`.
- **Contenido clave**:
  - Lee config JSON y extrae lista de modelos
  - Valida existencia de checkpoints
  - Construye y ejecuta `subprocess.run` con flags TTA/CLAHE del config
  - Soporte para guardar output en archivo log
- **Dependencias**: json, subprocess, sys, pathlib
- **Referenciado en CLAUDE.md**: Si (paso 6 del pipeline, `python scripts/evaluate_ensemble_from_config.py --config configs/ensemble_best.json`)
- **Importancia**: ALTO
- **Justificacion**: Parte del pipeline documentado para evaluar el ensemble. Es un wrapper delgado sobre el CLI, util para reproducibilidad con configs.

**Observaciones**:
- Muy limpio y minimalista. Bien diseñado como thin wrapper.
- No pasa parametros CLAHE extras (clip, tile) al CLI -- solo `--clahe` boolean. Esto podria ser una limitacion si el config especifica tile_size diferente.

---

### 3. train.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/train.py`
- **Lineas/Tamano**: 429 lineas / 16 KB
- **Proposito**: Script principal de entrenamiento para modelos de prediccion de landmarks. Implementa el flujo completo de dos fases (frozen backbone + fine-tuning).
- **Contenido clave**:
  - `parse_args()`: 40+ argumentos CLI con soporte de JSON config override
  - `set_global_seed()`: Semilla global con soporte deterministic mode
  - `setup_device()`: Auto-deteccion GPU/CPU
  - `create_loss_function()`: Factory para WingLoss, WeightedWingLoss, CombinedLandmarkLoss
  - `main()`: Flujo completo -- dataloaders, modelo, loss, trainer, evaluacion final, guardado de history/config/report
  - Soporte phase1-only, phase2-only, checkpoint resume
  - Category weights (COVID oversampling)
- **Dependencias**: torch, numpy, json, src_v2.data.dataset.create_dataloaders, src_v2.models.resnet_landmark, src_v2.models.losses, src_v2.training.trainer.LandmarkTrainer, src_v2.evaluation.metrics
- **Referenciado en CLAUDE.md**: No directamente (se documenta como `python -m src_v2 train`), pero es usado por quickstart_landmarks.sh y run_seed_sweep.sh
- **Importancia**: ALTO
- **Justificacion**: Aunque el CLI tiene `train`, este script es el que se usa en los shell scripts de automatizacion. Es la interfaz principal para entrenar modelos de landmarks individualmente.

**Observaciones**:
- Buen soporte de config JSON + CLI override pattern
- El default de `--clahe-tile` es 8, mientras que CLAUDE.md documenta que tile_size=4 es mejor. El config `landmarks_train_base.json` probablemente corrige esto, pero el default del CLI podria inducir a error.
- `parse_known_args()` seguido de `parse_args()` para el patron config-then-CLI es correcto pero un poco fragil.

---

### 4. train_classifier.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/train_classifier.py`
- **Lineas/Tamano**: 25 lineas / 4 KB
- **Proposito**: Wrapper minimo que delega al CLI `python -m src_v2 train-classifier`. Mantiene compatibilidad hacia atras.
- **Contenido clave**: Simplemente ejecuta `subprocess.call` pasando todos los argumentos al CLI
- **Dependencias**: subprocess, sys, pathlib
- **Referenciado en CLAUDE.md**: No (se documenta el CLI directamente)
- **Importancia**: BAJO
- **Justificacion**: Puro wrapper sin logica propia. Util solo para backward compat. Podria eliminarse si todos los scripts se actualizan para usar el CLI directamente.

---

### 5. train_hierarchical.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/train_hierarchical.py`
- **Lineas/Tamano**: 270 lineas / 12 KB
- **Proposito**: Entrenamiento del modelo jerarquico de landmarks (HierarchicalLandmarkModel). Alternativa al ResNet18Landmarks estandar.
- **Contenido clave**:
  - `train_epoch()`: Entrenamiento con main_loss + axis_loss (ponderada)
  - `validate()`: Validacion con error en pixeles
  - `main()`: Flujo two-phase con CosineAnnealingLR en phase2
  - Soporte config JSON, early stopping manual
  - Evaluacion final con TTA
- **Dependencias**: torch, src_v2.data.dataset, src_v2.models.hierarchical (HierarchicalLandmarkModel, AxisLoss), src_v2.models.losses.WingLoss, src_v2.evaluation.metrics
- **Referenciado en CLAUDE.md**: No (modelo mencionado como alternativa en `src_v2/models/hierarchical.py`)
- **Importancia**: MEDIO
- **Justificacion**: Es un modelo experimental alternativo. No es parte del pipeline principal (el ensemble usa ResNet18Landmarks), pero se mantiene como opcion. Tiene implementacion propia del training loop en lugar de usar LandmarkTrainer.

**Observaciones**:
- Duplicacion del training loop -- no reutiliza LandmarkTrainer de `src_v2/training/trainer.py`
- Hardcodea paths relativos: `data/coordenadas/coordenadas_maestro.csv`, `data`
- El default `--clahe` es `action="store_true", default=True` lo cual hace que CLAHE siempre este habilitado sin posibilidad de desactivarlo facilmente
- `clahe_tile_size=4` hardcodeado correctamente (coincide con best practice)
- Usa `random_state=args.seed` para split, no tiene `--split-seed` separado. Esto difiere del patron de train.py donde training seed y split seed son independientes.

---

### 6. predict.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/predict.py`
- **Lineas/Tamano**: 331 lineas / 12 KB
- **Proposito**: Script de inferencia para prediccion de landmarks en imagenes individuales. Incluye clase EnsemblePredictor y visualizacion.
- **Contenido clave**:
  - `EnsemblePredictor`: Clase con init (carga modelos), preprocess (CLAHE en LAB), predict (ensemble + TTA), predict_batch
  - DEFAULT_MODEL_PATHS: Solo 2 modelos (simplificado)
  - `visualize_prediction()`: Dibuja landmarks coloreados por grupo anatomico
  - `main()`: CLI para imagen individual con output JSON y visualizacion
- **Dependencias**: torch, PIL, cv2, numpy, src_v2.models.resnet_landmark.create_model
- **Referenciado en CLAUDE.md**: Si (marcado como "Old prediction wrapper" en seccion Legacy)
- **Importancia**: BAJO
- **Justificacion**: Marcado explicitamente como legacy en CLAUDE.md. Reemplazado por `predict_landmarks_dataset.py` para uso en pipeline y el CLI `python -m src_v2 predict` para uso interactivo.

**Observaciones**:
- CLAHE se aplica en espacio LAB (diferente de predict_landmarks_dataset.py que usa grayscale directamente). Esta inconsistencia podria dar resultados ligeramente diferentes.
- Hardcodea hidden_dim=768, deep_head=True, coord_attention=True -- no auto-detecta desde checkpoint
- El docstring referencia "3.71 px" (Sesion 43) que ya no es el best (3.61 px en GROUND_TRUTH.json)
- DEFAULT_MODEL_PATHS solo tiene 2 de los 4 modelos del ensemble

---

### 7. extract_predictions.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/extract_predictions.py`
- **Lineas/Tamano**: 854 lineas / 32 KB
- **Proposito**: Extrae predicciones del ensemble para el test set, genera triangulacion de Delaunay, metricas, y visualizaciones.
- **Contenido clave**:
  - `load_model()` / `load_ensemble()`: Carga modelos con arch hardcodeada (hidden_dim=768)
  - `predict_with_tta()` / `predict_ensemble_tta()`: TTA con flip + symmetric pairs
  - `extract_all_predictions()`: Loop sobre test_loader con ground truth comparison
  - `save_predictions_csv/json/npz()`: Guardado en 3 formatos
  - `compute_delaunay_triangulation()` / `get_canonical_triangulation()`: Triangulacion sobre mean landmarks
  - `compute_delaunay_metrics()`: Metricas de areas de triangulos por imagen/categoria
  - Visualizaciones: `visualize_delaunay()`, `visualize_delaunay_comparison()`, `generate_category_visualizations()`, `generate_area_heatmap()`
  - Soporte config JSON y CLI overrides
- **Dependencias**: torch, numpy, pandas, scipy.spatial.Delaunay, matplotlib, PIL, src_v2.data.dataset, src_v2.models.resnet_landmark, src_v2.data.utils
- **Referenciado en CLAUDE.md**: No directamente (pero referenciado en quickstart_landmarks.sh)
- **Importancia**: MEDIO
- **Justificacion**: Util para analisis detallado del ensemble (predicciones + triangulacion + visualizaciones). No es parte del pipeline principal de produccion (que usa predict_landmarks_dataset.py + generate-dataset CLI). Es usado por quickstart_landmarks.sh.

**Observaciones**:
- Archivo muy grande (854 lineas). Mezcla funcionalidad de prediccion, I/O, triangulacion, y visualizacion. Candidato a refactorizacion.
- Hardcodea arquitectura del modelo (hidden_dim=768, etc.) en lugar de auto-detectar
- Escala landmarks 224->299 en visualizaciones (linea 558-559) asumiendo imagen original de 299px -- esto es fragil
- El DEFAULT_ENSEMBLE_CHECKPOINTS usa 4 modelos (session10/session13) que ya no son el best ensemble (CLAUDE.md documenta seed666 combo)
- Duplica logica de TTA que existe en src_v2/evaluation/metrics.py

---

### 8. extract_dataset_splits.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/extract_dataset_splits.py`
- **Lineas/Tamano**: 506 lineas / 20 KB
- **Proposito**: Extrae los splits exactos train/val/test del modelo warped_lung_best, copiando imagenes originales y warped a un directorio estructurado para uso en GUI.
- **Contenido clave**:
  - `create_directory_structure()`: Crea arbol original/warped x train/val/test x 3 categorias
  - `copy_image()`: Copia con verificacion MD5 opcional
  - `process_split()`: Lee CSV del warped dataset, copia imagenes originales y warped
  - `generate_dataset_info()`: Metadata JSON con configuracion completa
  - `generate_readme()`: README.md con documentacion del dataset extraido
  - Verificacion de integridad por checksum
- **Dependencias**: pandas, shutil, hashlib, tqdm, pathlib
- **Referenciado en CLAUDE.md**: No
- **Importancia**: MEDIO
- **Justificacion**: Script utilitario para preparar datos para la GUI. No es parte del pipeline de entrenamiento/evaluacion, pero si del despliegue. Genera un dataset auto-documentado.

**Observaciones**:
- Bien diseñado con verificacion de integridad
- Referencia hardcodeada a "98.05% accuracy" que ya no es el best (99.10% es warped_96)
- Genera README.md y metadata -- buena practica para trazabilidad
- Emojis en output (unicode characters) que podrian fallar en algunos terminales

---

### 9. sweep_ensemble_combos.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/sweep_ensemble_combos.py`
- **Lineas/Tamano**: 76 lineas / 4 KB
- **Proposito**: Prueba todas las combinaciones de k modelos de un pool y encuentra la mejor combinacion de ensemble.
- **Contenido clave**:
  - `parse_error()`: Extrae metrica de error desde output del CLI usando regex
  - `evaluate_combo()`: Ejecuta `python -m src_v2 evaluate-ensemble` via subprocess
  - `main()`: Genera combinaciones con itertools, evalua cada una, ordena por error, reporta mejor
- **Dependencias**: itertools, subprocess, sys, pathlib, re
- **Referenciado en CLAUDE.md**: No directamente (pero usado por run_seed_sweep.sh)
- **Importancia**: MEDIO
- **Justificacion**: Script clave para la busqueda del mejor ensemble. Fue instrumental para encontrar el ensemble de 3.61 px. No se ejecuta frecuentemente pero es necesario cuando se añaden nuevos modelos.

**Observaciones**:
- Limpio y funcional. Buen diseño minimalista.
- Delega toda la evaluacion al CLI, evitando duplicacion
- No tiene timeout por evaluacion -- si un modelo falla, el script se cuelga

---

### 10. compute_cv_test_aggregated_metrics.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/compute_cv_test_aggregated_metrics.py`
- **Lineas/Tamano**: 159 lineas / 8 KB
- **Proposito**: Agrega metricas de validacion cruzada (5-fold) en el test set del clasificador. Calcula mean+-std de accuracy, F1, matrices de confusion sumadas.
- **Contenido clave**:
  - `load_fold_test_results()`: Lee test_results.json de cada fold
  - `compute_aggregated_metrics()`: Estadisticas globales, por clase, confusion matrix sumada, best fold
  - `main()`: Hardcodea `base_dir = Path("outputs/classifier_cv")`
- **Dependencias**: json, numpy, pathlib
- **Referenciado en CLAUDE.md**: No
- **Importancia**: BAJO
- **Justificacion**: Script post-hoc para analisis de CV. Solo util cuando se ejecuta cross-validation del clasificador. El path de entrada esta hardcodeado, limitando reutilizacion.

**Observaciones**:
- Hardcodea base_dir sin argparse -- deberia ser parametrizable
- Buena estructura de output JSON con estadisticas completas
- Falta de argparse es inconsistente con el resto de scripts

---

### 11. gpa_analysis.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/gpa_analysis.py`
- **Lineas/Tamano**: 738 lineas / 28 KB
- **Proposito**: Implementacion standalone de Generalized Procrustes Analysis (GPA) para calcular la forma canonica de 15 landmarks. Incluye PCA sobre formas alineadas.
- **Contenido clave**:
  - Parte 1: Funciones base (center_shape, scale_shape, optimal_rotation_matrix, align_shape, procrustes_distance)
  - Parte 2: `gpa_iterative()` -- GPA iterativo con convergencia
  - Parte 3: `scale_canonical_to_image()`, `save_canonical_shape()` -- conversion y guardado
  - Parte 4: Visualizaciones (convergencia, forma canonica, formas alineadas)
  - Parte 5: PCA informativo (eigenvalues, eigenvectors, scatter por categoria)
  - `main()`: Carga NPZ, ejecuta GPA, PCA, genera visualizaciones
- **Dependencias**: numpy, matplotlib, json (NO importa de src_v2)
- **Referenciado en CLAUDE.md**: No directamente (el pipeline usa `python -m src_v2 compute-canonical`)
- **Importancia**: BAJO
- **Justificacion**: Script de Session 19, reemplazado por `src_v2/processing/gpa.py::gpa_iterative()`. Las funciones de GPA fueron migradas a src_v2. Se mantiene por referencia historica y porque `verify_gpa_correctness.py` y visualizacion scripts lo importan.

**Observaciones**:
- DUPLICACION SIGNIFICATIVA con `src_v2/processing/gpa.py`. La implementacion de GPA aqui es esencialmente la misma.
- Importado por scripts de verificacion y visualizacion como modulo (`from scripts.gpa_analysis import ...`)
- Data path hardcodeado a `outputs/predictions/all_landmarks.npz` -- formato legacy
- Buena documentacion pero deberia considerarse eliminacion con migracion de dependientes a src_v2

---

### 12. piecewise_affine_warp.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/piecewise_affine_warp.py`
- **Lineas/Tamano**: 791 lineas / 28 KB
- **Proposito**: Implementacion standalone de warping piecewise affine para normalizacion geometrica. Session 20.
- **Contenido clave**:
  - Parte 1: Carga de datos (canonical_shape, delaunay_triangles, test_predictions)
  - Parte 2: `piecewise_affine_warp()` -- Warping core con boundary points, Delaunay, triangle masks, affine per triangle
  - `add_boundary_points()`: 8 puntos (4 esquinas + 4 midpoints)
  - `warp_triangle()`: In-place warping de un triangulo usando bounding box optimization
  - Parte 3: `normalize_image_geometry()` -- API de alto nivel
  - Parte 4: Visualizaciones detalladas
  - Parte 5: main() -- Demo con test images, fill rate stats, grilla comparativa
- **Dependencias**: numpy, cv2, scipy.spatial.Delaunay, matplotlib, `scripts.landmark_connections` (EJE_CENTRAL, PULMON_IZQUIERDO, PULMON_DERECHO)
- **Referenciado en CLAUDE.md**: No directamente (el pipeline usa `src_v2/processing/warp.py`)
- **Importancia**: BAJO
- **Justificacion**: Script de Session 20, reemplazado por `src_v2/processing/warp.py::piecewise_affine_warp()`. Sin embargo, es importado por scripts legacy (`generate_warped_dataset.py`, `generate_full_warped_dataset.py`, `benchmark_inference.py`, etc.).

**Observaciones**:
- DUPLICACION SIGNIFICATIVA con `src_v2/processing/warp.py`. El core de warping fue migrado.
- Importa `landmark_connections.py` via `sys.path` -- dependencia fragil
- Todavia referenciado por scripts legacy y de benchmark. Migracion pendiente de dependientes.
- La logica de `warp_triangle()` aqui usa bounding box ROI optimization que tambien esta en src_v2

---

### 13. landmark_connections.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/landmark_connections.py`
- **Lineas/Tamano**: 78 lineas / 4 KB
- **Proposito**: Define las conexiones anatomicas entre los 15 landmarks para visualizacion (eje central, contornos pulmonares).
- **Contenido clave**:
  - `EJE_CENTRAL = [0, 8, 9, 10, 1]` -- L1 -> L9 -> L10 -> L11 -> L2
  - `PULMON_IZQUIERDO = [0, 11, 2, 4, 6, 13, 1]` -- Contorno izquierdo
  - `PULMON_DERECHO = [0, 12, 3, 5, 7, 14, 1]` -- Contorno derecho
  - `TODAS_CONEXIONES`, `COLORES_CONEXIONES`
  - `plot_landmark_connections()`: Funcion de visualizacion con matplotlib
- **Dependencias**: numpy (solo en __main__), matplotlib (en plot function)
- **Referenciado en CLAUDE.md**: No
- **Importancia**: MEDIO
- **Justificacion**: Modulo utilitario importado por `piecewise_affine_warp.py` y potencialmente otros scripts de visualizacion. Define constantes anatomicas que no estan duplicadas en src_v2 (src_v2/constants.py tiene SYMMETRIC_PAIRS y CENTRAL_LANDMARKS pero no las conexiones de contorno).

**Observaciones**:
- Este modulo llena un gap -- las conexiones anatomicas para visualizacion no estan en src_v2/constants.py
- Deberia considerarse migrar estas constantes a `src_v2/constants.py` para centralizar definiciones anatomicas
- Los indices de PULMON_IZQUIERDO y PULMON_DERECHO deberian verificarse contra la documentacion de landmarks (L12 es indice 11, etc.)

---

### 14. analyze_data.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/analyze_data.py`
- **Lineas/Tamano**: 156 lineas / 8 KB
- **Proposito**: Analisis exploratorio del dataset de landmarks: distribucion, estadisticas de coordenadas, simetria, alineacion.
- **Contenido clave**:
  - Distribucion por categoria
  - Verificacion de existencia de imagenes
  - Estadisticas de coordenadas por landmark (variabilidad, dificultad)
  - Error de simetria por par bilateral
  - Rango de coordenadas
  - Alineacion de landmarks centrales con eje L1-L2
- **Dependencias**: numpy, pandas, PIL, matplotlib, src_v2.data.utils (load_coordinates_csv, get_image_path, get_landmarks_array, compute_statistics, compute_symmetry_error, constantes)
- **Referenciado en CLAUDE.md**: No
- **Importancia**: BAJO
- **Justificacion**: Script de analisis exploratorio one-shot. Util en fase inicial del proyecto para entender el dataset. No se ejecuta como parte del pipeline de produccion.

---

### 15. analyze_hospital_marks.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/analyze_hospital_marks.py`
- **Lineas/Tamano**: 430 lineas / 16 KB
- **Proposito**: Comparacion visual de imagenes originales vs warped para verificar si el warping elimina marcas hospitalarias (texto, logos en esquinas).
- **Contenido clave**:
  - `collect_sample_images()`: Muestrea imagenes por clase, encuentra pares original/warped
  - `create_comparison_figure()`: Side-by-side con anotaciones de regiones criticas
  - `create_mosaic_figure()`: Grid de comparaciones
  - `analyze_regional_intensities()`: Estadisticas de intensidad en regiones de marcas
  - Regiones definidas: top_left, top_right, bottom_left, bottom_right (esquinas tipicas)
- **Dependencias**: cv2, matplotlib, numpy, src_v2.processing.warp.compute_fill_rate
- **Referenciado en CLAUDE.md**: No
- **Importancia**: BAJO
- **Justificacion**: Script de analisis visual para la tesis. One-shot, no parte del pipeline. Verifica hipotesis especifica sobre warping y marcas hospitalarias.

**Observaciones**:
- Referencia a `WARPED_DATASET_DIR` y `FULL_COVERAGE_WARPED_DIR` que son paths de datasets legacy
- Unico import de src_v2 es `compute_fill_rate`

---

### 16. run_demo.py
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/run_demo.py`
- **Lineas/Tamano**: 270 lineas / 8 KB
- **Proposito**: Launcher para la GUI de demostracion (Gradio). Soporta modo desarrollo y PyInstaller standalone.
- **Contenido clave**:
  - `get_base_path()`: Detecta si es PyInstaller frozen o desarrollo
  - `check_dependencies()`: Verifica torch, gradio, numpy, cv2, matplotlib, pandas, PIL
  - `check_models()`: Verifica existencia de landmark models, canonical shape, triangulation, classifier
  - `print_device_info()`: Info de GPU
  - `main()`: Parsea args (--share, --port, --host), crea demo via `src_v2.gui.app.create_demo()`, lanza Gradio
- **Dependencias**: src_v2.gui.config (LANDMARK_MODELS, etc.), src_v2.gui.app.create_demo, gradio (indirecto)
- **Referenciado en CLAUDE.md**: No
- **Importancia**: ALTO
- **Justificacion**: Punto de entrada principal para la GUI de demostracion. Esencial para despliegue y presentacion. Soporte dual dev/PyInstaller es importante para portabilidad.

**Observaciones**:
- Buen diseño defensivo con checks de dependencias y modelos
- Soporte PyInstaller con `sys._MEIPASS` y env vars
- Emojis en output que podrian fallar en terminales limitados

---

### 17. quickstart_landmarks.sh
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/quickstart_landmarks.sh`
- **Lineas/Tamano**: 34 lineas / 4 KB
- **Proposito**: Automatiza el pipeline de landmarks: entrenar 4 seeds, evaluar ensemble, extraer predicciones.
- **Contenido clave**:
  - Entrena 4 modelos con seeds 123, 321, 111, 666
  - Evalua ensemble con `python -m src_v2 evaluate-ensemble`
  - Extrae predicciones con `extract_predictions.py`
  - Variables de entorno configurables (SESSION, TRAIN_CONFIG, OUTPUT_ROOT, CHECKPOINT_ROOT)
- **Dependencias**: train.py, python -m src_v2 evaluate-ensemble, extract_predictions.py
- **Referenciado en CLAUDE.md**: Si (`bash scripts/quickstart_landmarks.sh`)
- **Importancia**: ALTO
- **Justificacion**: Automatizacion documentada en CLAUDE.md. Permite reproducir todo el pipeline de landmarks en un solo comando.

**Observaciones**:
- Usa `set -euo pipefail` -- buena practica de scripting
- Los 4 seeds corresponden a los splits usados en el ensemble best
- Usa `extract_predictions.py` (no predict_landmarks_dataset.py) -- el quickstart genera predicciones del test set, no del dataset completo

---

### 18. quickstart_warping.sh
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/quickstart_warping.sh`
- **Lineas/Tamano**: 38 lineas / 4 KB
- **Proposito**: Automatiza el pipeline de warping: canonical shape -> predicciones -> dataset warped.
- **Contenido clave**:
  1. Verifica/genera canonical shape y triangulation con `python -m src_v2 compute-canonical`
  2. Genera cache de predicciones con `predict_landmarks_dataset.py` (TTA + CLAHE)
  3. Genera dataset warped con `python -m src_v2 generate-dataset` (margin 1.05, no-full-coverage)
  - Variables de entorno configurables
- **Dependencias**: python -m src_v2 compute-canonical, predict_landmarks_dataset.py, python -m src_v2 generate-dataset
- **Referenciado en CLAUDE.md**: Si (`nohup bash scripts/quickstart_warping.sh > outputs/warping_quickstart.log 2>&1 &`)
- **Importancia**: CRITICO
- **Justificacion**: Es el script de automatizacion mas importante del pipeline de warping. Documentado prominentemente en CLAUDE.md. Conecta los 3 pasos del pipeline (canonical, predictions, warping).

**Observaciones**:
- Bien diseñado con check condicional de canonical shape (no regenera si existe)
- Parametros hardcodeados coinciden con GROUND_TRUTH.json (margin 1.05, clahe-tile 4, no-full-coverage)
- Buen uso de variables de entorno para override

---

### 19. run_best_ensemble.sh
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/run_best_ensemble.sh`
- **Lineas/Tamano**: 8 lineas / 4 KB
- **Proposito**: Ejecuta la evaluacion del mejor ensemble desde configs/ensemble_best.json.
- **Contenido clave**: Wrapper de una linea sobre evaluate_ensemble_from_config.py
- **Dependencias**: evaluate_ensemble_from_config.py
- **Referenciado en CLAUDE.md**: No
- **Importancia**: BAJO
- **Justificacion**: Convenience script. Simplifica el comando pero no añade valor significativo.

---

### 20. run_classifier_sweep_accuracy.sh
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/run_classifier_sweep_accuracy.sh`
- **Lineas/Tamano**: 51 lineas / 4 KB
- **Proposito**: Sweep de hiperparametros para el clasificador CNN: learning rates x seeds x class weights.
- **Contenido clave**:
  - Grid: LR={5e-5, 2e-4} x seeds={42, 123, 321} x class_weights={on, off}
  - Skip si best_classifier.pt ya existe (resume)
  - Logging con tee
  - Variables de entorno configurables (OUTPUT_ROOT, CONFIG, DEVICE, SEEDS, LRS, etc.)
- **Dependencias**: python -m src_v2 train-classifier
- **Referenciado en CLAUDE.md**: No directamente (pero implicitamente como el proceso que genero warped_96)
- **Importancia**: MEDIO
- **Justificacion**: Fue el script que genero los resultados del clasificador validados. Util para reproduccion de sweeps.

**Observaciones**:
- Buen diseño con skip logic y variables de entorno
- El OUTPUT_ROOT tiene fecha hardcodeada `2026-01-12` -- deberia parametrizarse o usar fecha actual

---

### 21. run_seed_sweep.sh
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/run_seed_sweep.sh`
- **Lineas/Tamano**: 73 lineas / 4 KB
- **Proposito**: Entrena modelos con seeds arbitrarios y ejecuta sweep de combinaciones de ensemble (incluyendo modelos base existentes).
- **Contenido clave**:
  - Acepta seeds como argumentos de CLI
  - Entrena modelos nuevos con train.py (con o sin TRAIN_CONFIG)
  - Combina con BASE_MODELS (session10/session13) y EXISTING_MODELS
  - Ejecuta sweep_ensemble_combos.py con pool completo (opcion 1 y opcion 2)
  - `append_if_exists()`: Helper para añadir checkpoints opcionales
- **Dependencias**: train.py, sweep_ensemble_combos.py
- **Referenciado en CLAUDE.md**: No directamente (documentado en README.md de scripts)
- **Importancia**: MEDIO
- **Justificacion**: Orquestador principal para busqueda de mejores ensembles. Fue instrumental para encontrar el combo de 3.61 px. Diseño flexible con soporte de config.

**Observaciones**:
- Los hyperparameters hardcodeados (sin TRAIN_CONFIG) coinciden con los del best model
- Lista de BASE_MODELS corresponde al ensemble historico de 3.71 px

---

### 22. run_benchmark.sh
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/run_benchmark.sh`
- **Lineas/Tamano**: 55 lineas / 4 KB
- **Proposito**: Ejecuta benchmark de inferencia copiando imagenes de muestra a un directorio temporal y corriendo benchmark_inference.py.
- **Contenido clave**:
  - Crea directorio temporal con 100 imagenes (30 COVID + 40 Normal + 30 Viral)
  - Ejecuta benchmark_inference.py con ensemble + classifier
  - Limpia directorio temporal
- **Dependencias**: benchmark_inference.py (que importa de scripts.piecewise_affine_warp)
- **Referenciado en CLAUDE.md**: No
- **Importancia**: BAJO
- **Justificacion**: Script utilitario para medir rendimiento de inferencia. No es parte del pipeline de produccion. Referencia a `outputs/classifier_cropped_10/best_classifier.pt` que parece ser un checkpoint legacy.

**Observaciones**:
- Referencia a classifier_cropped_10 que no es el clasificador actual (warped_lung_best)
- Emojis en output
- Usa `find` para seleccionar imagenes -- podria ser mas robusto

---

### 23. README.md
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/README.md`
- **Lineas/Tamano**: 170 lineas / 8 KB
- **Proposito**: Documentacion del directorio de scripts: estructura, tabla de scripts activos, comandos de ejemplo, fuente de verdad.
- **Contenido clave**:
  - Tabla de scripts de produccion activos con CLI equivalentes
  - Tabla de scripts de automatizacion
  - Documentacion de configs (opcion 3)
  - Referencia a GROUND_TRUTH.json con metricas validadas
  - Seccion de scripts archivados
  - Valores clave: 3.61 px, 99.10%, warped_96 recomendado
- **Dependencias**: N/A
- **Referenciado en CLAUDE.md**: No
- **Importancia**: MEDIO
- **Justificacion**: Documentacion esencial para navegar el directorio de scripts. Mantiene informacion actualizada sobre el estado de cada script.

**Observaciones**:
- Bien mantenido y actualizado con los valores mas recientes
- Recomendacion correcta de usar CLI sobre scripts directos
- Lista scripts que ya no existen o fueron archivados

---

## Matriz de Dependencias entre Scripts

```
quickstart_warping.sh
  -> predict_landmarks_dataset.py (cache de predicciones)
  -> python -m src_v2 compute-canonical
  -> python -m src_v2 generate-dataset

quickstart_landmarks.sh
  -> train.py x4 (seeds 123, 321, 111, 666)
  -> python -m src_v2 evaluate-ensemble
  -> extract_predictions.py

run_seed_sweep.sh
  -> train.py (per seed)
  -> sweep_ensemble_combos.py
    -> python -m src_v2 evaluate-ensemble

run_classifier_sweep_accuracy.sh
  -> python -m src_v2 train-classifier

run_best_ensemble.sh
  -> evaluate_ensemble_from_config.py
    -> python -m src_v2 evaluate-ensemble

run_benchmark.sh
  -> benchmark_inference.py
    -> scripts/piecewise_affine_warp.py (legacy)

piecewise_affine_warp.py (legacy)
  -> landmark_connections.py

run_demo.py
  -> src_v2.gui.app.create_demo
```

## Clasificacion por Importancia

### CRITICO (2)
| Script | Justificacion |
|--------|---------------|
| `predict_landmarks_dataset.py` | Paso esencial del pipeline: genera cache de predicciones para warping |
| `quickstart_warping.sh` | Automatiza pipeline completo de warping, documentado en CLAUDE.md |

### ALTO (4)
| Script | Justificacion |
|--------|---------------|
| `evaluate_ensemble_from_config.py` | Evaluacion del ensemble, documentado en CLAUDE.md |
| `train.py` | Entrenamiento de landmarks, usado por quickstarts y sweeps |
| `quickstart_landmarks.sh` | Automatiza pipeline de landmarks, documentado en CLAUDE.md |
| `run_demo.py` | Punto de entrada de la GUI, soporte PyInstaller |

### MEDIO (7)
| Script | Justificacion |
|--------|---------------|
| `train_hierarchical.py` | Modelo experimental alternativo |
| `extract_predictions.py` | Analisis detallado de ensemble + triangulacion |
| `extract_dataset_splits.py` | Preparacion de datos para GUI |
| `sweep_ensemble_combos.py` | Busqueda de mejor ensemble |
| `landmark_connections.py` | Constantes anatomicas usadas como dependencia |
| `run_classifier_sweep_accuracy.sh` | Sweep que genero resultados validados |
| `run_seed_sweep.sh` | Busqueda de mejores ensembles |
| `README.md` | Documentacion del directorio |

### BAJO (8)
| Script | Justificacion |
|--------|---------------|
| `train_classifier.py` | Puro wrapper sin logica |
| `predict.py` | Legacy, reemplazado por predict_landmarks_dataset.py |
| `compute_cv_test_aggregated_metrics.py` | Post-hoc, hardcoded paths |
| `gpa_analysis.py` | Legacy Session 19, duplicado en src_v2 |
| `piecewise_affine_warp.py` | Legacy Session 20, duplicado en src_v2 |
| `analyze_data.py` | Analisis exploratorio one-shot |
| `analyze_hospital_marks.py` | Analisis visual para tesis |
| `run_best_ensemble.sh` | Convenience wrapper trivial |
| `run_benchmark.sh` | Benchmark con references legacy |

### Candidatos a ELIMINACION o ARCHIVADO
| Script | Razon |
|--------|-------|
| `train_classifier.py` | Wrapper trivial, el CLI es suficiente |
| `predict.py` | Marcado como legacy en CLAUDE.md |
| `gpa_analysis.py` | Duplicado completo de src_v2/processing/gpa.py (requiere migrar dependientes primero) |
| `piecewise_affine_warp.py` | Duplicado completo de src_v2/processing/warp.py (requiere migrar dependientes primero) |
| `run_best_ensemble.sh` | 8 lineas, trivial |

---

## Problemas Detectados

### 1. Duplicacion de logica GPA/Warping
**Severidad**: MEDIA
**Archivos**: `gpa_analysis.py`, `piecewise_affine_warp.py` vs `src_v2/processing/gpa.py`, `src_v2/processing/warp.py`
**Descripcion**: Las implementaciones de GPA y warping existen duplicadas: como scripts standalone (Session 19/20) y como modulos en src_v2. Los scripts legacy siguen siendo importados por otros scripts (benchmark, visualizacion, verificacion).
**Recomendacion**: Migrar todos los dependientes para importar de `src_v2.processing.*` y archivar los scripts legacy.

### 2. Inconsistencia en preprocessing de imagenes
**Severidad**: MEDIA
**Archivos**: `predict_landmarks_dataset.py` (CLAHE en grayscale), `predict.py` (CLAHE en LAB), `src_v2/data/transforms.py`
**Descripcion**: Tres implementaciones diferentes de preprocessing de imagenes. `predict_landmarks_dataset.py` aplica CLAHE directamente sobre grayscale, `predict.py` lo aplica en espacio LAB, y el training pipeline usa `src_v2/data/transforms.py`. Esto podria causar discrepancias sutiles entre training y inference.
**Recomendacion**: Unificar preprocessing en una funcion en `src_v2/data/transforms.py` y reutilizar en todos los scripts.

### 3. Defaults de CLAHE tile inconsistentes
**Severidad**: BAJA
**Archivos**: `train.py` (default=8), `src_v2/constants.py` (DEFAULT_CLAHE_TILE_SIZE=4)
**Descripcion**: El CLI default de train.py usa tile=8, mientras que el proyecto valido tile=4 como optimo. Los configs JSON corrigen esto, pero si alguien ejecuta train.py sin config podria usar el valor suboptimo.
**Recomendacion**: Cambiar default de `--clahe-tile` en train.py a 4, o importar de constants.py.

### 4. Metricas obsoletas hardcodeadas
**Severidad**: BAJA
**Archivos**: `predict.py` (3.71 px), `extract_predictions.py` (DEFAULT_ENSEMBLE_CHECKPOINTS), `extract_dataset_splits.py` (98.05%)
**Descripcion**: Varios scripts referencian metricas o checkpoints que ya no son los best validados (3.61 px, 99.10%).
**Recomendacion**: Actualizar o eliminar referencias hardcodeadas; referenciar GROUND_TRUTH.json.

### 5. datetime.utcnow() deprecado
**Severidad**: BAJA
**Archivos**: `predict_landmarks_dataset.py`
**Descripcion**: `datetime.utcnow()` esta deprecado desde Python 3.12; deberia usar `datetime.now(timezone.utc)`.
**Recomendacion**: Actualizar a API moderna.

### 6. Hardcoded architecture en scripts legacy
**Severidad**: BAJA
**Archivos**: `predict.py`, `extract_predictions.py`
**Descripcion**: Hardcodean `hidden_dim=768, deep_head=True, coord_attention=True` en lugar de auto-detectar desde checkpoint como hace `predict_landmarks_dataset.py`.
**Recomendacion**: Estos scripts son legacy, por lo que la solucion es simplemente no usarlos.

---

## Estadisticas Globales

| Metrica | Valor |
|---------|-------|
| Total archivos | 23 |
| Total lineas Python | 5,763 |
| Total lineas Shell | 259 |
| Total lineas Markdown | 170 |
| Total lineas | 6,192 |
| Scripts CRITICOS | 2 |
| Scripts ALTOS | 4 |
| Scripts MEDIOS | 8 |
| Scripts BAJOS | 9 |
| Candidatos a archivado | 5 |
| Scripts documentados en CLAUDE.md | 5 |
| Scripts con duplicacion vs src_v2 | 3 (gpa_analysis, piecewise_affine_warp, predict) |
