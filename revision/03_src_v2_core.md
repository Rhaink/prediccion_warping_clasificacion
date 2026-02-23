# Revision 03: src_v2 Core Files

**Fecha**: 2026-02-11
**Alcance**: Archivos raiz del modulo `src_v2/` - entry points, constantes y CLI principal
**Archivos revisados**: 4

---

## Indice

1. [src_v2/__init__.py](#1-src_v2initpy)
2. [src_v2/__main__.py](#2-src_v2mainpy)
3. [src_v2/constants.py](#3-src_v2constantspy)
4. [src_v2/cli.py](#4-src_v2clipy)
   - [Tabla Resumen de Comandos CLI](#tabla-resumen-de-comandos-cli)
   - [Funciones Helper](#funciones-helper-no-comandos)
   - [Detalle por Comando](#detalle-por-comando)

---

## 1. src_v2/__init__.py

| Campo | Valor |
|-------|-------|
| **Ruta** | `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/__init__.py` |
| **Lineas** | 15 |
| **Proposito** | Entry point del paquete. Define metadata (`__version__`, `__author__`) y docstring con resultados clave. |
| **Contenido clave** | `__version__ = "2.1.0"`, docstring con landmark error (3.71 px) y classification accuracy (99.10%). |
| **Dependencias** | Ninguna |
| **Importancia** | **MEDIO** |
| **Justificacion** | Necesario como paquete Python pero contiene minima logica. Los valores en el docstring estan parcialmente desactualizados: el docstring menciona 3.71 px como resultado del ensemble, pero GROUND_TRUTH.json indica que el best actual es 3.61 px (ensemble_4_models_tta_best_20260111). El valor de 99.10% corresponde a warped_96 que esta marcado como obsoleto en GROUND_TRUTH.json; el actual es 98.05% (warped_lung_best). |

### Observaciones

- **Inconsistencia de metricas**: El docstring indica "3.71 px (ensemble 4 models + TTA)" y "99.10% (warped_96, RECOMMENDED)". Segun GROUND_TRUTH.json:
  - El ensemble de 3.71 px esta marcado como `obsolete: true` (superado por 3.61 px con seeds 123,321,111,666).
  - warped_96 esta marcado como `obsolete: true` con razon "Superseded by warped_lung_best (98.05%)".
- **Recomendacion**: Actualizar docstring para reflejar valores actuales validados: 3.61 px y 98.05%.

---

## 2. src_v2/__main__.py

| Campo | Valor |
|-------|-------|
| **Ruta** | `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/__main__.py` |
| **Lineas** | 15 |
| **Proposito** | Permite ejecutar el paquete como modulo (`python -m src_v2`). Importa y llama `main()` desde `src_v2.cli`. |
| **Contenido clave** | `from src_v2.cli import main` + `main()` |
| **Dependencias** | `src_v2.cli` |
| **Importancia** | **CRITICO** |
| **Justificacion** | Es el unico entry point documentado para toda la CLI. Sin este archivo, `python -m src_v2` no funcionaria. El docstring documenta correctamente los comandos principales. |

### Observaciones

- Codigo minimo y correcto. No requiere cambios.

---

## 3. src_v2/constants.py

| Campo | Valor |
|-------|-------|
| **Ruta** | `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/constants.py` |
| **Lineas** | 299 |
| **Proposito** | Centraliza todas las constantes del dominio: landmarks, dimensiones de imagen, normalizacion, categorias, hiperparametros de entrenamiento, warping y modelo. |
| **Contenido clave** | `NUM_LANDMARKS=15`, `SYMMETRIC_PAIRS`, `CENTRAL_LANDMARKS`, `DEFAULT_IMAGE_SIZE=224`, `ORIGINAL_IMAGE_SIZE=299`, `IMAGENET_MEAN/STD`, `CATEGORIES`, `OPTIMAL_MARGIN_SCALE=1.05`, hiperparametros de entrenamiento, CLAHE defaults, quick mode limits. |
| **Dependencias** | Solo `typing` (stdlib) |
| **Importancia** | **CRITICO** |
| **Justificacion** | Referenciado por practicamente todos los modulos del proyecto. Define la geometria de landmarks, pares simetricos, y parametros validados experimentalmente (margin=1.05, CLAHE tile=4). Consistente con GROUND_TRUTH.json. |

### Contenido por Seccion

| Seccion | Lineas | Constantes principales |
|---------|--------|----------------------|
| LANDMARKS | 37-88 | `NUM_LANDMARKS=15`, `NUM_COORDINATES=30`, `LANDMARK_NAMES`, `SYMMETRIC_PAIRS` (5 pares), `CENTRAL_LANDMARKS` [8,9,10], `CENTRAL_LANDMARKS_T`, `AXIS_LANDMARKS` [0,1] |
| DIMENSIONES DE IMAGEN | 90-98 | `DEFAULT_IMAGE_SIZE=224`, `ORIGINAL_IMAGE_SIZE=299` |
| NORMALIZACION | 100-106 | `IMAGENET_MEAN`, `IMAGENET_STD` |
| CATEGORIAS | 108-123 | `CATEGORIES=['COVID','Normal','Viral_Pneumonia']`, `NUM_CLASSES=3`, `DEFAULT_CATEGORY_WEIGHTS` |
| CLASIFICADOR | 125-133 | `DEFAULT_CLASSIFIER_BACKBONE="resnet18"`, `CLASSIFIER_CLASSES` |
| MODELO | 135-146 | `BACKBONE_FEATURE_DIM=512`, `DEFAULT_HIDDEN_DIM=768`, `DEFAULT_DROPOUT_RATE=0.3` |
| ENTRENAMIENTO | 148-162 | `DEFAULT_BATCH_SIZE=16`, LRs (1e-3, 2e-5, 2e-4), epochs (15, 100) |
| LOSS | 164-173 | `DEFAULT_WING_OMEGA=10.0`, `DEFAULT_WING_EPSILON=2.0`, `DEFAULT_SYMMETRY_MARGIN=6.0` |
| DATA AUGMENTATION | 175-188 | `DEFAULT_FLIP_PROB=0.5`, `DEFAULT_ROTATION_DEGREES=10.0`, `DEFAULT_CLAHE_CLIP_LIMIT=2.0`, `DEFAULT_CLAHE_TILE_SIZE=4` |
| QUICK MODE | 190-202 | `QUICK_MODE_MAX_TRAIN=500`, `QUICK_MODE_MAX_VAL=100`, `QUICK_MODE_MAX_TEST=100` |
| WARPING | 204-222 | `OPTIMAL_MARGIN_SCALE=1.05`, `DEFAULT_MARGIN_SCALE=1.25` |
| COMBINED LOSS | 224-232 | `DEFAULT_CENTRAL_WEIGHT=1.0`, `DEFAULT_SYMMETRY_WEIGHT=0.5` |
| HIERARCHICAL LOSS | 234-245 | `HIERARCHICAL_DT_SCALE=0.1`, `HIERARCHICAL_T_SCALE=0.2`, `HIERARCHICAL_D_MAX=0.7` |
| ENTRENAMIENTO adicional | 247-255 | `DEFAULT_WEIGHT_DECAY=0.01`, `DEFAULT_FINE_TUNE_LR=1e-4` |
| DATA AUG adicional | 257-265 | `DEFAULT_BRIGHTNESS_RANGE`, `DEFAULT_CONTRAST_RANGE` |
| BILATERAL LANDMARKS | 267-275 | `BILATERAL_T_POSITIONS=[0.25,0.50,0.75,0.00,1.00]` |
| WARP params | 277-285 | `DEFAULT_WARP_MARGIN=2`, `DEFAULT_WARP_MAX_SIZE=224` |
| HIERARCHICAL MODEL | 287-298 | `HIERARCHICAL_HIDDEN_DIM=512`, `HIERARCHICAL_NUM_GROUPS=32`, `HIERARCHICAL_NUM_GROUPS_HALF=16` |

### Observaciones

- **Consistencia con GROUND_TRUTH.json**: Los valores criticos son consistentes:
  - `OPTIMAL_MARGIN_SCALE=1.05` coincide con `preprocessing.warping.margin_scale_optimal=1.05`
  - `DEFAULT_CLAHE_TILE_SIZE=4` coincide con `preprocessing.clahe.tile_size=4`
  - `DEFAULT_CLAHE_CLIP_LIMIT=2.0` coincide con `preprocessing.clahe.clip_limit=2.0`
  - `DEFAULT_HIDDEN_DIM=768` coincide con `preprocessing.model_architecture.hidden_dim=768`
  - `DEFAULT_DROPOUT_RATE=0.3` coincide con `preprocessing.model_architecture.dropout=0.3`
- **Consistencia con CLAUDE.md**: Correcta. CLAUDE.md referencia `SYMMETRIC_PAIRS`, `CENTRAL_LANDMARKS`, `OPTIMAL_MARGIN_SCALE`, `DEFAULT_CLAHE_TILE_SIZE`.
- **Documentacion interna**: Excelente. Cada seccion tiene comentarios claros, la estructura geometrica de los 15 landmarks esta bien explicada en el docstring del modulo.
- **Constantes de GROUND_TRUTH.json que usan `use_full_coverage: true`**: Nota que en GROUND_TRUTH.json `preprocessing.warping.use_full_coverage=true`, pero la configuracion warping_best.json usa `use_full_coverage=false`. Esto podria ser una inconsistencia historica en GROUND_TRUTH.json.

---

## 4. src_v2/cli.py

| Campo | Valor |
|-------|-------|
| **Ruta** | `/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/cli.py` |
| **Lineas** | 10,895 |
| **Proposito** | CLI monolitica basada en Typer que implementa TODOS los comandos del proyecto: entrenamiento, evaluacion, warping, clasificacion, visualizacion, analisis y utilidades. |
| **Contenido clave** | 31 comandos CLI, 3 funciones helper globales, ~10 funciones helper de utilidad, ~3 funciones helper para comparacion de arquitecturas. |
| **Dependencias** | `typer`, `logging`, `random`, `sys`, `numpy`, `pathlib`, `typing`, + imports lazy de `torch`, `cv2`, `sklearn`, `matplotlib`, `seaborn`, `tqdm`, `json`, `csv`, `time`, `shutil`, `multiprocessing`, y todos los modulos de `src_v2` |
| **Importancia** | **CRITICO** |
| **Justificacion** | Es la interfaz principal del proyecto. Todos los workflows documentados en CLAUDE.md pasan por este archivo. Sin embargo, su tamano (10,895 lineas) es un problema serio de mantenibilidad. |

### Tabla Resumen de Comandos CLI

| # | Comando | Lineas | En CLAUDE.md | En GROUND_TRUTH | Importancia | Descripcion |
|---|---------|--------|-------------|-----------------|-------------|-------------|
| 1 | `train` | 201-536 | Indirecta (quickstart) | Si (modelo) | **CRITICO** | Entrenar modelo de landmarks (2 fases) |
| 2 | `evaluate` | 538-797 | Indirecta | Si (metricas) | **ALTO** | Evaluar modelo de landmarks en test |
| 3 | `predict` | 799-1014 | Si (uso basico) | No | **ALTO** | Predecir landmarks en imagen individual |
| 4 | `warp` | 1017-1271 | Indirecta | No | **MEDIO** | Warping de dataset de imagenes |
| 5 | `version` | 1273-1278 | No | No | **BAJO** | Mostrar version del paquete |
| 6 | `evaluate-ensemble` | 1280-1623 | Si (paso 6) | Si (3.61px) | **CRITICO** | Evaluar ensemble de modelos + TTA |
| 7 | `classify` | 1625-2017 | No | No | **MEDIO** | Clasificar imagenes (pipeline completo) |
| 8 | `train-classifier` | 2019-2426 | Si (paso 4) | Si (warped_lung) | **CRITICO** | Entrenar clasificador CNN |
| 9 | `evaluate-classifier` | 2428-2601 | Si (paso 5) | Si (98.05%) | **CRITICO** | Evaluar clasificador |
| 10 | `evaluate-classifier-ensemble` | 2604-2977 | No | Si (CV ensemble) | **ALTO** | Evaluar ensemble 5-fold CV |
| 11 | `cross-validate-classifier` | 2979-3618 | No | Si (CV metricas) | **ALTO** | K-fold cross-validation |
| 12 | `cross-evaluate` | 3620-3923 | No | Si (cross_eval) | **MEDIO** | Cross-evaluacion 2 modelos x 2 datasets |
| 13 | `evaluate-external` | 3926-4225 | No | Si (FedCOVIDx) | **MEDIO** | Evaluacion en dataset externo binario |
| 14 | `test-robustness` | 4228-4456 | No | Si (robustness) | **MEDIO** | Test con perturbaciones JPEG/blur/ruido |
| 15 | `compute-canonical` | 4459-4708 | Si (paso 1) | No | **CRITICO** | Computar forma canonica via GPA |
| 16 | `generate-dataset` | 4711-5474 | Si (paso 3) | Si (warping cfg) | **CRITICO** | Generar dataset warped con splits |
| 17 | `generate-cropped-dataset` | 5482-5891 | No | No | **BAJO** | Generar dataset recortado sin warping |
| 18 | `apply-contrast-dataset` | 6281-6391 | No | No | **BAJO** | Aplicar SAHS/CLAHE a dataset |
| 19 | `crop-dataset` | 6531-6606 | No | No | **BAJO** | Recortar imagenes warped (quitar fondo negro) |
| 20 | `compare-architectures` | 7073-7518 | No | No | **MEDIO** | Comparar multiples arquitecturas CNN |
| 21 | `gradcam` | 7521-7798 | No | No | **MEDIO** | Visualizaciones Grad-CAM |
| 22 | `analyze-errors` | 7801-8063 | No | No | **BAJO** | Analizar errores de clasificacion |
| 23 | `pfs-analysis` | 8065-8447 | No | Si (PFS obsoleto) | **BAJO** | Analisis Pulmonary Focus Score |
| 24 | `generate-lung-masks` | 8450-8585 | No | No | **BAJO** | Generar mascaras pulmonares aproximadas |
| 25 | `optimize-margin` | 8593-9447 | No | Si (margin 1.05) | **MEDIO** | Grid search para margen optimo de warping |
| 26 | `extract-dataset-splits` | 9449-9553 | No | No | **BAJO** | Extraer splits para GUI |
| 27 | `generate-landmark-visualization-dataset` | 9555-9957 | Si (paso 6b) | No | **ALTO** | Dataset de visualizacion de landmarks |
| 28 | `generate-delaunay-mesh-dataset` | 9959-10429 | No | No | **MEDIO** | Dataset con malla Delaunay |
| 29 | `generate-landmark-comparison-dataset` | 10432-10567 | No | No | **MEDIO** | Comparacion pred vs GT en imagenes originales |
| 30 | `generate-landmark-comparison-dataset-aligned` | 10569-10734 | No | No | **MEDIO** | Comparacion alineada con warped dataset |
| 31 | `generate-viz-dataset` | 10737-10886 | No | No | **MEDIO** | Visualizaciones cientificas con landmarks+malla |

**Totales por importancia:**
- CRITICO: 7 comandos (train, evaluate-ensemble, train-classifier, evaluate-classifier, compute-canonical, generate-dataset, + archivo en si)
- ALTO: 4 comandos (evaluate, predict, evaluate-classifier-ensemble, cross-validate-classifier, generate-landmark-visualization-dataset)
- MEDIO: 11 comandos
- BAJO: 8 comandos
- ELIMINABLE: 0 (todos tienen algun uso potencial, aunque varios son experimentales/historicos)

### Funciones Helper (no comandos)

| Funcion | Lineas | Proposito | Importancia |
|---------|--------|-----------|-------------|
| `verbose_callback()` | 59-64 | Configura logging DEBUG cuando se pasa `--verbose` | MEDIO |
| `app callback (main)` | 74-90 | Callback de Typer para `--verbose/-v` global | MEDIO |
| `get_device()` | 93-104 | Detecta dispositivo (CUDA > MPS > CPU) | CRITICO |
| `get_optimal_num_workers()` | 107-151 | Workers optimos para DataLoader (Windows/Linux/sandbox) | ALTO |
| `detect_architecture_from_checkpoint()` | 154-198 | Detecta coord_attention, deep_head, hidden_dim desde state_dict | CRITICO |
| `_parse_contrast_exts()` | 5899-5917 | Parsea extensiones de imagen separadas por coma | BAJO |
| `_is_subpath()` | 5920-5935 | Verifica si un path es hijo de otro | BAJO |
| `_to_gray()` | 5938-5956 | Convierte imagen a escala de grises | BAJO |
| `_build_foreground_mask()` | 5959-5970 | Construye mascara de foreground por threshold | BAJO |
| `_apply_sahs_to_gray()` | 5973-6035 | Aplica SAHS a imagen grayscale | BAJO |
| `_apply_clahe_to_gray()` | 6038-6078 | Aplica CLAHE a imagen grayscale | MEDIO |
| `_crop_to_foreground()` | 6081-6125 | Recorta al bounding box de foreground y resize | BAJO |
| `_apply_contrast_dataset()` | 6128-6279 | Implementacion interna de apply-contrast-dataset | BAJO |
| `_crop_dataset_impl()` | 6399-6528 | Implementacion interna de crop-dataset | BAJO |
| `_train_single_architecture()` | 6635-6847 | Entrena una arquitectura y retorna metricas (para compare-architectures) | MEDIO |
| `_generate_comparison_figures()` | 6850-7022 | Genera graficos comparativos matplotlib | BAJO |
| `_generate_comparison_reports()` | 7025-7070 | Genera JSON y CSV con resultados | BAJO |
| `main()` | 10889-10891 | Entry point que llama `app()` | CRITICO |

### Detalle por Comando

---

#### Comando 1: `train`

| Campo | Valor |
|-------|-------|
| **Lineas** | 201-536 |
| **Decorador** | `@app.command()` |
| **Funcion** | `train()` |
| **En CLAUDE.md** | Indirecta (mencionado en quickstart_landmarks.sh) |
| **En GROUND_TRUTH** | Si (los modelos entrenados con este comando producen los valores de landmarks) |
| **Importancia** | **CRITICO** |

**Parametros:**
- `csv_path` (Argument): Path a CSV con landmarks
- `image_dir` (Argument): Directorio de imagenes
- `--output-dir`: Directorio de salida (default: "outputs/landmark_training")
- `--config`: JSON config file
- `--image-size`: Tamano de imagen (default: 224)
- `--batch-size`: (default: 16)
- `--phase1-lr`, `--phase2-backbone-lr`, `--phase2-head-lr`: Learning rates
- `--phase1-epochs`, `--phase2-epochs`: Epocas por fase
- `--hidden-dim`: Dimension oculta (default: 768)
- `--dropout`: (default: 0.3)
- `--flip-prob`, `--rotation-degrees`: Augmentacion
- `--wing-omega`, `--wing-epsilon`: Parametros Wing Loss
- `--symmetry-margin`: Margen de simetria
- `--use-coord-attention/--no-coord-attention`: Coordinate Attention (default: True)
- `--deep-head/--simple-head`: Deep head (default: True)
- `--clahe/--no-clahe`: CLAHE preprocessing (default: True)
- `--clahe-clip`, `--clahe-tile`: Parametros CLAHE
- `--device`: auto/cuda/cpu/mps
- `--seed`: Semilla
- `--split-seed`: Semilla para splits
- `--patience`: Early stopping
- `--val-split`, `--test-split`: Ratios de split

**Notas:**
- Soporta JSON config con override por CLI flags.
- Implementa entrenamiento 2 fases (frozen backbone + fine-tuning).
- Usa `LandmarkTrainer` de `src_v2/training/trainer.py`.

---

#### Comando 2: `evaluate`

| Campo | Valor |
|-------|-------|
| **Lineas** | 538-797 |
| **Decorador** | `@app.command()` |
| **Funcion** | `evaluate()` |
| **En CLAUDE.md** | Indirecta |
| **En GROUND_TRUTH** | Si (metricas individuales) |
| **Importancia** | **ALTO** |

**Parametros:**
- `checkpoint` (Argument): Path al checkpoint .pt
- `--csv-path`: CSV con landmarks GT
- `--image-dir`: Directorio de imagenes
- `--image-size`: (default: 224)
- `--batch-size`: (default: 16)
- `--device`: auto/cuda/cpu/mps
- `--split`: train/val/test (default: "test")
- `--split-seed`: Semilla para splits
- `--tta/--no-tta`: Test-Time Augmentation
- `--clahe/--no-clahe`: CLAHE
- `--clahe-clip`, `--clahe-tile`: Parametros CLAHE
- `--per-landmark/--no-per-landmark`: Metricas por landmark
- `--per-category/--no-per-category`: Metricas por categoria

**Notas:**
- Usa `detect_architecture_from_checkpoint()` para inferir arquitectura.
- Reporta error medio, std, mediana en pixeles.
- Soporta TTA con swap de pares simetricos.

---

#### Comando 3: `predict`

| Campo | Valor |
|-------|-------|
| **Lineas** | 799-1014 |
| **Decorador** | `@app.command()` |
| **Funcion** | `predict()` |
| **En CLAUDE.md** | Si (ejemplo basico en docstring de __main__.py) |
| **En GROUND_TRUTH** | No |
| **Importancia** | **ALTO** |

**Parametros:**
- `image_path` (Argument): Path a imagen de rayos X
- `--checkpoint`: Path al modelo (requerido)
- `--image-size`: (default: 224)
- `--device`: auto/cuda/cpu/mps
- `--output-dir`: Directorio de salida (default: None, imprime a stdout)
- `--clahe/--no-clahe`: CLAHE
- `--clahe-clip`, `--clahe-tile`: Parametros CLAHE
- `--tta/--no-tta`: TTA
- `--visualize/--no-visualize`: Guardar imagen con landmarks

**Notas:**
- Prediccion de landmarks en una sola imagen.
- Util para demostraciones y debugging.

---

#### Comando 4: `warp`

| Campo | Valor |
|-------|-------|
| **Lineas** | 1017-1271 |
| **Decorador** | `@app.command()` |
| **Funcion** | `warp()` |
| **En CLAUDE.md** | Indirecta (pipeline paso 3 usa generate-dataset) |
| **En GROUND_TRUTH** | No directamente |
| **Importancia** | **MEDIO** |

**Parametros:**
- `input_dir` (Argument): Directorio con imagenes
- `output_dir` (Argument): Directorio de salida
- `--checkpoint`: Modelo de landmarks
- `--canonical-shape`: JSON forma canonica
- `--triangulation`: JSON triangulacion Delaunay
- `--margin-scale`: Escala de margen (default: 1.25)
- `--image-size`: (default: 224)
- `--device`: auto/cuda/cpu/mps
- `--clahe/--no-clahe`, `--clahe-clip`, `--clahe-tile`
- `--tta/--no-tta`
- `--fill-threshold`: Umbral de fill rate

**Notas:**
- Warping de todas las imagenes en un directorio.
- Diferente de `generate-dataset` que ademas genera splits y usa predicciones cacheadas.
- Usa `DEFAULT_MARGIN_SCALE=1.25` (legacy), no el optimo 1.05.

---

#### Comando 5: `version`

| Campo | Valor |
|-------|-------|
| **Lineas** | 1273-1278 |
| **Decorador** | `@app.command()` |
| **Funcion** | `version()` |
| **En CLAUDE.md** | No |
| **En GROUND_TRUTH** | No |
| **Importancia** | **BAJO** |

**Parametros:** Ninguno.

**Notas:**
- Imprime `__version__` del paquete (2.1.0). Codigo trivial.

---

#### Comando 6: `evaluate-ensemble`

| Campo | Valor |
|-------|-------|
| **Lineas** | 1280-1623 |
| **Decorador** | `@app.command("evaluate-ensemble")` |
| **Funcion** | `evaluate_ensemble()` |
| **En CLAUDE.md** | Si (paso 6: scripts/evaluate_ensemble_from_config.py) |
| **En GROUND_TRUTH** | Si (ensemble_4_models_tta_best_20260111: 3.61 px) |
| **Importancia** | **CRITICO** |

**Parametros:**
- `checkpoints` (Argument, multiple): Paths a checkpoints
- `--csv-path`: CSV con landmarks GT
- `--image-dir`: Directorio imagenes
- `--image-size`: (default: 224)
- `--batch-size`: (default: 16)
- `--device`: auto/cuda/cpu/mps
- `--split`: train/val/test
- `--split-seed`: Semilla splits
- `--tta/--no-tta`: TTA
- `--clahe/--no-clahe`, `--clahe-clip`, `--clahe-tile`
- `--per-landmark/--no-per-landmark`
- `--per-category/--no-per-category`
- `--save-predictions`: Path para guardar NPZ
- `--config`: JSON config para ensemble

**Notas:**
- Comando clave que produce el resultado de 3.61 px.
- Soporta config JSON (ensemble_best.json) con lista de modelos.
- Promedia predicciones de N modelos (con TTA opcional).
- Guarda predicciones en formato NPZ para uso posterior (warping, visualizacion).

---

#### Comando 7: `classify`

| Campo | Valor |
|-------|-------|
| **Lineas** | 1625-2017 |
| **Decorador** | `@app.command()` |
| **Funcion** | `classify()` |
| **En CLAUDE.md** | No |
| **En GROUND_TRUTH** | No |
| **Importancia** | **MEDIO** |

**Parametros:**
- `image_path` (Argument): Path a imagen
- `--classifier-checkpoint`: Modelo clasificador
- `--landmark-checkpoint`: Modelo landmarks (opcional, para pipeline completo)
- `--canonical-shape`, `--triangulation`: Para warping
- `--margin-scale`: (default: 1.25)
- `--image-size`: (default: 224)
- `--device`: auto/cuda/cpu/mps
- `--clahe/--no-clahe`, `--clahe-clip`, `--clahe-tile`
- `--top-k`: Numero de clases a mostrar

**Notas:**
- Pipeline de inferencia completo: imagen -> landmarks -> warping -> clasificacion.
- Util para demos y testing individual, pero no usado en workflows de produccion.

---

#### Comando 8: `train-classifier`

| Campo | Valor |
|-------|-------|
| **Lineas** | 2019-2426 |
| **Decorador** | `@app.command("train-classifier")` |
| **Funcion** | `train_classifier()` |
| **En CLAUDE.md** | Si (paso 4) |
| **En GROUND_TRUTH** | Si (warped_lung_best: 98.05%) |
| **Importancia** | **CRITICO** |

**Parametros:**
- `data_dir` (Argument): Directorio del dataset warped
- `--output-dir`: Directorio salida
- `--config`: JSON config
- `--backbone`: Arquitectura (default: "resnet18")
- `--epochs`: (default: 30)
- `--batch-size`: (default: 16)
- `--lr`: Learning rate (default: 1e-4)
- `--weight-decay`: (default: 0.01)
- `--patience`: Early stopping (default: 10)
- `--device`: auto/cuda/cpu/mps
- `--seed`: Semilla
- `--dropout`: (default: 0.3)
- `--pretrained/--no-pretrained`: ImageNet weights (default: True)
- `--save-best/--no-save-best`: Guardar mejor modelo
- `--save-history/--no-save-history`: Guardar historial

**Notas:**
- Soporta config JSON (classifier_warped_base.json).
- Usa ImageClassifier con backbone configurable.
- Implementa early stopping por F1-macro en validacion.
- Guarda checkpoint, historial y metricas.

---

#### Comando 9: `evaluate-classifier`

| Campo | Valor |
|-------|-------|
| **Lineas** | 2428-2601 |
| **Decorador** | `@app.command("evaluate-classifier")` |
| **Funcion** | `evaluate_classifier()` |
| **En CLAUDE.md** | Si (paso 5) |
| **En GROUND_TRUTH** | Si (98.05% warped_lung_best) |
| **Importancia** | **CRITICO** |

**Parametros:**
- `checkpoint` (Argument): Path al clasificador .pt
- `--data-dir`: Directorio del dataset
- `--split`: train/val/test (default: "test")
- `--batch-size`: (default: 16)
- `--device`: auto/cuda/cpu/mps
- `--tta/--no-tta`: TTA horizontal flip
- `--output-dir`: Para guardar resultados

**Notas:**
- Reporta accuracy, F1-macro, F1-weighted, confusion matrix.
- Soporta TTA para clasificador (flip horizontal).

---

#### Comando 10: `evaluate-classifier-ensemble`

| Campo | Valor |
|-------|-------|
| **Lineas** | 2604-2977 |
| **Decorador** | `@app.command("evaluate-classifier-ensemble")` |
| **Funcion** | `evaluate_classifier_ensemble()` |
| **En CLAUDE.md** | No directamente |
| **En GROUND_TRUTH** | Si (classifier_ensemble_cv: 98.26% con TTA) |
| **Importancia** | **ALTO** |

**Parametros:**
- `checkpoints` (Argument, multiple): Paths a checkpoints de folds
- `--data-dir`: Directorio del dataset
- `--split`: train/val/test
- `--batch-size`: (default: 16)
- `--device`: auto/cuda/cpu/mps
- `--voting`: soft/hard (default: "soft")
- `--tta/--no-tta`: TTA
- `--output-dir`: Para guardar resultados
- `--config`: JSON config

**Notas:**
- Combina N clasificadores de K-fold CV.
- Soft voting: promedio de probabilidades. Hard voting: voto mayoritario.
- Produce el resultado de 98.26% en GROUND_TRUTH.json.

---

#### Comando 11: `cross-validate-classifier`

| Campo | Valor |
|-------|-------|
| **Lineas** | 2979-3618 |
| **Decorador** | `@app.command("cross-validate-classifier")` |
| **Funcion** | `cross_validate_classifier()` |
| **En CLAUDE.md** | No |
| **En GROUND_TRUTH** | Si (cross_validation: 98.60% +/- 0.26%) |
| **Importancia** | **ALTO** |

**Parametros:**
- `data_dir` (Argument): Directorio dataset
- `--output-dir`: Directorio salida
- `--backbone`: (default: "resnet18")
- `--folds`: Numero de folds (default: 5)
- `--epochs`: (default: 30)
- `--batch-size`: (default: 16)
- `--lr`: (default: 1e-4)
- `--weight-decay`: (default: 0.01)
- `--patience`: (default: 10)
- `--device`: auto/cuda/cpu/mps
- `--seed`: Semilla
- `--dropout`: (default: 0.3)
- `--eval-test/--no-eval-test`: Evaluar en test holdout
- `--config`: JSON config

**Notas:**
- K-fold CV estratificada sobre train+val, con test holdout opcional.
- Guarda modelo de cada fold para uso posterior en ensemble.
- Comando largo (~640 lineas) con logica de splits, entrenamiento por fold, y reportes.

---

#### Comando 12: `cross-evaluate`

| Campo | Valor |
|-------|-------|
| **Lineas** | 3620-3923 |
| **Decorador** | `@app.command("cross-evaluate")` |
| **Funcion** | `cross_evaluate()` |
| **En CLAUDE.md** | No |
| **En GROUND_TRUTH** | Si (cross_evaluation, obsoleto) |
| **Importancia** | **MEDIO** |

**Parametros:**
- `--model-a-checkpoint`, `--model-b-checkpoint`: Dos modelos
- `--dataset-a-dir`, `--dataset-b-dir`: Dos datasets
- `--model-a-name`, `--model-b-name`: Nombres display
- `--dataset-a-name`, `--dataset-b-name`: Nombres display
- `--split`: (default: "test")
- `--batch-size`, `--device`
- `--output-dir`

**Notas:**
- Genera matriz 2x2: cada modelo evaluado en cada dataset.
- Los resultados en GROUND_TRUTH.json estan marcados como obsoletos.

---

#### Comando 13: `evaluate-external`

| Campo | Valor |
|-------|-------|
| **Lineas** | 3926-4225 |
| **Decorador** | `@app.command("evaluate-external")` |
| **Funcion** | `evaluate_external()` |
| **En CLAUDE.md** | No |
| **En GROUND_TRUTH** | Si (external_validation, obsoleto) |
| **Importancia** | **MEDIO** |

**Parametros:**
- `data_dir` (Argument): Directorio dataset externo
- `--checkpoint`: Modelo clasificador
- `--positive-class`: Clase positiva (default: "COVID")
- `--image-size`: (default: 224)
- `--batch-size`, `--device`
- `--output-dir`

**Notas:**
- Evaluacion binaria en dataset externo (FedCOVIDx).
- Mapea 3 clases a 2 (COVID=positivo, rest=negativo).
- Resultados en GROUND_TRUTH.json marcados como obsoletos.

---

#### Comando 14: `test-robustness`

| Campo | Valor |
|-------|-------|
| **Lineas** | 4228-4456 |
| **Decorador** | `@app.command("test-robustness")` |
| **Funcion** | `test_robustness()` |
| **En CLAUDE.md** | No |
| **En GROUND_TRUTH** | Si (robustness, obsoleto) |
| **Importancia** | **MEDIO** |

**Parametros:**
- `data_dir` (Argument): Dataset
- `--checkpoint`: Modelo
- `--perturbations`: Lista de perturbaciones (default: "jpeg_q50,jpeg_q30,blur_sigma1,noise_sigma10")
- `--split`: (default: "test")
- `--batch-size`, `--device`
- `--output-dir`

**Notas:**
- Aplica perturbaciones (JPEG, blur, ruido) y mide degradacion.
- Resultados en GROUND_TRUTH.json marcados como obsoletos (hechos con warped_96/warped_99).

---

#### Comando 15: `compute-canonical`

| Campo | Valor |
|-------|-------|
| **Lineas** | 4459-4708 |
| **Decorador** | `@app.command("compute-canonical")` |
| **Funcion** | `compute_canonical()` |
| **En CLAUDE.md** | Si (paso 1) |
| **En GROUND_TRUTH** | No directamente (es prerequisito del warping) |
| **Importancia** | **CRITICO** |

**Parametros:**
- `csv_path` (Argument): CSV con landmarks maestro
- `--output-dir`: Directorio salida (default: "outputs/shape_analysis")
- `--image-size`: (default: 224)
- `--visualize/--no-visualize`: Generar visualizaciones
- `--margin-scale`: Escala de margen (default: 1.05)
- `--max-iter`: Max iteraciones GPA (default: 100)
- `--tol`: Tolerancia convergencia (default: 1e-6)

**Notas:**
- Primer paso del pipeline: calcula forma canonica via GPA iterativo.
- Genera `canonical_shape_gpa.json` y `canonical_delaunay_triangles.json`.
- Usa el margen optimo de 1.05.

---

#### Comando 16: `generate-dataset`

| Campo | Valor |
|-------|-------|
| **Lineas** | 4711-5474 |
| **Decorador** | `@app.command("generate-dataset")` |
| **Funcion** | `generate_dataset()` |
| **En CLAUDE.md** | Si (paso 3) |
| **En GROUND_TRUTH** | Si (warping config, datasets) |
| **Importancia** | **CRITICO** |

**Parametros:**
- `input_dir` (Argument, opcional): Directorio dataset original
- `output_dir` (Argument, opcional): Directorio salida
- `--config`: JSON config (warping_best.json)
- `--predictions`: NPZ con predicciones cacheadas
- `--canonical`: JSON forma canonica
- `--triangles`: JSON triangulacion
- `--csv-path`: CSV maestro para splits
- `--margin-scale`: (default: 1.05)
- `--image-size`: (default: 224)
- `--splits`: Ratios train,val,test
- `--seed`: Semilla
- `--grayscale/--rgb`: Modo color
- `--clahe/--no-clahe`, `--clahe-clip`, `--clahe-tile`
- `--use-full-coverage/--no-full-coverage`
- `--fill-threshold`

**Notas:**
- Comando principal para generar dataset warped completo.
- Usa predicciones cacheadas de NPZ (no re-infiere).
- Genera splits reproducibles y metadata JSON detallada.
- Comando largo (~764 lineas) con mucha logica de procesamiento inline.
- Soporta config JSON (warping_best.json) con override por CLI.

---

#### Comando 17: `generate-cropped-dataset`

| Campo | Valor |
|-------|-------|
| **Lineas** | 5482-5891 |
| **Decorador** | `@app.command("generate-cropped-dataset")` |
| **Funcion** | `generate_cropped_dataset()` |
| **En CLAUDE.md** | No |
| **En GROUND_TRUTH** | No |
| **Importancia** | **BAJO** |

**Parametros:**
- `input_dir` (Argument): Dataset original
- `output_dir` (Argument): Directorio salida
- `--predictions`: NPZ predicciones
- `--csv-path`: CSV maestro
- `--image-size`: (default: 224)
- `--margin-scale`: (default: 1.05)
- `--splits`: Ratios
- `--seed`: Semilla
- `--grayscale/--rgb`

**Notas:**
- Genera dataset recortado (bounding box de landmarks) SIN warping.
- ~410 lineas. Util como baseline de comparacion, pero no parte del pipeline principal.

---

#### Comando 18: `apply-contrast-dataset`

| Campo | Valor |
|-------|-------|
| **Lineas** | 6281-6391 |
| **Decorador** | `@app.command("apply-contrast-dataset")` |
| **Funcion** | `apply_contrast_dataset()` |
| **En CLAUDE.md** | No |
| **En GROUND_TRUTH** | No |
| **Importancia** | **BAJO** |

**Parametros:**
- `input_dir` (Argument): Dataset entrada
- `output_dir` (Argument): Dataset salida
- `--method`: "sahs" o "clahe" (default: "sahs")
- `--exts`: Extensiones (default: ".png,.jpg,.jpeg")
- `--copy-non-images/--no-copy-non-images`
- `--overwrite`
- `--preserve-background/--no-preserve-background`
- `--background-threshold`: (default: 0)
- `--crop-foreground/--no-crop-foreground`
- `--output-size`: (default: 224)
- `--pad-to-square/--stretch`
- `--pad-value`: (default: 0)
- `--upper-factor`, `--lower-factor`: SAHS params
- `--clahe-clip`, `--clahe-tile`: CLAHE params

**Notas:**
- Aplicacion batch de SAHS o CLAHE a todo un dataset.
- Preserva estructura de directorios.
- La implementacion real esta en `_apply_contrast_dataset()` (L6128-6279).
- Incluye ~7 funciones helper privadas (L5899-6125) que representan ~230 lineas de utilidades de procesamiento de imagen inline en cli.py.

---

#### Comando 19: `crop-dataset`

| Campo | Valor |
|-------|-------|
| **Lineas** | 6531-6606 |
| **Decorador** | `@app.command("crop-dataset")` |
| **Funcion** | `crop_dataset()` |
| **En CLAUDE.md** | No |
| **En GROUND_TRUTH** | No |
| **Importancia** | **BAJO** |

**Parametros:**
- `input_dir` (Argument): Imagenes warped
- `output_dir` (Argument): Imagenes recortadas
- `--exts`: Extensiones
- `--copy-non-images/--no-copy-non-images`
- `--overwrite`
- `--background-threshold`: (default: 0)
- `--output-size`: (default: 224)
- `--pad-to-square/--stretch`
- `--pad-value`: (default: 0)

**Notas:**
- Recorta fondo negro de imagenes warped y resize.
- Implementacion en `_crop_dataset_impl()` (L6399-6528).

---

#### Comando 20: `compare-architectures`

| Campo | Valor |
|-------|-------|
| **Lineas** | 7073-7518 |
| **Decorador** | `@app.command("compare-architectures")` |
| **Funcion** | `compare_architectures()` |
| **En CLAUDE.md** | No |
| **En GROUND_TRUTH** | No |
| **Importancia** | **MEDIO** |

**Parametros:**
- `data_dir` (Argument): Dataset warped
- `--output-dir`: Directorio salida
- `--architectures`: Lista de arquitecturas (default: "resnet18,resnet50,efficientnet_b0,densenet121")
- `--epochs`: (default: 30)
- `--batch-size`: (default: 16)
- `--lr`: (default: 1e-4)
- `--patience`: (default: 10)
- `--device`: auto/cuda/cpu/mps
- `--seed`: Semilla
- `--quick/--no-quick`: Modo rapido con datos reducidos

**Notas:**
- Entrena y compara multiples arquitecturas en el mismo dataset.
- Incluye ~460 lineas de funciones helper (L6614-7070: constantes, _train_single_architecture, _generate_comparison_figures, _generate_comparison_reports).
- Genera figuras matplotlib, JSON y CSV comparativos.

---

#### Comando 21: `gradcam`

| Campo | Valor |
|-------|-------|
| **Lineas** | 7521-7798 |
| **Decorador** | `@app.command()` |
| **Funcion** | `gradcam()` |
| **En CLAUDE.md** | No |
| **En GROUND_TRUTH** | No |
| **Importancia** | **MEDIO** |

**Parametros:**
- `image_path` (Argument): Imagen de entrada
- `--checkpoint`: Modelo clasificador
- `--output-dir`: Directorio salida
- `--device`: auto/cuda/cpu/mps
- `--target-class`: Clase objetivo (default: None = prediccion)
- `--backbone`: (default: "resnet18")

**Notas:**
- Genera heatmaps Grad-CAM para interpretar decisiones del clasificador.
- ~278 lineas con logica de extraccion de features y visualizacion.

---

#### Comando 22: `analyze-errors`

| Campo | Valor |
|-------|-------|
| **Lineas** | 7801-8063 |
| **Decorador** | `@app.command("analyze-errors")` |
| **Funcion** | `analyze_errors()` |
| **En CLAUDE.md** | No |
| **En GROUND_TRUTH** | No |
| **Importancia** | **BAJO** |

**Parametros:**
- `data_dir` (Argument): Dataset
- `--checkpoint`: Modelo clasificador
- `--split`: (default: "test")
- `--batch-size`, `--device`
- `--output-dir`
- `--gradcam/--no-gradcam`: Generar Grad-CAM para errores
- `--max-errors`: Maximo de errores a analizar (default: 50)

**Notas:**
- Identifica y analiza imagenes mal clasificadas.
- Opcionalmente genera Grad-CAM para cada error.
- ~263 lineas.

---

#### Comando 23: `pfs-analysis`

| Campo | Valor |
|-------|-------|
| **Lineas** | 8065-8447 |
| **Decorador** | `@app.command("pfs-analysis")` |
| **Funcion** | `pfs_analysis()` |
| **En CLAUDE.md** | No |
| **En GROUND_TRUTH** | Si (PFS, obsoleto) |
| **Importancia** | **BAJO** |

**Parametros:**
- `data_dir` (Argument): Dataset warped
- `--checkpoint`: Clasificador
- `--lung-masks-dir`: Directorio mascaras
- `--split`: (default: "test")
- `--batch-size`, `--device`
- `--output-dir`
- `--backbone`: (default: "resnet18")

**Notas:**
- Calcula Pulmonary Focus Score (proporcion de atencion Grad-CAM en region pulmonar).
- Resultados en GROUND_TRUTH.json marcados como obsoletos (PFS ~0.49, no hay evidencia de foco pulmonar).
- ~383 lineas.

---

#### Comando 24: `generate-lung-masks`

| Campo | Valor |
|-------|-------|
| **Lineas** | 8450-8585 |
| **Decorador** | `@app.command("generate-lung-masks")` |
| **Funcion** | `generate_lung_masks()` |
| **En CLAUDE.md** | No |
| **En GROUND_TRUTH** | No |
| **Importancia** | **BAJO** |

**Parametros:**
- `data_dir` (Argument): Dataset warped
- `--output-dir`: Directorio salida
- `--threshold`: Umbral binarizacion (default: 10)
- `--kernel-size`: Tamano kernel morfologico (default: 5)
- `--min-area`: Area minima contorno (default: 100)

**Notas:**
- Genera mascaras binarias aproximadas para PFS analysis.
- ~136 lineas, dependencia de pfs-analysis.

---

#### Comando 25: `optimize-margin`

| Campo | Valor |
|-------|-------|
| **Lineas** | 8593-9447 |
| **Decorador** | `@app.command("optimize-margin")` |
| **Funcion** | `optimize_margin()` |
| **En CLAUDE.md** | Mencionado en Critical Implementation Details |
| **En GROUND_TRUTH** | Si (margin_scale_optimal=1.05) |
| **Importancia** | **MEDIO** |

**Parametros:**
- `csv_path` (Argument): CSV landmarks
- `image_dir` (Argument): Directorio imagenes
- `--output-dir`
- `--margins`: Lista de margenes (default: "1.00,1.05,1.10,1.15,1.20,1.25,1.30")
- `--image-size`: (default: 224)
- `--batch-size`, `--device`
- `--seed`, `--split-seed`
- `--quick/--no-quick`: Modo rapido
- `--epochs`: (default: 15)
- `--clahe/--no-clahe`, `--clahe-clip`, `--clahe-tile`

**Notas:**
- Grid search que encontro OPTIMAL_MARGIN_SCALE=1.05.
- Entrena modelos con diferentes margenes y compara errores.
- ~855 lineas, uno de los comandos mas largos.
- Ya se ejecuto y su resultado esta validado; probablemente no necesita re-ejecucion.

---

#### Comando 26: `extract-dataset-splits`

| Campo | Valor |
|-------|-------|
| **Lineas** | 9449-9553 |
| **Decorador** | `@app.command("extract-dataset-splits")` |
| **Funcion** | `extract_dataset_splits()` |
| **En CLAUDE.md** | No |
| **En GROUND_TRUTH** | No |
| **Importancia** | **BAJO** |

**Parametros:**
- `data_dir` (Argument): Dataset warped
- `--output-dir`: Directorio salida
- `--format`: "json" o "csv" (default: "json")

**Notas:**
- Extrae informacion de splits para uso en GUI o herramientas externas.
- ~105 lineas, utilidad auxiliar.

---

#### Comando 27: `generate-landmark-visualization-dataset`

| Campo | Valor |
|-------|-------|
| **Lineas** | 9555-9957 |
| **Decorador** | `@app.command("generate-landmark-visualization-dataset")` |
| **Funcion** | `generate_landmark_visualization_dataset()` |
| **En CLAUDE.md** | Si (paso 6b) |
| **En GROUND_TRUTH** | No |
| **Importancia** | **ALTO** |

**Parametros:**
- `input_dir` (Argument, opcional): Dataset original
- `output_dir` (Argument, opcional): Directorio salida
- `--config`: JSON config (landmark_viz_best.json)
- `--predictions`: NPZ predicciones
- `--warped-dataset`: Dataset warped para splits
- `--splits`: Ratios
- `--seed`: Semilla
- `--classes`: Clases
- `--image-size`: (default: 299)
- `--preserve-size/--resize`
- `--landmark-color`: (default: "red")
- `--landmark-size`: (default: 3)
- `--cross-thickness`: (default: 2)

**Notas:**
- Genera dataset con landmarks visualizados sobre imagenes originales.
- Usa splits del warped dataset para consistencia.
- ~403 lineas. Soporta config JSON.

---

#### Comando 28: `generate-delaunay-mesh-dataset`

| Campo | Valor |
|-------|-------|
| **Lineas** | 9959-10429 |
| **Decorador** | `@app.command("generate-delaunay-mesh-dataset")` |
| **Funcion** | `generate_delaunay_mesh_dataset()` |
| **En CLAUDE.md** | No |
| **En GROUND_TRUTH** | No |
| **Importancia** | **MEDIO** |

**Parametros:**
- `input_dir` (Argument, opcional): Dataset original
- `output_dir` (Argument, opcional): Directorio salida
- `--config`: JSON config
- `--predictions`: NPZ predicciones
- `--triangles`: JSON triangulacion
- `--warped-dataset`: Dataset warped para splits
- `--splits`: Ratios
- `--seed`: Semilla
- `--classes`: Clases
- `--image-size`: (default: 299)
- `--preserve-size/--resize`
- `--mesh-color`: (default: "cyan")
- `--mesh-thickness`: (default: 1)

**Notas:**
- Dibuja malla de Delaunay sobre imagenes originales.
- ~471 lineas con logica inline de procesamiento.
- Soporta config JSON.

---

#### Comando 29: `generate-landmark-comparison-dataset`

| Campo | Valor |
|-------|-------|
| **Lineas** | 10432-10567 |
| **Decorador** | `@app.command("generate-landmark-comparison-dataset")` |
| **Funcion** | `generate_landmark_comparison_dataset()` |
| **En CLAUDE.md** | No |
| **En GROUND_TRUTH** | No |
| **Importancia** | **MEDIO** |

**Parametros:**
- `input_dir` (Argument): Dataset original
- `output_dir` (Argument): Directorio salida
- `--ground-truth-csv`: CSV GT (default: coordenadas_maestro.csv)
- `--predictions`: NPZ predicciones
- `--seed`: Semilla
- `--pred-color`: Color predicciones (default: "red")
- `--gt-color`: Color GT (default: "green")
- `--cross-size`: (default: 5)
- `--cross-thickness`: (default: 2)
- `--show-error-lines/--no-error-lines`
- `--max-per-split`: Limite por split

**Notas:**
- Genera visualizaciones comparativas predicciones vs ground truth.
- Delega a `src_v2.visualization.comparison_viz.generate_comparison_dataset()`.
- ~136 lineas.

---

#### Comando 30: `generate-landmark-comparison-dataset-aligned`

| Campo | Valor |
|-------|-------|
| **Lineas** | 10569-10734 |
| **Decorador** | `@app.command("generate-landmark-comparison-dataset-aligned")` |
| **Funcion** | `generate_landmark_comparison_dataset_aligned()` |
| **En CLAUDE.md** | No |
| **En GROUND_TRUTH** | No |
| **Importancia** | **MEDIO** |

**Parametros:**
- `warped_dataset_dir` (Argument): Dataset warped
- `output_dir` (Argument): Directorio salida
- `--original-images`: Directorio imagenes 299x299
- `--ground-truth-csv`: CSV GT
- `--predictions`: NPZ predicciones
- `--pred-color`: (default: "red")
- `--gt-color`: (default: "green")
- `--cross-size`: (default: 5)
- `--cross-thickness`: (default: 2)
- `--show-error-lines/--no-error-lines`

**Notas:**
- Similar al comando 29 pero usa splits del warped dataset (no crea nuevos).
- Permite correlacionar errores de landmarks con errores de clasificacion.
- Delega a `src_v2.visualization.comparison_viz.generate_comparison_dataset_aligned()`.
- ~166 lineas.

---

#### Comando 31: `generate-viz-dataset`

| Campo | Valor |
|-------|-------|
| **Lineas** | 10737-10886 |
| **Decorador** | `@app.command()` |
| **Funcion** | `generate_viz_dataset()` |
| **En CLAUDE.md** | No |
| **En GROUND_TRUTH** | No |
| **Importancia** | **MEDIO** |

**Parametros:**
- `input_dir` (Argument): Dataset original
- `output_dir` (Argument): Directorio salida
- `--predictions`: NPZ predicciones
- `--canonical`: JSON forma canonica
- `--triangles`: JSON triangulacion
- `--csv-path`: CSV maestro
- `--viz-size`: (default: 299)
- `--model-size`: (default: 224)
- `--seed`: (default: 42)
- `--max-per-split`: Limite por split
- `--dpi`: (default: 300)
- `--no-numbers`: No mostrar numeros en landmarks

**Notas:**
- Genera visualizaciones cientificas con landmarks + malla Delaunay.
- Delega a `src_v2.visualization.scientific_viz.generate_viz_dataset()`.
- ~150 lineas.

---

### Analisis Estructural de cli.py

#### Distribucion del codigo por tipo

| Tipo | Lineas aprox. | % del total |
|------|---------------|-------------|
| Comandos CRITICOS del pipeline (train, evaluate-ensemble, train-classifier, evaluate-classifier, compute-canonical, generate-dataset) | ~3,900 | 36% |
| Comandos ALTOS (evaluate, predict, evaluate-classifier-ensemble, cross-validate-classifier, generate-landmark-visualization-dataset) | ~2,200 | 20% |
| Comandos MEDIOS (classify, warp, cross-evaluate, evaluate-external, test-robustness, compare-architectures, gradcam, optimize-margin, vizualizaciones) | ~3,600 | 33% |
| Comandos BAJOS (version, generate-cropped-dataset, apply-contrast-dataset, crop-dataset, analyze-errors, pfs-analysis, generate-lung-masks, extract-dataset-splits) | ~1,000 | 9% |
| Funciones helper + imports + boilerplate | ~200 | 2% |

#### Problemas identificados

1. **Tamano excesivo (10,895 lineas)**: Este es el problema mas grave. Un archivo CLI de casi 11K lineas es dificil de mantener, navegar y testear. La recomendacion seria dividir en modulos tematicos:
   - `cli_landmarks.py`: train, evaluate, predict, evaluate-ensemble
   - `cli_classifier.py`: train-classifier, evaluate-classifier, evaluate-classifier-ensemble, cross-validate-classifier, classify
   - `cli_warping.py`: compute-canonical, generate-dataset, warp, optimize-margin
   - `cli_analysis.py`: cross-evaluate, evaluate-external, test-robustness, gradcam, analyze-errors, pfs-analysis
   - `cli_datasets.py`: generate-cropped-dataset, apply-contrast-dataset, crop-dataset, extract-dataset-splits
   - `cli_visualization.py`: generate-landmark-visualization-dataset, generate-delaunay-mesh-dataset, generate-landmark-comparison-dataset, generate-landmark-comparison-dataset-aligned, generate-viz-dataset

2. **Funciones de procesamiento inline**: Las funciones `_apply_sahs_to_gray()`, `_apply_clahe_to_gray()`, `_crop_to_foreground()`, `_build_foreground_mask()`, `_to_gray()` (L5899-6125) son utilidades de procesamiento de imagen que deberian estar en `src_v2/processing/` y no en el CLI.

3. **Logica de entrenamiento inline**: `_train_single_architecture()` (L6635-6847) es un loop de entrenamiento completo de ~213 lineas que duplica funcionalidad que deberia estar en `src_v2/training/`.

4. **Logica de visualizacion inline**: `_generate_comparison_figures()` (L6850-7022) y `_generate_comparison_reports()` (L7025-7070) deberian estar en `src_v2/visualization/`.

5. **Comandos con resultado obsoleto**: Los comandos `cross-evaluate` (12), `evaluate-external` (13), `test-robustness` (14), y `pfs-analysis` (23) tienen resultados marcados como obsoletos en GROUND_TRUTH.json. Los comandos son funcionales pero sus resultados validados corresponden a datasets/metodos anteriores (warped_96/warped_99), no al metodo actual (warped_lung_best).

6. **Margen por defecto inconsistente**: El comando `warp` (4) usa `DEFAULT_MARGIN_SCALE=1.25` (legacy) mientras que `compute-canonical` (15) y `generate-dataset` (16) usan 1.05 (optimo). Esto puede causar confusion si alguien usa `warp` directamente.

7. **Duplicacion entre comandos de visualizacion**: Los comandos 27-31 (generate-landmark-visualization-dataset, generate-delaunay-mesh-dataset, generate-landmark-comparison-dataset, generate-landmark-comparison-dataset-aligned, generate-viz-dataset) tienen logica similar con variaciones menores. Podrian consolidarse con un sistema de "capas" de visualizacion.

---

### Resumen de Consistencia con Archivos de Referencia

#### Consistencia con CLAUDE.md

| Aspecto | Estado |
|---------|--------|
| Pipeline de 6 pasos documentado | Los 6 comandos mencionados existen y funcionan |
| Entry point `python -m src_v2` | Correcto via `__main__.py` |
| Referencia a `detect_architecture_from_checkpoint()` | Presente en cli.py L154-198 |
| Referencia a configs JSON | Implementado en train, train-classifier, generate-dataset, etc. |
| Mention de `SYMMETRIC_PAIRS`, `CENTRAL_LANDMARKS` | Definidos correctamente en constants.py |
| 25 de 31 comandos NO documentados en CLAUDE.md | Gran brecha de documentacion |

#### Consistencia con GROUND_TRUTH.json

| Aspecto | Estado |
|---------|--------|
| Landmark ensemble 3.61 px | Producido por `evaluate-ensemble` |
| Classifier warped_lung_best 98.05% | Producido por `evaluate-classifier` |
| margin_scale_optimal=1.05 | Coincide con `compute-canonical` y `generate-dataset` defaults |
| CLAHE clip=2.0, tile=4 | Coincide con constants.py defaults |
| Cross-validation 98.60% | Producido por `cross-validate-classifier` |
| Classifier ensemble 98.26% | Producido por `evaluate-classifier-ensemble` |
| Valores obsoletos (robustness, cross_eval, PFS, external) | Comandos existen pero resultados corresponden a metodos anteriores |

---

### Recomendaciones Priorizadas

1. **P1 (Alta)**: Actualizar docstring de `__init__.py` con valores actuales (3.61 px, 98.05%).
2. **P1 (Alta)**: Refactorizar `cli.py` dividiendo en modulos tematicos (~6 archivos) para mejorar mantenibilidad.
3. **P2 (Media)**: Mover funciones de procesamiento inline (`_apply_sahs_to_gray`, etc.) a `src_v2/processing/`.
4. **P2 (Media)**: Mover `_train_single_architecture()` a `src_v2/training/`.
5. **P2 (Media)**: Corregir DEFAULT_MARGIN_SCALE en comando `warp` de 1.25 a 1.05 o documentar claramente por que difiere.
6. **P3 (Baja)**: Documentar los 25 comandos faltantes en CLAUDE.md.
7. **P3 (Baja)**: Considerar consolidar los 5 comandos de visualizacion en un sistema mas modular.
8. **P3 (Baja)**: Re-ejecutar comandos obsoletos (robustness, cross-evaluate, pfs-analysis) con warped_lung_best y actualizar GROUND_TRUTH.json, o marcar los comandos como "pendiente re-validacion".
