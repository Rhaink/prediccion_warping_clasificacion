# 22. Data, Checkpoints, Build and Distribution

Analisis de las estructuras de datos, modelos entrenados, y artefactos de compilacion.

---

## 1. data/

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/data/`
- **Tamano total**: Estimado ~2-3 GB (15,153 imagenes PNG 299x299 + dataset externo + coordenadas)
- **Importancia**: CRITICO
- **En .gitignore**: Si (`data/`, `data/dataset/`, `data/coordenadas/`)

### Estructura

```
data/
├── coordenadas/
│   └── coordenadas_maestro.csv          # Landmark annotations (15 landmarks x N images)
├── dataset/
│   └── COVID-19_Radiography_Dataset/
│       ├── COVID/
│       │   ├── images/                   # ~3,616 imagenes COVID PNG
│       │   └── masks/                    # Mascaras de segmentacion correspondientes
│       ├── Normal/
│       │   ├── images/                   # ~10,192 imagenes Normal PNG
│       │   └── masks/
│       ├── Viral Pneumonia/
│       │   ├── images/                   # ~1,345 imagenes Viral Pneumonia PNG
│       │   └── masks/
│       ├── COVID.metadata.xlsx
│       ├── Normal.metadata.xlsx
│       ├── Viral Pneumonia.metadata.xlsx
│       └── README.md.txt
├── external_datasets/
│   └── covid-chestxray-dataset/          # Repositorio git clonado (ieee8023/covid-chestxray-dataset)
│       ├── images/                       # Centenares de imagenes JPG/PNG (multi-formato)
│       ├── annotations/
│       │   ├── covid-severity-scores.csv
│       │   ├── imageannotation_ai_lung_bounding_boxes.json
│       │   └── lungVAE-masks/            # Mascaras VAE para pulmones
│       ├── metadata.csv
│       ├── README.md
│       ├── SCHEMA.md
│       └── .git/                         # Repositorio git completo con packfile
├── tmp_subset_cli.csv                    # Subconjunto temporal para pruebas CLI
└── tmp_subset_warp.csv                   # Subconjunto temporal para pruebas warping
```

### Contenido clave

**coordenadas_maestro.csv**: Archivo CSV maestro de anotaciones de landmarks. Cada fila contiene un indice, 30 coordenadas (15 puntos x,y) y un identificador de imagen. Formato: `idx,L1x,L1y,L2x,L2y,...,L15x,L15y,image_id`. Este archivo es la base del entrenamiento de landmarks.

**COVID-19 Radiography Dataset**: Dataset principal de 15,153 imagenes en total (confirmado por `dataset_summary.json` y `VERIFICATION_REPORT.txt`):
- COVID: 3,616 imagenes
- Normal: 10,192 imagenes
- Viral Pneumonia: 1,345 imagenes
- Cada clase tiene subdirectorios `images/` y `masks/`
- Formato PNG, resolucion original 299x299

**covid-chestxray-dataset**: Dataset externo (Joseph Paul Cohen et al.) clonado como repositorio git. Utilizado para validacion externa (Session 55). Contiene imagenes en formatos mixtos (JPG, PNG) y metadatos CSV. Referenciado en `GROUND_TRUTH.json` bajo `external_validation`.

### Notas

- **Archivos temporales**: `tmp_subset_cli.csv` y `tmp_subset_warp.csv` son subconjuntos de ~3 filas cada uno, aparentemente residuos de pruebas. ELIMINABLES.
- **Dataset externo con .git**: El directorio `covid-chestxray-dataset/.git/` contiene un packfile completo (~centenares de MB potencialmente). Solo se usa para referencia, no para el pipeline principal.
- **Mascaras no usadas en pipeline**: Las mascaras (`masks/`) del dataset principal no se utilizan en el pipeline actual (landmark + warping + clasificacion). Son parte del dataset original de Kaggle.
- **No hay directorio `data/dataset/` en la raiz de git** (en .gitignore correctamente).

---

## 2. checkpoints/

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/checkpoints/`
- **Tamano total**: ~629 MB (segun CLAUDE.md, post-limpieza del 2026-01-20)
- **Importancia**: CRITICO
- **En .gitignore**: Si (`checkpoints/`)

### Estructura

```
checkpoints/
├── best_model_7.84px.pt                  # Historico temprano (early baseline)
├── best_model_7.21px.pt                  # Historico temprano (early baseline)
├── best_model_session10.pt               # Historico session 10
├── session10/
│   └── ensemble/
│       ├── seed123/
│       │   └── final_model.pt            # CRITICO - Modelo del ensemble actual
│       └── seed456/
│           └── final_model.pt            # HISTORICO - Mejor individual (4.04 px)
├── session13/
│   ├── seed321/
│   │   └── final_model.pt               # CRITICO - Modelo del ensemble actual
│   ├── seed789/
│   │   └── final_model.pt               # HISTORICO - Ensemble obsoleto (3.71 px)
│   └── hierarchical/
│       ├── final_model.pt               # HISTORICO - Modelo jerarquico (46.62 px, fallido)
│       └── results.json                 # Resultados del modelo jerarquico
├── repro_split111/
│   └── session14/
│       └── seed111/
│           ├── final_model.pt            # CRITICO - Modelo del ensemble actual
│           ├── repro_split111.zip        # Backup ZIP del modelo
│           └── repro_split111/
│               └── session14/
│                   └── seed111/
│                       └── final_model.pt  # DUPLICADO anidado
└── repro_split666/
    └── session16/
        └── seed666/
            ├── final_model.pt            # CRITICO - Modelo del ensemble actual
            ├── repro_split666.zip        # Backup ZIP del modelo
            └── repro_split666/
                └── session16/
                    └── seed666/
                        └── final_model.pt  # DUPLICADO anidado
```

### Contenido clave

**Ensemble actual (3.61 px - `ensemble_best.json`)** - Los 4 modelos criticos:

| Modelo | Ruta | Estado | Rol |
|--------|------|--------|-----|
| seed123 | `session10/ensemble/seed123/final_model.pt` | PRESENTE | Ensemble actual |
| seed321 | `session13/seed321/final_model.pt` | PRESENTE | Ensemble actual |
| seed111 | `repro_split111/session14/seed111/final_model.pt` | PRESENTE | Ensemble actual |
| seed666 | `repro_split666/session16/seed666/final_model.pt` | PRESENTE | Ensemble actual |

**Modelos historicos** (no usados en pipeline actual):

| Modelo | Ruta | Error | Nota |
|--------|------|-------|------|
| seed456 | `session10/ensemble/seed456/final_model.pt` | 4.04 px | Mejor individual, reemplazado en ensemble |
| seed789 | `session13/seed789/final_model.pt` | 3.71 px (combo) | Ensemble obsoleto |
| hierarchical | `session13/hierarchical/final_model.pt` | 46.62 px | Experimento fallido |
| best_model_7.84px | `best_model_7.84px.pt` | 7.84 px | Baseline temprano |
| best_model_7.21px | `best_model_7.21px.pt` | 7.21 px | Baseline temprano |
| best_model_session10 | `best_model_session10.pt` | ? | Baseline session 10 |

### Notas

- **Duplicados anidados**: Tanto `repro_split111` como `repro_split666` tienen copias duplicadas del `final_model.pt` dentro de una estructura anidada redundante (`seed111/repro_split111/session14/seed111/final_model.pt`). Esto parece un artefacto del proceso de extraccion del ZIP. Los ZIP (`repro_split111.zip`, `repro_split666.zip`) tambien estan presentes. Son 2 copias extra de cada modelo. ELIMINABLES (los duplicados anidados y/o los ZIPs).
- **Modelo jerarquico**: El modelo `session13/hierarchical/final_model.pt` obtuvo 46.62 px de error (vs 3.61 px del ensemble), confirmando que el enfoque jerarquico fue descartado. El `results.json` documenta la configuracion. Podria eliminarse, pero ocupa poco espacio.
- **Baselines historicos en raiz**: Los 3 archivos `.pt` en la raiz de checkpoints (`best_model_7.84px.pt`, `best_model_7.21px.pt`, `best_model_session10.pt`) son baselines tempranos. ELIMINABLES si no se requieren para comparacion.
- **Post-limpieza**: Segun `CHECKPOINTS_CLEANUP_REPORT.md`, ya se liberaron ~133 GB el 2026-01-20. El backup existe en `checkpoints_backup_20260120.tar.gz`.

---

## 3. build/

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/build/`
- **Tamano total**: Estimado ~5-10 GB (5 releases con wheels PyTorch ~300 MB c/u + modelos)
- **Importancia**: MEDIO
- **En .gitignore**: Si (`build/`)

### Estructura

```
build/
├── models_staging/                       # Modelos preparados para distribucion
│   ├── landmarks/
│   │   ├── resnet18_seed123_best.pt
│   │   ├── resnet18_seed321_best.pt
│   │   ├── resnet18_seed111_best.pt
│   │   └── resnet18_seed666_best.pt
│   ├── shape_analysis/
│   │   ├── canonical_shape_gpa.json
│   │   └── canonical_delaunay_triangles.json
│   └── classifier/
│       └── best_classifier.pt
└── releases/
    ├── python-3.12.8-embed-amd64.zip     # Python embebido para Windows
    ├── get-pip.py                         # Instalador de pip
    ├── INSTRUCCIONES_v1.0.13.txt          # Instrucciones legacy
    │
    ├── covid19-demo-v1.0.12-portable-windows/     # Release v1.0.12 (descomprimido)
    ├── covid19-demo-v1.0.12-portable-windows.zip  # Release v1.0.12 (ZIP)
    ├── covid19-demo-v1.0.13-portable-windows/     # Release v1.0.13 (descomprimido)
    ├── covid19-demo-v1.0.13-portable-windows.zip  # Release v1.0.13 (ZIP)
    ├── covid19-demo-v14-portable-windows/          # Release v14 (descomprimido)
    ├── covid19-demo-v14-portable-windows.zip       # Release v14 (ZIP)
    ├── covid19-demo-v15-portable-windows/          # Release v15 (descomprimido)
    ├── covid19-demo-v15-portable-windows.zip       # Release v15 (ZIP)
    ├── covid19-demo-v16-portable-windows/          # Release v16 - ULTIMA (descomprimido)
    └── covid19-demo-v16-portable-windows.zip       # Release v16 - ULTIMA (ZIP)
```

### Estructura de un release (v16, version actual)

```
covid19-demo-v16-portable-windows/
├── INSTALL.bat                           # Instalador automatico
├── RUN_DEMO.bat                          # Launcher local
├── RUN_DEMO_SHARE.bat                    # Launcher con compartir Gradio (72h link)
├── README.txt
├── VERSION.txt                           # Metadatos JSON: version, checksums, features
├── MANIFEST.txt                          # Lista de 87 wheels incluidos
├── GROUND_TRUTH.json
├── requirements.txt
├── get-pip.py
├── install_deps.py                       # Script Python para instalar dependencias
├── configs/
│   ├── classifier_warped_base.json
│   ├── classifier_original_base.json
│   ├── ensemble_best.json
│   ├── hierarchical_train_base.json
│   ├── landmarks_train_base.json
│   └── final_config.json
├── models/
│   ├── landmarks/
│   │   ├── resnet18_seed123_best.pt      # 4 modelos de landmarks
│   │   ├── resnet18_seed321_best.pt
│   │   ├── resnet18_seed111_best.pt
│   │   └── resnet18_seed666_best.pt
│   ├── classifier/
│   │   └── best_classifier.pt            # Clasificador entrenado
│   └── shape_analysis/
│       ├── canonical_shape_gpa.json
│       └── canonical_delaunay_triangles.json
├── src_v2/                               # Codigo fuente completo
│   ├── __init__.py, __main__.py, cli.py, constants.py
│   ├── data/ (dataset.py, transforms.py, utils.py)
│   ├── models/ (classifier.py, resnet_landmark.py, hierarchical.py, losses.py)
│   ├── processing/ (warp.py, gpa.py)
│   ├── training/ (trainer.py, callbacks.py)
│   ├── evaluation/ (metrics.py, ensemble.py)
│   ├── visualization/ (gradcam.py, pfs_analysis.py, etc.)
│   ├── gui/ (app.py, model_manager.py, inference_pipeline.py, etc.)
│   └── utils/ (geometry.py)
└── wheels/                               # 87 wheels pre-descargados para offline
    ├── torch-2.4.1+cpu-cp312-cp312-win_amd64.whl  # ~300 MB
    ├── torchvision-0.19.1+cpu-cp312-cp312-win_amd64.whl
    ├── gradio-6.5.0-py3-none-any.whl
    ├── numpy-2.4.1-cp312-cp312-win_amd64.whl
    ├── opencv_python_headless-4.13.0.90-cp37-abi3-win_amd64.whl
    ├── pandas-3.0.0-cp312-cp312-win_amd64.whl
    ├── scipy-1.17.0-cp312-cp312-win_amd64.whl
    ├── scikit_learn-1.8.0-cp312-cp312-win_amd64.whl
    ├── tzdata-2025.3-py2.py3-none-any.whl  # Hotfix critico para pandas 3.0+
    └── ... (77 mas)
```

### Contenido clave

**models_staging/**: Directorio de staging con 7 artefactos listos para empaquetar en releases. Contiene los 4 modelos de landmarks, el clasificador, y los archivos de forma canonica.

**Releases**: 5 versiones de paquetes portables para Windows (v1.0.12, v1.0.13, v14, v15, v16). La version actual es **v16** (build date: 2026-01-29). Cada release es un paquete completamente autocontenido con:
- Python 3.12.8 embebido
- 87 wheels Python para instalacion offline
- Modelos pre-entrenados con checksums SHA-256
- GUI Gradio con opciones local y publica

### Notas

- **Duplicacion masiva**: Cada release descomprimido contiene copias identicas de los modelos (~4 modelos landmarks + 1 clasificador + 2 shape analysis). Con 5 releases, hay 5x la misma data de modelos. Ademas, cada release tiene un ZIP correspondiente, lo que duplica nuevamente el contenido.
- **v16 no tiene shape_analysis en models/**: La version v16 SI incluye `canonical_shape_gpa.json` y `canonical_delaunay_triangles.json` en `models/shape_analysis/`.
- **Nota sobre versionado**: Los nombres cambian de formato (v1.0.12, v1.0.13 -> v14, v15, v16), sugiriendo un cambio de esquema de versionado.
- **Releases anteriores**: v1.0.12, v1.0.13, v14, v15 son versiones historicas. Potencialmente ELIMINABLES si v16 es la version final para la defensa de tesis. Esto ahorraria ~4 GB estimados.

---

## 4. dist/

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/dist/`
- **Tamano total**: Minimo (directorio vacio o con subdirectorio vacio)
- **Importancia**: BAJO
- **En .gitignore**: Si (`dist/`)

### Estructura

```
dist/
└── COVID19_Demo/                         # Directorio vacio (sin archivos)
```

### Contenido clave

El directorio `dist/COVID19_Demo/` existe pero esta vacio (sin archivos hijos encontrados por glob). Esto probablemente fue creado durante un intento anterior de empaquetado (posiblemente PyInstaller o similar) que fue abandonado en favor del enfoque de paquete portable en `build/releases/`.

### Notas

- **Directorio residual**: ELIMINABLE completamente. No contiene datos utiles.

---

## 5. outputs/

- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/outputs/`
- **Tamano total**: Estimado ~5-8 GB (15,153 imagenes warped x2 + 15,153 originales + modelos clasificador)
- **Importancia**: ALTO
- **En .gitignore**: Si (`outputs/`)

### Estructura

```
outputs/
├── shape_analysis/                       # Resultados de GPA
│   ├── canonical_shape_gpa.json          # Forma canonica (15 puntos)
│   ├── canonical_delaunay_triangles.json # Triangulacion Delaunay
│   ├── aligned_shapes.npz               # Formas alineadas
│   └── canonical_shape.npy              # Formato numpy de forma canonica
│
├── landmark_predictions/                 # Cache de predicciones
│   └── session_warping/
│       └── predictions.npz              # 15,153 predicciones de landmarks (~tamanio moderado)
│
├── warped_lung_best/                    # Dataset warped principal
│   └── session_warping/
│       ├── dataset_summary.json          # Metadatos completos del dataset
│       ├── train/                        # 11,364 imagenes warped
│       │   ├── COVID/                    # 2,712 imagenes
│       │   ├── Normal/                   # 7,644 imagenes
│       │   └── Viral_Pneumonia/          # 1,008 imagenes
│       ├── val/                          # 1,894 imagenes warped
│       │   ├── COVID/                    # 452 imagenes
│       │   ├── Normal/                   # 1,274 imagenes
│       │   └── Viral_Pneumonia/          # 168 imagenes
│       └── test/                         # 1,895 imagenes warped
│           ├── COVID/                    # 452 imagenes
│           ├── Normal/                   # 1,274 imagenes
│           └── Viral_Pneumonia/          # 169 imagenes
│
├── classifier_warped_lung_best/         # Clasificadores entrenados
│   ├── session_2026-01-12/
│   │   ├── best_classifier.pt           # Clasificador base
│   │   └── results.json
│   ├── sweeps_2026-01-12/               # Sweep de hiperparametros (12 configs)
│   │   ├── lr5e-5_seed42_on/  (train.log, best_classifier.pt, results.json)
│   │   ├── lr5e-5_seed42_off/
│   │   ├── lr5e-5_seed123_on/
│   │   ├── lr5e-5_seed123_off/
│   │   ├── lr5e-5_seed321_on/
│   │   ├── lr5e-5_seed321_off/
│   │   ├── lr2e-4_seed42_on/
│   │   ├── lr2e-4_seed42_off/
│   │   ├── lr2e-4_seed123_on/
│   │   ├── lr2e-4_seed123_off/
│   │   ├── lr2e-4_seed321_on/  <-- BEST (98.05%, referenciado en GROUND_TRUTH.json)
│   │   └── lr2e-4_seed321_off/
│   ├── sweeps_2026-01-13_lr_tune/       # Ajuste fino de LR (3 configs)
│   │   ├── lr1.5e-4_seed321_on/
│   │   ├── lr2e-4_seed321_on/
│   │   └── lr2.5e-4_seed321_on/
│   └── sweeps_2026-01-13_lr_tune_confirm/ # Confirmacion (2 configs)
│       ├── lr1.5e-4_seed42_on/
│       └── lr1.5e-4_seed123_on/
│
├── classifier_cv/                       # Cross-validation del clasificador
│   ├── cross_validation_results.json
│   ├── cross_validation_test_results.json
│   ├── ensemble_test_results.json
│   ├── ensemble_test_results_no_tta.json
│   ├── ensemble_test_results_tta.json
│   ├── ensemble_predictions.csv
│   └── ensemble_predictions_tta.csv
│
├── dataset_splits_for_gui/              # Dataset pre-dividido para GUI/defensa
│   ├── README.md
│   ├── VERIFICATION_REPORT.txt          # Verificacion de integridad (PASS)
│   ├── CERTIFICACION_DEFENSA_TESIS.md
│   ├── original/                        # 15,153 imagenes originales organizadas
│   │   ├── train/ (COVID, Normal, Viral_Pneumonia)
│   │   ├── val/   (COVID, Normal, Viral_Pneumonia)
│   │   └── test/  (COVID, Normal, Viral_Pneumonia)
│   └── warped/                          # 15,153 imagenes warped organizadas
│       ├── train/ (COVID, Normal, Viral_Pneumonia)
│       ├── val/   (COVID, Normal, Viral_Pneumonia)
│       └── test/  (COVID, Normal, Viral_Pneumonia)
│
├── training_config.json                 # Config de entrenamiento (legacy)
├── training_history.json                # Historial de entrenamiento (legacy)
├── evaluation_report_20260110_013843.txt # Reporte de evaluacion
│
├── [Logs de reproduccion]
│   ├── repro_split_all_run.log
│   ├── repro_split_all_run2.log
│   ├── repro_split456_rerun_run.log
│   ├── option1_new_seeds.log
│   ├── option1_333_444.log
│   ├── option1_555_666.log
│   ├── quickstart_landmarks.log
│   ├── warping_quickstart.log
│   └── viz_dataset_generation.log
│
├── [Logs y resultados de ensemble sweep]
│   ├── ensemble_combo_sweep_111_222.txt
│   ├── ensemble_combo_sweep_333_444.txt
│   ├── ensemble_combo_sweep_555_666.txt
│   ├── ensemble_combo_sweep_option2.txt
│   └── ensemble_combo_sweep_option2_333_444.txt
│   └── ensemble_combo_sweep_option2_555_666.txt
│
└── [Benchmark]
    ├── benchmark_execution.log
    ├── benchmark_execution_full.log
    ├── benchmark_final.log
    └── benchmark_results.json
```

### Contenido clave

**shape_analysis/**: Resultados del Generalized Procrustes Analysis. La forma canonica (`canonical_shape_gpa.json`) y la triangulacion Delaunay (`canonical_delaunay_triangles.json`) son CRITICOS para el pipeline de warping. Se copian a `build/models_staging/` para las releases.

**landmark_predictions/**: Cache de predicciones de landmarks para las 15,153 imagenes. El archivo `predictions.npz` evita re-inferencia durante el warping. Contiene coordenadas predichas, rutas de imagenes, y metadatos del ensemble (4 modelos, TTA, CLAHE). CRITICO para reproducibilidad.

**warped_lung_best/**: Dataset warped principal con 15,153 imagenes (margin=1.05, fill_rate ~47%). Dividido en train (11,364) / val (1,894) / test (1,895) con 0 errores de procesamiento. CRITICO como input del clasificador.

**classifier_warped_lung_best/**: Contiene 17 clasificadores entrenados de multiples sweeps de hiperparametros. El mejor es `sweeps_2026-01-12/lr2e-4_seed321_on/best_classifier.pt` con 98.05% accuracy (referenciado en `GROUND_TRUTH.json`).

**classifier_cv/**: Resultados de 5-fold cross-validation con metricas:
- Val accuracy media: 98.60% (+/- 0.26%)
- Ensemble test accuracy (sin TTA): 98.10%
- Ensemble test accuracy (con TTA): 98.26%

**dataset_splits_for_gui/**: Copia del dataset completo (original + warped) organizado por splits, preparado para la GUI de demostracion y la defensa de tesis. Verificacion de integridad exitosa (15,153/15,153 imagenes identicas a fuente). ALTO consumo de espacio (duplicacion completa).

### Notas

- **Duplicacion significativa**: `dataset_splits_for_gui/` duplica tanto las imagenes originales de `data/dataset/` como las imagenes warped de `warped_lung_best/`. Esto triplica efectivamente el almacenamiento de imagenes.
- **Sweeps conservados**: Se conservan 17 clasificadores de los sweeps, cada uno con un `best_classifier.pt` (~44 MB). Solo 1 es el "best". Los demas son utiles para documentacion/analisis pero ELIMINABLES si se necesita espacio.
- **Logs de reproduccion**: Multiples archivos `.log` y `.txt` documentan los experimentos de reproduccion. Valor bajo individual, util en conjunto para trazabilidad.

---

## 6. Resumen de integridad

### Verificacion de checkpoints criticos

| Checkpoint | Esperado (CLAUDE.md) | Existe | Usado en ensemble_best.json |
|------------|----------------------|--------|----------------------------|
| `session10/ensemble/seed123/final_model.pt` | Si | SI | SI |
| `session10/ensemble/seed456/final_model.pt` | Si | SI | NO (historico) |
| `session13/seed321/final_model.pt` | Si | SI | SI |
| `session13/seed789/final_model.pt` | Si (historico) | SI | NO (historico) |
| `repro_split111/session14/seed111/final_model.pt` | Si | SI | SI |
| `repro_split666/session16/seed666/final_model.pt` | Si | SI | SI |

**Resultado**: Todos los checkpoints documentados en CLAUDE.md existen. Los 4 modelos criticos del ensemble actual estan presentes y funcionales.

### Consistencia entre directorios

| Artefacto | checkpoints/ | build/models_staging/ | build/releases/v16/ | outputs/ |
|-----------|-------------|----------------------|---------------------|----------|
| seed123 model | SI | SI (renamed) | SI (renamed) | - |
| seed321 model | SI | SI (renamed) | SI (renamed) | - |
| seed111 model | SI | SI (renamed) | SI (renamed) | - |
| seed666 model | SI | SI (renamed) | SI (renamed) | - |
| Canonical shape | - | SI | SI | SI |
| Delaunay triangles | - | SI | SI | SI |
| Best classifier | - | SI | SI | SI (en sweeps) |
| Predictions cache | - | - | - | SI |
| Warped dataset | - | - | - | SI |

### Flujo de datos verificado

```
data/coordenadas/coordenadas_maestro.csv
    --> (GPA) --> outputs/shape_analysis/canonical_shape_gpa.json

data/dataset/COVID-19_Radiography_Dataset/
    --> (Landmark inference) --> outputs/landmark_predictions/session_warping/predictions.npz
    --> (Warping) --> outputs/warped_lung_best/session_warping/
    --> (Classification) --> outputs/classifier_warped_lung_best/

checkpoints/ (4 modelos ensemble)
    --> build/models_staging/ (renombrados)
    --> build/releases/v16/ (empaquetados)
```

---

## 7. Problemas detectados y recomendaciones

### Problemas

1. **Duplicados en checkpoints/**: Los directorios `repro_split111` y `repro_split666` contienen copias anidadas redundantes de los modelos (`seed111/repro_split111/session14/seed111/final_model.pt`), mas archivos ZIP. Esto desperdicia ~200-400 MB.

2. **Duplicacion en outputs/dataset_splits_for_gui/**: Este directorio duplica completamente el dataset original (~15,153 imagenes) y el warped (~15,153 imagenes) que ya existen en `data/dataset/` y `outputs/warped_lung_best/`. Estimado ~2-3 GB adicionales.

3. **5 releases en build/**: Se mantienen 5 versiones de releases portables, cada una con ~87 wheels Python (incluyendo torch ~300 MB) y modelos. Solo v16 es la version actual. Las anteriores consumen ~4 GB estimados.

4. **dist/ vacio**: Directorio residual sin contenido util.

5. **Archivos temporales en data/**: `tmp_subset_cli.csv` y `tmp_subset_warp.csv` son residuos de pruebas.

6. **Falta v16 como ZIP en releases**: Existe el directorio descomprimido `covid19-demo-v16-portable-windows/` y el ZIP `covid19-demo-v16-portable-windows.zip`. Es la unica version que necesita ambos.

### Recomendaciones

| Accion | Estimado ahorro | Prioridad |
|--------|----------------|-----------|
| Eliminar `dist/COVID19_Demo/` | ~0 | BAJA |
| Eliminar `data/tmp_subset_*.csv` | ~0 | BAJA |
| Eliminar duplicados anidados en `checkpoints/repro_split*/` | ~200 MB | MEDIA |
| Eliminar releases anteriores (v1.0.12, v1.0.13, v14, v15) | ~4 GB | MEDIA |
| Evaluar eliminacion de `dataset_splits_for_gui/` si no se necesita offline | ~3 GB | MEDIA |
| Eliminar sweeps de clasificador no-best | ~700 MB | BAJA |
| Eliminar baselines tempranos en raiz de checkpoints/ | ~100 MB | BAJA |

**Espacio total potencialmente recuperable**: ~8 GB

**ADVERTENCIA**: Antes de eliminar cualquier release o modelo, verificar que no se necesite para la defensa de tesis o distribucion a terceros.
