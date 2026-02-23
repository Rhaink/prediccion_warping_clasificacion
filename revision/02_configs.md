# 02. Configuration Files

Analisis de los archivos de configuracion JSON del proyecto.

**Archivos analizados**: 12
**Directorio**: `/home/donrobot/Projects/prediccion_warping_clasificacion/configs/`
**Tamano total**: 402 lineas, ~10.9 KB

---

## Resumen ejecutivo

El proyecto usa archivos JSON para parametrizar cada etapa del pipeline (landmarks, warping, clasificacion, ensembles). De los 12 configs, **5 son criticos** para la reproduccion del pipeline actual (`ensemble_best.json`, `warping_best.json`, `classifier_warped_base.json`, `landmarks_train_base.json`, `ensemble_classifier.json`). Hay **3 configs de baja importancia o eliminables** que corresponden a experimentos historicos o funcionalidad no utilizada activamente (`classifier_original_base.json`, `ensemble_test_no_tta.json`, `final_config.json`). Se detectan inconsistencias menores entre configs y GROUND_TRUTH.json que se documentan abajo.

---

## Analisis individual

### 1. ensemble_best.json
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/configs/ensemble_best.json`
- **Lineas/Tamano**: 11 lineas / 336 bytes
- **Proposito**: Define el ensemble de 4 modelos de landmarks que logra el mejor error (3.61 px). Es el config central para prediccion de landmarks en el pipeline actual.
- **Contenido clave**:
  - `name`: `"ensemble_best_20260111"`
  - `models`: 4 checkpoints (seed123, seed321, seed111, seed666) -- coincide exactamente con `GROUND_TRUTH.json` entry `ensemble_4_models_tta_best_20260111`
  - `tta`: `true` (Test-Time Augmentation habilitado)
  - `clahe`: `true` (preprocesamiento CLAHE habilitado)
- **Usado por**:
  - `scripts/predict_landmarks_dataset.py` (default `--ensemble-config`)
  - `scripts/quickstart_warping.sh` (variable `ENSEMBLE_CONFIG`)
  - `scripts/run_best_ensemble.sh`
  - `scripts/extract_predictions.py`
  - `scripts/benchmark_inference.py`
  - `src_v2/gui/config.py` (GUI config reference)
  - Documentado en `CLAUDE.md`, `README.md`, multiples docs
- **Importancia**: CRITICO
- **Justificacion**: Es la definicion autoritativa del mejor ensemble de landmarks. Toda la cadena de prediccion de landmarks depende de este config. Sin el, no se puede reproducir el dataset warpeado.

---

### 2. warping_best.json
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/configs/warping_best.json`
- **Lineas/Tamano**: 16 lineas / 544 bytes
- **Proposito**: Parametros optimos para generar el dataset warpeado (normalizacion geometrica). Controla el comando `generate-dataset` del CLI.
- **Contenido clave**:
  - `input_dir`: `"data/dataset/COVID-19_Radiography_Dataset"`
  - `output_dir`: `"outputs/warped_lung_best/sessionXX"` (placeholder para sesion)
  - `predictions`: `"outputs/landmark_predictions/sessionXX/predictions.npz"` (placeholder)
  - `canonical`: `"outputs/shape_analysis/canonical_shape_gpa.json"`
  - `triangles`: `"outputs/shape_analysis/canonical_delaunay_triangles.json"`
  - `margin`: `1.05` -- coincide con `GROUND_TRUTH.json` `preprocessing.warping.margin_scale_optimal`
  - `use_full_coverage`: `false` -- **NOTA**: contradice `GROUND_TRUTH.json` `preprocessing.warping.use_full_coverage: true`. Ver observacion abajo.
  - `clahe`: `true`, `clahe_clip`: `2.0`, `clahe_tile`: `4`
  - `splits`: `"0.75,0.125,0.125"`, `seed`: `42`
  - `tta`: `true`
- **Usado por**:
  - `python -m src_v2 generate-dataset --config configs/warping_best.json`
  - `scripts/quickstart_warping.sh` (indirectamente)
  - Documentado en `CLAUDE.md`, `README.md`, multiples docs
- **Importancia**: CRITICO
- **Justificacion**: Define completamente como se genera el dataset warpeado. Es imprescindible para reproduccion.
- **Observacion**: El campo `use_full_coverage: false` contradice `GROUND_TRUTH.json` donde `preprocessing.warping.use_full_coverage: true`. Segun `GROUND_TRUTH.json`, los datasets `warped_96` y `warped_99` usaban `use_full_coverage=true`, pero el dataset actual `warped_lung_best` (fill_rate=47%) usa `false`. El valor en `GROUND_TRUTH.json` bajo `preprocessing.warping` parece ser un vestigio de configuraciones anteriores y no refleja la metodologia actual. **El config es correcto; el GROUND_TRUTH.json tiene informacion desactualizada en esa seccion.**

---

### 3. classifier_warped_base.json
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/configs/classifier_warped_base.json`
- **Lineas/Tamano**: 11 lineas / 255 bytes
- **Proposito**: Configuracion base para entrenar el clasificador ResNet-18 sobre imagenes warpeadas. Es el config por defecto del pipeline de clasificacion.
- **Contenido clave**:
  - `data_dir`: `"outputs/warped_lung_best/session_warping"`
  - `backbone`: `"resnet18"`
  - `epochs`: `50`, `batch_size`: `32`, `lr`: `0.0001`
  - `patience`: `10` (early stopping)
  - `use_class_weights`: `true`
  - `output_dir`: `"outputs/classifier_warped_lung_best"`
  - `seed`: `42`
- **Usado por**:
  - `python -m src_v2 train-classifier --config configs/classifier_warped_base.json`
  - `python -m src_v2 cross-validate-classifier --config configs/classifier_warped_base.json`
  - `scripts/run_classifier_sweep_accuracy.sh`
  - Documentado en `CLAUDE.md`, `README.md`, multiples docs
- **Importancia**: CRITICO
- **Justificacion**: Es el punto de entrada estandar para entrenamiento del clasificador. Referenciado directamente en CLAUDE.md como parte del pipeline principal.
- **Observacion**: El `lr: 0.0001` difiere del mejor resultado validado en `GROUND_TRUTH.json` que usa `lr2e-4` (0.0002) con `seed321`. Este config es un punto de partida base; el sweep de hiperparametros encontro mejores valores.

---

### 4. landmarks_train_base.json
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/configs/landmarks_train_base.json`
- **Lineas/Tamano**: 22 lineas / 439 bytes
- **Proposito**: Configuracion base para entrenar modelos individuales de deteccion de landmarks. Define la arquitectura del modelo y los hiperparametros de entrenamiento de dos fases.
- **Contenido clave**:
  - **Arquitectura**: `coord_attention: true`, `deep_head: true`, `hidden_dim: 768`, `dropout: 0.3`
  - **Fase 1** (backbone congelado): `phase1_epochs: 15`, `phase1_lr: 0.001`, `phase1_patience: 5`
  - **Fase 2** (fine-tuning): `phase2_epochs: 100`, `phase2_backbone_lr: 2e-5`, `phase2_head_lr: 2e-4`, `phase2_patience: 15`
  - **Augmentation**: `flip_prob: 0.5`, `rotation: 10.0`
  - **Preprocessing**: `clahe: true`, `clahe_clip: 2.0`, `clahe_tile: 4`
  - **Inference**: `tta: true`
  - `loss`: `"wing"` (Wing Loss), `batch_size`: `16`, `num_workers`: `4`
- **Usado por**:
  - `scripts/quickstart_landmarks.sh` (variable `TRAIN_CONFIG`)
  - `scripts/run_seed_sweep.sh` (via `TRAIN_CONFIG` env var)
  - `scripts/README.md` (documented usage)
- **Importancia**: CRITICO
- **Justificacion**: Define completamente la arquitectura y entrenamiento de los modelos de landmarks. Todos los modelos del ensemble fueron entrenados con estos parametros base. Coincide exactamente con los valores en `final_config.json` y `GROUND_TRUTH.json` para arquitectura y entrenamiento.

---

### 5. ensemble_classifier.json
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/configs/ensemble_classifier.json`
- **Lineas/Tamano**: 22 lineas / 727 bytes
- **Proposito**: Configuracion para el ensemble de clasificadores (5-fold CV) evaluado sobre el test set. Controla el comando `evaluate-classifier-ensemble`.
- **Contenido clave**:
  - `description`: "5-fold cross-validation ensemble configuration for classifier evaluation"
  - `use_tta`: `true`
  - `checkpoint_paths`: 5 paths (`outputs/classifier_cv/fold_01..05/best_classifier.pt`)
  - `data_dir`: `"outputs/warped_lung_best/session_warping"`
  - `split`: `"test"`
  - `baseline_accuracy`: `0.9768`, `baseline_std`: `0.0016`
  - `class_names`: `["COVID", "Normal", "Viral_Pneumonia"]`
  - `expected_samples`: total=1895 (COVID=452, Normal=1274, Viral_Pneumonia=169)
- **Usado por**:
  - `python -m src_v2 evaluate-classifier-ensemble --config configs/ensemble_classifier.json`
  - Referenciado en `GROUND_TRUTH.json` como `classification.classifier_ensemble_cv.config`
- **Importancia**: CRITICO
- **Justificacion**: Define la evaluacion final del pipeline de clasificacion (ensemble de 5 folds). Los resultados validados en GROUND_TRUTH (accuracy=0.9826 con TTA) dependen de este config. Incluye metadatos valiosos (expected_samples, baseline) para verificacion.

---

### 6. landmark_viz_best.json
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/configs/landmark_viz_best.json`
- **Lineas/Tamano**: 11 lineas / 369 bytes
- **Proposito**: Configuracion para generar dataset de visualizacion de landmarks (imagenes originales con landmarks superpuestos como cruces rojas).
- **Contenido clave**:
  - `input_dir`: `"data/dataset/COVID-19_Radiography_Dataset"`
  - `output_dir`: `"outputs/landmark_visualizations/session_warping"`
  - `predictions`: `"outputs/landmark_predictions/session_warping/predictions.npz"`
  - `splits`: `"0.75,0.125,0.125"`, `seed`: `42`
  - `classes`: `"COVID,Normal,Viral Pneumonia"`
  - Visualizacion: `cross_size: 5`, `cross_thickness: 2`, `cross_color: "red"`
- **Usado por**:
  - `python -m src_v2 generate-landmark-visualization-dataset --config configs/landmark_viz_best.json`
  - Documentado en `CLAUDE.md`
- **Importancia**: MEDIO
- **Justificacion**: Util para verificacion visual y generacion de figuras, pero no es parte del pipeline de entrenamiento/evaluacion. El dataset de visualizacion es un producto auxiliar.

---

### 7. classifier_warped_sahs_masked.json
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/configs/classifier_warped_sahs_masked.json`
- **Lineas/Tamano**: 11 lineas / 241 bytes
- **Proposito**: Configuracion para entrenar clasificador sobre imagenes warpeadas con mascaras SAHS (Simulated Anatomical Heat Signatures). Experimento complementario para el reporte.
- **Contenido clave**:
  - `data_dir`: `"outputs/warped_lung_sahs"` (dataset diferente al principal)
  - `backbone`: `"resnet18"`
  - `epochs`: `50`, `batch_size`: `32`, `lr`: `0.0001`
  - `patience`: `10`, `use_class_weights`: `true`
  - `output_dir`: `"outputs/classifier_warped_sahs_masked"`
  - `seed`: `42`
- **Usado por**:
  - No hay referencia directa en CLI o scripts principales
  - Resultados referenciados por scripts de visualizacion: `generate_F5_8_comparison_improved.py`, `generate_confusion_matrix_sahs.py`, `generate_F5_9_misclassified.py`
- **Importancia**: MEDIO
- **Justificacion**: Parte de un experimento complementario (SAHS masking) para el reporte. No es parte del pipeline principal pero contribuye a figuras del reporte final. Los scripts que lo referencian son de generacion de figuras.

---

### 8. cropping_10.json
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/configs/cropping_10.json`
- **Lineas/Tamano**: 9 lineas / 252 bytes
- **Proposito**: Configuracion para generar un dataset recortado al 10% (crop central). Usado por el comando `generate-cropped-dataset`.
- **Contenido clave**:
  - `input_dir`: `"data/dataset/COVID-19_Radiography_Dataset"`
  - `output_dir`: `"outputs/cropped_lung_10/sessionXX"` (placeholder)
  - `crop_pct`: `0.10` (10% de recorte)
  - `min_fill_rate`: `0.47`
  - `splits`: `"0.75,0.125,0.125"`, `seed`: `42`
  - `classes`: `"COVID,Normal,Viral Pneumonia"`
- **Usado por**:
  - `python -m src_v2 generate-cropped-dataset --config configs/cropping_10.json`
- **Importancia**: BAJO
- **Justificacion**: Es un experimento de ablacion (cropping vs. warping). No forma parte del pipeline principal documentado en CLAUDE.md. El resultado no aparece en GROUND_TRUTH.json como metrica validada. Puede ser util para futuras comparaciones pero no es esencial.

---

### 9. classifier_original_base.json
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/configs/classifier_original_base.json`
- **Lineas/Tamano**: 12 lineas / 336 bytes
- **Proposito**: Configuracion para entrenar clasificador sobre imagenes originales (sin warping) como baseline de comparacion.
- **Contenido clave**:
  - `warped_data_dir`: `"outputs/warped_lung_best/session_warping"` (referencia a splits warpeados)
  - `original_data_dir`: `"data/dataset/COVID-19_Radiography_Dataset"`
  - `model`: `"resnet18"` (usa clave `model` en vez de `backbone`)
  - `epochs`: `50`, `batch_size`: `32`, `lr`: `0.0001`
  - `patience`: `10`, `use_class_weights`: `true`
  - `output_dir`: `"outputs/classifier_original_warped_lung_best"`
  - `seed`: `42`
- **Usado por**:
  - `scripts/archive/classification/train_classifier_original.py` (script archivado)
  - Referenciado en `scripts/README.md`
- **Importancia**: BAJO
- **Justificacion**: Solo usado por un script archivado (`scripts/archive/`). El resultado correspondiente en GROUND_TRUTH (`original_100`) esta marcado como `obsolete`. Ademas, usa `model` como clave en vez de `backbone`, inconsistente con los demas configs de clasificador. Tiene un campo `warped_data_dir` inusual que se usa para alinear splits entre original y warpeado.

---

### 10. ensemble_test_no_tta.json
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/configs/ensemble_test_no_tta.json`
- **Lineas/Tamano**: 11 lineas / 336 bytes
- **Proposito**: Variante de `ensemble_best.json` con TTA y CLAHE deshabilitados. Para medir el impacto de TTA/CLAHE por separado.
- **Contenido clave**:
  - `name`: `"ensemble_test_no_tta"`
  - `models`: mismos 4 checkpoints que `ensemble_best.json`
  - `tta`: `false`
  - `clahe`: `false`
- **Usado por**:
  - Ningun script o comando lo referencia directamente en el codebase (busqueda en `scripts/` y `src_v2/` no arroja resultados)
- **Importancia**: BAJO
- **Justificacion**: Es un config de ablacion para uso manual (linea de comandos ad-hoc). No esta integrado en ningun pipeline automatizado ni referenciado en documentacion principal. Util para debugging pero no esencial.

---

### 11. final_config.json
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/configs/final_config.json`
- **Lineas/Tamano**: 255 lineas / 7.1 KB
- **Proposito**: Documento historico exhaustivo que captura el estado completo del proyecto al final de la Sesion 13. Incluye resultados, arquitectura, hiperparametros, dataset splits, y descripcion de landmarks.
- **Contenido clave**:
  - `project.version`: `"2.1"`, `project.date`: `"2024-11-27"` (fecha incorrecta, deberia ser 2025/2026)
  - `results.ensemble_4models`: error 3.71 px (ensemble obsoleto con seed456/seed789)
  - `architecture`: Descripcion completa del head (Flatten -> Linear 512 -> GroupNorm -> ReLU -> Dropout -> Linear 768 -> GroupNorm -> ReLU -> Dropout -> Linear 30 -> Sigmoid)
  - `preprocessing`, `training`, `augmentation`: Duplica informacion de `landmarks_train_base.json` con mas detalle
  - `dataset`: 957 muestras, splits 75/15/10, 15 landmarks con descripciones
  - `ensemble_4models`, `ensemble_optimal`, `ensemble_all`: 3 configuraciones de ensemble (todas obsoletas)
  - `hardware`: AMD Radeon RX 6600, ROCm compatible
- **Usado por**:
  - `scripts/visualization/generate_f4_5_autogen.py` (como `DEFAULT_CONFIG`)
- **Importancia**: BAJO
- **Justificacion**: Es esencialmente un snapshot historico. La informacion de resultados esta desactualizada (usa el ensemble 3.71 px, no el actual de 3.61 px). Los modelos referenciados incluyen seed456 y seed789 que ya no forman parte del ensemble actual. La unica referencia activa es un script de generacion de figuras. La informacion relevante esta mejor mantenida en `GROUND_TRUTH.json` y en los configs individuales. La fecha `2024-11-27` parece incorrecta.
- **Observacion**: Contiene informacion valiosa sobre la estructura del head del modelo que no esta en ningun otro config. Si se eliminara, se perderia la documentacion de `head_structure` (capas, dimensiones, GroupNorm groups). Considerar migrar esa informacion a documentacion apropiada antes de cualquier limpieza.

---

### 12. hierarchical_train_base.json
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/configs/hierarchical_train_base.json`
- **Lineas/Tamano**: 11 lineas / 202 bytes
- **Proposito**: Configuracion para entrenar el modelo jerarquico de landmarks (HierarchicalLandmarkModel), una arquitectura alternativa que no fue seleccionada como la mejor.
- **Contenido clave**:
  - `seed`: `42`, `hidden_dim`: `512` (menor que el estandar de 768)
  - `dropout`: `0.3`
  - `phase1_epochs`: `15`, `phase2_epochs`: `100`, `patience`: `15`
  - `axis_weight`: `0.5` (peso para penalizacion de eje en modelo jerarquico)
  - `save_dir`: `"checkpoints/hierarchical"`
  - `clahe`: `true`
- **Usado por**:
  - `scripts/train_hierarchical.py`
  - Referenciado en `scripts/README.md`
- **Importancia**: BAJO
- **Justificacion**: El modelo jerarquico fue explorado pero no seleccionado como mejor aproximacion. No produce resultados en GROUND_TRUTH.json. El `hidden_dim: 512` difiere del estandar 768 usado en el modelo principal. Solo es relevante si se quisiera retomar la investigacion de arquitecturas alternativas.

---

## Tabla resumen

| # | Archivo | Lineas | Tamano | Importancia | Pipeline actual |
|---|---------|--------|--------|-------------|-----------------|
| 1 | `ensemble_best.json` | 11 | 336 B | CRITICO | Si - prediccion landmarks |
| 2 | `warping_best.json` | 16 | 544 B | CRITICO | Si - generacion dataset |
| 3 | `classifier_warped_base.json` | 11 | 255 B | CRITICO | Si - entrenamiento clasificador |
| 4 | `landmarks_train_base.json` | 22 | 439 B | CRITICO | Si - entrenamiento landmarks |
| 5 | `ensemble_classifier.json` | 22 | 727 B | CRITICO | Si - evaluacion ensemble clasificadores |
| 6 | `landmark_viz_best.json` | 11 | 369 B | MEDIO | Auxiliar - visualizacion |
| 7 | `classifier_warped_sahs_masked.json` | 11 | 241 B | MEDIO | Auxiliar - experimento reporte |
| 8 | `cropping_10.json` | 9 | 252 B | BAJO | No - experimento ablacion |
| 9 | `classifier_original_base.json` | 12 | 336 B | BAJO | No - script archivado |
| 10 | `ensemble_test_no_tta.json` | 11 | 336 B | BAJO | No - ablacion manual |
| 11 | `final_config.json` | 255 | 7.1 KB | BAJO | No - snapshot historico |
| 12 | `hierarchical_train_base.json` | 11 | 202 B | BAJO | No - arquitectura descartada |

---

## Observaciones transversales

### 1. Inconsistencia `use_full_coverage` entre warping_best.json y GROUND_TRUTH.json

- `warping_best.json`: `use_full_coverage: false`
- `GROUND_TRUTH.json` seccion `preprocessing.warping`: `use_full_coverage: true`
- **Diagnostico**: El config es correcto para la metodologia actual (`warped_lung_best`, fill_rate=47%). La seccion `preprocessing.warping` en GROUND_TRUTH.json parece reflejar configuraciones historicas (warped_96/warped_99) y deberia actualizarse.
- **Accion sugerida**: Actualizar `GROUND_TRUTH.json` seccion `preprocessing.warping.use_full_coverage` a `false`.

### 2. Inconsistencia de clave `model` vs `backbone`

- `classifier_original_base.json` usa `"model": "resnet18"`
- `classifier_warped_base.json` y `classifier_warped_sahs_masked.json` usan `"backbone": "resnet18"`
- **Diagnostico**: `classifier_original_base.json` fue creado para un script diferente (`train_classifier_original.py`, ahora archivado) que esperaba la clave `model`. Los otros configs usan `backbone` que es lo que espera el CLI (`src_v2/cli.py`).
- **Impacto**: Bajo, ya que el config solo lo usa un script archivado.

### 3. Placeholders `sessionXX` en configs

- `warping_best.json`: `output_dir` y `predictions` contienen `"sessionXX"` como placeholder.
- `cropping_10.json`: `output_dir` contiene `"sessionXX"`.
- **Diagnostico**: Estos placeholders requieren sustitucion manual o por script antes de ejecutar. El `quickstart_warping.sh` los sobreescribe con argumentos CLI, pero si se usa el config directamente, crearia directorios con nombre literal `sessionXX`.
- **Accion sugerida**: Documentar claramente que estos campos son placeholders, o usar un valor por defecto funcional como `session_warping`.

### 4. Duplicacion de informacion en final_config.json

- `final_config.json` (255 lineas, 7.1 KB) contiene toda la informacion de `landmarks_train_base.json` mas resultados, arquitectura detallada, y dataset metadata.
- Esta informacion esta mejor mantenida en `GROUND_TRUTH.json` (resultados), `landmarks_train_base.json` (hiperparametros), y el codigo fuente (arquitectura).
- El unico contenido unico es `architecture.head_structure` (capa por capa del head) y `dataset.landmarks.description` (significado anatomico de cada landmark).
- **Accion sugerida**: Extraer la informacion unica a documentacion apropiada y considerar marcar `final_config.json` como historico/legacy.

### 5. Consistencia de splits y seeds

Todos los configs que definen splits usan valores consistentes:
- `splits`: `"0.75,0.125,0.125"` (excepto `final_config.json` que documenta `0.75/0.15/0.10` -- un split diferente para el dataset de landmarks vs. clasificacion)
- `seed`: `42` (universal)
- Esta consistencia es correcta y deseable para reproducibilidad.

**Nota importante sobre splits**: El dataset de landmarks usa split 75/15/10 (957 muestras anotadas), mientras que el dataset de clasificacion usa split 75/12.5/12.5 (~15,000+ imagenes). Son datasets diferentes con diferentes splits, lo cual es correcto.

### 6. Configs sin referencia activa en codigo

Los siguientes configs no tienen referencia directa en ningun script no-archivado ni en el CLI:
- `ensemble_test_no_tta.json`: Sin referencias en scripts/ o src_v2/
- `classifier_warped_sahs_masked.json`: Solo referenciado por scripts de visualizacion (no por CLI)

Estos configs se usan presumiblemente via invocacion manual de linea de comandos.

### 7. Coherencia con GROUND_TRUTH.json

| Parametro | Config | GROUND_TRUTH.json | Coincide |
|-----------|--------|-------------------|----------|
| Ensemble models (4) | ensemble_best.json | landmarks.ensemble_4_models_tta_best_20260111 | Si |
| Margin scale | warping_best.json (1.05) | preprocessing.warping.margin_scale_optimal (1.05) | Si |
| CLAHE tile_size | warping_best.json (4) | preprocessing.clahe.tile_size (4) | Si |
| CLAHE clip_limit | warping_best.json (2.0) | preprocessing.clahe.clip_limit (2.0) | Si |
| use_full_coverage | warping_best.json (false) | preprocessing.warping (true) | **No** |
| hidden_dim | landmarks_train_base.json (768) | preprocessing.model_architecture.hidden_dim (768) | Si |
| dropout | landmarks_train_base.json (0.3) | preprocessing.model_architecture.dropout (0.3) | Si |
| coord_attention | landmarks_train_base.json (true) | preprocessing.model_architecture.coord_attention (true) | Si |
| Wing Loss | landmarks_train_base.json ("wing") | final_config.json (WingLoss) | Si |
| Classifier ensemble config | ensemble_classifier.json | classification.classifier_ensemble_cv.config | Si |
