# Guía de Uso del CLI

**Referencia Completa de la Interfaz de Línea de Comandos**

Este documento describe los comandos principales del sistema para detección de landmarks pulmonares y clasificación COVID-19.

---

## Tabla de Contenidos

1. [Introducción al CLI](#introducción-al-cli)
2. [Comandos Principales del Pipeline](#comandos-principales-del-pipeline)
3. [Workflows Comunes](#workflows-comunes)
4. [Comandos Auxiliares](#comandos-auxiliares)
5. [Tips y Mejores Prácticas](#tips-y-mejores-prácticas)

---

## Introducción al CLI

### Acceso al CLI

El sistema se ejecuta mediante el módulo `src_v2`:

```bash
python -m src_v2 [COMMAND] [OPTIONS]
```

### Ayuda General

Ver todos los comandos disponibles:

```bash
python -m src_v2 --help
```

Ver ayuda de un comando específico:

```bash
python -m src_v2 [COMMAND] --help
```

Ejemplo:
```bash
python -m src_v2 compute-canonical --help
```

### Opciones Globales

```bash
--verbose, -v    # Modo verbose (nivel DEBUG de logging)
--help           # Mostrar ayuda
```

### Configuración mediante JSON

La mayoría de los comandos aceptan `--config CONFIG.json` para evitar flags largos en la línea de comandos. Los configs están en `04_Configuraciones/`.

**Ventajas:**
- Reproducibilidad: mismo config → mismos resultados
- Legibilidad: JSON documentado
- Versionamiento: configs en Git

---

## Comandos Principales del Pipeline

Esta sección documenta los 7 comandos críticos para el pipeline completo.

---

### 1. compute-canonical

**Propósito:** Calcular la forma pulmonar canónica mediante Generalized Procrustes Analysis (GPA) y generar la triangulación de Delaunay.

**Cuándo usar:** Una sola vez al inicio, antes de generar el dataset normalizado. Genera los archivos de forma canónica necesarios para el warping.

#### Sintaxis

```bash
python -m src_v2 compute-canonical LANDMARKS_CSV [OPTIONS]
```

#### Parámetros

| Parámetro | Tipo | Descripción | Default |
|-----------|------|-------------|---------|
| `LANDMARKS_CSV` | Ruta | **Requerido.** Path al CSV con coordenadas de landmarks | - |
| `--output-dir, -o` | Ruta | Directorio de salida para archivos JSON | `outputs/shape_analysis` |
| `--visualize` | Flag | Generar visualización de la forma canónica | False |
| `--max-iterations` | Int | Máximo de iteraciones para GPA | 100 |
| `--tolerance` | Float | Tolerancia para convergencia de GPA | 1e-8 |
| `--image-size` | Int | Tamaño de imagen para escalar forma canónica | 224 |
| `--padding` | Float | Padding relativo (0.1 = 10%) | 0.1 |

#### Entradas

- **CSV de anotaciones:** Formato `image_name,category,L1_x,L1_y,...,L15_x,L15_y`
- 15 landmarks × 2 coordenadas = 30 valores por imagen
- Coordenadas en píxeles [0, 299] (tamaño original)

#### Salidas

Genera en `--output-dir`:

1. **`canonical_shape_gpa.json`** - Forma canónica (15 landmarks en coordenadas normalizadas [0,1])
   ```json
   {
     "landmarks": [[x1,y1], [x2,y2], ..., [x15,y15]],
     "image_size": 224,
     "num_landmarks": 15
   }
   ```

2. **`canonical_delaunay_triangles.json`** - Triangulación (24 triángulos)
   ```json
   {
     "triangles": [[v1, v2, v3], ...],
     "num_triangles": 24
   }
   ```

3. **`canonical_shape_visualization.png`** (si `--visualize`)
   - Landmarks en rojo
   - Triangulación en azul

#### Ejemplo Concreto

```bash
python -m src_v2 compute-canonical \
  data/coordenadas/coordenadas_maestro.csv \
  --output-dir outputs/shape_analysis \
  --visualize
```

**Salida esperada:**
```
Computing canonical shape via GPA...
  Initial alignment...
  Iteration 1: mean shape change = 0.0234
  Iteration 2: mean shape change = 0.0045
  ...
  Converged after 4 iterations.

Computing Delaunay triangulation...
  Generated 24 triangles from 15 landmarks.

Saved:
  ✓ outputs/shape_analysis/canonical_shape_gpa.json
  ✓ outputs/shape_analysis/canonical_delaunay_triangles.json
  ✓ outputs/shape_analysis/canonical_shape_visualization.png
```

**Tiempo:** ~30 segundos

---

### 2. evaluate-ensemble

**Propósito:** Evaluar el ensemble de modelos de landmarks en el conjunto de test.

**Cuándo usar:** Para verificar las métricas del ensemble (3.61 px) o evaluar nuevos ensembles.

#### Sintaxis

```bash
python scripts/evaluate_ensemble_from_config.py --config ENSEMBLE_CONFIG.json
```

**Nota:** Este comando usa un script auxiliar en `scripts/` en lugar del CLI principal.

#### Parámetros

| Parámetro | Tipo | Descripción | Default |
|-----------|------|-------------|---------|
| `--config` | Ruta | **Requerido.** Config JSON con lista de modelos | - |
| `--split` | Str | Split a evaluar: train, val, test | test |
| `--output` | Ruta | Directorio de salida para resultados | `outputs/ensemble_evaluation` |

#### Entradas

- **Config JSON** (ej: `configs/ensemble_best.json`):
  ```json
  {
    "name": "ensemble_best_20260111",
    "models": [
      "checkpoints/session10/ensemble/seed123/final_model.pt",
      "checkpoints/session13/seed321/final_model.pt",
      "checkpoints/repro_split111/session14/seed111/final_model.pt",
      "checkpoints/repro_split666/session16/seed666/final_model.pt"
    ],
    "tta": true,
    "clahe": true
  }
  ```

- **Dataset:** Imágenes en `data/dataset/COVID-19_Radiography_Dataset/`
- **Anotaciones:** `data/coordenadas/coordenadas_maestro.csv`

#### Salidas

Genera en `--output`:

1. **`results.json`** - Métricas completas
   ```json
   {
     "mean_error_px": 3.61,
     "std_px": 2.48,
     "median_px": 3.07,
     "error_by_category": {
       "Normal": 3.22,
       "COVID": 3.93,
       "Viral_Pneumonia": 4.11
     }
   }
   ```

2. **`predictions_per_image.csv`** - Predicciones por imagen
3. **`error_distribution.png`** - Histograma de errores
4. **`error_by_landmark.png`** - Error por cada landmark

#### Ejemplo Concreto

```bash
python scripts/evaluate_ensemble_from_config.py \
  --config configs/ensemble_best.json
```

**Salida esperada:**
```
=== Evaluating Ensemble ===
Models: 4
TTA: True
CLAHE: True

Evaluating on test set (3030 images)...
100%|████████████████| 3030/3030 [05:23<00:00, 9.37it/s]

=== Results ===
Mean Pixel Error: 3.61 px
Std:              2.48 px
Median:           3.07 px

Error by Category:
  Normal:           3.22 px
  COVID:            3.93 px
  Viral_Pneumonia:  4.11 px
```

**Tiempo:** 5-8 minutos (GPU) / 30-45 minutos (CPU)

---

### 3. generate-dataset

**Propósito:** Generar el dataset completo de imágenes normalizadas geométricamente mediante warping afín por partes.

**Cuándo usar:** Después de tener la forma canónica y predicciones de landmarks. Genera el dataset para entrenar el clasificador.

#### Sintaxis

```bash
python -m src_v2 generate-dataset [INPUT_DIR] [OUTPUT_DIR] [OPTIONS]
```

O usando config:

```bash
python -m src_v2 generate-dataset --config CONFIG.json
```

#### Parámetros Principales

| Parámetro | Tipo | Descripción | Default |
|-----------|------|-------------|---------|
| `INPUT_DIR` | Ruta | Directorio del dataset original | - |
| `OUTPUT_DIR` | Ruta | Directorio de salida para dataset warped | - |
| `--config` | Ruta | Config JSON con todos los parámetros | - |
| `--predictions` | Ruta | **Recomendado.** Cache .npz con landmarks predichos (evita re-inferencia) | - |
| `--checkpoint, -c` | Ruta | Checkpoint del modelo (si no hay `--predictions`) | - |
| `--ensemble-config` | Ruta | Config de ensemble (si no hay `--predictions`) | - |
| `--canonical` | Ruta | Forma canónica (.json) | `outputs/shape_analysis/canonical_shape_gpa.json` |
| `--triangles` | Ruta | Triangulación de Delaunay (.json) | `outputs/shape_analysis/canonical_delaunay_triangles.json` |
| `--margin, -m` | Float | Factor de escala para márgenes (1.05 = 5% expansión) | 1.05 |
| `--splits` | Str | Ratios train,val,test separados por coma | `0.75,0.125,0.125` |
| `--seed, -s` | Int | Semilla para reproducibilidad de splits | 42 |
| `--clahe` | Flag | Aplicar CLAHE | False |
| `--clahe-clip` | Float | Clip limit para CLAHE | 2.0 |
| `--clahe-tile` | Int | Tile size para CLAHE | 4 |
| `--no-full-coverage` | Flag | **Importante.** Usa ROI basado en landmarks (NO expansión a imagen completa) | False |
| `--tta` | Flag | Usar Test-Time Augmentation con flip horizontal | False |

#### Entradas

1. **Dataset original:** Estructura `INPUT_DIR/COVID/images/, Normal/images/, Viral Pneumonia/images/`
2. **Predicciones cache** (recomendado): `.npz` generado por `predict_landmarks_dataset.py`
3. **Forma canónica:** JSON con 15 landmarks
4. **Triangulación:** JSON con 24 triángulos

#### Salidas

Genera en `OUTPUT_DIR`:

```
OUTPUT_DIR/
├── train/
│   ├── COVID/
│   ├── Normal/
│   └── Viral_Pneumonia/
├── val/
├── test/
├── metadata.json               # Parámetros usados
├── dataset_summary.json        # Estadísticas
└── {split}/
    ├── landmarks.json          # Landmarks predichos por imagen
    └── images.csv              # Lista de imágenes procesadas
```

**Características del dataset warped:**
- Imágenes: 224×224 PNG
- Fill rate medio: ~47% (con `--no-full-coverage` y margin=1.05)
- Preprocesamiento: Escala de grises + CLAHE (opcional)

#### Ejemplo Concreto

**Usando cache de predicciones (recomendado):**

```bash
python -m src_v2 generate-dataset \
  data/dataset/COVID-19_Radiography_Dataset \
  outputs/warped_lung_best/session_warping \
  --canonical outputs/shape_analysis/canonical_shape_gpa.json \
  --triangles outputs/shape_analysis/canonical_delaunay_triangles.json \
  --predictions outputs/landmark_predictions/session_warping/predictions.npz \
  --margin 1.05 \
  --splits 0.75,0.125,0.125 \
  --seed 42 \
  --clahe --clahe-clip 2.0 --clahe-tile 4 \
  --no-full-coverage
```

**Usando config (más limpio):**

```bash
python -m src_v2 generate-dataset --config configs/warping_best.json
```

**Salida esperada:**
```
Loading cached predictions...
  ✓ Loaded 15153 predictions from predictions.npz

Applying piecewise affine warping...
Processing: 100%|████████| 15153/15153 [07:23<00:00, 34.16it/s]

Dataset statistics:
  Total images: 15153
  Train: 11364 (75.0%)
  Val:    1894 (12.5%)
  Test:   1895 (12.5%)

Fill rate:
  Mean: 47.2%
  Std:   8.3%

Saved to: outputs/warped_lung_best/session_warping/
```

**Tiempo:** 5-10 minutos

**¿Por qué usar cache de predicciones?**
- Evita re-inferencia (ahorra 10-45 min)
- Garantiza predicciones idénticas
- Permite experimentar con diferentes parámetros de warping sin re-predecir

---

### 4. train-classifier

**Propósito:** Entrenar un clasificador CNN (ResNet-18) para COVID-19 en el dataset normalizado.

**Cuándo usar:** Después de generar el dataset warped. Para entrenar desde cero o experimentar con hiperparámetros.

#### Sintaxis

```bash
python -m src_v2 train-classifier [DATA_DIR] [OPTIONS]
```

O usando config:

```bash
python -m src_v2 train-classifier --config CONFIG.json
```

#### Parámetros

| Parámetro | Tipo | Descripción | Default |
|-----------|------|-------------|---------|
| `DATA_DIR` | Ruta | **Requerido.** Directorio del dataset (debe contener train/, val/, test/) | - |
| `--output-dir, -o` | Ruta | Directorio de salida para modelo y resultados | `outputs/classifier` |
| `--backbone, -b` | Str | Arquitectura: `resnet18`, `efficientnet_b0`, `densenet121` | `resnet18` |
| `--epochs, -e` | Int | Número de épocas | 50 |
| `--batch-size` | Int | Tamaño de batch | 32 |
| `--lr` | Float | Learning rate | 0.0001 |
| `--class-weights` | Flag | Usar pesos de clase para balanceo | True |
| `--patience` | Int | Paciencia para early stopping | 10 |
| `--device` | Str | Dispositivo: `auto`, `cuda`, `cpu`, `mps` | `auto` |
| `--seed, -s` | Int | Semilla aleatoria | 42 |
| `--config` | Ruta | JSON con defaults para reproducir entrenamiento | - |

#### Entradas

- **Dataset warped:** Estructura `DATA_DIR/train/, val/, test/` con subdirectorios por clase
- Imágenes 224×224 PNG

#### Salidas

Genera en `--output-dir`:

1. **`best_classifier.pt`** - Mejor modelo según val accuracy
2. **`final_classifier.pt`** - Modelo al final del entrenamiento
3. **`training_history.json`** - Historial de métricas por época
4. **`training_curves.png`** - Gráficos de loss y accuracy
5. **`config.json`** - Configuración usada
6. **`test_results.json`** - Evaluación en test set

#### Ejemplo Concreto

```bash
python -m src_v2 train-classifier \
  outputs/warped_lung_best/session_warping \
  --backbone resnet18 \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.0001 \
  --class-weights \
  --patience 10 \
  --seed 42 \
  --output-dir outputs/classifier_warped_lung_best
```

**O usando config:**

```bash
python -m src_v2 train-classifier --config configs/classifier_warped_base.json
```

**Salida esperada:**
```
Training classifier...
Backbone: resnet18
Dataset: outputs/warped_lung_best/session_warping
  Train: 11364 images
  Val:    1894 images
  Test:   1895 images

Class distribution (train):
  COVID:            2712 (23.9%)
  Normal:           7644 (67.3%)
  Viral_Pneumonia:  1008 ( 8.9%)

Using class weights: [1.398, 0.496, 3.763]

Epoch 1/50
  Train Loss: 0.8234 | Acc: 72.34%
  Val   Loss: 0.5123 | Acc: 85.67%

...

Epoch 50/50
  Train Loss: 0.0834 | Acc: 97.12%
  Val   Loss: 0.1245 | Acc: 98.53%

Early stopping at epoch 47

Best model: epoch 47, val_acc = 98.67%
Saved to: outputs/classifier_warped_lung_best/best_classifier.pt

Evaluating on test set...
  Test Accuracy:    98.05%
  Test F1-macro:    97.12%
  Test F1-weighted: 98.04%
```

**Tiempo:** 1-2 horas (GPU con 50 epochs)

---

### 5. evaluate-classifier

**Propósito:** Evaluar un clasificador entrenado en un dataset específico.

**Cuándo usar:** Para validar métricas de un modelo entrenado, comparar modelos, o evaluar en diferentes splits.

#### Sintaxis

```bash
python -m src_v2 evaluate-classifier CHECKPOINT [OPTIONS]
```

#### Parámetros

| Parámetro | Tipo | Descripción | Default |
|-----------|------|-------------|---------|
| `CHECKPOINT` | Ruta | **Requerido.** Path al modelo .pt | - |
| `--data-dir` | Ruta | **Requerido.** Directorio del dataset | - |
| `--split` | Str | Split a evaluar: `train`, `val`, `test` | `test` |
| `--batch-size` | Int | Tamaño de batch | 32 |
| `--output` | Ruta | Archivo JSON de salida con resultados | `evaluation_results.json` |
| `--save-predictions` | Flag | Guardar predicciones por imagen en CSV | False |
| `--confusion-matrix` | Flag | Generar matriz de confusión (PNG) | False |

#### Entradas

- **Modelo:** Checkpoint `.pt`
- **Dataset:** Estructura con `train/`, `val/`, `test/`

#### Salidas

1. **`evaluation_results.json`** - Métricas detalladas
   ```json
   {
     "accuracy": 0.9805,
     "f1_macro": 0.9712,
     "f1_weighted": 0.9804,
     "precision_macro": 0.9721,
     "recall_macro": 0.9704,
     "per_class_f1": {
       "COVID": 0.9682,
       "Normal": 0.9845,
       "Viral_Pneumonia": 0.9581
     }
   }
   ```

2. **`predictions.csv`** (si `--save-predictions`)
3. **`confusion_matrix.png`** (si `--confusion-matrix`)

#### Ejemplo Concreto

```bash
python -m src_v2 evaluate-classifier \
  outputs/classifier_warped_lung_best/best_classifier.pt \
  --data-dir outputs/warped_lung_best/session_warping \
  --split test \
  --save-predictions \
  --confusion-matrix
```

**Salida esperada:**
```
Evaluating classifier...
Model: outputs/classifier_warped_lung_best/best_classifier.pt
Dataset: outputs/warped_lung_best/session_warping
Split: test (1895 images)

Evaluating: 100%|████████| 60/60 [00:15<00:00, 3.85it/s]

=== Test Results ===
Accuracy:    98.05%
F1-macro:    97.12%
F1-weighted: 98.04%

Per-class metrics:
              Precision  Recall  F1-Score  Support
COVID            97.23%  96.41%    96.82%     723
Normal           98.67%  98.24%    98.45%    1276
Viral_Pneumonia  94.87%  96.77%    95.81%     268

Saved results to: evaluation_results.json
Saved predictions to: predictions.csv
Saved confusion matrix to: confusion_matrix.png
```

**Tiempo:** 1-2 minutos (GPU) / 10-15 minutos (CPU)

---

### 6. cross-validate-classifier

**Propósito:** Realizar validación cruzada k-fold para estimar el desempeño del clasificador con mayor robustez estadística.

**Cuándo usar:** Para reportar métricas académicas con intervalos de confianza, o cuando se quiere evaluar la estabilidad del modelo.

#### Sintaxis

```bash
python -m src_v2 cross-validate-classifier [OPTIONS]
```

#### Parámetros

| Parámetro | Tipo | Descripción | Default |
|-----------|------|-------------|---------|
| `--data-dir` | Ruta | **Requerido.** Directorio del dataset | - |
| `--output-dir` | Ruta | Directorio de salida | `outputs/classifier_cv` |
| `--backbone` | Str | Arquitectura | `resnet18` |
| `--folds` | Int | Número de folds | 5 |
| `--epochs` | Int | Épocas por fold | 50 |
| `--batch-size` | Int | Tamaño de batch | 32 |
| `--lr` | Float | Learning rate | 0.0001 |
| `--seed` | Int | Semilla aleatoria | 42 |
| `--eval-test` | Flag | Evaluar cada fold en test set (no recomendado) | False |

#### Entradas

- **Dataset:** `DATA_DIR/train/`, `val/`, `test/`
- **Nota:** CV combina train+val y hace folds. Test se mantiene separado (holdout).

#### Salidas

Genera en `--output-dir`:

1. **`cross_validation_results.json`** - Métricas agregadas
   ```json
   {
     "val_accuracy_mean": 98.60,
     "val_accuracy_std": 0.26,
     "val_f1_macro_mean": 98.00,
     "val_f1_macro_std": 0.36
   }
   ```

2. **`fold_1/`, ..., `fold_5/`** - Modelos y resultados por fold
   - `best_classifier.pt`
   - `training_history.json`
   - `fold_results.json`

3. **`cv_summary.png`** - Boxplots de métricas por fold

#### Ejemplo Concreto

```bash
python -m src_v2 cross-validate-classifier \
  --data-dir outputs/warped_lung_best/session_warping \
  --output-dir outputs/classifier_cv \
  --backbone resnet18 \
  --folds 5 \
  --epochs 50 \
  --batch-size 32 \
  --seed 42
```

**Salida esperada:**
```
=== 5-Fold Cross-Validation ===
Dataset: outputs/warped_lung_best/session_warping
Train+Val: 13258 images
Test: 1895 images (held out)

Fold 1/5...
  Training 50 epochs...
  Best val_acc: 98.72% (epoch 43)

Fold 2/5...
  Best val_acc: 98.53% (epoch 47)

...

Fold 5/5...
  Best val_acc: 98.67% (epoch 45)

=== Cross-Validation Results ===
Validation Accuracy:  98.60% ± 0.26%
Validation F1-macro:  98.00% ± 0.36%

Individual folds:
  Fold 1: 98.72%
  Fold 2: 98.53%
  Fold 3: 98.65%
  Fold 4: 98.42%
  Fold 5: 98.67%

Results saved to: outputs/classifier_cv/cross_validation_results.json
```

**Tiempo:** 10-15 horas (GPU, 5 folds × 50 epochs)

---

### 7. evaluate-classifier-ensemble

**Propósito:** Evaluar el ensemble de los 5 modelos de cross-validation en el test set, opcionalmente con TTA.

**Cuándo usar:** Después de completar la validación cruzada, para obtener las métricas finales del ensemble en el conjunto de test.

#### Sintaxis

```bash
python -m src_v2 evaluate-classifier-ensemble [OPTIONS]
```

#### Parámetros

| Parámetro | Tipo | Descripción | Default |
|-----------|------|-------------|---------|
| `--data-dir` | Ruta | **Requerido.** Directorio del dataset | - |
| `--models` | Str | Patrón glob para modelos (ej: `fold_*/best_classifier.pt`) | - |
| `--output` | Ruta | Archivo JSON de salida | `ensemble_test_results.json` |
| `--split` | Str | Split a evaluar | `test` |
| `--tta` | Flag | Usar Test-Time Augmentation | False |
| `--save-predictions` | Flag | Guardar predicciones CSV | False |

#### Entradas

- **Modelos:** 5 checkpoints de CV (uno por fold)
- **Dataset:** Test set

#### Salidas

1. **`ensemble_test_results.json`** - Métricas del ensemble
2. **`predictions.csv`** (si `--save-predictions`)
3. **Log en consola** con impacto de TTA caso por caso

#### Ejemplo Concreto

**Sin TTA:**

```bash
python -m src_v2 evaluate-classifier-ensemble \
  --data-dir outputs/warped_lung_best/session_warping \
  --models "outputs/classifier_cv/fold_*/best_classifier.pt" \
  --output outputs/classifier_cv/ensemble_test_results.json \
  --split test
```

**Con TTA:**

```bash
python -m src_v2 evaluate-classifier-ensemble \
  --data-dir outputs/warped_lung_best/session_warping \
  --models "outputs/classifier_cv/fold_*/best_classifier.pt" \
  --output outputs/classifier_cv/ensemble_test_results_tta.json \
  --split test \
  --tta \
  --save-predictions
```

**Salida esperada (con TTA):**
```
Loading 5 models from CV folds...
  ✓ fold_1/best_classifier.pt
  ✓ fold_2/best_classifier.pt
  ...
  ✓ fold_5/best_classifier.pt

Evaluating ensemble on test set (1895 images) with TTA...
100%|████████████████| 60/60 [00:32<00:00, 1.87it/s]

=== Ensemble Test Results (With TTA) ===
Accuracy:    98.26%
F1-macro:    97.12%
F1-weighted: 98.25%

Improvement over No TTA:
  Accuracy:    +0.16%
  F1-macro:    +0.09%

TTA Impact on Cases:
  Helped:   6 cases (corrected by TTA)
  Hurt:     3 cases (worsened by TTA)
  Neutral:  1886 cases (no change)
  Net improvement: +3 cases

Per-class F1 delta:
  COVID:            +0.44%
  Normal:           +0.12%
  Viral_Pneumonia:  -0.28%

Results saved to: outputs/classifier_cv/ensemble_test_results_tta.json
```

**Tiempo:** 2-3 minutos (GPU sin TTA) / 4-6 minutos (GPU con TTA)

---

## Workflows Comunes

### Workflow 1: Pipeline Completo Desde Cero

**Objetivo:** Entrenar todo desde cero (landmarks + clasificador).

```bash
# 1. Forma canónica (GPA)
python -m src_v2 compute-canonical \
  data/coordenadas/coordenadas_maestro.csv \
  --output-dir outputs/shape_analysis --visualize

# 2. Entrenar modelos de landmarks (4 modelos con diferentes seeds)
# (Este paso toma ~24-48 horas por modelo en GPU)
python -m src_v2 train \
  --config configs/landmarks_train_base.json \
  --seed 123 --output outputs/landmarks/seed123

python -m src_v2 train \
  --config configs/landmarks_train_base.json \
  --seed 321 --output outputs/landmarks/seed321

# ... (repetir para seeds 111 y 666)

# 3. Generar cache de predicciones del ensemble
python scripts/predict_landmarks_dataset.py \
  --input-dir data/dataset/COVID-19_Radiography_Dataset \
  --output outputs/landmark_predictions/session_warping/predictions.npz \
  --ensemble-config configs/ensemble_best.json \
  --tta --clahe --clahe-clip 2.0 --clahe-tile 4

# 4. Generar dataset warped
python -m src_v2 generate-dataset --config configs/warping_best.json

# 5. Cross-validation del clasificador
python -m src_v2 cross-validate-classifier \
  --data-dir outputs/warped_lung_best/session_warping \
  --output-dir outputs/classifier_cv \
  --folds 5 --epochs 50

# 6. Evaluar ensemble del clasificador en test
python -m src_v2 evaluate-classifier-ensemble \
  --data-dir outputs/warped_lung_best/session_warping \
  --models "outputs/classifier_cv/fold_*/best_classifier.pt" \
  --split test --tta
```

**Tiempo total:** ~5-10 días (depende de hardware)

### Workflow 2: Solo Inferencia (Modelos Pre-entrenados)

**Objetivo:** Usar modelos del USB para validar métricas.

```bash
# 1. Forma canónica (rápido)
python -m src_v2 compute-canonical \
  data/coordenadas/coordenadas_maestro.csv \
  --output-dir outputs/shape_analysis --visualize

# 2. Evaluar ensemble de landmarks
python scripts/evaluate_ensemble_from_config.py \
  --config configs/ensemble_best.json

# 3. Generar cache de predicciones
python scripts/predict_landmarks_dataset.py \
  --input-dir data/dataset/COVID-19_Radiography_Dataset \
  --output outputs/landmark_predictions/session_warping/predictions.npz \
  --ensemble-config configs/ensemble_best.json \
  --tta --clahe --clahe-clip 2.0 --clahe-tile 4

# 4. Generar dataset warped
python -m src_v2 generate-dataset --config configs/warping_best.json
```

**Tiempo total:** ~30-60 minutos

### Workflow 3: Experimentar con Parámetros de Warping

**Objetivo:** Probar diferentes valores de margin sin re-predecir landmarks.

```bash
# Generar dataset con margin=1.0
python -m src_v2 generate-dataset \
  data/dataset/COVID-19_Radiography_Dataset \
  outputs/warped_margin_1.0/session_warping \
  --predictions outputs/landmark_predictions/session_warping/predictions.npz \
  --canonical outputs/shape_analysis/canonical_shape_gpa.json \
  --triangles outputs/shape_analysis/canonical_delaunay_triangles.json \
  --margin 1.0 \
  --splits 0.75,0.125,0.125 --seed 42 \
  --clahe --clahe-clip 2.0 --clahe-tile 4 --no-full-coverage

# Generar dataset con margin=1.1
python -m src_v2 generate-dataset \
  data/dataset/COVID-19_Radiography_Dataset \
  outputs/warped_margin_1.1/session_warping \
  --predictions outputs/landmark_predictions/session_warping/predictions.npz \
  --canonical outputs/shape_analysis/canonical_shape_gpa.json \
  --triangles outputs/shape_analysis/canonical_delaunay_triangles.json \
  --margin 1.1 \
  --splits 0.75,0.125,0.125 --seed 42 \
  --clahe --clahe-clip 2.0 --clahe-tile 4 --no-full-coverage

# Entrenar clasificadores en ambos
python -m src_v2 train-classifier \
  outputs/warped_margin_1.0/session_warping \
  --output-dir outputs/classifier_margin_1.0 --epochs 50

python -m src_v2 train-classifier \
  outputs/warped_margin_1.1/session_warping \
  --output-dir outputs/classifier_margin_1.1 --epochs 50
```

---

## Comandos Auxiliares

### version

Mostrar versión del paquete:

```bash
python -m src_v2 version
```

### generate-landmark-visualization-dataset

Generar dataset con landmarks visualizados sobre imágenes originales (útil para papers):

```bash
python -m src_v2 generate-landmark-visualization-dataset \
  --config configs/landmark_viz_best.json
```

### optimize-margin

Búsqueda automática del margin óptimo para warping:

```bash
python -m src_v2 optimize-margin \
  --checkpoint checkpoints/model.pt \
  --canonical outputs/shape_analysis/canonical_shape_gpa.json \
  --min-margin 1.0 --max-margin 1.2 --step 0.05
```

---

## Tips y Mejores Prácticas

### 1. Uso de Configuraciones JSON

**Recomendado:**
```bash
python -m src_v2 generate-dataset --config configs/warping_best.json
```

**Evitar (difícil de reproducir):**
```bash
python -m src_v2 generate-dataset INPUT OUTPUT --margin 1.05 --splits 0.75,0.125,0.125 --seed 42 --clahe --clahe-clip 2.0 --clahe-tile 4 --no-full-coverage --predictions predictions.npz --canonical canonical.json --triangles triangles.json
```

### 2. Siempre Usar Seeds para Reproducibilidad

```bash
--seed 42  # Todos los comandos que involucren splits o entrenamiento
```

### 3. Cachear Predicciones de Landmarks

**Primera vez:**
```bash
python scripts/predict_landmarks_dataset.py ... --output predictions.npz
```

**Uso posterior:**
```bash
python -m src_v2 generate-dataset --predictions predictions.npz
```

Ahorra 10-45 minutos cada vez que regenera el dataset.

### 4. Monitorear Entrenamientos Largos

```bash
nohup python -m src_v2 cross-validate-classifier \
  --data-dir outputs/warped_lung_best/session_warping \
  --output-dir outputs/classifier_cv \
  > cv_training.log 2>&1 &

# Monitorear en otra terminal
tail -f cv_training.log
```

### 5. Verificar Salidas Inmediatamente

Después de cada comando crítico:

```bash
# Después de compute-canonical
ls -lh outputs/shape_analysis/
cat outputs/shape_analysis/canonical_shape_gpa.json | head -n 20

# Después de generate-dataset
find outputs/warped_*/session_warping -name "*.png" | wc -l
cat outputs/warped_*/session_warping/metadata.json | jq '.fill_rate_mean'

# Después de entrenar clasificador
cat outputs/classifier_*/training_history.json | jq '.best_val_acc'
```

### 6. Usar --help Liberalmente

```bash
python -m src_v2 <command> --help
```

Cada comando tiene documentación detallada.

### 7. Gestión de Salidas

Organice outputs por experimento:

```
outputs/
├── experiment_baseline/
│   ├── shape_analysis/
│   ├── landmark_predictions/
│   ├── warped_dataset/
│   └── classifier/
├── experiment_margin_1.1/
│   └── ...
└── experiment_no_clahe/
    └── ...
```

---

## Comandos Adicionales Disponibles

Además de los comandos principales documentados, el sistema incluye:

- `train` - Entrenar modelo de landmarks desde cero
- `evaluate` - Evaluar modelo de landmarks individual
- `predict` - Predecir landmarks en una sola imagen
- `warp` - Aplicar warping a una imagen individual
- `classify` - Clasificar una sola imagen
- `test-robustness` - Evaluar robustez ante perturbaciones
- `generate-cropped-dataset` - Dataset recortado sin warping
- `extract-dataset-splits` - Extraer splits exactos del modelo warped_lung_best
- `generate-delaunay-mesh-dataset` - Visualizaciones con malla de Delaunay

Para ver todos los comandos:

```bash
python -m src_v2 --help
```

---

**Última actualización:** 28 de enero de 2026

**Contacto:**
- Estudiante: Rafael Alejandro Cruz Ovando, BUAP
- Director: Dr. Leopoldo Altamirano Robles (robles@inaoep.mx), INAOE
