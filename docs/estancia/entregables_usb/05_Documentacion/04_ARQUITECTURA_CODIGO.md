# Arquitectura del Código

**Guía de la Estructura del Sistema para Modificaciones y Extensiones**

Este documento describe la arquitectura completa del código del sistema de detección de landmarks pulmonares y clasificación COVID-19, facilitando la comprensión para extensiones o modificaciones.

---

## Tabla de Contenidos

1. [Visión General](#visión-general)
2. [Estructura de Directorios](#estructura-de-directorios)
3. [Módulos Principales](#módulos-principales)
4. [Flujo de Datos](#flujo-de-datos)
5. [Patrones de Diseño](#patrones-de-diseño)
6. [Puntos de Extensión](#puntos-de-extensión)
7. [Convenciones del Código](#convenciones-del-código)

---

## Visión General

### Estadísticas del Proyecto

- **Total de líneas:** ~25,000 líneas de código Python
- **Archivos Python:** 43 archivos principales
- **Módulos:** 7 módulos principales + 1 CLI
- **Dependencias:** 13 paquetes externos principales
- **Testing:** Suite de tests con pytest (cobertura >80%)

### Organización General

```
src_v2/                      # Código fuente principal (~14,264 líneas)
├── models/                  # Arquitecturas de redes neuronales
├── processing/              # GPA y warping geométrico
├── data/                    # Datasets y transformaciones
├── training/                # Entrenamiento y callbacks
├── evaluation/              # Métricas y ensemble
├── visualization/           # Generación de gráficos
├── gui/                     # Interfaz gráfica Gradio (opcional)
├── utils/                   # Utilidades generales
└── cli.py                   # Interfaz de línea de comandos (~10,895 líneas)
```

### Principios de Diseño

1. **Modularidad:** Cada módulo tiene responsabilidad única y bien definida
2. **Reproducibilidad:** Seeds aleatorias, configuraciones JSON, datasets versionados
3. **Extensibilidad:** Interfaces claras para agregar nuevos modelos, métricas, o visualizaciones
4. **Claridad:** Nombres descriptivos, type hints, docstrings completos

---

## Estructura de Directorios

### Árbol Completo

```
prediccion_warping_clasificacion/
│
├── src_v2/                          # Código fuente principal
│   ├── __init__.py                  # (267 B)
│   ├── cli.py                       # CLI principal (10,895 líneas)
│   ├── constants.py                 # Constantes globales
│   │
│   ├── models/                      # Arquitecturas de redes neuronales
│   │   ├── __init__.py              # (835 B)
│   │   ├── resnet_landmark.py       # (11 KB) ResNet-18 para landmarks
│   │   ├── classifier.py            # (12 KB) Clasificador COVID-19
│   │   ├── hierarchical.py          # (13 KB) Modelo jerárquico (experimental)
│   │   └── losses.py                # (14 KB) Wing Loss, Symmetry Loss, etc.
│   │
│   ├── processing/                  # Procesamiento geométrico
│   │   ├── __init__.py              # (970 B)
│   │   ├── gpa.py                   # (8.9 KB) Generalized Procrustes Analysis
│   │   └── warp.py                  # (13 KB) Warping afín por partes
│   │
│   ├── data/                        # Datasets y transformaciones
│   │   ├── __init__.py              # (365 B)
│   │   ├── dataset.py               # (13 KB) LandmarkDataset, create_dataloaders
│   │   ├── transforms.py            # (13 KB) CLAHE, TTA, aumentación
│   │   └── utils.py                 # (8.3 KB) Splits estratificados, normalización
│   │
│   ├── training/                    # Entrenamiento y callbacks
│   │   ├── __init__.py              # (267 B)
│   │   ├── trainer.py               # (14 KB) LandmarkTrainer (2 fases)
│   │   └── callbacks.py             # (6.6 KB) EarlyStopping, ModelCheckpoint
│   │
│   ├── evaluation/                  # Métricas y evaluación
│   │   ├── __init__.py              # (399 B)
│   │   ├── metrics.py               # (16 KB) Error de píxeles, métricas de clasificación
│   │   └── ensemble.py              # (15 KB) Evaluación de ensemble
│   │
│   ├── visualization/               # Generación de gráficos científicos
│   │   ├── __init__.py              # (2.5 KB)
│   │   ├── gradcam.py               # (12 KB) GradCAM para explicabilidad
│   │   ├── scientific_viz.py        # (16 KB) Visualizaciones para papers
│   │   ├── comparison_viz.py        # (34 KB) Comparaciones lado a lado
│   │   ├── error_analysis.py        # (17 KB) Análisis de fallos
│   │   ├── plot_roc_curves.py       # (30 KB) Curvas ROC multi-clase
│   │   ├── plot_failure_cases.py    # (25 KB) Casos mal clasificados
│   │   ├── pfs_analysis.py          # (22 KB) Pulmonary Focus Score
│   │   ├── feature_visualizer.py    # (16 KB) Visualización de features
│   │   ├── feature_extractor.py     # (9.0 KB) Extracción de features
│   │   ├── diagramming.py           # (9.2 KB) Diagramas de arquitectura
│   │   └── utils.py                 # (14 KB) Utilidades de visualización
│   │
│   ├── gui/                         # Interfaz gráfica (Gradio)
│   │   ├── __init__.py              # (207 B)
│   │   ├── app.py                   # (19 KB) Aplicación principal Gradio
│   │   ├── config.py                # (16 KB) Configuración de la GUI
│   │   ├── model_manager.py         # (15 KB) Gestión de modelos
│   │   ├── inference_pipeline.py    # (11 KB) Pipeline de inferencia
│   │   ├── gradcam_utils.py         # (7.9 KB) Utilidades GradCAM
│   │   └── visualizer.py            # (23 KB) Visualizaciones interactivas
│   │
│   └── utils/                       # Utilidades generales
│       ├── __init__.py
│       ├── logging.py               # Configuración de logging
│       └── file_io.py               # Operaciones de archivos
│
├── scripts/                         # Scripts auxiliares
│   ├── evaluate_ensemble_from_config.py   # Evaluar ensemble
│   ├── predict_landmarks_dataset.py       # Generar cache de predicciones
│   ├── quickstart_warping.sh              # Pipeline automático
│   └── verify_landmark_viz_dataset.py     # Verificaciones
│
├── configs/                         # Configuraciones JSON
│   ├── ensemble_best.json
│   ├── landmarks_train_base.json
│   ├── warping_best.json
│   └── classifier_warped_base.json
│
├── tests/                           # Suite de tests (pytest)
│   ├── test_processing.py           # Tests de GPA y warping
│   ├── test_models.py               # Tests de arquitecturas
│   ├── test_data.py                 # Tests de datasets
│   └── test_evaluation.py           # Tests de métricas
│
├── data/                            # Datos (no en repo)
│   ├── dataset/                     # Dataset original (~2 GB)
│   └── coordenadas/                 # Anotaciones de landmarks
│
├── checkpoints/                     # Modelos entrenados (184 MB en USB)
│   ├── session10/ensemble/seed123/
│   ├── session13/seed321/
│   ├── repro_split111/session14/seed111/
│   └── repro_split666/session16/seed666/
│
├── outputs/                         # Salidas generadas (no en repo)
│   ├── shape_analysis/              # Forma canónica y triangulación
│   ├── landmark_predictions/        # Cache de predicciones
│   ├── warped_lung_best/            # Dataset normalizado
│   └── classifier_*/                # Modelos de clasificación
│
├── docs/                            # Documentación
│   ├── REPRO_FULL_PIPELINE.md
│   ├── REPRO_ENSEMBLE_3_71.md
│   └── EXPERIMENTS.md
│
├── requirements.txt                 # Dependencias Python
├── pyproject.toml                   # Configuración del proyecto
├── README.md                        # Overview del proyecto
└── CLAUDE.md                        # Guía de desarrollo (interna)
```

---

## Módulos Principales

### 1. models/ - Arquitecturas de Redes Neuronales

**Propósito:** Definir las arquitecturas de deep learning para detección de landmarks y clasificación.

#### `resnet_landmark.py` (11 KB)

**Clase principal:** `ResNet18Landmarks`

**Responsabilidades:**
- Backbone: ResNet-18 pre-entrenado en ImageNet
- Cabeza de regresión: Predice 30 valores (15 landmarks × 2 coordenadas)
- Módulos opcionales:
  - **Coordinate Attention:** Módulo de atención para mejorar localización espacial
  - **Deep Head:** Cabeza de regresión profunda (3 capas FC vs. 1 capa)

**Arquitectura típica:**
```
Input (224×224×3)
    ↓
ResNet-18 Backbone (congelado en fase 1)
    ↓ [512 features]
CoordinateAttention (opcional)
    ↓
DeepHead (FC 512→768→512→30) o SimpleHead (FC 512→30)
    ↓
Output (30) → [L1_x, L1_y, ..., L15_x, L15_y] en [0, 1]
```

**Métodos clave:**
- `forward(x)` → Predicción de landmarks
- `freeze_backbone()` → Congelar ResNet para fase 1
- `unfreeze_backbone()` → Descongelar para fase 2

**Ejemplo de uso:**
```python
from src_v2.models import ResNet18Landmarks

model = ResNet18Landmarks(
    num_landmarks=15,
    coord_attention=True,
    deep_head=True,
    hidden_dim=768,
    dropout=0.3
)

# Fase 1: congelar backbone
model.freeze_backbone()

# Fase 2: descongelar
model.unfreeze_backbone()
```

#### `classifier.py` (12 KB)

**Clase principal:** `ImageClassifier`

**Responsabilidades:**
- Clasificación multi-clase (COVID, Normal, Viral Pneumonia)
- Backbones soportados: ResNet-18, EfficientNet-B0, DenseNet-121
- Cabeza de clasificación simple (avg pooling → FC → softmax)

**Métodos clave:**
- `forward(x)` → logits (N, 3)
- `predict(x)` → clases predichas
- `predict_proba(x)` → probabilidades

#### `losses.py` (14 KB)

**Funciones de pérdida implementadas:**

1. **Wing Loss** - Para regresión de landmarks (suave en valores pequeños, lineal en grandes)
   ```python
   WingLoss(w=10.0, epsilon=2.0)
   ```

2. **Symmetry Loss** - Penaliza asimetrías entre pares de landmarks (L3-L4, L5-L6, etc.)
   ```python
   SymmetryLoss(symmetric_pairs=SYMMETRIC_PAIRS, weight=0.1)
   ```

3. **Combined Loss** - Wing Loss + Symmetry Loss
   ```python
   CombinedLoss(wing_weight=1.0, symmetry_weight=0.1)
   ```

4. **Distance Preservation Loss** - Preserva distancias relativas entre landmarks (experimental)

**Decisión de diseño:** Wing Loss es la pérdida principal (mejor que MSE/L1 para landmarks). Symmetry Loss es opcional y aporta ~0.1 px de mejora.

---

### 2. processing/ - Procesamiento Geométrico

**Propósito:** Calcular forma canónica mediante GPA y aplicar warping afín por partes.

#### `gpa.py` (8.9 KB)

**Funciones principales:**

1. **`gpa_iterative(landmarks, max_iter=100, tol=1e-8)`**
   - Implementa Generalized Procrustes Analysis iterativo
   - Alinea múltiples formas y calcula forma de consenso
   - Pasos:
     1. Centrar cada forma (traslación)
     2. Escalar a tamaño unitario
     3. Rotar para alinear con forma media
     4. Repetir hasta convergencia

   **Entrada:** Array (N, 15, 2) con N formas
   **Salida:** Array (15, 2) con forma canónica

2. **`compute_delaunay_triangulation(landmarks)`**
   - Calcula triangulación de Delaunay de los 15 landmarks
   - **Salida:** Lista de 24 triángulos (cada uno: 3 índices de vértices)

3. **`scale_shape_to_image(shape, image_size=224, padding=0.1)`**
   - Escala forma canónica normalizada [0,1] a coordenadas de imagen

**Uso en el pipeline:**
```python
from src_v2.processing.gpa import gpa_iterative, compute_delaunay_triangulation

# Cargar landmarks de entrenamiento
landmarks = load_training_landmarks()  # (N, 15, 2)

# Calcular forma canónica
canonical_shape = gpa_iterative(landmarks)

# Calcular triangulación
triangles = compute_delaunay_triangulation(canonical_shape)
```

#### `warp.py` (13 KB)

**Función principal:** `piecewise_affine_warp(image, src_landmarks, dst_landmarks, triangles)`

**Responsabilidades:**
- Warping afín por partes usando triangulación de Delaunay
- Para cada triángulo:
  1. Calcular matriz de transformación afín (src → dst)
  2. Aplicar `cv2.warpAffine` con máscara del triángulo
- Componer resultado final

**Funciones auxiliares:**

1. **`scale_landmarks_from_centroid(landmarks, margin_scale=1.05)`**
   - Expande landmarks desde el centroide por un factor
   - Usado para agregar margen alrededor del ROI

2. **`compute_fill_rate(image)`**
   - Calcula porcentaje de píxeles no negros
   - Métrica de calidad del warping

3. **`extract_roi_from_landmarks(image, landmarks, margin=10)`**
   - Extrae región rectangular alrededor de landmarks

**Proceso completo de warping:**
```
Input Image (299×299)
    ↓ Predict landmarks
Landmarks src (15×2) en imagen original
    ↓ Scale from centroid (margin=1.05)
Landmarks expanded
    ↓ Piecewise Affine Warp (24 triángulos)
Warped Image aligned to canonical shape
    ↓ Crop to bounding box
Output Image (224×224) normalized
```

---

### 3. data/ - Datasets y Transformaciones

**Propósito:** Cargar y preparar datos para entrenamiento e inferencia.

#### `dataset.py` (13 KB)

**Clase principal:** `LandmarkDataset(torch.utils.data.Dataset)`

**Responsabilidades:**
- Cargar imágenes de radiografías
- Cargar anotaciones de landmarks desde CSV
- Aplicar transformaciones (CLAHE, aumentación, normalización)
- Soportar splits (train/val/test)

**Atributos:**
- `image_paths`: Lista de rutas a imágenes
- `landmarks`: Array (N, 30) con coordenadas [0, 299]
- `labels`: Array (N,) con índices de clase
- `transform`: Transformaciones de torchvision

**Métodos:**
- `__getitem__(idx)` → (image_tensor, landmarks_tensor, label)
- `__len__()` → Número de muestras

**Función auxiliar:** `create_dataloaders(csv_path, image_dir, batch_size, splits=(0.75, 0.125, 0.125), seed=42)`
- Crea DataLoaders para train/val/test con splits estratificados
- Garantiza reproducibilidad con `seed`

#### `transforms.py` (13 KB)

**Transformaciones implementadas:**

1. **`CLAHETransform(clip_limit=2.0, tile_grid_size=(4, 4))`**
   - Contrast Limited Adaptive Histogram Equalization
   - Mejora contraste local en radiografías

2. **`TTATransform(apply_flip=True, symmetric_pairs=SYMMETRIC_PAIRS)`**
   - Test-Time Augmentation con flip horizontal
   - Corrige índices de landmarks simétricos después del flip
   - Ejemplo: L3 (izquierda) ↔ L4 (derecha) tras flip

3. **`LandmarkAugmentation(rotation=10.0, flip_prob=0.5)`**
   - Aumentación durante entrenamiento
   - Rotación aleatoria ± rotation grados
   - Flip horizontal con probabilidad flip_prob

4. **`NormalizeToModel(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`**
   - Normalización con estadísticas de ImageNet
   - Requerido para backbones pre-entrenados

**Ejemplo de pipeline de transformaciones:**
```python
from src_v2.data.transforms import CLAHETransform, NormalizeToModel

transform = transforms.Compose([
    CLAHETransform(clip_limit=2.0, tile_grid_size=(4, 4)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    NormalizeToModel()
])
```

#### `utils.py` (8.3 KB)

**Funciones útiles:**
- `create_stratified_splits(labels, ratios=(0.75, 0.125, 0.125), seed=42)` → Splits balanceados
- `denormalize_landmarks(landmarks, image_size=224)` → [0,1] → [0, image_size]
- `normalize_landmarks(landmarks, image_size=299)` → [0, image_size] → [0,1]
- `compute_class_weights(labels)` → Pesos para CrossEntropyLoss

---

### 4. training/ - Entrenamiento y Callbacks

#### `trainer.py` (14 KB)

**Clase principal:** `LandmarkTrainer`

**Responsabilidades:**
- Entrenamiento en 2 fases para modelos de landmarks
- Gestión de learning rates diferenciados (backbone vs. head)
- Logging de métricas (loss, error de píxeles)
- Guardado de checkpoints

**Método principal:** `train(model, train_loader, val_loader, config)`

**Fases de entrenamiento:**

**Fase 1 (15 epochs):**
- Backbone congelado (pesos pre-entrenados de ImageNet)
- Solo entrena cabeza de regresión
- LR alto: 1e-3
- Objetivo: Estabilizar predicción inicial

**Fase 2 (100 epochs):**
- Backbone descongelado
- Learning rates diferenciados:
  - Backbone: 2e-5 (muy bajo para no destruir features pre-entrenadas)
  - Head: 2e-4 (10× mayor)
- Early stopping con paciencia=15

**Ejemplo de uso:**
```python
from src_v2.training import LandmarkTrainer

trainer = LandmarkTrainer()
history = trainer.train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config={
        'phase1_epochs': 15,
        'phase1_lr': 1e-3,
        'phase2_epochs': 100,
        'phase2_backbone_lr': 2e-5,
        'phase2_head_lr': 2e-4,
    }
)
```

#### `callbacks.py` (6.6 KB)

**Callbacks implementados:**

1. **`EarlyStopping(patience=15, min_delta=0.001)`**
   - Detiene entrenamiento si val_loss no mejora por `patience` epochs
   - Previene overfitting

2. **`ModelCheckpoint(save_dir, monitor='val_loss', mode='min')`**
   - Guarda mejor modelo según métrica monitoreada
   - Guarda checkpoint cada N epochs

---

### 5. evaluation/ - Métricas y Evaluación

#### `metrics.py` (16 KB)

**Métricas para landmarks:**

1. **`compute_pixel_error(pred_landmarks, gt_landmarks, image_size=224)`**
   - Calcula distancia euclidiana en píxeles
   - **Entrada:** Landmarks en [0,1]
   - **Salida:** Error medio, std, mediana, máximo

2. **`compute_per_landmark_error(pred, gt, image_size=224)`**
   - Error por cada uno de los 15 landmarks

3. **`compute_error_by_category(pred, gt, labels, categories)`**
   - Error estratificado por categoría (COVID, Normal, Viral Pneumonia)

**Métricas para clasificación:**

1. **`compute_classification_metrics(y_true, y_pred)`**
   - Accuracy, precision, recall, F1 (macro y weighted)
   - Matriz de confusión
   - Métricas por clase

2. **`compute_roc_auc(y_true, y_proba, num_classes=3)`**
   - AUC-ROC multi-clase (one-vs-rest)

#### `ensemble.py` (15 KB)

**Función principal:** `evaluate_ensemble(models, dataloader, tta=True, clahe=True)`

**Proceso:**
1. Cada modelo predice landmarks para cada imagen
2. Promedio de predicciones (ensemble simple)
3. Si TTA:
   - Predicción en imagen original
   - Predicción en imagen flipped
   - Promedio con corrección de simetría
4. Calcular métricas finales

**Mejora típica del ensemble:**
- Ensemble de 4 modelos: ~0.5 px mejor que mejor modelo individual
- TTA adicional: ~0.1 px mejor

---

### 6. visualization/ - Generación de Gráficos

Este módulo es el más extenso (~200 KB) y contiene herramientas para generar todas las visualizaciones del paper.

#### `gradcam.py` (12 KB)

**Clase:** `GradCAM`

**Responsabilidades:**
- Gradient-weighted Class Activation Mapping
- Visualiza regiones importantes para la decisión del clasificador

**Uso:**
```python
from src_v2.visualization import GradCAM

gradcam = GradCAM(model, target_layer='layer4')
heatmap = gradcam.generate(image, target_class=0)  # COVID class
```

#### `scientific_viz.py` (16 KB)

Visualizaciones para papers científicos:
- `plot_landmarks_on_image(image, landmarks)` - Landmarks sobre imagen
- `plot_triangulation(image, landmarks, triangles)` - Malla de Delaunay
- `plot_canonical_shape(shape, triangles)` - Forma canónica
- `plot_error_distribution(errors)` - Histograma de errores
- `plot_training_curves(history)` - Loss y accuracy vs. epochs

#### Otros módulos de visualización

- `error_analysis.py` - Análisis de casos fallidos
- `plot_roc_curves.py` - Curvas ROC multi-clase
- `comparison_viz.py` - Comparaciones lado a lado (original vs. warped)
- `pfs_analysis.py` - Análisis de Pulmonary Focus Score

---

### 7. cli.py - Interfaz de Línea de Comandos

**Tamaño:** 10,895 líneas (archivo más grande del proyecto)

**Framework:** Typer (CLI moderno con type hints)

**Comandos implementados:** 40+ comandos

**Estructura típica de un comando:**
```python
@app.command()
def train_classifier(
    data_dir: str = typer.Argument(..., help="Dataset directory"),
    output_dir: str = typer.Option("outputs/classifier", help="Output directory"),
    config: Optional[str] = typer.Option(None, help="Config JSON"),
    ...
):
    """Entrenar clasificador CNN para COVID-19."""
    # 1. Cargar config si existe
    # 2. Crear dataloaders
    # 3. Inicializar modelo
    # 4. Entrenar
    # 5. Evaluar en test
    # 6. Guardar resultados
```

**Ventajas del diseño:**
- Documentación automática (`--help`)
- Validación de tipos automática
- Configs JSON para reproducibilidad
- Logging consistente

---

## Flujo de Datos

### Pipeline Completo (Diagrama ASCII)

```
┌─────────────────────────────────────────────────────────────────┐
│                  FASE 1: FORMA CANÓNICA (GPA)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
    coordenadas_maestro.csv   │
    (15153 imágenes, 15 landmarks c/u)
                              ↓
                 ┌─────────────────────┐
                 │  gpa_iterative()    │
                 │  - Centrar          │
                 │  - Escalar          │
                 │  - Rotar            │
                 │  - Promediar        │
                 └─────────────────────┘
                              ↓
            canonical_shape_gpa.json (15 landmarks)
            canonical_delaunay_triangles.json (24 triángulos)


┌─────────────────────────────────────────────────────────────────┐
│           FASE 2: ENTRENAMIENTO DE MODELOS DE LANDMARKS         │
│                     (Opcional, modelos pre-entrenados)          │
└─────────────────────────────────────────────────────────────────┘
                              │
    Dataset original (15153 imágenes 299×299)
    + Anotaciones CSV
                              ↓
              ┌─────────────────────────────┐
              │  LandmarkDataset            │
              │  + CLAHETransform           │
              │  + LandmarkAugmentation     │
              └─────────────────────────────┘
                              ↓
              ┌─────────────────────────────┐
              │  ResNet18Landmarks          │
              │  + CoordinateAttention      │
              │  + DeepHead                 │
              └─────────────────────────────┘
                              ↓
              ┌─────────────────────────────┐
              │  LandmarkTrainer            │
              │  Fase 1: backbone frozen    │
              │  Fase 2: fine-tuning        │
              │  Loss: Wing Loss            │
              └─────────────────────────────┘
                              ↓
      4 modelos entrenados (seeds 123, 321, 111, 666)
                 ~46 MB c/u, error ~4.1 px individual


┌─────────────────────────────────────────────────────────────────┐
│           FASE 3: PREDICCIÓN DE LANDMARKS (ENSEMBLE)            │
└─────────────────────────────────────────────────────────────────┘
                              │
    Dataset original (15153 imágenes)
    + 4 modelos del ensemble
                              ↓
              ┌─────────────────────────────┐
              │  Ensemble Prediction        │
              │  - Cargar 4 modelos         │
              │  - Predecir c/imagen        │
              │  - Promediar predicciones   │
              │  + TTA (flip horizontal)    │
              └─────────────────────────────┘
                              ↓
      predictions.npz (cache: 3.6 MB)
      - Predictions: (15153, 30)
      - Image paths
      - Metadata: models, TTA, CLAHE

      Error del ensemble: 3.61 px


┌─────────────────────────────────────────────────────────────────┐
│         FASE 4: GENERACIÓN DE DATASET NORMALIZADO (WARPING)     │
└─────────────────────────────────────────────────────────────────┘
                              │
    Dataset original (15153×299×299)
    + predictions.npz (landmarks predichos)
    + canonical_shape_gpa.json
    + canonical_delaunay_triangles.json
                              ↓
          ┌───────────────────────────────────┐
          │  Para cada imagen:                │
          │  1. Cargar imagen original        │
          │  2. Cargar landmarks predichos    │
          │  3. Scale from centroid (1.05×)   │
          │  4. Piecewise affine warp         │
          │     (24 triángulos)               │
          │  5. Crop a bounding box           │
          │  6. Resize a 224×224              │
          │  7. CLAHE (opcional)              │
          └───────────────────────────────────┘
                              ↓
    Dataset normalizado (15153×224×224)
    - train/ (11364 imágenes, 75%)
    - val/   (1894 imágenes, 12.5%)
    - test/  (1895 imágenes, 12.5%)
    Fill rate medio: ~47%


┌─────────────────────────────────────────────────────────────────┐
│        FASE 5: CLASIFICACIÓN COVID-19 (CROSS-VALIDATION)        │
└─────────────────────────────────────────────────────────────────┘
                              │
    Dataset normalizado (train+val: 13258)
                              ↓
          ┌───────────────────────────────────┐
          │  5-Fold Cross-Validation          │
          │                                   │
          │  Para cada fold:                  │
          │  1. Split train/val               │
          │  2. ImageClassifier (ResNet-18)   │
          │  3. Train 50 epochs               │
          │     Loss: CrossEntropyLoss        │
          │     + Class weights               │
          │  4. Early stopping                │
          │  5. Guardar best_classifier.pt    │
          └───────────────────────────────────┘
                              ↓
    5 modelos de CV (~45 MB c/u)
    Val accuracy: 98.60% ± 0.26%
    Val F1-macro: 98.00% ± 0.36%


┌─────────────────────────────────────────────────────────────────┐
│           FASE 6: EVALUACIÓN FINAL EN TEST SET                  │
└─────────────────────────────────────────────────────────────────┘
                              │
    Test set (1895 imágenes)
    + 5 modelos de CV
                              ↓
          ┌───────────────────────────────────┐
          │  Ensemble de Clasificadores       │
          │  - Promedio de logits             │
          │  + TTA (flip horizontal)          │
          └───────────────────────────────────┘
                              ↓
         Test accuracy: 98.26%
         Test F1-macro: 97.12%
         TTA improvement: +3 casos
```

---

## Patrones de Diseño

### 1. Factory Pattern (Creación de Modelos)

```python
# models/__init__.py
def create_landmark_model(architecture='resnet18', **kwargs):
    if architecture == 'resnet18':
        return ResNet18Landmarks(**kwargs)
    elif architecture == 'hierarchical':
        return HierarchicalLandmarkModel(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
```

### 2. Strategy Pattern (Funciones de Pérdida)

```python
# Selección dinámica de loss function
loss_functions = {
    'wing': WingLoss(),
    'mse': torch.nn.MSELoss(),
    'combined': CombinedLoss(wing_weight=1.0, symmetry_weight=0.1)
}

criterion = loss_functions[config['loss']]
```

### 3. Observer Pattern (Callbacks de Entrenamiento)

```python
# training/callbacks.py
class TrainingCallback:
    def on_epoch_end(self, epoch, logs):
        pass

class EarlyStopping(TrainingCallback):
    def on_epoch_end(self, epoch, logs):
        if logs['val_loss'] > self.best_loss + self.min_delta:
            self.patience_counter += 1
        ...
```

### 4. Singleton Pattern (Configuración Global)

```python
# constants.py
SYMMETRIC_PAIRS = [(2, 3), (4, 5), (6, 7), (11, 12), (13, 14)]
CENTRAL_LANDMARKS = [0, 8, 9, 10, 1]
DEFAULT_IMAGE_SIZE = 224
OPTIMAL_MARGIN_SCALE = 1.05
DEFAULT_CLAHE_TILE_SIZE = 4
```

---

## Puntos de Extensión

### 1. Agregar Nuevo Backbone para Landmarks

```python
# src_v2/models/resnet_landmark.py

class EfficientNetLandmarks(nn.Module):
    def __init__(self, num_landmarks=15, ...):
        super().__init__()
        # Cargar EfficientNet pre-entrenado
        self.backbone = models.efficientnet_b0(weights='DEFAULT')
        # ... resto de la implementación
```

**Archivos a modificar:**
- `src_v2/models/resnet_landmark.py` (o crear nuevo archivo)
- `src_v2/models/__init__.py` (agregar a imports)
- `src_v2/cli.py` (agregar opción en `--architecture`)

### 2. Agregar Nueva Función de Pérdida

```python
# src_v2/models/losses.py

class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super().__init__()
        self.omega = omega
        # ... implementación

    def forward(self, pred, target):
        # ... cálculo de pérdida
        return loss
```

**Integración:**
```python
# src_v2/training/trainer.py
loss_functions['adaptive_wing'] = AdaptiveWingLoss()
```

### 3. Agregar Nueva Métrica

```python
# src_v2/evaluation/metrics.py

def compute_normalized_error(pred, gt, image_size=224):
    """
    Compute Normalized Mean Error (NME).
    """
    # Normalizar por tamaño de imagen
    error = np.linalg.norm(pred - gt, axis=1)
    nme = error / image_size
    return nme.mean()
```

**Uso:**
```python
# En scripts de evaluación
nme = compute_normalized_error(predictions, ground_truth)
print(f"NME: {nme:.4f}")
```

### 4. Agregar Nueva Transformación

```python
# src_v2/data/transforms.py

class GaussianNoiseTransform:
    def __init__(self, mean=0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        noise = torch.randn_like(img) * self.std + self.mean
        return img + noise
```

**Integración:**
```python
# En dataset
transform = transforms.Compose([
    CLAHETransform(),
    GaussianNoiseTransform(std=0.01),  # Nueva transformación
    transforms.ToTensor(),
])
```

### 5. Agregar Nuevo Comando CLI

```python
# src_v2/cli.py

@app.command()
def analyze_symmetry(
    checkpoint: str = typer.Argument(..., help="Model checkpoint"),
    output_dir: str = typer.Option("outputs/symmetry_analysis")
):
    """
    Analizar simetría de las predicciones del modelo.
    """
    # 1. Cargar modelo
    model = torch.load(checkpoint)

    # 2. Evaluar en test set
    predictions = evaluate_model(model, test_loader)

    # 3. Calcular simetría
    symmetry_errors = compute_symmetry_errors(predictions)

    # 4. Generar visualizaciones
    plot_symmetry_analysis(symmetry_errors, output_dir)

    print(f"Results saved to {output_dir}")
```

---

## Convenciones del Código

### Naming Conventions

- **Variables:** `snake_case` (ej: `canonical_shape`, `val_accuracy`)
- **Funciones:** `snake_case` (ej: `compute_pixel_error`, `gpa_iterative`)
- **Clases:** `PascalCase` (ej: `ResNet18Landmarks`, `LandmarkDataset`)
- **Constantes:** `UPPER_SNAKE_CASE` (ej: `SYMMETRIC_PAIRS`, `DEFAULT_IMAGE_SIZE`)

### Type Hints

Todos los parámetros de funciones usan type hints:

```python
def compute_pixel_error(
    pred_landmarks: np.ndarray,    # (N, 30)
    gt_landmarks: np.ndarray,      # (N, 30)
    image_size: int = 224
) -> Tuple[float, float, float, float]:
    """
    Compute pixel error between predicted and ground truth landmarks.

    Args:
        pred_landmarks: Predicted landmarks in [0,1]
        gt_landmarks: Ground truth landmarks in [0,1]
        image_size: Image size for denormalization

    Returns:
        Tuple of (mean_error, std, median, max_error) in pixels
    """
    ...
```

### Docstrings

Google-style docstrings para todas las funciones públicas.

### Imports

Organizados en 3 bloques:

```python
# 1. Standard library
import os
import json
from pathlib import Path
from typing import List, Tuple, Optional

# 2. Third-party
import numpy as np
import torch
import cv2
from sklearn.metrics import accuracy_score

# 3. Local imports
from src_v2.models import ResNet18Landmarks
from src_v2.data import LandmarkDataset
from src_v2.constants import SYMMETRIC_PAIRS
```

### Logging

Uso consistente de Python logging:

```python
import logging

logger = logging.getLogger(__name__)

logger.info(f"Training {model_name} for {epochs} epochs")
logger.warning(f"Validation loss increased: {val_loss:.4f}")
logger.error(f"Failed to load checkpoint: {checkpoint_path}")
```

---

## Referencias

### Documentos Relacionados

- `01_GUIA_INICIO_RAPIDO.md` - Instalación y ejecución rápida
- `03_GUIA_USO_CLI.md` - Referencia completa de comandos
- `06_CONFIGURACIONES_JSON.md` - Sistema de configuración
- `07_MODELOS_ENTRENADOS.md` - Modelos del ensemble

### Código Fuente

- **GitHub:** (no disponible públicamente actualmente)
- **USB:** `02_Codigo/src_v2/`

---

**Última actualización:** 28 de enero de 2026

**Contacto:**
- Estudiante: Rafael Alejandro Cruz Ovando, BUAP
- Director: Dr. Leopoldo Altamirano Robles (robles@inaoep.mx), INAOE
