# Formatos de Datos

**Especificación Completa de Formatos de Entrada y Salida**

Este documento describe todos los formatos de datos utilizados en el sistema, desde las imágenes originales hasta las salidas de clasificación.

---

## Tabla de Contenidos

1. [Dataset de Entrada](#dataset-de-entrada)
2. [Formato de Anotaciones](#formato-de-anotaciones)
3. [Formato de Predicciones (Cache)](#formato-de-predicciones-cache)
4. [Dataset Normalizado (Warped)](#dataset-normalizado-warped)
5. [Salidas de Clasificación](#salidas-de-clasificación)
6. [Formatos Auxiliares](#formatos-auxiliares)

---

## Dataset de Entrada

### Imágenes Originales

**Formato:** PNG (Portable Network Graphics)

**Especificaciones:**
- **Resolución:** 299×299 píxeles
- **Profundidad de color:** 8 bits por canal
- **Canales:** 1 canal (escala de grises)
- **Compresión:** PNG sin pérdida
- **Tamaño de archivo:** ~10-30 KB por imagen

**Características:**
- Radiografías de tórax posteroanterior (PA)
- Sin información DICOM (imágenes anónimas)
- Formato estándar compatible con OpenCV, PIL, etc.

**Ejemplo de lectura:**
```python
import cv2

# Leer como escala de grises
image = cv2.imread('COVID-1.png', cv2.IMREAD_GRAYSCALE)
print(f"Shape: {image.shape}")  # (299, 299)
print(f"Dtype: {image.dtype}")  # uint8
print(f"Range: [{image.min()}, {image.max()}]")  # [0, 255]
```

### Estructura de Directorios

```
data/dataset/COVID-19_Radiography_Dataset/
├── COVID/
│   ├── images/
│   │   ├── COVID-1.png
│   │   ├── COVID-2.png
│   │   ├── ...
│   │   └── COVID-3616.png          (3,616 imágenes)
│   └── masks/                      (no usado en este proyecto)
│       ├── COVID-1.png
│       └── ...
├── Normal/
│   ├── images/
│   │   ├── Normal-1.png
│   │   ├── ...
│   │   └── Normal-10192.png        (10,192 imágenes)
│   └── masks/
└── Viral Pneumonia/
    ├── images/
    │   ├── Viral Pneumonia-1.png
    │   ├── ...
    │   └── Viral Pneumonia-1345.png (1,345 imágenes)
    └── masks/
```

**Total:** 15,153 imágenes

**Distribución:**
- **Normal:** 67.3% (10,192 imágenes)
- **COVID:** 23.9% (3,616 imágenes)
- **Viral Pneumonia:** 8.9% (1,345 imágenes)

**Nota:** El dataset está **desbalanceado**. Por esto se usan `class_weights` durante entrenamiento del clasificador.

### Naming Convention

**Formato:** `<CATEGORY>-<ID>.png`

**Ejemplos:**
- `COVID-1.png`
- `Normal-5432.png`
- `Viral Pneumonia-123.png`

**ID:** Número secuencial sin relleno de ceros (1, 2, ..., 10192, no 00001, 00002).

---

## Formato de Anotaciones

### Archivo CSV de Anotaciones

**Ubicación:** `data/coordenadas/coordenadas_maestro.csv`

**Tamaño:** ~2 MB

**Formato:** CSV con header

**Estructura:**
```csv
image_name,category,L1_x,L1_y,L2_x,L2_y,L3_x,L3_y,...,L15_x,L15_y
```

**Columnas:**

| Índice | Nombre | Tipo | Descripción |
|--------|--------|------|-------------|
| 0 | `image_name` | String | Nombre de archivo (ej: `COVID-1.png`) |
| 1 | `category` | String | Categoría: `COVID`, `Normal`, `Viral_Pneumonia` |
| 2-31 | `L1_x`, `L1_y`, ..., `L15_x`, `L15_y` | Float | Coordenadas de 15 landmarks en píxeles |

**Total de columnas:** 32 (1 nombre + 1 categoría + 30 coordenadas)

**Total de filas:** 15,154 (1 header + 15,153 imágenes)

### Especificación de Landmarks

**Número de landmarks:** 15

**Coordenadas:**
- Sistema: Cartesiano (origen en esquina superior izquierda)
- Unidades: Píxeles
- Rango: [0, 299] (resolución de imagen original)
- Precisión: Flotante (ej: 145.67)

**Orden de landmarks:**

| ID | Nombre | Ubicación Anatómica | Tipo |
|----|--------|---------------------|------|
| L1 | Landmark 1 | Tope del contorno central | Central |
| L2 | Landmark 2 | Base del contorno central | Central |
| L3 | Landmark 3 | Contorno lateral izquierdo (superior) | Left |
| L4 | Landmark 4 | Contorno lateral derecho (superior) | Right |
| L5 | Landmark 5 | Contorno lateral izquierdo (medio-superior) | Left |
| L6 | Landmark 6 | Contorno lateral derecho (medio-superior) | Right |
| L7 | Landmark 7 | Contorno lateral izquierdo (medio-inferior) | Left |
| L8 | Landmark 8 | Contorno lateral derecho (medio-inferior) | Right |
| L9 | Landmark 9 | Contorno central (t=0.25) | Central |
| L10 | Landmark 10 | Contorno central (t=0.50) | Central |
| L11 | Landmark 11 | Contorno central (t=0.75) | Central |
| L12 | Landmark 12 | Ápex izquierdo | Left |
| L13 | Landmark 13 | Ápex derecho | Right |
| L14 | Landmark 14 | Base izquierda | Left |
| L15 | Landmark 15 | Base derecha | Right |

**Pares simétricos (5 pares):**
- (L3, L4) - Superior lateral
- (L5, L6) - Medio-superior lateral
- (L7, L8) - Medio-inferior lateral
- (L12, L13) - Ápex
- (L14, L15) - Base

**Landmarks centrales (5):**
- L1, L9, L10, L11, L2 (eje vertical del pulmón)

**Nota:** Los landmarks definen el **contorno pulmonar**, NO puntos anatómicos específicos (como carina, ángulos costofrénicos, etc.).

### Ejemplo de Entrada CSV

```csv
image_name,category,L1_x,L1_y,L2_x,L2_y,L3_x,L3_y,L4_x,L4_y,L5_x,L5_y,L6_x,L6_y,L7_x,L7_y,L8_x,L8_y,L9_x,L9_y,L10_x,L10_y,L11_x,L11_y,L12_x,L12_y,L13_x,L13_y,L14_x,L14_y,L15_x,L15_y
COVID-1.png,COVID,149.5,45.2,149.3,253.8,72.1,78.3,226.8,78.5,45.2,125.6,253.7,125.9,67.4,173.2,231.2,173.5,149.4,98.7,149.3,150.1,149.3,201.5,85.3,58.9,213.6,59.1,92.7,237.4,206.2,237.6
Normal-1.png,Normal,150.1,42.8,150.0,255.3,75.3,75.1,224.9,75.3,48.6,122.4,251.5,122.7,70.8,169.8,229.3,170.1,150.1,96.2,150.0,147.5,150.0,198.9,88.7,56.5,211.4,56.7,95.1,239.7,205.0,239.9
```

### Lectura de Anotaciones

**Con pandas:**
```python
import pandas as pd

# Leer CSV
df = pd.read_csv('data/coordenadas/coordenadas_maestro.csv')

print(f"Total de imágenes: {len(df)}")  # 15153
print(f"Columnas: {df.columns.tolist()}")

# Extraer landmarks de una imagen
row = df.iloc[0]
image_name = row['image_name']
category = row['category']

# Coordenadas (30 valores)
coords = row.iloc[2:].values.astype(float)  # [L1_x, L1_y, ..., L15_y]
landmarks = coords.reshape(15, 2)  # (15, 2)

print(f"Imagen: {image_name}, Categoría: {category}")
print(f"Landmarks shape: {landmarks.shape}")
print(f"L1: ({landmarks[0, 0]:.1f}, {landmarks[0, 1]:.1f})")
```

**Con NumPy (solo coordenadas):**
```python
import numpy as np

# Leer solo coordenadas (skip header, tomar columnas 2-31)
data = np.loadtxt('data/coordenadas/coordenadas_maestro.csv',
                  delimiter=',', skiprows=1, usecols=range(2, 32))

print(f"Shape: {data.shape}")  # (15153, 30)
landmarks_all = data.reshape(-1, 15, 2)  # (15153, 15, 2)
```

---

## Formato de Predicciones (Cache)

### Archivo NPZ de Predicciones

**Propósito:** Cachear predicciones de landmarks para todo el dataset, evitando re-inferencia.

**Ubicación típica:** `outputs/landmark_predictions/session_warping/predictions.npz`

**Formato:** NumPy compressed archive (`.npz`)

**Tamaño:** ~3-4 MB (comprimido)

**Generación:**
```bash
python scripts/predict_landmarks_dataset.py \
  --input-dir data/dataset/COVID-19_Radiography_Dataset \
  --output outputs/landmark_predictions/session_warping/predictions.npz \
  --ensemble-config configs/ensemble_best.json \
  --tta --clahe --clahe-clip 2.0 --clahe-tile 4
```

### Contenido del NPZ

**Arrays almacenados:**

| Key | Shape | Dtype | Descripción |
|-----|-------|-------|-------------|
| `predictions` | (15153, 30) | float32 | Landmarks predichos en [0,1] |
| `image_paths` | (15153,) | object (str) | Rutas completas a imágenes |

**Metadata (atributos):**

| Key | Tipo | Descripción |
|-----|------|-------------|
| `models` | List[str] | Rutas a checkpoints usados |
| `tta` | bool | Si se usó TTA |
| `clahe` | bool | Si se usó CLAHE |
| `clahe_clip` | float | Clip limit de CLAHE |
| `clahe_tile_size` | int | Tile size de CLAHE |
| `timestamp` | str | Fecha/hora de generación |
| `num_images` | int | Total de imágenes procesadas |

### Lectura del Cache

```python
import numpy as np

# Cargar cache
cache = np.load('outputs/landmark_predictions/session_warping/predictions.npz',
                allow_pickle=True)

# Extraer arrays
predictions = cache['predictions']  # (15153, 30) en [0,1]
image_paths = cache['image_paths']  # (15153,) strings

# Extraer metadata
metadata = {k: cache[k].item() for k in cache.files if k not in ['predictions', 'image_paths']}

print(f"Imágenes: {len(image_paths)}")
print(f"Predictions shape: {predictions.shape}")
print(f"Modelos usados: {metadata.get('models', 'N/A')}")
print(f"TTA: {metadata.get('tta', False)}")
print(f"CLAHE: {metadata.get('clahe', False)}")

# Landmarks de una imagen específica
idx = 0
landmarks_normalized = predictions[idx]  # (30,) en [0,1]
landmarks_xy = landmarks_normalized.reshape(15, 2)
print(f"\nImagen: {image_paths[idx]}")
print(f"Landmarks (normalized):\n{landmarks_xy}")
```

### Ventajas del Cache

1. **Velocidad:** Evita re-inferencia (~10-45 min dependiendo de hardware)
2. **Reproducibilidad:** Mismas predicciones exactas en múltiples experimentos
3. **Experimentación:** Probar diferentes parámetros de warping sin re-predecir
4. **Portabilidad:** Archivo pequeño (~4 MB) fácil de compartir

---

## Dataset Normalizado (Warped)

### Imágenes Warpeadas

**Formato:** PNG

**Especificaciones:**
- **Resolución:** 224×224 píxeles
- **Profundidad de color:** 8 bits
- **Canales:** 1 canal (escala de grises)
- **Compresión:** PNG sin pérdida
- **Tamaño de archivo:** ~3-8 KB por imagen

**Características:**
- Geometría normalizada mediante warping afín por partes
- Pulmones centrados y alineados a forma canónica
- Fill rate medio: ~47% (con margin=1.05, use_full_coverage=false)
- Preprocesamiento: CLAHE aplicado (opcional según config)

### Estructura de Directorios

```
outputs/warped_lung_best/session_warping/
├── train/
│   ├── COVID/
│   │   ├── COVID-1.png
│   │   ├── COVID-2.png
│   │   └── ...                     (2,712 imágenes)
│   ├── Normal/
│   │   ├── Normal-1.png
│   │   └── ...                     (7,644 imágenes)
│   └── Viral_Pneumonia/
│       ├── Viral Pneumonia-1.png
│       └── ...                     (1,008 imágenes)
├── val/
│   ├── COVID/                       (452 imágenes)
│   ├── Normal/                      (1,274 imágenes)
│   └── Viral_Pneumonia/             (168 imágenes)
├── test/
│   ├── COVID/                       (452 imágenes)
│   ├── Normal/                      (1,274 imágenes)
│   └── Viral_Pneumonia/             (169 imágenes)
├── dataset_summary.json             # Estadísticas del dataset
├── metadata.json                    # Parámetros de warping usados
└── train/
    ├── landmarks.json               # Landmarks predichos por imagen (train)
    └── images.csv                   # Lista de imágenes procesadas (train)
```

**Splits:**
- **Train:** 11,364 imágenes (75%)
- **Val:** 1,894 imágenes (12.5%)
- **Test:** 1,895 imágenes (12.5%)

### Metadata del Dataset (metadata.json)

```json
{
  "input_dir": "data/dataset/COVID-19_Radiography_Dataset",
  "output_dir": "outputs/warped_lung_best/session_warping",
  "canonical_shape": "outputs/shape_analysis/canonical_shape_gpa.json",
  "triangles": "outputs/shape_analysis/canonical_delaunay_triangles.json",
  "margin_scale": 1.05,
  "use_full_coverage": false,
  "clahe": true,
  "clahe_clip_limit": 2.0,
  "clahe_tile_size": 4,
  "fill_rate_mean": 47.2,
  "fill_rate_std": 8.3,
  "fill_rate_min": 28.1,
  "fill_rate_max": 68.9,
  "image_size": 224,
  "total_images": 15153,
  "timestamp": "2026-01-13T10:45:23",
  "splits": {
    "train": 11364,
    "val": 1894,
    "test": 1895
  },
  "seed": 42
}
```

### Dataset Summary (dataset_summary.json)

```json
{
  "total_images": 15153,
  "class_distribution": {
    "COVID": 3616,
    "Normal": 10192,
    "Viral_Pneumonia": 1345
  },
  "splits": {
    "train": {
      "total": 11364,
      "COVID": 2712,
      "Normal": 7644,
      "Viral_Pneumonia": 1008
    },
    "val": {
      "total": 1894,
      "COVID": 452,
      "Normal": 1274,
      "Viral_Pneumonia": 168
    },
    "test": {
      "total": 1895,
      "COVID": 452,
      "Normal": 1274,
      "Viral_Pneumonia": 169
    }
  },
  "fill_rate_statistics": {
    "mean": 47.23,
    "std": 8.35,
    "median": 46.81,
    "min": 28.14,
    "max": 68.92
  }
}
```

### Landmarks por Split (train/landmarks.json)

```json
{
  "COVID-1.png": {
    "landmarks": [
      [112.3, 45.1],
      [112.1, 178.9],
      ...
    ],
    "predicted": true,
    "fill_rate": 48.23
  },
  "Normal-1.png": {
    "landmarks": [...],
    "predicted": true,
    "fill_rate": 47.56
  },
  ...
}
```

**Nota:** Landmarks en este JSON están en coordenadas de la imagen normalizada (224×224), NO de la original.

### Fill Rate

**Definición:** Porcentaje de píxeles no negros en la imagen warpeada.

**Rango:** [0, 100]%

**Interpretación:**
- **~47% (actual):** ROI basado en landmarks con margin=1.05 → contiene principalmente región pulmonar
- **~99% (obsoleto):** Imagen completa → incluye mucho fondo negro innecesario

**Cálculo:**
```python
def compute_fill_rate(image):
    """Compute percentage of non-black pixels."""
    total_pixels = image.shape[0] * image.shape[1]
    non_black_pixels = np.count_nonzero(image > 0)
    return (non_black_pixels / total_pixels) * 100.0
```

---

## Salidas de Clasificación

### Predicciones del Clasificador

**Formato:** CSV o JSON

#### Formato CSV (predictions.csv)

```csv
image_path,true_label,predicted_label,confidence,covid_prob,normal_prob,viral_pneumonia_prob
test/COVID/COVID-1.png,COVID,COVID,0.9823,0.9823,0.0154,0.0023
test/Normal/Normal-1.png,Normal,Normal,0.9901,0.0012,0.9901,0.0087
test/Viral_Pneumonia/Viral Pneumonia-1.png,Viral_Pneumonia,Viral_Pneumonia,0.8745,0.0234,0.1021,0.8745
```

**Columnas:**

| Nombre | Tipo | Descripción |
|--------|------|-------------|
| `image_path` | String | Ruta relativa a la imagen |
| `true_label` | String | Etiqueta real (ground truth) |
| `predicted_label` | String | Etiqueta predicha por el modelo |
| `confidence` | Float [0,1] | Probabilidad de la clase predicha |
| `covid_prob` | Float [0,1] | P(COVID) |
| `normal_prob` | Float [0,1] | P(Normal) |
| `viral_pneumonia_prob` | Float [0,1] | P(Viral_Pneumonia) |

**Nota:** `covid_prob + normal_prob + viral_pneumonia_prob = 1.0` (softmax)

#### Formato JSON (predictions.json)

```json
{
  "test/COVID/COVID-1.png": {
    "true_label": "COVID",
    "predicted_label": "COVID",
    "confidence": 0.9823,
    "probabilities": {
      "COVID": 0.9823,
      "Normal": 0.0154,
      "Viral_Pneumonia": 0.0023
    },
    "logits": [4.23, -2.15, -3.87]
  },
  ...
}
```

### Resultados de Evaluación

**Formato:** JSON

**Ubicación típica:** `outputs/classifier_*/test_results.json` o `evaluation_results.json`

```json
{
  "accuracy": 0.9805,
  "f1_macro": 0.9712,
  "f1_weighted": 0.9804,
  "precision_macro": 0.9721,
  "recall_macro": 0.9704,
  "per_class_metrics": {
    "COVID": {
      "precision": 0.9723,
      "recall": 0.9641,
      "f1_score": 0.9682,
      "support": 723
    },
    "Normal": {
      "precision": 0.9867,
      "recall": 0.9824,
      "f1_score": 0.9845,
      "support": 1276
    },
    "Viral_Pneumonia": {
      "precision": 0.9487,
      "recall": 0.9677,
      "f1_score": 0.9581,
      "support": 268
    }
  },
  "confusion_matrix": [
    [695, 21, 7],
    [18, 1253, 5],
    [7, 2, 259]
  ],
  "timestamp": "2026-01-13T12:34:56",
  "model_path": "outputs/classifier_warped_lung_best/best_classifier.pt",
  "data_dir": "outputs/warped_lung_best/session_warping",
  "split": "test"
}
```

### Matriz de Confusión

**Formato:** NumPy array (3×3) o nested list en JSON

**Interpretación:**
```
                Predicted
              COVID  Normal  Viral_Pneumonia
True  COVID     695      21                7
      Normal     18    1253                5
      Viral     7        2              259
```

**Lectura:**
- Diagonal: Predicciones correctas
- Fila i, Columna j: Clase i predicha como clase j
- Ejemplo: 21 casos de COVID predichos como Normal (falsos negativos)

---

## Formatos Auxiliares

### Forma Canónica (canonical_shape_gpa.json)

**Ubicación:** `outputs/shape_analysis/canonical_shape_gpa.json`

```json
{
  "landmarks": [
    [0.5, 0.1],
    [0.5, 0.9],
    [0.2, 0.25],
    [0.8, 0.25],
    ...
  ],
  "image_size": 224,
  "num_landmarks": 15,
  "scale": 1.0,
  "method": "gpa_iterative",
  "num_iterations": 4,
  "convergence_threshold": 1e-8,
  "timestamp": "2026-01-13T09:12:34"
}
```

**`landmarks`:** Array (15, 2) con coordenadas normalizadas [0, 1]

### Triangulación Delaunay (canonical_delaunay_triangles.json)

**Ubicación:** `outputs/shape_analysis/canonical_delaunay_triangles.json`

```json
{
  "triangles": [
    [0, 8, 11],
    [8, 11, 9],
    [9, 11, 1],
    ...
  ],
  "num_triangles": 24,
  "num_landmarks": 15,
  "method": "scipy.spatial.Delaunay"
}
```

**`triangles`:** Lista de 24 triángulos, cada uno con 3 índices de vértices (landmarks).

### Historial de Entrenamiento (training_history.json)

**Ubicación:** `outputs/*/training_history.json`

```json
{
  "phase1": {
    "epochs": 15,
    "train_loss": [0.8234, 0.6543, ..., 0.2341],
    "val_loss": [0.7123, 0.5987, ..., 0.2456],
    "val_error_px": [12.34, 9.87, ..., 5.23]
  },
  "phase2": {
    "epochs": 78,
    "train_loss": [0.2234, 0.1876, ..., 0.0987],
    "val_loss": [0.2345, 0.1987, ..., 0.1234],
    "val_error_px": [5.12, 4.67, ..., 4.15],
    "best_epoch": 65,
    "best_val_loss": 0.1123,
    "best_val_error_px": 4.10
  },
  "total_epochs": 93,
  "early_stopped": true,
  "early_stop_epoch": 78
}
```

---

## Conversiones Comunes

### Coordenadas Normalizadas ↔ Píxeles

**Normalizar (px → [0,1]):**
```python
landmarks_normalized = landmarks_pixels / image_size
```

**Denormalizar ([0,1] → px):**
```python
landmarks_pixels = landmarks_normalized * image_size
```

### Reshape de Landmarks

**Flat (30,) → Pares (15, 2):**
```python
landmarks_xy = landmarks_flat.reshape(15, 2)
```

**Pares (15, 2) → Flat (30,):**
```python
landmarks_flat = landmarks_xy.reshape(30)
```

### Índices de Clase ↔ Nombres

**Mapping:**
```python
class_to_idx = {'COVID': 0, 'Normal': 1, 'Viral_Pneumonia': 2}
idx_to_class = {0: 'COVID', 1: 'Normal', 2: 'Viral_Pneumonia'}

# Índice a nombre
predicted_class = idx_to_class[predicted_idx]

# Nombre a índice
class_idx = class_to_idx[class_name]
```

---

## Validación de Formatos

### Validar Imagen

```python
import cv2

def validate_image(image_path, expected_shape=(299, 299)):
    """Validate image format and dimensions."""
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return False, "Failed to load image"
        if image.shape != expected_shape:
            return False, f"Wrong shape: {image.shape}, expected {expected_shape}"
        if image.dtype != np.uint8:
            return False, f"Wrong dtype: {image.dtype}, expected uint8"
        return True, "Valid"
    except Exception as e:
        return False, str(e)

# Uso
valid, msg = validate_image('COVID-1.png')
print(f"Valid: {valid}, Message: {msg}")
```

### Validar CSV de Anotaciones

```python
import pandas as pd

def validate_annotations_csv(csv_path):
    """Validate annotations CSV format."""
    try:
        df = pd.read_csv(csv_path)

        # Verificar columnas
        expected_cols = ['image_name', 'category'] + \
                        [f'L{i}_{coord}' for i in range(1, 16) for coord in ['x', 'y']]
        if df.columns.tolist() != expected_cols:
            return False, "Incorrect columns"

        # Verificar número de filas
        if len(df) != 15153:
            return False, f"Expected 15153 rows, got {len(df)}"

        # Verificar coordenadas en rango
        coords = df.iloc[:, 2:].values
        if coords.min() < 0 or coords.max() > 299:
            return False, "Coordinates out of range [0, 299]"

        return True, "Valid"
    except Exception as e:
        return False, str(e)

# Uso
valid, msg = validate_annotations_csv('data/coordenadas/coordenadas_maestro.csv')
print(f"Valid: {valid}, Message: {msg}")
```

---

## Referencias

### Documentos Relacionados

- `01_GUIA_INICIO_RAPIDO.md` - Instalación y ejecución
- `03_GUIA_USO_CLI.md` - Comandos para generar/procesar datos
- `04_ARQUITECTURA_CODIGO.md` - Código que lee/escribe estos formatos
- `05_REPRODUCIBILIDAD_COMPLETA.md` - Generación de datos paso a paso

### Datasets

- **COVID-19 Radiography Database:** https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
- **Formato PNG:** https://www.w3.org/TR/PNG/
- **NumPy NPZ:** https://numpy.org/doc/stable/reference/generated/numpy.savez.html

---

**Última actualización:** 28 de enero de 2026

**Contacto:**
- Estudiante: Rafael Alejandro Cruz Ovando, BUAP
- Director: Dr. Leopoldo Altamirano Robles (robles@inaoep.mx), INAOE
