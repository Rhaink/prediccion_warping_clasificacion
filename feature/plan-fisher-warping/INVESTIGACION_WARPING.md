# Investigación: Warping y Landmarks - Resumen Ejecutivo

**Fecha:** 2026-01-06
**Investigador:** Claude (Agente de Exploración)

---

## 🎯 HALLAZGOS CLAVE

### 1. DATOS DE LANDMARKS

**Archivo principal de landmarks predichos:**
```
/home/donrobot/Projects/prediccion_warping_clasificacion/outputs/predictions/all_landmarks.npz
```

**Estructura:**
```python
import numpy as np

data = np.load('outputs/predictions/all_landmarks.npz')

# Disponible:
data['all_landmarks']        # (957, 15, 2) - Todos los landmarks en pixeles 224x224
data['all_image_names']      # (957,) - Nombres como "COVID-269", "Normal-1234"
data['all_categories']       # (957,) - "COVID", "Normal", "Viral_Pneumonia"

# Por split:
data['train_landmarks']      # (717, 15, 2)
data['val_landmarks']        # (144, 15, 2)
data['test_landmarks']       # (96, 15, 2)
```

**Archivo de forma canónica (GPA):**
```
/home/donrobot/Projects/prediccion_warping_clasificacion/outputs/shape_analysis/canonical_shape_gpa.json
```

**Contenido:**
```json
{
  "canonical_shape_pixels": [[112.3, 33.7], [111.8, 190.5], ...],  // 15 landmarks
  "image_size": 224,
  "n_landmarks": 15,
  "convergence": {
    "n_iterations": 5,
    "converged": true
  }
}
```

---

## 2. IMÁGENES PARA VISUALIZACIÓN

### Imágenes Originales

**Ruta base:**
```
/home/donrobot/Projects/prediccion_warping_clasificacion/data/dataset/
```

**Estructura:**
```
dataset/
├── COVID/              # 324 imágenes PNG
│   ├── COVID-269.png
│   ├── COVID-1234.png
│   └── ...
├── Normal/             # 475 imágenes PNG
│   ├── Normal-5678.png
│   └── ...
└── Viral_Pneumonia/    # 200 imágenes PNG
    ├── Viral Pneumonia-123.png
    └── ...
```

**Características:**
- Formato: PNG grayscale
- Resolución: 299x299 pixeles (original)
- Total: 999 imágenes

### Imágenes Warped

**Ruta base:**
```
/home/donrobot/Projects/prediccion_warping_clasificacion/outputs/full_warped_dataset/
```

**Estructura:**
```
full_warped_dataset/
├── train/
│   ├── COVID/
│   │   └── COVID-269_warped.png
│   ├── Normal/
│   └── Viral_Pneumonia/
├── val/
│   └── [misma estructura]
└── test/
    └── [misma estructura]
```

**Total warped:** 15,153 imágenes PNG (224x224 pixeles)

---

## 3. CÓDIGO DE WARPING

### Función Principal de Warping

**Archivo:**
```
/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/piecewise_affine_warp.py
/home/donrobot/Projects/prediccion_warping_clasificacion/src_v2/processing/warp.py
```

**Función clave:**
```python
def piecewise_affine_warp(
    image: np.ndarray,              # Imagen original (224, 224)
    source_landmarks: np.ndarray,   # (15, 2) landmarks predichos
    target_landmarks: np.ndarray,   # (15, 2) forma canónica
    triangles: Optional[np.ndarray] = None,
    output_size: int = 224,
    use_full_coverage: bool = True
) -> np.ndarray:
    """
    Warping piecewise affine usando triangulación Delaunay.

    Proceso:
    1. Extender landmarks con 8 puntos de borde (15 → 23)
    2. Calcular triangulación Delaunay (~18 triángulos)
    3. Warpear cada triángulo individualmente
    4. Retornar imagen warped
    """
```

### Función de Extensión de Puntos

```python
def add_boundary_points(
    landmarks: np.ndarray,  # (15, 2)
    image_size: int = 224
) -> np.ndarray:  # (23, 2)
    """
    Agrega 8 puntos de borde:
    - 4 esquinas: (0,0), (223,0), (0,223), (223,223)
    - 4 midpoints: (112,0), (0,112), (223,112), (112,223)

    Total: 15 landmarks + 8 borde = 23 puntos
    """
    corners = np.array([
        [0, 0], [image_size-1, 0],
        [0, image_size-1], [image_size-1, image_size-1]
    ])

    mid = image_size // 2
    midpoints = np.array([
        [mid, 0], [0, mid],
        [image_size-1, mid], [mid, image_size-1]
    ])

    return np.vstack([landmarks, corners, midpoints])
```

---

## 4. TRIANGULACIÓN DELAUNAY

**Biblioteca usada:** `scipy.spatial.Delaunay`

**Código:**
```python
from scipy.spatial import Delaunay

# Calcular triangulación
extended_landmarks = add_boundary_points(landmarks, 224)  # (23, 2)
tri = Delaunay(extended_landmarks)
triangles = tri.simplices  # (N, 3) donde N ≈ 18 triángulos

# Cada triángulo es un array [i, j, k] con índices a los 23 puntos
```

**Archivo pre-calculado:**
```
/home/donrobot/Projects/prediccion_warping_clasificacion/outputs/shape_analysis/canonical_delaunay_triangles.json
```

```json
{
  "triangles": [[i1,j1,k1], [i2,j2,k2], ..., [i18,j18,k18]],
  "n_triangles": 18,
  "n_landmarks": 15
}
```

---

## 5. LOS 15 LANDMARKS ANATÓMICOS

**Definición** (según documentación del proyecto):

### Eje Central Vertical (5 landmarks)
- **L1**: Punto superior (clavícula/tráquea superior)
- **L9**: Cuarto superior
- **L10**: Punto medio (centro del pecho)
- **L11**: Cuarto inferior
- **L2**: Punto inferior (diafragma)

### Contorno Pulmón Izquierdo (5 landmarks)
- **L12**: Borde superior izquierdo
- **L3**: Zona superior izquierda
- **L5**: Zona media izquierda (hilio)
- **L7**: Zona inferior izquierda
- **L14**: Esquina inferior izquierda (ángulo costofrénico)

### Contorno Pulmón Derecho (5 landmarks)
- **L13**: Borde superior derecho
- **L4**: Zona superior derecha
- **L6**: Zona media derecha (hilio)
- **L8**: Zona inferior derecha
- **L15**: Esquina inferior derecha (ángulo costofrénico)

**Pares simétricos:** (L3,L4), (L5,L6), (L7,L8), (L12,L13), (L14,L15)

---

## 6. MAPEO PARA VISUALIZACIONES

### Para TAREA 1.1: Landmarks Overlay

**Input necesario:**
```python
# 1. Cargar imagen original
image_path = "data/dataset/COVID/COVID-269.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 2. Cargar landmarks
data = np.load('outputs/predictions/all_landmarks.npz')
idx = np.where(data['all_image_names'] == 'COVID-269')[0][0]
landmarks = data['all_landmarks'][idx]  # (15, 2)

# 3. Resize imagen a 224x224 (landmarks ya están en esa escala)
image_224 = cv2.resize(image, (224, 224))

# 4. Dibujar landmarks
for i, (x, y) in enumerate(landmarks):
    cv2.circle(image_224, (int(x), int(y)), radius=3, color=255, thickness=-1)
    cv2.putText(image_224, f'L{i+1}', (int(x)+5, int(y)), ...)
```

### Para TAREA 1.2: Triangulación

**Input necesario:**
```python
# 1. Misma imagen + landmarks
# 2. Extender con boundary points
extended = add_boundary_points(landmarks, 224)  # (23, 2)

# 3. Calcular triangulación
from scipy.spatial import Delaunay
tri = Delaunay(extended)

# 4. Dibujar triángulos
for simplex in tri.simplices:
    pts = extended[simplex].astype(int)
    cv2.polylines(image_224, [pts], isClosed=True, color=255, thickness=1)
```

### Para TAREA 1.3: Warping Step-by-Step

**Input necesario:**
```python
# 1. Imagen original
original = cv2.imread('data/dataset/COVID/COVID-269.png', cv2.IMREAD_GRAYSCALE)
original_224 = cv2.resize(original, (224, 224))

# 2. Landmarks predichos
landmarks_pred = data['all_landmarks'][idx]  # (15, 2)

# 3. Forma canónica
import json
with open('outputs/shape_analysis/canonical_shape_gpa.json') as f:
    canonical = json.load(f)
canonical_landmarks = np.array(canonical['canonical_shape_pixels'])  # (15, 2)

# 4. Imagen warped
from src_v2.processing.warp import piecewise_affine_warp
warped = piecewise_affine_warp(original_224, landmarks_pred, canonical_landmarks)

# 5. Crear panel 2x2:
#    [original, original+landmarks]
#    [original+triangulation, warped]
```

---

## 7. EJEMPLOS DE IMÁGENES A USAR

### Casos Sugeridos para Visualización

**COVID (bien alineado):**
- `COVID-269` - Caso típico de COVID con landmarks claros

**Normal (bien alineado):**
- `Normal-1234` - Caso típico normal con landmarks claros

**Viral Pneumonia:**
- `Viral Pneumonia-123` - Para mostrar la tercera clase

**Para errores/casos difíciles:**
- Buscar en `outputs/predictions/test_predictions.npz` el campo `errors`
- Ordenar por error promedio y seleccionar:
  - Mejor caso (error mínimo)
  - Caso promedio (error mediano)
  - Peor caso (error máximo)

---

## 8. LIBRERÍAS NECESARIAS

```python
import numpy as np
import cv2
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import json
from pathlib import Path
```

**Todas ya instaladas en el proyecto.**

---

## 9. ESTRUCTURA DE DIRECTORIOS PARA SALIDA

**Crear:**
```bash
mkdir -p results/figures/warping_explained
mkdir -p results/figures/pca_explained
mkdir -p results/figures/statistical_analysis
mkdir -p results/figures/knn_explained
mkdir -p results/figures/fisher_explained
mkdir -p results/figures/interpretation
```

**Ruta base de figuras:**
```
/home/donrobot/Projects/prediccion_warping_clasificacion/feature/plan-fisher-warping/results/figures/
```

---

## 10. PRÓXIMOS PASOS

### Listo para FASE 1, TAREA 1.1

**Tenemos todo lo necesario:**
- ✅ Ubicación de landmarks predichos
- ✅ Ubicación de imágenes originales
- ✅ Código de referencia para cargar datos
- ✅ Definición de los 15 landmarks
- ✅ Librerías disponibles

**Siguiente acción:**
Crear `scripts/visualize_landmarks_overlay.py` que:
1. Carga imagen original (resize a 224x224)
2. Carga landmarks correspondientes de all_landmarks.npz
3. Dibuja círculos en coordenadas
4. Agrega labels L1-L15
5. Guarda figura profesional

---

## 📊 DATOS CLAVE ENCONTRADOS

| Archivo | Ubicación | Contenido |
|---------|-----------|-----------|
| **all_landmarks.npz** | `outputs/predictions/` | 957 landmarks predichos (15 puntos × 2 coords cada uno) |
| **canonical_shape_gpa.json** | `outputs/shape_analysis/` | Forma canónica obtenida por GPA |
| **Imágenes originales** | `data/dataset/` | 999 radiografías PNG (299x299) |
| **Imágenes warped** | `outputs/full_warped_dataset/` | 15,153 imágenes warped (224x224) |
| **Código warping** | `scripts/piecewise_affine_warp.py` | Implementación completa |
| **Código GPA** | `scripts/gpa_analysis.py` | Generalized Procrustes Analysis |

---

**Estado:** ✅ Investigación completa - Listo para implementación
