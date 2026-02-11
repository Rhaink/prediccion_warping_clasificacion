# Reporte de Estancia de Investigación

---

## Datos Generales

| Campo | Información |
|-------|-------------|
| **Proyecto** | Normalización y alineación automática de la forma de la región pulmonar integrada con selección de características discriminantes para detección automática de neumonía y COVID-19 |
| **Director** | Dr. Leopoldo Altamirano Robles |
| **Estudiante** | Rafael Alejandro Cruz Ovando |
| **Período** | 9 de octubre – 9 de noviembre de 2025 |
| **Lugar** | Laboratorio de Visión por Computadora, INAOE |
| **Institución de origen** | Benemérita Universidad Autónoma de Puebla (BUAP) |

---

## Resumen Ejecutivo

Este reporte documenta el trabajo realizado durante la estancia de investigación en el Laboratorio de Visión por Computadora del INAOE. Se desarrolló un sistema completo para la detección automática de landmarks pulmonares y normalización geométrica de radiografías de tórax para clasificación de COVID-19.

### Resultados Principales

| Métrica | Valor Obtenido |
|---------|----------------|
| Error de landmarks (ensemble) | **3.61 px** |
| Error de landmarks (mejor individual) | 4.04 px |
| Accuracy clasificación | **98.60%** (CV) |
| F1-macro validación cruzada | 98.00% ± 0.36% |

### Contribuciones

1. **Sistema de detección de landmarks**: Ensemble de 4 modelos ResNet-18 con Coordinate Attention y Test-Time Augmentation
2. **Normalización geométrica**: Pipeline de warping afín por partes usando triangulación de Delaunay
3. **Funciones de pérdida geométricas**: Wing Loss con restricciones de simetría y alineación central
4. **Infraestructura reproducible**: Sistema de configuraciones JSON y documentación exhaustiva

---

## 1. Introducción

### 1.1 Contexto del Proyecto

La pandemia de COVID-19 resaltó la necesidad de herramientas de diagnóstico rápido y automatizado. Las radiografías de tórax ofrecen una alternativa accesible a las pruebas PCR, pero su interpretación requiere experiencia especializada.

Este proyecto aborda dos desafíos fundamentales:

1. **Detección de landmarks anatómicos**: Localización automática de 15 puntos que definen el contorno pulmonar
2. **Normalización geométrica**: Transformación de imágenes a una forma canónica que elimina variabilidad anatómica

### 1.2 Importancia de los Landmarks Pulmonares

La región pulmonar en radiografías presenta alta variabilidad inter-paciente debido a:
- Diferencias anatómicas individuales
- Posicionamiento durante la adquisición
- Rotaciones y escalas variables

La detección precisa de landmarks permite normalizar estas variaciones, mejorando la robustez de sistemas de clasificación downstream.

### 1.3 Estructura de 15 Landmarks

El sistema utiliza 15 landmarks que definen el contorno pulmonar bilateral:

```
Eje central vertical:     L1 (apex) → L9 → L10 → L11 → L2 (base)
Contorno izquierdo:       L12, L3, L5, L7, L14
Contorno derecho:         L13, L4, L6, L8, L15
Pares simétricos:         (L3,L4), (L5,L6), (L7,L8), (L12,L13), (L14,L15)
```

*Referencia visual: Figura F4.3_landmarks_15.png*

---

## 2. Metodología Implementada

### 2.1 Preparación de Datos

#### Dataset Utilizado

| Característica | Valor |
|---------------|-------|
| Dataset base | COVID-19 Radiography Dataset |
| Total imágenes | 21,165 |
| Clases | COVID, Normal, Viral_Pneumonia |
| Anotaciones | 15 landmarks por imagen |
| División | 75% train / 15% val / 10% test |

#### Preprocesamiento

El pipeline de preprocesamiento implementa:

1. **CLAHE** (Contrast Limited Adaptive Histogram Equalization):
   - `clip_limit`: 2.0
   - `tile_size`: 4 (optimizado experimentalmente)

2. **Redimensionamiento**: 224×224 píxeles (tamaño de entrada ResNet)

3. **Normalización**: Media y desviación estándar de ImageNet

**Implementación**: `src_v2/data/transforms.py`

```python
# Configuración CLAHE optimizada
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
```

#### Aumentación de Datos

Durante entrenamiento se aplican:
- Rotación: ±10°
- Traslación: ±5%
- Escalado: 0.95–1.05
- Flip horizontal (con corrección de pares simétricos)

*Referencia visual: Figura F4.12_aumento_datos.png*

### 2.2 Arquitectura del Modelo

#### Backbone: ResNet-18

Se utiliza ResNet-18 pre-entrenado en ImageNet como extractor de características:

| Componente | Especificación |
|------------|----------------|
| Backbone | ResNet-18 (pretrained ImageNet) |
| Coordinate Attention | Sí (después de layer4) |
| Cabeza de regresión | Deep Head (768 dim) |
| Dropout | 0.3 |
| Salida | 30 valores (15 landmarks × 2 coordenadas) |

**Implementación**: `src_v2/models/resnet_landmark.py`

```python
class ResNet18Landmarks(nn.Module):
    def __init__(self, pretrained=True, num_landmarks=15,
                 use_coord_attention=True, use_deep_head=True,
                 hidden_dim=768, dropout=0.3):
        # Backbone ResNet-18
        self.backbone = models.resnet18(pretrained=pretrained)

        # Coordinate Attention (opcional)
        if use_coord_attention:
            self.coord_attention = CoordAttention(512, 512)

        # Cabeza de regresión profunda
        if use_deep_head:
            self.regression_head = nn.Sequential(
                nn.Linear(512, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_landmarks * 2)
            )
```

*Referencia visual: Figura F4.5_arquitectura_modelo.png*

#### Coordinate Attention

Mecanismo de atención que captura dependencias espaciales de largo alcance:

- Codifica información posicional en los canales
- Captura relaciones horizontales y verticales independientemente
- Mejora la localización precisa de landmarks

*Referencia visual: Figura coord_attention_v10_mechanism_real.png*

### 2.3 Funciones de Pérdida

Se implementó un sistema de pérdidas combinadas que incorpora restricciones geométricas:

**Implementación**: `src_v2/models/losses.py`

#### Wing Loss (Pérdida Principal)

Función robusta para regresión de landmarks que da mayor peso a errores pequeños:

```python
def wing_loss(pred, target, omega=10, epsilon=2):
    """
    Wing Loss: robusto a outliers, sensible a errores pequeños
    L(x) = omega * ln(1 + |x|/epsilon)  si |x| < omega
    L(x) = |x| - C                       si |x| >= omega
    """
    diff = torch.abs(pred - target)
    small = diff < omega
    loss = torch.where(
        small,
        omega * torch.log(1 + diff / epsilon),
        diff - (omega - omega * np.log(1 + omega / epsilon))
    )
    return loss.mean()
```

*Referencia visual: Figura F4.6_wing_loss_grafica.png*

#### Central Alignment Loss

Penaliza desviaciones de landmarks centrales (L9, L10, L11) respecto al eje vertical:

```python
def central_alignment_loss(pred, central_indices=[8, 9, 10]):
    """Penaliza desviación del eje central (x debería ser ~0.5)"""
    central_x = pred[:, central_indices, 0]
    target_x = 0.5  # Centro normalizado
    return F.mse_loss(central_x, torch.full_like(central_x, target_x))
```

#### Soft Symmetry Loss

Penaliza asimetrías excesivas entre pares de landmarks simétricos:

```python
SYMMETRIC_PAIRS = [(2, 3), (4, 5), (6, 7), (11, 12), (13, 14)]

def soft_symmetry_loss(pred, margin=6.0):
    """
    Penaliza solo cuando asimetría excede margen (6px)
    Permite asimetría natural en anatomía real
    """
    total_loss = 0
    for left_idx, right_idx in SYMMETRIC_PAIRS:
        left = pred[:, left_idx]
        right = pred[:, right_idx]
        # Distancia al eje central
        left_dist = torch.abs(left[:, 0] - 0.5)
        right_dist = torch.abs(right[:, 0] - 0.5)
        diff = torch.abs(left_dist - right_dist) * 224  # A píxeles
        # Solo penalizar si excede margen
        excess = F.relu(diff - margin)
        total_loss += excess.mean()
    return total_loss / len(SYMMETRIC_PAIRS)
```

#### Combined Loss

Combinación ponderada de todas las pérdidas:

```python
def combined_loss(pred, target,
                  wing_weight=1.0,
                  symmetry_weight=0.1,
                  alignment_weight=0.05):
    loss = wing_weight * wing_loss(pred, target)
    loss += symmetry_weight * soft_symmetry_loss(pred)
    loss += alignment_weight * central_alignment_loss(pred)
    return loss
```

### 2.4 Plan de Entrenamiento

Se implementó un esquema de entrenamiento en dos fases con tasas de aprendizaje diferenciadas:

**Implementación**: `src_v2/training/trainer.py`

#### Fase 1: Cabeza Congelada

| Parámetro | Valor |
|-----------|-------|
| Épocas | 15 |
| Learning rate | 1e-3 |
| Backbone | Congelado |
| Objetivo | Estabilizar predicción inicial |

```python
# Fase 1: Solo entrenar cabeza de regresión
for param in model.backbone.parameters():
    param.requires_grad = False

optimizer = optim.AdamW(model.regression_head.parameters(), lr=1e-3)
```

#### Fase 2: Fine-tuning Diferenciado

| Parámetro | Valor |
|-----------|-------|
| Épocas | 100 (con early stopping) |
| LR backbone | 2e-5 |
| LR cabeza | 2e-4 |
| Scheduler | ReduceLROnPlateau |
| Early stopping | 20 épocas sin mejora |

```python
# Fase 2: Fine-tuning con LR diferenciado
for param in model.backbone.parameters():
    param.requires_grad = True

optimizer = optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 2e-5},
    {'params': model.regression_head.parameters(), 'lr': 2e-4}
])

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)
```

*Referencia visual: Figura F4.16_estrategia_entrenamiento_ensemble_tta.png*

### 2.5 Estrategia de Ensemble

El sistema final combina 4 modelos entrenados con diferentes semillas:

| Modelo | Semilla | Error Individual |
|--------|---------|------------------|
| 1 | seed123 | 4.15 px |
| 2 | seed321 | 4.08 px |
| 3 | seed111 | 4.12 px |
| 4 | seed666 | 4.10 px |
| **Ensemble** | - | **3.61 px** |

#### Test-Time Augmentation (TTA)

Durante inferencia, cada imagen se procesa normal y con flip horizontal:

```python
def predict_with_tta(model, image):
    # Predicción original
    pred_orig = model(image)

    # Predicción con flip horizontal
    image_flip = torch.flip(image, dims=[3])
    pred_flip = model(image_flip)

    # Corregir coordenadas del flip
    pred_flip[:, :, 0] = 1.0 - pred_flip[:, :, 0]

    # Intercambiar pares simétricos
    for left, right in SYMMETRIC_PAIRS:
        pred_flip[:, [left, right]] = pred_flip[:, [right, left]]

    # Promediar
    return (pred_orig + pred_flip) / 2
```

**Implementación**: `src_v2/data/transforms.py`, `src_v2/evaluation/metrics.py`

### 2.6 Normalización Geométrica (Warping)

Una vez detectados los landmarks, se normaliza la geometría de las imágenes:

#### Generalized Procrustes Analysis (GPA)

Se calcula la forma canónica a partir de las anotaciones de entrenamiento:

```python
def gpa_iterative(shapes, max_iter=100, tol=1e-6):
    """
    Alinea iterativamente un conjunto de formas
    hasta converger a la forma media (canónica)
    """
    # Inicializar con primera forma
    mean_shape = shapes[0].copy()

    for _ in range(max_iter):
        # Alinear todas las formas a la media actual
        aligned = []
        for shape in shapes:
            aligned.append(procrustes_align(shape, mean_shape))

        # Calcular nueva media
        new_mean = np.mean(aligned, axis=0)

        # Verificar convergencia
        if np.linalg.norm(new_mean - mean_shape) < tol:
            break
        mean_shape = new_mean

    return mean_shape
```

**Implementación**: `src_v2/processing/gpa.py`

*Referencia visual: Figura F4.7_proceso_gpa.png*

#### Triangulación de Delaunay

Se genera una malla de triángulos sobre los landmarks:

```python
def compute_delaunay_triangulation(landmarks):
    """
    Calcula triangulación de Delaunay sobre los 15 landmarks
    más 4 esquinas de la imagen
    """
    # Agregar esquinas
    corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    all_points = np.vstack([landmarks, corners])

    # Triangulación
    tri = Delaunay(all_points)
    return tri.simplices
```

*Referencia visual: Figura F4.8_triangulacion_delaunay.png*

#### Warping Afín por Partes

Cada triángulo se transforma independientemente:

```python
def piecewise_affine_warp(image, src_landmarks, dst_landmarks, triangles):
    """
    Warping afín por partes usando triangulación
    """
    output = np.zeros_like(image)

    for tri_indices in triangles:
        # Obtener vértices del triángulo
        src_tri = src_landmarks[tri_indices]
        dst_tri = dst_landmarks[tri_indices]

        # Calcular transformación afín
        M = cv2.getAffineTransform(
            src_tri.astype(np.float32),
            dst_tri.astype(np.float32)
        )

        # Aplicar transformación
        warped = cv2.warpAffine(image, M, image.shape[:2])

        # Crear máscara del triángulo
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_tri.astype(np.int32), 255)

        # Combinar
        output = np.where(mask[..., None], warped, output)

    return output
```

**Implementación**: `src_v2/processing/warp.py`

*Referencia visual: Figura F4.9_original_vs_warped.png*

#### Parámetro de Margen

Se aplica una expansión del 5% desde el centroide de landmarks para incluir contexto:

```python
OPTIMAL_MARGIN_SCALE = 1.05  # Validado experimentalmente

def scale_landmarks_from_centroid(landmarks, scale=1.05):
    """Expande landmarks desde su centroide"""
    centroid = landmarks.mean(axis=0)
    return centroid + (landmarks - centroid) * scale
```

---

## 3. Resultados y Análisis

### 3.1 Métricas de Detección de Landmarks

#### Error Global

| Configuración | Error Medio | Std | Mediana |
|--------------|-------------|-----|---------|
| Mejor modelo individual (seed456) | 4.04 px | - | - |
| Ensemble 4 modelos | 3.71 px | 2.42 px | 3.17 px |
| **Ensemble + TTA (mejor)** | **3.61 px** | 2.48 px | 3.07 px |

#### Error por Categoría

| Categoría | Error (px) |
|-----------|------------|
| Normal | 3.22 |
| COVID | 3.93 |
| Viral_Pneumonia | 4.11 |

#### Error por Landmark

| Landmark | Error (px) | Ubicación |
|----------|------------|-----------|
| L10 | **2.44** | Centro (mejor) |
| L9 | 2.76 | Centro |
| L5 | 2.88 | Lateral izq. |
| L12 | 5.43 | Apex izq. (peor) |
| L13 | 5.35 | Apex der. |
| L14 | 4.39 | Base izq. |

**Observación**: Los landmarks centrales (L9, L10, L11) tienen menor error debido a la menor variabilidad anatómica. Los landmarks en ápices (L12, L13) presentan mayor error por la dificultad de definir límites precisos.

*Referencia visual: Figura F5.1_error_por_landmark.png*

### 3.2 Clasificación

#### Validación Cruzada (5-fold)

| Métrica | Valor | Std |
|---------|-------|-----|
| Accuracy | 98.60% | ±0.26% |
| F1-macro | 98.00% | ±0.36% |
| F1-weighted | 98.60% | ±0.25% |

**Configuración**:
- Dataset: `outputs/warped_lung_best/session_warping`
- Backbone: ResNet-18
- Folds: 5
- Seed: 42

#### Ensemble de Clasificadores con TTA

| Métrica | Sin TTA | Con TTA |
|---------|---------|---------|
| Accuracy | 98.10% | **98.26%** |
| F1-macro | 97.03% | 97.12% |
| F1-weighted | 98.09% | 98.25% |

**Impacto de TTA**:
- Casos mejorados: 6
- Casos empeorados: 3
- Mejora neta: +3 casos

*Referencia visual: Figura F5.7_matriz_confusion_sahs.png*

### 3.3 Comparación con Plan Original

#### Objetivos Cumplidos

| Objetivo del Plan | Estado | Evidencia |
|-------------------|--------|-----------|
| Recopilar y normalizar dataset | ✅ Completado | `data/coordenadas/coordenadas_maestro.csv` |
| Arquitectura ResNet-18 con pérdidas geométricas | ✅ Completado | `src_v2/models/` |
| Plan de entrenamiento en fases | ✅ Completado | `src_v2/training/trainer.py` |
| Protocolo de evaluación | ✅ Completado | `src_v2/evaluation/` |
| Documentación | ✅ Completado | `docs/`, `CLAUDE.md` |

#### Extensiones Realizadas (Más allá del Plan)

1. **Ensemble de 4 modelos**: No contemplado originalmente, redujo error de 4.04 a 3.61 px
2. **Test-Time Augmentation**: Mejora adicional con flip horizontal
3. **Sistema de configuración JSON**: Reproducibilidad mejorada
4. **Validación cruzada 5-fold**: Evaluación más robusta
5. **Pipeline completo de clasificación**: Incluyendo warping y clasificador

---

## 4. Entregables Producidos

### 4.1 Código Fuente

```
src_v2/
├── models/
│   ├── resnet_landmark.py      # Arquitectura ResNet-18 + CoordAttention
│   ├── losses.py               # Wing Loss, Symmetry, Alignment
│   └── classifier.py           # Clasificador COVID-19
├── training/
│   ├── trainer.py              # Entrenamiento 2 fases
│   └── callbacks.py            # Early stopping, checkpointing
├── data/
│   ├── dataset.py              # LandmarkDataset
│   └── transforms.py           # CLAHE, TTA, aumentación
├── processing/
│   ├── gpa.py                  # Generalized Procrustes Analysis
│   └── warp.py                 # Warping afín por partes
├── evaluation/
│   ├── metrics.py              # Error de landmarks
│   └── ensemble.py             # Evaluación de ensemble
└── cli.py                      # Interfaz de línea de comandos
```

### 4.2 Modelos Entrenados

4 modelos del ensemble disponibles en `checkpoints/`:

| Archivo | Descripción |
|---------|-------------|
| `session10/ensemble/seed123/final_model.pt` | Modelo ensemble 1 |
| `session13/seed321/final_model.pt` | Modelo ensemble 2 |
| `repro_split111/session14/seed111/final_model.pt` | Modelo ensemble 3 |
| `repro_split666/session16/seed666/final_model.pt` | Modelo ensemble 4 |

### 4.3 Configuraciones Reproducibles

```
configs/
├── ensemble_best.json          # Configuración del mejor ensemble
├── landmarks_train_base.json   # Hiperparámetros de entrenamiento
├── warping_best.json           # Parámetros de warping óptimos
└── classifier_warped_base.json # Configuración del clasificador
```

### 4.4 Documentación

| Documento | Descripción |
|-----------|-------------|
| `CLAUDE.md` | Guía principal del proyecto |
| `GROUND_TRUTH.json` | Valores validados experimentalmente |
| `docs/REPRO_FULL_PIPELINE.md` | Guía de reproducción completa |
| `docs/REPRO_ENSEMBLE_3_71.md` | Detalles del ensemble |
| `docs/CONFIGS.md` | Sistema de configuraciones |

---

## 5. Cronograma Ejecutado

### Mapeo Actividades vs Plan Original

| Semana | Plan Original | Actividades Ejecutadas |
|--------|--------------|------------------------|
| **1** (9-15 Oct) | Revisión estado del arte, análisis dataset | ✅ Análisis de 21,165 imágenes, definición de 15 landmarks, configuración de entorno |
| **2** (16-22 Oct) | Arquitectura base, Fase 1 | ✅ Implementación ResNet-18 + CoordAttention, entrenamiento Fase 1 |
| **3** (23-29 Oct) | Fine-tuning, Fases 2-3 | ✅ Fine-tuning diferenciado, pérdidas geométricas, sweep de semillas |
| **4** (30 Oct-9 Nov) | Experimentos finales, documentación | ✅ Ensemble 4 modelos, TTA, pipeline clasificación, documentación |

### Sesiones de Desarrollo

El desarrollo se documentó en sesiones incrementales:

- **Sesiones 1-10**: Arquitectura base y entrenamiento inicial
- **Sesiones 11-15**: Optimización de hiperparámetros y ensemble
- **Sesiones 16-25**: Pipeline de warping y clasificación
- **Sesiones 26-55**: Validación, robustez y documentación

---

## 6. Conclusiones

### 6.1 Objetivos Cumplidos

1. **Detección de landmarks**: Error de 3.61 px con ensemble, superando expectativas
2. **Arquitectura robusta**: ResNet-18 + Coordinate Attention + Deep Head
3. **Pérdidas geométricas**: Wing Loss + Soft Symmetry + Central Alignment
4. **Entrenamiento en fases**: Backbone congelado → Fine-tuning diferenciado
5. **Clasificación COVID-19**: 98.60% accuracy con validación cruzada

### 6.2 Contribuciones Técnicas

- Sistema end-to-end desde radiografía hasta clasificación
- Normalización geométrica que reduce variabilidad anatómica
- Configuraciones JSON para reproducibilidad total
- Documentación exhaustiva del proceso experimental

### 6.3 Limitaciones Identificadas

1. **Domain shift**: Modelos no generalizan a datos externos sin fine-tuning
2. **Landmarks extremos**: Mayor error en ápices pulmonares (L12, L13)
3. **Dependencia de anotaciones**: Calidad limitada por anotaciones manuales

### 6.4 Trabajo Futuro

1. Explorar arquitecturas más recientes (ViT, ConvNeXt)
2. Domain adaptation para generalización a otros hospitales
3. Detección de landmarks en 3D (CT)
4. Incorporación de incertidumbre en predicciones

---

## 7. Referencias y Anexos

### 7.1 Comandos CLI Principales

```bash
# Calcular forma canónica
python -m src_v2 compute-canonical data/coordenadas/coordenadas_maestro.csv \
  --output-dir outputs/shape_analysis --visualize

# Generar predicciones de landmarks
python scripts/predict_landmarks_dataset.py \
  --input-dir data/dataset/COVID-19_Radiography_Dataset \
  --output outputs/landmark_predictions/predictions.npz \
  --ensemble-config configs/ensemble_best.json --tta --clahe

# Generar dataset normalizado
python -m src_v2 generate-dataset --config configs/warping_best.json

# Entrenar clasificador
python -m src_v2 train-classifier --config configs/classifier_warped_base.json

# Evaluar ensemble de landmarks
python scripts/evaluate_ensemble_from_config.py --config configs/ensemble_best.json
```

### 7.2 Estructura del Repositorio

```
prediccion_warping_clasificacion/
├── src_v2/                 # Código fuente principal
├── scripts/                # Scripts de utilidad
├── configs/                # Configuraciones JSON
├── checkpoints/            # Modelos entrenados
├── data/                   # Datos (no en repo)
├── outputs/                # Resultados (no en repo)
├── docs/                   # Documentación
├── tests/                  # Tests unitarios
├── CLAUDE.md               # Guía del proyecto
└── GROUND_TRUTH.json       # Valores validados
```

### 7.3 Figuras de Referencia

| Figura | Archivo | Descripción |
|--------|---------|-------------|
| Landmarks 15 | `F4.3_landmarks_15.png` | Estructura de landmarks pulmonares |
| Arquitectura | `F4.5_arquitectura_modelo.png` | ResNet-18 + cabeza de regresión |
| Wing Loss | `F4.6_wing_loss_grafica.png` | Función de pérdida |
| GPA | `F4.7_proceso_gpa.png` | Proceso de alineación |
| Triangulación | `F4.8_triangulacion_delaunay.png` | Malla de triángulos |
| Warping | `F4.9_original_vs_warped.png` | Comparación original/normalizado |
| Error landmarks | `F5.1_error_por_landmark.png` | Error por cada landmark |
| Matriz confusión | `F5.7_matriz_confusion_sahs.png` | Resultados clasificación |

---

**Firma:**

_________________________
Rafael Alejandro Cruz Ovando
Estudiante de Maestría
Benemérita Universidad Autónoma de Puebla

_________________________
Dr. Leopoldo Altamirano Robles
Director del Proyecto
Instituto Nacional de Astrofísica, Óptica y Electrónica

**Fecha de elaboración:** Enero 2026
