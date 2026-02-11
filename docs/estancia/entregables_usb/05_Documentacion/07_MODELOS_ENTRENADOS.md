# Modelos Entrenados

**Documentación del Ensemble de Modelos de Detección de Landmarks**

Este documento describe los 4 modelos del ensemble que alcanzaron **3.61 px de error medio** en la detección de landmarks pulmonares.

---

## Tabla de Contenidos

1. [Visión General del Ensemble](#visión-general-del-ensemble)
2. [Modelos Individuales](#modelos-individuales)
3. [Arquitectura](#arquitectura)
4. [Proceso de Entrenamiento](#proceso-de-entrenamiento)
5. [Cómo Cargar y Usar los Modelos](#cómo-cargar-y-usar-los-modelos)
6. [Cuándo Usar Ensemble vs. Individual](#cuándo-usar-ensemble-vs-individual)
7. [Troubleshooting](#troubleshooting)

---

## Visión General del Ensemble

### Métricas del Ensemble

**Configuración:** 4 modelos + Test-Time Augmentation (TTA)

| Métrica | Valor | Comparación |
|---------|-------|-------------|
| **Error medio** | **3.61 px** | Best individual: 4.10 px |
| **Desviación estándar** | 2.48 px | Variabilidad moderada |
| **Error mediano** | 3.07 px | 50% de predicciones < 3.07 px |
| **Error máximo** | 27.3 px | Caso outlier extremo |
| **Mejora vs. best individual** | 0.49 px (12.0%) | Ensemble > individual |

**Fuente:** `GROUND_TRUTH.json` v2.1.0, validado el 2026-01-13

### Error por Categoría

| Categoría | Error Medio | Observación |
|-----------|-------------|-------------|
| **Normal** | 3.22 px | Mejor performance (anatomía más regular) |
| **COVID** | 3.93 px | Desafío moderado (opacidades en vidrio esmerilado) |
| **Viral Pneumonia** | 4.11 px | Más difícil (consolidaciones, derrame pleural) |

**Interpretación:**
- Normal: Pulmones sin patología → contornos más claros
- COVID: Opacidades periféricas → dificulta detección de landmarks exteriores
- Viral Pneumonia: Consolidaciones densas → landmarks menos visibles

### Composición del Ensemble

El ensemble está formado por **4 modelos ResNet-18** entrenados con diferentes semillas aleatorias:

| Modelo | Semilla | Error Individual (con TTA) | Ubicación |
|--------|---------|---------------------------|-----------|
| **Model 1** | 123 | 4.20 px | `checkpoints/session10/ensemble/seed123/final_model.pt` |
| **Model 2** | 321 | 4.15 px | `checkpoints/session13/seed321/final_model.pt` |
| **Model 3** | 111 | 4.18 px | `checkpoints/repro_split111/session14/seed111/final_model.pt` |
| **Model 4** | 666 | **4.10 px** (best) | `checkpoints/repro_split666/session16/seed666/final_model.pt` |

**Estrategia de ensemble:** Promedio simple de predicciones de los 4 modelos.

**¿Por qué funciona el ensemble?**
- Cada modelo aprende features ligeramente diferentes (debido a diferentes inicializaciones)
- Promedio reduce varianza y errores aleatorios
- Diferentes modelos capturan diferentes aspectos de la anatomía pulmonar

---

## Modelos Individuales

### Model 1: seed123

**Checkpoints:**
- `checkpoints/session10/ensemble/seed123/final_model.pt` (46 MB)

**Métricas (con TTA):**
- Error medio: 4.20 px
- Std: 2.55 px
- Mediana: 3.58 px

**Características:**
- Primer modelo entrenado del ensemble
- Performance sólida pero no la mejor
- Contribuye diversidad al ensemble

**Fecha de entrenamiento:** Diciembre 2025 (Session 10)

---

### Model 2: seed321

**Checkpoints:**
- `checkpoints/session13/seed321/final_model.pt` (46 MB)

**Métricas (con TTA):**
- Error medio: 4.15 px
- Std: 2.51 px
- Mediana: 3.52 px

**Características:**
- Segunda mejor performance individual
- Excelente generalización en Viral Pneumonia
- Complementa bien a seed666

**Fecha de entrenamiento:** Diciembre 2025 (Session 13)

---

### Model 3: seed111

**Checkpoints:**
- `checkpoints/repro_split111/session14/seed111/final_model.pt` (46 MB)

**Métricas (con TTA):**
- Error medio: 4.18 px
- Std: 2.53 px
- Mediana: 3.55 px

**Características:**
- Entrenado con split reproducible (split_seed=111)
- Performance intermedia
- Agrega robustez al ensemble

**Fecha de entrenamiento:** Enero 2026 (Session 14, reproducibility sweep)

---

### Model 4: seed666 ⭐ (Best Individual)

**Checkpoints:**
- `checkpoints/repro_split666/session16/seed666/final_model.pt` (46 MB)

**Métricas (con TTA):**
- Error medio: **4.10 px** (mejor individual)
- Std: 2.47 px
- Mediana: 3.46 px

**Características:**
- **Mejor modelo individual del proyecto**
- Excelente performance en todas las categorías
- Entrenado con split reproducible (split_seed=666)
- Líder del ensemble

**Fecha de entrenamiento:** Enero 2026 (Session 16, reproducibility sweep)

**Nota histórica:** Este modelo reemplazó a seed456 (4.04 px) como mejor individual. seed456 fue parte del ensemble histórico (3.71 px) pero fue reemplazado por seed666 para alcanzar 3.61 px.

---

## Arquitectura

### Visión General

```
Input Image (224×224×3)
    ↓
┌─────────────────────────────────────┐
│         ResNet-18 Backbone          │
│         (ImageNet pre-trained)      │
│                                     │
│  Conv1 (7×7, 64)                    │
│  MaxPool                            │
│  Layer1: 2× BasicBlock (64)        │
│  Layer2: 2× BasicBlock (128)       │
│  Layer3: 2× BasicBlock (256)       │
│  Layer4: 2× BasicBlock (512)       │
└─────────────────────────────────────┘
    ↓ [512 features, 7×7 spatial]
┌─────────────────────────────────────┐
│    Coordinate Attention Module      │
│    (Spatial-aware attention)        │
└─────────────────────────────────────┘
    ↓ [512 features, attended]
    AdaptiveAvgPool2d → [512]
    ↓
┌─────────────────────────────────────┐
│         Deep Regression Head        │
│                                     │
│  FC1: 512 → 768                     │
│  ReLU + Dropout(0.3)                │
│  FC2: 768 → 512                     │
│  ReLU + Dropout(0.3)                │
│  FC3: 512 → 30                      │
└─────────────────────────────────────┘
    ↓
Output: 30 valores [L1_x, L1_y, ..., L15_x, L15_y] en [0, 1]
```

### Componentes Clave

#### 1. ResNet-18 Backbone

**Propósito:** Extracción de features robustas

**Características:**
- Pre-entrenado en ImageNet (1.2M imágenes, 1000 clases)
- 11.7M parámetros en el backbone
- Proven architecture para visión por computadora

**¿Por qué ResNet-18 y no más profundo?**
- ResNet-18: Balance ideal entre capacidad y velocidad
- ResNet-34/50: Más parámetros, no mejora significativamente para este task
- Dataset de 15k imágenes: ResNet-18 suficiente, evita overfitting

#### 2. Coordinate Attention Module

**Propósito:** Mejorar localización espacial de landmarks

**Mecanismo:**
1. Pooling horizontal y vertical (en lugar de global)
2. Preserva información espacial en dos dimensiones
3. Genera mapas de atención específicos para coordenadas (x, y)

**Impacto medido:**
- Con Coordinate Attention: ~4.1 px
- Sin Coordinate Attention: ~4.3 px
- **Mejora:** ~0.2 px

**Paper:** "Coordinate Attention for Efficient Mobile Network Design" (Hou et al., 2021)

#### 3. Deep Regression Head

**Propósito:** Mapear features (512-D) a coordenadas de landmarks (30-D)

**Arquitectura:**
- **Simple Head (baseline):** FC 512 → 30 (1 capa)
- **Deep Head (usado):** FC 512 → 768 → 512 → 30 (3 capas)

**Ventajas del Deep Head:**
- Mayor capacidad de regresión no lineal
- Dropout entre capas → regularización
- **Mejora medida:** ~0.15 px vs. Simple Head

**Dropout:** 0.3 entre cada capa FC (previene overfitting)

### Tamaño del Modelo

| Componente | Parámetros | Porcentaje |
|------------|------------|------------|
| ResNet-18 Backbone | 11.7M | 94.7% |
| Coordinate Attention | 0.3M | 2.4% |
| Deep Regression Head | 0.36M | 2.9% |
| **Total** | **12.36M** | 100% |

**Tamaño en disco:** ~46 MB por modelo (fp32)

---

## Proceso de Entrenamiento

### Two-Phase Training

El entrenamiento se realiza en **2 fases** para aprovechar los pesos pre-entrenados de ImageNet:

#### Fase 1: Backbone Congelado (Warm-up)

**Duración:** 15 epochs

**Configuración:**
- Backbone: **Congelado** (pesos de ImageNet fijos)
- Head: **Entrenable** (inicializado aleatoriamente)
- Learning rate: 1e-3 (alto)
- Optimizer: Adam
- Loss: Wing Loss

**Objetivo:**
- Entrenar la cabeza de regresión desde cero
- Estabilizar predicciones iniciales
- Evitar destruir features pre-entrenadas con gradientes grandes

**Resultado tras Fase 1:**
- Error ~8-10 px (predicciones razonables pero poco precisas)
- Head estabilizada y lista para fine-tuning

#### Fase 2: Fine-tuning Completo

**Duración:** 100 epochs (con early stopping típicamente ~70-80 epochs)

**Configuración:**
- Backbone: **Descongelado** (entrenable)
- Head: **Entrenable**
- Learning rates diferenciados:
  - Backbone: 2e-5 (muy bajo, preservar features pre-entrenadas)
  - Head: 2e-4 (10× mayor que backbone)
- Optimizer: Adam con weight decay
- Loss: Wing Loss
- Early stopping: patience=15

**Objetivo:**
- Adaptar features de ImageNet a radiografías de tórax
- Fine-tuning suave del backbone (no destruir conocimiento pre-entrenado)
- Permitir que el head se especialice más

**Resultado tras Fase 2:**
- Error ~4.1-4.2 px (modelos individuales)
- Convergencia estable

### Wing Loss

**Función de pérdida usada:**

```
Wing Loss(x) = {
    w * ln(1 + |x|/ε)           si |x| < w
    |x| - C                     si |x| ≥ w
}

Parámetros: w=10.0, ε=2.0
```

**Características:**
- **Suave** cerca de cero (|x| < w): Enfoca en predicciones cercanas, evita penalizar demasiado errores pequeños
- **Lineal** lejos de cero (|x| ≥ w): No amplifica outliers como MSE

**Ventaja vs. MSE/L1:**
- MSE: Penaliza mucho outliers (x²) → sensible a casos difíciles
- L1: Trata igual errores pequeños y grandes → menos enfoque en precisión fina
- **Wing Loss:** Balance ideal para regresión de landmarks (~0.3 px mejor que MSE)

### Aumentación de Datos

Durante entrenamiento (solo train set):

1. **Flip horizontal:** 50% probabilidad
   - Corrige índices de landmarks simétricos (L3↔L4, L5↔L6, etc.)
2. **Rotación aleatoria:** ±10 grados
   - Simula variaciones en posicionamiento del paciente
3. **CLAHE:** Contrast Limited Adaptive Histogram Equalization
   - Clip limit: 2.0
   - Tile size: 4×4
4. **Normalización:** Media y std de ImageNet

**NO se usa durante validación/test** (excepto TTA en test).

### Tiempo de Entrenamiento

Por modelo individual:

| Fase | Epochs | Tiempo (GPU) | Tiempo (CPU) |
|------|--------|--------------|--------------|
| Fase 1 | 15 | ~30-45 min | ~4-6 horas |
| Fase 2 | ~70-80 (con early stop) | ~4-6 horas | ~36-48 horas |
| **Total** | | **~5-7 horas** | **~2 días** |

**Hardware usado (GPU):**
- NVIDIA RTX 3090 (24 GB VRAM)
- Batch size: 16
- Épocas efectivas: ~85-95 (early stopping)

**Para entrenar los 4 modelos del ensemble:**
- GPU: ~24-28 horas totales (pueden entrenar en paralelo si hay múltiples GPUs)
- CPU: ~8 días totales

---

## Cómo Cargar y Usar los Modelos

### Cargar un Modelo Individual

```python
import torch

# Cargar modelo
model_path = 'checkpoints/repro_split666/session16/seed666/final_model.pt'
model = torch.load(model_path, map_location='cpu')  # 'cuda' si tienes GPU
model.eval()

print(f"Modelo cargado: {model.__class__.__name__}")
print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")
```

**Salida esperada:**
```
Modelo cargado: ResNet18Landmarks
Parámetros: 12,361,470
```

### Inferencia con un Modelo

```python
import torch
import cv2
from torchvision import transforms
from src_v2.data.transforms import CLAHETransform, NormalizeToModel

# Preparar transformaciones
transform = transforms.Compose([
    CLAHETransform(clip_limit=2.0, tile_grid_size=(4, 4)),
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    NormalizeToModel()  # ImageNet normalization
])

# Cargar y preparar imagen
image = cv2.imread('path/to/xray.png', cv2.IMREAD_GRAYSCALE)
image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
image_tensor = transform(image_rgb).unsqueeze(0)  # (1, 3, 224, 224)

# Inferencia
with torch.no_grad():
    landmarks_normalized = model(image_tensor)  # (1, 30) en [0, 1]

# Denormalizar a píxeles
landmarks_pixels = landmarks_normalized.squeeze(0).cpu().numpy() * 224
landmarks_xy = landmarks_pixels.reshape(15, 2)  # (15, 2)

print(f"Landmarks predichos (px):")
for i, (x, y) in enumerate(landmarks_xy, start=1):
    print(f"  L{i}: ({x:.1f}, {y:.1f})")
```

### Usar el Ensemble Completo

```python
import numpy as np

# Cargar los 4 modelos
models = [
    torch.load('checkpoints/session10/ensemble/seed123/final_model.pt'),
    torch.load('checkpoints/session13/seed321/final_model.pt'),
    torch.load('checkpoints/repro_split111/session14/seed111/final_model.pt'),
    torch.load('checkpoints/repro_split666/session16/seed666/final_model.pt'),
]

for model in models:
    model.eval()

# Inferencia con ensemble
predictions = []
with torch.no_grad():
    for model in models:
        pred = model(image_tensor)
        predictions.append(pred.cpu().numpy())

# Promedio de predicciones
ensemble_prediction = np.mean(predictions, axis=0)  # (1, 30)
ensemble_landmarks = ensemble_prediction.squeeze(0) * 224
ensemble_xy = ensemble_landmarks.reshape(15, 2)

print(f"Ensemble predictions (px):")
for i, (x, y) in enumerate(ensemble_xy, start=1):
    print(f"  L{i}: ({x:.1f}, {y:.1f})")
```

### Usar TTA (Test-Time Augmentation)

```python
from src_v2.data.transforms import TTATransform
from src_v2.constants import SYMMETRIC_PAIRS

def predict_with_tta(model, image_tensor):
    """Predict with TTA (horizontal flip)."""
    # Predicción original
    with torch.no_grad():
        pred_original = model(image_tensor).cpu().numpy()

    # Predicción con flip
    image_flipped = torch.flip(image_tensor, dims=[3])  # Flip horizontal
    with torch.no_grad():
        pred_flipped = model(image_flipped).cpu().numpy()

    # Flip predicciones de vuelta y corregir landmarks simétricos
    pred_flipped_corrected = pred_flipped.copy()
    pred_flipped_corrected = pred_flipped_corrected.reshape(-1, 15, 2)
    pred_flipped_corrected[:, :, 0] = 1.0 - pred_flipped_corrected[:, :, 0]  # Flip x

    # Swap landmarks simétricos
    for i, j in SYMMETRIC_PAIRS:
        pred_flipped_corrected[:, [i, j]] = pred_flipped_corrected[:, [j, i]]

    pred_flipped_corrected = pred_flipped_corrected.reshape(-1, 30)

    # Promedio
    pred_tta = (pred_original + pred_flipped_corrected) / 2.0
    return pred_tta

# Uso
pred_tta = predict_with_tta(model, image_tensor)
```

### Usando el Script de Evaluación del Ensemble

**Forma más simple (recomendada):**

```bash
python scripts/evaluate_ensemble_from_config.py \
  --config configs/ensemble_best.json
```

Esto automáticamente:
- Carga los 4 modelos
- Aplica TTA
- Aplica CLAHE
- Calcula métricas completas

---

## Cuándo Usar Ensemble vs. Individual

### Usar Ensemble Cuando:

✅ **Máxima precisión es crítica**
- Error medio: 3.61 px (12% mejor que mejor individual)
- Aplicaciones clínicas o investigación de alta precisión

✅ **Tienes recursos computacionales suficientes**
- 4× más lento que modelo individual (~20 segundos vs. 5 segundos para 1000 imágenes en GPU)
- 4× más memoria RAM (~180 MB vs. 45 MB)

✅ **Robustez ante outliers es importante**
- Ensemble reduce varianza → menos predicciones extremas

### Usar Modelo Individual Cuando:

✅ **Velocidad es prioritaria**
- Inferencia en tiempo real
- Aplicaciones móviles o edge devices
- Procesamiento de datasets masivos (>100k imágenes)

✅ **Recursos limitados**
- Solo 1 GPU o CPU
- Memoria RAM limitada (<8 GB)

✅ **Diferencia de 0.5 px no es crítica**
- Análisis exploratorio
- Aplicaciones no clínicas

**Recomendación:** Use **seed666** (best individual, 4.10 px) si opta por modelo individual.

### Comparación de Performance

| Métrica | seed666 (individual) | Ensemble (4 modelos) | Diferencia |
|---------|---------------------|----------------------|------------|
| Error medio | 4.10 px | 3.61 px | -0.49 px (12.0%) |
| Std | 2.47 px | 2.48 px | +0.01 px |
| Mediana | 3.46 px | 3.07 px | -0.39 px |
| Tiempo (GPU, 1000 imgs) | ~5 seg | ~20 seg | 4× |
| Memoria | 46 MB | 184 MB | 4× |

---

## Troubleshooting

### Problema: "RuntimeError: CUDA out of memory"

**Causa:** Intentar cargar 4 modelos en GPU con VRAM insuficiente.

**Solución 1 (recomendada):** Usar CPU para ensemble
```python
models = [
    torch.load(path, map_location='cpu')
    for path in model_paths
]
```

**Solución 2:** Cargar modelos uno a uno
```python
predictions = []
for model_path in model_paths:
    model = torch.load(model_path, map_location='cuda')
    model.eval()
    with torch.no_grad():
        pred = model(image_tensor)
        predictions.append(pred.cpu())
    del model  # Liberar memoria
    torch.cuda.empty_cache()

ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
```

### Problema: Predicciones fuera de rango [0, 1]

**Causa:** Modelo no está en modo evaluación o falta normalización.

**Solución:**
```python
model.eval()  # Crítico: pone batch norm y dropout en modo inference
with torch.no_grad():
    pred = model(image_tensor)
```

### Problema: Error en landmarks simétricos con TTA

**Causa:** No se corrigieron índices simétricos tras flip.

**Solución:** Usar `SYMMETRIC_PAIRS` de `src_v2/constants.py`:
```python
from src_v2.constants import SYMMETRIC_PAIRS

# Tras flip, intercambiar landmarks simétricos
for i, j in SYMMETRIC_PAIRS:
    landmarks[:, [i, j]] = landmarks[:, [j, i]]
```

### Problema: Modelos no se cargan

**Síntoma:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'checkpoints/...'
```

**Solución:** Verificar que los checkpoints estén en las ubicaciones correctas (ver sección "Modelos Individuales").

```bash
ls -lh checkpoints/**/final_model.pt
```

---

## Referencias

### Documentos Relacionados

- `05_REPRODUCIBILIDAD_COMPLETA.md` - Reproducir entrenamiento del ensemble
- `06_CONFIGURACIONES_JSON.md` - Config usado (`landmarks_train_base.json`)
- `GROUND_TRUTH.json` - Métricas validadas del ensemble

### Papers

- **ResNet:** He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)
- **Coordinate Attention:** Hou et al., "Coordinate Attention for Efficient Mobile Network Design" (CVPR 2021)
- **Wing Loss:** Feng et al., "Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks" (CVPR 2018)

### Archivos de Modelos

Los 4 checkpoints están en el USB:
```
03_Modelos/
├── seed123_final_model.pt    (46 MB)
├── seed321_final_model.pt    (46 MB)
├── seed111_final_model.pt    (46 MB)
└── seed666_final_model.pt    (46 MB)
```

**Total:** 184 MB

---

**Última actualización:** 28 de enero de 2026

**Contacto:**
- Estudiante: Rafael Alejandro Cruz Ovando, BUAP
- Director: Dr. Leopoldo Altamirano Robles (robles@inaoep.mx), INAOE
