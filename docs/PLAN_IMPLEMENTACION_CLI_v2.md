# Plan de Implementacion - Completar CLI v2

**Fecha:** 2025-12-08
**Basado en:** Analisis de Gaps Session 19

## Objetivo
Alcanzar ~90% de reproducibilidad de experimentos originales via CLI.

---

## FASE 1: Pipeline Basico Completo (Sesiones 20-21)

### Sesion 20: Generacion de Datasets

#### Tarea 1.1: Comando `generate-dataset`
**Archivo:** `src_v2/cli.py`
**Lineas estimadas:** ~200

```bash
# Uso propuesto
python -m src_v2 generate-dataset \
    --input data/COVID-19_Radiography_Dataset \
    --output outputs/warped_dataset \
    --checkpoint checkpoints/final_model.pt \
    --ensemble checkpoints/seed123.pt checkpoints/seed456.pt \
    --margin 1.05 \
    --canonical outputs/shape_analysis/canonical_shape_gpa.json \
    --triangles outputs/shape_analysis/canonical_delaunay_triangles.json \
    --splits 0.75,0.125,0.125 \
    --seed 42
```

**Funcionalidades:**
- [ ] Cargar dataset original (ImageFolder o estructura custom)
- [ ] Predecir landmarks con modelo/ensemble
- [ ] Aplicar warping con margen configurable
- [ ] Crear splits train/val/test
- [ ] Guardar metadata (landmarks.json, images.csv)
- [ ] Generar dataset_summary.json
- [ ] Progreso con tqdm
- [ ] Soporte para TTA en prediccion de landmarks

**Basado en:** `scripts/generate_full_warped_dataset.py`

#### Tarea 1.2: Comando `compute-canonical`
**Archivo:** `src_v2/cli.py`
**Lineas estimadas:** ~150

```bash
# Uso propuesto
python -m src_v2 compute-canonical \
    --landmarks-csv data/coordenadas/coordenadas_maestro.csv \
    --output-dir outputs/shape_analysis \
    --visualize
```

**Funcionalidades:**
- [ ] Cargar coordenadas de landmarks
- [ ] Implementar Generalized Procrustes Analysis (GPA)
- [ ] Calcular forma canonica (mean shape)
- [ ] Generar triangulacion de Delaunay
- [ ] Guardar canonical_shape_gpa.json
- [ ] Guardar canonical_delaunay_triangles.json
- [ ] Visualizacion opcional de forma canonica

**Basado en:** `scripts/gpa_analysis.py`

### Sesion 21: Arquitecturas Adicionales

#### Tarea 1.3: Agregar Arquitecturas al Clasificador
**Archivo:** `src_v2/models/classifier.py`

**Arquitecturas a agregar:**
- [ ] AlexNet
- [ ] ResNet-50
- [ ] MobileNetV2
- [ ] VGG-16

**Cambios:**
```python
SUPPORTED_BACKBONES = [
    "resnet18",      # Existente
    "efficientnet_b0", # Existente
    "densenet121",   # Existente
    "alexnet",       # NUEVO
    "resnet50",      # NUEVO
    "mobilenet_v2",  # NUEVO
    "vgg16",         # NUEVO
]
```

#### Tarea 1.4: Tests para Nuevas Arquitecturas
**Archivo:** `tests/test_classifier.py`

- [ ] Test de creacion de cada arquitectura
- [ ] Test de forward pass
- [ ] Test de guardado/carga de checkpoint

---

## FASE 2: Analisis y Comparacion (Sesiones 22-23)

### Sesion 22: Comparacion Multi-Arquitectura

#### Tarea 2.1: Comando `compare-architectures`
**Archivo:** `src_v2/cli.py`
**Lineas estimadas:** ~300

```bash
# Uso propuesto
python -m src_v2 compare-architectures \
    --data-dir outputs/full_warped_dataset \
    --architectures resnet18,resnet50,efficientnet_b0,densenet121,mobilenet_v2,alexnet,vgg16 \
    --epochs 30 \
    --output-dir outputs/architecture_comparison \
    --seed 42
```

**Funcionalidades:**
- [ ] Entrenar cada arquitectura secuencialmente
- [ ] Evaluar en test set
- [ ] Generar tabla comparativa (CSV)
- [ ] Generar graficas de comparacion
- [ ] Calcular tiempo de entrenamiento por arquitectura
- [ ] Guardar todos los checkpoints
- [ ] Generar comparison_summary.json

**Basado en:** `scripts/train_all_architectures.py`, `scripts/compare_classifiers.py`

### Sesion 23: Analisis de Errores y Margenes

#### Tarea 2.2: Comando `analyze-errors`
**Archivo:** `src_v2/cli.py`
**Lineas estimadas:** ~200

```bash
# Uso propuesto
python -m src_v2 analyze-errors \
    --checkpoint outputs/classifier/best.pt \
    --data-dir outputs/full_warped_dataset \
    --output-dir outputs/error_analysis \
    --top-k 50
```

**Funcionalidades:**
- [ ] Identificar imagenes mal clasificadas
- [ ] Calcular confusion matrix detallada
- [ ] Analizar patrones de error por clase
- [ ] Guardar imagenes de errores mas confiados (falsos positivos/negativos)
- [ ] Generar error_analysis.json con estadisticas
- [ ] Visualizar distribucion de confianzas en errores

**Basado en:** `scripts/session30_error_analysis.py`

#### Tarea 2.3: Comando `optimize-margin`
**Archivo:** `src_v2/cli.py`
**Lineas estimadas:** ~250

```bash
# Uso propuesto
python -m src_v2 optimize-margin \
    --data-dir data/COVID-19_Radiography_Dataset \
    --checkpoint checkpoints/landmark_model.pt \
    --margins 1.00,1.05,1.10,1.15,1.20,1.25,1.30 \
    --samples 3000 \
    --epochs 20 \
    --output-dir outputs/margin_optimization
```

**Funcionalidades:**
- [ ] Generar dataset warped para cada margen
- [ ] Entrenar clasificador rapido (epochs reducidos)
- [ ] Evaluar accuracy para cada margen
- [ ] Identificar margen optimo
- [ ] Generar grafica de accuracy vs margen
- [ ] Guardar margin_optimization_results.json

**Basado en:** `scripts/margin_optimization_experiment.py`

---

## FASE 3: Explicabilidad (Sesion 24)

### Sesion 24: Grad-CAM y Visualizacion

#### Tarea 3.1: Comando `gradcam`
**Archivo:** `src_v2/cli.py`
**Lineas estimadas:** ~250

```bash
# Uso propuesto - imagen individual
python -m src_v2 gradcam \
    --checkpoint outputs/classifier/best.pt \
    --image test_image.png \
    --output gradcam_output.png \
    --layer layer4

# Uso propuesto - directorio
python -m src_v2 gradcam \
    --checkpoint outputs/classifier/best.pt \
    --data-dir outputs/full_warped_dataset/test \
    --output-dir outputs/gradcam_analysis \
    --samples 100
```

**Funcionalidades:**
- [ ] Implementar Grad-CAM para clasificadores
- [ ] Soporte para seleccion de capa target
- [ ] Overlay de heatmap sobre imagen original
- [ ] Procesamiento batch para multiples imagenes
- [ ] Calcular Pulmonary Focus Score (PFS) opcional
- [ ] Guardar gradcam_analysis.json con estadisticas

**Basado en:** `scripts/gradcam_comparison.py`, `scripts/gradcam_pfs_analysis.py`

#### Tarea 3.2: Modulo de Grad-CAM
**Archivo:** `src_v2/evaluation/gradcam.py` (NUEVO)

```python
class GradCAM:
    def __init__(self, model, target_layer):
        ...

    def generate(self, input_tensor, target_class=None):
        ...

    def visualize(self, image, heatmap, alpha=0.5):
        ...

def compute_pulmonary_focus_score(heatmap, lung_mask):
    """Calcula que porcentaje de atencion esta en area pulmonar."""
    ...
```

---

## FASE 4: Produccion (Sesiones Futuras)

### Tareas Pendientes para Produccion

#### 4.1: API REST
**Archivo:** `src_v2/api/` (NUEVO directorio)

```python
# src_v2/api/main.py
from fastapi import FastAPI, UploadFile
app = FastAPI()

@app.post("/predict/landmarks")
async def predict_landmarks(image: UploadFile):
    ...

@app.post("/predict/classify")
async def classify_image(image: UploadFile, warp: bool = False):
    ...

@app.post("/warp")
async def warp_image(image: UploadFile):
    ...
```

#### 4.2: Domain Adaptation
**Comando propuesto:** `domain-adapt`
**Tecnicas:** DANN, CORAL, self-training

#### 4.3: Modelo 4 Clases
**Cambio:** Flag `--num-classes 4` en train-classifier
**Clases:** COVID, Normal, Viral_Pneumonia, Lung_Opacity

---

## Resumen de Comandos a Implementar

| Fase | Comando | Prioridad | Complejidad | Sesion |
|------|---------|-----------|-------------|--------|
| 1 | `generate-dataset` | ALTA | Media | 20 |
| 1 | `compute-canonical` | ALTA | Media | 20 |
| 1 | Arquitecturas adicionales | ALTA | Baja | 21 |
| 2 | `compare-architectures` | MEDIA | Alta | 22 |
| 2 | `analyze-errors` | MEDIA | Media | 23 |
| 2 | `optimize-margin` | MEDIA | Media | 23 |
| 3 | `gradcam` | MEDIA | Media | 24 |
| 4 | API REST | BAJA | Alta | Futuro |

---

## Metricas de Exito

### Al completar Fase 1:
- [ ] Poder generar dataset warped desde cero via CLI
- [ ] Poder calcular forma canonica via CLI
- [ ] 7 arquitecturas soportadas en clasificador

### Al completar Fase 2:
- [ ] Comparacion automatica de arquitecturas
- [ ] Analisis de errores reproducible
- [ ] Busqueda de margen optimo automatizada

### Al completar Fase 3:
- [ ] Visualizaciones de Grad-CAM via CLI
- [ ] Pulmonary Focus Score calculable

### Cobertura Final Esperada:
- **Experimentos reproducibles via CLI:** ~90%
- **Comandos CLI totales:** 18 (vs 12 actuales)
- **Tests adicionales:** ~50 nuevos tests

---

## Notas de Implementacion

### Reutilizacion de Codigo
- `generate-dataset` puede reutilizar logica de `warp`
- `compare-architectures` puede reutilizar `train-classifier`
- `gradcam` requiere nuevo modulo pero usa modelo existente

### Dependencias Nuevas
- `scipy` para GPA (ya instalado)
- `pytorch-grad-cam` o implementacion custom para Grad-CAM

### Estructura de Archivos Propuesta
```
src_v2/
├── cli.py                    # +5 comandos nuevos
├── models/
│   └── classifier.py         # +4 arquitecturas
├── evaluation/
│   ├── metrics.py            # Existente
│   └── gradcam.py            # NUEVO
└── data/
    └── gpa.py                # NUEVO - Procrustes Analysis
```

---

## Siguiente Sesion Recomendada

**Sesion 20:** Implementar `generate-dataset` y `compute-canonical`

Esto desbloqueara la capacidad de recrear todo el pipeline desde cero, que es el gap mas critico actualmente.
