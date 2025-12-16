# Sesion 23: Comandos gradcam y analyze-errors

## Fecha: 2025-12-09

## Objetivo
Implementar comandos para explainabilidad del clasificador COVID-19:
1. `gradcam` - Visualizaciones Grad-CAM para interpretar decisiones del modelo
2. `analyze-errors` - Analisis detallado de errores de clasificacion

## Resumen Ejecutivo

| Aspecto | Detalle |
|---------|---------|
| **Comandos implementados** | `gradcam`, `analyze-errors` |
| **Modulo nuevo** | `src_v2/visualization/` |
| **Lineas de codigo** | ~1100 lineas nuevas |
| **Tests nuevos** | 31 tests (de 327 a 358) |
| **Bugs corregidos** | 15+ bugs (8 criticos, 7 medios) |
| **Arquitecturas soportadas** | 7 (resnet18, resnet50, efficientnet_b0, densenet121, alexnet, vgg16, mobilenet_v2) |
| **Validacion** | Probado con datos reales y 3 arquitecturas |

## Archivos Creados

### src_v2/visualization/__init__.py
- Modulo de visualizacion con exports para GradCAM y ErrorAnalyzer

### src_v2/visualization/gradcam.py (~380 lineas)
- **Clases:**
  - `GradCAM` - Implementacion de Gradient-weighted Class Activation Mapping
- **Funciones:**
  - `get_target_layer()` - Auto-deteccion de capa objetivo por arquitectura
  - `calculate_pfs()` - Pulmonary Focus Score
  - `overlay_heatmap()` - Superposicion de heatmap sobre imagen
  - `create_gradcam_visualization()` - Visualizacion completa con anotaciones
- **Constantes:**
  - `TARGET_LAYER_MAP` - Mapeo de arquitectura a capa objetivo

### src_v2/visualization/error_analysis.py (~480 lineas)
- **Dataclasses:**
  - `ErrorDetail` - Detalles de un error individual
  - `ErrorSummary` - Estadisticas resumen
- **Clases:**
  - `ErrorAnalyzer` - Analizador de errores de clasificacion
- **Funciones:**
  - `analyze_classification_errors()` - Wrapper de alto nivel
  - `create_error_visualizations()` - Generar figuras

### tests/test_visualization.py (~200 lineas)
- `TestGradCAMModule` - 11 tests para GradCAM
- `TestErrorAnalysisModule` - 10 tests para ErrorAnalyzer

## Archivos Modificados

### src_v2/cli.py
- **Lineas agregadas:** 4770-5318 (~550 lineas)
- **Comandos nuevos:**
  - `gradcam` - Lineas 4770-5047
  - `analyze-errors` - Lineas 5050-5311

### tests/test_cli.py
- **Tests agregados:** 12 nuevos tests
  - `TestGradCAMCommand` - 7 tests
  - `TestAnalyzeErrorsCommand` - 5 tests
- **Actualizacion:** Conteo de comandos de 15 a 17

## Comandos Implementados

### 1. Comando gradcam

#### Uso
```bash
# Imagen individual
python -m src_v2 gradcam \
    --checkpoint outputs/classifier/best.pt \
    --image test.png --output gradcam.png

# Modo batch (multiples imagenes)
python -m src_v2 gradcam \
    --checkpoint outputs/classifier/best.pt \
    --data-dir outputs/warped_dataset/test \
    --output-dir outputs/gradcam_analysis \
    --num-samples 20

# Personalizar colormap y transparencia
python -m src_v2 gradcam \
    --checkpoint outputs/classifier/best.pt \
    --image test.png --output gradcam.png \
    --colormap viridis --alpha 0.7
```

#### Parametros

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `--checkpoint` | Path | Requerido | Checkpoint del clasificador |
| `--image` | Path | None | Imagen individual |
| `--data-dir` | Path | None | Directorio batch |
| `--output` | Path | None | Salida imagen individual |
| `--output-dir` | Path | None | Directorio salida batch |
| `--layer` | str | auto | Capa objetivo (auto-detecta) |
| `--num-samples` | int | 10 | Muestras por clase |
| `--colormap` | str | jet | jet, hot, viridis |
| `--alpha` | float | 0.5 | Transparencia heatmap (0-1) |
| `--device` | str | auto | Dispositivo |

#### Capas Objetivo por Arquitectura
```python
TARGET_LAYER_MAP = {
    "resnet18": "backbone.layer4",
    "resnet50": "backbone.layer4",
    "densenet121": "backbone.features.denseblock4",
    "efficientnet_b0": "backbone.features.8",
    "vgg16": "backbone.features.30",
    "alexnet": "backbone.features.12",
    "mobilenet_v2": "backbone.features.18",
}
```

### 2. Comando analyze-errors

#### Uso
```bash
# Analisis basico
python -m src_v2 analyze-errors \
    --checkpoint outputs/classifier/best.pt \
    --data-dir outputs/warped_dataset/test \
    --output-dir outputs/error_analysis

# Con visualizaciones y GradCAM para errores
python -m src_v2 analyze-errors \
    --checkpoint outputs/classifier/best.pt \
    --data-dir outputs/warped_dataset/test \
    --output-dir outputs/error_analysis \
    --visualize --gradcam --top-k 30
```

#### Parametros

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `--checkpoint` | Path | Requerido | Checkpoint del clasificador |
| `--data-dir` | Path | Requerido | Directorio del dataset |
| `--output-dir` | Path | Requerido | Directorio de salida |
| `--visualize` | bool | False | Generar graficos |
| `--gradcam` | bool | False | GradCAM para errores |
| `--top-k` | int | 20 | Top errores a procesar |
| `--batch-size` | int | 32 | Tamano de batch |
| `--device` | str | auto | Dispositivo |

#### Estructura de Salida
```
outputs/error_analysis/
├── error_summary.json           # Resumen estadistico
├── error_details.csv            # Detalles por imagen
├── confusion_analysis.json      # Matriz de confusion
├── figures/
│   ├── error_distribution.png   # Errores por clase
│   ├── confidence_histogram.png # Distribucion confianza
│   ├── confusion_matrix.png     # Heatmap confusion
│   └── misclassified/          # Imagenes copiadas
└── gradcam_errors/             # Si --gradcam
    ├── COVID_as_Normal/
    │   └── img_conf0.95_gradcam.png
    └── ...
```

## Implementacion Tecnica

### Grad-CAM
```python
class GradCAM:
    def __init__(self, model, target_layer):
        # Registrar hooks para capturar activaciones y gradientes
        self._register_hooks()

    def __call__(self, input_tensor, target_class=None):
        # Forward pass
        output = self.model(input_tensor)

        # Backward pass para clase objetivo
        output[0, target_class].backward()

        # Calcular pesos via GAP de gradientes
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Combinacion ponderada + ReLU
        cam = F.relu((weights * self.activations).sum(dim=1))

        # Normalizar a [0, 1]
        return cam / cam.max()
```

### Pulmonary Focus Score (PFS)
```python
def calculate_pfs(heatmap, mask):
    """
    PFS = sum(heatmap * mask) / sum(heatmap)

    - 1.0 = Modelo enfocado 100% en pulmones
    - 0.5 = Enfoque igual dentro/fuera de pulmones
    - <0.5 = Enfocado mas fuera de pulmones
    """
    overlap = (heatmap * mask).sum()
    total = heatmap.sum()
    return overlap / total
```

## Bugs Corregidos

### Criticos (8)

#### 1. Batch Size Validation en GradCAM
**Ubicacion:** `gradcam.py:166-167`
**Problema:** GradCAM asumia batch size=1 sin validar
**Solucion:**
```python
if input_tensor.shape[0] != 1:
    raise ValueError(f"Batch size must be 1, got {input_tensor.shape[0]}")
```

#### 2. None Check para Gradientes
**Ubicacion:** `gradcam.py:195-199`
**Problema:** No se verificaba si hooks capturaron datos
**Solucion:**
```python
if self.gradients is None or self.activations is None:
    raise RuntimeError("Gradients or activations not captured...")
```

#### 3. Division por Cero en Normalizacion
**Ubicacion:** `gradcam.py:220-224`
**Problema:** cam.max() podia ser cero
**Solucion:**
```python
if cam_max > 1e-8:
    cam = cam / cam_max
else:
    cam = torch.zeros_like(cam)
```

#### 4. Empty class_names Validation
**Ubicacion:** `error_analysis.py:72-73`
**Problema:** Lista vacia causaba errores
**Solucion:**
```python
if not class_names:
    raise ValueError("class_names cannot be empty")
```

#### 5. Label Out of Bounds
**Ubicacion:** `error_analysis.py:100-101`
**Problema:** Labels invalidos no detectados
**Solucion:**
```python
if not (0 <= label < self.num_classes):
    raise ValueError(f"Label {label} out of range...")
```

#### 6. GPU Memory Leak en gradcam
**Ubicacion:** `cli.py:5037-5043`
**Problema:** Modelo no liberado despues de uso
**Solucion:**
```python
finally:
    gradcam.remove_hooks()
    if torch_device.type == "cuda":
        del model
        torch.cuda.empty_cache()
```

#### 7. GPU Memory Leak en analyze-errors
**Ubicacion:** `cli.py:5307-5311`
**Problema:** Memoria GPU no liberada
**Solucion:**
```python
if torch_device.type == "cuda":
    del model
    torch.cuda.empty_cache()
```

#### 8. GradCAM Hooks No Liberados en Errores
**Ubicacion:** `cli.py:5263-5297`
**Problema:** Si fallaba el loop, hooks quedaban registrados
**Solucion:**
```python
try:
    for error in top_errors:
        # proceso
finally:
    gradcam.remove_hooks()
```

### Medios (7)

1. **File Encoding Issues** - Agregado `encoding="utf-8"` a todos los archivos
2. **Counter no serializable** - Convertido a `dict()` antes de JSON
3. **Parameter Validation (alpha)** - Validar rango 0.0-1.0
4. **Parameter Validation (num_samples)** - Validar > 0
5. **Parameter Validation (batch_size)** - Validar > 0
6. **Parameter Validation (top_k)** - Validar > 0
7. **target_class Validation** - Validar rango valido en GradCAM

## Validacion

### Tests Ejecutados
```bash
$ python -m pytest tests/ -v
...
====================== 358 passed, 23 warnings in 17.92s =======================
```

### Tests por Modulo
- `test_visualization.py`: 21 tests (GradCAM: 11, ErrorAnalysis: 10)
- `test_cli.py`: 12 tests nuevos (GradCAM: 7, AnalyzeErrors: 5)

### Validacion con Datos Reales

#### GradCAM - DenseNet-121
```
Prediccion: COVID (98.5%)
Visualizacion: Heatmap enfocado en region pulmonar inferior
```

#### GradCAM - EfficientNet-B0
```
Prediccion: COVID (100.0%)
Visualizacion: Activacion distribuida en region central pulmonar
```

#### Analyze-Errors Output
```
Total muestras: 315
Total errores: 26 (8.3%)
Confianza promedio (errores): 86.2%
Confianza promedio (correctos): 97.8%

Pares de confusion mas frecuentes:
  Viral_Pneumonia->COVID: 12
  COVID->Normal: 8
  Normal->COVID: 6
```

## Metricas de Calidad

| Metrica | Objetivo | Logrado |
|---------|----------|---------|
| Tests nuevos | >= 9 | 31 |
| Tests totales pasando | >= 327 | 358 |
| Arquitecturas soportadas | 7 | 7 |
| Memory leaks | 0 | 0 (corregidos) |
| Datos hardcodeados | 0 | 0 |

## Dependencias Utilizadas

- `torch` - Hooks, backward pass
- `matplotlib` - Visualizaciones
- `cv2` - Anotaciones en imagenes
- `PIL` - Manipulacion de imagenes
- `numpy` - Operaciones numericas

## Notas de Implementacion

1. **Context Manager para GradCAM**: Se implemento `__enter__`/`__exit__` para uso con `with`:
   ```python
   with GradCAM(model, layer) as gradcam:
       heatmap, _, _ = gradcam(tensor)
   # hooks automaticamente removidos
   ```

2. **Auto-deteccion de Capa**: El sistema detecta automaticamente la mejor capa segun arquitectura, pero permite override manual.

3. **Compatibilidad Multi-arquitectura**: Probado con ResNet, EfficientNet y DenseNet. Las 7 arquitecturas tienen capas objetivo definidas.

## Proximos Pasos (Sesion 24)

1. **Analisis PFS completo**: Implementar calculo de PFS con mascaras pulmonares
2. **Reporte de interpretabilidad**: Documento con hallazgos de GradCAM
3. **Optimizacion de memoria**: Profiling para uso intensivo
4. **Tests de integracion E2E**: Pipeline completo de explainabilidad
