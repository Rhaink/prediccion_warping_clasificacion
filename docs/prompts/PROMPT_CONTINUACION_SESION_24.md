# Prompt de Continuacion - Sesion 24: Analisis PFS y Reporte de Interpretabilidad

**Fecha:** 2025-12-09
**Sesion anterior:** 23 (Comandos gradcam y analyze-errors)
**Estado previo:** 17 comandos CLI, 358 tests, 15+ bugs corregidos

## Resumen de Sesiones Anteriores

### Sesion 22 - Comando compare-architectures
- **Comando implementado:** `compare-architectures`
- **9 bugs corregidos** incluyendo memory leaks y bugs de matplotlib
- **15 tests nuevos**
- **327 tests pasando**
- **Validado:** ResNet-18 → 83.33% acc, 84% F1

### Sesion 23 - Comandos gradcam y analyze-errors
- **Comandos implementados:** `gradcam`, `analyze-errors`
- **Modulo nuevo:** `src_v2/visualization/`
- **15+ bugs corregidos:**
  - Batch size validation en GradCAM
  - None check para gradientes/activaciones
  - Division por cero en normalizacion
  - Empty class_names validation
  - Label out of bounds
  - GPU memory leaks (2)
  - GradCAM hooks no liberados
  - File encoding issues
  - Parameter validations (alpha, num_samples, batch_size, top_k)
- **31 tests nuevos**
- **358 tests pasando**
- **Validado con 3 arquitecturas:** ResNet-18, EfficientNet-B0, DenseNet-121

## Estado Actual del CLI

### Comandos Implementados (17 total)

#### Landmarks (5)
- `train` - Entrenar modelo de landmarks
- `evaluate` - Evaluar modelo individual
- `evaluate-ensemble` - Evaluar ensemble
- `predict` - Predecir en imagen
- `warp` - Aplicar warping

#### Clasificacion (9)
- `classify` - Clasificar imagenes
- `train-classifier` - Entrenar clasificador
- `evaluate-classifier` - Evaluar clasificador
- `cross-evaluate` - Evaluacion cruzada
- `evaluate-external` - Validacion externa
- `test-robustness` - Pruebas de robustez
- `compare-architectures` - Comparar arquitecturas
- `gradcam` - **NUEVO** Visualizaciones Grad-CAM
- `analyze-errors` - **NUEVO** Analisis de errores

#### Procesamiento (2)
- `compute-canonical` - Calcular forma canonica
- `generate-dataset` - Generar dataset warped

#### Utilidad (1)
- `version` - Mostrar version

### Metricas
- **Tests:** 358 pasando
- **Arquitecturas soportadas:** 7 (resnet18, resnet50, efficientnet_b0, densenet121, alexnet, vgg16, mobilenet_v2)
- **Cobertura experimentos:** ~85%

## Modulo visualization Implementado

### src_v2/visualization/gradcam.py

```python
# Capas objetivo por arquitectura
TARGET_LAYER_MAP = {
    "resnet18": "backbone.layer4",
    "resnet50": "backbone.layer4",
    "densenet121": "backbone.features.denseblock4",
    "efficientnet_b0": "backbone.features.8",
    "vgg16": "backbone.features.30",
    "alexnet": "backbone.features.12",
    "mobilenet_v2": "backbone.features.18",
}

# Funciones disponibles
get_target_layer(model, backbone_name, layer_name=None)
calculate_pfs(heatmap, mask)  # Pulmonary Focus Score
overlay_heatmap(image, heatmap, alpha=0.5, colormap='jet')
create_gradcam_visualization(image, heatmap, prediction, confidence, ...)

# Clase GradCAM
class GradCAM:
    def __init__(self, model, target_layer): ...
    def __call__(self, input_tensor, target_class=None) -> (heatmap, pred_class, confidence): ...
    def remove_hooks(self): ...  # IMPORTANTE: llamar al finalizar
```

### src_v2/visualization/error_analysis.py

```python
# Dataclasses
@dataclass
class ErrorDetail:
    image_path: str
    true_class: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]

@dataclass
class ErrorSummary:
    total_samples: int
    total_errors: int
    error_rate: float
    errors_by_true_class: Dict[str, int]
    confusion_matrix: List[List[int]]
    ...

# Clase principal
class ErrorAnalyzer:
    def __init__(self, class_names): ...
    def add_prediction(self, output, label, image_path): ...
    def add_batch(self, outputs, labels, image_paths): ...
    def get_summary(self) -> ErrorSummary: ...
    def get_top_errors(self, k=20, by='confidence', descending=True): ...
    def save_reports(self, output_dir): ...

# Funciones de alto nivel
analyze_classification_errors(model, dataloader, class_names, device, output_dir)
create_error_visualizations(analyzer, output_dir, copy_images=False)
```

## Objetivos de Sesion 24

### Objetivo Principal: Analisis PFS Completo y Documentacion

Implementar el analisis completo de Pulmonary Focus Score (PFS) con mascaras pulmonares y generar reportes de interpretabilidad del modelo.

### Tarea 1: Comando `pfs-analysis` (Prioridad ALTA)

#### Proposito
Calcular el Pulmonary Focus Score (PFS) para evaluar si el modelo enfoca su atencion en las regiones pulmonares relevantes.

#### Formula PFS
```
PFS = sum(heatmap * mask) / sum(heatmap)

- 1.0 = Modelo enfocado 100% en pulmones
- 0.5 = Enfoque igual dentro/fuera de pulmones
- <0.5 = Enfocado mas fuera de pulmones (preocupante)
```

#### Uso Esperado
```bash
# Analisis PFS basico
python -m src_v2 pfs-analysis \
    --checkpoint outputs/classifier/best.pt \
    --data-dir outputs/warped_dataset/test \
    --mask-dir data/lung_masks \
    --output-dir outputs/pfs_analysis

# Comparar PFS entre arquitecturas
python -m src_v2 pfs-analysis \
    --checkpoint outputs/arch_comparison/checkpoints/resnet18_warped_best.pt \
    --checkpoint outputs/arch_comparison/checkpoints/efficientnet_b0_warped_best.pt \
    --data-dir outputs/warped_dataset/test \
    --mask-dir data/lung_masks \
    --output-dir outputs/pfs_comparison \
    --compare
```

#### Parametros Propuestos

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `--checkpoint` | Path | Requerido | Checkpoint(s) del clasificador |
| `--data-dir` | Path | Requerido | Directorio con imagenes |
| `--mask-dir` | Path | None | Directorio con mascaras pulmonares |
| `--output-dir` | Path | Requerido | Directorio de salida |
| `--num-samples` | int | 50 | Muestras por clase |
| `--compare` | bool | False | Comparar multiples checkpoints |
| `--threshold` | float | 0.5 | Umbral PFS minimo aceptable |

#### Salida Esperada
```
outputs/pfs_analysis/
├── pfs_summary.json             # Resumen estadistico
├── pfs_by_class.csv             # PFS por clase
├── pfs_details.csv              # PFS por imagen
├── figures/
│   ├── pfs_distribution.png     # Histograma PFS
│   ├── pfs_by_class.png         # PFS promedio por clase
│   ├── pfs_vs_confidence.png    # Correlacion PFS-confianza
│   └── low_pfs_samples/         # Imagenes con PFS bajo
└── gradcam_overlays/            # Visualizaciones con mascara
```

#### Metricas a Calcular
- PFS promedio global
- PFS por clase (COVID, Normal, Viral_Pneumonia)
- PFS por prediccion correcta vs incorrecta
- Correlacion PFS vs confianza
- Porcentaje de predicciones con PFS < umbral

### Tarea 2: Generar Mascaras Pulmonares (Prioridad MEDIA)

#### Opciones
1. **Usar mascaras existentes:** Si el dataset tiene mascaras de segmentacion
2. **Aproximacion rectangular:** Region central de la imagen
3. **Segmentacion automatica:** Usar modelo pre-entrenado (opcional)

#### Script sugerido
```bash
# Generar mascaras aproximadas (rectangulo central)
python -m src_v2 generate-lung-masks \
    --data-dir outputs/warped_dataset \
    --output-dir outputs/lung_masks \
    --method rectangular \
    --margin 0.15  # 15% de margen
```

### Tarea 3: Reporte de Interpretabilidad (Prioridad MEDIA)

Generar un documento markdown con hallazgos del analisis de interpretabilidad:

```
outputs/interpretability_report/
├── INTERPRETABILITY_REPORT.md   # Reporte completo
├── figures/
│   ├── gradcam_examples.png     # Ejemplos por clase
│   ├── pfs_summary.png          # Resumen PFS
│   └── attention_patterns.png   # Patrones de atencion
└── data/
    ├── gradcam_stats.json
    └── pfs_stats.json
```

#### Contenido del Reporte
1. **Resumen ejecutivo:** Conclusiones principales
2. **Analisis Grad-CAM:** Donde mira el modelo por clase
3. **Analisis PFS:** Enfoque pulmonar del modelo
4. **Patrones de error:** Relacion entre atencion y errores
5. **Recomendaciones:** Mejoras sugeridas

## Referencia de Funciones Existentes

### calculate_pfs (ya implementado)
```python
from src_v2.visualization.gradcam import calculate_pfs

# Ejemplo de uso
pfs = calculate_pfs(heatmap, lung_mask)
print(f"PFS: {pfs:.2%}")  # "PFS: 85.32%"
```

### Scripts de Referencia
```
scripts/gradcam_pfs_analysis.py  # Implementacion original PFS
scripts/session30_error_analysis.py  # Analisis de errores
```

## Implementacion Sugerida

### Arquitectura del Comando pfs-analysis

```python
@app.command("pfs-analysis")
def pfs_analysis(
    checkpoint: str,
    data_dir: str,
    mask_dir: Optional[str] = None,
    output_dir: str = "outputs/pfs_analysis",
    num_samples: int = 50,
    threshold: float = 0.5,
    device: str = "auto",
):
    """Analizar Pulmonary Focus Score del clasificador."""

    # 1. Cargar modelo y mascaras
    model = create_classifier(checkpoint=checkpoint, device=device)
    masks = load_lung_masks(mask_dir) if mask_dir else generate_approximate_masks()

    # 2. Para cada imagen
    pfs_results = []
    for image, mask in samples:
        # Generar GradCAM
        heatmap, pred, conf = gradcam(image)

        # Calcular PFS
        pfs = calculate_pfs(heatmap, mask)

        pfs_results.append({
            'image': image_path,
            'class': true_class,
            'predicted': pred_class,
            'confidence': conf,
            'pfs': pfs,
            'correct': pred_class == true_class,
        })

    # 3. Generar estadisticas y reportes
    generate_pfs_reports(pfs_results, output_dir)
    generate_pfs_figures(pfs_results, output_dir)

    # 4. Alertas
    low_pfs = [r for r in pfs_results if r['pfs'] < threshold]
    if low_pfs:
        logger.warning(f"{len(low_pfs)} imagenes con PFS < {threshold}")
```

## Tests Requeridos

### Tests para pfs-analysis

```python
class TestPFSAnalysisCommand:
    def test_pfs_analysis_help(self):
        result = runner.invoke(app, ['pfs-analysis', '--help'])
        assert result.exit_code == 0
        assert '--checkpoint' in result.stdout
        assert '--mask-dir' in result.stdout

    def test_pfs_analysis_requires_checkpoint(self):
        result = runner.invoke(app, ['pfs-analysis', '--data-dir', '/path'])
        assert result.exit_code != 0

    def test_calculate_pfs_known_values(self):
        # Heatmap uniforme, mascara mitad
        heatmap = np.ones((100, 100))
        mask = np.zeros((100, 100))
        mask[:, 50:] = 1  # Mitad derecha
        pfs = calculate_pfs(heatmap, mask)
        assert abs(pfs - 0.5) < 0.01

    def test_pfs_full_overlap(self):
        heatmap = np.ones((100, 100))
        mask = np.ones((100, 100))
        pfs = calculate_pfs(heatmap, mask)
        assert pfs == 1.0
```

## Criterios de Exito

### Comando pfs-analysis
- [ ] `pfs-analysis --help` muestra documentacion
- [ ] Calcula PFS correctamente para cada imagen
- [ ] Genera `pfs_summary.json` con estadisticas
- [ ] Genera visualizaciones de distribucion PFS
- [ ] Detecta imagenes con PFS bajo (<0.5)
- [ ] Tests nuevos pasan (minimo 5)

### Reporte de Interpretabilidad
- [ ] Documento markdown generado
- [ ] Incluye ejemplos de Grad-CAM
- [ ] Incluye analisis PFS
- [ ] Visualizaciones legibles
- [ ] Conclusiones claras

### General
- [ ] 358+ tests siguen pasando
- [ ] Sin datos hardcodeados
- [ ] Sin memory leaks
- [ ] Documentacion de sesion creada

## Archivos Clave a Consultar

```
src_v2/visualization/gradcam.py       # calculate_pfs() ya implementado
src_v2/visualization/error_analysis.py # ErrorAnalyzer
src_v2/cli.py                         # Comandos gradcam y analyze-errors
scripts/gradcam_pfs_analysis.py       # Referencia original PFS
docs/sesiones/SESION_23_GRADCAM_ERROR_ANALYSIS.md  # Sesion anterior
```

## Gaps Restantes Despues de Sesion 24

| Funcionalidad | Prioridad | Estado |
|---------------|-----------|--------|
| `pfs-analysis` | Alta | En progreso |
| `generate-lung-masks` | Media | Pendiente |
| Reporte interpretabilidad | Media | Pendiente |
| `optimize-margin` | Baja | Pendiente |

## Consideraciones Tecnicas

### Mascaras Pulmonares
1. **Si existen mascaras:** Cargarlas directamente
2. **Si no existen:**
   - Opcion A: Usar aproximacion rectangular (region central)
   - Opcion B: Implementar segmentacion basica
3. **Formato esperado:** PNG binario (0=fondo, 255=pulmon)

### Memoria GPU
- Procesar en batches pequenos
- Liberar gradientes despues de cada imagen
- Usar `with GradCAM(...) as gradcam:` para limpieza automatica

### Validacion de Mascaras
```python
def validate_mask(mask, image_shape):
    """Validar que mascara es compatible con imagen."""
    if mask.shape[:2] != image_shape[:2]:
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]))
    if mask.max() > 1:
        mask = mask / 255.0
    return mask
```

## Notas Importantes

1. **PFS es metrica clave para confianza clinica:**
   - Modelos con PFS alto son mas confiables
   - PFS bajo sugiere que modelo mira regiones irrelevantes

2. **Validacion cruzada de PFS:**
   - Comparar PFS entre arquitecturas
   - PFS para predicciones correctas vs incorrectas

3. **Umbrales sugeridos:**
   - PFS > 0.7: Excelente enfoque pulmonar
   - PFS 0.5-0.7: Aceptable
   - PFS < 0.5: Preocupante, revisar modelo

---

**Para comenzar la sesion:**
1. Lee este prompt completo
2. Revisa `scripts/gradcam_pfs_analysis.py` para entender PFS
3. Verifica si existen mascaras pulmonares en el dataset
4. Implementa el comando `pfs-analysis`
5. Genera el reporte de interpretabilidad
6. Documenta hallazgos

**Usa ultrathink para disenar la estrategia de mascaras pulmonares antes de implementar.**
