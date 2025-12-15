# Prompt de Continuacion - Sesion 23: Comandos gradcam y analyze-errors

**Fecha:** 2025-12-09
**Sesion anterior:** 22 (Comando compare-architectures)
**Estado previo:** 15 comandos CLI, 327 tests, 9 bugs corregidos

## Resumen de Sesiones Anteriores

### Sesion 21 - Validacion Funcional
- 3 bugs corregidos en GPA y warping
- 19 tests nuevos
- 312 tests pasando

### Sesion 22 - Comando compare-architectures
- **Comando implementado:** `compare-architectures`
- **9 bugs corregidos:**
  - Memory leak en GPU (critico)
  - Retorno de modelo incorrecto (critico)
  - Figuras sobrescritas (alto)
  - Bug de axes matplotlib (alto)
  - Sin validacion loaders vacios (alto)
  - Clases hardcodeadas (medio)
  - Sin validacion CSV (medio)
  - Imagenes no encontradas silenciosas (bajo)
  - Solo extension .png (bajo)
- **15 tests nuevos** incluyendo tests de sincronizacion
- **327 tests pasando**
- **Validado con datos reales:** ResNet-18 → 83.33% acc, 84% F1

## Estado Actual del CLI

### Comandos Implementados (15 total)

#### Landmarks (5)
- `train` - Entrenar modelo de landmarks
- `evaluate` - Evaluar modelo individual
- `evaluate-ensemble` - Evaluar ensemble
- `predict` - Predecir en imagen
- `warp` - Aplicar warping

#### Clasificacion (7)
- `classify` - Clasificar imagenes
- `train-classifier` - Entrenar clasificador
- `evaluate-classifier` - Evaluar clasificador
- `cross-evaluate` - Evaluacion cruzada
- `evaluate-external` - Validacion externa
- `test-robustness` - Pruebas de robustez
- `compare-architectures` - **NUEVO** Comparar arquitecturas

#### Procesamiento (2)
- `compute-canonical` - Calcular forma canonica
- `generate-dataset` - Generar dataset warped

#### Utilidad (1)
- `version` - Mostrar version

### Metricas
- **Tests:** 327 pasando
- **Arquitecturas clasificador:** 7
- **Cobertura experimentos:** ~80%

## Objetivos de Sesion 23

### Objetivo Principal: Explicabilidad y Analisis

Implementar comandos para explicabilidad del modelo y analisis de errores, criticos para aplicaciones clinicas.

### Tarea 1: Comando `gradcam` (Prioridad ALTA)

#### Proposito
Generar visualizaciones Grad-CAM para explicar las decisiones del clasificador COVID-19, mostrando que regiones de la imagen influyen en la prediccion.

#### Uso Esperado
```bash
# Visualizar una imagen
python -m src_v2 gradcam \
    --checkpoint outputs/classifier/best.pt \
    --image test_image.png \
    --output gradcam_output.png

# Visualizar batch de imagenes
python -m src_v2 gradcam \
    --checkpoint outputs/classifier/best.pt \
    --data-dir outputs/warped_dataset/test \
    --output-dir outputs/gradcam_analysis \
    --num-samples 20

# Con capa especifica
python -m src_v2 gradcam \
    --checkpoint outputs/classifier/best.pt \
    --image test.png \
    --layer layer4  # ultima capa convolucional
    --output gradcam.png
```

#### Parametros Propuestos

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `--checkpoint` | Path | Requerido | Checkpoint del clasificador |
| `--image` | Path | None | Imagen individual |
| `--data-dir` | Path | None | Directorio con imagenes |
| `--output` | Path | None | Archivo de salida (imagen individual) |
| `--output-dir` | Path | None | Directorio de salida (batch) |
| `--layer` | str | "auto" | Capa para Grad-CAM |
| `--num-samples` | int | 10 | Muestras por clase |
| `--colormap` | str | "jet" | Mapa de colores |
| `--alpha` | float | 0.5 | Transparencia del heatmap |

#### Funcionalidades
1. **Visualizacion individual:** Imagen + heatmap superpuesto
2. **Visualizacion batch:** Grid de ejemplos por clase
3. **Soporte multi-arquitectura:** Detectar automaticamente la capa correcta segun backbone
4. **Pulmonary Focus Score (PFS):** Calcular porcentaje de atencion en region pulmonar

#### Referencia
- `scripts/gradcam_comparison.py`
- `scripts/gradcam_multi_architecture.py`
- `scripts/gradcam_pfs_analysis.py`

### Tarea 2: Comando `analyze-errors` (Prioridad MEDIA)

#### Proposito
Analizar los errores de clasificacion para entender patrones de fallo del modelo.

#### Uso Esperado
```bash
# Analisis completo
python -m src_v2 analyze-errors \
    --checkpoint outputs/classifier/best.pt \
    --data-dir outputs/warped_dataset/test \
    --output-dir outputs/error_analysis

# Con visualizaciones
python -m src_v2 analyze-errors \
    --checkpoint outputs/classifier/best.pt \
    --data-dir outputs/warped_dataset/test \
    --output-dir outputs/error_analysis \
    --visualize \
    --gradcam  # Agregar Grad-CAM a errores
```

#### Parametros Propuestos

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `--checkpoint` | Path | Requerido | Checkpoint del clasificador |
| `--data-dir` | Path | Requerido | Directorio de test |
| `--output-dir` | Path | Requerido | Directorio de salida |
| `--visualize` | bool | False | Generar visualizaciones |
| `--gradcam` | bool | False | Agregar Grad-CAM a errores |
| `--top-k` | int | 20 | Top K errores por categoria |

#### Salida Esperada
```
outputs/error_analysis/
├── error_summary.json          # Resumen de errores
├── error_details.csv           # Detalles por imagen
├── confusion_analysis.json     # Analisis de confusion
├── figures/
│   ├── error_distribution.png  # Distribucion de errores
│   ├── confidence_histogram.png # Histograma de confianza
│   ├── misclassified_COVID/    # Imagenes COVID mal clasificadas
│   ├── misclassified_Normal/
│   └── misclassified_Viral_Pneumonia/
└── gradcam_errors/             # (si --gradcam)
    ├── COVID_as_Normal/
    ├── Normal_as_COVID/
    └── ...
```

#### Metricas a Reportar
- Errores por clase (count, %)
- Matriz de confusion detallada
- Confianza promedio en errores vs aciertos
- Top-K errores mas "seguros" (alta confianza incorrecta)
- Patrones de confusion (COVID→Normal, Normal→COVID, etc.)

#### Referencia
- `scripts/session30_error_analysis.py`

## Implementacion Sugerida

### Arquitectura Grad-CAM

```python
# src_v2/visualization/__init__.py (nuevo modulo)

class GradCAM:
    """Grad-CAM para visualizacion de atencion."""

    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer or self._detect_target_layer()
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _detect_target_layer(self):
        """Detectar ultima capa convolucional segun arquitectura."""
        backbone = self.model.backbone_name
        layer_map = {
            'resnet18': 'backbone.layer4',
            'resnet50': 'backbone.layer4',
            'densenet121': 'backbone.features.denseblock4',
            'efficientnet_b0': 'backbone.features.8',
            'vgg16': 'backbone.features.30',
            'alexnet': 'backbone.features.12',
            'mobilenet_v2': 'backbone.features.18',
        }
        return layer_map.get(backbone, 'backbone.layer4')

    def generate(self, input_tensor, target_class=None):
        """Generar mapa de calor Grad-CAM."""
        ...

    def visualize(self, image, heatmap, alpha=0.5, colormap='jet'):
        """Superponer heatmap en imagen."""
        ...
```

### Estructura de Archivos Sugerida

```
src_v2/
├── visualization/           # NUEVO modulo
│   ├── __init__.py
│   ├── gradcam.py          # Implementacion Grad-CAM
│   └── error_analysis.py   # Analisis de errores
├── cli.py                  # Agregar comandos gradcam y analyze-errors
└── ...
```

## Tests Requeridos

### Tests para gradcam

```python
class TestGradCAMCommand:
    def test_gradcam_help(self):
        result = runner.invoke(app, ['gradcam', '--help'])
        assert result.exit_code == 0
        assert '--checkpoint' in result.stdout
        assert '--image' in result.stdout

    def test_gradcam_requires_checkpoint(self):
        result = runner.invoke(app, ['gradcam', '--image', 'test.png'])
        assert result.exit_code != 0

    def test_gradcam_requires_image_or_data_dir(self):
        result = runner.invoke(app, [
            'gradcam',
            '--checkpoint', '/path/to/model.pt'
        ])
        assert result.exit_code != 0

    def test_gradcam_invalid_layer(self):
        result = runner.invoke(app, [
            'gradcam',
            '--checkpoint', '/path/to/model.pt',
            '--image', 'test.png',
            '--layer', 'invalid_layer'
        ])
        assert result.exit_code != 0
```

### Tests para analyze-errors

```python
class TestAnalyzeErrorsCommand:
    def test_analyze_errors_help(self):
        result = runner.invoke(app, ['analyze-errors', '--help'])
        assert result.exit_code == 0
        assert '--checkpoint' in result.stdout
        assert '--data-dir' in result.stdout

    def test_analyze_errors_requires_args(self):
        result = runner.invoke(app, ['analyze-errors'])
        assert result.exit_code != 0

    def test_analyze_errors_missing_checkpoint(self):
        result = runner.invoke(app, [
            'analyze-errors',
            '--data-dir', '/path/to/data'
        ])
        assert result.exit_code != 0
```

## Criterios de Exito

### Comando gradcam
- [ ] `gradcam --help` muestra documentacion
- [ ] Genera visualizacion para imagen individual
- [ ] Genera visualizaciones batch
- [ ] Detecta capa automaticamente segun arquitectura
- [ ] Soporta las 7 arquitecturas
- [ ] Tests nuevos pasan (minimo 5)

### Comando analyze-errors
- [ ] `analyze-errors --help` muestra documentacion
- [ ] Genera `error_summary.json`
- [ ] Genera `error_details.csv`
- [ ] Genera visualizaciones opcionales
- [ ] Integra Grad-CAM para errores (opcional)
- [ ] Tests nuevos pasan (minimo 4)

### General
- [ ] 327+ tests siguen pasando
- [ ] Documentacion de sesion creada
- [ ] Sin datos hardcodeados
- [ ] Sin memory leaks

## Archivos Clave a Consultar

```
scripts/gradcam_comparison.py           # Implementacion original Grad-CAM
scripts/gradcam_multi_architecture.py   # Multi-arquitectura
scripts/gradcam_pfs_analysis.py         # Pulmonary Focus Score
scripts/session30_error_analysis.py     # Analisis de errores
src_v2/models/classifier.py             # ImageClassifier
src_v2/cli.py                           # CLI actual (15 comandos)
docs/sesiones/SESION_22_COMPARE_ARCHITECTURES.md  # Sesion anterior
```

## Gaps Restantes Despues de Sesion 23

| Funcionalidad | Prioridad | Estado |
|---------------|-----------|--------|
| `gradcam` | Alta | En progreso |
| `analyze-errors` | Media | En progreso |
| `optimize-margin` | Baja | Pendiente |

## Notas Importantes

1. **Grad-CAM es critico para aplicaciones clinicas:**
   - Los medicos necesitan entender por que el modelo hace predicciones
   - El Pulmonary Focus Score (PFS) mide si el modelo mira los pulmones

2. **Compatibilidad con arquitecturas:**
   - Cada backbone tiene diferente estructura
   - Necesario mapeo de capas por arquitectura

3. **Analisis de errores:**
   - Identificar patrones de fallo ayuda a mejorar el modelo
   - COVID→Normal y Normal→COVID son los errores mas criticos

4. **Consideraciones de memoria:**
   - Grad-CAM requiere guardar gradientes
   - Procesar en batches pequenos para imagenes grandes

---

**Para comenzar la sesion:**
1. Lee este prompt completo
2. Revisa `scripts/gradcam_comparison.py` para entender la implementacion
3. Crea el modulo `src_v2/visualization/` con gradcam.py
4. Implementa el comando `gradcam` primero (mas complejo)
5. Luego implementa `analyze-errors`
6. Agrega tests antes de probar con datos reales

**Usa ultrathink para disenar la arquitectura del modulo visualization antes de implementar.**
