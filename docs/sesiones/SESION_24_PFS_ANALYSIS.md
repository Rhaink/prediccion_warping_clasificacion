# Sesión 24: Análisis PFS y Comandos de Máscaras

**Fecha:** 2025-12-09
**Sesión anterior:** 23 (Comandos gradcam y analyze-errors)
**Estado previo:** 17 comandos CLI, 358 tests

## Resumen de Implementación

### Objetivos Cumplidos

1. ✅ Implementar comando `pfs-analysis` para análisis de Pulmonary Focus Score
2. ✅ Implementar comando `generate-lung-masks` para generar máscaras aproximadas
3. ✅ Crear módulo `src_v2/visualization/pfs_analysis.py`
4. ✅ Escribir tests (29 nuevos tests)
5. ✅ Verificar compatibilidad (387 tests pasando)

### Nuevos Comandos CLI (19 total)

#### `pfs-analysis`
Analiza el Pulmonary Focus Score (PFS) del clasificador.

```bash
# Con máscaras reales del dataset
python -m src_v2 pfs-analysis \
    --checkpoint outputs/classifier/best.pt \
    --data-dir outputs/warped_dataset/test \
    --mask-dir data/dataset/COVID-19_Radiography_Dataset

# Con máscaras aproximadas (rectangulares)
python -m src_v2 pfs-analysis \
    --checkpoint outputs/classifier/best.pt \
    --data-dir outputs/warped_dataset/test \
    --approximate --margin 0.15 \
    --output-dir outputs/pfs_analysis
```

**Parámetros:**
| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `--checkpoint` | Path | Requerido | Checkpoint del clasificador |
| `--data-dir` | Path | Requerido | Directorio con imágenes |
| `--mask-dir` | Path | None | Directorio con máscaras pulmonares |
| `--output-dir` | Path | outputs/pfs_analysis | Directorio de salida |
| `--num-samples` | int | 50 | Total de muestras (divididas entre clases) |
| `--threshold` | float | 0.5 | Umbral PFS mínimo aceptable |
| `--approximate` | bool | False | Usar máscaras rectangulares |
| `--margin` | float | 0.15 | Margen para máscaras aproximadas |

**Salidas generadas:**
```
outputs/pfs_analysis/
├── pfs_summary.json          # Estadísticas globales
├── pfs_details.csv           # PFS por imagen
├── pfs_by_class.csv          # PFS promedio por clase
├── low_pfs_samples.csv       # Imágenes con PFS bajo
├── figures/
│   ├── pfs_distribution.png  # Histograma PFS
│   ├── pfs_by_class.png      # PFS por clase
│   ├── pfs_vs_confidence.png # Correlación PFS-confianza
│   └── pfs_correct_vs_incorrect.png
└── low_pfs_samples/          # GradCAM de casos problemáticos
```

#### `generate-lung-masks`
Genera máscaras pulmonares aproximadas.

```bash
python -m src_v2 generate-lung-masks \
    --data-dir outputs/warped_dataset \
    --output-dir outputs/lung_masks \
    --method rectangular \
    --margin 0.15
```

**Parámetros:**
| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `--data-dir` | Path | Requerido | Directorio del dataset |
| `--output-dir` | Path | Requerido | Directorio de salida |
| `--method` | str | rectangular | Método de generación |
| `--margin` | float | 0.15 | Margen para método rectangular |

### Nuevo Módulo: src_v2/visualization/pfs_analysis.py

```python
# Dataclasses
@dataclass
class PFSResult:
    image_path: str
    true_class: str
    predicted_class: str
    confidence: float
    pfs: float
    correct: bool

@dataclass
class PFSSummary:
    total_samples: int
    mean_pfs: float
    std_pfs: float
    median_pfs: float
    pfs_by_class: Dict[str, Dict[str, float]]
    pfs_correct_vs_incorrect: Dict[str, Dict[str, float]]
    low_pfs_count: int
    low_pfs_rate: float
    ...

# Clase principal
class PFSAnalyzer:
    def __init__(self, class_names, threshold=0.5): ...
    def add_result(self, result): ...
    def get_summary(self) -> PFSSummary: ...
    def get_low_pfs_results(self) -> List[PFSResult]: ...
    def save_reports(self, output_dir): ...

# Funciones auxiliares
load_lung_mask(mask_path) -> np.ndarray
find_mask_for_image(image_path, mask_dir, class_name) -> Path|None
generate_approximate_mask(image_shape, margin) -> np.ndarray
run_pfs_analysis(model, dataloader, class_names, ...) -> Tuple[PFSAnalyzer, List]
create_pfs_visualizations(results, output_dir, summary) -> Dict[str, Path]
save_low_pfs_gradcam_samples(results, output_dir, threshold, max_samples) -> int
```

## Métricas

### Tests
- **Tests previos:** 358
- **Tests nuevos:** 29
  - 16 tests unitarios para PFSAnalyzer, PFSResult, generate_approximate_mask
  - 13 tests CLI para pfs-analysis y generate-lung-masks
- **Tests totales:** 387 pasando

### Comandos CLI
- **Comandos previos:** 17
- **Comandos nuevos:** 2 (pfs-analysis, generate-lung-masks)
- **Comandos totales:** 19

### Archivos Modificados/Creados
```
src_v2/visualization/pfs_analysis.py   # NUEVO - 450 líneas
src_v2/visualization/__init__.py       # Actualizado exports
src_v2/cli.py                          # +500 líneas (2 comandos)
tests/test_visualization.py            # +240 líneas (16 tests)
tests/test_cli.py                      # +135 líneas (13 tests)
```

## Interpretación del PFS

El **Pulmonary Focus Score (PFS)** mide qué fracción de la atención del modelo
(según Grad-CAM) se concentra en las regiones pulmonares.

```
PFS = sum(heatmap * mask) / sum(heatmap)
```

**Umbrales de interpretación:**
- **PFS > 0.7:** Excelente - El modelo enfoca principalmente en pulmones
- **PFS 0.5-0.7:** Aceptable - Enfoque pulmonar satisfactorio
- **PFS < 0.5:** Preocupante - El modelo mira regiones no pulmonares

**Métricas calculadas:**
1. PFS promedio global con std
2. PFS por clase (COVID, Normal, Viral_Pneumonia)
3. PFS para predicciones correctas vs incorrectas
4. Correlación PFS vs confianza
5. Porcentaje de muestras con PFS bajo

## Máscaras Pulmonares Disponibles

El proyecto tiene 21,165 máscaras pulmonares en:
```
data/dataset/COVID-19_Radiography_Dataset/
├── COVID/masks/           # 3,616 máscaras
├── Normal/masks/          # 10,192 máscaras
├── Viral Pneumonia/masks/ # 1,345 máscaras
└── Lung_Opacity/masks/    # 6,012 máscaras
```

**Formato:** PNG 299x299, binario (0=fondo, 255=pulmón)

## Uso Típico

### 1. Análisis PFS con máscaras reales
```bash
python -m src_v2 pfs-analysis \
    --checkpoint outputs/arch_comparison/checkpoints/resnet18_warped_best.pt \
    --data-dir outputs/warped_dataset/test \
    --mask-dir data/dataset/COVID-19_Radiography_Dataset \
    --num-samples 100 \
    --threshold 0.5 \
    --output-dir outputs/pfs_resnet18
```

### 2. Análisis PFS con máscaras aproximadas
```bash
python -m src_v2 pfs-analysis \
    --checkpoint outputs/classifier/best.pt \
    --data-dir outputs/warped_dataset/test \
    --approximate \
    --margin 0.15 \
    --output-dir outputs/pfs_approximate
```

### 3. Generar máscaras para dataset sin ellas
```bash
python -m src_v2 generate-lung-masks \
    --data-dir outputs/external_dataset \
    --output-dir outputs/generated_masks \
    --method rectangular \
    --margin 0.15
```

## Investigación: Landmarks vs Máscaras

### Hallazgo 1: Son Conceptos Complementarios

El proyecto utiliza **DOS** sistemas de referencia pulmonar que son complementarios:

| Aspecto | Landmarks (Manual) | Máscaras (Dataset) |
|---------|-------------------|-------------------|
| **Origen** | Anotados manualmente | COVID-19_Radiography_Dataset |
| **Formato** | 15 puntos (x,y) en CSV/JSON | PNG binario 299x299 |
| **Propósito** | Warping geométrico | Segmentación/PFS |
| **Precisión** | Contorno simplificado | Segmentación detallada |
| **Cantidad** | ~650 imágenes anotadas | 21,165 máscaras |

**Conclusión:** Los landmarks definen el CONTORNO para warping, las máscaras definen la REGIÓN para PFS.
Ambos son válidos para sus propósitos.

### Hallazgo 2: Limitación Crítica - Máscaras NO Warped

⚠️ **ADVERTENCIA IMPORTANTE:**

Las máscaras pulmonares del dataset COVID-19_Radiography están en coordenadas de las
imágenes **ORIGINALES**, no de las imágenes **WARPED**.

```
Imagen Original (299x299) ──[Warping]──> Imagen Warped (224x224)
        ↓                                        ↓
  Máscara Original                      ⚠️ Máscara NO transformada
  (geometría correcta)                  (geometría desalineada)
```

**Impacto:**
- Para imágenes ORIGINALES: PFS es preciso
- Para imágenes WARPED: PFS tiene desalineación geométrica
- Los resultados previos (~35% PFS) pueden estar afectados por esta limitación

**Mitigación implementada:**
- Se agregó advertencia automática cuando `data-dir` contiene "warped"
- Para PFS preciso, se recomienda usar imágenes originales

### Hallazgo 3: Máscaras del Dataset Son Válidas

Las 21,165 máscaras en `COVID-19_Radiography_Dataset/*/masks/` son:
- Segmentaciones profesionales de radiografías reales
- Formato correcto (PNG binario)
- Apropiadas para calcular PFS en imágenes originales

## Bugs Encontrados y Corregidos

### BUG 1 (CRÍTICO): max_per_class incorrecto
**Archivo:** `src_v2/cli.py:5496`
**Problema:** `max_per_class = num_samples` procesaba 3x más imágenes de las solicitadas
**Corrección:**
```python
# ANTES (incorrecto):
max_per_class = num_samples  # Si num_samples=50, procesaba 150 total

# DESPUÉS (correcto):
max_per_class = max(1, num_samples // len(class_names))  # Procesa ~50 total
```

### BUG 2: find_mask_for_image sin sufijo _mask
**Archivo:** `src_v2/visualization/pfs_analysis.py:284-295`
**Problema:** No buscaba archivos con sufijo `_mask.png` generados por `generate-lung-masks`
**Corrección:** Añadidas rutas con sufijo `_mask.png` a `possible_paths`

### BUG 3: Progress bar con total incorrecto
**Archivo:** `src_v2/cli.py:5523`
**Problema:** `total=num_samples * len(class_names)` no coincidía con samples reales
**Corrección:** `total=max_per_class * len(class_names)` (expected_total)

### Mejora: Advertencia para imágenes warped
**Archivo:** `src_v2/cli.py:5489-5503`
**Cambio:** Agregada advertencia automática cuando se detectan imágenes warped

## Decisión: PFS Pospuesto

**Fecha decisión:** 2025-12-09

Después de análisis exhaustivo con múltiples agentes, se decidió **posponer** el trabajo adicional en PFS:

### Razones:
1. **Hipótesis principal ya demostrada** sin PFS (11x generalización, 30x robustez)
2. **Limitación técnica** requiere implementar `warp_mask()` (~2-3 horas)
3. **PFS no existe en literatura** - requeriría justificación adicional en tesis
4. **Resultados actuales inválidos** para comparación warped vs original

### Documentación:
- Ver: `docs/TRABAJO_FUTURO_PFS.md` para detalles completos
- Checklist de implementación incluido para retomar cuando sea conveniente

## Próximos Pasos (Sesión 25+)

1. **Comando `optimize-margin`** (PRIORITARIO)
   - Buscar margen óptimo para warping automáticamente
   - Último comando pendiente para ~95% cobertura CLI
   - Basado en: `scripts/margin_optimization_experiment.py`

2. **Trabajo futuro opcional:**
   - Implementar `warp_mask()` para PFS válido
   - Reporte de interpretabilidad con métricas de literatura
   - API REST con FastAPI

## Archivos Clave

```
src_v2/visualization/pfs_analysis.py    # Módulo PFS
src_v2/visualization/gradcam.py         # calculate_pfs()
src_v2/cli.py                           # Comandos CLI
tests/test_visualization.py             # Tests unitarios
tests/test_cli.py                       # Tests CLI
```

---

**Comandos CLI Implementados (19 total):**

| # | Comando | Descripción | Sesión |
|---|---------|-------------|--------|
| 1 | train | Entrenar modelo de landmarks | - |
| 2 | evaluate | Evaluar modelo individual | - |
| 3 | predict | Predecir landmarks en imagen | - |
| 4 | warp | Aplicar warping geométrico | - |
| 5 | version | Mostrar versión | - |
| 6 | evaluate-ensemble | Evaluar ensemble | - |
| 7 | classify | Clasificar imagen | - |
| 8 | train-classifier | Entrenar clasificador | - |
| 9 | evaluate-classifier | Evaluar clasificador | - |
| 10 | cross-evaluate | Evaluación cruzada | 18 |
| 11 | evaluate-external | Validación externa | 18 |
| 12 | test-robustness | Pruebas de robustez | 18 |
| 13 | compute-canonical | Calcular forma canónica | 20 |
| 14 | generate-dataset | Generar dataset warped | 20 |
| 15 | compare-architectures | Comparar arquitecturas | 22 |
| 16 | gradcam | Visualizaciones Grad-CAM | 23 |
| 17 | analyze-errors | Análisis de errores | 23 |
| 18 | **pfs-analysis** | Análisis PFS | **24** |
| 19 | **generate-lung-masks** | Generar máscaras | **24** |
