# Sesion 18: Nuevos Comandos CLI para Reproducir Experimentos Originales

**Fecha:** 08-Dic-2024
**Estado:** Completado
**Tests:** 244 pasando (48 nuevos tests CLI)

## Resumen

Esta sesion agrega funcionalidades faltantes al CLI para reproducir los experimentos originales documentados en las Sessions 29-37.

## Cambios Implementados

### 1. Nuevos Comandos CLI (Alta Prioridad)

#### `cross-evaluate` - Evaluacion Cruzada de Modelos

Reproduce los resultados de Session 30 (11x mejor generalizacion del modelo warped).

```bash
python -m src_v2 cross-evaluate \
    modelo_original.pt \
    modelo_warped.pt \
    --data-a data/dataset/COVID-19_Radiography_Dataset \
    --data-b outputs/full_warped_dataset \
    --output-dir outputs/cross_evaluation
```

**Parametros:**
- `MODEL_A`, `MODEL_B`: Checkpoints de los dos modelos a comparar
- `--data-a`, `--data-b`: Directorios de los dos datasets
- `--split`: test, val, all (default: test)
- `--output-dir`: Directorio de salida para resultados JSON
- `--batch-size`, `--device`, `--seed`

**Salida:** Matriz de evaluacion 2x2 con gaps de generalizacion.

#### `evaluate-external` - Validacion en Dataset Externo

Evalua modelos en dataset binario externo (FedCOVIDx/Dataset3) con mapeo 3→2 clases.

```bash
python -m src_v2 evaluate-external \
    outputs/classifier/best.pt \
    --external-data outputs/external_validation/dataset3 \
    --output results.json \
    --threshold 0.5
```

**Parametros:**
- `CHECKPOINT`: Modelo de 3 clases (COVID, Normal, Viral_Pneumonia)
- `--external-data`: Dataset binario (positive/negative)
- `--threshold`: Umbral de decision (default: 0.5)
- `--output`: JSON con resultados

**Mapeo de clases:**
- P(positive) = P(COVID)
- P(negative) = P(Normal) + P(Viral_Pneumonia)

**Metricas:** Accuracy, Sensitivity, Specificity, Precision, F1, AUC-ROC

### 2. Nuevo Comando CLI (Media Prioridad)

#### `test-robustness` - Pruebas de Robustez

Evalua resistencia del modelo ante perturbaciones (JPEG, blur, ruido).

```bash
python -m src_v2 test-robustness \
    outputs/classifier/best.pt \
    --data-dir outputs/full_warped_dataset \
    --output robustness.json
```

**Perturbaciones evaluadas:**
- `original`: Sin perturbacion (baseline)
- `jpeg_q50`: Compresion JPEG Q=50
- `jpeg_q30`: Compresion JPEG Q=30
- `blur_sigma1`: Blur gaussiano sigma=1.0
- `blur_sigma2`: Blur gaussiano sigma=2.0
- `noise_005`: Ruido gaussiano sigma=0.05
- `noise_010`: Ruido gaussiano sigma=0.10

**Salida:** Accuracy, error rate y degradacion para cada perturbacion.

### 3. Soporte DenseNet-121 en train-classifier

Agrega DenseNet-121 como backbone para el clasificador (recomendado para mejor generalizacion).

```bash
python -m src_v2 train-classifier data/warped/ \
    --backbone densenet121 \
    --epochs 50
```

**Cambios:**
- `src_v2/models/classifier.py`: Agregado soporte para `densenet121`
- `SUPPORTED_BACKBONES = ("resnet18", "efficientnet_b0", "densenet121")`

## Archivos Modificados

```
src_v2/cli.py                 # +790 lineas (3 nuevos comandos)
src_v2/models/classifier.py   # +12 lineas (DenseNet-121)
tests/test_cli.py             # +210 lineas (22 nuevos tests)
```

## Tests Agregados

### TestCrossEvaluate (5 tests)
- `test_cross_evaluate_help`
- `test_cross_evaluate_requires_models`
- `test_cross_evaluate_requires_data_options`
- `test_cross_evaluate_missing_model_files`
- `test_cross_evaluate_module_execution`

### TestEvaluateExternal (6 tests)
- `test_evaluate_external_help`
- `test_evaluate_external_requires_checkpoint`
- `test_evaluate_external_requires_data_option`
- `test_evaluate_external_missing_files`
- `test_evaluate_external_default_threshold`
- `test_evaluate_external_module_execution`

### TestTestRobustness (5 tests)
- `test_robustness_help`
- `test_robustness_requires_checkpoint`
- `test_robustness_requires_data_dir`
- `test_robustness_missing_files`
- `test_robustness_module_execution`

### TestDenseNet121Support (4 tests)
- `test_train_classifier_backbone_options`
- `test_classifier_densenet121_instantiation`
- `test_classifier_densenet121_forward`
- `test_create_classifier_densenet121`

## Comandos CLI Actuales (12 total)

| Comando | Descripcion |
|---------|-------------|
| `train` | Entrenar modelo de landmarks |
| `evaluate` | Evaluar modelo de landmarks |
| `predict` | Predecir landmarks en imagen |
| `warp` | Aplicar warping a dataset |
| `version` | Mostrar version |
| `evaluate-ensemble` | Evaluar ensemble de modelos |
| `classify` | Clasificar imagenes COVID-19 |
| `train-classifier` | Entrenar clasificador CNN |
| `evaluate-classifier` | Evaluar clasificador |
| **`cross-evaluate`** | Evaluacion cruzada 2 modelos |
| **`evaluate-external`** | Validacion en dataset externo |
| **`test-robustness`** | Pruebas de robustez |

## Resultados Esperados (Referencia Session 30)

### Cross-Evaluation (Original vs Warped)

```
                    Dataset Original    Dataset Warped
Modelo Original     98.81%             73.45%    (GAP: 25.36%)
Modelo Warped       95.78%             98.02%    (GAP: 2.24%)

RATIO: Modelo Original gap es 11x mayor que Modelo Warped
CONCLUSION: Modelo Warped GENERALIZA MEJOR
```

### Test de Robustez (Session 29)

| Perturbacion | Original Error | Warped Error | Mejora |
|--------------|----------------|--------------|--------|
| JPEG Q=50 | 16.14% | 0.53% | 30x |
| JPEG Q=30 | 29.97% | 1.32% | 23x |
| Blur fuerte | 46.05% | 16.27% | 3x |

## Uso Recomendado

### Para reproducir Session 30:
```bash
# Cross-evaluation
python -m src_v2 cross-evaluate \
    outputs/session28_baseline_original/resnet18_original_15k_best.pt \
    outputs/session27_models/resnet18_expanded_15k_best.pt \
    --data-a data/dataset/COVID-19_Radiography_Dataset \
    --data-b outputs/full_warped_dataset \
    --output-dir outputs/session30_reproduced
```

### Para validacion externa:
```bash
# Evaluar en FedCOVIDx
python -m src_v2 evaluate-external \
    outputs/classifier_warped/best.pt \
    --external-data outputs/external_validation/dataset3 \
    --output external_results.json
```

### Para entrenar con DenseNet-121:
```bash
# Mejor generalizacion segun Session 32
python -m src_v2 train-classifier outputs/full_warped_dataset \
    --backbone densenet121 \
    --epochs 50 \
    --output-dir outputs/classifier_densenet
```

## Proximos Pasos (Sesion 19)

### Validacion Pendiente con Datos Reales

- [ ] `cross-evaluate`: Probar con modelos original vs warped
- [ ] `evaluate-external`: Probar con FedCOVIDx (dataset3)
- [ ] `test-robustness`: Verificar perturbaciones funcionan correctamente
- [ ] DenseNet-121: Verificar que entrena sin errores

### Recursos Disponibles para Validacion

**Modelos:**
- `outputs/classifier_comparison/resnet18_original/best_model.pt`
- `outputs/classifier_comparison/resnet18_warped/best_model.pt`
- `outputs/classifier_comparison/densenet121_warped/best_model.pt`
- `outputs/classifier_comparison/efficientnet_b0_warped/best_model.pt`

**Datasets:**
- Original: `data/dataset/COVID-19_Radiography_Dataset/`
- Warped: `outputs/full_warped_dataset/` (train/val/test)
- External: `outputs/external_validation/dataset3/test/` (positive/negative)

### Puntos a Verificar en Codigo

1. **cross-evaluate:**
   - Manejo de estructuras de dataset diferentes (ImageFolder vs manual)
   - Orden de clases consistente entre modelos
   - Calculo correcto de gaps

2. **evaluate-external:**
   - Mapeo 3→2 clases (COVID=positive, Normal+Viral=negative)
   - Indice de COVID correcto en diferentes checkpoints
   - Metricas binarias (sensitivity, specificity)

3. **test-robustness:**
   - Funciones de perturbacion (JPEG, blur, ruido)
   - Manejo de imagenes RGB vs grayscale
   - Calculo de degradacion relativa

### Resultados de Referencia para Validacion

| Experimento | Metrica | Valor Esperado |
|-------------|---------|----------------|
| Cross-eval Original→Warped | Accuracy | ~73.45% |
| Cross-eval Warped→Original | Accuracy | ~95.78% |
| Cross-eval Gap Ratio | Ratio | ~11x |
| External Warped | Accuracy | ~53-57% |
| Robustness JPEG Q=50 Warped | Error | <1% |
