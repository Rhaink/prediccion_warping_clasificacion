# Analisis de Gaps: Version Original vs CLI

**Fecha:** 2025-12-08
**Ultima actualizacion:** 2025-12-09 (Sesion 22)
**Objetivo:** Identificar funcionalidades faltantes en la version CLI para reproducir todos los experimentos originales.

## Resumen Ejecutivo

| Categoria | Scripts Originales | En CLI | Gap |
|-----------|-------------------|--------|-----|
| Entrenamiento | 9 | 2 | 7 |
| Evaluacion | 18 | 5 | 13 |
| Procesamiento | 12 | 3 | 9 |
| Visualizacion | 20 | 0 | 20 |
| Analisis | 6 | 0 | 6 |
| Prediccion | 4 | 2 | 2 |
| **TOTAL** | **69** | **12** | **57** |

> **Nota Sesion 20:** Se agregaron 2 comandos de procesamiento (`compute-canonical`, `generate-dataset`)

## 1. Comandos CLI Actuales (15 total)

### Landmarks (5 comandos)
- `train` - Entrenar modelo de landmarks
- `evaluate` - Evaluar modelo individual
- `evaluate-ensemble` - Evaluar ensemble de modelos
- `predict` - Predecir en imagen individual
- `warp` - Aplicar warping a dataset

### Clasificacion (7 comandos)
- `classify` - Clasificar imagenes
- `train-classifier` - Entrenar clasificador
- `evaluate-classifier` - Evaluar clasificador
- `cross-evaluate` - Evaluacion cruzada de generalizacion
- `evaluate-external` - Validacion externa (FedCOVIDx)
- `test-robustness` - Pruebas de robustez
- `compare-architectures` - Comparar multiples arquitecturas **NUEVO Sesion 22**

### Procesamiento (2 comandos)
- `compute-canonical` - Calcular forma canonica con GPA
- `generate-dataset` - Generar dataset warped con splits

### Utilidad (1 comando)
- `version` - Mostrar version

## 2. Funcionalidades Faltantes por Prioridad

### PRIORIDAD ALTA - Reproducir Experimentos Clave

#### 2.1 Generacion de Datasets Warped ✅ COMPLETADO (Sesion 20)
**Scripts originales:**
- `scripts/generate_warped_dataset.py`
- `scripts/generate_full_warped_dataset.py`
- `scripts/session31_generate_dataset_margin125.py`

**Comando CLI implementado:** `generate-dataset`
```bash
python -m src_v2 generate-dataset \
    data/COVID-19_Radiography_Dataset \
    outputs/warped_dataset \
    --checkpoint checkpoints_v2/final_model.pt \
    --margin 1.05 \
    --splits 0.75,0.125,0.125
```

**Estado:** Implementado en `src_v2/cli.py` y `src_v2/processing/warp.py`

#### 2.2 Comparacion Multi-Arquitectura ✅ COMPLETADO (Sesion 22)
**Scripts originales:**
- `scripts/train_all_architectures.py`
- `scripts/compare_classifiers.py`
- `scripts/session31_train_multi_arch.py`

**Comando CLI implementado:** `compare-architectures`
```bash
python -m src_v2 compare-architectures outputs/full_warped_dataset \
    --architectures resnet18,efficientnet_b0,densenet121,mobilenet_v2 \
    --output-dir outputs/arch_comparison \
    --epochs 30 --seed 42
```

**Estado:** Implementado en Sesion 22. Incluye:
- Entrenamiento secuencial de 7 arquitecturas
- Generacion de reportes (JSON/CSV)
- Visualizaciones (accuracy, F1, confusion matrices, training curves)
- Soporte para comparar warped vs original
- Modo rapido (`--quick`) para pruebas
- 9 bugs corregidos, 15 tests nuevos

#### 2.3 Analisis GPA (Generalized Procrustes Analysis) ✅ COMPLETADO (Sesion 20)
**Scripts originales:**
- `scripts/gpa_analysis.py`
- `scripts/verify_gpa_correctness.py`
- `scripts/visualize_gpa_methodology.py`

**Comando CLI implementado:** `compute-canonical`
```bash
python -m src_v2 compute-canonical \
    data/coordenadas/coordenadas_maestro.csv \
    --output-dir outputs/shape_analysis \
    --visualize
```

**Estado:** Implementado en `src_v2/cli.py` y `src_v2/processing/gpa.py`

### PRIORIDAD MEDIA - Analisis y Visualizacion

#### 2.4 Grad-CAM y Explicabilidad
**Scripts originales:**
- `scripts/gradcam_comparison.py`
- `scripts/gradcam_multi_architecture.py`
- `scripts/gradcam_pfs_analysis.py`

**Comando CLI propuesto:** `gradcam`
```bash
python -m src_v2 gradcam \
    --checkpoint outputs/classifier/best.pt \
    --image test_image.png \
    --output gradcam_visualization.png
```

**Justificacion:** Explicabilidad es critica para aplicaciones clinicas.

#### 2.5 Analisis de Errores
**Scripts originales:**
- `scripts/session30_error_analysis.py`

**Comando CLI propuesto:** `analyze-errors`
```bash
python -m src_v2 analyze-errors \
    --checkpoint outputs/classifier/best.pt \
    --data-dir outputs/full_warped_dataset \
    --output-dir outputs/error_analysis
```

**Justificacion:** Entender donde falla el modelo es importante para mejoras.

#### 2.6 Experimentos de Margenes
**Scripts originales:**
- `scripts/margin_optimization_experiment.py`
- `scripts/experiment_extended_margins.py`

**Comando CLI propuesto:** `optimize-margin`
```bash
python -m src_v2 optimize-margin \
    --data-dir data/COVID-19_Radiography_Dataset \
    --margins 1.00,1.05,1.10,1.15,1.20,1.25,1.30 \
    --output outputs/margin_optimization
```

**Justificacion:** Margin 1.25 demostro ser optimo. Automatizar busqueda.

### PRIORIDAD BAJA - Visualizacion para Tesis

#### 2.7 Generacion de Figuras
**Scripts originales:** 20 scripts en `scripts/visualization/`
- `generate_bloque1_*.py` hasta `generate_bloque8_*.py`
- `generate_architecture_diagrams.py`
- `generate_results_figures.py`

**Recomendacion:** NO migrar al CLI. Son scripts one-shot para generacion de figuras de tesis. Mantener en `/scripts/visualization/` como referencia.

#### 2.8 Validacion de Integridad de Datos
**Scripts originales:**
- `scripts/verify_data_leakage.py`
- `scripts/verify_val_vs_test.py`
- `scripts/verify_no_tta.py`

**Recomendacion:** Incorporar como tests automaticos en `tests/` en lugar de comandos CLI.

## 3. Arquitecturas en Clasificador ✅ COMPLETADO (Sesion 20)

### Estado Actual (7 arquitecturas soportadas)
El comando `train-classifier` soporta:
- ResNet-18 ✅
- ResNet-50 ✅ **NUEVO**
- EfficientNet-B0 ✅
- DenseNet-121 ✅
- AlexNet ✅ **NUEVO**
- VGG-16 ✅ **NUEVO**
- MobileNetV2 ✅ **NUEVO**

**Estado:** Todas las arquitecturas usadas en experimentos originales estan disponibles en `src_v2/models/classifier.py`.

## 4. Funcionalidades de Investigacion Pendientes

### 4.1 Domain Adaptation
**Problema:** FedCOVIDx muestra ~50% accuracy (domain shift)
**Solucion propuesta:** Comando `domain-adapt`
**Tecnicas:** DANN, CORAL, self-training

### 4.2 Modelo de 4 Clases
**Problema:** Dataset tiene Lung_Opacity (12,024 imagenes no usadas)
**Solucion propuesta:** Flag `--num-classes 4` en train-classifier

### 4.3 API REST
**Problema:** CLI no es deployable en produccion clinica
**Solucion propuesta:** FastAPI wrapper sobre comandos existentes

## 5. Plan de Implementacion Propuesto

### Fase 1: Completar Pipeline Basico ✅ COMPLETADO (Sesion 20)
1. [x] `generate-dataset` - Crear datasets warped completos
2. [x] `compute-canonical` - Calcular forma canonica con GPA
3. [x] Agregar arquitecturas faltantes (AlexNet, ResNet-50, MobileNetV2, VGG-16)

### Fase 2: Analisis y Comparacion (1-2 sesiones) - PENDIENTE
4. [ ] `compare-architectures` - Comparacion sistematica
5. [ ] `analyze-errors` - Analisis de errores de clasificacion
6. [ ] `optimize-margin` - Busqueda de margen optimo

### Fase 3: Explicabilidad (1 sesion) - PENDIENTE
7. [ ] `gradcam` - Visualizacion de atencion
8. [ ] Integrar Pulmonary Focus Score (PFS)

### Fase 4: Produccion (futuro)
9. [ ] API REST con FastAPI
10. [ ] Domain adaptation para datasets externos
11. [ ] Modelo de 4 clases

## 6. Scripts que NO Necesitan Migracion

Los siguientes scripts son especificos de sesiones historicas y no necesitan ser comandos CLI:

| Script | Razon |
|--------|-------|
| `session30_*.py` | Experimentos historicos documentados |
| `validation_session26*.py` | Validaciones one-shot |
| `train_baseline_original_15k.py` | Caso especifico |
| `debug_*.py` | Herramientas de desarrollo |
| `test_*.py` en scripts/ | Deberian ser tests formales |
| `visualization/*.py` | Generacion de figuras de tesis |

## 7. Matriz de Reproducibilidad

| Experimento Original | Reproducible via CLI | Gap |
|---------------------|---------------------|-----|
| Entrenar landmarks | ✅ `train` | - |
| Evaluar ensemble 3.71px | ✅ `evaluate-ensemble` | - |
| Generar dataset warped | ✅ `generate-dataset` | - |
| Clasificar warped | ✅ `train-classifier` + `evaluate-classifier` | - |
| Cross-evaluation | ✅ `cross-evaluate` | - |
| Robustez a perturbaciones | ✅ `test-robustness` | - |
| Validacion externa | ✅ `evaluate-external` | - |
| Comparar 7 arquitecturas | ✅ `compare-architectures` | - |
| Grad-CAM | ❌ | `gradcam` |
| GPA forma canonica | ✅ `compute-canonical` | - |
| Optimizacion de margenes | ❌ | `optimize-margin` |

## 8. Conclusiones

### Funcionalidades Criticas Completadas
1. ~~**`generate-dataset`** - Crear datasets warped completos~~ ✅ (Sesion 20)
2. ~~**`compute-canonical`** - Calcular forma canonica~~ ✅ (Sesion 20)
3. ~~**Arquitecturas adicionales** - AlexNet, ResNet-50, MobileNetV2, VGG-16~~ ✅ (Sesion 20)
4. ~~**`compare-architectures`** - Comparacion multi-arquitectura~~ ✅ (Sesion 22)

### Funcionalidades Criticas Pendientes (1)
1. **`gradcam`** - Explicabilidad con Grad-CAM

### Funcionalidades Deseables (3)
1. **`analyze-errors`** - Analisis de errores
2. **`optimize-margin`** - Busqueda de margen optimo
3. **API REST** - Despliegue en produccion

### Estado del Proyecto (Actualizado Sesion 22)
- **CLI actual:** 15 comandos funcionales y validados
- **Cobertura de experimentos:** ~85% reproducibles via CLI
- **Tests:** 327 pasando (+15 nuevos en Sesion 22)
- **Documentacion:** 22 sesiones documentadas
- **Arquitecturas clasificador:** 7 de 7 completas
- **Bugs corregidos:** 9 en Sesion 22

### Recomendacion Final
Implementar `gradcam` y `analyze-errors` en Sesion 23 para alcanzar ~95% de reproducibilidad y completar explicabilidad clinica.
