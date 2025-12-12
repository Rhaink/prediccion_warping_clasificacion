# Scripts del Proyecto

> **Recomendacion**: Usar el CLI `covid-landmarks` en lugar de ejecutar scripts directamente.
> Los scripts aqui son principalmente para desarrollo, investigacion y generacion de figuras.

## Scripts de Produccion (Activos)

| Script | Proposito | CLI Equivalente |
|--------|-----------|-----------------|
| `predict.py` | Prediccion de landmarks | `covid-landmarks predict` |
| `evaluate_ensemble.py` | Evaluacion del ensemble | `covid-landmarks evaluate` |
| `train.py` | Entrenamiento de modelos | `covid-landmarks train` |
| `verify_individual_models.py` | Verificacion de modelos | - |
| `create_thesis_figures.py` | Figuras GradCAM para tesis | - |

## Scripts de Visualizacion

Ubicacion: `visualization/`

| Script | Proposito | Sesion |
|--------|-----------|--------|
| `generate_bloque1_*.py` | Figuras introduccion | - |
| `generate_bloque2_*.py` | Metodologia y datos | - |
| `generate_bloque3_*.py` | Preprocesamiento | - |
| `generate_bloque4_*.py` | Arquitectura | - |
| `generate_bloque5_*.py` | Ensemble y TTA | - |
| `generate_bloque6_*.py` | Resultados | - |
| `generate_bloque7_*.py` | Evidencia visual | - |
| `generate_bloque8_*.py` | Conclusiones | - |
| `generate_results_figures.py` | Figuras de resultados | - |
| `generate_architecture_diagrams.py` | Diagramas de arquitectura | - |

## Scripts Historicos

Estos scripts documentan el desarrollo experimental pero no se usan activamente.
Considere moverlos a `legacy/` si se necesita limpiar el directorio.

### Funcionalidad Duplicada en src_v2/
- `gpa_analysis.py` -> `src_v2/processing/gpa.py`
- `piecewise_affine_warp.py` -> `src_v2/processing/warp.py`
- `landmark_connections.py` -> `src_v2/constants.py`

### Scripts de Sesiones
- `session30_*.py` - Cross-evaluation y analisis de errores
- `session31_*.py` - Evaluacion multi-arquitectura
- `validation_session*.py` - Validaciones historicas

### Scripts de Generacion de Datasets
- `generate_warped_dataset.py` - Dataset warpeado (957 imagenes)
- `generate_full_warped_dataset.py` - Dataset con cobertura completa
- `filter_dataset_3_classes.py` - Filtrado a 3 clases

### Scripts de Evaluacion Externa
- `evaluate_external_baseline.py` - Evaluacion en FedCOVIDx
- `warp_dataset3.py` - Warping de Dataset3

## Fuente de Verdad

Los valores experimentales validados estan en `GROUND_TRUTH.json`.
Cualquier script de visualizacion debe usar valores de este archivo.

### Valores Clave (Sesion 50)
- Error ensemble 4 + TTA: **3.71 px**
- Error best individual (seed 456): **4.04 px**
- Ensemble 2 modelos: **3.79 px**
- Accuracy clasificacion: **98.73%**
- Robustez JPEG Q50: **30.45x** mejor que original
