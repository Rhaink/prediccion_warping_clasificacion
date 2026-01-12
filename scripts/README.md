# Scripts del Proyecto

> **Recomendacion**: Usar el CLI `python -m src_v2` en lugar de ejecutar scripts directamente.
> Los scripts aqui son principalmente para desarrollo, investigacion y generacion de figuras.

## Estructura

```
scripts/
├── archive/           # Scripts archivados (debug, sesiones antiguas, tests)
├── visualization/     # Scripts de visualización para tesis
└── *.py              # Scripts de utilidad activos
```

## Scripts de Produccion (Activos)

| Script | Proposito | CLI Equivalente |
|--------|-----------|-----------------|
| `predict.py` | Prediccion de landmarks | `python -m src_v2 predict` |
| `train.py` | Entrenamiento de modelos | `python -m src_v2 train` |
| `train_classifier.py` | Entrenamiento clasificador | `python -m src_v2 train-classifier` |
| `train_classifier_original.py` | Clasificador en imagenes originales | - |
| `train_hierarchical.py` | Entrenar modelo jerarquico | - |
| `evaluate_ensemble_from_config.py` | Evaluar ensemble desde config | - |
| `verify_individual_models.py` | Verificacion de modelos | - |
| `create_thesis_figures.py` | Figuras para tesis | - |
| `generate_thesis_figure.py` | Figura trade-off fill rate | - |

## Scripts de Automatizacion

| Script | Proposito |
|--------|-----------|
| `run_seed_sweep.sh` | Entrenar seeds arbitrarios y correr opcion 1 + opcion 2 |
| `run_best_ensemble.sh` | Evaluar el mejor ensemble actual |
| `quickstart_landmarks.sh` | Entrenar + evaluar + extraer predicciones (hasta pre-warping) |

Nota: `run_repro_split_ensemble.sh` y `run_option1_new_seeds.sh` fueron movidos a
`scripts/archive/` (reemplazados por `run_seed_sweep.sh`).

## Configs (opcion 3)

Usa JSONs en `configs/` para estandarizar hiperparametros y evitar copias manuales:

```bash
python scripts/train.py --config configs/landmarks_train_base.json \
  --seed 123 --split-seed 123 --save-dir checkpoints/... --output-dir outputs/...
```

Para sweeps con config:
```bash
TRAIN_CONFIG=configs/landmarks_train_base.json bash scripts/run_seed_sweep.sh 333 444
```

Predicciones del ensemble desde config:
```bash
python scripts/extract_predictions.py --config configs/ensemble_best.json \
  --output-dir outputs/predictions_best
```

Clasificador warpeado:
```bash
python scripts/train_classifier.py --config configs/classifier_warped_base.json
```

Clasificador original:
```bash
python scripts/train_classifier_original.py --config configs/classifier_original_base.json
```

Modelo jerarquico:
```bash
python scripts/train_hierarchical.py --config configs/hierarchical_train_base.json
```

## Scripts de Visualizacion

Ubicacion: `visualization/`

| Script | Proposito |
|--------|-----------|
| `generate_bloque*_*.py` | Figuras por bloque de tesis |
| `generate_results_figures.py` | Figuras de resultados |
| `generate_architecture_diagrams.py` | Diagramas de arquitectura |

## Scripts de Generacion de Datasets

| Script | Proposito |
|--------|-----------|
| `generate_warped_dataset.py` | Dataset warpeado (47% fill) |
| `generate_full_warped_dataset.py` | Dataset completo warpeado |
| `generate_original_cropped_47.py` | Dataset control (47% fill) |
| `filter_dataset_3_classes.py` | Filtrado a 3 clases |

## Scripts de Evaluacion

| Script | Proposito |
|--------|-----------|
| `compare_classifiers.py` | Comparar clasificadores |
| `evaluate_external_baseline.py` | Evaluacion en FedCOVIDx |
| `evaluate_external_warped.py` | Evaluacion externa warpeada |
| `extract_predictions.py` | Predicciones del ensemble + triangulacion (test split) |
| `gradcam_*.py` | Analisis Grad-CAM |
| `calculate_pfs_warped.py` | Calculo PFS en warped |

## Scripts de Verificacion

| Script | Proposito |
|--------|-----------|
| `verify_data_leakage.py` | Verificar data leakage |
| `verify_canonical_delaunay.py` | Verificar triangulacion |
| `verify_gpa_correctness.py` | Verificar GPA |
| `verify_no_tta.py` | Verificar sin TTA |

## Directorio archive/

Scripts archivados de sesiones anteriores (19 scripts):

- `debug_*.py` - Scripts de debugging temporales
- `session30_*.py`, `session31_*.py` - Scripts de sesiones especificas
- `validation_session*.py` - Validaciones de sesiones antiguas
- `experiment_*.py` - Experimentos puntuales
- `test_*.py` - Tests que deberian estar en `tests/`

Estos se mantienen por referencia historica pero no se usan directamente.

Nota: `archive/invalid_warping/generate_warped_dataset_full_coverage.py` fue movido
por generar datasets con warping incorrecto (ver docs/reportes/REPORTE_VERIFICACION_DESCUBRIMIENTOS_GEOMETRICOS.md).

## Fuente de Verdad

Los valores experimentales validados estan en `GROUND_TRUTH.json` (v2.1.0).
Cualquier script de visualizacion debe usar valores de este archivo.

### Valores Clave (Sesion 53 + sweeps)

| Metrica | Valor |
|---------|-------|
| Error ensemble best (seed666 combo) | **3.61 px** |
| Error ensemble 4 + TTA | **3.71 px** |
| Error best individual (seed 456) | **4.04 px** |
| Accuracy warped_96 (RECOMENDADO) | **99.10%** |
| Accuracy warped_99 | 98.73% |
| Robustez JPEG Q50 (warped_47) | **30x** mejor que original |
| Robustez JPEG Q50 (warped_96) | 3.06% degradacion |

### Clasificador Recomendado

**warped_96** es el punto optimo entre accuracy y robustez:
- Mejor accuracy: 99.10%
- Buena robustez: 3.06% degradacion JPEG Q50
- 2.4x mas robusto que warped_99

Ver `docs/sesiones/SESION_53_FILL_RATE_TRADEOFF.md` para analisis completo.
