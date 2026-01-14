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
| `predict_landmarks_dataset.py` | Cache de landmarks para dataset completo | - |
| `train.py` | Entrenamiento de modelos | `python -m src_v2 train` |
| `train_classifier.py` | Wrapper del CLI (clasificador) | `python -m src_v2 train-classifier` |
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
| `quickstart_warping.sh` | Pipeline warping con cache de landmarks + dataset warped |
| `run_classifier_sweep_accuracy.sh` | Sweep de accuracy para clasificador CNN |

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

Clasificador warpeado (CLI canonico):
```bash
python -m src_v2 train-classifier --config configs/classifier_warped_base.json
```

Wrapper equivalente:
```bash
python scripts/train_classifier.py --config configs/classifier_warped_base.json
```

Clasificador original (legacy, archivado):
```bash
python scripts/archive/classification/train_classifier_original.py --config configs/classifier_original_base.json
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
| `predict_landmarks_dataset.py` | Cache de landmarks para warping (recomendado) |
| `generate_warped_dataset.py` | Legacy (Sesion 21): warping con GT, fill ~47% |
| `generate_full_warped_dataset.py` | Legacy (Sesion 25): warping inline, sin cache |

Nota: el flujo actual usa `python -m src_v2 generate-dataset` con `--predictions`
o `--config configs/warping_best.json` (ver `docs/QUICKSTART_WARPING.md`).

Nota: `generate_original_cropped_47.py` y `filter_dataset_3_classes.py` fueron movidos a
`scripts/archive/classification/`.

## Scripts de Evaluacion

| Script | Proposito |
|--------|-----------|
| `extract_predictions.py` | Predicciones del ensemble + triangulacion (test split) |
| `calculate_pfs_warped.py` | Calculo PFS en warped |

Nota: `compare_classifiers.py` y `gradcam_*.py` fueron movidos a
`scripts/archive/classification/`. Para Grad-CAM usar el CLI:
`python -m src_v2 gradcam`.

Nota: la pipeline de validacion externa (`evaluate_external_*.py`,
`analyze_class_mapping.py`, `prepare_dataset3.py`, `warp_dataset3.py`) fue movida a
`scripts/archive/classification/`.

## Scripts de Verificacion

| Script | Proposito |
|--------|-----------|
| `verify_data_leakage.py` | Verificar data leakage |
| `verify_canonical_delaunay.py` | Verificar triangulacion |
| `verify_gpa_correctness.py` | Verificar GPA |
| `verify_no_tta.py` | Verificar sin TTA |

## Directorio archive/

Scripts archivados de sesiones anteriores:

- `debug_*.py` - Scripts de debugging temporales
- `session30_*.py`, `session31_*.py` - Scripts de sesiones especificas
- `validation_session*.py` - Validaciones de sesiones antiguas
- `experiment_*.py` - Experimentos puntuales
- `test_*.py` - Tests que deberian estar en `tests/`

Estos se mantienen por referencia historica pero no se usan directamente.
Ver `docs/ARCHIVE_LOG.md` para el detalle de movimientos recientes.

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
