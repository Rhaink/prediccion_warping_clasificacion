# Reproduccion completa: landmarks -> warping -> clasificacion (pipeline actual)

Este documento describe el flujo actual de punta a punta y los archivos
necesarios para reproducir los resultados. Incluye los scripts/funciones clave
y nota elementos legacy que ya no se usan en el pipeline actual.

## Entradas necesarias (desde cero)
- Datos y checkpoints requeridos para reproducir el pipeline.
- El autor los proveera fuera del repositorio.
- Los comandos asumen la estructura `data/` y `checkpoints/` del proyecto.

## 0) Entorno base
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Entrenar modelos de landmarks (si necesitas reentrenar)
Script: `scripts/train.py`

Funciones clave:
- `src_v2/data/dataset.create_dataloaders`: crea splits y DataLoaders.
- `src_v2/models.create_model`: construye la red de landmarks.
- `src_v2/training/LandmarkTrainer`: entrenamiento en 2 fases + early stopping.
- `src_v2/evaluation/metrics.evaluate_model`: evalua error en pixeles.

Ejemplo (usa config base):
```bash
python scripts/train.py --config configs/landmarks_train_base.json \
  --seed 123 --split-seed 123 \
  --save-dir checkpoints/repro_split123/session10/ensemble/seed123 \
  --output-dir outputs/repro_split123/session10/ensemble/seed123
```

Notas:
- `--split-seed` fija el split para reproducibilidad.
- El error reportado en `scripts/train.py` usa escala 299; en el CLI se reporta en 224.

## 2) Evaluar ensemble de landmarks
CLI: `python -m src_v2 evaluate-ensemble`

Funciones clave:
- `src_v2/evaluation/metrics.compute_pixel_error`: error por landmark.
- `src_v2/constants.SYMMETRIC_PAIRS`: corrige flip para TTA.

Ejemplo (ensemble best 3.61):
```bash
python scripts/evaluate_ensemble_from_config.py --config configs/ensemble_best.json
```

## 3) Predicciones de landmarks para TODO el dataset (cache)
Script: `scripts/predict_landmarks_dataset.py`

Funciones clave:
- `detect_architecture_from_checkpoint`: infiere arquitectura desde el state_dict.
- `create_model`: crea cada modelo del ensemble.
- `SYMMETRIC_PAIRS`: corrige landmarks con flip (TTA).

Salida:
- `outputs/landmark_predictions/<SESSION>/predictions.npz`
- Metadata dentro de `predictions.npz` (models, tta, clahe, seed, etc).

Ejemplo:
```bash
python scripts/predict_landmarks_dataset.py \
  --input-dir data/dataset/COVID-19_Radiography_Dataset \
  --output outputs/landmark_predictions/session_warping/predictions.npz \
  --ensemble-config configs/ensemble_best.json \
  --tta --clahe --clahe-clip 2.0 --clahe-tile 4
```

## 4) Forma canonica y triangulos (GPA)
CLI: `python -m src_v2 compute-canonical`

Funciones clave (en `src_v2/processing/gpa.py`):
- `gpa_iterative`: alinea landmarks y obtiene forma consenso.
- `scale_canonical_to_image`: escala a coordenadas de imagen (224).
- `compute_delaunay_triangulation`: triangulacion para warping.

Salida:
- `outputs/shape_analysis/canonical_shape_gpa.json`
- `outputs/shape_analysis/canonical_delaunay_triangles.json`

Ejemplo:
```bash
python -m src_v2 compute-canonical data/coordenadas/coordenadas_maestro.csv \
  --output-dir outputs/shape_analysis --visualize
```

## 5) Warping del dataset (usa cache de landmarks)
CLI: `python -m src_v2 generate-dataset` (recomendado con config)

Funciones clave:
- `src_v2/processing/warp.piecewise_affine_warp`: warping por triangulos.
- `scale_landmarks_from_centroid`: aplica margin_scale.
- `clip_landmarks_to_image`: evita landmarks fuera de imagen.
- `compute_fill_rate`: porcentaje de pixeles no negros.

Config recomendado:
- `configs/warping_best.json` (usa `use_full_coverage=false`, margin 1.05)

Ejemplo:
```bash
python -m src_v2 generate-dataset --config configs/warping_best.json
```

Salida:
- `outputs/warped_lung_best/<SESSION>/train/`, `val/`, `test/`
- `images.csv` y `landmarks.json` por split
- `dataset_summary.json` con metadata de warping y predicciones

Dataset actual:
- `outputs/warped_lung_best/session_warping`
- `dataset_summary.json` indica `landmarks.source = predictions`
- `fill_rate_mean` ~ 0.47 (warping solo pulmones)

Quickstart automatizado:
```bash
nohup bash scripts/quickstart_warping.sh > outputs/warping_quickstart.log 2>&1 &
```

## 6) Entrenar clasificador en warped
CLI: `python -m src_v2 train-classifier`

Funciones clave:
- `torchvision.datasets.ImageFolder`: carga train/val/test.
- `src_v2/models/ImageClassifier`: backbone + head.
- `src_v2/models.get_classifier_transforms`: transforms train/eval.
- `src_v2/models.get_class_weights`: pesos por clase.

Config recomendado:
- `configs/classifier_warped_base.json`

Ejemplo:
```bash
python -m src_v2 train-classifier --config configs/classifier_warped_base.json
```

Salida:
- `outputs/classifier_warped_lung_best/best_classifier.pt`
- `outputs/classifier_warped_lung_best/results.json`

## 7) Evaluar clasificador
```bash
python -m src_v2 evaluate-classifier \
  outputs/classifier_warped_lung_best/best_classifier.pt \
  --data-dir outputs/warped_lung_best/session_warping --split test
```

## Donde se generan hoy las predicciones de landmarks
- Archivo actual: `outputs/landmark_predictions/session_warping/predictions.npz`
- Generado por: `scripts/predict_landmarks_dataset.py`
- Consumido por: `python -m src_v2 generate-dataset --predictions ...`
- Confirmacion: `outputs/warped_lung_best/session_warping/dataset_summary.json`
  contiene `landmarks.predictions_path`.

## Elementos legacy / obsoletos (no parte del pipeline actual)
- `scripts/generate_warped_dataset.py`: usa GT y flujo de sesion 21.
- `scripts/generate_full_warped_dataset.py`: warping inline sin cache (sesion 25).
- `scripts/predict.py`: wrapper antiguo para landmarks (reemplazado por CLI y cache).
- `outputs/warped_dataset/` y `outputs/full_warped_dataset/`: datasets previos.
- `docs/PLAN_WARPING_AGENT.md`: plan historico ya implementado.
- `scripts/archive/`: scripts antiguos y experimentales, no usados hoy.

Si necesitas validar otros cambios recientes, consulta los commits de la rama
`warping-pipeline` (restaura los scripts y config del pipeline de warping) y los
docs en `docs/REPRO_ENSEMBLE_3_71.md` / `docs/REPRO_CLASSIFIER_RESNET18.md`.
