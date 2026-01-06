# ROI Warping: Guia para Variantes de Dataset (Manual y Full)

## Objetivo
Evaluar como cambia la clasificacion al variar el aumento de ROI (contorno pulmonar) en datasets warped.

## Fuentes y artefactos
- Landmarks GT: `data/coordenadas/coordenadas_maestro.csv` (escala 299)
- Landmarks GT escalados a 224: `outputs/predictions/all_landmarks.npz`
- Forma canonica: `outputs/shape_analysis/canonical_shape_gpa.json`
- Triangulos Delaunay (18): `outputs/shape_analysis/canonical_delaunay_triangles.json`

## Pipeline de warping (comun)
- Warping piecewise affine: `scripts/piecewise_affine_warp.py`
- Con `use_full_coverage=False`: solo area pulmonar (fondo negro, fill ~47%)
- Con `use_full_coverage=True`: cobertura total de imagen (fill ~99%)

## Control del aumento de ROI
Parametro principal: `margin_scale` (escala desde el centroide)
- margin_scale > 1.0 => expande ROI (contorno mas amplio)
- margin_scale = 1.0 => ROI base
- margin_scale < 1.0 => contrae ROI
Siempre aplicar `clip_landmarks_to_image` para evitar salir del borde.

## Dataset manual (GT, 957)
Script: `scripts/generate_warped_dataset.py`
Estado actual:
- Usa GT en 224 desde `outputs/predictions/all_landmarks.npz`
- No aplica margin_scale
- `use_full_coverage=False`

Para crear variantes:
- Introducir margin_scale (ej. 1.10, 1.15) sobre GT
- Mantener triangulos canonicos fijos
- Salida a carpeta nueva para no pisar base

## Dataset full (predicho, 15k)
Script: `scripts/generate_full_warped_dataset.py`
Estado actual:
- `MARGIN_SCALE = 1.05`
- Prediccion con ensemble (`scripts/predict.py`)
- `use_full_coverage=False`

Para crear variantes:
- Cambiar `MARGIN_SCALE` (ej. 1.10, 1.15)
- Salida a carpeta nueva
- Mantener splits seed=42

## Clasificacion y comparabilidad
- Usar mismos hiperparametros de clasificador
- Mantener splits identicos entre variantes
- Registrar: margin_scale, fill_rate, dataset_summary.json
- Entrenamiento: `python -m src_v2 train-classifier <dataset_dir>`

## Riesgos / notas
- `outputs/full_coverage_warped_dataset` fue invalidado en el pasado; evitar replicarlo sin validar pipeline.
- Cambiar forma canonica o triangulos rompe comparabilidad entre variantes.
