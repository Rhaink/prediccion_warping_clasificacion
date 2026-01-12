# Quickstart: Warping (solo pulmones)

Objetivo: generar el dataset warpeado **solo pulmones** usando el ensemble best
(3.61 px) con TTA + CLAHE y cache de landmarks, igual al estilo de
`outputs/full_warped_dataset`.

## Requisitos
- Entorno activo: `source .venv/bin/activate`
- Dataset en `data/dataset/COVID-19_Radiography_Dataset/`
- CSV con coordenadas en `data/coordenadas/coordenadas_maestro.csv`

## Quickstart (una sola linea)
```bash
nohup bash scripts/quickstart_warping.sh > outputs/warping_quickstart.log 2>&1 &
tail -f outputs/warping_quickstart.log
```

## Que hace el script
1) Verifica/genera la forma canonica y triangulos (GPA).
2) Predice landmarks para todo el dataset y guarda cache (JSON/NPZ).
3) Genera el dataset warpeado con `--predictions` (sin inferencia en el paso de warping).

## Salidas esperadas
- Forma canonica:
  - `outputs/shape_analysis/canonical_shape_gpa.json`
  - `outputs/shape_analysis/canonical_delaunay_triangles.json`
- Predicciones:
  - `outputs/landmark_predictions/<SESSION>/predictions.npz`
- Dataset warpeado:
  - `outputs/warped_lung_best/<SESSION>/dataset_summary.json`
  - `outputs/warped_lung_best/<SESSION>/{train,val,test}/...`

## Variables opcionales
```bash
SESSION=session_warping \
INPUT_DIR=data/dataset/COVID-19_Radiography_Dataset \
ENSEMBLE_CONFIG=configs/ensemble_best.json \
PREDICTIONS_PATH=outputs/landmark_predictions/session_warping/predictions.npz \
WARPED_OUTPUT_DIR=outputs/warped_lung_best/session_warping \
nohup bash scripts/quickstart_warping.sh > outputs/warping_quickstart.log 2>&1 &
```

## Config recomendado
El config base para este flujo es:
- `configs/warping_best.json` (usa `use_full_coverage=false`)

Si quieres usarlo directo, ajusta `sessionXX` y ejecuta:
```bash
python -m src_v2 generate-dataset --config configs/warping_best.json
```

## Validacion rapida
- Revisa `outputs/warped_lung_best/<SESSION>/dataset_summary.json`
- Esperado: `fill_rate_mean` ~0.47
- Verifica que el resumen mencione `predictions_path`

## Scripts relevantes
- `scripts/predict_landmarks_dataset.py`
- `scripts/quickstart_warping.sh`
- `python -m src_v2 generate-dataset`
