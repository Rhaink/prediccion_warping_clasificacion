# Quickstart: Landmarks (hasta predicciones)

Objetivo: reproducir desde cero el pipeline de landmarks hasta obtener
predicciones del ensemble, antes de pasar a warping.

## Requisitos
- Entorno activo: `source .venv/bin/activate`
- Datos en `data/`
- CSV en `data/coordenadas/coordenadas_maestro.csv`

## Quickstart (una sola linea)
```bash
nohup bash scripts/quickstart_landmarks.sh > outputs/quickstart_landmarks.log 2>&1 &
tail -f outputs/quickstart_landmarks.log
```

## Que hace el script
1) Entrena 4 seeds (123, 321, 111, 666) con el config base.
2) Evalua el ensemble de esos 4 modelos (TTA + CLAHE).
3) Genera predicciones en el split test (seed=42) en CSV/JSON/NPZ y triangulacion.

## Salidas esperadas
- Checkpoints:
  - `checkpoints/repro_quickstart/<SESSION>/seed123/final_model.pt`
  - `checkpoints/repro_quickstart/<SESSION>/seed321/final_model.pt`
  - `checkpoints/repro_quickstart/<SESSION>/seed111/final_model.pt`
  - `checkpoints/repro_quickstart/<SESSION>/seed666/final_model.pt`
- Predicciones:
  - `outputs/repro_quickstart/<SESSION>/predictions/test_predictions.csv`
  - `outputs/repro_quickstart/<SESSION>/predictions/test_predictions.json`
  - `outputs/repro_quickstart/<SESSION>/predictions/test_predictions.npz`
  - `outputs/repro_quickstart/<SESSION>/predictions/delaunay_triangles.json`

## Variables opcionales
```bash
SESSION=session17 \
TRAIN_CONFIG=configs/landmarks_train_base.json \
OUTPUT_ROOT=outputs/repro_quickstart/session17 \
CHECKPOINT_ROOT=checkpoints/repro_quickstart/session17 \
nohup bash scripts/quickstart_landmarks.sh > outputs/quickstart_landmarks.log 2>&1 &
```

## Validacion rapida
- Esperado (escala 224): mean ~3.61 px con TTA+CLAHE.
- Verifica con:
  - `python -m src_v2 evaluate-ensemble <model1> <model2> <model3> <model4> --tta --clahe`

Si ya tienes los checkpoints del best actual:
```bash
python scripts/extract_predictions.py --config configs/ensemble_best.json \
  --output-dir outputs/predictions_best
```

## Antes de warping
Este quickstart termina en la generacion de predicciones. Para warping, usar:
`python -m src_v2 generate-dataset` (ver README), pero esto no forma parte
del flujo de predicciones.

## Scripts relevantes (minimo)
- `scripts/train.py`
- `scripts/quickstart_landmarks.sh`
- `scripts/run_seed_sweep.sh`
- `scripts/run_best_ensemble.sh`
- `scripts/extract_predictions.py`
- `scripts/sweep_ensemble_combos.py`
- `scripts/evaluate_ensemble_from_config.py`

## Scripts movidos a archive
- `scripts/archive/run_repro_split_ensemble.sh`
- `scripts/archive/run_option1_new_seeds.sh`
- `scripts/archive/evaluate_ensemble.py`
