# Repro: Clasificador ResNet18 (warped_lung_best)

Objetivo: reproducir el clasificador CNN (ResNet18) usando el dataset
warpeado actual y documentar el flujo canonico.

Este flujo NO regenera el warping. Usa el dataset ya existente en:
`outputs/warped_lung_best/session_warping`.

## Requisitos
- Entorno activo: `source .venv/bin/activate`
- Dataset original: `data/dataset/COVID-19_Radiography_Dataset`
- Dataset warpeado: `outputs/warped_lung_best/session_warping`
  - Debe contener `train/`, `val/`, `test/` + `images.csv`

## Entrenar clasificador con dataset warpeado (canonico)
```bash
python -m src_v2 train-classifier --config configs/classifier_warped_base.json
```

Salidas esperadas:
- `outputs/classifier_warped_lung_best/best_classifier.pt`
- `outputs/classifier_warped_lung_best/results.json`

Evaluar:
```bash
python -m src_v2 evaluate-classifier \
  outputs/classifier_warped_lung_best/best_classifier.pt \
  --data-dir outputs/warped_lung_best/session_warping --split test
```

## Entrenar clasificador en imagenes originales (mismos splits)
Este script usa los splits del warpeado pero lee imagenes originales.

```bash
python scripts/train_classifier_original.py --config configs/classifier_original_base.json
```

Salidas esperadas:
- `outputs/classifier_original_warped_lung_best/best_classifier.pt`
- `outputs/classifier_original_warped_lung_best/results.json`

## Notas
- `train_classifier.py` es solo un wrapper del CLI.
- El directorio original usa `Viral Pneumonia` (con espacio).
  El script hace el mapeo automatico desde `Viral_Pneumonia`.
- Si deseas overrides, puedes pasar flags del CLI despues de `--config`.

## Decision de alcance (2026-01-12)
- Prioridad actual: reproducir el clasificador y maximizar accuracy en warped_lung_best.
- Se pospone la limpieza/alineacion de scripts legacy hasta terminar este objetivo.

## Smoke test (2026-01-12)
- Comando: `python -m src_v2 train-classifier --config configs/classifier_warped_base.json --epochs 1 --batch-size 4`
- Resultados (test): accuracy 0.9203, F1 macro 0.8948, F1 weighted 0.9186
- Artefactos archivados: `outputs/archive/classifier_warped_lung_best_smoketest_2026-01-12`

## Run completo (2026-01-12, baseline)
- Comando:
  `python -m src_v2 train-classifier --config configs/classifier_warped_base.json --output-dir outputs/classifier_warped_lung_best/session_2026-01-12 --device cuda`
- Mejor val F1: 0.9791 (epoch 36), early stopping en epoch 46
- Resultados (test):
  - accuracy 0.9794
  - F1 macro 0.9675
  - F1 weighted 0.9793
- Artefactos:
  - `outputs/classifier_warped_lung_best/session_2026-01-12/best_classifier.pt`
  - `outputs/classifier_warped_lung_best/session_2026-01-12/results.json`
- Nota: superado por el sweep 2026-01-13 (ver Opcion C).

## Evaluacion con TTA (opcional)
```bash
python -m src_v2 evaluate-classifier \
  outputs/classifier_warped_lung_best/sweeps_2026-01-12/lr2e-4_seed321_on/best_classifier.pt \
  --data-dir outputs/warped_lung_best/session_warping --split test --tta
```

## Opcion C: Sweep de accuracy (2026-01-13)
Objetivo: maximizar accuracy en test, seleccionando por val accuracy.

Script recomendado:
```bash
bash scripts/run_classifier_sweep_accuracy.sh
```

Variables utiles:
```bash
OUTPUT_ROOT=outputs/classifier_warped_lung_best/sweeps_2026-01-12 \
SEEDS="42 123 321" \
LRS="5e-5 2e-4" \
CLASS_WEIGHTS_MODES="on off" \
DEVICE=cuda \
bash scripts/run_classifier_sweep_accuracy.sh
```

Resultado best actual (sweep):
- Run: `lr2e-4_seed321_on`
- Resultados (test):
  - accuracy 0.9805
  - F1 macro 0.9712
  - F1 weighted 0.9804
- Artefactos:
  - `outputs/classifier_warped_lung_best/sweeps_2026-01-12/lr2e-4_seed321_on/best_classifier.pt`
  - `outputs/classifier_warped_lung_best/sweeps_2026-01-12/lr2e-4_seed321_on/results.json`

## Futuras pruebas (no implementadas)
- Label smoothing: suaviza targets (ej. 0.9/0.1) para reducir overfitting y mejorar calibracion.
- Focal loss: enfoca el aprendizaje en ejemplos dificiles; puede mejorar recall de clases minoritarias.

## Baseline existente (referencia)
- `outputs/classifier_full/results.json`
- `outputs/classifier_original_3classes/results.json`
- `outputs/classifier_original_cropped_47/results.json`
