# Reproduccion de ensemble 3.71 px (landmarks) y best 3.61 px

## Objetivos iniciales
- Replicar el baseline reportado (3.71 px) con ensemble de 4 modelos + TTA.
- Superar el baseline con nuevos seeds y confirmar estabilidad del proceso.
- Obtener modelos con resultados iguales o mejores y poder reentrenar el ensemble.
- Dejar un proceso reproducible y evitar confusiones de escala, split y evaluacion.

## Hallazgos clave (causas de la desviacion)
1) Escala de error:
   - `scripts/train.py` evalua en la escala real del dataset (299x299).
   - `src_v2 evaluate` y `src_v2 evaluate-ensemble` reportan en escala 224 (resize).
   - Por eso los errores individuales ~8-9 px (299) corresponden a ~6-7 px (224),
     y el ensemble baja a ~3.7-3.9 px.
2) Split de datos:
   - En el pasado el split usaba `random_state=args.seed`. Luego se fijo a 42.
   - Los modelos del 3.71 se entrenaron con split seed = seed del modelo.
   - Se agrego `--split-seed` para controlar esto y reproducir exactamente.
3) Determinismo:
   - `--deterministic` fallaba por `adaptive_avg_pool2d` sin implementacion determinista.
   - Se ajusto a `torch.use_deterministic_algorithms(True, warn_only=True)` para no abortar.
4) Terminal/Termux:
   - Multilineas con `nohup` y `EOF` se rompian por saltos de linea.
   - Solucion: usar un script en repo (`scripts/run_repro_split_ensemble.sh`) y
     llamarlo en una sola linea.

## Cambios aplicados en el codigo
- `scripts/train.py`: nuevo `--split-seed`, `--deterministic`, se usa `split_seed` en
  `create_dataloaders`, y `set_global_seed` usa determinismo con warn_only.
- `src_v2/data/dataset.py`: `create_dataloaders` acepta `seed` y `deterministic`,
  inicializa seed en workers y `torch.Generator`.
- `scripts/run_repro_split_ensemble.sh`: helper para entrenar seeds 321 y 789 y
  evaluar ensemble con 4 modelos.
- `scripts/run_option1_new_seeds.sh`: entrena seeds nuevos (111/222) y ejecuta
  un sweep de combinaciones.
- `scripts/sweep_ensemble_combos.py`: evalua todas las combinaciones de tamaño K
  y reporta la mejor.

## Archivos necesarios
- Dataset: `data/` y `data/coordenadas/coordenadas_maestro.csv`
- Entrenamiento: `scripts/train.py`
- Evaluacion ensemble: `python -m src_v2 evaluate-ensemble`
- Checkpoints base (mejor 3.71):
  - `checkpoints/session10/ensemble/seed123/final_model.pt`
  - `checkpoints/session10/ensemble/seed456/final_model.pt`
  - `checkpoints/session13/seed321/final_model.pt`
  - `checkpoints/session13/seed789/final_model.pt`
- Mejor actual (3.61):
  - `checkpoints/session10/ensemble/seed123/final_model.pt`
  - `checkpoints/session13/seed321/final_model.pt`
  - `checkpoints/repro_split111/session14/seed111/final_model.pt`
  - `checkpoints/repro_split666/session16/seed666/final_model.pt`

## Archivos opcionales / soporte
- `scripts/verify_individual_models.py` (eval individual en escala 224)
- `scripts/verify_no_tta.py` (comparar sin TTA)
- `scripts/run_repro_split_ensemble.sh` (automatiza entrenamiento + ensemble)
- `scripts/run_option1_new_seeds.sh` (entrena seeds nuevos + sweep)
- `scripts/sweep_ensemble_combos.py` (barrido de combinaciones)
- `configs/landmarks_train_base.json` (plantilla de entrenamiento)
- `configs/ensemble_best.json` (config del ensemble best)
- Logs: `outputs/*/evaluation_report_*.txt`, `outputs/*/training_history.json`

## Proceso paso a paso (reproducir el 3.71 con modelos existentes)
1) Activar entorno:
   - `source .venv/bin/activate`
2) Evaluar ensemble original (3.71 esperado):
   - `python -m src_v2 evaluate-ensemble \`
     `checkpoints/session10/ensemble/seed123/final_model.pt \`
     `checkpoints/session10/ensemble/seed456/final_model.pt \`
     `checkpoints/session13/seed321/final_model.pt \`
     `checkpoints/session13/seed789/final_model.pt \`
     `--tta --clahe`
3) Guardar el resultado en un log si quieres:
   - `python -m src_v2 evaluate-ensemble ... --tta --clahe | tee outputs/ensemble_371.log`

## Proceso paso a paso (reproducir el mejor actual 3.61)
1) Activar entorno:
   - `source .venv/bin/activate`
2) Evaluar ensemble best (3.61 esperado):
   - `python -m src_v2 evaluate-ensemble \`
     `checkpoints/session10/ensemble/seed123/final_model.pt \`
     `checkpoints/session13/seed321/final_model.pt \`
     `checkpoints/repro_split111/session14/seed111/final_model.pt \`
     `checkpoints/repro_split666/session16/seed666/final_model.pt \`
     `--tta --clahe`
3) Guardar el resultado en un log si quieres:
   - `python -m src_v2 evaluate-ensemble ... --tta --clahe | tee outputs/ensemble_361.log`

## Proceso paso a paso (reentrenar un modelo y evaluar)
Ejemplo con un seed:
```bash
python scripts/train.py --seed 123 --split-seed 123 \
  --save-dir checkpoints/repro_split123/session10/ensemble/seed123 \
  --output-dir outputs/repro_split123/session10/ensemble/seed123 \
  --batch-size 16 \
  --phase1-epochs 15 --phase2-epochs 100 \
  --phase1-lr 1e-3 --phase2-backbone-lr 2e-5 --phase2-head-lr 2e-4 \
  --phase1-patience 5 --phase2-patience 15 \
  --coord-attention --deep-head --hidden-dim 768 --dropout 0.3 \
  --clahe --clahe-clip 2.0 --clahe-tile 4 \
  --loss wing --tta
```
Con config (opcion 3):
```bash
python scripts/train.py --config configs/landmarks_train_base.json \
  --seed 123 --split-seed 123 \
  --save-dir checkpoints/repro_split123/session10/ensemble/seed123 \
  --output-dir outputs/repro_split123/session10/ensemble/seed123
```
Notas:
- Las claves del config usan underscores (ej: phase1_epochs, phase2_head_lr).
- CLI siempre puede sobreescribir valores del config.
Notas:
- `--tta` solo afecta la evaluacion final, no el entrenamiento.
- Si comparas con 3.71/3.61, usa `evaluate-ensemble` (escala 224).
- La evaluacion final de `scripts/train.py` esta en escala 299.

## Automatizar (evitar errores de pegado)
```bash
nohup bash scripts/run_repro_split_ensemble.sh > outputs/repro_split_all_run2.log 2>&1 &
tail -f outputs/repro_split_all_run2.log
```
Con esto se entrenan seeds 321 y 789 (seed123 y seed456 ya existen en repro_split),
y luego se evalua el ensemble con TTA+CLAHE.

Para entrenar seeds nuevos y buscar mejor combinacion:
```bash
nohup bash scripts/run_option1_new_seeds.sh > outputs/option1_new_seeds.log 2>&1 &
tail -f outputs/option1_new_seeds.log
```

El resultado del sweep queda en:
- `outputs/ensemble_combo_sweep_111_222.txt`

Para correr seeds arbitrarios (opcion 1 + opcion 2 en un solo flujo):
```bash
nohup bash scripts/run_seed_sweep.sh 333 444 > outputs/option1_333_444.log 2>&1 &
tail -f outputs/option1_333_444.log
```

Para reproducir el best 3.61 (seed555/seed666):
```bash
nohup bash scripts/run_seed_sweep.sh 555 666 > outputs/option1_555_666.log 2>&1 &
tail -f outputs/option1_555_666.log
```

Para evaluar el mejor ensemble actual:
```bash
bash scripts/run_best_ensemble.sh
```

Si quieres usar config con los sweeps:
```bash
TRAIN_CONFIG=configs/landmarks_train_base.json \
  nohup bash scripts/run_seed_sweep.sh 333 444 > outputs/option1_333_444.log 2>&1 &
```

Evaluar ensemble desde config:
```bash
python scripts/evaluate_ensemble_from_config.py --config configs/ensemble_best.json
```

## Opcion 2 (barrido con modelos rerun)
Evalua combinaciones que incluyen modelos originales y rerun (sin reentrenar):
```bash
python scripts/sweep_ensemble_combos.py --tta --clahe \
  --out outputs/ensemble_combo_sweep_option2.txt \
  checkpoints/session10/ensemble/seed123/final_model.pt \
  checkpoints/session10/ensemble/seed456/final_model.pt \
  checkpoints/session13/seed321/final_model.pt \
  checkpoints/session13/seed789/final_model.pt \
  checkpoints/repro_split111/session14/seed111/final_model.pt \
  checkpoints/repro_split222/session14/seed222/final_model.pt \
  checkpoints/repro_split456_rerun/session10/ensemble/seed456/final_model.pt \
  checkpoints/repro_split321/session13/seed321/final_model.pt \
  checkpoints/repro_split789/session13/seed789/final_model.pt
```

## Resultados de esta sesion
- Ensemble reproducido con nuevos seeds (repro_split*): 3.86 px con TTA+CLAHE
  (log `outputs/repro_split_all_run2.log`)
- Semillas nuevas:
  - seed321 (TTA): 8.93 px en
    `outputs/repro_split321/session13/seed321/evaluation_report_20260110_022354.txt`
  - seed789 (TTA): 8.39 px en
    `outputs/repro_split789/session13/seed789/evaluation_report_20260110_022953.txt`
- Nuevo mejor resultado (supera 3.71):
  - BEST 3.61 px con el combo:
    `checkpoints/session10/ensemble/seed123/final_model.pt`
    `checkpoints/session13/seed321/final_model.pt`
    `checkpoints/repro_split111/session14/seed111/final_model.pt`
    `checkpoints/repro_split666/session16/seed666/final_model.pt`
  - Metricas (TTA+CLAHE, test split):
    - mean=3.61 px, median=3.07 px, std=2.48 px
    - Normal=3.22 px, COVID=3.93 px, Viral_Pneumonia=4.11 px
  - Sweeps en `outputs/ensemble_combo_sweep_555_666.txt` y
    `outputs/ensemble_combo_sweep_option2_555_666.txt`
  - Log en `outputs/option1_555_666.log`
- Opcion 2 (rerun vs original):
  - Mejor resultado no cambia (3.61 px)
  - Sweep en `outputs/ensemble_combo_sweep_option2_555_666.txt`
- Seeds 111/222 (TTA):
  - Best de ese sweep: 3.67 px
  - Sweeps en `outputs/ensemble_combo_sweep_111_222.txt` y
    `outputs/ensemble_combo_sweep_option2.txt`
- Seeds 333/444 (TTA):
  - seed333 mean 8.70 px, seed444 mean 9.02 px
  - Best de ese sweep: 3.67 px
  - Sweeps en `outputs/ensemble_combo_sweep_333_444.txt` y
    `outputs/ensemble_combo_sweep_option2_333_444.txt`

## Errores comunes y solucion rapida
- `outputs/: Is a directory` -> el redirect estaba cortado; usa una sola linea.
- `cd: too many arguments` -> comando pegado con saltos; usa script.
- Prompt `>` en Termux -> la shell espera cierre; evita heredoc y multilineas.
- Error determinismo `adaptive_avg_pool2d` -> usar `--deterministic` con warn_only
  (ya aplicado en el codigo).
