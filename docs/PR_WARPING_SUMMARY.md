# Resumen tecnico: Warping reproducible (solo pulmones)

Objetivo: dejar el warping reproducible usando el ensemble best (3.61 px),
con TTA + CLAHE, y generar un dataset warpeado solo pulmones (sin full coverage),
visualmente consistente con `outputs/full_warped_dataset`.

## Cambios clave
- Nuevo cache de landmarks para dataset completo en `scripts/predict_landmarks_dataset.py`.
- Quickstart de warping con cache en `scripts/quickstart_warping.sh`.
- Soporte en CLI para `--predictions`, `--ensemble-config` y `--config` en
  `python -m src_v2 generate-dataset`.
- Config de warping reproducible en `configs/warping_best.json`.
- Documentacion y guias actualizadas en `docs/QUICKSTART_WARPING.md`,
  `docs/PLAN_WARPING_AGENT.md`, `README.md`, `docs/README.md`, `scripts/README.md`.
- Scripts legacy de generacion de datasets movidos a `scripts/archive/legacy_warping/`.
- Scripts de analisis warping movidos a `scripts/archive/warping_analysis/` con README.

## Flujo recomendado (resumen)
1) Cache de landmarks (ensemble best 3.61 + TTA + CLAHE).
2) Warping con `--predictions` y `--no-full-coverage`.

Salida esperada: `outputs/warped_lung_best/<SESSION>/` con fill rate ~0.47.

## Notas
- Los scripts legacy siguen disponibles en `scripts/archive/legacy_warping/`,
  pero el flujo recomendado es el nuevo (cache + generate-dataset).
