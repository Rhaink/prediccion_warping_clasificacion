# Experiments Index

Short index of validated experiments and the authoritative artifacts
(checkpoints, logs, and source files) to reproduce results.

## Landmark Ensemble (224 scale)

### Best (2026-01-11)
- Mean error: 3.61 px (TTA + CLAHE)
- Models:
  - checkpoints/session10/ensemble/seed123/final_model.pt
  - checkpoints/session13/seed321/final_model.pt
  - checkpoints/repro_split111/session14/seed111/final_model.pt
  - checkpoints/repro_split666/session16/seed666/final_model.pt
- Evaluation:
  - bash scripts/run_best_ensemble.sh
- Sources:
  - docs/REPRO_ENSEMBLE_3_71.md
  - GROUND_TRUTH.json (landmarks ensemble_4_models_tta_best_20260111)

### Baseline (Session 10/13)
- Mean error: 3.71 px (TTA + CLAHE)
- Models:
  - checkpoints/session10/ensemble/seed123/final_model.pt
  - checkpoints/session10/ensemble/seed456/final_model.pt
  - checkpoints/session13/seed321/final_model.pt
  - checkpoints/session13/seed789/final_model.pt
- Sources:
  - docs/REPRO_ENSEMBLE_3_71.md
  - GROUND_TRUTH.json (landmarks ensemble_4_models_tta)

## Seed Sweeps

- Seeds 111/222:
  - outputs/ensemble_combo_sweep_111_222.txt
  - outputs/option1_new_seeds.log
- Seeds 333/444:
  - outputs/ensemble_combo_sweep_333_444.txt
  - outputs/ensemble_combo_sweep_option2_333_444.txt
  - outputs/option1_333_444.log
- Seeds 555/666:
  - outputs/ensemble_combo_sweep_555_666.txt
  - outputs/ensemble_combo_sweep_option2_555_666.txt
  - outputs/option1_555_666.log

## Notes
- Use docs/REPRO_ENSEMBLE_3_71.md for step-by-step runs.
- Update this file when a new best result is validated.
