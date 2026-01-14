# Archive Log

Registro de archivos movidos a `scripts/archive/` para reducir ruido operativo
sin perder referencia historica.

## 2026-01-13 - Clasificacion y Grad-CAM (legacy)

**Motivo:** scripts de experimentos antiguos que dependen de rutas legacy
(`outputs/warped_dataset`, `outputs/classifier_comparison`,
`outputs/full_warped_dataset`) o fueron reemplazados por comandos CLI en
`python -m src_v2` (gradcam, compare-architectures, optimize-margin).

Movidos a `scripts/archive/classification/`:
- `scripts/gradcam_comparison.py` -> `scripts/archive/classification/gradcam_comparison.py`
- `scripts/gradcam_multi_architecture.py` -> `scripts/archive/classification/gradcam_multi_architecture.py`
- `scripts/gradcam_pfs_analysis.py` -> `scripts/archive/classification/gradcam_pfs_analysis.py`
- `scripts/train_all_architectures.py` -> `scripts/archive/classification/train_all_architectures.py`
- `scripts/margin_optimization_experiment.py` -> `scripts/archive/classification/margin_optimization_experiment.py`
- `scripts/train_expanded_dataset.py` -> `scripts/archive/classification/train_expanded_dataset.py`
- `scripts/train_resnet18_expanded.py` -> `scripts/archive/classification/train_resnet18_expanded.py`
- `scripts/train_baseline_original_15k.py` -> `scripts/archive/classification/train_baseline_original_15k.py`
- `scripts/compare_classifiers.py` -> `scripts/archive/classification/compare_classifiers.py`
