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
- `scripts/train_classifier_original.py` -> `scripts/archive/classification/train_classifier_original.py`
- `scripts/filter_dataset_3_classes.py` -> `scripts/archive/classification/filter_dataset_3_classes.py`
- `scripts/generate_original_cropped_47.py` -> `scripts/archive/classification/generate_original_cropped_47.py`
- `scripts/evaluate_external_baseline.py` -> `scripts/archive/classification/evaluate_external_baseline.py`
- `scripts/evaluate_external_warped.py` -> `scripts/archive/classification/evaluate_external_warped.py`
- `scripts/analyze_class_mapping.py` -> `scripts/archive/classification/analyze_class_mapping.py`
- `scripts/prepare_dataset3.py` -> `scripts/archive/classification/prepare_dataset3.py`
- `scripts/warp_dataset3.py` -> `scripts/archive/classification/warp_dataset3.py`

Movidos a `outputs/archive/`:
- `outputs/classifier_full` -> `outputs/archive/classifier_full`
- `outputs/classifier_original_3classes` -> `outputs/archive/classifier_original_3classes`
- `outputs/classifier_original_cropped_47` -> `outputs/archive/classifier_original_cropped_47`
- `outputs/classifier_efficientnet` -> `outputs/archive/classifier_efficientnet`
- `outputs/classifier_config_smoketest` -> `outputs/archive/classifier_config_smoketest`

Movidos a `results/figures/archive/`:
- `results/figures/gradcam` -> `results/figures/archive/gradcam`
- `results/figures/gradcam_multi` -> `results/figures/archive/gradcam_multi`
