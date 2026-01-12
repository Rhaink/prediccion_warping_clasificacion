# Warping Analysis Archive

Estos scripts fueron movidos aqui para mantener minimo el pipeline de warping
reproducible. Sirven para analisis, comparaciones o experimentos historicos.

## Contenido
- `calculate_pfs_warped.py`: calcula PFS usando mascaras warpeadas (valida
  atencion pulmonar en modelos warped). Requiere dataset warped + mascaras.
- `compare_classifiers.py`: compara metricas entre clasificadores original vs
  warped (accuracy/F1/confusion matrix).
- `evaluate_external_warped.py`: evalua modelos warped en Dataset3 warpeado
  (pipeline de validacion externa).
- `filter_dataset_3_classes.py`: filtra datasets a 3 clases para comparaciones
  consistentes con warped.
- `gradcam_comparison.py`: Grad-CAM comparando atencion en original vs warped.
- `gradcam_multi_architecture.py`: Grad-CAM comparando varias arquitecturas en
  original vs warped.
- `gradcam_pfs_analysis.py`: analisis PFS con Grad-CAM (hipotesis foco pulmonar).
- `margin_optimization_experiment.py`: grid search de margin_scale generando
  datasets warpeados para encontrar fill rate optimo.
- `piecewise_affine_warp.py`: implementacion legacy de warping usada por scripts
  antiguos; el flujo actual usa `src_v2.processing.warp`.
- `train_baseline_original_15k.py`: baseline en dataset original 15k para
  comparar contra warped.
- `warp_dataset3.py`: warpea Dataset3 para evaluacion externa.

## Uso
Estos scripts no son parte del flujo principal. Se mantienen por referencia y
para reproducir analisis historicos cuando sea necesario.
