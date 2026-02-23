# Revision: scripts/archive/

**Fecha**: 2026-02-11
**Archivos analizados**: 40
**Lineas totales**: ~16,339
**Tamano total**: ~495 KB

---

## Grupo 1: Root Archive (22 archivos)

### 1. debug_hierarchical.py

- **Ruta**: `scripts/archive/debug_hierarchical.py`
- **Lineas/Tamano**: 183 lineas / ~6.1 KB
- **Proposito**: Analiza parametros geometricos del HierarchicalLandmarkModel comparando valores hardcoded con estadisticas reales del dataset de landmarks. Identifica bugs en `bilateral_t_base` que causaban predicciones incorrectas de landmarks laterales.
- **Importancia**: BAJO
- **Justificacion**: Script de debugging one-shot de la sesion de desarrollo del modelo jerarquico. El bug ya fue identificado y documentado. No tiene valor para reproduccion futura ya que el modelo jerarquico no se usa en el pipeline actual.

### 2. debug_landmark_visualization.py

- **Ruta**: `scripts/archive/debug_landmark_visualization.py`
- **Lineas/Tamano**: 175 lineas / ~6.7 KB
- **Proposito**: Visualiza landmarks numerados sobre la forma canonica y sobre imagenes reales para verificar que las conexiones anatomicas son correctas. Genera figuras de debug con lineas entre landmarks.
- **Importancia**: BAJO
- **Justificacion**: Script de verificacion visual one-shot. La visualizacion de landmarks ya esta integrada en el pipeline actual a traves de `generate-landmark-visualization-dataset`. No aporta funcionalidad unica.

### 3. evaluate_ensemble.py

- **Ruta**: `scripts/archive/evaluate_ensemble.py`
- **Lineas/Tamano**: 319 lineas / ~10.9 KB
- **Proposito**: Sesion 12 - Evalua combinaciones de modelos de ensemble (seeds 42, 123, 456) con TTA y promedios ponderados. Calcula error en pixeles para cada combinacion.
- **Importancia**: BAJO
- **Justificacion**: Supersedido por `scripts/evaluate_ensemble_from_config.py` y `configs/ensemble_best.json`. Los resultados de ensemble 3.61px ya estan validados. Usa seeds/modelos que no corresponden al ensemble actual (seed666 combo).

### 4. experiment_black_background.py

- **Ruta**: `scripts/archive/experiment_black_background.py`
- **Lineas/Tamano**: 467 lineas / ~15.1 KB
- **Proposito**: Sesion 28 - Experimento critico que verifica si el fondo negro en imagenes warped es un shortcut de clasificacion. Entrena modelos con 5 tipos de fondo (gaussian noise, uniform noise, mean fill, high noise, original black) y compara accuracy usando ANOVA y Kruskal-Wallis.
- **Importancia**: MEDIO
- **Justificacion**: Resultados del experimento son cientificamente importantes para la tesis (demuestran que el fondo negro NO es un shortcut). Sin embargo, el script usa paths hardcoded a modelos de sesiones antiguas y no se integra con el pipeline actual. Los resultados ya fueron documentados en las notas de sesion.

### 5. experiment_extended_margins.py

- **Ruta**: `scripts/archive/experiment_extended_margins.py`
- **Lineas/Tamano**: 424 lineas / ~12.7 KB
- **Proposito**: Sesion 28 - Prueba margenes extendidos [1.05 a 1.30] con warping on-the-fly. Entrena modelos canario (AlexNet/ResNet-18) para cada margen.
- **Importancia**: BAJO
- **Justificacion**: Supersedido por `python -m src_v2 optimize-margin` (Sesion 25) que encontro margin=1.05 como optimo. Usa imports legacy (`scripts.predict.EnsemblePredictor`, `scripts.piecewise_affine_warp`). Resultado ya consolidado en `GROUND_TRUTH.json`.

### 6. run_option1_new_seeds.sh

- **Ruta**: `scripts/archive/run_option1_new_seeds.sh`
- **Lineas/Tamano**: 41 lineas / ~1.5 KB
- **Proposito**: Shell script para entrenar seeds 111 y 222, luego hacer sweep de combinaciones de ensemble con 6 modelos.
- **Importancia**: BAJO
- **Justificacion**: Script de ejecucion one-shot para exploracion de seeds. El ensemble final usa seeds diferentes (seed666 combo). No aporta valor para reproduccion.

### 7. run_repro_split_ensemble.sh

- **Ruta**: `scripts/archive/run_repro_split_ensemble.sh`
- **Lineas/Tamano**: 50 lineas / ~1.7 KB
- **Proposito**: Shell script para entrenar seeds 321 y 789 con split seeds separados, luego evaluar ensemble con `repro_split` checkpoint paths.
- **Importancia**: BAJO
- **Justificacion**: Script de ejecucion one-shot. Los checkpoints de repro_split ya existen. El proceso de reproduccion esta documentado en `docs/REPRO_ENSEMBLE_3_71.md`.

### 8. session30_cross_evaluation.py

- **Ruta**: `scripts/archive/session30_cross_evaluation.py`
- **Lineas/Tamano**: 450 lineas / ~16.8 KB
- **Proposito**: Sesion 30 - Evaluacion cruzada completa comparando modelos original y warped en ambos datasets. Evalua 4 combinaciones: original->original, original->warped, warped->warped, warped->original. Calcula gaps de generalizacion.
- **Importancia**: MEDIO
- **Justificacion**: Analisis de cross-evaluation es relevante para la tesis. Sin embargo, usa paths hardcoded a modelos antiguos (`session27_models`, `session28_baseline_original`) y no es reproducible con el pipeline actual. Los resultados ya estan documentados.

### 9. session30_error_analysis.py

- **Ruta**: `scripts/archive/session30_error_analysis.py`
- **Lineas/Tamano**: 660 lineas / ~24.2 KB
- **Proposito**: Genera visualizaciones consolidadas (heatmaps, bar plots, confusion matrices) y tabla markdown desde resultados de cross-evaluation de sesion 30. Contiene valores hardcoded de resultados.
- **Importancia**: BAJO
- **Justificacion**: Script de generacion de figuras one-shot con datos hardcoded. Las figuras ya fueron generadas y los resultados documentados. No tiene utilidad futura.

### 10. session30_robustness_figure.py

- **Ruta**: `scripts/archive/session30_robustness_figure.py`
- **Lineas/Tamano**: 346 lineas / ~13.0 KB
- **Proposito**: Genera figuras de calidad publicacion para analisis de robustez usando resultados JSON de sesion 29. Crea graficos comparativos de artifact y geometric robustness.
- **Importancia**: BAJO
- **Justificacion**: Script de generacion de figuras one-shot. Las figuras ya fueron generadas. Se puede regenerar desde los datos JSON si fuera necesario, pero los paths son hardcoded.

### 11. session31_cross_evaluation.py

- **Ruta**: `scripts/archive/session31_cross_evaluation.py`
- **Lineas/Tamano**: 378 lineas / ~12.3 KB
- **Proposito**: Sesion 31 - Cross-evaluation multi-arquitectura (AlexNet, MobileNet, EfficientNet, DenseNet) en datasets original, warped_105, y warped_125.
- **Importancia**: BAJO
- **Justificacion**: Extension de session30_cross_evaluation con mas arquitecturas. Usa paths hardcoded a modelos de sesion 31. Resultados ya documentados en notas de sesion.

### 12. session31_generate_dataset_margin125.py

- **Ruta**: `scripts/archive/session31_generate_dataset_margin125.py`
- **Lineas/Tamano**: 315 lineas / ~10.5 KB
- **Proposito**: Genera dataset warped completo de 15K imagenes con margin_scale=1.25 para comparacion con margin=1.05. Usa `scripts.predict.EnsemblePredictor` (legacy).
- **Importancia**: BAJO
- **Justificacion**: Experimento de exploracion de margenes ya concluido. El margin optimo (1.05) ya esta validado. Usa imports legacy. Supersedido por `python -m src_v2 generate-dataset --config`.

### 13. session31_train_multi_arch.py

- **Ruta**: `scripts/archive/session31_train_multi_arch.py`
- **Lineas/Tamano**: 505 lineas / ~16.3 KB
- **Proposito**: Sesion 31 - Entrena AlexNet, MobileNetV2, EfficientNet-B0, DenseNet-121 en datasets original, warped_105, warped_125 con hiperparametros identicos para comparacion justa.
- **Importancia**: BAJO
- **Justificacion**: Experimento multi-arquitectura one-shot. Resultados documentados. No se integra con el pipeline actual basado en ResNet-18. Contiene codigo de entrenamiento duplicado que ya existe en `src_v2/training/`.

### 14. test_dataset.py

- **Ruta**: `scripts/archive/test_dataset.py`
- **Lineas/Tamano**: 357 lineas / ~12.6 KB
- **Proposito**: Tests de LandmarkDataset: carga de datos, horizontal flip con correccion de pares simetricos, funcionalidad de DataLoader, y generacion de figuras de verificacion.
- **Importancia**: BAJO
- **Justificacion**: Tests ad-hoc que deberian estar en `tests/`. La funcionalidad ya esta cubierta por los tests formales en `tests/test_data.py` y similares. Genera figuras de verificacion que ya no son necesarias.

### 15. test_forward_pass.py

- **Ruta**: `scripts/archive/test_forward_pass.py`
- **Lineas/Tamano**: 416 lineas / ~12.9 KB
- **Proposito**: Sesion 2 - Verificacion exhaustiva de: creacion de modelo, freeze/unfreeze de backbone, grupos de LR diferenciados, forward/backward pass, todas las funciones de loss (Wing, Weighted Wing, Central Alignment, Soft Symmetry, Combined).
- **Importancia**: BAJO
- **Justificacion**: Script de verificacion de sesion 2 (muy temprano en el desarrollo). Todas estas funcionalidades ya estan cubiertas por tests formales en `tests/`. No tiene valor para el pipeline actual.

### 16. test_hierarchical_forward.py

- **Ruta**: `scripts/archive/test_hierarchical_forward.py`
- **Lineas/Tamano**: 243 lineas / ~8.5 KB
- **Proposito**: Tests de forward pass del HierarchicalLandmarkModel. Compara predicciones de modelo entrenado vs no-entrenado. Analiza errores por tipo de landmark (eje, bilaterales).
- **Importancia**: BAJO
- **Justificacion**: El modelo jerarquico no se usa en el pipeline actual. Tests especificos de un modelo descartado. Sin valor para reproduccion.

### 17. test_reconstruct.py

- **Ruta**: `scripts/archive/test_reconstruct.py`
- **Lineas/Tamano**: 297 lineas / ~10.5 KB
- **Proposito**: Test aislado de la funcion de reconstruccion del modelo jerarquico. Verifica rangos de parametros (tanh, sigmoid), direcciones de vectores perpendiculares, e identifica bugs en `bilateral_t_base`.
- **Importancia**: BAJO
- **Justificacion**: Complemento de debug_hierarchical.py para el modelo jerarquico descartado. Bug ya identificado y documentado.

### 18. test_robustness_artifacts.py

- **Ruta**: `scripts/archive/test_robustness_artifacts.py`
- **Lineas/Tamano**: 462 lineas / ~14.9 KB
- **Proposito**: Sesion 29 - Test de robustez a artefactos: ruido (gaussiano, salt-and-pepper), blur (gaussiano, motion), cambios de contraste/brillo, compresion JPEG, y perturbaciones combinadas. Compara modelo original vs warped.
- **Importancia**: MEDIO
- **Justificacion**: Los resultados de robustez son fundamentales para la tesis. Sin embargo, el script usa paths hardcoded y modelos de sesiones antiguas. Los resultados estan documentados y el concepto esta implementado en `python -m src_v2 test-robustness`.

### 19. test_robustness_geometric.py

- **Ruta**: `scripts/archive/test_robustness_geometric.py`
- **Lineas/Tamano**: 413 lineas / ~13.7 KB
- **Proposito**: Sesion 29 - Test de robustez geometrica: rotacion, escala, traslacion, flip, y perturbaciones combinadas. Compara modelo original vs warped con metricas por perturbacion.
- **Importancia**: MEDIO
- **Justificacion**: Similar a test_robustness_artifacts.py - resultados relevantes para la tesis pero script no reproducible con pipeline actual. La funcionalidad de robustez geometrica esta en `src_v2`.

### 20. validation_session26.py

- **Ruta**: `scripts/archive/validation_session26.py`
- **Lineas/Tamano**: 800 lineas / ~28.1 KB
- **Proposito**: Sesion 26 - Validacion avanzada con 3 experimentos: (1) Grad-CAM con Pulmonary Focus Score, (2) Inyeccion de artefactos sinteticos, (3) Analisis de errores. El archivo mas largo del archive.
- **Importancia**: BAJO
- **Justificacion**: Script monolitico de sesion 26 con paths hardcoded a modelos viejos. La funcionalidad de PFS y Grad-CAM fue refinada en scripts posteriores (validation_session26_v2, validation_session27_pfs). Demasiado largo y no modular para reutilizacion.

### 21. validation_session26_v2.py

- **Ruta**: `scripts/archive/validation_session26_v2.py`
- **Lineas/Tamano**: 611 lineas / ~21.2 KB
- **Proposito**: Version 2 de la validacion de sesion 26 con artefactos agresivos (watermark, corner box, border), Grad-CAM con artefactos, y analisis de invariancia de warping.
- **Importancia**: BAJO
- **Justificacion**: Iteracion sobre validation_session26.py. Los artefactos agresivos (watermark, corner box) fueron un experimento exploratorio. Usa paths hardcoded. Funcionalidad de Grad-CAM disponible en otros scripts mas recientes.

### 22. validation_session27_pfs.py

- **Ruta**: `scripts/archive/validation_session27_pfs.py`
- **Lineas/Tamano**: 348 lineas / ~11.4 KB
- **Proposito**: Sesion 27 - Comparacion de PFS (Pulmonary Focus Score) entre el modelo de 98.02% accuracy y el modelo original. Usa Grad-CAM y mascaras pulmonares para calcular PFS.
- **Importancia**: BAJO
- **Justificacion**: Calculo de PFS refinado en gradcam_pfs_analysis.py (sesion 29). Usa paths a modelos de sesion 27 que ya no son los modelos actuales. Metrica PFS documentada en resultados de sesion.

---

## Grupo 2: Classification Archive (17 archivos)

### 23. analyze_class_mapping.py

- **Ruta**: `scripts/archive/classification/analyze_class_mapping.py`
- **Lineas/Tamano**: 616 lineas / ~22.8 KB
- **Proposito**: Sesion 37 - Analisis de 4 estrategias de mapeo de 3 clases (COVID, Normal, Viral_Pneumonia) a 2 clases binarias (COVID vs no-COVID): Estrategia A (suma de probabilidades), B (solo Normal como negativo), C (excluir con umbral), D (ponderada). Evalua en Dataset3/FedCOVIDx.
- **Importancia**: BAJO
- **Justificacion**: Analisis exploratorio de validacion externa. La evaluacion externa en Dataset3 fue un experimento complementario, no parte del pipeline principal. Usa modelos y datasets especificos de sesion 37.

### 24. compare_classifiers.py

- **Ruta**: `scripts/archive/classification/compare_classifiers.py`
- **Lineas/Tamano**: 144 lineas / ~5.1 KB
- **Proposito**: Sesion 22 - Script simple de comparacion que carga resultados JSON de clasificadores warped vs original y genera graficos de barras comparativos.
- **Importancia**: ELIMINABLE
- **Justificacion**: Script trivial de visualizacion que depende de paths antiguos (`outputs/classifier/results.json`, `outputs/classifier_original/results_original.json`). La comparacion ya fue documentada y las figuras generadas.

### 25. evaluate_external_baseline.py

- **Ruta**: `scripts/archive/classification/evaluate_external_baseline.py`
- **Lineas/Tamano**: 692 lineas / ~22.0 KB
- **Proposito**: Sesion 36 - Evaluacion de modelos existentes de 3 clases en Dataset3 externo binario (FedCOVIDx). Mapea P(positive)=P(COVID), P(negative)=P(Normal)+P(Viral). Incluye evaluacion con multiples thresholds.
- **Importancia**: BAJO
- **Justificacion**: Experimento de validacion externa one-shot. Dataset3 no forma parte del pipeline principal. Usa modelos y paths especificos de sesiones 27-28.

### 26. evaluate_external_warped.py

- **Ruta**: `scripts/archive/classification/evaluate_external_warped.py`
- **Lineas/Tamano**: 443 lineas / ~15.1 KB
- **Proposito**: Sesion 37 - Evaluacion especifica de modelos warped en la version warped de Dataset3 para comparacion justa (warped model -> warped external data).
- **Importancia**: BAJO
- **Justificacion**: Complemento de evaluate_external_baseline.py para la comparacion fair de modelos warped. Mismo contexto de validacion externa one-shot. Resultados documentados.

### 27. filter_dataset_3_classes.py

- **Ruta**: `scripts/archive/classification/filter_dataset_3_classes.py`
- **Lineas/Tamano**: 248 lineas / ~9.0 KB
- **Proposito**: Filtra el dataset original de 4 clases a 3 clases (excluye Lung_Opacity) para permitir cross-evaluation valida con datasets warped que solo tienen 3 clases.
- **Importancia**: BAJO
- **Justificacion**: Script utilitario one-shot. El pipeline actual ya maneja las 3 clases directamente. La funcionalidad de filtrado no se necesita porque el dataset warped se genera con las 3 clases desde el inicio.

### 28. generate_original_cropped_47.py

- **Ruta**: `scripts/archive/classification/generate_original_cropped_47.py`
- **Lineas/Tamano**: 295 lineas / ~10.1 KB
- **Proposito**: Experimento de control: genera imagenes originales recortadas a ~47% fill rate (154x154 centrado en 224x224 negro) para determinar si la robustez del modelo warped 47% viene de reduccion de informacion o normalizacion geometrica.
- **Importancia**: MEDIO
- **Justificacion**: Experimento de control cientificamente relevante para la tesis. Si el dataset cropped 47% es robusto, la robustez = reduccion de info; si no, robustez = normalizacion geometrica. Sin embargo, el script es autocontenido y no depende del pipeline actual.

### 29. gradcam_comparison.py

- **Ruta**: `scripts/archive/classification/gradcam_comparison.py`
- **Lineas/Tamano**: 437 lineas / ~14.5 KB
- **Proposito**: Sesion 22 - Comparacion Grad-CAM entre modelos warped y original. Genera visualizaciones de heatmaps superpuestos y PFS (Pulmonary Focus Score) basico.
- **Importancia**: BAJO
- **Justificacion**: Primera implementacion de Grad-CAM, supersedida por versiones mas completas (gradcam_multi_architecture.py, gradcam_pfs_analysis.py). Usa paths a modelos de sesion 22.

### 30. gradcam_multi_architecture.py

- **Ruta**: `scripts/archive/classification/gradcam_multi_architecture.py`
- **Lineas/Tamano**: 489 lineas / ~16.4 KB
- **Proposito**: Sesion 24 - Visualizacion Grad-CAM multi-arquitectura. Soporta AlexNet, ResNet-18, ResNet-50, MobileNetV2, EfficientNet-B0, DenseNet-121, VGG-16 con seleccion automatica de target layer por arquitectura.
- **Importancia**: BAJO
- **Justificacion**: Extension de gradcam_comparison.py para multiples arquitecturas. El pipeline actual usa solo ResNet-18. Las visualizaciones ya fueron generadas. Codigo de Grad-CAM podria ser util como referencia pero no se integra con el sistema actual.

### 31. gradcam_pfs_analysis.py

- **Ruta**: `scripts/archive/classification/gradcam_pfs_analysis.py`
- **Lineas/Tamano**: 430 lineas / ~14.6 KB
- **Proposito**: Sesion 29 - Analisis Grad-CAM comparativo con PFS cuantitativo. Calcula PFS = (gradcam * mask).sum() / gradcam.sum() para 300 muestras por modelo. Usa t-test para comparar PFS entre modelos original y warped.
- **Importancia**: MEDIO
- **Justificacion**: Analisis estadistico riguroso de PFS con test de hipotesis. Resultados importantes para la tesis. Sin embargo, usa paths hardcoded a modelos de sesiones 27-28 y mascaras del dataset original.

### 32. margin_optimization_experiment.py

- **Ruta**: `scripts/archive/classification/margin_optimization_experiment.py`
- **Lineas/Tamano**: 630 lineas / ~20.5 KB
- **Proposito**: Sesion 25 - Experimento completo de optimizacion de margen. Genera datasets warped con diferentes margin_scale [0.95, 1.0, 1.05, 1.10, 1.15], entrena modelos canario (AlexNet, ResNet-18), y compara resultados.
- **Importancia**: BAJO
- **Justificacion**: Supersedido por `python -m src_v2 optimize-margin` que implementa la misma logica de forma integrada. El resultado (margin=1.05) ya esta validado y en `GROUND_TRUTH.json`. Usa imports legacy (`scripts.piecewise_affine_warp`).

### 33. prepare_dataset3.py

- **Ruta**: `scripts/archive/classification/prepare_dataset3.py`
- **Lineas/Tamano**: 338 lineas / ~11.2 KB
- **Proposito**: Sesion 36 - Preparacion de Dataset3 (FedCOVIDx) para validacion externa: parsea archivos de etiquetas (train.txt, val.txt, test.txt), redimensiona imagenes a 299x299, crea estructura de directorios compatible con DataLoaders.
- **Importancia**: BAJO
- **Justificacion**: Script de preprocesamiento one-shot para un dataset externo que no forma parte del pipeline principal. Dataset3 fue un experimento complementario de validacion.

### 34. train_all_architectures.py

- **Ruta**: `scripts/archive/classification/train_all_architectures.py`
- **Lineas/Tamano**: 943 lineas / ~31.5 KB
- **Proposito**: Sesion 23 - Script generalizado para entrenar 7 arquitecturas CNN (AlexNet, ResNet-18/50, MobileNetV2, EfficientNet-B0, DenseNet-121, VGG-16) en datasets original y warped. Incluye generacion de reportes comparativos.
- **Importancia**: BAJO
- **Justificacion**: El archivo mas grande de classification archive. Contiene toda la logica de entrenamiento inline (no usa src_v2). El pipeline actual usa solo ResNet-18 via `python -m src_v2 train-classifier`. Los resultados multi-arquitectura ya fueron documentados. Mucho codigo duplicado con train_classifier_original.py y train_expanded_dataset.py.

### 35. train_baseline_original_15k.py

- **Ruta**: `scripts/archive/classification/train_baseline_original_15k.py`
- **Lineas/Tamano**: 467 lineas / ~15.5 KB
- **Proposito**: Sesion 28 - Entrena ResNet-18 baseline en 15K imagenes originales (sin warping) con splits identicos al modelo warped para comparacion justa. Compara con modelo warped de 98.02%.
- **Importancia**: BAJO
- **Justificacion**: Replicado por `python -m src_v2 train-classifier` con el dataset original. El resultado de la comparacion original vs warped ya esta documentado. Usa paths hardcoded a `session27_models` y `session28_baseline_original`.

### 36. train_classifier_original.py

- **Ruta**: `scripts/archive/classification/train_classifier_original.py`
- **Lineas/Tamano**: 552 lineas / ~18.1 KB
- **Proposito**: Sesion 22 - Entrenamiento de clasificador CNN en imagenes originales usando los mismos splits que el dataset warped. Soporta ResNet-18 y EfficientNet-B0 con argparse y config JSON.
- **Importancia**: BAJO
- **Justificacion**: Supersedido por `python -m src_v2 train-classifier` que ofrece la misma funcionalidad de forma integrada. El script era la version original de sesion 22 del entrenamiento en imagenes originales. Contiene mucho codigo de entrenamiento duplicado.

### 37. train_expanded_dataset.py

- **Ruta**: `scripts/archive/classification/train_expanded_dataset.py`
- **Lineas/Tamano**: 407 lineas / ~13.1 KB
- **Proposito**: Sesion 25 - Compara rendimiento de modelos canario (AlexNet, ResNet-18) entre dataset baseline (957 imagenes) y dataset expandido (~15K imagenes) con margin=1.05.
- **Importancia**: BAJO
- **Justificacion**: Resultado ya conocido: dataset expandido mejora significativamente sobre baseline. El pipeline actual siempre usa el dataset completo. Usa paths hardcoded a `outputs/full_warped_dataset` y `outputs/margin_experiment`.

### 38. train_resnet18_expanded.py

- **Ruta**: `scripts/archive/classification/train_resnet18_expanded.py`
- **Lineas/Tamano**: 308 lineas / ~10.4 KB
- **Proposito**: Sesion 27 - Re-entrenamiento de ResNet-18 en dataset expandido para recuperar el modelo de 97.76% accuracy que no fue guardado en Sesion 25.
- **Importancia**: BAJO
- **Justificacion**: Script de recuperacion one-shot. El modelo fue recuperado exitosamente y guardado. El pipeline actual produce modelos con 99.10% accuracy (warped_96), superando este resultado. No tiene valor de reproduccion.

### 39. warp_dataset3.py

- **Ruta**: `scripts/archive/classification/warp_dataset3.py`
- **Lineas/Tamano**: 299 lineas / ~9.9 KB
- **Proposito**: Sesion 37 - Aplica warping geometrico a imagenes de Dataset3 (FedCOVIDx) usando el ensemble de landmarks para evaluacion justa: modelos warped evaluados en Dataset3 warped.
- **Importancia**: BAJO
- **Justificacion**: Script de procesamiento one-shot para validacion externa. Usa imports legacy (`scripts.predict.EnsemblePredictor`, `scripts.piecewise_affine_warp`). Dataset3 no forma parte del pipeline principal.

---

## Grupo 3: Invalid Warping Archive (1 archivo)

### 40. generate_warped_dataset_full_coverage.py

- **Ruta**: `scripts/archive/invalid_warping/generate_warped_dataset_full_coverage.py`
- **Lineas/Tamano**: 341 lineas / ~12.0 KB
- **Proposito**: Genera dataset warped con `use_full_coverage=True` para obtener ~96% fill rate, resolviendo el sesgo metodologico identificado en Sesion 35 donde el dataset original tenia ~47% fill rate. Usa un solo modelo de checkpoint (no ensemble).
- **Importancia**: BAJO
- **Justificacion**: Supersedido por `python -m src_v2 generate-dataset --config configs/warping_best.json` que usa el ensemble completo y cached predictions. Este script usa un solo checkpoint (`seed123/final_model.pt`) en lugar del ensemble, lo que produce landmarks de menor calidad. Ubicado en `invalid_warping/` lo que sugiere que ya fue marcado como invalido.

---

## Resumen por Importancia

| Importancia | Cantidad | Archivos |
|-------------|----------|----------|
| CRITICO     | 0        | - |
| ALTO        | 0        | - |
| MEDIO       | 5        | experiment_black_background.py, session30_cross_evaluation.py, test_robustness_artifacts.py, test_robustness_geometric.py, generate_original_cropped_47.py, gradcam_pfs_analysis.py |
| BAJO        | 34       | (ver lista completa arriba) |
| ELIMINABLE  | 1        | compare_classifiers.py |

**Nota**: Los 5 archivos MEDIO contienen resultados cientificos relevantes para la tesis (robustez, cross-evaluation, PFS, experimento control). Sin embargo, ninguno es necesario para el pipeline actual ya que sus resultados estan documentados y las funcionalidades clave estan integradas en `src_v2`.

## Patrones Observados

1. **Codigo duplicado masivo**: Los scripts de entrenamiento (train_all_architectures.py, train_baseline_original_15k.py, train_classifier_original.py, train_expanded_dataset.py, train_resnet18_expanded.py) contienen loops de entrenamiento, funciones `create_model()`, `train_epoch()`, `evaluate()` practicamente identicos. Esto suma ~2,677 lineas de codigo duplicado.

2. **Imports legacy**: Varios scripts (experiment_extended_margins.py, session31_generate_dataset_margin125.py, warp_dataset3.py) usan `scripts.predict.EnsemblePredictor` y `scripts.piecewise_affine_warp` que ya no son los modulos actuales.

3. **Paths hardcoded**: Todos los scripts usan paths hardcoded a modelos de sesiones especificas (session27_models, session28_baseline_original, etc.) que no corresponden a la estructura actual del proyecto.

4. **Evolucion temporal**: Se observa la evolucion del pipeline desde scripts standalone (sesiones 22-25) hacia el sistema modular actual (src_v2). Los scripts de sesiones mas recientes (36-37) son de validacion externa, no del pipeline principal.

5. **Redundancia de Grad-CAM**: Hay 4 implementaciones de Grad-CAM (gradcam_comparison.py, gradcam_multi_architecture.py, gradcam_pfs_analysis.py, y dentro de validation_session26.py) con grados crecientes de sofisticacion.

## Recomendacion General

Todos los archivos estan correctamente archivados en `scripts/archive/`. Ninguno es necesario para el pipeline actual. Los 5 archivos MEDIO podrian conservarse como referencia historica de resultados experimentales, pero no requieren mantenimiento. El directorio completo podria eliminarse sin afectar la funcionalidad del proyecto, siempre que los resultados clave ya esten documentados en `docs/sesiones/` y `docs/reportes/`.
