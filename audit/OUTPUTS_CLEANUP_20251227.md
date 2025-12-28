# Reporte de Limpieza y Reorganización de Outputs
**Fecha:** 27 de Diciembre de 2025
**Ejecutado por:** Agente Gemini CLI

## Resumen Ejecutivo
Se realizó una limpieza masiva del directorio `outputs/` para separar los entregables "Core" (esenciales) de la evidencia histórica, archivos temporales y visualizaciones secundarias. El directorio raíz pasó de tener >40 elementos a solo 9 elementos esenciales.

## 1. Elementos Eliminados
Se eliminaron carpetas obsoletas o dependientes de datasets ya borrados.
- `outputs/warped_replication/` (Dataset legacy, replicación fallida)
- `outputs/warped_subset/` (Subconjunto manual sin uso actual)
- `outputs/original_cropped_47/` (Dataset intermedio, regenerable)
- `outputs/original_3_classes/` (Dataset intermedio, regenerable)
- `outputs/classifier_replication/` (Modelo asociado a warped_replication)
- `outputs/shape_analysis_test/` (Pruebas de geometría, versión final en `shape_analysis`)

## 2. Elementos Consolidados en `results/`
Se movieron archivos de análisis, métricas y gráficos al directorio `results/` para centralizar la evidencia.

### Logs y Métricas
- `outputs/evaluation_report_*.txt` -> **`results/logs/historical/`**
- `outputs/classify_*.json` -> **`results/predictions/`**
- `outputs/robustness_*.json`, `eval_subset.json` -> **`results/metrics/`**
- `outputs/training_config.json`, `history.json` -> **`results/logs/latest_run/`**

### Visualizaciones (`results/figures/`)
- `outputs/gradcam/` -> `results/figures/gradcam/`
- `outputs/gradcam_multi/` -> `results/figures/gradcam_multi/`
- `outputs/pipeline_viz/` -> `results/figures/pipeline_viz/`
- `outputs/diagrams/` -> `results/figures/diagrams/`
- `outputs/visual_analysis/` -> `results/figures/visual_analysis/`
- `outputs/visualizations/` -> `results/figures/visualizations/`
- `outputs/flip_comparison.png`, etc. -> `results/figures/archive/`

### Validación (`results/validation/`)
- `outputs/external_validation/` -> `results/validation/external_validation/`
- `outputs/cross_evaluation_valid_3classes/` -> `results/validation/cross_evaluation_valid_3classes/`
- `outputs/pfs_warped_valid/` (y full) -> `results/validation/pfs_warped_valid/`

### Artefactos Geométricos (`results/geometry_artifacts/`)
- Contenido de `outputs/predictions/` (delaunay, landmarks, etc.) -> `results/geometry_artifacts/`

## 3. Elementos Archivados (`outputs/archive/`)
Se movieron carpetas de experimentos pasados que sirven como evidencia histórica pero no se usan activamente.
- **Sesiones:** `session9` a `session32` (todas las carpetas `session*`).
- **Experimentos:** `exp_clahe_*`, `exp_more_epochs`, `margin_experiment`, `expanded_experiment`.
- **Modelos Legacy:** `classifier` (genérico), `classifier_binary_*`, `classifier_original`, `classifier_comparison`.

## 4. Estado Final de `outputs/` (Core)
Solo permanecen los directorios fundamentales para el funcionamiento actual y la tesis.

| Carpeta | Descripción |
| :--- | :--- |
| **`full_warped_dataset/`** | Dataset principal (Warping Ground Truth, 15k imgs). |
| **`warped_dataset/`** | Dataset legacy (Warping 47% fill) para referencia histórica. |
| **`shape_analysis/`** | Definición geométrica canónica y triángulos de Delaunay. |
| **`thesis_figures/`** | Figuras finales listas para el documento de tesis. |
| **`classifier_efficientnet/`** | Modelo Warped (Referencia histórica/Legacy). |
| **`classifier_full/`** | Modelo Warped actual (Entrenado en full_warped_dataset). |
| **`classifier_original_cropped_47/`** | Modelo Control (Entrenado en original recortado). |
| **`classifier_original_3classes/`** | Modelo Baseline (Entrenado en original 3 clases). |
| **`archive/`** | Contenedor de toda la evidencia histórica. |

---
**Próximos pasos:** Actualizar scripts que apunten a rutas antiguas (ej. `outputs/gradcam`) si se planea re-ejecutarlos, o dejarlos como están sabiendo que sus salidas antiguas están archivadas.
