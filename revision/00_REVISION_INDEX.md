# 00. Indice Maestro - Revision Completa del Proyecto

**Proyecto**: prediccion_warping_clasificacion
**Fecha de revision**: 2026-02-11
**Total de archivos analizados**: ~449 entradas individuales + grupos
**Total de secciones**: 22

---

## Resumen de Importancia

| Nivel | Cantidad | Porcentaje |
|-------|----------|------------|
| CRITICO | 70 | 16.4% |
| ALTO | 82 | 19.2% |
| MEDIO | 114 | 26.7% |
| BAJO | 133 | 31.2% |
| ELIMINABLE | 27 | 6.3% |
| **Total** | **426** | **100%** |

---

## Indice de Secciones

### Codigo Fuente Principal (src_v2)

| Seccion | Archivo | Entradas | Lineas |
|---------|---------|----------|--------|
| [03](03_src_v2_core.md) | Core (cli.py, constants, init, main) | 37 | 1,181 |
| [04](04_src_v2_data.md) | Data (dataset, transforms, utils) | 5 | 210 |
| [05](05_src_v2_models.md) | Models (ResNet18, classifier, losses) | 6 | 242 |
| [06](06_src_v2_processing.md) | Processing (GPA, warping) | 4 | 183 |
| [07](07_src_v2_training.md) | Training (trainer, callbacks) | 3 | 200 |
| [08](08_src_v2_evaluation.md) | Evaluation (metrics, ensemble) | 5 | 178 |
| [09](09_src_v2_utils.md) | Utils (geometry) | 3 | 102 |
| [10](10_src_v2_visualization.md) | Visualization (GradCAM, PFS, ROC, etc.) | 13 | 446 |
| [11](11_src_v2_gui.md) | GUI (Gradio app) | 11 | 355 |

### Configuracion y Raiz

| Seccion | Archivo | Entradas | Lineas |
|---------|---------|----------|--------|
| [01](01_root_project_files.md) | Root Project Files | 25 | 376 |
| [02](02_configs.md) | Configuration Files (JSON) | 13 | 341 |

### Scripts

| Seccion | Archivo | Entradas | Lineas |
|---------|---------|----------|--------|
| [12](12_scripts_pipeline.md) | Pipeline Scripts | 24 | 655 |
| [13](13_scripts_figure_generation.md) | Figure Generation Scripts | 49 | 550 |
| [14](14_scripts_verification.md) | Verification & Dataset Gen Scripts | 21 | 394 |
| [15](15_scripts_fisher.md) | Fisher Analysis Scripts | 13 | 244 |
| [16](16_scripts_auxiliary.md) | Auxiliary Scripts (build, deploy) | 16 | 265 |
| [17](17_scripts_archive.md) | Archived/Legacy Scripts | 42 | 368 |

### Documentacion

| Seccion | Archivo | Entradas | Lineas |
|---------|---------|----------|--------|
| [18](18_docs_technical.md) | Technical Documentation | 44 | 543 |
| [19](19_docs_thesis.md) | Thesis Documentation (LaTeX) | 41 | 522 |
| [20](20_docs_deliverables.md) | Deliverables (estancia, manual, carta) | 28 | 328 |

### Infraestructura y Datos

| Seccion | Archivo | Entradas | Lineas |
|---------|---------|----------|--------|
| [21](21_hidden_and_meta.md) | Hidden & Meta Files (.planning, .claude, results) | 41 | 467 |
| [22](22_data_checkpoints_build.md) | Data, Checkpoints, Build & Dist | 5 | 504 |

### Resumen

| Seccion | Archivo | Lineas |
|---------|---------|--------|
| [99](99_FINAL_SUMMARY.md) | Resumen Final + Recomendaciones | - |

---

## Distribucion de Importancia por Seccion

| Seccion | CRITICO | ALTO | MEDIO | BAJO | ELIMINABLE | Total |
|---------|---------|------|-------|------|------------|-------|
| 01 Root | 4 | 3 | 6 | 8 | 4 | 25 |
| 02 Configs | 5 | 0 | 2 | 5 | 0 | 12 |
| 03 Core | 9 | 5 | 13 | 8 | 0 | 35 |
| 04 Data | 3 | 1 | 0 | 0 | 0 | 4 |
| 05 Models | 3 | 1 | 0 | 1 | 0 | 5 |
| 06 Processing | 2 | 0 | 1 | 0 | 0 | 3 |
| 07 Training | 1 | 1 | 0 | 1 | 0 | 3 |
| 08 Evaluation | 1 | 1 | 1 | 0 | 0 | 3 |
| 09 Utils | 0 | 0 | 2 | 1 | 0 | 3 |
| 10 Visualization | 1 | 7 | 3 | 1 | 0 | 12 |
| 11 GUI | 4 | 1 | 1 | 3 | 0 | 9 |
| 12 Pipeline | 2 | 4 | 8 | 9 | 0 | 23 |
| 13 Figuras | 4 | 11 | 18 | 6 | 9 | 48 |
| 14 Verificacion | 2 | 6 | 4 | 5 | 2 | 19 |
| 15 Fisher | 0 | 4 | 4 | 3 | 0 | 11 |
| 16 Auxiliar | 0 | 3 | 8 | 4 | 0 | 15 |
| 17 Archive | 0 | 0 | 6 | 33 | 1 | 40 |
| 18 Docs Tech | 3 | 7 | 13 | 20 | 0 | 43 |
| 19 Tesis | 19 | 10 | 3 | 5 | 3 | 40 |
| 20 Entregables | 4 | 7 | 6 | 5 | 7 | 29 |
| 21 Hidden/Meta | 1 | 9 | 14 | 14 | 1 | 39 |
| 22 Data/Ckpt | 2 | 1 | 1 | 1 | 0 | 5 |
| **Total** | **70** | **82** | **114** | **133** | **27** | **426** |

---

## Como Usar Esta Revision

1. **Para entender el pipeline**: Leer secciones 03-08 (core + data + models + processing + training + evaluation)
2. **Para limpiar el proyecto**: Ver seccion 99 (resumen final) con recomendaciones de eliminacion
3. **Para preparar la defensa**: Leer secciones 19 (tesis) y 20 (entregables)
4. **Para evaluar deuda tecnica**: Ver seccion 99, apartado "Deuda Tecnica"
5. **Para encontrar un archivo especifico**: Buscar en la seccion correspondiente segun su ubicacion
