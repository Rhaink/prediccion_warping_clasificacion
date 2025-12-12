# REFERENCE_INDEX.md

**Índice de Referencia Completo del Proyecto**

Generado: 2025-12-12
Versión del proyecto: 2.0.0

---

## Sección 1: Resumen del Proyecto

| Propiedad | Valor |
|-----------|-------|
| **Ruta raíz** | `/home/donrobot/Projects/prediccion_warping_clasificacion/` |
| **Entry point** | `python -m src_v2` |
| **Versión** | 2.0.0 |
| **Branch actual** | audit/main |
| **Último commit** | 834e5e5 - docs: agregar documentación sesiones 42-50 y actualizar configs |

### Propósito del Proyecto

Sistema completo de predicción de landmarks anatómicos en radiografías de tórax con warping y clasificación multi-arquitectura. Incluye:
- Predicción de 98 landmarks anatómicos usando ResNet-18 + CoordAttention
- Generalized Procrustes Analysis (GPA) para normalización geométrica
- Piecewise Affine Warping para alineación de imágenes
- Clasificación jerárquica multi-arquitectura (ResNet, DenseNet, EfficientNet, ConvNeXt)
- Pipeline completo de entrenamiento, evaluación y visualización

---

## Sección 2: Código Core (src_v2/)

**Total: 27 archivos | 13,060 líneas**

### 2.1 Nivel raíz

| Archivo | Líneas | Descripción | Estado Auditoría |
|---------|--------|-------------|------------------|
| `__init__.py` | 6 | Package initialization | Pendiente |
| `__main__.py` | 15 | Entry point para ejecución como módulo | Pendiente |
| `cli.py` | 6,686 | CLI principal con 20+ comandos Click | Pendiente |
| `constants.py` | 293 | Constantes centralizadas del proyecto | Pendiente |

### 2.2 data/ (Data loading y transformaciones)

| Archivo | Líneas | Descripción | Estado Auditoría |
|---------|--------|-------------|------------------|
| `data/__init__.py` | 15 | Exports principales del módulo | Pendiente |
| `data/dataset.py` | 307 | LandmarkDataset class con augmentation | Pendiente |
| `data/transforms.py` | 390 | CLAHE, augmentation pipeline | Pendiente |
| `data/utils.py` | 281 | CSV loading, split helpers | Pendiente |

**Subtotal data/**: 993 líneas

### 2.3 models/ (Arquitecturas de modelos)

| Archivo | Líneas | Descripción | Estado Auditoría |
|---------|--------|-------------|------------------|
| `models/__init__.py` | 39 | Factory functions, model registry | Pendiente |
| `models/losses.py` | 434 | WingLoss, CombinedLoss, AdaptiveWingLoss | Pendiente |
| `models/resnet_landmark.py` | 325 | ResNet-18 + CoordAttention para landmarks | Pendiente |
| `models/classifier.py` | 394 | ImageClassifier multi-arquitectura | Pendiente |
| `models/hierarchical.py` | 368 | Modelo jerárquico multinivel | Pendiente |

**Subtotal models/**: 1,560 líneas

### 2.4 training/ (Entrenamiento)

| Archivo | Líneas | Descripción | Estado Auditoría |
|---------|--------|-------------|------------------|
| `training/__init__.py` | 13 | Exports del módulo de training | Pendiente |
| `training/trainer.py` | 432 | LandmarkTrainer con mixed precision | Pendiente |
| `training/callbacks.py` | 240 | EarlyStopping, LRScheduler, ModelCheckpoint | Pendiente |

**Subtotal training/**: 685 líneas

### 2.5 evaluation/ (Métricas y evaluación)

| Archivo | Líneas | Descripción | Estado Auditoría |
|---------|--------|-------------|------------------|
| `evaluation/__init__.py` | 19 | Exports de métricas | Pendiente |
| `evaluation/metrics.py` | 437 | Pixel error metrics, MRE, SDR | Pendiente |

**Subtotal evaluation/**: 456 líneas

### 2.6 processing/ (Procesamiento geométrico)

| Archivo | Líneas | Descripción | Estado Auditoría |
|---------|--------|-------------|------------------|
| `processing/__init__.py` | 42 | Exports de GPA y warp | Pendiente |
| `processing/gpa.py` | 298 | Generalized Procrustes Analysis | Pendiente |
| `processing/warp.py` | 448 | Piecewise Affine Warping | Pendiente |

**Subtotal processing/**: 788 líneas

### 2.7 visualization/ (Visualización y análisis)

| Archivo | Líneas | Descripción | Estado Auditoría |
|---------|--------|-------------|------------------|
| `visualization/__init__.py` | 45 | Exports de visualización | Pendiente |
| `visualization/gradcam.py` | 376 | Grad-CAM visualization | Pendiente |
| `visualization/pfs_analysis.py` | 630 | Progression-Free Survival analysis | Pendiente |
| `visualization/error_analysis.py` | 478 | Error heatmaps, distribution analysis | Pendiente |

**Subtotal visualization/**: 1,529 líneas

### 2.8 utils/ (Utilidades)

| Archivo | Líneas | Descripción | Estado Auditoría |
|---------|--------|-------------|------------------|
| `utils/__init__.py` | 5 | Exports de utilidades | Pendiente |
| `utils/geometry.py` | 44 | Helpers de geometría | Pendiente |

**Subtotal utils/**: 49 líneas

---

## Sección 3: Tests (tests/)

**Total: 21 archivos | 11,778 líneas**

| Archivo | Líneas | Descripción | Estado Auditoría |
|---------|--------|-------------|------------------|
| `conftest.py` | 507 | Fixtures compartidas pytest | Pendiente |
| `__init__.py` | 1 | Package init | Pendiente |
| `test_callbacks.py` | 276 | Tests de callbacks (EarlyStopping, etc) | Pendiente |
| `test_classifier.py` | 484 | Tests de ImageClassifier | Pendiente |
| `test_cli_integration.py` | 2,748 | Tests de integración CLI | Pendiente |
| `test_cli.py` | 1,142 | Tests unitarios CLI | Pendiente |
| `test_constants.py` | 300 | Tests de constantes | Pendiente |
| `test_cross_evaluation_classes.py` | 254 | Tests evaluación cruzada | Pendiente |
| `test_evaluation_metrics.py` | 410 | Tests de métricas | Pendiente |
| `test_fill_rate_full_coverage.py` | 184 | Tests de cobertura completa | Pendiente |
| `test_losses.py` | 468 | Tests de funciones de pérdida | Pendiente |
| `test_lung_masks_integration.py` | 526 | Tests integración máscaras pulmonares | Pendiente |
| `test_optimize_margin_integration.py` | 826 | Tests optimización de márgenes | Pendiente |
| `test_pfs_integration.py` | 527 | Tests integración PFS | Pendiente |
| `test_pipeline.py` | 405 | Tests pipeline completo | Pendiente |
| `test_processing.py` | 901 | Tests GPA y warping | Pendiente |
| `test_robustness_comparative.py` | 315 | Tests comparativos de robustez | Pendiente |
| `test_trainer.py` | 283 | Tests de LandmarkTrainer | Pendiente |
| `test_transforms.py` | 409 | Tests de transformaciones | Pendiente |
| `test_visualization.py` | 554 | Tests de visualización | Pendiente |
| `test_warp_mask_consistency.py` | 258 | Tests consistencia warp-mask | Pendiente |

### Cobertura de Tests

Los tests cubren:
- Unidades individuales (models, losses, metrics)
- Integración (CLI, pipeline completo, warping)
- Regresión (robustez, validación cruzada)
- Validación de datos (constantes, configuraciones)

---

## Sección 4: Documentación Principal

| Archivo | Descripción |
|---------|-------------|
| `README.md` | Introducción y guía rápida del proyecto |
| `REPRODUCIBILITY.md` | Guía de reproducibilidad de resultados |
| `CONTRIBUTING.md` | Guía de contribución al proyecto |
| `CHANGELOG.md` | Registro de cambios por versión |
| `pyproject.toml` | Configuración del proyecto (build, dependencies) |
| `requirements.txt` | Dependencias Python |

### Reportes de Verificación

| Archivo | Descripción |
|---------|-------------|
| `referencia_auditoria.md` | Documento de referencia para auditoría |
| `REPORTE_REVISION_ENSEMBLE_TTA.md` | Verificación ensemble y TTA |
| `REPORTE_VERIFICACION_01_analisis_exploratorio.md` | Verificación análisis exploratorio |
| `REPORTE_VERIFICACION_03_funciones_perdida.md` | Verificación funciones de pérdida |
| `REPORTE_VERIFICACION_06_optimizacion_arquitectura.md` | Verificación optimización |
| `REPORTE_VERIFICACION_DESCUBRIMIENTOS_GEOMETRICOS.md` | Verificación descubrimientos |
| `REPORTE_VERIFICACION_DOCS_16_17.md` | Verificación docs finales |
| `RESUMEN_REVISION_ENSEMBLE_TTA.md` | Resumen revisión ensemble |
| `RESUMEN_VERIFICACION_03.md` | Resumen verificación cap. 3 |
| `RESUMEN_VERIFICACION_06.md` | Resumen verificación cap. 6 |
| `RESUMEN_VERIFICACION.md` | Resumen general de verificación |

---

## Sección 5: Documentación Académica (documentación/)

**Total: 21 archivos LaTeX | 13,820 líneas**

### Capítulos Principales (00-17)

| Archivo | Descripción |
|---------|-------------|
| `00_preambulo.tex` | Preámbulo y configuración LaTeX |
| `01_analisis_exploratorio_datos.tex` | Análisis exploratorio del dataset |
| `02_arquitectura_modelo_landmarks.tex` | Arquitectura ResNet + CoordAttention |
| `03_funciones_perdida.tex` | WingLoss y variantes |
| `04_preprocesamiento_clahe.tex` | CLAHE y preprocesamiento |
| `05_entrenamiento_dos_fases.tex` | Estrategia de entrenamiento |
| `06_optimizacion_arquitectura.tex` | Optimización de hiperparámetros |
| `07_ensemble_tta.tex` | Ensemble y Test-Time Augmentation |
| `08_arquitectura_jerarquica.tex` | Modelo jerárquico multinivel |
| `09_descubrimientos_geometricos.tex` | Hallazgos geométricos |
| `10_analisis_procrustes_gpa.tex` | Generalized Procrustes Analysis |
| `11_warping_piecewise_affine.tex` | Transformación piecewise affine |
| `12_generacion_dataset_warpeado.tex` | Generación de dataset warpeado |
| `13_clasificacion_multi_arquitectura.tex` | Clasificación multi-arquitectura |
| `14_validacion_cruzada.tex` | Validación cruzada |
| `15_analisis_robustez.tex` | Análisis de robustez |
| `16_validacion_externa.tex` | Validación externa |
| `17_resultados_consolidados.tex` | Resultados finales consolidados |

### Apéndices (A-C)

| Archivo | Descripción |
|---------|-------------|
| `A_derivaciones_matematicas.tex` | Derivaciones matemáticas detalladas |
| `B_hiperparametros_configuraciones.tex` | Tablas de hiperparámetros |
| `C_codigo_fuente.tex` | Referencias al código fuente |

---

## Sección 6: Sesiones de Desarrollo (docs/sesiones/)

**Total: 52 archivos Markdown documentados**

Las sesiones documentan el desarrollo completo del proyecto desde SESION_00 hasta SESION_50, incluyendo:
- Análisis exploratorio inicial
- Desarrollo de arquitecturas
- Experimentación con losses
- Implementación de GPA y warping
- Desarrollo del clasificador jerárquico
- Validación cruzada y robustez
- Optimizaciones y mejoras

Ruta: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/sesiones/`

---

## Sección 7: Scripts Auxiliares (scripts/)

**Total: 84 archivos Python | 38,532 líneas**

Los scripts auxiliares cubren:
- Experimentación y prototipos
- Generación de datasets
- Análisis de resultados
- Visualizaciones específicas
- Utilidades de pre/post-procesamiento
- Scripts de validación y verificación
- Herramientas de debugging

Ruta: `/home/donrobot/Projects/prediccion_warping_clasificacion/scripts/`

---

## Sección 8: Estadísticas Globales

| Componente | Archivos | Líneas | Porcentaje |
|------------|----------|--------|------------|
| **src_v2/** | 27 | 13,060 | 16.9% |
| **tests/** | 21 | 11,778 | 15.3% |
| **scripts/** | 84 | 38,532 | 49.9% |
| **documentación/** | 21 | 13,820 | 17.9% |
| **TOTAL** | **153** | **77,190** | **100%** |

### Desglose por Módulo (src_v2/)

| Módulo | Archivos | Líneas | Porcentaje |
|--------|----------|--------|------------|
| cli.py | 1 | 6,686 | 51.2% |
| visualization/ | 4 | 1,529 | 11.7% |
| models/ | 5 | 1,560 | 11.9% |
| data/ | 4 | 993 | 7.6% |
| processing/ | 3 | 788 | 6.0% |
| training/ | 3 | 685 | 5.2% |
| evaluation/ | 2 | 456 | 3.5% |
| constants.py | 1 | 293 | 2.2% |
| utils/ | 2 | 49 | 0.4% |
| Otros | 2 | 21 | 0.2% |

### Desglose Tests por Tipo

| Tipo | Archivos | Líneas | Porcentaje |
|------|----------|--------|------------|
| Integración | 7 | 6,032 | 51.2% |
| Unitarios | 13 | 5,239 | 44.5% |
| Fixtures | 1 | 507 | 4.3% |

---

## Sección 9: Comandos CLI Disponibles

El archivo `cli.py` (6,686 líneas) implementa 20+ comandos principales:

### Comandos de Entrenamiento
- `train` - Entrenamiento de modelo landmark
- `train-classifier` - Entrenamiento de clasificador
- `train-hierarchical` - Entrenamiento jerárquico

### Comandos de Procesamiento
- `gpa` - Generalized Procrustes Analysis
- `warp` - Piecewise Affine Warping
- `generate-warped-dataset` - Generación dataset warpeado

### Comandos de Evaluación
- `evaluate` - Evaluación de modelo
- `cross-validate` - Validación cruzada
- `evaluate-classifier` - Evaluación clasificador
- `robustness` - Análisis de robustez

### Comandos de Visualización
- `visualize` - Visualización general
- `gradcam` - Grad-CAM
- `error-analysis` - Análisis de errores
- `pfs-analysis` - Análisis PFS

### Comandos de Utilidades
- `optimize-clahe` - Optimización CLAHE
- `optimize-margin` - Optimización márgenes
- `lung-masks` - Generación máscaras pulmonares
- `export-landmarks` - Exportar landmarks
- `predict` - Predicción sobre nuevas imágenes

---

## Sección 10: Dependencias y Configuración

### Dependencias Principales (requirements.txt)

- **Deep Learning**: torch, torchvision, timm
- **Visión por Computadora**: opencv-python, scikit-image, albumentations
- **Científicas**: numpy, scipy, pandas
- **Visualización**: matplotlib, seaborn, plotly
- **CLI**: click, rich, tqdm
- **Testing**: pytest, pytest-cov
- **Métricas**: scikit-learn, lifelines (PFS)

### Configuración del Proyecto (pyproject.toml)

- **Build System**: setuptools
- **Python Version**: >=3.8
- **Package Name**: prediccion-warping-clasificacion
- **Version**: 2.0.0

---

## Sección 11: Estructura de Directorios Completa

```
/home/donrobot/Projects/prediccion_warping_clasificacion/
│
├── src_v2/                      # Código fuente principal (13,060 líneas)
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py                   # CLI principal (6,686 líneas)
│   ├── constants.py             # Constantes (293 líneas)
│   ├── data/                    # Data loading (993 líneas)
│   ├── models/                  # Arquitecturas (1,560 líneas)
│   ├── training/                # Entrenamiento (685 líneas)
│   ├── evaluation/              # Métricas (456 líneas)
│   ├── processing/              # GPA/Warp (788 líneas)
│   ├── visualization/           # Visualización (1,529 líneas)
│   └── utils/                   # Utilidades (49 líneas)
│
├── tests/                       # Tests (21 archivos, 11,778 líneas)
│   ├── conftest.py
│   ├── test_*.py (20 archivos)
│   └── __init__.py
│
├── scripts/                     # Scripts auxiliares (84 archivos, 38,532 líneas)
│
├── docs/                        # Documentación de desarrollo
│   └── sesiones/                # 52 sesiones documentadas
│
├── documentación/               # Documentación académica (21 archivos LaTeX, 13,820 líneas)
│   ├── 00-17 (capítulos)
│   └── A-C (apéndices)
│
├── audit/                       # Directorio de auditoría
│   └── REFERENCE_INDEX.md       # Este archivo
│
├── README.md
├── REPRODUCIBILITY.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── pyproject.toml
└── requirements.txt
```

---

## Sección 12: Próximos Pasos de Auditoría

### Fase 1: Revisión de Código Core
1. Auditar `cli.py` (6,686 líneas) - comando por comando
2. Revisar módulos de models/ (5 archivos, 1,560 líneas)
3. Validar training/ y evaluation/ (5 archivos, 1,141 líneas)
4. Verificar processing/ (GPA/Warp - 3 archivos, 788 líneas)

### Fase 2: Validación de Tests
1. Revisar cobertura de tests (21 archivos, 11,778 líneas)
2. Ejecutar suite completa de tests
3. Verificar fixtures y mocks (conftest.py)

### Fase 3: Consistencia Documentación-Código
1. Comparar documentación LaTeX con implementación
2. Verificar fórmulas matemáticas
3. Validar resultados reportados

### Fase 4: Scripts y Reproducibilidad
1. Auditar scripts auxiliares (84 archivos)
2. Verificar reproducibilidad de experimentos
3. Validar configuraciones y constantes

---

## Notas Finales

- **Generado**: 2025-12-12
- **Método**: Análisis automático con `find`, `wc -l`, `ls`
- **Precisión**: Conteos exactos de archivos y líneas
- **Scope**: Proyecto completo excepto datos binarios y modelos guardados

**Estado**: Este índice proporciona una referencia completa y precisa de todos los componentes del proyecto para facilitar la auditoría sistemática.
