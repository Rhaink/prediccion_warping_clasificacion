# Progreso del Proyecto

Ultima actualizacion: 2026-01-04 (Fase 6 COMPLETADA)

## Fase 0: Reorganizacion

- [x] Crear estructura de directorios
- [x] Crear README.md
- [x] Crear checklist/PROGRESS.md
- [x] Crear docs/01_MATEMATICAS.md
- [x] Crear docs/02_PIPELINE.md
- [x] Crear docs/03_ASESOR_CHECKLIST.md
- [x] Mover archivos a config/
- [x] Crear docs/00_OBJETIVOS.md
- [x] Eliminar archivos redundantes (PLAN.md, TASKS.md, WEEKLY_CHECKLIST.md, STATE.md, NOTA_OBJETIVOS.md)
- [x] Commit de reorganizacion (2285dedf)

## Fase 1: Preparacion de Datos (COMPLETADA)

- [x] Definir objetivo y criterios de evidencia
- [x] Inventario de datos (ver `results/logs/00_dataset_report.txt`)
- [x] Definir escenarios 2-clases y 3-clases
- [x] Crear splits fijos con seed documentada
- [x] Documentar balanceo (ver `results/logs/01_full_balance_summary.txt`)

Entregables generados:
- `results/metrics/00_dataset_counts.csv`
- `results/metrics/00_dataset_splits_manual.csv`
- `results/metrics/00_dataset_splits_full_original.csv`
- `results/metrics/01_full_balanced_3class_original.csv`
- `results/metrics/01_full_balanced_3class_warped.csv`
- `results/metrics/02_full_balanced_2class_original.csv`
- `results/metrics/02_full_balanced_2class_warped.csv`
- `results/metrics/03_manual_warped.csv` (NUEVO)
- `results/metrics/03_manual_original.csv` (NUEVO)

## Fase 2: Visualizacion para Asesor (COMPLETADA)

- [x] Crear `src/visualize.py` con funciones de visualizacion profesional
- [x] Generar 6 paneles dataset manual: Original vs Normalizada
- [x] Generar 6 paneles dataset completo: Original vs Normalizada
- [ ] Enviar muestras al asesor para revision

Entregables (reorganizados):
- `results/figures/phase2_samples/panel_manual_01.png` a `panel_manual_06.png`
- `results/figures/phase2_samples/panel_full_01.png` a `panel_full_06.png`

## Fase 3: Implementacion PCA (COMPLETADA)

- [x] Crear `src/data_loader.py`
  - Cargar imagenes como vectores
  - Separar por split (train/val/test)
  - Mapeo de etiquetas (2 clases: Enfermo/Sano)
  - Normalizacion de intensidad a [0, 1]
- [x] Crear `src/pca.py`
  - PCA desde cero con explicacion matematica detallada
  - Truco de covarianza pequena (N x N en lugar de D x D)
  - Calculo con SOLO datos de training
- [x] Crear `src/generate_all_pca_figures.py` - Procesa los 4 datasets
- [x] Crear `src/create_manual_csvs.py` - Genera CSVs para dataset manual
- [x] Crear `src/verify_eigenface_visualization.py` - Verificacion cientifica

### Datasets Procesados (4 escenarios):

| Dataset | Train | Val | Test | Dimensiones | CSV |
|---------|-------|-----|------|-------------|-----|
| Full Warped | 5,040 | 1,005 | 680 | 224x224 | 01_full_balanced_3class_warped.csv |
| Full Original | 5,040 | 1,005 | 680 | 299x299 | 01_full_balanced_3class_original.csv |
| Manual Warped | 717 | 144 | 96 | 224x224 | 03_manual_warped.csv |
| Manual Original | 717 | 144 | 96 | 299x299 | 03_manual_original.csv |

### Resultados Clave:

| Dataset | PC1 (%) | Top 10 (%) | K@95% |
|---------|---------|------------|-------|
| Full Warped | 46.4 | 82.0 | 50 |
| Full Original | 27.1 | 72.3 | 50 |
| Manual Warped | 49.0 | 82.0 | 50 |
| Manual Original | 28.3 | 72.7 | 50 |

**Hallazgo principal**: El warping concentra ~70% mas varianza en PC1 (46-49% vs 27-28%).
Esto es consistente tanto en Full como en Manual dataset.

### Verificacion Cientifica:

- Varianza del fondo en imagenes warped = 0 (confirmado)
- Visualizacion con fondo negro es cientificamente valida para warped
- Para originales se usa normalizacion min-max estandar
- Referencia: Turk & Pentland (1991), scikit-learn eigenfaces example

### Entregables:

```
results/figures/phase3_pca/
├── full_warped/
│   ├── mean_face.png
│   ├── eigenfaces.png
│   └── variance.png
├── full_original/
│   └── (mismas 3 figuras)
├── manual_warped/
│   └── (mismas 3 figuras)
├── manual_original/
│   └── (mismas 3 figuras)
├── comparisons/
│   ├── eigenfaces_4datasets.png   <- CLAVE para asesor
│   ├── variance_4datasets.png     <- CLAVE para asesor
│   └── summary_table.png
└── verification/
    └── eigenface_visualization_comparison.png
```

## Fase 4: Caracteristicas y Estandarizacion (COMPLETADA)

- [x] Crear `src/features.py`
  - StandardScaler desde cero (sin sklearn)
  - Funciones de verificacion y visualizacion
  - Documentacion matematica completa
- [x] Crear `src/generate_features.py` - Procesa los 4 datasets
- [x] Extraer ponderantes (pesos) para todas las imagenes
  - 50 componentes PCA por imagen
  - Guardados en `results/metrics/phase4_features/{dataset}_{split}_features.csv`
- [x] Implementar estandarizacion Z-score
  - Media/sigma calculados SOLO con training
  - Aplicado a train, val, test
- [x] Verificar media~0 y std~1 en training
  - Entregables: `results/figures/phase4_features/{dataset}/distribution.png`

### Resultados de Verificacion:

| Dataset | Train | Val | Test | Media≈0? | Std≈1? |
|---------|-------|-----|------|----------|--------|
| full_warped | 5,040 | 1,005 | 680 | ✓ | ✓ |
| full_original | 5,040 | 1,005 | 680 | ✓ | ✓ |
| manual_warped | 717 | 144 | 96 | ✓ | ✓ |
| manual_original | 717 | 144 | 96 | ✓ | ✓ |

### Entregables:

```
results/
├── metrics/phase4_features/
│   ├── full_warped_train_features.csv
│   ├── full_warped_val_features.csv
│   ├── full_warped_test_features.csv
│   ├── full_original_train_features.csv
│   ├── full_original_val_features.csv
│   ├── full_original_test_features.csv
│   ├── manual_warped_train_features.csv
│   ├── manual_warped_val_features.csv
│   ├── manual_warped_test_features.csv
│   ├── manual_original_train_features.csv
│   ├── manual_original_val_features.csv
│   ├── manual_original_test_features.csv
│   └── summary.json
└── figures/phase4_features/
    ├── full_warped/
    │   ├── distribution.png
    │   ├── scaler_params.png
    │   └── verification_stats.json
    ├── full_original/
    │   └── (mismas 3 figuras)
    ├── manual_warped/
    │   └── (mismas 3 figuras)
    └── manual_original/
        └── (mismas 3 figuras)
```

## Fase 5: Fisher (COMPLETADA)

- [x] Crear `src/fisher.py`
  - Clase FisherRatio implementada desde cero (sin sklearn)
  - Documentacion matematica completa en docstrings
  - Formula: J = (mu1-mu2)^2 / (sigma1^2 + sigma2^2)
- [x] Crear `src/generate_fisher.py` - Procesa los 4 datasets
- [x] Calcular Fisher ratio por caracteristica
  - Entregables: `results/metrics/phase5_fisher/{dataset}_fisher_ratios.csv`
- [x] Visualizar Fisher ratios (top-K)
  - Entregables: `results/figures/phase5_fisher/{dataset}/fisher_ratios.png`
  - Entregables: `results/figures/phase5_fisher/{dataset}/class_separation.png`
- [x] Amplificar caracteristicas multiplicando por Fisher
  - Entregables: `results/metrics/phase5_fisher/{dataset}_{split}_amplified.csv`

### Resultados Clave:

| Dataset | J max | J mean | Top 3 PCs |
|---------|-------|--------|-----------|
| full_warped | 0.262 | 0.018 | PC1, PC3, PC2 |
| full_original | 0.357 | 0.020 | PC2, PC5, PC4 |
| manual_warped | 0.300 | 0.019 | PC1, PC3, PC6 |
| manual_original | 0.259 | 0.018 | PC2, PC3, PC5 |

**Hallazgo principal**:
- En datasets WARPED: **PC1** tiene el mejor Fisher ratio (mejor separacion)
- En datasets ORIGINALES: **PC2** tiene el mejor Fisher ratio
- Esto sugiere que el warping reorganiza la informacion discriminativa hacia PC1

### Entregables:

```
results/
├── metrics/phase5_fisher/
│   ├── full_warped_fisher_ratios.csv
│   ├── full_warped_train_amplified.csv
│   ├── full_warped_val_amplified.csv
│   ├── full_warped_test_amplified.csv
│   ├── full_original_fisher_ratios.csv
│   ├── full_original_{train,val,test}_amplified.csv
│   ├── manual_warped_fisher_ratios.csv
│   ├── manual_warped_{train,val,test}_amplified.csv
│   ├── manual_original_fisher_ratios.csv
│   ├── manual_original_{train,val,test}_amplified.csv
│   └── summary.json
└── figures/phase5_fisher/
    ├── full_warped/
    │   ├── fisher_ratios.png
    │   ├── class_separation.png
    │   └── amplification_effect.png
    ├── full_original/
    │   └── (mismas 3 figuras)
    ├── manual_warped/
    │   └── (mismas 3 figuras)
    ├── manual_original/
    │   └── (mismas 3 figuras)
    └── comparisons/
        ├── fisher_4datasets.png      <- CLAVE para asesor
        ├── warped_vs_original.png    <- CLAVE para asesor
        └── summary_table.png
```

## Fase 6: Clasificacion KNN (COMPLETADA)

- [x] Crear `src/classifier.py` (KNN)
  - Clase KNNClassifier desde cero (sin sklearn)
  - Documentacion matematica completa en docstrings
  - Funciones de evaluacion: accuracy, precision, recall, F1
  - Funciones de visualizacion: matriz de confusion, comparaciones
- [x] Crear `src/generate_classification.py` - Procesa los 4 datasets
- [x] Entrenar y evaluar con imagenes WARPED
- [x] Comparar CON vs SIN warping (usar imagenes originales)
- [x] Generar matrices de confusion

### Resultados Clave:

| Dataset | K optimo | Val Acc | Test Acc | Macro F1 |
|---------|----------|---------|----------|----------|
| full_warped | 11 | 84.58% | **81.47%** | 0.804 |
| full_original | 15 | 79.40% | 79.26% | 0.779 |
| manual_warped | 5 | 76.39% | **71.88%** | 0.719 |
| manual_original | 31 | 75.69% | 66.67% | 0.666 |

### Comparacion Warped vs Original:

| Dataset | Warped | Original | Mejora |
|---------|--------|----------|--------|
| Full | 81.47% | 79.26% | **+2.21%** |
| Manual | 71.88% | 66.67% | **+5.21%** |

**Hallazgo principal**: El warping mejora la clasificacion en ambos datasets:
- En el dataset Full: +2.21% de accuracy
- En el dataset Manual: +5.21% de accuracy (mejora mas significativa)

### Metricas por Clase (Test Set):

**Full Warped (mejor modelo):**
- Enfermo: Precision=0.83, Recall=0.88, F1=0.85
- Normal: Precision=0.79, Recall=0.72, F1=0.76

**Observacion**: El modelo detecta mejor a los enfermos (recall alto),
pero tiene mas falsos positivos en normales.

### Entregables:

```
results/
├── metrics/phase6_classification/
│   ├── full_warped_results.csv
│   ├── full_warped_predictions.csv
│   ├── full_original_results.csv
│   ├── full_original_predictions.csv
│   ├── manual_warped_results.csv
│   ├── manual_warped_predictions.csv
│   ├── manual_original_results.csv
│   ├── manual_original_predictions.csv
│   ├── comparison_summary.csv        <- CLAVE para asesor
│   └── summary.json
└── figures/phase6_classification/
    ├── full_warped/
    │   ├── k_optimization.png
    │   ├── confusion_matrix.png
    │   └── confusion_matrix_normalized.png
    ├── full_original/
    │   └── (mismas 3 figuras)
    ├── manual_warped/
    │   └── (mismas 3 figuras)
    ├── manual_original/
    │   └── (mismas 3 figuras)
    └── comparisons/
        ├── metrics_4datasets.png           <- CLAVE para asesor
        ├── warped_vs_original_full.png     <- CLAVE para asesor
        ├── warped_vs_original_manual.png
        └── confusion_matrices_4datasets.png <- CLAVE para asesor
```

## Fase 7: Experimento 2 vs 3 Clases

- [ ] Repetir pipeline con 3 clases
- [ ] Tabla comparativa de 4 escenarios:
  - 2C sin warp / 2C con warp / 3C sin warp / 3C con warp
  - Entregable: `results/metrics/comparacion_final.csv`

## Fase 8: Documentacion Final

- [ ] Analisis de errores (casos mal clasificados)
  - Entregable: `results/logs/analisis_errores.txt`
- [ ] Documento final con matematicas explicadas
  - Entregable: Documento en Documentos/docs/

---

## Notas de Sesion

### 2025-12-30

- Inicio de reorganizacion de archivos
- Creacion de nueva estructura de directorios

### 2025-12-31 (Sesion 1)

- Fase 3 implementada, pendiente de revision
- Creados: `data_loader.py`, `pca.py`, `generate_pca_figures.py`
- Analisis profundo del problema de visualizacion (fondo gris variable)
- Solucion: normalizar solo contenido, fondo negro fijo
- El fondo negro de imagenes warped NO afecta PCA (varianza 0)
- Hallazgo: warping concentra varianza (PC1: 46% vs 27% en original)

### 2025-12-31 (Sesion 2 - Revision Fase 3)

**Revision solicitada:**
1. Estandares academicos - VERIFICADO
2. Cobertura de datasets - CORREGIDO (ahora 4 datasets)
3. Alineacion con plan asesor - VERIFICADO
4. Figuras para asesor - REGENERADAS

**Acciones tomadas:**
- Creados CSVs para dataset manual (warped y original) desde estructura existente
- Reorganizada estructura de `results/figures/` con subdirectorios claros
- Verificada implementacion cientifica de eigenfaces vs sklearn
- Creado script unificado `generate_all_pca_figures.py` para 4 datasets
- Generadas figuras comparativas de 4 escenarios

**Resultado:**
- Fase 3 marcada como COMPLETADA
- Figuras clave para asesor: `eigenfaces_4datasets.png`, `variance_4datasets.png`
- Evidencia clara del beneficio del warping en los 4 escenarios

### 2025-12-31 (Sesion 3 - Fase 4)

**Implementacion de Fase 4: Caracteristicas y Estandarizacion**

1. Creado `src/features.py`:
   - Clase StandardScaler desde cero (sin sklearn)
   - Estandarizacion Z-score con documentacion matematica
   - Funciones de verificacion y visualizacion

2. Creado `src/generate_features.py`:
   - Procesa los 4 datasets automaticamente
   - Aplica PCA (50 componentes) + Z-score
   - Genera CSVs y figuras de verificacion

3. Resultados:
   - 12 CSVs generados (4 datasets x 3 splits)
   - Verificacion exitosa: media≈0 y std≈1 en todos los trainings
   - Figuras de distribucion para cada dataset

**Observaciones tecnicas:**
- Media de training: <10^-8 (esencialmente 0)
- Std de training: 1.000000 (exacto)
- Val/Test pueden diferir de 0/1 (es normal y esperado)

**Resultado:**
- Fase 4 marcada como COMPLETADA
- Listos para Fase 5 (Fisher Ratio)

### 2025-12-31 (Sesion 4 - Fase 5)

**Implementacion de Fase 5: Criterio de Fisher**

1. Creado `src/fisher.py`:
   - Clase FisherRatio desde cero (sin sklearn)
   - Formula: J = (mu1-mu2)^2 / (sigma1^2 + sigma2^2)
   - Documentacion matematica extensa en docstrings
   - Funciones de visualizacion: histogramas por clase, barras de ratios
   - Funcion de amplificacion: X_amp = X * J

2. Creado `src/generate_fisher.py`:
   - Procesa los 4 datasets automaticamente
   - Carga caracteristicas de Fase 4
   - Calcula Fisher usando SOLO training
   - Amplifica train/val/test con los mismos ratios
   - Genera 3 figuras por dataset + 3 comparativas

3. Resultados:
   - Fisher ratios calculados para las 50 caracteristicas
   - 12 CSVs de caracteristicas amplificadas (4 datasets x 3 splits)
   - Verificacion exitosa en todos los datasets

**Hallazgo interesante:**
- En datasets WARPED: PC1 es la mejor caracteristica (J~0.26-0.30)
- En datasets ORIGINALES: PC2 es la mejor caracteristica (J~0.26-0.36)
- El warping reorganiza la varianza discriminativa hacia PC1
- Esto complementa el hallazgo de Fase 3: warping concentra varianza en PC1

**Observaciones tecnicas:**
- Fisher ratios son relativamente pequenos (max ~0.3)
- Esto indica que la separacion de clases no es perfecta
- Pero hay caracteristicas que separan mejor que otras
- La amplificacion ponderable estas diferencias

**Resultado:**
- Fase 5 marcada como COMPLETADA
- Listos para Fase 6 (Clasificacion KNN)

### 2026-01-04 (Sesion 5 - Fase 6)

**Implementacion de Fase 6: Clasificacion KNN**

1. Creado `src/classifier.py`:
   - Clase KNNClassifier desde cero (sin sklearn)
   - Distancia euclidiana vectorizada con NumPy
   - Votacion mayoritaria con desempate por vecino mas cercano
   - Documentacion matematica extensa en docstrings
   - Funciones de evaluacion completas (accuracy, precision, recall, F1)
   - Funciones de visualizacion (matrices de confusion, comparaciones)

2. Creado `src/generate_classification.py`:
   - Procesa los 4 datasets automaticamente
   - Busca K optimo usando conjunto de validacion
   - Evalua en conjunto de test
   - Genera figuras comparativas

3. Resultados:
   - El warping mejora consistentemente la clasificacion
   - Full dataset: 81.47% (warped) vs 79.26% (original) = +2.21%
   - Manual dataset: 71.88% (warped) vs 66.67% (original) = +5.21%
   - K optimo varia por dataset (5 a 31)

**Hallazgos tecnicos:**
- El warping produce mejoras mas significativas en el dataset manual (+5.21%)
- Esto puede deberse a que el dataset manual es mas pequeno y el warping
  ayuda a normalizar mejor las variaciones de pose
- El modelo tiene mejor recall para enfermos (0.88) que para normales (0.72)
- Esto es deseable en un contexto medico (mejor detectar enfermos)

**Resultado:**
- Fase 6 marcada como COMPLETADA
- Listos para Fase 7 (Experimento 2 vs 3 Clases)
