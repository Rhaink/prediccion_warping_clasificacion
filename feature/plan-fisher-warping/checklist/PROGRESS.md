# Progreso del Proyecto

Ultima actualizacion: 2025-12-31 (Fase 3 COMPLETADA)

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

## Fase 4: Caracteristicas y Estandarizacion

- [ ] Extraer ponderantes (pesos) para todas las imagenes
  - Guardar en `results/metrics/ponderantes_train.csv` etc.
- [ ] Implementar estandarizacion Z-score
  - Calcular media/sigma con SOLO training
  - Aplicar a train, val, test
- [ ] Verificar media~0 y std~1 en training
  - Entregable: `results/figures/distribucion_estandarizada.png`

## Fase 5: Fisher

- [ ] Crear `src/fisher.py`
  - Implementar formula: J = (mu1-mu2)^2 / (sigma1^2 + sigma2^2)
- [ ] Calcular Fisher ratio por caracteristica
  - Entregable: `results/metrics/fisher_ratios.csv`
- [ ] Visualizar Fisher ratios (top-K)
  - Entregable: `results/figures/fisher_top_k.png`
- [ ] Amplificar caracteristicas multiplicando por Fisher

## Fase 6: Clasificacion

- [ ] Crear `src/classifier.py` (KNN)
- [ ] Entrenar y evaluar con imagenes WARPED
  - Entregable: `results/metrics/clasificacion_warped.csv`
- [ ] Comparar CON vs SIN warping (usar imagenes originales)
  - Entregable: `results/metrics/comparacion_warping.csv`
- [ ] Generar matriz de confusion
  - Entregable: `results/figures/confusion_matrix.png`

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
