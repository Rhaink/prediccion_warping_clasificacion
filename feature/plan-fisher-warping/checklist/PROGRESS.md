# Progreso del Proyecto

Ultima actualizacion: 2025-12-30

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
- [ ] Commit de reorganizacion

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

## Fase 2: Visualizacion para Asesor

- [ ] Generar panel de imagenes warped (9 ejemplos: 3 COVID, 3 Normal, 3 Viral)
  - Entregable: `results/figures/panel_warped_samples.png`
- [ ] Enviar muestras al asesor para revision

## Fase 3: Implementacion PCA

- [ ] Crear `src/data_loader.py`
  - Cargar imagenes warped como vectores
  - Separar por split (train/val/test)
  - Aplicar mapeo de etiquetas (2 clases)
- [ ] Crear `src/pca.py`
  - Implementar PCA (con explicacion matematica)
  - Calcular con SOLO datos de training
- [ ] Calcular eigenfaces con training
  - Entregable: `results/figures/eigenfaces_top10.png`
- [ ] Visualizar curva de varianza explicada
  - Entregable: `results/figures/varianza_explicada.png`
- [ ] Seleccionar numero de componentes (K)

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
