Estado del proyecto (persistente entre sesiones)

Resumen corto
- Proyecto: plan de trabajo para alineacion + PCA + Fisher + clasificador simple.
- Rama actual: feature/plan-fisher-warping.
- Objetivo: seguir el flujo del asesor y documentar cada paso sin cajas negras.

Hecho en esta sesion
- Se creo la rama `feature/plan-fisher-warping`.
- Se creo `feature/plan-fisher-warping/PLAN.md` con el plan detallado.
- Se creo `feature/plan-fisher-warping/TASKS.md` con tareas y entregables.
- Se creo `feature/plan-fisher-warping/WEEKLY_CHECKLIST.md` con calendario semanal.
- Se ajusto la checklist para incluir:
  - Clasificador simple como evidencia principal.
  - PCA solo con imagenes warped.
  - Fisher despues de estandarizar.
  - Entrega de muestras warped al asesor.
  - Pedir aclaraciones cuando falte info.
  - Documentar avances y hacer commits por cambios.
- Semana 1 documentada con entregables:
  - Objetivos y criterios: `feature/plan-fisher-warping/NOTA_OBJETIVOS.md`.
  - Mapeo 2-clases/3-clases: `feature/plan-fisher-warping/LABEL_MAPPING.md`.
  - Protocolo de split/balanceo: `feature/plan-fisher-warping/SPLIT_PROTOCOL.md`.
  - Reporte de dataset: `feature/plan-fisher-warping/results/logs/00_dataset_report.txt`.
  - Conteos por clase/split: `feature/plan-fisher-warping/results/metrics/00_dataset_counts.csv`.
  - Splits base: `feature/plan-fisher-warping/results/metrics/00_dataset_splits_manual.csv` y
    `feature/plan-fisher-warping/results/metrics/00_dataset_splits_full_original.csv`.
  - Balanceo full: `feature/plan-fisher-warping/results/logs/01_full_balance_summary.txt`.
  - CSVs balanceados: `feature/plan-fisher-warping/results/metrics/01_full_balanced_3class_*.csv` y
    `feature/plan-fisher-warping/results/metrics/02_full_balanced_2class_*.csv`.

Pendiente inmediato (proxima sesion)
- Iniciar Semana 2 usando `feature/plan-fisher-warping/WEEKLY_CHECKLIST.md`.

Dudas abiertas / info requerida
- Preferencias de formato para reportes (CSV, PNG, PDF).

Notas sobre landmarks/mascaras
- Landmarks: disponibles en `data/coordenadas/coordenadas_maestro.csv` (contorno pulmonar manual), no relevantes para el plan actual.
- Mascaras: existen en el dataset original, pero no se van a usar.

Archivos clave
- `feature/plan-fisher-warping/PLAN.md`
- `feature/plan-fisher-warping/TASKS.md`
- `feature/plan-fisher-warping/WEEKLY_CHECKLIST.md`

Notas de trabajo
- Pedir aclaraciones cuando no se conozcan rutas/parametros.
- Mantener commits pequenos por cambio.
