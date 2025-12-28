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

Pendiente inmediato (proxima sesion)
- Iniciar Semana 1 usando `feature/plan-fisher-warping/WEEKLY_CHECKLIST.md`.
- Confirmar ubicacion de datos y rutas reales con el usuario.
- Crear estructura `results/` y reporte de dataset cuando se tenga la ruta.

Dudas abiertas / info requerida
- Ubicacion exacta de los datos (imagenes, labels, splits previos si existen).
- Si hay landmarks o mascaras disponibles para el warping.
- Preferencias de formato para reportes (CSV, PNG, PDF).

Archivos clave
- `feature/plan-fisher-warping/PLAN.md`
- `feature/plan-fisher-warping/TASKS.md`
- `feature/plan-fisher-warping/WEEKLY_CHECKLIST.md`

Notas de trabajo
- Pedir aclaraciones cuando no se conozcan rutas/parametros.
- Mantener commits pequenos por cambio.
