Checklist de tareas ejecutables con entregables (derivado de PLAN.md)

T1: Definir objetivo y criterios de evidencia
- Entregables: nota breve con objetivo, pregunta central y criterios de evidencia.

T2: Inventario de datos
- Entregables: tabla de conteos por clase y split (CSV); reporte breve en `results/logs/00_dataset_report.txt`.

T3: Definir escenarios 2-clases y 3-clases
- Entregables: mapeo de etiquetas y reglas de re-etiquetado documentadas.

T4: Crear split fijo (train/val/test)
- Entregables: CSVs de splits; seed y metodo de split documentados.

T5: Preprocesamiento base (sin warping)
- Entregables: panel de ejemplos antes/despues de CLAHE; parametros en `results/logs/01_preprocess_params.txt`.

T6: Alineacion (warping)
- Entregables: panel original vs warped; metrica de calidad (distancia landmarks) en CSV; notas en `results/logs/02_warping_validation.txt`; paquete de muestras warped para revision del asesor.

T7: PCA/Eigenspace
- Entregables: curva de varianza explicada; galeria de eigenfaces; reconstrucciones con distintos K.

T8: Caracteristicas (ponderantes)
- Entregables: tabla con ejemplos de ponderantes; scatter de 2-3 dimensiones principales.

T9: Estandarizacion de caracteristicas
- Entregables: distribuciones antes/despues; verificacion de media~0 y std~1 en training.

T10: Criterio de Fisher
- Entregables: tabla de Fisher por dimension (ordenada); grafica de top-K.

T11: Clasificacion simple
- Entregables: metricas (accuracy, F1 macro), matrices de confusion; comparativa con/sin Fisher y con/sin warping.

T12: Experimento 2-clases vs 3-clases
- Entregables: tabla comparativa final (4 escenarios); figuras de separacion 2D/3D.

T13: Analisis de errores
- Entregables: lista de casos mal clasificados y analisis breve por caso.

T14: Documentacion final y auditoria
- Entregables: documento en `Documentos/docs/` con explicacion matematica paso a paso + resultados, incluyendo formula explicita de Fisher y ejemplo numerico; checklist de auditoria completada.
