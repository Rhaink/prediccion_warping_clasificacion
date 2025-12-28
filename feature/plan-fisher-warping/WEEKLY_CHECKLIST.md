Checklist semanal (referencia desde hoy, trabajo diario)

Semana 1
- Definir objetivo y criterios de evidencia.
- Enfatizar que la mejora debe demostrarse con clasificador simple (no CNN como prueba principal).
- Aclarar que se trabajara paso a paso con explicaciones de matematicas, geometria y conceptos abstractos.
- Registrar dudas abiertas y solicitar informacion faltante al usuario; no asumir rutas ni datos.
- Documentar cada avance y hacer commits por cambios para mantener orden.
- Inventario de datos: conteos por clase y split; reporte de dataset.
- Definir escenarios 2-clases y 3-clases; documentar mapeos.
- Crear split fijo (train/val/test) con seed documentada.
- Entregables: nota de objetivos; CSVs de conteos y splits; reporte en `results/logs/00_dataset_report.txt`.

Semana 2
- Preprocesamiento base: normalizacion geometrica y CLAHE con comparativos.
- Documentar parametros de CLAHE y ejemplos visuales por clase.
- Entregables: paneles antes/despues; `results/logs/01_preprocess_params.txt`.

Semana 3
- Alineacion (warping): definir puntos de referencia y metodo geometrico.
- Aplicar warping y validar con distancia de landmarks.
- Preparar muestras warped para el asesor.
- Entregables: panel original vs warped; CSV de distancia; `results/logs/02_warping_validation.txt`; paquete de muestras warped.

Semana 4
- PCA/Eigenspace: calcular PCA con training (solo warped), seleccionar K.
- Generar curva de varianza explicada, eigenfaces y reconstrucciones.
- Entregables: figuras de varianza, eigenfaces y reconstrucciones por K.

Semana 5
- Extraer caracteristicas (ponderantes) y visualizar dispersions principales.
- Estandarizar caracteristicas con estadisticas del training.
- Entregables: tabla de ponderantes; scatter 2D/3D; distribuciones antes/despues; verificacion media~0/std~1.

Semana 6
- Calcular criterio de Fisher por dimension y aplicar ponderacion despues de estandarizar.
- Entregables: tabla ordenada de Fisher; grafica top-K.

Semana 7
- Clasificacion simple: KNN y/o MLP pequeno.
- Evaluar con y sin Fisher, con y sin warping.
- Entregables: accuracy, F1 macro, matrices de confusion; comparativas.

Semana 8
- Experimento 2-clases vs 3-clases: repetir pipeline y comparar.
- Entregables: tabla comparativa final (4 escenarios); figuras de separacion 2D/3D.

Semana 9
- Analisis de errores: casos mal clasificados y explicacion breve.
- Sensibilidad a K (componentes) y estabilidad de resultados.
- Entregables: tabla de errores; resumen de sensibilidad.

Semana 10
- Documentacion final: explicar matematicas y geometria sin cajas negras.
- Incluir formula explicita de Fisher y ejemplo numerico.
- Auditoria final del plan con checklist completada.
- Confirmar entrega de muestras warped al asesor si no se enviaron antes.
- Entregables: documento en `Documentos/docs/`; checklist de auditoria completa.
