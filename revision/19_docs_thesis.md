# 19. Thesis Documentation (LaTeX)

Analisis de los archivos LaTeX de la tesis de maestria.

**Archivos analizados**: ~40 .tex + artifacts + figures
**Directorio base**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/`
**Titulo**: "Normalizacion y alineacion automatica de la forma de la region pulmonar integrada con seleccion de caracteristicas discriminantes para deteccion de neumonia y COVID-19"
**Autor**: Lic. Rafael Alejandro Cruz Ovando
**Universidad**: BUAP - Maestria en Ingenieria Electronica
**Fecha**: Enero 2026

---

## Estructura general de la tesis

La tesis esta organizada en 6 capitulos + anexos, con paginas preliminares (portada, dedicatoria, agradecimientos, resumen/abstract) y un glosario al final. El documento principal `main.tex` usa `\input{}` para incluir cada seccion. Usa estilo IEEE para bibliografia con `natbib`.

### Flujo de inclusion en main.tex

```
main.tex
  |-- portada/title-page.tex
  |-- dedicatoria/dedicatoria.tex
  |-- agradecimientos/agradecimientos.tex
  |-- resumen/resumen.tex
  |-- resumen/abstract.tex
  |-- [TOC, LOF, LOT]
  |-- capitulo1/1_introduccion.tex
  |-- objetivos/0-Objetivos.tex
  |-- capitulo2/2_marco_teorico.tex
  |-- capitulo3/3_estado_del_arte.tex
  |     |-- tabla_3_1_covid19_detection.tex
  |     |-- tabla_3_2_landmark_detection.tex
  |     |-- tabla_3_3_normalizacion_geometrica.tex
  |-- capitulo4/4_1_descripcion_general.tex
  |-- capitulo4/4_2_dataset_preprocesamiento.tex
  |-- capitulo4/4_3_modelo_landmarks.tex
  |-- capitulo4/4_4_normalizacion_geometrica.tex
  |-- capitulo4/4_5_clasificacion.tex
  |-- capitulo4/4_6_inferencia_metricas.tex
  |-- capitulo5/5_1_resultados_landmarks.tex
  |-- capitulo5/5_2_forma_canonica.tex
  |-- capitulo5/5_3_resultados_clasificacion.tex  (NO la version _CV)
  |-- capitulo5/5_4_eficiencia_computacional.tex
  |-- capitulo6/6_conclusiones.tex
  |-- references.bib
  |-- anexos/anexo_B_articulos_publicados.tex
  |-- anexos/anexo_C_certificados.tex
  |-- anexos/anexo_D_codigo_fuente.tex
  |-- glosario.tex
```

**NOTA**: `anexos/anexo_A_manual_usuario.tex` esta COMENTADO en main.tex (linea 223).
**NOTA**: `setup/settings.tex` NO esta incluido en main.tex (los paquetes se cargan directamente en main.tex).
**NOTA**: `acronimos/lista_acronimos.tex` NO esta incluido en main.tex.

---

## Analisis archivo por archivo

### 1. main.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/main.tex`
- **Lineas/Tamano**: 240 lineas / 8 KB
- **Proposito**: Archivo principal de la tesis. Define documentclass (report, 12pt, letterpaper), carga todos los paquetes (babel, geometry, graphicx, booktabs, natbib, tikz, etc.), configura nombres en espanol, numeracion por capitulo y orquesta la inclusion de todos los capitulos y secciones via `\input{}`.
- **Incluido en main.tex**: Es el archivo raiz
- **Importancia**: CRITICO
- **Justificacion**: Sin este archivo no se puede compilar la tesis. Contiene toda la configuracion de paquetes y la estructura completa del documento.

### 2. setup/settings.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/setup/settings.tex`
- **Lineas/Tamano**: 98 lineas / 4 KB
- **Proposito**: Archivo alternativo de configuracion de paquetes (biblatex, fancyhdr, tocloft, parskip, chngcntr). Define estilos de pagina, contadores por seccion y formato de capitulos diferente al de main.tex.
- **Incluido en main.tex**: **No** -- No aparece ningun `\input{setup/settings}` en main.tex
- **Importancia**: ELIMINABLE
- **Justificacion**: Es una version alternativa/legacy de la configuracion. main.tex ya carga todos los paquetes necesarios directamente. Este archivo usa biblatex (style=ieee) mientras main.tex usa natbib -- son incompatibles. Contiene configuraciones que difieren de las activas (e.g., counterwithin{figure}{section} vs counterwithin{figure}{chapter}). Se puede eliminar para evitar confusion.

### 3. portada/title-page.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/portada/title-page.tex`
- **Lineas/Tamano**: 42 lineas / 4 KB
- **Proposito**: Pagina de titulo institucional de la BUAP con logo, titulo de tesis, nombre del autor, directores (Dr. Ayala Raggi, Dr. Barreto Flores), fecha y nota de becario SECIHTI.
- **Incluido en main.tex**: Si
- **Importancia**: CRITICO
- **Justificacion**: Portada obligatoria del documento de tesis.

### 4. dedicatoria/dedicatoria.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/dedicatoria/dedicatoria.tex`
- **Lineas/Tamano**: 41 lineas / 4 KB
- **Proposito**: Dedicatoria a padres, familia y amigos con lenguaje metaforico relacionado con la tematica (algoritmos, metricas, variables).
- **Incluido en main.tex**: Si
- **Importancia**: ALTO
- **Justificacion**: Seccion estilistica obligatoria en tesis de maestria BUAP.

### 5. dedicatoria/dedicatoria copy.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/dedicatoria/dedicatoria copy.tex`
- **Lineas/Tamano**: 49 lineas / 4 KB
- **Proposito**: Version alternativa de la dedicatoria. Difiere en que incluye una seccion "A mi mismo" (8 lineas adicionales sobre resiliencia) y omite la seccion del archivo principal. Las secciones a padres, familia y amigos son identicas.
- **Incluido en main.tex**: **No** -- main.tex incluye `dedicatoria/dedicatoria` (sin "copy")
- **Importancia**: ELIMINABLE
- **Justificacion**: Es una copia de respaldo/alternativa. El nombre "dedicatoria copy.tex" (con espacio) indica que es un duplicado de trabajo. No se referencia en ningun lugar.

### 6. resumen/resumen.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/resumen/resumen.tex`
- **Lineas/Tamano**: 18 lineas / 4 KB
- **Proposito**: Resumen en espanol de la tesis. Describe el sistema de normalizacion geometrica con landmarks, los cuatro componentes (CLAHE/SAHS, ensemble de 4 ResNet-18, GPA + Delaunay + warping, clasificador CNN) y los resultados principales (3.61 px error landmarks, 98.60% accuracy CV, analisis de caracteristicas espurias). Incluye palabras clave.
- **Incluido en main.tex**: Si
- **Importancia**: CRITICO
- **Justificacion**: Seccion obligatoria de la tesis. Contiene la sintesis completa del trabajo.

### 7. resumen/abstract.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/resumen/abstract.tex`
- **Lineas/Tamano**: 18 lineas / 4 KB
- **Proposito**: Abstract en ingles. Traduccion fiel del resumen en espanol con las mismas cifras y estructura.
- **Incluido en main.tex**: Si
- **Importancia**: CRITICO
- **Justificacion**: Requerimiento estandar para tesis de posgrado.

### 8. agradecimientos/agradecimientos.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/agradecimientos/agradecimientos.tex`
- **Lineas/Tamano**: 17 lineas / 4 KB
- **Proposito**: Agradecimientos a SECIHTI (beca), BUAP/FCE, directores de tesis, comite revisor (M.C. Quiroz, M.C. Rodriguez, Dra. Morin Castillo), Dr. Altamirano Robles (estancia INAOE) y familia.
- **Incluido en main.tex**: Si
- **Importancia**: CRITICO
- **Justificacion**: Seccion obligatoria solicitada explicitamente por el jurado (M.C. Ana Maria).

### 9. objetivos/0-Objetivos.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/objetivos/0-Objetivos.tex`
- **Lineas/Tamano**: 16 lineas / 4 KB
- **Proposito**: Objetivo general y 5 objetivos especificos del trabajo. Incluye: (1) alineacion/normalizacion deformable, (2) extraccion/seleccion de caracteristicas, (3) evaluacion KNN y CNN, (4) validacion cruzada y metricas, (5) comparacion con/sin normalizacion.
- **Incluido en main.tex**: Si (despues de capitulo 1, como seccion no numerada dentro del flujo)
- **Importancia**: CRITICO
- **Justificacion**: Los objetivos son evaluados directamente por el jurado. El jurado solicito revision y actualizacion de estos.

### 10. acronimos/lista_acronimos.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/acronimos/lista_acronimos.tex`
- **Lineas/Tamano**: 50 lineas / 4 KB
- **Proposito**: Lista de acronimos y abreviaturas organizada por categorias (Aprendizaje Profundo, Procesamiento de Imagenes, Analisis Geometrico, Machine Learning, Medico, Metricas, Otros). Usa entorno multicols.
- **Incluido en main.tex**: **No** -- no hay `\input{acronimos/...}` en main.tex
- **Importancia**: BAJO
- **Justificacion**: Contenido duplicado con la seccion de acronimos ya incluida al final de `glosario.tex` (lineas 717-859). Podria incluirse si se desea una lista separada previa a los capitulos, pero actualmente no se usa.

### 11. glosario.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/glosario.tex`
- **Lineas/Tamano**: 859 lineas / 56 KB
- **Proposito**: Glosario academico exhaustivo (A-Z) con definiciones de todos los terminos tecnicos de la tesis, seguido de una tabla formal de acronimos y abreviaturas (lineas 717-859). Contiene ~100 terminos con definiciones detalladas y parametros especificos del trabajo.
- **Incluido en main.tex**: Si (al final del documento, despues de anexos)
- **Importancia**: ALTO
- **Justificacion**: Archivo extenso y bien elaborado que sirve como referencia tecnica del documento. La tabla de acronimos (longtable) es formal y completa.

### 12. capitulo1/1_introduccion.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/capitulo1/1_introduccion.tex`
- **Lineas/Tamano**: 26 lineas / 12 KB
- **Proposito**: Capitulo 1 - Introduccion. Establece el contexto (neumonia, COVID-19, necesidad de diagnostico automatico), plantea el problema (variabilidad geometrica, caracteristicas espurias), revisa trabajos previos (COVIDNet, CheXNet, STERN, Picazo-Castillo, Ayala-Raggi), propone la solucion (deformacion afin por partes con 15 landmarks) y describe la organizacion del documento.
- **Incluido en main.tex**: Si
- **Importancia**: CRITICO
- **Justificacion**: Capitulo introductorio que enmarca toda la tesis.

### 13. capitulo2/2_marco_teorico.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/capitulo2/2_marco_teorico.tex`
- **Lineas/Tamano**: 653 lineas / 44 KB
- **Proposito**: Capitulo 2 - Marco Teorico. Cubre: representacion mediante landmarks (15 puntos), CLAHE, SAHS, redes neuronales convolucionales (convolucion, pooling, ResNet, skip connections), mecanismos de atencion (Coordinate Attention), funciones de perdida (Wing Loss), GPA (Analisis Procrustes), triangulacion de Delaunay, transformacion afin por partes, y metricas de evaluacion (error en pixeles, accuracy, precision, recall, F1, validacion cruzada).
- **Incluido en main.tex**: Si
- **Importancia**: CRITICO
- **Justificacion**: Capitulo fundamental que establece los fundamentos matematicos y computacionales usados en la tesis.

### 14. capitulo3/3_estado_del_arte.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/capitulo3/3_estado_del_arte.tex`
- **Lineas/Tamano**: 298 lineas / 44 KB
- **Proposito**: Capitulo 3 - Estado del Arte. Revision sistematica con 8 secciones: DL para diagnostico medico (evolucion CNN, transfer learning, casos de exito, desafios), deteccion de COVID-19 (datasets, metodos, limitaciones), deteccion de landmarks (facial/vertebral, brecha en torax), normalizacion geometrica (STN, STERN, trabajos del grupo), mecanismos de atencion (SE, CBAM, Coordinate Attention, ViT), preprocesamiento de contraste, robustez/generalizacion, y sintesis. Incluye 3 tablas comparativas via `\input`.
- **Incluido en main.tex**: Si
- **Importancia**: CRITICO
- **Justificacion**: Revision de literatura y posicionamiento del trabajo.

### 15. capitulo3/tabla_3_1_covid19_detection.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/capitulo3/tabla_3_1_covid19_detection.tex`
- **Lineas/Tamano**: 38 lineas / 4 KB
- **Proposito**: Tabla comparativa de metodos de deteccion de COVID-19 en radiografias (COVIDNet, CheXNet, CovC-ReDRNet, RegNetX032, VGG19, DenseNet-121, DenseNet-201). Muestra arquitectura, dataset, exactitud y sensibilidad.
- **Incluido en main.tex**: Indirectamente (via `\input` en 3_estado_del_arte.tex)
- **Importancia**: ALTO
- **Justificacion**: Tabla comparativa clave para el posicionamiento del trabajo.

### 16. capitulo3/tabla_3_2_landmark_detection.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/capitulo3/tabla_3_2_landmark_detection.tex`
- **Lineas/Tamano**: 38 lineas / 4 KB
- **Proposito**: Tabla comparativa de metodos de deteccion de landmarks (Wing Loss facial, Adaptive Wing, Yeh columna vertebral). Identifica la brecha en deteccion de contorno pulmonar completo.
- **Incluido en main.tex**: Indirectamente (via `\input` en 3_estado_del_arte.tex)
- **Importancia**: ALTO
- **Justificacion**: Sustenta la brecha identificada en la literatura.

### 17. capitulo3/tabla_3_3_normalizacion_geometrica.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/capitulo3/tabla_3_3_normalizacion_geometrica.tex`
- **Lineas/Tamano**: 38 lineas / 4 KB
- **Proposito**: Tabla comparativa de normalizacion geometrica (STN, STERN, Picazo-Castillo, Ayala-Raggi). Identifica la brecha en deformacion afin por partes para clasificacion medica.
- **Incluido en main.tex**: Indirectamente (via `\input` en 3_estado_del_arte.tex)
- **Importancia**: ALTO
- **Justificacion**: Sustenta la novedad del enfoque propuesto.

### 18. capitulo4/4_1_descripcion_general.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/capitulo4/4_1_descripcion_general.tex`
- **Lineas/Tamano**: 110 lineas / 12 KB
- **Proposito**: Seccion 4.1 - Descripcion General del Sistema. Define `\chapter{Metodologia}` y describe las 3 fases del sistema (entrenamiento landmarks, entrenamiento clasificador, prueba). Incluye 4 figuras de diagrama: fases del sistema, pipeline de operacion, flujo de datos, y diseno modular.
- **Incluido en main.tex**: Si
- **Importancia**: CRITICO
- **Justificacion**: Define la arquitectura general del sistema y el capitulo de metodologia.

### 19. capitulo4/4_2_dataset_preprocesamiento.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/capitulo4/4_2_dataset_preprocesamiento.tex`
- **Lineas/Tamano**: 227 lineas / 16 KB
- **Proposito**: Seccion 4.2 - Dataset y Preprocesamiento. Describe COVID-19 Radiography Database (15,153 imagenes, 3 clases), proceso de anotacion manual (957 imagenes, 15 landmarks), preprocesamiento CLAHE (tile=4, clip=2.0), division del dataset (75/15/10 con seed=42, estratificado).
- **Incluido en main.tex**: Si
- **Importancia**: CRITICO
- **Justificacion**: Describe los datos y su preparacion, fundamentales para reproducibilidad.

### 20. capitulo4/4_3_modelo_landmarks.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/capitulo4/4_3_modelo_landmarks.tex`
- **Lineas/Tamano**: 451 lineas / 32 KB
- **Proposito**: Seccion 4.3 - Modelo de Prediccion de Landmarks. Archivo mas extenso del Cap 4. Describe: arquitectura ResNet-18 con Coordinate Attention, cabeza de regresion (GAP + GN + FC + Dropout + Sigmoid), Wing Loss, Combined Loss (simetria, preservacion de distancia), entrenamiento en dos fases (frozen 15 epochs, fine-tune 100 epochs), ensemble de 4 modelos, TTA con correccion de simetria.
- **Incluido en main.tex**: Si
- **Importancia**: CRITICO
- **Justificacion**: Describe el componente central del sistema: la deteccion de landmarks.

### 21. capitulo4/4_4_normalizacion_geometrica.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/capitulo4/4_4_normalizacion_geometrica.tex`
- **Lineas/Tamano**: 246 lineas / 20 KB
- **Proposito**: Seccion 4.4 - Normalizacion Geometrica. Detalla GPA iterativo (centrado, escalado, rotacion optima via SVD), triangulacion de Delaunay (16 triangulos), transformacion afin por partes (mapeo triangulo por triangulo), margin scale (1.05), y flujo completo del modulo.
- **Incluido en main.tex**: Si
- **Importancia**: CRITICO
- **Justificacion**: La normalizacion geometrica es la contribucion principal de la tesis.

### 22. capitulo4/4_5_clasificacion.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/capitulo4/4_5_clasificacion.tex`
- **Lineas/Tamano**: 294 lineas / 20 KB
- **Proposito**: Seccion 4.5 - Clasificacion de Enfermedades Pulmonares. Describe preprocesamiento SAHS para imagenes warped, arquitectura del clasificador (ResNet-18 transfer learning, 3 clases), estrategia de entrenamiento (50 epochs, batch=32, AdamW), aumento de datos (flip, rotacion, desplazamiento), pesos por clase, y configuracion completa.
- **Incluido en main.tex**: Si
- **Importancia**: CRITICO
- **Justificacion**: Describe el modulo final del pipeline.

### 23. capitulo4/4_6_inferencia_metricas.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/capitulo4/4_6_inferencia_metricas.tex`
- **Lineas/Tamano**: 82 lineas / 8 KB
- **Proposito**: Seccion 4.6 - Protocolo de Inferencia y Evaluacion. Describe flujo de inferencia end-to-end, metricas (error en pixeles, accuracy, precision, recall, F1 Macro/Weighted, matriz de confusion), protocolo de evaluacion y especificaciones de hardware.
- **Incluido en main.tex**: Si
- **Importancia**: ALTO
- **Justificacion**: Define como se evalua el sistema.

### 24. capitulo5/5_1_resultados_landmarks.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/capitulo5/5_1_resultados_landmarks.tex`
- **Lineas/Tamano**: 64 lineas / 8 KB
- **Proposito**: Seccion 5.1 - Resultados de Deteccion de Landmarks. Define `\chapter{Resultados}`. Presenta: ensemble 3.61 px vs individual 4.04 px (10.6% mejora), error por landmark (centrales 2.44-2.94 px, esquinas superiores 5.35-5.43 px), figuras de distribucion de error y ejemplos de prediccion.
- **Incluido en main.tex**: Si
- **Importancia**: CRITICO
- **Justificacion**: Resultados cuantitativos del modelo de landmarks.

### 25. capitulo5/5_2_forma_canonica.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/capitulo5/5_2_forma_canonica.tex`
- **Lineas/Tamano**: 64 lineas / 8 KB
- **Proposito**: Seccion 5.2 - Normalizacion Geometrica. Presenta forma canonica obtenida por GPA, triangulacion de Delaunay (16 triangulos), y ejemplos visuales de imagenes normalizadas + SAHS para las tres categorias.
- **Incluido en main.tex**: Si
- **Importancia**: CRITICO
- **Justificacion**: Resultados visuales de la normalizacion geometrica.

### 26. capitulo5/5_3_resultados_clasificacion.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/capitulo5/5_3_resultados_clasificacion.tex`
- **Lineas/Tamano**: 193 lineas / 16 KB
- **Proposito**: Seccion 5.3 - Clasificacion (version single-split). Reporta: 98.60% accuracy, 98.00% F1-Macro, rendimiento por clase, matriz de confusion, analisis de errores (36 imagenes), y comparacion de 4 configuraciones de preprocesamiento (Original 98.68%, Normalizado 98.60%, ALP 96.40%, Recortado 95.36%) que demuestra la hipotesis sobre caracteristicas espurias.
- **Incluido en main.tex**: **Si** -- es la version activa (linea 197: `\input{capitulo5/5_3_resultados_clasificacion}`)
- **Importancia**: CRITICO
- **Justificacion**: Contiene la evidencia experimental principal de la tesis.

### 27. capitulo5/5_3_resultados_clasificacion_CV.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/capitulo5/5_3_resultados_clasificacion_CV.tex`
- **Lineas/Tamano**: 269 lineas / 24 KB
- **Proposito**: Version alternativa de la Seccion 5.3 con resultados de validacion cruzada (k=5). Reporta: 97.68% +/- 0.16% accuracy, 96.47% +/- 0.27% F1-Macro. Incluye desglose por fold, resultados por clase (COVID 95.67%, Normal 98.88%, VP 93.30%), comparacion 4 configuraciones, y validacion con clasificador PCA+Fisher+KNN.
- **Incluido en main.tex**: **No** -- la version sin _CV es la incluida
- **Importancia**: MEDIO
- **Justificacion**: Version alternativa mas robusta estadisticamente (con cross-validation). Podria sustituir o complementar la version actual. El resumen/abstract de la tesis ya reporta cifras de CV (98.60% y 98.00%), asi que hay inconsistencia con esta version que reporta 97.68%.

### 28. capitulo5/5_4_eficiencia_computacional.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/capitulo5/5_4_eficiencia_computacional.tex`
- **Lineas/Tamano**: 36 lineas / 4 KB
- **Proposito**: Seccion 5.4 - Eficiencia Computacional. Tabla de tiempos de inferencia: 89.92 ms/imagen total (11.1 img/s). Desglose: CLAHE 0.45ms, Landmarks 66.08ms (73.5%), Warping 4.86ms, SAHS 0.53ms, Clasificacion 8.59ms.
- **Incluido en main.tex**: Si
- **Importancia**: ALTO
- **Justificacion**: Responde a la pregunta del jurado sobre costo computacional.

### 29. capitulo6/6_conclusiones.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/capitulo6/6_conclusiones.tex`
- **Lineas/Tamano**: 391 lineas / 28 KB
- **Proposito**: Capitulo 6 - Conclusiones y Trabajos Futuros. Sintetiza contribuciones (validacion experimental de hipotesis sobre caracteristicas espurias, modelo de landmarks 3.61 px, sistema completo de normalizacion, metodo SAHS), verifica cumplimiento de objetivos, discute limitaciones (dataset unico, resolución 224x224, solo PA, dependencia del modelo de landmarks), y propone trabajo futuro (mas datasets, segmentacion, ViT, resolucion mayor, explicabilidad).
- **Incluido en main.tex**: Si
- **Importancia**: CRITICO
- **Justificacion**: Capitulo final que sintetiza el trabajo y responde preguntas del jurado.

### 30. anexos/anexo_A_manual_usuario.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/anexos/anexo_A_manual_usuario.tex`
- **Lineas/Tamano**: 270 lineas / 12 KB
- **Proposito**: Manual de usuario de la interfaz grafica Gradio para demostracion del sistema. Incluye requisitos, modelos necesarios, instrucciones de uso y documentacion de la GUI.
- **Incluido en main.tex**: **No** -- esta COMENTADO (linea 223: `%\input{anexos/anexo_A_manual_usuario}`)
- **Importancia**: BAJO
- **Justificacion**: Comentado intencionalmente para revision posterior. El jurado pregunto sobre la interfaz, pero el comentario indica que aun requiere revision antes de inclusion definitiva.

### 31. anexos/anexo_B_articulos_publicados.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/anexos/anexo_B_articulos_publicados.tex`
- **Lineas/Tamano**: 60 lineas / 8 KB
- **Proposito**: Anexo B - Articulos publicados. Incluye dos articulos sobre SAHS: (1) Computacion y Sistemas Vol. 29 No. 4, 2025 (ingles), (2) Abstraction & Application Vol. 48, 2024 (espanol). Ambos con metadata completa y PDFs embebidos via `\includepdf`.
- **Incluido en main.tex**: Si
- **Importancia**: ALTO
- **Justificacion**: Evidencia de produccion cientifica, requisito comun en tesis de maestria.

### 32. anexos/anexo_C_certificados.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/anexos/anexo_C_certificados.tex`
- **Lineas/Tamano**: 115 lineas / 8 KB
- **Proposito**: Anexo C - Certificados y reconocimientos. Incluye: reconocimiento Encuentro NOVA, IEEE DAY, constancia RVP-AI ROC&C 2025, ponente taller Vision por Computadora (RAS IEEE BUAP), poster congreso IEEE NOVA, y constancia curso educacion continua "Reconocimiento de Patrones en Imagenes". Cada uno con metadata descriptiva y PDF embebido.
- **Incluido en main.tex**: Si
- **Importancia**: MEDIO
- **Justificacion**: Documentacion de difusion academica del trabajo.

### 33. anexos/anexo_D_codigo_fuente.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/anexos/anexo_D_codigo_fuente.tex`
- **Lineas/Tamano**: 68 lineas / 4 KB
- **Proposito**: Anexo D - Codigo Fuente. Referencia al repositorio GitHub (https://github.com/Rhaink/LungAlignment), descripcion del contenido (4 modulos), instrucciones de instalacion y reproducibilidad.
- **Incluido en main.tex**: Si
- **Importancia**: ALTO
- **Justificacion**: Permite reproducibilidad de los experimentos. El jurado (M.C. Ana Maria) solicito no incluir codigo inline sino referencia web.

### 34. references.bib
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/references.bib`
- **Lineas/Tamano**: 735 lineas / 28 KB
- **Proposito**: Base de datos bibliografica en formato BibTeX. Contiene ~68 referencias organizadas por categoria: redes neuronales y deep learning, COVID-19, landmarks, normalizacion geometrica, mecanismos de atencion, preprocesamiento, datasets, metricas. Incluye articulos seminales (LeCun 1998, He 2016, Krizhevsky 2012) y trabajos recientes (2023-2025).
- **Incluido en main.tex**: Si (via `\bibliography{references}`)
- **Importancia**: CRITICO
- **Justificacion**: Todas las citas del documento dependen de este archivo.

---

## Archivos de apoyo (no LaTeX)

### 35. capitulo3/PAPERS_IDENTIFICADOS.md
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/capitulo3/PAPERS_IDENTIFICADOS.md`
- **Lineas/Tamano**: 537 lineas
- **Proposito**: Documento de trabajo con papers identificados para el Cap 3, organizados por prioridad con DOI, citas y uso previsto en la tesis. Incluye surveys clave (Litjens 2017), papers de COVID-19, landmarks y normalizacion.
- **Incluido en main.tex**: No (archivo de trabajo markdown)
- **Importancia**: BAJO
- **Justificacion**: Documento de trabajo/investigacion del autor. No se incluye en la tesis compilada. Util como referencia del proceso de revision bibliografica.

### 36. capitulo3/RESUMEN_CAPITULO3.md
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/capitulo3/RESUMEN_CAPITULO3.md`
- **Lineas/Tamano**: 310 lineas
- **Proposito**: Resumen del estado de completitud del capitulo 3 con estadisticas (8 secciones, ~68 referencias), marcado como COMPLETADO.
- **Incluido en main.tex**: No (archivo de trabajo markdown)
- **Importancia**: BAJO
- **Justificacion**: Control de progreso. Eliminable tras finalizacion de la tesis.

### 37. anexos/plan_trabajo_inaoe.md
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/anexos/plan_trabajo_inaoe.md`
- **Lineas/Tamano**: 96 lineas
- **Proposito**: Plan de trabajo para la estancia de investigacion en INAOE (Oct-Nov 2025) con el Dr. Leopoldo Altamirano Robles. Define objetivos, metodologia y cronograma de la estancia.
- **Incluido en main.tex**: No (documento administrativo markdown)
- **Importancia**: BAJO
- **Justificacion**: Documento administrativo de la estancia. No se incluye en la tesis.

### 38. instrucciones_observaciones.txt
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/instrucciones_observaciones.txt`
- **Lineas/Tamano**: 178 lineas
- **Proposito**: Notas del coloquio del 23 y 29 de enero 2026 con observaciones del jurado (Dra. Morin, M.C. Ana Maria, M.C. Nicolas). Contiene feedback especifico: preguntas sobre aportacion, pesos, costo computacional, balanceo de datos, manual de usuario, objetivos, estado del arte.
- **Incluido en main.tex**: No (notas de revision)
- **Importancia**: ALTO (como referencia de revisiones pendientes)
- **Justificacion**: Documento clave para saber que correcciones solicito el jurado. No se incluye en la tesis compilada pero es fundamental para la revision.

---

## Artefactos de compilacion LaTeX (GRUPO)

### 39. Artefactos de compilacion LaTeX
- **Archivos**:
  - `main.aux` (96 KB) - Referencias cruzadas auxiliares
  - `main.bbl` (16 KB) - Bibliografia compilada por BibTeX
  - `main.blg` (4 KB) - Log de BibTeX
  - `main.fdb_latexmk` (40 KB) - Base de datos de latexmk
  - `main.fls` (56 KB) - Lista de archivos de latexmk
  - `main.lof` (20 KB) - Lista de figuras
  - `main.log` (124 KB) - Log de compilacion LaTeX
  - `main.lot` (8 KB) - Lista de tablas
  - `main.out` (36 KB) - Outlines para hyperref
  - `main.synctex.gz` (1000 KB) - Sincronizacion editor-PDF
  - `main.toc` (24 KB) - Tabla de contenido
- **Tamano total**: ~1.4 MB
- **Importancia**: ELIMINABLE
- **Justificacion**: Son artefactos regenerables automaticamente al compilar `main.tex`. Se recomienda agregarlos a `.gitignore` y eliminarlos del repositorio. El archivo `main.synctex.gz` (1 MB) es particularmente innecesario.

### 39b. PDF compilado de la tesis
- **Archivo**: `Tesis_Rafael_Cruz_223470443.pdf` (19 MB)
- **Importancia**: MEDIO
- **Justificacion**: PDF compilado de la tesis. Util como referencia pero regenerable. Podria mantenerse para acceso rapido sin necesidad de compilar LaTeX.

---

## Figuras (GRUPO)

**Directorio**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/Tesis/Figures/`
**Tamano total**: ~19 MB
**Cantidad**: ~62 archivos de imagen (PNG + JPG)

### Figuras referenciadas en archivos .tex activos

| Figura | Usado en | Descripcion |
|--------|----------|-------------|
| `Logo_de_la_BUAP.png` | title-page.tex | Logo institucional BUAP |
| `F4.1_fases_sistema.png` | 4_1 | Diagrama de 3 fases del sistema |
| `F4.2_pipeline_operacion_es.png` | 4_1 | Pipeline de operacion (espanol) |
| `F4.3_landmarks_15.png` | 2_marco, 4_2 | 15 landmarks anatomicos |
| `F4.4_clahe_comparacion.png` | 2_marco, 4_2 | Comparacion CLAHE tile 4 vs 8 |
| `F4.5_arquitectura_modelo_es.png` | 4_3 | Arquitectura ResNet-18 + CA (espanol) |
| `F4.6_wing_loss_grafica.png` | 4_3 | Grafica Wing Loss vs L1/L2 |
| `F4.7_proceso_gpa.png` | 2_marco | Proceso GPA (version original) |
| `F4.7_proceso_gpa_es.png` | 4_4 | Proceso GPA (version espanol) |
| `F4.8_triangulacion_delaunay.png` | 2_marco | Triangulacion Delaunay (original) |
| `F4.8_triangulacion_delaunay_es.png` | 4_4 | Triangulacion Delaunay (espanol) |
| `F4.9_original_vs_warped_es.png` | 2_marco, 4_4 | Original vs warped (espanol) |
| `F4.10_margin_scale.png` | -- | **NO REFERENCIADA** |
| `F4.11_flujo_normalizacion.png` | 4_4 | Flujo normalizacion |
| `F4.12_aumento_datos.png` | 4_5 | Aumento de datos |
| `F4.13_warped_sahs.png` | 4_5 | Warped + SAHS |
| `F4.14_flujo_clasificacion.png` | 4_5 | Flujo clasificacion |
| `F4.15_flujo_inferencia_evaluacion.png` | 4_6 | Flujo inferencia y evaluacion |
| `F4.16_estrategia_entrenamiento_ensemble_tta.png` | 4_3 | Estrategia ensemble + TTA |
| `F4.17_division_dataset.png` | 4_2 | Division del dataset |
| `F4.18_transfer_learning_clasificador.png` | 4_5 | Transfer learning clasificador |
| `F4.19_rotacion_optima_svd.png` | 4_4 | Rotacion optima SVD |
| `F4.20_diseno_modular_justificacion.png` | 4_1 | Diseno modular |
| `F4.21_proceso_anotacion.png` | 4_2 | Proceso de anotacion |
| `F4.22_transformacion_triangulo.png` | 4_4 | Transformacion triangulo |
| `F4.23_flujo_datos.png` | 4_1 | Flujo de datos |
| `F4.24_cabeza_regresion.png` | 4_3 | Cabeza de regresion |
| `F2.3_clahe_vs_sahs.png` | 2_marco | CLAHE vs SAHS |
| `coord_attention_v10_mechanism_real.png` | 4_3 | Mecanismo Coordinate Attention |
| `F5.1_error_por_landmark.png` | 5_1 | Error por landmark |
| `F5.2_ejemplos_prediccion.png` | 5_1 | Ejemplos de prediccion |
| `F5.3_forma_canonica.png` | 5_2 | Forma canonica GPA |
| `F5.4_triangulacion_resultados.png` | 5_2 | Triangulacion resultados |
| `F5.6_ejemplos_warping.png` | 5_2 | Ejemplos warping |
| `F5.7_matriz_confusion_sahs_es.png` | 5_3 | Matriz confusion (espanol, activa) |
| `F5.8_comparacion_sahs_v2.png` | 5_3 | Comparacion SAHS v2 (activa) |
| `F5.9_casos_mal_clasificados.png` | 5_3 | Casos mal clasificados (activa) |
| `F5.11_comparacion_preprocesamiento_sahs.png` | 5_3, 5_3_CV | Comparacion preprocesamiento |

### Figuras NO referenciadas en archivos .tex activos

Estas figuras existen en `Figures/` pero no se referencian en ninguno de los `.tex` incluidos en `main.tex`:

| Figura | Razon probable |
|--------|----------------|
| `F4.2b_interfaz_etiquetadocopia.png` | **NO REFERENCIADA** -- version duplicada con "copia" |
| `F4.10_margin_scale.png` | **NO REFERENCIADA** -- posiblemente eliminada del texto |
| `F4.13_warped_sahs_simple.png` | **NO REFERENCIADA** -- version simplificada no usada |
| `F4.2_pipeline_operacion.jpg` | Reemplazada por `_es.png` |
| `F4.1_fases_sistema.jpg` | Reemplazada por `.png` |
| `F4.5_arquitectura_modelo.png` | Reemplazada por `_es.png` |
| `F4.9_original_vs_warped.png` | Reemplazada por `_es.png` |
| `F4.9_original_vs_warped_en.png` | Version en ingles no usada |
| `F5.1_error_por_landmark_en.png` | Version en ingles no usada |
| `F5.7_matriz_confusion_sahs.png` | Reemplazada por `_es.png` |
| `F5.7_matriz_confusion_cv.png` | Solo en `_CV.tex` (no incluido) |
| `F5.8_comparacion_sahs.png` | Reemplazada por `_v2.png` |
| `F5.8_comparacion_cv.png` | Solo en `_CV.tex` (no incluido) |
| `F5.9_casos_mal_clasificados_cv.png` | Solo en `_CV.tex` (no incluido) |
| `F5.9_misclassified_cases.png` | Version en ingles no usada |
| `F4.2b_interfaz_etiquetado.png` | Referenciada en 4_2 |
| `setup/img/Logo_de_la_BUAP.png` | Duplicado del logo en Figures/ |

**Total figuras no referenciadas**: ~15 archivos (versiones en ingles, duplicados, versiones antiguas)

### PDFs embebidos en anexos

| Archivo PDF | Usado en | Tamano |
|-------------|----------|--------|
| `articulo_ingles_sahs_2024.pdf` | anexo_B | 588 KB |
| `articulo_espanol_sahs_2024.pdf` | anexo_B | 468 KB |
| `reconocimiento_nova.pdf` | anexo_C | 152 KB |
| `reconocimiento_ieee_day.pdf` | anexo_C | 212 KB |
| `constancia_participacion_rafael.pdf` | anexo_C | 764 KB |
| `reconocimiento_ras_day.pdf` | anexo_C | 160 KB |
| `POSTER-Normalizacion...pdf` | anexo_C | 844 KB |
| `Rafael Alejandro CO.pdf` | anexo_C | 592 KB |
| `inaoe.pdf` | **NO REFERENCIADA** | 152 KB |
| `MX2025012404.pdf` | **NO REFERENCIADA** | 848 KB |

---

## Resumen de importancia

| Importancia | Cantidad | Archivos |
|-------------|----------|----------|
| **CRITICO** | 18 | main.tex, title-page, resumen, abstract, agradecimientos, objetivos, Cap 1-6 (12 archivos), references.bib |
| **ALTO** | 8 | dedicatoria, glosario, tablas 3.1-3.3, 4_6_inferencia, 5_4_eficiencia, anexo_B, anexo_D, instrucciones_observaciones |
| **MEDIO** | 3 | 5_3_clasificacion_CV, anexo_C, PDF compilado |
| **BAJO** | 4 | acronimos, PAPERS_IDENTIFICADOS.md, RESUMEN_CAPITULO3.md, plan_trabajo_inaoe.md |
| **ELIMINABLE** | 4+ | settings.tex, dedicatoria copy.tex, artefactos LaTeX (11 archivos), ~15 figuras no referenciadas, 2 PDFs no referenciados |

---

## Observaciones criticas

### 1. Inconsistencia en resultados entre versiones de Sec 5.3
La version activa (`5_3_resultados_clasificacion.tex`) reporta 98.60% accuracy y 98.00% F1-Macro (single split). La version alternativa (`_CV.tex`) con validacion cruzada reporta 97.68% +/- 0.16%. El resumen/abstract de la tesis usa las cifras de la version activa (98.60%), pero menciona "validacion cruzada de cinco pliegues". Esto es inconsistente: o se usan las cifras de CV o se reporta como single-split.

### 2. Archivo settings.tex no utilizado
El archivo `setup/settings.tex` usa `biblatex` mientras `main.tex` usa `natbib`. Son incompatibles. El archivo no se incluye y deberia eliminarse para evitar confusion.

### 3. Figuras con versiones duplicadas (EN/ES)
Existen multiples figuras con versiones en ingles y espanol (e.g., `F4.7_proceso_gpa.png` vs `_es.png`, `F4.5_arquitectura_modelo.png` vs `_es.png`). Las versiones en espanol son las activas. Las versiones en ingles/originales podrian eliminarse para reducir tamano (~15 archivos sobrantes).

### 4. Anexo A comentado
El manual de usuario (Anexo A, 270 lineas) esta comentado en `main.tex`. El jurado pregunto sobre la interfaz de usuario, por lo que podria necesitar activarse tras revision.

### 5. PDFs no referenciados
`inaoe.pdf` y `MX2025012404.pdf` existen en `anexos/` pero no se referencian en ningun `.tex`. Podrian ser documentos administrativos que deberian eliminarse del directorio de la tesis o incluirse si son relevantes.

### 6. Artefactos de compilacion en el repositorio
Los 11 archivos de artefactos LaTeX (~1.4 MB) y `main.synctex.gz` (1 MB) no deberian estar en el repositorio Git. Se recomienda agregarlos a `.gitignore`.

### 7. Nombre de archivo con espacio
`dedicatoria copy.tex` contiene un espacio en el nombre, lo cual puede causar problemas en ciertos entornos. Ademas es un duplicado no referenciado.
