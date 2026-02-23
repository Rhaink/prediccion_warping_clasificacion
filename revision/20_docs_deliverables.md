# 20. Deliverables Documentation

Analisis de documentacion de entregables: reporte de estancia, manual de usuario, cartas formales y snapshot USB.

**Archivos analizados**: ~110 (muchos son duplicados del snapshot USB)

---

## A. Reporte de Estancia (docs/estancia/)

### 1. AUDITORIA_REPORTE_INAOE.md
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/estancia/AUDITORIA_REPORTE_INAOE.md`
- **Lineas/Tamano**: 258 lineas / 12 KB
- **Proposito**: Auditoria sistematica que verifica el cumplimiento del reporte de estancia contra el plan de trabajo firmado y el GROUND_TRUTH.json. Incluye tablas detalladas comparando datos generales, objetivos, metodologia, metricas y entregables.
- **Importancia**: ALTO
- **Justificacion**: Documento de trazabilidad unico que valida la correspondencia entre el plan firmado, los resultados reportados y los valores de GROUND_TRUTH.json. Util para revisores academicos y para demostrar rigor en la verificacion. Todas las metricas revisadas coinciden correctamente. La nota sobre CLAUDE.md (99.10% obsoleto vs. 98.60% CV correcto en el reporte) es un detalle valioso de auditoria.

### 2. LISTA_ENTREGABLES.md
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/estancia/LISTA_ENTREGABLES.md`
- **Lineas/Tamano**: 249 lineas / 12 KB
- **Proposito**: Inventario completo de entregables de la estancia INAOE, con tamanos, rutas y comandos para preparar el paquete de entrega. Incluye dos opciones de entrega (completa y ligera).
- **Importancia**: MEDIO
- **Justificacion**: Documento organizativo util para la preparacion de entregables. Gran parte de su contenido fue materializado en el snapshot USB (entregables_usb/), por lo que ahora es mas historico que funcional. Los comandos de preparacion de entrega son utiles como referencia pero ya fueron ejecutados.

### 3. REPORTE_ESTANCIA_INAOE.md
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/estancia/REPORTE_ESTANCIA_INAOE.md`
- **Lineas/Tamano**: 722 lineas / 24 KB
- **Proposito**: Version Markdown del reporte de estancia INAOE. Documento completo con metodologia, resultados, entregables, cronograma y conclusiones. Incluye pseudocodigo Python extenso para cada componente del pipeline.
- **Importancia**: MEDIO
- **Justificacion**: Borrador en Markdown del reporte. Contiene el mismo contenido que las versiones LaTeX pero en formato menos formal. Dado que existen dos versiones LaTeX (V1 y V2) que son las versiones finales, este archivo es un precursor/borrador. **Problema detectado**: El MD dice 21,165 imagenes mientras las versiones LaTeX dicen 15,153. El numero correcto es 15,153 (suma de COVID:3616 + Normal:10192 + Viral_Pneumonia:1345). La cifra 21,165 incluye la clase "Lung_Opacity" que no se usa en este proyecto. La auditoria no detecto esta discrepancia porque referenciaba numeros de linea del .tex (correcto), no del .md.

### 4. REPORTE_ESTANCIA_INAOE.tex (V1)
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/estancia/REPORTE_ESTANCIA_INAOE.tex`
- **Lineas/Tamano**: 946 lineas / 32 KB
- **Proposito**: Version LaTeX V1 del reporte formal de estancia INAOE. Documento completo con portada institucional, indice de contenidos, 7 secciones principales (Resumen Ejecutivo, Introduccion, Metodologia, Resultados, Entregables, Cronograma, Conclusiones). Incluye codigo Python en listings, tablas de resultados, y espacio para firmas.
- **Importancia**: ALTO
- **Justificacion**: Es el reporte formal academico de la estancia. Incluye portada institucional completa, indice de contenidos (ToC), codigo embebido extenso con listings de Python, y firmas formales. Los datos del dataset (15,153 imagenes) son correctos. Incluye mencion de SAHS para clasificacion, que no aparece en V2. Sin embargo, la version V1 es mas larga y verbosa que V2 -- incluye mucho codigo repetido en listings. La fecha es 28 de enero de 2026.

### 5. REPORTE_ESTANCIA_INAOE_V2.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/estancia/REPORTE_ESTANCIA_INAOE_V2.tex`
- **Lineas/Tamano**: 369 lineas / 20 KB
- **Proposito**: Version LaTeX V2 (compacta) del reporte de estancia. Version mas concisa que V1, sin ToC y sin mucho codigo embebido. Mismos resultados pero presentacion mas densa y profesional.
- **Importancia**: CRITICO
- **Justificacion**: Version final y mas refinada del reporte. Eliminada la tabla de contenidos (documento mas corto no la necesita). Texto mas denso y profesional -- los subsistemas se describen en prosa en vez de con listings extensos. Agrega tabla completa de error por landmark individual (todos los 15 landmarks vs. solo 6 en V1). Agrega seccion "Impacto de la Normalizacion Geometrica". No tiene seccion de Conclusiones/Firmas al final (termina en Cronograma), lo cual es una omision que deberia corregirse -- la V1 tiene firmas formales.

### Artefactos LaTeX de estancia (grupo)
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/estancia/REPORTE_ESTANCIA_INAOE.{aux,fdb_latexmk,fls,log,out,synctex.gz,toc}` y `REPORTE_ESTANCIA_INAOE_V2.{aux,fdb_latexmk,fls,log,out,synctex.gz}`
- **Lineas/Tamano**: ~13 archivos / ~444 KB total (el synctex.gz de V1 solo es 260 KB)
- **Proposito**: Archivos intermedios generados por pdflatex durante la compilacion de los reportes.
- **Importancia**: ELIMINABLE
- **Justificacion**: Artefactos de compilacion LaTeX. Se regeneran automaticamente al compilar con `pdflatex`. No aportan valor al repositorio y solo anaden ruido. Deben eliminarse y anadirse al .gitignore.

### PDFs de estancia (grupo)
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/estancia/REPORTE_ESTANCIA_INAOE.pdf`, `REPORTE_ESTANCIA_INAOE_V2.pdf`, `plan_trabajo.pdf`
- **Lineas/Tamano**: 3 archivos PDF
- **Proposito**: PDFs compilados de los reportes y el plan de trabajo original firmado.
- **Importancia**: ALTO (plan_trabajo.pdf es CRITICO como referencia original firmada)
- **Justificacion**: Los PDFs compilados son la version "final" entregable. El plan_trabajo.pdf es el documento de referencia original firmado que define los objetivos de la estancia. Los PDFs de reporte se pueden regenerar desde los .tex, pero es conveniente tenerlos pre-compilados. El plan de trabajo es un documento externo no regenerable.

---

## B. Cartas Formales (docs/carta/)

### 6. carta_modificaciones_tesis.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/carta/carta_modificaciones_tesis.tex`
- **Lineas/Tamano**: 111 lineas / 8 KB
- **Proposito**: Carta formal que describe las modificaciones realizadas a la tesis en respuesta a observaciones del jurado de coloquio. Documenta 8 categorias de cambios: estructura, objetivos, validacion experimental, eficiencia computacional, contribuciones, estado del arte, figuras/tablas, redaccion, y anexos.
- **Importancia**: CRITICO
- **Justificacion**: Documento administrativo obligatorio para el proceso de titulacion. Debe acompanar la version corregida de la tesis. Documenta cambios sustanciales: eliminacion de publicaciones como objetivo, adicion de validacion cruzada 5-fold, comparacion con clasificador PCA+KNN, incorporacion del metodo ALP, tiempos de inferencia. Este documento es evidencia formal de la respuesta a observaciones del jurado.

### 7. CONSTANCIA_ESTANCIA_INAOE.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/carta/CONSTANCIA_ESTANCIA_INAOE.tex`
- **Lineas/Tamano**: 87 lineas / 4 KB
- **Proposito**: Constancia formal firmada por el Dr. Leopoldo Altamirano Robles certificando que el estudiante realizo la estancia de investigacion en el INAOE. Incluye titulo del proyecto, periodo, y afirmacion de cumplimiento.
- **Importancia**: CRITICO
- **Justificacion**: Documento administrativo obligatorio para el expediente academico. Es la constancia oficial del director de proyecto. **Discrepancia detectada**: El periodo de estancia dice "marzo a julio de 2025" mientras que el reporte dice "9 de octubre - 9 de noviembre de 2025". Estos son periodos diferentes, lo cual podria ser un error o podria referirse a dos estancias distintas. Debe verificarse y corregirse si es un error.

### Artefactos LaTeX de carta (grupo)
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/carta/{carta_modificaciones_tesis,CONSTANCIA_ESTANCIA_INAOE}.{aux,fdb_latexmk,fls,log,synctex.gz}`
- **Lineas/Tamano**: ~10 archivos / ~68 KB total
- **Proposito**: Archivos intermedios de compilacion LaTeX para las cartas.
- **Importancia**: ELIMINABLE
- **Justificacion**: Artefactos de compilacion, se regeneran al compilar. Deben eliminarse.

### PDFs de carta (grupo)
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/carta/modificaciones_tesis.pdf`, `CONSTANCIA_ESTANCIA_INAOE.pdf`
- **Lineas/Tamano**: 2 archivos PDF
- **Proposito**: PDFs compilados de las cartas formales.
- **Importancia**: ALTO
- **Justificacion**: Versiones finales listas para firma e impresion. Regenerables desde los .tex.

---

## C. Manual de Usuario (docs/manual/)

### 8. manual_usuario.tex
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/manual/manual_usuario.tex`
- **Lineas/Tamano**: 839 lineas / 32 KB
- **Proposito**: Manual de usuario completo en LaTeX para la interfaz grafica (GUI) v15 del sistema de deteccion COVID-19. Orientado a usuarios sin conocimientos tecnicos. Incluye 9 capitulos: Introduccion, Instalacion, Demostracion Completa, Vista Rapida, Solucion de Problemas, Preguntas Frecuentes, Glosario, Contacto, y Anexos.
- **Importancia**: ALTO
- **Justificacion**: Manual profesional bien estructurado para usuarios finales. Usa comandos LaTeX personalizados (paso, boton, nota, advertencia, consejo). Referencia 16 imagenes (la mayoria faltan -- solo 2 de 16 capturadas: interfaz_principal.png y sistema_cargado.png). Incluye disclaimer medico repetido multiples veces. Version del sistema: v15. Especificaciones tecnicas correctas (3.61 px, 98.60%). El manual compila pero tiene muchas imagenes faltantes que aparecen como espacios vacios.

### 9. compilar.sh
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/manual/compilar.sh`
- **Lineas/Tamano**: 149 lineas / 8 KB
- **Proposito**: Script bash para compilar el manual de usuario. Incluye verificaciones de pdflatex, triple compilacion para referencias, conteo de imagenes y oferta de abrir el PDF resultante.
- **Importancia**: BAJO
- **Justificacion**: Script de conveniencia para compilar LaTeX. Funcionalidad trivial (tres pdflatex consecutivos) envuelta en mucho output decorativo. Util pero no esencial -- cualquier usuario de LaTeX sabe ejecutar pdflatex.

### 10. INSTRUCCIONES_IMAGENES.md
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/manual/INSTRUCCIONES_IMAGENES.md`
- **Lineas/Tamano**: 326 lineas / 12 KB
- **Proposito**: Guia detallada para capturar las 16 imagenes requeridas por el manual de usuario. Incluye especificaciones tecnicas por imagen (contenido, ubicacion en LaTeX, sugerencias de captura, formato).
- **Importancia**: BAJO
- **Justificacion**: Documento de proceso interno para completar el manual. Solo 2 de 16 imagenes han sido capturadas, por lo que aun es relevante. Sin embargo, es documentacion de proceso, no un entregable en si. Podria consolidarse como seccion del README.md del directorio.

### 11. README.md (manual)
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/manual/README.md`
- **Lineas/Tamano**: 243 lineas / 8 KB
- **Proposito**: README del directorio del manual. Describe contenido del manual, estructura de capitulos, instrucciones de compilacion, y comandos LaTeX personalizados.
- **Importancia**: BAJO
- **Justificacion**: Documentacion del directorio. Repite informacion que ya esta en INSTRUCCIONES_IMAGENES.md y RESUMEN_GENERACION.md. Hay tres archivos de meta-documentacion (README, INSTRUCCIONES, RESUMEN) para un solo archivo de contenido (manual_usuario.tex). Excesivo.

### 12. RESUMEN_GENERACION.md
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/manual/RESUMEN_GENERACION.md`
- **Lineas/Tamano**: 377 lineas / 12 KB
- **Proposito**: Resumen detallado de la generacion del manual de usuario. Documenta estructura del contenido, estado de imagenes, estadisticas, y proximos pasos.
- **Importancia**: ELIMINABLE
- **Justificacion**: Documento generado automaticamente por Claude Code que describe lo que se acaba de crear. Es meta-documentacion redundante -- el README.md y INSTRUCCIONES_IMAGENES.md ya cubren esta informacion. El archivo es excesivamente largo (377 lineas) para un resumen de generacion. Contiene mucha repeticion de informacion del README.md.

### Artefactos LaTeX de manual (grupo)
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/manual/manual_usuario.{aux,fdb_latexmk,fls,log,out,toc}`
- **Lineas/Tamano**: 6 archivos / ~112 KB total
- **Proposito**: Archivos intermedios de compilacion LaTeX del manual.
- **Importancia**: ELIMINABLE
- **Justificacion**: Artefactos de compilacion. Se regeneran al compilar.

### manual_usuario.pdf
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/manual/manual_usuario.pdf`
- **Lineas/Tamano**: 1 archivo PDF (23 paginas segun README)
- **Proposito**: Manual de usuario compilado en PDF.
- **Importancia**: ALTO
- **Justificacion**: Entregable final del manual. Aunque tiene imagenes faltantes (14 de 16), la estructura y el texto estan completos.

### Imagenes del manual (grupo)
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/manual/imagenes/`
- **Lineas/Tamano**: 2 archivos PNG (interfaz_principal.png, sistema_cargado.png)
- **Proposito**: Capturas de pantalla de la interfaz GUI v15 para el manual.
- **Importancia**: MEDIO
- **Justificacion**: Solo 2 de 16 imagenes capturadas. Las 14 restantes estan pendientes. Sin estas imagenes, el manual compilado tiene muchos espacios vacios.

---

## D. Snapshot USB de Entregables (docs/estancia/entregables_usb/)

### 13. 01_Reporte/ (Reporte + Figuras)
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/estancia/entregables_usb/01_Reporte/`
- **Lineas/Tamano**: 1 .tex (946 lineas, identico a V1), 1 PDF, 8 figuras PNG
- **Proposito**: Copia del reporte de estancia con figuras para entrega en USB. Incluye 8 figuras (F4.3 landmarks, F4.5 arquitectura, F4.6 wing loss, F4.7 GPA, F4.8 Delaunay, F4.9 warped, F5.1 error por landmark, F5.7 matriz confusion SAHS).
- **Importancia**: MEDIO
- **Justificacion**: El .tex es duplicado exacto de la V1 en docs/estancia/. Las figuras PNG son valiosas -- son las unicas copias de estas visualizaciones en el repositorio. El PDF es la version compilada con figuras integradas. Si se elimina el snapshot USB, las figuras deben preservarse en otro lugar.

### 14. 02_Codigo/src_v2/ (Snapshot de codigo)
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/estancia/entregables_usb/02_Codigo/src_v2/`
- **Lineas/Tamano**: ~40 archivos .py + 43 archivos .pyc (en __pycache__/)
- **Proposito**: Copia congelada del directorio src_v2/ completo para entrega en USB. Incluye todos los modulos: models, training, data, processing, evaluation, visualization, gui, utils, cli.
- **Importancia**: ELIMINABLE (o BAJO si se conserva para archivo)
- **Justificacion**: Duplicado exacto del src_v2/ principal del proyecto. Ademas incluye directorios __pycache__/ con archivos .pyc compilados (43 archivos) que no deberian estar en un repositorio. Si se necesita un snapshot congelado, debe hacerse via git tags, no copiando archivos. Los .pyc son innecesarios y anaden peso al repositorio sin razon.

### 15. 03_Modelos/ (Checkpoints)
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/estancia/entregables_usb/03_Modelos/`
- **Lineas/Tamano**: 4 archivos .pt (~46 MB cada uno, ~184 MB total)
- **Proposito**: Copia de los 4 modelos del ensemble renombrados con prefijo seed para entrega en USB.
- **Importancia**: BAJO (duplicados de checkpoints/)
- **Justificacion**: Son copias de los modelos en checkpoints/ con nombres simplificados (seed123_final_model.pt vs. session10/ensemble/seed123/final_model.pt). Duplicar ~184 MB de modelos dentro del repositorio es ineficiente. Si estan versionados con Git LFS, es aun mas problematico. Deben eliminarse del repositorio y referenciarse desde su ubicacion canonica en checkpoints/.

### 16. 04_Configuraciones/ (JSON configs)
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/estancia/entregables_usb/04_Configuraciones/`
- **Lineas/Tamano**: 4 archivos JSON (~1.6 KB total)
- **Proposito**: Copia de las 4 configuraciones JSON criticas para entrega en USB.
- **Importancia**: ELIMINABLE
- **Justificacion**: Duplicado exacto de los archivos en configs/. Tamano negligible pero introduce riesgo de divergencia si se modifica uno pero no el otro.

### 17. 05_Documentacion/ - Indice Maestro (00_INDICE_MAESTRO.md)
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/estancia/entregables_usb/05_Documentacion/00_INDICE_MAESTRO.md`
- **Lineas/Tamano**: 613 lineas / ~25 KB
- **Proposito**: Punto de entrada unico a toda la documentacion tecnica del USB. Define 4 rutas de navegacion segun el perfil del usuario (Revisor Academico, Usuario Final, Investigador, Estudiante). Incluye descripcion de cada documento, mapa de navegacion por tarea, y verificacion de integridad.
- **Importancia**: MEDIO
- **Justificacion**: Excelente documento de navegacion para un paquete de entrega independiente. Sin embargo, dentro del repositorio principal, su valor es limitado porque el CLAUDE.md ya cumple esta funcion. Las rutas de lectura recomendadas son bien pensadas. Referencia documentos que no existen (06_CONFIGURACIONES_JSON.md, 09_GLOSARIO_TERMINOS.md, 10_PREGUNTAS_FRECUENTES.md, 00_LEEME.txt), lo que sugiere que la generacion del snapshot USB quedo incompleta.

### 18. 05_Documentacion/ - Guias principales (01-05, 07-08)
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/estancia/entregables_usb/05_Documentacion/`
- **Lineas/Tamano**: 8 archivos .md:
  - `01_GUIA_INICIO_RAPIDO.md` (391 lineas)
  - `02_INSTALACION_REQUISITOS.md` (844 lineas)
  - `03_GUIA_USO_CLI.md` (no leido, referenciado como 29 KB)
  - `04_ARQUITECTURA_CODIGO.md` (no leido, referenciado como 36 KB)
  - `05_REPRODUCIBILIDAD_COMPLETA.md` (no leido, referenciado como 22 KB)
  - `07_MODELOS_ENTRENADOS.md` (no leido, referenciado como 19 KB)
  - `08_FORMATOS_DATOS.md` (no leido, referenciado como 21 KB)
- **Proposito**: Suite de documentacion tecnica completa para el snapshot USB. Cubre desde inicio rapido hasta formatos de datos, pasando por arquitectura, CLI, reproducibilidad y modelos.
- **Importancia**: MEDIO
- **Justificacion**: Documentacion tecnica exhaustiva creada especificamente para el snapshot de entrega. Contenido original valioso no presente en otros archivos del repositorio (especialmente 02_INSTALACION con troubleshooting detallado, 04_ARQUITECTURA con mapa del codigo, 07_MODELOS con detalles del ensemble). Sin embargo, parte de esta informacion duplica docs/ existentes (REPRO_FULL_PIPELINE.md, CONFIGS.md, etc.). Los archivos 06 (configs), 09 (glosario) y 10 (FAQ) referenciados en el indice maestro NO EXISTEN -- el snapshot esta incompleto.

### 19. 05_Documentacion/ - Configs y requirements
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/estancia/entregables_usb/05_Documentacion/configs/` y `requirements.txt`
- **Lineas/Tamano**: 4 JSON + 1 txt
- **Proposito**: Copia de configuraciones y dependencias para el snapshot USB.
- **Importancia**: ELIMINABLE
- **Justificacion**: Duplicados de configs/ y requirements.txt de la raiz del proyecto.

### 20. Figuras del reporte USB (01_Reporte/Figures/)
- **Ruta**: `/home/donrobot/Projects/prediccion_warping_clasificacion/docs/estancia/entregables_usb/01_Reporte/Figures/`
- **Lineas/Tamano**: 8 archivos PNG
- **Proposito**: Figuras cientificas generadas para el reporte de estancia. Incluyen visualizaciones de landmarks, arquitectura del modelo, Wing Loss, GPA, triangulacion de Delaunay, warping, error por landmark, y matriz de confusion.
- **Importancia**: ALTO
- **Justificacion**: Son las unicas copias de estas figuras en el repositorio. Si se elimina el snapshot USB, estas figuras deben preservarse. Son necesarias para compilar el reporte V1 con figuras integradas.

---

## E. Observaciones Generales

### Discrepancias Detectadas

1. **Tamano del dataset**: El REPORTE_ESTANCIA_INAOE.md dice 21,165 imagenes; las versiones LaTeX (V1 y V2) dicen 15,153. El AUDITORIA_REPORTE_INAOE.md audita contra el .tex pero reporta 21,165 en sus propias tablas. El valor correcto es 15,153 (el subconjunto de 3 clases usado en el proyecto).

2. **Periodo de estancia en CONSTANCIA**: La constancia dice "marzo a julio de 2025"; el reporte dice "9 de octubre - 9 de noviembre de 2025". Deben verificarse ambas fechas. Si son estancias diferentes, la constancia deberia especificarlo. Si es un error, debe corregirse.

3. **Snapshot USB incompleto**: El indice maestro (00_INDICE_MAESTRO.md) referencia archivos que no existen: `06_CONFIGURACIONES_JSON.md`, `09_GLOSARIO_TERMINOS.md`, `10_PREGUNTAS_FRECUENTES.md`, `00_LEEME.txt`. Esto sugiere que la generacion del snapshot se interrumpio antes de completarse.

4. **V2 sin conclusiones/firmas**: La REPORTE_ESTANCIA_INAOE_V2.tex termina en el cronograma sin seccion de Conclusiones ni espacio para firmas, a diferencia de V1 que las tiene. Si V2 es la version final, debe completarse.

### Redundancia Excesiva

El directorio docs/estancia/ contiene un nivel preocupante de redundancia:

- **Reporte de estancia**: 3 versiones (MD, V1.tex, V2.tex) + 1 copia en USB + 2-3 PDFs
- **Configuraciones JSON**: 3 copias (configs/, entregables_usb/04_Configuraciones/, entregables_usb/05_Documentacion/configs/)
- **Codigo fuente**: 2 copias (src_v2/ + entregables_usb/02_Codigo/src_v2/)
- **Checkpoints**: 2 copias (checkpoints/ + entregables_usb/03_Modelos/, ~368 MB duplicados)
- **Manual**: 3 archivos de meta-documentacion (README, INSTRUCCIONES, RESUMEN) para 1 archivo de contenido

### Imagenes del Manual Incompletas

El manual referencia 16 imagenes pero solo 2 estan capturadas (interfaz_principal.png y sistema_cargado.png). Esto afecta la calidad del PDF compilado. Las 14 imagenes pendientes requieren ejecutar la GUI v15 y capturar pantallas segun INSTRUCCIONES_IMAGENES.md.

### Archivos __pycache__ en Snapshot

El snapshot USB incluye 43 archivos .pyc compilados en directorios __pycache__/. Estos no deberian estar en un repositorio ni en un snapshot de entrega. Deben eliminarse.

### Artefactos LaTeX

Hay ~29 archivos de artefactos LaTeX (.aux, .log, .fdb_latexmk, .fls, .synctex.gz, .out, .toc) distribuidos en docs/estancia/, docs/carta/ y docs/manual/. Estos consumen ~624 KB y son completamente regenerables. Deben eliminarse y anadirse al .gitignore.

---

## F. Resumen por Importancia

### CRITICO (3 archivos)
| Archivo | Ruta | Justificacion |
|---------|------|---------------|
| REPORTE_ESTANCIA_INAOE_V2.tex | docs/estancia/ | Version final compacta del reporte |
| carta_modificaciones_tesis.tex | docs/carta/ | Respuesta obligatoria a observaciones del jurado |
| CONSTANCIA_ESTANCIA_INAOE.tex | docs/carta/ | Constancia oficial del director (verificar fechas) |

### ALTO (8 archivos/grupos)
| Archivo | Ruta | Justificacion |
|---------|------|---------------|
| REPORTE_ESTANCIA_INAOE.tex (V1) | docs/estancia/ | Reporte formal con codigo y firmas |
| AUDITORIA_REPORTE_INAOE.md | docs/estancia/ | Verificacion de cumplimiento |
| PDFs de estancia (3) | docs/estancia/*.pdf | Entregables compilados + plan original |
| PDFs de carta (2) | docs/carta/*.pdf | Cartas firmables |
| manual_usuario.tex | docs/manual/ | Manual de usuario completo |
| manual_usuario.pdf | docs/manual/ | Manual compilado |
| Figuras reporte USB (8 PNG) | entregables_usb/01_Reporte/Figures/ | Unicas copias de figuras cientificas |

### MEDIO (5 archivos/grupos)
| Archivo | Ruta | Justificacion |
|---------|------|---------------|
| LISTA_ENTREGABLES.md | docs/estancia/ | Inventario, ya materializado |
| REPORTE_ESTANCIA_INAOE.md | docs/estancia/ | Borrador MD (discrepancia 21,165 vs. 15,153) |
| 00_INDICE_MAESTRO.md | entregables_usb/05_Documentacion/ | Navegacion de USB (refs rotas) |
| 05_Documentacion/ guias (8 md) | entregables_usb/05_Documentacion/ | Contenido original pero parcialmente duplicado |
| Imagenes manual (2 PNG) | docs/manual/imagenes/ | Solo 2/16 capturadas |

### BAJO (5 archivos/grupos)
| Archivo | Ruta | Justificacion |
|---------|------|---------------|
| compilar.sh | docs/manual/ | Script de conveniencia trivial |
| INSTRUCCIONES_IMAGENES.md | docs/manual/ | Proceso interno para completar manual |
| README.md (manual) | docs/manual/ | Redundante con otros meta-docs |
| 01_Reporte/ (.tex + PDF) | entregables_usb/01_Reporte/ | Duplicado de V1 |
| 03_Modelos/ (4 .pt) | entregables_usb/03_Modelos/ | Duplicado de checkpoints/ (~184 MB) |

### ELIMINABLE (7 archivos/grupos)
| Archivo | Ruta | Justificacion |
|---------|------|---------------|
| RESUMEN_GENERACION.md | docs/manual/ | Meta-doc autogenerado, totalmente redundante |
| Artefactos LaTeX estancia (~13) | docs/estancia/*.{aux,log,...} | ~444 KB regenerables |
| Artefactos LaTeX carta (~10) | docs/carta/*.{aux,log,...} | ~68 KB regenerables |
| Artefactos LaTeX manual (~6) | docs/manual/*.{aux,log,...} | ~112 KB regenerables |
| 02_Codigo/src_v2/ (+ __pycache__) | entregables_usb/02_Codigo/ | Duplicado + .pyc indeseados |
| 04_Configuraciones/ (4 JSON) | entregables_usb/04_Configuraciones/ | Duplicado de configs/ |
| 05_Documentacion/configs + req | entregables_usb/05_Documentacion/ | Triple duplicado de configs/ |

---

## G. Recomendaciones

1. **Limpiar artefactos LaTeX**: Eliminar ~29 archivos (.aux, .log, .fdb_latexmk, .fls, .synctex.gz, .out, .toc) y agregar patrones al .gitignore.

2. **Resolver discrepancia de fechas en constancia**: Verificar si el periodo "marzo a julio 2025" es correcto o es un error respecto a "9 oct - 9 nov 2025".

3. **Completar V2 del reporte**: Agregar seccion de Conclusiones y firmas formales al final de REPORTE_ESTANCIA_INAOE_V2.tex.

4. **Corregir dataset count en MD**: El REPORTE_ESTANCIA_INAOE.md y la AUDITORIA deben usar 15,153 consistentemente, no 21,165.

5. **Eliminar snapshot USB del repositorio**: El directorio entregables_usb/ duplica ~184 MB de modelos + todo el src_v2 + configs. Deberia existir como un tar.gz separado o generarse via script, no vivir en el repositorio.

6. **Preservar figuras antes de eliminar snapshot**: Las 8 figuras PNG en 01_Reporte/Figures/ son las unicas copias y deben moverse a un directorio permanente (ej: docs/estancia/Figures/).

7. **Completar imagenes del manual**: Capturar las 14 imagenes pendientes de la GUI v15 siguiendo INSTRUCCIONES_IMAGENES.md.

8. **Consolidar meta-documentacion del manual**: Fusionar README.md, INSTRUCCIONES_IMAGENES.md y RESUMEN_GENERACION.md en un solo archivo (o eliminar RESUMEN_GENERACION.md como minimo).

9. **Limpiar __pycache__ del snapshot**: Si se conserva el snapshot, eliminar los 43 archivos .pyc.

10. **Completar documentos faltantes del USB**: Si el snapshot USB se mantiene, crear los archivos referenciados pero inexistentes (06, 09, 10, 00_LEEME.txt) o actualizar el indice maestro para no referenciarlos.
