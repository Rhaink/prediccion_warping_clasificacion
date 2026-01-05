# Verificacion de Requisitos del Asesor

Este documento rastrea el cumplimiento de cada requisito especificado por el
asesor en la conversacion (ver `conversacion.txt`).

## Tabla de Verificacion

| # | Requisito | Cita del Asesor | Estado | Evidencia |
|---|-----------|-----------------|--------|-----------|
| 1 | Usar 2 clases | "solo va a haber dos clases... con neumonia (ya sea causada por otras razones o por COVID), pero neumonia a fin de cuentas, y las que estan sanas" | DOCUMENTADO | `config/LABEL_MAPPING.md` |
| 2 | Usar imagenes warped | "estoy hablando de las imagenes que ya pasaron por la alineacion, o sea que ya fueron warped" | PENDIENTE | |
| 3 | Mostrar imagenes al asesor | "a mi me gustaria ver esas imagenes" | PENDIENTE | |
| 4 | Construir Eigen-space con warped | "construyes un solo Eigen-space" con imagenes "ya ajustadas, ya warped" | PENDIENTE | |
| 5 | Seleccionar N Eigenfaces | "suponte que fueron 10 Eigenfaces son las principales... las 10 que corresponden a las 10 mayores varianzas" | PENDIENTE | |
| 6 | Caracteristicas = Ponderantes | "las caracteristicas no serian las Eigenfaces. Las caracteristicas serian los ponderantes... Los pesos" | DOCUMENTADO | `docs/01_MATEMATICAS.md` |
| 7 | Estandarizar Z-score | "de esos 1,000 valores sacas la media. Y le sacas la desviacion estandar... a cada valor le restas la media y luego esa diferencia la divides entre la desviacion estandar" | DOCUMENTADO | `docs/01_MATEMATICAS.md` |
| 8 | Fisher por caracteristica | "tomas los que nada mas son para la neumonia... Media 1... Para los que son sanos... Media numero 2... Y con esos sacas el Criterio de Fisher" | DOCUMENTADO | `docs/01_MATEMATICAS.md` |
| 9 | Fisher como amplificador | "Con esa Razon de Fisher, puedes usarla como un ponderante... todos los datos de la caracteristica los podrias multiplicar por esa cosa" | DOCUMENTADO | `docs/01_MATEMATICAS.md` |
| 10 | Clasificador simple (KNN) | "puede ser hasta uno bien simple. Puede ser un KNN" | OK | `src/classifier.py` - KNN desde cero |
| 11 | NO usar CNN | "la red neuronal convolucional en realidad ya esta preparada para que las imagenes no esten alineadas... no es tan fiable para demostrar" | OK | No se usa CNN |
| 12 | Demostrar que warping importa | "Lo que hace la diferencia es que hayas alineado esas canijas imagenes... van a funcionar mejor incluso con un clasificador tan chafa o tan simple como un KNN" | OK | Full: +2.21%, Manual: +5.21% |

## Estados

- **OK**: Requisito cumplido completamente
- **DOCUMENTADO**: Requisito entendido y documentado, pendiente implementacion
- **PENDIENTE**: Aun no se ha trabajado en este requisito
- **EN PROGRESO**: Actualmente trabajando en este requisito

## Resumen

- Total requisitos: 12
- Cumplidos (OK): 3
- Documentados: 5
- Pendientes: 4

## Notas

### Sobre el uso de 2 clases

El asesor explico que COVID causa neumonia y visualmente es similar a otros
tipos de neumonia en radiografias. Por eso simplificamos a:
- **Enfermo**: COVID + Viral_Pneumonia
- **Normal**: Normal

Ver `config/LABEL_MAPPING.md` para el mapeo completo.

### Sobre el clasificador

El asesor enfatizo que el clasificador NO es lo importante:
> "No va a ser mucha la diferencia en la clasificacion. Lo que hace la
> diferencia es que hayas alineado esas canijas imagenes."

Por eso usamos KNN simple. El objetivo es demostrar que con imagenes
alineadas (warped), incluso un clasificador basico funciona bien.

### Sobre CNN

El asesor advirtio que CNN no es adecuada para esta demostracion:
> "cuando se las das directamente a la red neuronal convolucional (CNN),
> la red neuronal convolucional en realidad ya esta preparada para que
> las imagenes no esten alineadas"

Las CNNs tienen mecanismos internos (pooling, convoluciones) que las hacen
parcialmente invariantes a desalineaciones. Usar CNN ocultaria el beneficio
del warping.

## Actualizaciones

| Fecha | Requisito | Cambio |
|-------|-----------|--------|
| 2025-12-30 | Todos | Creacion inicial del documento |
