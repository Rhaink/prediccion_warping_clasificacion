# Verificacion de Requisitos del Asesor

Este documento rastrea el cumplimiento de cada requisito especificado por el
asesor en la conversacion (ver `conversacion.txt`).

## Tabla de Verificacion

| # | Requisito | Cita del Asesor | Estado | Evidencia |
|---|-----------|-----------------|--------|-----------|
| 1 | Usar 2 clases (principal) | "solo va a haber dos clases... con neumonia (ya sea causada por otras razones o por COVID), pero neumonia a fin de cuentas, y las que estan sanas" | OK | Fase 6 completa. Ver `results/metrics/phase6_classification/` |
| 2 | Usar imagenes warped | "estoy hablando de las imagenes que ya pasaron por la alineacion, o sea que ya fueron warped" | OK | `outputs/full_warped_dataset/`, `outputs/warped_dataset/` |
| 3 | Mostrar imagenes al asesor | "a mi me gustaria ver esas imagenes" | LISTO | `results/figures/phase2_samples/` + `README_ASESOR.md` |
| 4 | Construir Eigen-space con warped | "construyes un solo Eigen-space" con imagenes "ya ajustadas, ya warped" | OK | `src/pca.py`, Fase 3 |
| 5 | Seleccionar N Eigenfaces | "suponte que fueron 10 Eigenfaces son las principales... las 10 que corresponden a las 10 mayores varianzas" | OK | Usamos 50 componentes. Ver `results/figures/phase3_pca/` |
| 6 | Caracteristicas = Ponderantes | "las caracteristicas no serian las Eigenfaces. Las caracteristicas serian los ponderantes... Los pesos" | OK | `src/features.py`, Fase 4 |
| 7 | Estandarizar Z-score | "de esos 1,000 valores sacas la media. Y le sacas la desviacion estandar... a cada valor le restas la media y luego esa diferencia la divides entre la desviacion estandar" | OK | `src/features.py`, Fase 4 |
| 8 | Fisher por caracteristica | "tomas los que nada mas son para la neumonia... Media 1... Para los que son sanos... Media numero 2... Y con esos sacas el Criterio de Fisher" | OK | `src/fisher.py`, Fase 5 |
| 9 | Fisher como amplificador | "Con esa Razon de Fisher, puedes usarla como un ponderante... todos los datos de la caracteristica los podrias multiplicar por esa cosa" | OK | `src/fisher.py`, Fase 5 |
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
- Cumplidos (OK): 12
- Pendientes: 0

**TODOS LOS REQUISITOS DEL ASESOR HAN SIDO CUMPLIDOS**

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

### Sobre Fase 7: Experimento con 3 clases

El asesor menciono que "en teoria podriamos establecer tres clases", aunque
recomendo simplificar a 2 clases. La Fase 7 implementa el escenario secundario
de 3 clases (COVID vs Normal vs Viral_Pneumonia) como experimento adicional.

**Extension de Fisher a multiclase (pairwise):**

El asesor solo explico Fisher para 2 clases. Para 3 clases, extendimos la
formula usando el enfoque pairwise (promedio de pares):

```
J = (J_COVID_Normal + J_COVID_Viral + J_Normal_Viral) / 3
```

Esta extension es matematicamente valida y consistente con la intuicion
del asesor: mide que tan bien separa cada caracteristica TODOS los pares
de clases.

**Resultados Fase 7:**
- El warping mejora clasificacion tanto en 2C como en 3C
- Mejora 2C: +2.21% (Full), +5.21% (Manual)
- Mejora 3C: +0.74% (Full), +2.08% (Manual)

## Actualizaciones

| Fecha | Requisito | Cambio |
|-------|-----------|--------|
| 2025-12-30 | Todos | Creacion inicial del documento |
| 2026-01-05 | 1-12 | Actualizacion de estados (11 OK, 1 pendiente) |
| 2026-01-05 | Fase 7 | Documentacion de extension Fisher multiclase |
| 2026-01-07 | Todos | Post error crítico: agregadas salvaguardas de verificación |

---

## IMPORTANTE: Salvaguardas Post-Error Crítico (2026-01-07)

### ¿Qué pasó?

Los experimentos de 2 clases fueron ejecutados con el CSV incorrecto (`01_full_balanced_3class_*.csv` en lugar de `02_full_balanced_2class_*.csv`), resultando en:
- Solo 680 de 1,245 imágenes test usadas (45%)
- Ratio de clases invertido (60% Enfermo vs 40% Normal)
- ~3 días de trabajo perdido

Ver `docs/POST_MORTEM_ERROR_CRITICO.md` para análisis completo.

### Salvaguardas Implementadas

Para asegurar que TODOS los requisitos del asesor se cumplan correctamente y que este tipo de error nunca vuelva a pasar:

1. **VERIFICATION_CHECKLIST.md** - Checklist obligatorio después de cada fase
2. **SPLIT_PROTOCOL.md actualizado** - Reglas prescriptivas (no solo descriptivas)
3. **Validaciones automáticas** - Asserts en el código para verificar tamaños
4. **Logging explícito** - Prints de archivos cargados y tamaños
5. **CORRECTION_PLAN.md** - Plan detallado para corregir experimentos

### Verificación Pre-Reunión con Asesor

Antes de presentar resultados al asesor, VERIFICAR:

- [ ] **Dataset correcto usado:** 02_full_balanced_2class_*.csv para 2 clases
- [ ] **Test size correcto:** 1,245 imágenes (no 680)
- [ ] **Ratio de clases:** 40% Enfermo, 60% Normal (no invertido)
- [ ] **Todos los requisitos cumplidos:** Revisar tabla de verificación arriba
- [ ] **Coherencia de números:** Summary.json coincide con notebooks
- [ ] **Figuras correctas:** Todas las rutas existen y tienen datos correctos

### Referencias

- `docs/POST_MORTEM_ERROR_CRITICO.md` - Análisis del error
- `docs/VERIFICATION_CHECKLIST.md` - Checklist obligatorio
- `docs/CORRECTION_PLAN.md` - Plan de corrección
- `config/SPLIT_PROTOCOL.md` - Reglas de uso de CSVs
