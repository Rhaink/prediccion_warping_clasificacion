# PROMPT DE CONTINUACIÓN - SESIÓN 07 DE REDACCIÓN DE TESIS (REVISIÓN)

## INSTRUCCIONES PARA CLAUDE

Lee el archivo `Documentos/Tesis/prompts/prompt_tesis.md` para entender tu rol como Asesor Senior de Tesis y el proceso de trabajo en fases.

**IMPORTANTE - ESTA SESIÓN ES DE REVISIÓN, NO DE REDACCIÓN NUEVA**

---

## CONTEXTO DE LA SESIÓN ANTERIOR

### Fecha de sesión anterior: 16 Diciembre 2025 (Sesión 06)

### Estado del Proyecto de Tesis

| Fase | Estado | Descripción |
|------|--------|-------------|
| Fase 1: Análisis del Proyecto | ✅ COMPLETADA | Análisis exhaustivo del código, resultados, documentación |
| Fase 2: Estructura de Tesis | ✅ COMPLETADA | Estructura de 6 capítulos aprobada |
| Fase 3: Redacción | 🔄 EN PROGRESO | **Capítulo 4 COMPLETADO** - Pendiente revisión visual |
| Fase 4: Revisión Final | ⏳ PENDIENTE | — |

---

## OBJETIVO DE ESTA SESIÓN

**SESIÓN DE REVISIÓN VISUAL Y CORRECCIÓN DEL CAPÍTULO 4**

El usuario ha compilado el documento PDF y ha identificado problemas visuales. Esta sesión se dedica exclusivamente a:

1. **Identificar y corregir problemas visuales en el PDF:**
   - Tablas desbordadas o mal formateadas
   - Ecuaciones cortadas o mal alineadas
   - Algoritmos con problemas de sintaxis LaTeX
   - Figuras placeholder que necesitan ajuste
   - Espaciado inadecuado
   - Viudas y huérfanas
   - Referencias cruzadas rotas

2. **Revisión exhaustiva del contenido del Capítulo 4:**
   - Verificar consistencia de datos entre secciones
   - Detectar errores de redacción o gramática
   - Verificar que todas las referencias cruzadas funcionan
   - Revisar numeración de ecuaciones, tablas y figuras

3. **NO avanzar con nuevo contenido** - Capítulo 5 queda pospuesto

---

## ARCHIVOS A REVISAR

### Documento compilado:
- `Documentos/Tesis/main.pdf` - **Revisar visualmente**

### Archivos LaTeX del Capítulo 4:

| Archivo | Páginas PDF aprox. | Contenido |
|---------|-------------------|-----------|
| `capitulo4/4_1_descripcion_general.tex` | 2-3 | Pipeline general |
| `capitulo4/4_2_dataset_preprocesamiento.tex` | 4-7 | Dataset, CLAHE |
| `capitulo4/4_3_modelo_landmarks.tex` | 8-17 | ResNet-18 + CoordAttn |
| `capitulo4/4_4_normalizacion_geometrica.tex` | 18-29 | GPA, Delaunay, Warping |
| `capitulo4/4_5_clasificacion.tex` | 30-40 | Clasificador CNN |
| `capitulo4/4_6_protocolo_evaluacion.tex` | 41-51 | Protocolo evaluación |

### Archivo principal:
- `main.tex` - Incluye portada + Capítulo 4

---

## PROBLEMAS COMUNES A BUSCAR

### 1. Problemas de Tablas
- [ ] Tablas que exceden el ancho de página (`tabularx` vs `tabular`)
- [ ] Celdas con texto cortado
- [ ] Alineación inconsistente de columnas
- [ ] Líneas horizontales excesivas o faltantes
- [ ] Tablas sin `\centering`

### 2. Problemas de Ecuaciones
- [ ] Ecuaciones numeradas inconsistentemente
- [ ] Ecuaciones demasiado largas sin split
- [ ] Símbolos matemáticos incorrectos
- [ ] Paréntesis desbalanceados
- [ ] `\text{}` faltante en texto dentro de ecuaciones

### 3. Problemas de Algoritmos
- [ ] Comandos `\STATE`, `\FOR`, `\IF` en mayúsculas (algpseudocode prefiere minúsculas)
- [ ] Indentación incorrecta
- [ ] `\RETURN` vs `\Return`
- [ ] Texto en español vs inglés mezclado

### 4. Problemas de Figuras
- [ ] Placeholders `[FIGURA PENDIENTE]` mal formateados
- [ ] Captions demasiado largos
- [ ] Figuras sin `\centering`
- [ ] Referencias a figuras inexistentes

### 5. Problemas de Formato General
- [ ] Viudas (línea huérfana al inicio de página)
- [ ] Huérfanas (línea sola al final de página)
- [ ] Espaciado inconsistente entre secciones
- [ ] Overflow de texto (badness warnings)

### 6. Problemas de Referencias
- [ ] `\ref{}` a labels inexistentes
- [ ] Labels duplicados
- [ ] Referencias cruzadas que muestran "??"

---

## PROCESO SUGERIDO

### Paso 1: Solicitar problemas específicos al usuario
```
El usuario mencionó que observó problemas visuales en el PDF.
PREGUNTAR:
- ¿Cuáles son los problemas específicos que observaste?
- ¿En qué páginas o secciones están?
- ¿Puedes describir qué ves mal?
```

### Paso 2: Revisar archivos LaTeX
Una vez identificados los problemas, revisar los archivos `.tex` correspondientes.

### Paso 3: Aplicar correcciones
- Usar el tool `Edit` para correcciones puntuales
- Recompilar después de cada grupo de correcciones

### Paso 4: Verificar correcciones
Recompilar el PDF y confirmar que los problemas fueron resueltos.

---

## ERRORES YA CORREGIDOS EN SESIONES ANTERIORES

**NO volver a introducir estos errores:**

| Sesión | Error | Corrección |
|--------|-------|------------|
| 03 | Estructura cabeza 2 capas | 3 capas (512→512→768→30) |
| 03 | CombinedLandmarkLoss | Solo WingLoss |
| 03 | Dropout 0.5/0.25 | 0.3/0.15 |
| 04 | Trade-off fill rate en Cap.4 | Reservado para Cap.5 |
| 05 | Ensemble de clasificadores existe | NO existe |
| 05 | TTA aplica a clasificación | Solo a landmarks |
| 06 | L9, L10 en "ápex pulmonar" | "eje central" |
| 06 | L12, L13 en "ángulos costofrénicos" | "bordes superiores" |
| 06 | Kernel blur 5×5 | Automático según σ |
| 06 | `\RETURN` en algoritmos | `\State \Return` |

---

## HISTORIAL DE CORRECCIONES LaTeX (Sesión 06)

```latex
% Error: \RETURN no existe en algpseudocode
% Corrección: Usar \State \Return
% Archivos afectados: 4_4_normalizacion_geometrica.tex (líneas 98, 235)
```

---

## VERIFICACIONES DE COMPILACIÓN

### Warnings a revisar en `main.log`:
- `Underfull \hbox` - Espaciado horizontal insuficiente
- `Overfull \hbox` - Texto excede el margen
- `Undefined reference` - Referencias rotas
- `Label multiply defined` - Labels duplicados

### Comando de compilación:
```bash
cd Documentos/Tesis && pdflatex -interaction=nonstopmode main.tex
```

Para ver warnings específicos:
```bash
grep -E "(Underfull|Overfull|Undefined|multiply)" main.log
```

---

## ARCHIVOS DE CONTEXTO

| Archivo | Contenido |
|---------|-----------|
| `Documentos/Tesis/DECISIONES_FASE_1.md` | Decisiones tomadas, claims validados/invalidados |
| `Documentos/Tesis/ESTRUCTURA_TESIS.md` | Estructura de 6 capítulos, historial de sesiones |
| `Documentos/Tesis/FIGURAS_PENDIENTES.md` | Lista de figuras por crear |
| `GROUND_TRUTH.json` | Valores validados experimentalmente |

---

## CLAIMS CIENTÍFICOS VALIDADOS (Para referencia)

| Claim | Valor | Fuente |
|-------|-------|--------|
| Error de landmarks (ensemble 4 + TTA) | **3.71 px** | GROUND_TRUTH.json |
| Accuracy clasificación (warped_96) | **99.10%** | GROUND_TRUTH.json |
| Mejora robustez JPEG Q50 | **30×** | GROUND_TRUTH.json |
| Mejora generalización cross-dataset | **2.4×** | GROUND_TRUTH.json |

---

## COMANDO INICIAL SUGERIDO

```
Estoy revisando el PDF compilado del Capítulo 4 y encontré los siguientes
problemas visuales:

[USUARIO: describir problemas específicos aquí]

Por favor ayúdame a corregirlos.
```

---

## DESPUÉS DE LA REVISIÓN

Una vez completada la revisión visual y las correcciones:

1. **Actualizar `ESTRUCTURA_TESIS.md`** con los problemas encontrados y corregidos
2. **Recompilar** el documento final
3. **Confirmar** con el usuario que los problemas fueron resueltos
4. **Generar prompt para Sesión 08** para continuar con Capítulo 5

---

## RECORDATORIOS

- Esta sesión es de **REVISIÓN**, no de contenido nuevo
- **NO modificar datos o valores científicos** - solo formato visual
- Si encuentras errores de contenido durante la revisión, **documentarlos** pero consultar antes de cambiar
- Mantener backup mental de cambios realizados para el historial

---

*Prompt generado: 16 Diciembre 2025 - Sesión 06 (para Sesión 07 de Revisión)*
