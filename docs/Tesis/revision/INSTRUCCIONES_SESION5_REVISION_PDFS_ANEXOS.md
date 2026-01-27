# Instrucciones para Sesión 5 - Revisión y Corrección de PDFs en Anexos

**Fecha de creación:** 2026-01-26
**Estado actual:** C3 (Anexos) REQUIERE REVISIÓN - Fase 1 AÚN NO COMPLETADA
**Prioridad:** 🔴 CRÍTICA - Debe completarse ANTES de cerrar Fase 1

---

## 📋 CONTEXTO RÁPIDO

En la Sesión 4 se crearon los Anexos B (Artículos Publicados) y C (Certificados y Reconocimientos), integrando 6 archivos PDF. Sin embargo, **NO se revisó el contenido real de cada PDF**, lo que resultó en:

1. **Títulos potencialmente incorrectos** en los archivos .tex
2. **Descripciones que no coinciden** con el contenido real de los PDFs
3. **Información incorrecta** sobre eventos, fechas, ISBN, etc.
4. **Anexo A (GUI) debe comentarse temporalmente** para revisión posterior

---

## 🎯 OBJETIVOS DE ESTA SESIÓN

### 1. Comentar Anexo A (Interfaz Gráfica) Temporalmente
**Razón:** Requiere revisión posterior, no bloquear avance de Anexos B y C

**Acción:**
- [ ] Comentar línea en `main.tex` que incluye `anexo_A_manual_usuario.tex`
- [ ] Agregar comentario explicativo: "% TEMPORAL: Comentado para revisión posterior"
- [ ] NO eliminar el archivo, solo desactivar su inclusión

### 2. Revisar Contenido Real de Artículos (Anexo B)

**PDFs a revisar:**
1. `articulo_ingles_sahs_2024.pdf`
2. `articulo_espanol_sahs_2024.pdf`

**Checklist por PDF:**
- [ ] Abrir y leer PDF completo
- [ ] Verificar título exacto del artículo
- [ ] Verificar autores (nombres completos y orden)
- [ ] Verificar evento/congreso real
- [ ] Verificar fecha de publicación
- [ ] Verificar ISBN (si aplica)
- [ ] Verificar editorial/institución publicadora
- [ ] Identificar tema principal del artículo
- [ ] Determinar si ambos artículos son sobre el mismo tema

**Preguntas a responder:**
1. ¿Ambos artículos hablan del método SAHS (Statistical Asymmetrical Histogram Stretching)?
2. ¿Son sobre la tesis actual o sobre trabajo previo/relacionado?
3. ¿Hay relación directa con landmarks y warping, o solo con preprocesamiento?
4. ¿Los eventos NOVA y RAS DAY son correctos?

### 3. Revisar Contenido Real de Certificados (Anexo C)

**PDFs a revisar:**
1. `reconocimiento_nova.pdf`
2. `reconocimiento_ieee_day.pdf`
3. `constancia_participacion_rafael.pdf`
4. `reconocimiento_ras_day.pdf`

**Checklist por PDF:**
- [ ] Abrir y leer PDF completo
- [ ] Verificar tipo de documento (reconocimiento, constancia, certificado)
- [ ] Verificar evento exacto
- [ ] Verificar fecha del evento
- [ ] Verificar institución emisora
- [ ] Verificar motivo del reconocimiento (participación, ponencia, asistencia, etc.)
- [ ] Identificar si está relacionado con artículos del Anexo B

**Preguntas a responder:**
1. ¿Qué eventos corresponden a qué artículos?
2. ¿Hay certificados de eventos donde NO se presentó artículo?
3. ¿Las fechas son coherentes con la línea temporal de la tesis?
4. ¿Todos los certificados son relevantes para la tesis?

### 4. Corregir anexo_B_articulos_publicados.tex

**Basado en revisión de PDFs:**
- [ ] Corregir títulos exactos de artículos
- [ ] Corregir nombres de autores (si hay errores)
- [ ] Corregir información de publicación (evento, fecha, ISBN)
- [ ] Actualizar descripción del contenido de cada artículo
- [ ] Verificar coherencia entre Anexo B y Anexo C (eventos)

**Estructura sugerida por artículo:**
```latex
\section{Título Exacto del Artículo}

\textbf{Autores:} [Lista exacta según PDF]

\textbf{Evento:} [Nombre completo del evento según certificado]

\textbf{Fecha:} [Fecha del evento]

\textbf{Publicación:} [Memorias/Proceedings exactos, ISBN si aplica]

\textbf{Resumen:} [1-2 párrafos describiendo qué trata el artículo y cómo se relaciona con la tesis]

\includepdf[pages=-,pagecommand={\thispagestyle{plain}}]{anexos/archivo.pdf}
```

### 5. Corregir anexo_C_certificados.tex

**Basado en revisión de PDFs:**
- [ ] Corregir nombres exactos de eventos
- [ ] Corregir tipo de documento (reconocimiento vs constancia vs certificado)
- [ ] Corregir instituciones emisoras
- [ ] Agregar fechas de eventos (si visibles en PDFs)
- [ ] Vincular con artículos del Anexo B (si aplica)

**Estructura sugerida por certificado:**
```latex
\section{[Tipo de Documento] - [Evento Exacto]}

[Párrafo breve describiendo el evento, la fecha, y el motivo del reconocimiento.
Si está relacionado con algún artículo del Anexo B, mencionarlo explícitamente.]

\textbf{Institución:} [Institución emisora]
\textbf{Fecha:} [Fecha del evento]
\textbf{Motivo:} [Ponencia, participación, asistencia, etc.]

\includepdf[pages=-,pagecommand={\thispagestyle{plain}}]{anexos/archivo.pdf}
```

---

## 🔍 METODOLOGÍA DE REVISIÓN

### Paso 1: Revisión Individual de PDFs (40 min)

**Para cada PDF:**
1. Abrir PDF en visor
2. Leer completamente (o escanear si es muy largo)
3. Tomar notas detalladas:
   - Título exacto
   - Autores
   - Evento/institución
   - Fecha
   - Tema principal
   - Relación con la tesis
4. Capturar información en tabla temporal

**Tabla de revisión sugerida:**

| Archivo PDF | Tipo | Título/Evento | Fecha | Institución | Tema/Motivo | Relación con Tesis |
|-------------|------|---------------|-------|-------------|-------------|-------------------|
| articulo_ingles_sahs_2024.pdf | Artículo | ? | ? | ? | ? | ? |
| articulo_espanol_sahs_2024.pdf | Artículo | ? | ? | ? | ? | ? |
| reconocimiento_nova.pdf | Certificado | ? | ? | ? | ? | ? |
| reconocimiento_ieee_day.pdf | Certificado | ? | ? | ? | ? | ? |
| constancia_participacion_rafael.pdf | Certificado | ? | ? | ? | ? | ? |
| reconocimiento_ras_day.pdf | Certificado | ? | ? | ? | ? | ? |

### Paso 2: Análisis de Coherencia (10 min)

**Verificar:**
1. ¿Los certificados corresponden a los artículos?
2. ¿Las fechas son coherentes?
3. ¿Hay información contradictoria?
4. ¿Todos los documentos son relevantes para la tesis?

### Paso 3: Corrección de Archivos .tex (20 min)

**Orden:**
1. Corregir `anexo_B_articulos_publicados.tex` primero
2. Corregir `anexo_C_certificados.tex` segundo (puede referenciar Anexo B)
3. Verificar coherencia entre ambos anexos

### Paso 4: Comentar Anexo A (5 min)

**En `main.tex`:**
```latex
% TEMPORAL: Comentado para revisión posterior (Sesión 6)
% \input{anexos/anexo_A_manual_usuario}
```

### Paso 5: Verificación Final (10 min)

- [ ] Compilar LaTeX y verificar que no haya errores
- [ ] Revisar que PDFs se incluyan correctamente
- [ ] Verificar que información sea coherente
- [ ] Actualizar documentación de progreso

**Tiempo total estimado:** ~85 minutos

---

## 📂 ARCHIVOS CLAVE PARA ESTA SESIÓN

### Archivos a Leer (PDFs):
1. `docs/Tesis/anexos/articulo_ingles_sahs_2024.pdf`
2. `docs/Tesis/anexos/articulo_espanol_sahs_2024.pdf`
3. `docs/Tesis/anexos/reconocimiento_nova.pdf`
4. `docs/Tesis/anexos/reconocimiento_ieee_day.pdf`
5. `docs/Tesis/anexos/constancia_participacion_rafael.pdf`
6. `docs/Tesis/anexos/reconocimiento_ras_day.pdf`

### Archivos a Modificar:
1. `docs/Tesis/anexos/anexo_B_articulos_publicados.tex` - Corregir información
2. `docs/Tesis/anexos/anexo_C_certificados.tex` - Corregir información
3. `docs/Tesis/main.tex` - Comentar Anexo A
4. `docs/Tesis/revision/CHECKLIST_OBSERVACIONES.md` - Actualizar C3
5. `docs/Tesis/revision/PROGRESO_REVISION.md` - Actualizar estado

### Archivos de Contexto:
1. `docs/Tesis/revision/RESUMEN_SESION4_ANEXOS_COMPLETADO.md` - Sesión anterior
2. Este archivo (`INSTRUCCIONES_SESION5_REVISION_PDFS_ANEXOS.md`)

---

## 💡 PREGUNTAS CRÍTICAS A RESPONDER

### Sobre los Artículos:
1. ¿Son ambos artículos sobre SAHS (mejora de contraste)?
2. ¿SAHS es parte de la metodología principal de la tesis o trabajo previo?
3. ¿Hay algún artículo sobre landmarks + warping específicamente?
4. ¿Los ISBNs son correctos?

### Sobre los Certificados:
1. ¿Qué certificados corresponden a ponencias de artículos?
2. ¿Qué certificados son por asistencia/participación sin ponencia?
3. ¿Hay certificados no relacionados con la tesis que deban excluirse?

### Sobre la Coherencia:
1. ¿La información en Anexo B coincide con certificados de Anexo C?
2. ¿Las fechas son coherentes?
3. ¿Los nombres de eventos coinciden entre artículos y certificados?

---

## ✅ CRITERIOS DE ÉXITO PARA ESTA SESIÓN

Al finalizar, se debe tener:
- ✅ Anexo A comentado temporalmente en main.tex
- ✅ Todos los PDFs revisados y documentados en tabla
- ✅ anexo_B_articulos_publicados.tex corregido con información exacta
- ✅ anexo_C_certificados.tex corregido con información exacta
- ✅ Coherencia verificada entre Anexos B y C
- ✅ LaTeX compila sin errores
- ✅ Documentación de progreso actualizada
- ✅ C3 marcado como "EN REVISIÓN FINAL" (no completado aún)

---

## 🔗 COMANDO PARA INICIAR LA SESIÓN 5

**Copiar y pegar al inicio de la nueva conversación:**

```
Continúo con la revisión de anexos de mi tesis (Sesión 5).

En la Sesión 4 creé los Anexos B y C, pero NO revisé el contenido real
de los PDFs, resultando en información potencialmente incorrecta.

Por favor lee el archivo de instrucciones:
docs/Tesis/revision/INSTRUCCIONES_SESION5_REVISION_PDFS_ANEXOS.md

Necesito:
1. Comentar temporalmente Anexo A (GUI) en main.tex
2. Revisar contenido REAL de 6 PDFs (2 artículos + 4 certificados)
3. Corregir títulos, descripciones y datos en anexo_B_articulos_publicados.tex
4. Corregir información en anexo_C_certificados.tex
5. Verificar coherencia entre artículos y certificados
6. Actualizar documentación

Archivos de contexto:
- docs/Tesis/revision/RESUMEN_SESION4_ANEXOS_COMPLETADO.md (sesión anterior)
- docs/Tesis/revision/PROGRESO_REVISION.md (estado general)
- docs/Tesis/revision/CHECKLIST_OBSERVACIONES.md (C3 pendiente)

PDFs en: docs/Tesis/anexos/
```

---

## 📝 PLANTILLA DE REPORTE DE REVISIÓN

**Al revisar cada PDF, usa este formato:**

```markdown
### [Nombre del archivo]

**Tipo:** [Artículo/Certificado/Reconocimiento/Constancia]

**Información encontrada:**
- Título/Evento: [texto exacto del PDF]
- Autores/Participante: [nombres completos]
- Fecha: [fecha del evento/publicación]
- Institución: [institución emisora]
- ISBN (si aplica): [número]
- Tema principal: [breve descripción]

**Relación con la tesis:**
[¿Cómo se relaciona este documento con la investigación de la tesis?]

**Errores encontrados en .tex actual:**
- Error 1: [descripción]
- Error 2: [descripción]

**Correcciones necesarias:**
- [ ] Corregir título/evento
- [ ] Corregir autores
- [ ] Corregir fecha
- [ ] Actualizar descripción
```

---

## 🚨 ADVERTENCIAS IMPORTANTES

1. **NO asumir información** - Leer PDFs completos
2. **NO usar información de Sesión 4** - Puede estar incorrecta
3. **Verificar ortografía** de nombres propios, eventos, instituciones
4. **Diferenciar** entre ponencia (presentación) y participación (asistencia)
5. **Vincular** artículos con certificados cuando corresponda
6. **Mantener** nombres de archivos PDF (no renombrar nuevamente)

---

## 🎯 RESULTADO ESPERADO

Al finalizar esta sesión, los Anexos B y C deberán:
1. Reflejar **exactamente** el contenido de los PDFs
2. Tener **coherencia interna** (artículos ↔ certificados)
3. Usar **terminología precisa** (ponencia vs participación, etc.)
4. Incluir **información completa** (fechas, instituciones, ISBN)
5. Tener **descripciones claras** de la relación con la tesis

Esto permitirá cerrar C3 definitivamente y completar la Fase 1 al 100%.

---

**Preparado por:** Claude Code (Sesión 4)
**Para:** Sesión 5 - Revisión y Corrección de PDFs en Anexos
**Estado C3:** EN REVISIÓN CRÍTICA - Fase 1 pendiente de completar
**Prioridad:** 🔴 CRÍTICA
