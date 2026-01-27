# Instrucciones para Sesión 4 - Revisión y Completado de Anexos

**Fecha de creación:** 2026-01-26
**Estado actual:** Fase 1 completada, pero C3 (Anexos) requiere revisión y expansión
**Prioridad:** 🔴 CRÍTICA - Debe completarse ANTES de iniciar Fase 2

---

## 📋 CONTEXTO RÁPIDO

En la Sesión 3 se completó la tarea C3 (Anexos) creando el Anexo A: Manual de Usuario de la GUI. Sin embargo, se identificaron las siguientes necesidades adicionales:

1. **Errores LaTeX en anexo_A_manual_usuario.tex** (verificar si persisten)
2. **PDFs nuevos agregados** que deben incluirse como anexos
3. **Versión de la GUI** - El manual es de v15, verificar si es la versión correcta
4. **Anexos adicionales críticos** - Identificar qué más debe incluirse

---

## 🎯 OBJETIVOS DE ESTA SESIÓN

### 1. Corregir Errores LaTeX en Anexo A
**Archivo:** `docs/Tesis/anexos/anexo_A_manual_usuario.tex`

**Errores reportados:**
- Línea 50: Overfull \hbox (100.9907pt too wide)
- Línea 59: Underfull \hbox (badness 2521)
- Líneas 96, 109, 121: Posibles emojis Unicode residuales

**Acciones:**
- [ ] Verificar si los errores persisten con lectura directa del archivo
- [ ] Corregir overfull/underfull (usar `\texttt{}`, `\small`, o `\linebreak`)
- [ ] Confirmar eliminación de todos los emojis Unicode

### 2. Integrar PDFs Existentes como Anexos
**Ubicación:** `docs/Tesis/anexos/` (PDFs ya presentes)

**PDFs identificados:**

**Artículos Publicados:**
1. `Statistical Asymmetrical Histogram Stretching for Contrast Enhancement in Chest X-ray Images for Pneumonia Detection.pdf`
2. `Enfoque de expansión estadística de histograma asimétrico para mejorar el contraste en imágenes de radiografías de tórax para la detección de neumonía.pdf`

**Certificados y Reconocimientos:**
3. `Reconocimientos_NOVA_Rafael.pdf`
4. `Rafael_IEEE_DAY_Reconocimeinto.pdf`
5. `Constancias Participantes-Rafael Alejandro Cruz Ovando.pdf`
6. `Reconocimientos_RAS_DAY_Rafael_PONENCIA.pdf`

**Acciones:**
- [ ] Investigar mejor práctica para incluir PDFs en tesis LaTeX
- [ ] Decidir estructura de anexos (¿Un anexo por categoría?)
- [ ] Crear archivos .tex que incluyan estos PDFs con `\includepdf`

### 3. Verificar Versión de la GUI
**Archivo a revisar:** `src_v2/gui/README.md` y `src_v2/gui/CHANGELOG.md`

**Pregunta:** ¿El manual de usuario describe la versión 15 de la GUI? ¿Es la versión correcta?

**Acciones:**
- [ ] Leer CHANGELOG.md para verificar versión actual
- [ ] Confirmar que las funcionalidades descritas coinciden con la versión final
- [ ] Actualizar Anexo A si hay discrepancias

### 4. Identificar Anexos Adicionales Críticos
**Objetivo:** Determinar qué otros anexos son de suma importancia para la tesis

**Categorías a considerar:**

**A. Anexos Técnicos:**
- Especificaciones del dataset (COVID-19 Radiography Dataset)
- Configuraciones de experimentos (archivos JSON de configs/)
- Detalles de arquitecturas de modelos
- Tablas de resultados completos por clase

**B. Anexos de Reproducibilidad:**
- Instrucciones completas de reproducción
- Lista de dependencias con versiones exactas
- Hardware utilizado y tiempos de entrenamiento

**C. Anexos Metodológicos:**
- Detalles de Generalized Procrustes Analysis
- Triangulación de Delaunay (visualización)
- Proceso de warping paso a paso

**D. Anexos de Validación:**
- Matrices de confusión completas
- Curvas ROC por clase
- Análisis de errores (imágenes mal clasificadas)

**Acciones:**
- [ ] Revisar capítulos 4 y 5 para identificar material que debería ir en anexos
- [ ] Priorizar anexos según criterio de importancia
- [ ] Crear lista de 2-3 anexos adicionales críticos

---

## 🔍 INVESTIGACIÓN REQUERIDA

### Tema 1: Incluir PDFs en LaTeX (Artículos y Certificados)

**Opciones comunes:**

**Opción A: `\includepdf` del paquete `pdfpages`**
```latex
\usepackage{pdfpages}

\chapter{Artículos Publicados}
\section{Artículo en IEEE}
\includepdf[pages=-,pagecommand={\thispagestyle{plain}}]{anexos/Statistical...pdf}
```

**Opción B: Referencias en texto con PDFs externos**
```latex
\chapter{Artículos y Reconocimientos}
\section{Artículos Publicados}
Los siguientes artículos fueron publicados como resultado de este trabajo:
\begin{itemize}
    \item Cruz, R. et al. (2024). Statistical Asymmetrical Histogram...
          Ver documento completo en Anexo Digital.
\end{itemize}
```

**Opción C: Híbrida (descripción + PDF integrado)**
```latex
\chapter{Publicaciones Derivadas}
\section{Artículo Principal}
Breve descripción del artículo, motivación, y resultados principales.

\subsection{Documento Completo}
\includepdf[pages=-]{anexos/Statistical...pdf}
```

**Pregunta para el usuario:**
- ¿Qué opción prefieres?
- ¿Los PDFs deben estar completamente integrados o solo referenciados?
- ¿Cuántos anexos separados? (¿Uno para artículos, otro para certificados?)

### Tema 2: Estructura Óptima de Anexos para Tesis

**Estructura tradicional:**
```
Anexo A: Manual de Usuario
Anexo B: Artículos Publicados
Anexo C: Certificados y Reconocimientos
Anexo D: Configuraciones de Experimentos
Anexo E: Resultados Detallados
...
```

**Estructura alternativa (temática):**
```
Anexo A: Documentación de Usuario
Anexo B: Producción Científica (artículos + certificados)
Anexo C: Detalles Metodológicos
Anexo D: Datos Experimentales
```

---

## 📝 ESTRUCTURA PROPUESTA DE ANEXOS (PRELIMINAR)

Basado en los archivos disponibles y necesidades típicas:

### Anexo A: Manual de Usuario de la Interfaz Gráfica ✅
**Estado:** Creado, requiere corrección de errores LaTeX
**Archivo:** `anexos/anexo_A_manual_usuario.tex`

### Anexo B: Artículos Publicados 🆕
**Estado:** Por crear
**Contenido:**
- Introducción a las publicaciones derivadas
- Artículo 1 (inglés): Statistical Asymmetrical Histogram Stretching...
- Artículo 2 (español): Enfoque de expansión estadística...
**Archivo propuesto:** `anexos/anexo_B_articulos_publicados.tex`

### Anexo C: Certificados y Reconocimientos 🆕
**Estado:** Por crear
**Contenido:**
- NOVA
- IEEE DAY
- RAS DAY
- Constancias de participación
**Archivo propuesto:** `anexos/anexo_C_certificados.tex`

### Anexo D: Configuraciones de Experimentos (Opcional pero recomendado) 🆕
**Estado:** Por crear
**Contenido:**
- Archivo JSON de mejor ensemble (ensemble_best.json)
- Archivo JSON de mejor warping (warping_best.json)
- Explicación de parámetros clave
**Archivo propuesto:** `anexos/anexo_D_configuraciones.tex`

### Anexo E: Resultados Detallados por Clase (Opcional) 🆕
**Estado:** Por crear
**Contenido:**
- Matrices de confusión ampliadas
- Métricas por clase (Precision, Recall, F1)
- Ejemplos de casos bien/mal clasificados
**Archivo propuesto:** `anexos/anexo_E_resultados_detallados.tex`

---

## 🚀 PLAN DE ACCIÓN SUGERIDO

### Paso 1: Corrección de Errores (15 min)
1. Leer `anexo_A_manual_usuario.tex` completo
2. Verificar errores de overfull/underfull
3. Aplicar correcciones

### Paso 2: Investigación de PDFs (10 min)
1. Buscar ejemplos de tesis con PDFs integrados
2. Decidir método de inclusión (usar `pdfpages`)
3. Verificar si `pdfpages` está en preamble de main.tex

### Paso 3: Creación de Anexo B (Artículos) (20 min)
1. Crear `anexo_B_articulos_publicados.tex`
2. Breve introducción sobre publicaciones derivadas
3. Usar `\includepdf` para cada artículo
4. Agregar en main.tex

### Paso 4: Creación de Anexo C (Certificados) (15 min)
1. Crear `anexo_C_certificados.tex`
2. Introducción breve
3. Incluir PDFs de certificados
4. Agregar en main.tex

### Paso 5: Evaluación de Anexos Adicionales (10 min)
1. Revisar capítulos 4-5 para identificar material complementario
2. Decidir si crear Anexo D (configuraciones) y E (resultados)
3. Priorizar según importancia para defensa

### Paso 6: Actualización de Documentación (5 min)
1. Actualizar CHECKLIST_OBSERVACIONES.md
2. Actualizar PROGRESO_REVISION.md
3. Crear resumen de sesión

**Tiempo total estimado:** ~75 minutos

---

## 📂 ARCHIVOS CLAVE PARA ESTA SESIÓN

### Archivos a Leer:
1. `docs/Tesis/anexos/anexo_A_manual_usuario.tex` - Revisar errores
2. `src_v2/gui/CHANGELOG.md` - Verificar versión GUI
3. `docs/Tesis/main.tex` - Ver preamble (¿tiene pdfpages?)
4. `configs/ensemble_best.json` - Posible Anexo D
5. `configs/warping_best.json` - Posible Anexo D

### Archivos a Crear:
1. `docs/Tesis/anexos/anexo_B_articulos_publicados.tex`
2. `docs/Tesis/anexos/anexo_C_certificados.tex`
3. (Opcional) `docs/Tesis/anexos/anexo_D_configuraciones.tex`
4. (Opcional) `docs/Tesis/anexos/anexo_E_resultados_detallados.tex`

### Archivos a Modificar:
1. `docs/Tesis/main.tex` - Agregar nuevos anexos
2. `docs/Tesis/anexos/anexo_A_manual_usuario.tex` - Corregir errores
3. `docs/Tesis/revision/CHECKLIST_OBSERVACIONES.md` - Actualizar C3
4. `docs/Tesis/revision/PROGRESO_REVISION.md` - Actualizar progreso

---

## 💡 PREGUNTAS PARA EL USUARIO (Responder al inicio de sesión)

1. **Artículos publicados:**
   - ¿Ambos artículos (inglés y español) son sobre el mismo trabajo?
   - ¿Deben incluirse completos o solo las primeras páginas?

2. **Certificados:**
   - ¿Todos los certificados deben incluirse o solo los más relevantes?
   - ¿Alguno requiere descripción contextual?

3. **Anexos adicionales:**
   - ¿Qué anexos consideras más críticos además del manual?
   - ¿Preferencia por anexos técnicos (configs) o de resultados?

4. **Versión de la GUI:**
   - ¿La versión 15 es la versión final presentada en la tesis?
   - ¿Algún cambio importante desde v15?

---

## ✅ CRITERIOS DE ÉXITO PARA ESTA SESIÓN

Al finalizar, se debe tener:
- ✅ Anexo A sin errores LaTeX (compila correctamente)
- ✅ Anexo B creado (artículos publicados integrados)
- ✅ Anexo C creado (certificados incluidos)
- ✅ Decisión sobre anexos D y E (crear ahora o después)
- ✅ Verificación de versión de GUI en Anexo A
- ✅ main.tex con paquete `pdfpages` y todos los anexos incluidos
- ✅ Documentación de progreso actualizada
- ✅ C3 completamente finalizado y listo para Fase 2

---

## 🔗 COMANDO PARA INICIAR LA SESIÓN

**Copiar y pegar al inicio de la nueva conversación:**

```
Estoy trabajando en la revisión de mi tesis según observaciones del jurado.
Completé la Fase 1 (tareas críticas C1, C2, C3), pero C3 (Anexos) requiere
revisión y expansión.

Por favor lee el archivo de instrucciones:
docs/Tesis/revision/INSTRUCCIONES_SESION4_ANEXOS_REVISION.md

Necesito:
1. Corregir errores LaTeX en Anexo A
2. Integrar PDFs de artículos publicados y certificados como anexos
3. Verificar versión de la GUI (v15)
4. Sugerir otros anexos críticos
5. Completar esto ANTES de iniciar Fase 2

Archivos de contexto adicionales:
- docs/Tesis/revision/RESUMEN_SESION3_FASE1_COMPLETA.md
- docs/Tesis/revision/PROGRESO_REVISION.md
```

---

**Preparado por:** Claude Code (Sesión 3)
**Para:** Sesión 4 - Revisión y Completado de Anexos
**Estado:** CRÍTICO - Completar antes de Fase 2
