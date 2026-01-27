# Checklist Detallado de Observaciones del Jurado

**Fecha de Coloquio:** 29 de enero de 2026
**Revisores:** Dra. Montes, M.C. Ana María, M.C. Nicolás, Observador

---

## 🔴 PRIORIDAD CRÍTICA (Bloquean Aprobación)

### ✅ C1. Objetivos Específicos - COMPLETADO
**Revisor:** M.C. Ana María + M.C. Nicolás
**Estado:** ✅ COMPLETADO (2026-01-26)
**Archivo principal:** `docs/Tesis/objetivos/0-Objetivos.tex`

**Problemas identificados:**
- [x] Objetivo 6 "Publicar resultados" NO es objetivo de investigación
- [x] Objetivo 3 menciona "KNN, CNN, MLP" pero solo se evaluó CNN
- [x] Ubicación de objetivos antes del índice es "poco ortodoxa"
- [x] Falta alineación explícita objetivos → resultados

**Acciones realizadas:**
- [x] Eliminado objetivo específico 6
- [x] Modificado objetivo específico 3: "KNN y CNN" (se evaluaron ambos)
- [x] Objetivos movidos después del índice (ubicación tradicional)
- [x] Tabla de cumplimiento de objetivos agregada en Cap 6.1.4

**Archivos modificados:**
- [x] `docs/Tesis/objetivos/0-Objetivos.tex` (modificado por usuario)
- [x] `docs/Tesis/capitulo6/6_conclusiones.tex` (tabla de cumplimiento)
- [x] `docs/Tesis/main.tex` (reubicación de sección)
- [x] `docs/Tesis/capitulo5/5_4_analisis_comparativo.tex` (4 menciones de KNN)

**Criterios de éxito cumplidos:**
- ✅ 5 objetivos específicos (eliminado el 6)
- ✅ Objetivo 3 correctamente formulado (KNN y CNN evaluados)
- ✅ Tabla de cumplimiento en Conclusiones (subsección 1.4.1)
- ✅ Coherencia objetivos ↔ resultados verificada
- ✅ BONUS: Resultados de KNN integrados en Cap 5 y 6

---

### ✅ C2. Agradecimientos - COMPLETADO
**Revisor:** M.C. Ana María
**Estado:** ✅ COMPLETADO (2026-01-26)
**Archivo principal:** `docs/Tesis/agradecimientos/agradecimientos.tex`

**Problema:**
- [x] No existe sección de agradecimientos

**Acciones realizadas:**
- [x] Crear archivo `docs/Tesis/agradecimientos/agradecimientos.tex`
- [x] Mencionar SECIHTI (Secretaría de Ciencia, Humanidades, Tecnología e Innovación)
- [x] Mencionar BUAP
- [x] Agradecer a directores de tesis (Dr. Salvador Ayala Raggi, Dr. Aldrin Barreto Flores)
- [x] Agradecer a comité revisor (Dra. Montes, M.C. Ana María, M.C. Nicolás)
- [x] Incluir en `main.tex` (después de portada, antes de índice)

**Archivos creados/modificados:**
- [x] `docs/Tesis/agradecimientos/agradecimientos.tex` (creado)
- [x] `docs/Tesis/main.tex` (agregada sección de agradecimientos)

**Archivos a modificar:**
- [ ] `docs/Tesis/main.tex` (incluir sección)

**Criterio de éxito:**
- ✅ Sección de agradecimientos creada
- ✅ Menciona SECIHTI, BUAP, directores, comité
- ✅ Tono apropiado y conciso
- ✅ Incluida en compilación de tesis

---

### ✅ C3. Anexos - COMPLETADO
**Revisor:** M.C. Ana María
**Estado:** ✅ COMPLETADO (2026-01-26, Sesión 4 + Sesión 5)
**Archivos principales:** `docs/Tesis/anexos/`

**Problema:**
- [x] No existen anexos (RESUELTO: 3 anexos creados)
- [x] Advertencia: NO incluir código fuente (CUMPLIDO)
- [x] Errores LaTeX en Anexo A corregidos
- [x] PDFs de artículos y certificados integrados
- [x] Versión de GUI verificada (v1.0.0, no v15)
- [x] Anexos críticos completados (manual, artículos, certificados)
- [x] Contenido REAL de PDFs revisado y corregido (Sesión 5)

**Acciones completadas (Sesión 3 + Sesión 4 + Sesión 5):**
- [x] Crear Anexo A: Manual de Usuario (GUI de demostración)
- [x] Corregir errores LaTeX en anexo_A_manual_usuario.tex
- [x] Crear Anexo B: Artículos Publicados (2 PDFs integrados)
- [x] Crear Anexo C: Certificados y Reconocimientos (4 PDFs integrados)
- [x] Agregar paquete `pdfpages` a main.tex
- [x] Incluir todos los anexos en `main.tex` con comando `\appendix`
- [x] Renombrar PDFs para evitar problemas con espacios en LaTeX
- [x] NO se incluyó código fuente (solo documentación)
- [x] Verificar versión de GUI (confirmada: v1.0.0)
- [x] **Revisar contenido REAL de los 6 PDFs incluidos (Sesión 5)**
- [x] **Corregir títulos, descripciones y datos en Anexo B según PDFs reales (Sesión 5)**
- [x] **Corregir información en Anexo C según certificados reales (Sesión 5)**
- [x] **Comentar temporalmente Anexo A (GUI) para revisión posterior (Sesión 5)**
- [x] **Verificar coherencia entre artículos (Anexo B) y certificados (Anexo C) (Sesión 5)**

**Anexo D (Configuraciones):**
- [x] Evaluado y descartado por decisión del usuario

**Archivos creados:**
- [x] `docs/Tesis/anexos/anexo_A_manual_usuario.tex` (corregido, temporalmente comentado)
- [x] `docs/Tesis/anexos/anexo_B_articulos_publicados.tex` (corregido con información real)
- [x] `docs/Tesis/anexos/anexo_C_certificados.tex` (corregido con información real)

**Archivos modificados:**
- [x] `docs/Tesis/main.tex` (pdfpages agregado, Anexo A comentado, Anexos B y C corregidos)
- [x] PDFs renombrados a nombres sin espacios

**PDFs integrados y REVISADOS:**
- [x] articulo_ingles_sahs_2024.pdf - Publicado en **Computación y Sistemas** (no NOVA)
- [x] articulo_espanol_sahs_2024.pdf - Publicado en **Abstraction & Application** (no RAS DAY)
- [x] reconocimiento_nova.pdf - Poster sobre **normalización geométrica** (no SAHS)
- [x] reconocimiento_ieee_day.pdf - Presentación sobre **redes neuronales** (no artículo)
- [x] constancia_participacion_rafael.pdf - Evento **RVP-AI ROC&C 2025** en Acapulco
- [x] reconocimiento_ras_day.pdf - **Taller de visión por computadora** (no congreso)

**Correcciones críticas realizadas en Sesión 5:**
- ✅ Artículo inglés: Revista "Computación y Sistemas" (no Memorias NOVA)
- ✅ Artículo español: Revista "Abstraction & Application" (no RAS DAY)
- ✅ Agregado 5to autor: José Francisco Portillo Robledo
- ✅ Agregada institución INAOE como segunda afiliación
- ✅ Reconocimiento NOVA: Corregido tema (normalización geométrica, no SAHS)
- ✅ Reconocimiento IEEE DAY: Agregada información completa (presentación redes neuronales)
- ✅ Constancia RVP-AI: Agregada información completa del evento
- ✅ Reconocimiento RAS: Corregido tipo (taller, no congreso)

**Criterios de éxito COMPLETADOS:**
- ✅ Anexo A sin errores LaTeX (compila correctamente, temporalmente comentado)
- ✅ Anexo B corregido con información REAL de los artículos
- ✅ Anexo C corregido con información REAL de los certificados
- ✅ Decisión sobre anexos adicionales tomada
- ✅ Versión de GUI verificada (v1.0.0)
- ✅ main.tex con paquete `pdfpages` y anexos correctos
- ✅ Coherencia verificada entre Anexos B y C (eventos independientes)
- ✅ Información exacta de publicaciones, eventos, fechas e instituciones

---

## 🟡 PRIORIDAD ALTA (Mejoran Sustancialmente la Tesis)

### ☐ H1. Aportación Científica - CLARIFICACIÓN REQUERIDA
**Revisor:** Dra. Montes + M.C. Ana María
**Estado:** ⏳ Pendiente

**Contribuciones identificadas:**
1. Ensemble+TTA landmark: 3.61 px (10.6% mejora)
2. Sistema completo GPA + Delaunay + warping
3. Clasificador: 98.10% accuracy, 97.17% F1-Macro
4. Metodología reproducible (configs JSON)
5. **PRINCIPAL:** Demostración de shortcut learning (98.68% → 95.36%)

**Acciones requeridas:**
- [ ] Cap 1: Agregar sección 1.X "Contribuciones de este Trabajo"
- [ ] Cap 3: Agregar sección 3.9 "Posicionamiento y Contribuciones"
- [ ] Cap 3: Tabla comparativa con 2-3 trabajos relacionados
- [ ] Cap 6: Reformatear sección 6.1.2 (lista numerada + recuadro)

---

### ☐ H2. Justificación de 15 Landmarks y 16 Triángulos
**Revisor:** Dra. Montes + Observador
**Estado:** ⏳ Pendiente

**Acciones requeridas:**
- [ ] Agregar subsección 4.4.1 "Diseño del Sistema de Landmarks"
- [ ] Justificación anatómica de 15 puntos
- [ ] Explicar relación landmarks → triángulos Delaunay
- [ ] Crear figura ilustrativa

---

### ☐ H3. Balanceo de Datos y Estrategia de Pesos
**Revisor:** Dra. Montes
**Estado:** ⏳ Pendiente

**Acciones requeridas:**
- [ ] Subsección 4.2.4 "Estrategia de Manejo de Desbalance"
- [ ] Subsección 4.5.3 "Análisis de Parámetros del Modelo"
- [ ] Reportar pesos de clase exactos
- [ ] ResNet-18: 11.2M parámetros (desglose)

---

### ☐ H4. Costo Computacional
**Revisor:** Dra. Montes + M.C. Nicolás
**Estado:** ⏳ Pendiente

**Acciones requeridas:**
- [ ] Subsección 4.6.3 "Análisis de Costo Computacional"
- [ ] Tiempo de entrenamiento
- [ ] Tiempo de inferencia
- [ ] Hardware usado

---

### ☐ H5. Hiperparámetros
**Revisor:** Dra. Montes
**Estado:** ⏳ Pendiente

**Acciones requeridas:**
- [ ] Tablas de hiperparámetros
- [ ] Justificar valores clave
- [ ] Proceso de selección

---

### ☐ H6. Influencia de Equipos de Rayos X
**Revisor:** Dra. Montes
**Estado:** ⏳ Pendiente

**Acciones requeridas:**
- [ ] Documentar homogeneidad del dataset
- [ ] Agregar limitación en Cap 5

---

### ☐ H7. Comparación con Referencia [5]
**Revisor:** Observador
**Estado:** ⏳ Pendiente

**Acciones requeridas:**
- [ ] Revisar Tabla 3.1
- [ ] Subsección 5.4.2 "Comparación Cuantitativa"

---

## 🟢 PRIORIDAD MEDIA

### ☐ M1. Manual de Usuario
- [ ] Crear Anexo A

### ☐ M2. Robustez ante Ruido
- [ ] Subsección 5.4.3 o agregar en Limitaciones

### ☐ M3. Metadatos de Pacientes
- [ ] Mencionar en 4.2.1 y Limitaciones

### ☐ M4. Referencias Web
- [ ] Revisar `references.bib`

### ☐ M5. Licencias de Software
- [ ] Subsección 4.1.4

---

## Resumen de Progreso

| Prioridad | Total | Completadas | Pendientes | % |
|-----------|-------|-------------|------------|---|
| 🔴 Crítica | 3 | 1 | 2 | 33% |
| 🟡 Alta | 7 | 0 | 7 | 0% |
| 🟢 Media | 5 | 0 | 5 | 0% |
| **TOTAL** | **15** | **1** | **14** | **7%** |

**Última actualización:** 2026-01-26 18:30
**Tareas completadas hoy:** C1 (Objetivos Específicos)
