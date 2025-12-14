# Prompt para Sesion 5b: Procesamiento Geometrico - Warping Piecewise Affine

Estoy realizando una auditoria academica de mi proyecto de tesis de maestria (clasificacion de radiografias de torax mediante deep learning). El proyecto esta en /home/donrobot/Projects/prediccion_warping_clasificacion/.

IMPORTANTE: Lee primero referencia_auditoria.md en la raiz del proyecto - contiene el protocolo COMPLETO que debes seguir A RAJA TABLA.

## ESTADO ACTUAL DE LA AUDITORIA

### Sesiones Completadas

| Sesion | Modulo                               | Estado        | Hallazgos                          |
|--------|--------------------------------------|---------------|-----------------------------------|
| 0      | Mapeo del proyecto                   | Completada    | 0🔴, 4🟠, 5🟡, 4⚪                 |
| 1      | Configuracion y utilidades           | APROBADO      | 0🔴, 0🟠, 1🟡, 4⚪                 |
| 2      | Gestion de datos (data/)             | APROBADO      | 0🔴, 2🟠 resueltos, 5🟡, 8⚪       |
| 3a     | Funciones de perdida (losses.py)     | APROBADO      | 0🔴, 1🟠 resuelto, 4🟡, 10⚪       |
| 3b     | ResNet Landmark (resnet_landmark.py) | APROBADO      | 0🔴, 0🟠, 2🟡, 15⚪                |
| 3c     | Clasificador (classifier.py)         | APROBADO      | 0🔴, 1🟠 resuelto, 2🟡, 15⚪       |
| 3d     | Jerarquico (hierarchical.py)         | APROBADO      | 0🔴, 0🟠, 2🟡, 20⚪ (experimental) |
| 4a     | Trainer (trainer.py)                 | APROBADO      | 0🔴, 0🟠, 5🟡, 18⚪                |
| 4b     | Callbacks (callbacks.py)             | APROBADO      | 0🔴, 0🟠, 1🟡, 18⚪                |
| 5a     | GPA (gpa.py)                         | APROBADO      | 0🔴, 1🟠 resuelto, 1🟡, 23⚪       |

### Hallazgos 🟠 Mayores PENDIENTES (de Sesion 0)

1. **M1:** Remover claim incorrecto sobre PFS en README.md
2. **M3:** Anadir seccion de sesgos y disclaimer medico
3. **M4:** Documentar justificacion del margen optimo 1.05

### Archivos de Referencia

- Protocolo: referencia_auditoria.md
- Plan maestro: audit/MASTER_PLAN.md
- Sesion anterior: audit/sessions/session_05a_gpa.md
- Hallazgos: audit/findings/consolidated_issues.md

---

## SESION 5b: PROCESAMIENTO GEOMETRICO - WARPING

### Archivo a Auditar

```
src_v2/processing/
├── gpa.py              (299 lineas) ← Sesion 5a COMPLETADA
├── warp.py             (449 lineas) ← ESTA SESION
└── __init__.py         (42 lineas)  ← Ya revisado en 5a
Total modulo: ~790 lineas (dividido en 2 sesiones)
```

### Contexto Tecnico de warp.py

Este archivo implementa **Piecewise Affine Warping**, tecnica fundamental para:
- Transformar imagenes a una forma canonica usando landmarks predichos
- Aplicar warping triangulo por triangulo usando transformaciones afines
- Garantizar cobertura completa de la imagen mediante puntos de borde
- Transformar mascaras binarias con la misma geometria (para PFS)

**Pipeline implementado:**
1. Imagen de entrada → Modelo predice 15 landmarks
2. Landmarks predichos + Forma canonica + Delaunay → Triangulacion
3. Warping por triangulos → Imagen geometricamente normalizada

**Importancia academica:** El warping es el puente entre la prediccion de landmarks y la clasificacion. El jurado evaluara:
- Correctitud de las transformaciones afines
- Manejo de casos degenerados (triangulos con area ~0)
- Preservacion de valores binarios en mascaras
- Documentacion del pipeline

### Estructura del Codigo (449 lineas)

| Funcion | Lineas | Descripcion |
|---------|--------|-------------|
| Docstring modulo | 1-11 | Descripcion del pipeline de warping |
| Imports | 13-18 | logging, numpy, cv2, warnings, typing, scipy |
| `_triangle_area_2x()` | 24-38 | Detectar triangulos degenerados |
| `scale_landmarks_from_centroid()` | 41-57 | Escalar landmarks desde centroide |
| `clip_landmarks_to_image()` | 60-77 | Recortar landmarks a limites de imagen |
| `add_boundary_points()` | 80-114 | Agregar 8 puntos de borde |
| `get_affine_transform_matrix()` | 117-134 | Matriz de transformacion afin |
| `create_triangle_mask()` | 137-154 | Mascara binaria para triangulo |
| `get_bounding_box()` | 157-175 | Bounding box de triangulo |
| `warp_triangle()` | 178-238 | Warping de un triangulo (IN-PLACE) |
| `piecewise_affine_warp()` | 241-301 | Funcion principal de warping |
| `compute_fill_rate()` | 304-320 | Calcular tasa de llenado |
| `warp_mask()` | 323-392 | Warping de mascaras binarias |
| `_warp_triangle_nearest()` | 395-448 | Warping con interpolacion NEAREST |

### TESTS EXISTENTES

| Archivo | Lineas | Descripcion |
|---------|--------|-------------|
| tests/test_processing.py | 901 | Tests para GPA y warping combinados |

Tests relevantes para warp.py:
- TestScaleLandmarksFromCentroid (3 tests)
- TestClipLandmarks (2 tests)
- TestAddBoundaryPoints (2 tests)
- TestGetAffineTransformMatrix (3 tests)
- TestCreateTriangleMask (3 tests)
- TestGetBoundingBox (3 tests)
- TestWarpTriangle (3 tests)
- TestPiecewiseAffineWarp (2 tests)
- TestComputeFillRate (2 tests)
- TestWarpMask (6 tests)
- Tests CLI: TestComputeCanonicalCommand, TestGenerateDatasetCommand

### Dependencias

- **Usa:** numpy, cv2 (OpenCV), scipy.spatial.Delaunay, gpa.py (add_boundary_points referenciado)
- **Es usado por:** CLI (generate-dataset), inference pipeline
- **Impacto:** Errores aqui afectarian la normalizacion geometrica de todas las imagenes

---

## LECCIONES APRENDIDAS DE SESIONES ANTERIORES (CUMPLIR ESTRICTAMENTE)

### De Sesion 5a (verificacion exhaustiva con multiples agentes):

1. **Verificacion exitosa:** Session 5a paso verificacion estricta con 0 desviaciones
2. **Conteo correcto:** El conteo manual 4+7+5+7=23 fue verificado y coincidio
3. **Clasificacion correcta:** D01 como 🟠 (jurado notara) y C01 como 🟡 (mejora recomendada)
4. **Checklist util:** El checklist pre-commit garantizo cumplimiento

### De Sesion 4b:

1. **Conteo MANUAL OBLIGATORIO:** Contar CADA hallazgo en las tablas antes de reportar totales
2. **Verificar coincidencia:** El conteo en veredicto DEBE coincidir con hallazgos listados
3. **V04 y V05 separados:** No combinar hallazgos en una sola linea de la lista de fortalezas
4. **Desviaciones:** Documentar TODAS las desviaciones detectadas y corregidas

### De Sesiones 3d y 4a:

1. **§4.4 paso 1 OBLIGATORIO:** Incluir seccion "Contexto de Sesion Anterior" con referencia explicita
2. **§5.1 CRITICO:** Si la solucion dice "Opcional", el hallazgo es ⚪ (Nota), NO 🟡 (Menor)
3. **Plantilla §6:** Son 14 secciones minimo

### De Sesiones 1-3c:

1. **Protocolo §7.2 OBLIGATORIO:** Antes de ejecutar CUALQUIER comando:
   ```
   📋 SOLICITUD DE VALIDACION
   - Comando a ejecutar: [comando]
   - Resultado esperado: [descripcion]
   - Importancia: [por que]
   - Criterio de exito: [como saber si paso]

   ¿Procedo? [Esperar mi confirmacion]
   ```

2. **Limite §4.3:** Maximo 500 lineas por sesion. warp.py (449 lineas) cumple.

3. **Orden de Auditores §3.2 (ESTRICTO):**
   1. Arquitecto de Software
   2. Revisor de Codigo
   3. Especialista en Documentacion
   4. Ingeniero de Validacion
   5. Auditor Maestro (con TABLA de veredicto)

4. **Clasificacion §5.1:**
   - 🔴 Critico: Bloquea aprobacion
   - 🟠 Mayor: Jurado notara
   - 🟡 Menor: Mejora recomendada (NO "Opcional")
   - ⚪ Nota: Fortalezas y observaciones opcionales

5. **Veredicto en formato TABLA:**
   ```
   | Metrica           | Valor              |
   |-------------------|--------------------|
   | Estado del modulo | APROBADO           |
   | Conteo            | 0🔴, X🟠, Y🟡, Z⚪ |
   | ...               |                    |
   ```

---

## AREAS DE ENFOQUE ESPECIAL

Dado que el warping es una tecnica de procesamiento de imagenes critica, prestar atencion especial a:

1. **Correctitud geometrica:**
   - Transformaciones afines correctas (cv2.getAffineTransform, cv2.warpAffine)
   - Manejo de triangulos degenerados (area < epsilon)
   - Coordenadas locales vs globales en warp_triangle

2. **Estabilidad numerica:**
   - Bounding boxes con valores negativos (get_bounding_box)
   - Division por cero o triangulos con area cero
   - Precision de punto flotante en transformaciones

3. **Integridad de mascaras:**
   - warp_mask usa INTER_NEAREST (critico para valores binarios)
   - Preservacion de valores 0/255 despues del warping
   - Documentacion de por que NEAREST es necesario

4. **Cobertura de imagen:**
   - add_boundary_points agrega 8 puntos (4 esquinas + 4 medios)
   - use_full_coverage=True garantiza cobertura completa
   - compute_fill_rate valida el resultado

5. **Documentacion del pipeline:**
   - Flujo claro: landmarks → triangulacion → warping
   - Diferencia entre warp_triangle (INTER_LINEAR) y _warp_triangle_nearest (INTER_NEAREST)
   - Modificacion IN-PLACE documentada

---

## INSTRUCCIONES

1. Lee referencia_auditoria.md completo
2. Lee audit/sessions/session_05a_gpa.md para contexto y formato correcto
3. Sigue el flujo §4.4 paso a paso
4. **INCLUYE seccion "Contexto de Sesion Anterior"** (leccion de 3d)
5. Aplica perspectiva de los 5 auditores EN ORDEN §3.2
6. ANTES de clasificar: Si la solucion es "Opcional" → usar ⚪, no 🟡
7. ANTES de reportar conteo: Contar manualmente cada severidad en las tablas
8. En ⚪: Listar CADA hallazgo separadamente (no combinar)
9. Documenta hallazgos con severidad calibrada segun §5.1
10. Veredicto en formato TABLA
11. Solicita validacion con protocolo §7.2 ANTES de ejecutar tests
12. Crea documento audit/sessions/session_05b_warp.md
13. ANTES del commit: Verificar que conteo coincide con tablas
14. Incluye seccion "Registro de Commit" y "Desviaciones de Protocolo"
15. Haz commit: `audit(session-5b): auditoria warp.py`

---

## CHECKLIST PRE-COMMIT (OBLIGATORIO)

Antes de hacer commit, verificar:
- [ ] Seccion "Contexto de Sesion Anterior" incluida
- [ ] Plantilla §6 cumple 14+ secciones
- [ ] Clasificacion §5.1 correcta (no "Opcional" en 🟡)
- [ ] Conteo manual coincide con hallazgos listados en tablas
- [ ] Cada ⚪ listado separadamente (no combinados)
- [ ] Flujo §4.4 completo (9/9 pasos)
- [ ] Orden de auditores §3.2 respetado (5/5 en orden)
- [ ] Protocolo §7.2 aplicado en validaciones
- [ ] Seccion "Registro de Commit" incluida
- [ ] Seccion "Desviaciones de Protocolo" incluida

---

## PROGRESO GLOBAL

| Metrica | Valor |
|---------|-------|
| Modulos completados | 9/12 |
| Modulo models/ | COMPLETADO (4/4) |
| Modulo training/ | COMPLETADO (2/2) |
| Modulo processing/ | 1/2 (gpa.py completado, warp.py pendiente) |
| Hallazgos totales | 🔴:0 \| 🟠:9 (6 resueltos, 3 pendientes) \| 🟡:28 \| ⚪:135 |
| Objetivo | Completar processing/ (esta sesion: warp.py) |

---

## AL FINALIZAR ESTA SESION

Con warp.py completado:
- Modulo processing/ estara 2/2 COMPLETADO
- Proxima sesion: 6 (inference/ o cli/)
- Progreso: 10/12 modulos completados

---

## NOTA SOBRE warp_mask Y PFS

El archivo contiene la funcion `warp_mask()` con un comentario importante:

```python
"""
IMPORTANT: This enables valid PFS (Pulmonary Focus Score) calculation
on warped images by ensuring mask-image geometric alignment.
"""
```

Sin embargo, en la sesion 0 se identifico M1 como hallazgo 🟠:
> **M1:** Remover claim incorrecto sobre PFS en README.md

Verificar que la documentacion en warp.py sobre PFS es:
1. Tecnica y correcta (describe lo que hace la funcion)
2. NO hace claims sobre resultados experimentales de PFS

---

¿Listo para comenzar con la Sesion 5b?
