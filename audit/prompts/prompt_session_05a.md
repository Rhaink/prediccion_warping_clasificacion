# Prompt para Sesion 5a: Procesamiento Geometrico - GPA (Generalized Procrustes Analysis)

Estoy realizando una auditoria academica de mi proyecto de tesis de maestria (clasificacion de radiografias de torax mediante deep learning). El proyecto esta en /home/donrobot/Projects/prediccion_warping_clasificacion/.

IMPORTANTE: Lee primero referencia_auditoria.md en la raiz del proyecto - contiene el protocolo COMPLETO que debes seguir A RAJA TABLA.

## ESTADO ACTUAL DE LA AUDITORIA

### Sesiones Completadas

| Sesion | Modulo                               | Estado        | Hallazgos                         |
|--------|--------------------------------------|---------------|-----------------------------------|
| 0      | Mapeo del proyecto                   | Completada    | 0🔴, 4🟠, 5🟡, 4⚪                |
| 1      | Configuracion y utilidades           | APROBADO      | 0🔴, 0🟠, 1🟡, 4⚪                |
| 2      | Gestion de datos (data/)             | APROBADO      | 0🔴, 2🟠 resueltos, 5🟡, 8⚪      |
| 3a     | Funciones de perdida (losses.py)     | APROBADO      | 0🔴, 1🟠 resuelto, 4🟡, 10⚪      |
| 3b     | ResNet Landmark (resnet_landmark.py) | APROBADO      | 0🔴, 0🟠, 2🟡, 15⚪               |
| 3c     | Clasificador (classifier.py)         | APROBADO      | 0🔴, 1🟠 resuelto, 2🟡, 15⚪      |
| 3d     | Jerarquico (hierarchical.py)         | APROBADO      | 0🔴, 0🟠, 2🟡, 20⚪ (experimental)|
| 4a     | Trainer (trainer.py)                 | APROBADO      | 0🔴, 0🟠, 5🟡, 18⚪               |
| 4b     | Callbacks (callbacks.py)             | APROBADO      | 0🔴, 0🟠, 1🟡, 18⚪               |

### Hallazgos 🟠 Mayores PENDIENTES (de Sesion 0)

1. **M1:** Remover claim incorrecto sobre PFS en README.md
2. **M3:** Anadir seccion de sesgos y disclaimer medico
3. **M4:** Documentar justificacion del margen optimo 1.05

### Archivos de Referencia

- Protocolo: referencia_auditoria.md
- Plan maestro: audit/MASTER_PLAN.md
- Sesion anterior: audit/sessions/session_04b_callbacks.md
- Hallazgos: audit/findings/consolidated_issues.md

---

## SESION 5a: PROCESAMIENTO GEOMETRICO - GPA

### Archivo a Auditar

```
src_v2/processing/
├── gpa.py              (298 lineas) ← ESTA SESION
├── warp.py             (448 lineas) ← Sesion 5b
└── __init__.py         (42 lineas)
Total modulo: ~788 lineas (dividido en 2 sesiones)
```

### Contexto Tecnico de gpa.py

Este archivo implementa **Generalized Procrustes Analysis (GPA)**, algoritmo fundamental para:
- Alinear configuraciones de landmarks entre multiples imagenes
- Calcular la forma media (mean shape) de un conjunto de landmarks
- Normalizar landmarks eliminando diferencias de escala, rotacion y traslacion
- Base matematica para el warping posterior

**Importancia academica:** GPA es uno de los pilares metodologicos del proyecto. El jurado evaluara:
- Correctitud de la implementacion matematica
- Justificacion de decisiones algoritmicas
- Documentacion clara del proceso

### TESTS EXISTENTES

| Archivo | Lineas | Descripcion |
|---------|--------|-------------|
| tests/test_processing.py | 901 | Tests para GPA y warping combinados |

### Dependencias

- **Usa:** numpy, torch, scipy (para operaciones matematicas)
- **Es usado por:** warp.py (para normalizar landmarks antes del warping)
- **Impacto:** Errores aqui propagarian a todo el pipeline de clasificacion

---

## LECCIONES APRENDIDAS DE SESIONES ANTERIORES (CUMPLIR ESTRICTAMENTE)

### De Sesion 4b (verificacion exhaustiva):

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

2. **Limite §4.3:** Maximo 500 lineas por sesion. gpa.py (298 lineas) cumple.

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

Dado que GPA es un algoritmo matematico critico, prestar atencion especial a:

1. **Correctitud matematica:**
   - Implementacion de Procrustes (rotacion optima, escalado, traslacion)
   - Convergencia del algoritmo iterativo
   - Manejo de casos degenerados (landmarks colineales, coincidentes)

2. **Estabilidad numerica:**
   - Division por cero en normalizacion
   - Overflow/underflow en operaciones matriciales
   - Precision de punto flotante

3. **Documentacion academica:**
   - Referencias a literatura (Gower, Dryden & Mardia)
   - Explicacion de decisiones algoritmicas
   - Docstrings con formulas matematicas

4. **Reproducibilidad:**
   - Seeds aleatorios si se usan
   - Determinismo de resultados
   - Tolerancias de convergencia documentadas

---

## INSTRUCCIONES

1. Lee referencia_auditoria.md completo
2. Lee audit/sessions/session_04b_callbacks.md para contexto y formato correcto
3. Usa ultrathinking y multiples agentes para analisis exhaustivo
4. Sigue el flujo §4.4 paso a paso
5. **INCLUYE seccion "Contexto de Sesion Anterior"** (leccion de 3d)
6. Aplica perspectiva de los 5 auditores EN ORDEN §3.2
7. ANTES de clasificar: Si la solucion es "Opcional" → usar ⚪, no 🟡
8. ANTES de reportar conteo: Contar manualmente cada severidad en las tablas
9. En ⚪: Listar CADA hallazgo separadamente (no combinar)
10. Documenta hallazgos con severidad calibrada segun §5.1
11. Veredicto en formato TABLA
12. Solicita validacion con protocolo §7.2 ANTES de ejecutar tests
13. Crea documento audit/sessions/session_05a_gpa.md
14. ANTES del commit: Verificar que conteo coincide con tablas
15. Incluye seccion "Registro de Commit" y "Desviaciones de Protocolo"
16. Haz commit: `audit(session-5a): auditoria gpa.py`

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
| Modulos completados | 8/12 |
| Modulo models/ | COMPLETADO (4/4) |
| Modulo training/ | COMPLETADO (2/2) |
| Modulo processing/ | 0/2 (gpa.py pendiente, warp.py pendiente) |
| Hallazgos totales | 🔴:0 \| 🟠:8 (5 resueltos, 3 pendientes) \| 🟡:27 \| ⚪:112 |
| Objetivo | Iniciar processing/ (esta sesion: gpa.py) |

---

## AL FINALIZAR ESTA SESION

Con gpa.py completado:
- Modulo processing/ estara 1/2 auditado
- Proxima sesion: 5b (warp.py - 448 lineas)
- Progreso: 9/12 modulos completados

---

¿Listo para comenzar con la Sesion 5a?
